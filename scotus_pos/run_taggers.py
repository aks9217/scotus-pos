"""Tag a column file with each pretrained tagger and add one column per tagger.

    python -m scotus_pos.run_taggers                     # legal dev and test, all taggers
    python -m scotus_pos.run_taggers --split dev --taggers nltk flair
    python -m scotus_pos.run_taggers --split wsj24       # WSJ section 24, for the in-domain baseline
    python -m scotus_pos.run_taggers --input wsj_test.pos --output data/tagged/wsj_test.tsv

A split names an input file (see SPLITS) and writes data/tagged/{split}.tsv
with columns TOKEN, GOLD, then one per tagger. If the output file already
exists, the columns of taggers you are not re-running are kept, so you can
add or redo one tagger at a time. Tags are written raw; normalize them with
tagset.normalize_tag when scoring or voting.

Tagging the test set is fine; scoring on it is what the test-set rule limits.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .io import read_columns, write_columns
from .paths import GOLD_DIR, TAGGED_DIR, WSJ_DIR
from .taggers import TAGGERS, load_tagger

SPLITS = {
    "dev": GOLD_DIR / "dev.tsv",
    "test": GOLD_DIR / "test.tsv",
    "wsj24": WSJ_DIR / "WSJ_24.pos",  # in-domain comparison (held out from HMM training)
}


def run(input_path: Path, output_path: Path, names: list[str]) -> None:
    gold_cols, sentences = read_columns(input_path)

    # Keep columns from an earlier run for taggers we are not re-running.
    if output_path.exists():
        old_cols, old_sents = read_columns(output_path)
        if [s.tokens for s in old_sents] != [s.tokens for s in sentences]:
            raise SystemExit(f"{output_path} has different tokens from {input_path}; delete it and rerun")
        for col in old_cols:
            if col not in gold_cols and col not in names:
                for sent, old in zip(sentences, old_sents):
                    sent.columns[col] = old.columns[col]

    token_lists = [s.tokens for s in sentences]
    ntok = sum(map(len, token_lists))
    for name in names:
        print(f"{name}: tagging {len(sentences)} sentences, {ntok} tokens ...", flush=True)
        start = time.perf_counter()
        tags = load_tagger(name).tag_sents(token_lists)
        for sent, sent_tags in zip(sentences, tags):
            sent.columns[name] = sent_tags
        print(f"  done in {time.perf_counter() - start:.1f}s", flush=True)

    # Base columns first, then taggers in registry order, then anything else (e.g. HMM).
    present = set().union(*(s.columns for s in sentences)) if sentences else set()
    columns = [c for c in gold_cols if c in present]
    columns += [c for c in TAGGERS if c in present and c not in columns]
    columns += sorted(present - set(columns))
    write_columns(output_path, sentences, columns)
    print(f"wrote {output_path} with columns {', '.join(columns)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=["dev", "test"], choices=list(SPLITS),
                    help="files to tag (default: dev test, the legal gold set)")
    ap.add_argument("--taggers", nargs="+", default=list(TAGGERS), type=str.upper, metavar="NAME",
                    help=f"taggers to run (default: all). Choices: {', '.join(TAGGERS)}")
    ap.add_argument("--input", type=Path, help="tag this file instead of a gold split")
    ap.add_argument("--output", type=Path, help="where to write --input's result")
    args = ap.parse_args()

    unknown = [t for t in args.taggers if t not in TAGGERS]
    if unknown:
        ap.error(f"unknown tagger(s) {unknown}; choose from {', '.join(TAGGERS)}")
    if (args.input is None) != (args.output is None):
        ap.error("--input and --output go together")

    if args.input:
        run(args.input, args.output, args.taggers)
    else:
        for split in args.split:
            run(SPLITS[split], TAGGED_DIR / f"{split}.tsv", args.taggers)


if __name__ == "__main__":
    main()
