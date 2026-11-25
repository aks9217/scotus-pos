"""Token accuracy of each tagger against the gold tags, as a Markdown table.

    python -m scotus_pos.stats                    # WSJ 24 and legal dev
    python -m scotus_pos.stats --split dev

Reads the tagged files in data/tagged/ (written by run_taggers). Gold and
predicted tags both go through tagset.normalize_tag first, so differences
such as "(" vs "-LRB-" or HYPH vs ":" are not counted as errors. A tagger
with no column in a file is shown as "–".

The legal test set is only scored with --final, because each final system
is scored on it once (see "Rules for the test set" in the README).
"""

from __future__ import annotations

import argparse

from .io import MISSING, read_columns
from .paths import TAGGED_DIR
from .tagset import normalize_tag
from .taggers import TAGGERS

SPLIT_LABELS = {"wsj24": "WSJ 24", "dev": "Legal dev", "test": "Legal test"}
TAGGER_LABELS = {
    "NLTK": "NLTK",
    "STANFORD": "Stanford",
    "SPACY_SM": "spaCy small",
    "SPACY_TRF": "spaCy transformer",
    "FLAIR": "Flair",
    "STANZA": "Stanza",
    "HMM": "HMM",
}


def accuracies(split: str) -> tuple[dict[str, float], int]:
    """Accuracy per tagger column in data/tagged/{split}.tsv, and the number of gold-tagged tokens."""
    columns, sentences = read_columns(TAGGED_DIR / f"{split}.tsv")
    gold = [normalize_tag(t) for s in sentences for t in s.columns["GOLD"]]
    scored = [i for i, g in enumerate(gold) if g != MISSING]
    result = {}
    for col in columns:
        if col == "GOLD":
            continue
        pred = [normalize_tag(t) for s in sentences for t in s.columns[col]]
        result[col] = sum(pred[i] == gold[i] for i in scored) / len(scored)
    return result, len(scored)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=["wsj24", "dev"], choices=list(SPLIT_LABELS),
                    help="tagged files to score, one table column each (default: wsj24 dev)")
    ap.add_argument("--final", action="store_true", help="allow scoring the legal test set")
    args = ap.parse_args()
    if "test" in args.split and not args.final:
        ap.error("scoring the legal test set needs --final (score each final system on it once)")

    results = {}
    sizes = {}
    for split in args.split:
        results[split], sizes[split] = accuracies(split)

    taggers = [t for t in TAGGERS if any(t in r for r in results.values())]
    taggers += sorted({t for r in results.values() for t in r} - set(taggers))  # e.g. ensembles later
    print("| Tagger | " + " | ".join(SPLIT_LABELS[s] for s in args.split) + " |")
    print("|---|" + "---:|" * len(args.split))
    for t in taggers:
        cells = [f"{results[s][t]:.3f}" if t in results[s] else "–" for s in args.split]
        print(f"| {TAGGER_LABELS.get(t, t)} | " + " | ".join(cells) + " |")
    print("\nTokens scored: " + ", ".join(f"{SPLIT_LABELS[s]} {sizes[s]:,}" for s in args.split))


if __name__ == "__main__":
    main()
