"""Assemble data/gold/dev.tsv and data/gold/test.tsv from the annotation files.

    python -m scotus_pos.build_gold                     # fails until every token is tagged
    python -m scotus_pos.build_gold --allow-incomplete  # writes "_" for untagged tokens

Sources:
  * the 50 agreement sentences come from annotation/adjudicated_agreement.tsv
    (written by ``python -m scotus_pos.agreement`` and then adjudicated);
  * every other sentence comes from annotation/individual/<annotator>.tsv.

Checks that tokens match the manifest exactly and every tag is a Penn Treebank
tag. "?" (an unadjudicated disagreement) counts as untagged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import paths
from .io import MISSING, Sentence, read_columns, write_columns
from .tagset import PTB_TAGS, normalize_tag

UNRESOLVED = "?"


def load_manifest(gold_dir: Path) -> list[dict]:
    with (gold_dir / "manifest.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _index(path: Path) -> dict[str, Sentence]:
    if not path.exists():
        return {}
    _, sents = read_columns(path)
    return {s.sent_id: s for s in sents}


def assemble(gold_dir: Path, allow_incomplete: bool = False, quiet: bool = False) -> bool:
    manifest = load_manifest(gold_dir)
    ann = gold_dir / "annotation"
    adjudicated = _index(ann / "adjudicated_agreement.tsv")
    individual: dict[str, dict[str, Sentence]] = {}
    for rec in manifest:
        if not rec["agreement"]:
            name = rec["annotators"][0]
            if name not in individual:
                individual[name] = _index(ann / "individual" / f"{name}.tsv")

    errors: list[str] = []
    done = defaultdict(lambda: [0, 0])  # source -> [tagged tokens, total tokens]
    out: dict[str, list[Sentence]] = {"dev": [], "test": []}
    for rec in manifest:
        tokens = [t[0] for t in rec["tokens"]]
        source = "agreement (adjudicated)" if rec["agreement"] else rec["annotators"][0]
        src = adjudicated.get(rec["sentence_id"]) if rec["agreement"] else individual[source].get(rec["sentence_id"])
        tags = [MISSING] * len(tokens)
        if src is not None:
            if src.tokens != tokens:
                errors.append(f"{rec['sentence_id']} ({source}): tokens differ from manifest; do not edit the TOKEN column")
            else:
                tags = []
                for i, raw in enumerate(src.columns.get("GOLD", [MISSING] * len(tokens))):
                    tag = raw.strip()
                    if tag in (MISSING, UNRESOLVED, ""):
                        tags.append(MISSING)
                        continue
                    tag = normalize_tag(tag)
                    if tag not in PTB_TAGS:
                        errors.append(f"{rec['sentence_id']} token {i + 1} {tokens[i]!r} ({source}): {raw!r} is not a PTB tag")
                        tag = MISSING
                    tags.append(tag)
        done[source][0] += sum(t != MISSING for t in tags)
        done[source][1] += len(tags)
        sent = Sentence(rec["sentence_id"], tokens, {"GOLD": tags},
                        meta={"text": rec["text"], "record_id": rec["record_id"]})
        out[rec["split"]].append(sent)

    complete = all(a == b for a, b in done.values())
    if not quiet:
        for source, (a, b) in sorted(done.items()):
            print(f"  {source:<28} {a:>6}/{b:<6} tokens tagged ({100 * a / b:5.1f}%)")
        for e in errors[:50]:
            print("  ERROR", e)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors")
    if (errors or not complete) and not allow_incomplete:
        print("Gold set is not complete; nothing written. Use --allow-incomplete to write a partial set.")
        return False
    for split, sents in out.items():
        write_columns(gold_dir / f"{split}.tsv", sents, ["GOLD"])
    if not quiet:
        print(f"Wrote {gold_dir / 'dev.tsv'} ({len(out['dev'])} sentences) and "
              f"{gold_dir / 'test.tsv'} ({len(out['test'])} sentences)"
              + ("" if complete and not errors else " [INCOMPLETE]"))
    return complete and not errors


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-dir", type=Path, default=paths.GOLD_DIR)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)
    ok = assemble(args.gold_dir, allow_incomplete=args.allow_incomplete)
    sys.exit(0 if ok or args.allow_incomplete else 1)


if __name__ == "__main__":
    main()
