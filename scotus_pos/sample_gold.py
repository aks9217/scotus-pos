"""Sample the legal gold set and split it into dev and test by opinion.

    python -m scotus_pos.sample_gold                 # defaults from the proposal
    python -m scotus_pos.sample_gold --annotators alice,bob,carol

Design (all randomness comes from --seed):

1. Opinions flagged ``short_document_review_before_annotation`` or with too few
   usable sentences are dropped; --n-opinions (90) are drawn from the rest.
2. The opinions are sorted by year and cut into blocks of 10. Within each block
   --dev-frac (30%) go to dev, the rest to test, so both splits span 1793-2006
   and no opinion is in both.
3. Sentences are drawn round-robin across a split's opinions (a shuffled
   sentence list per opinion), so every opinion contributes about the same
   number of sentences and long opinions do not dominate.
4. --n-agreement (50) sentences, taken from dev and test in proportion, are
   tagged by every annotator. The rest are dealt to annotators in a snake order
   by sentence length, which balances both sentences and tokens per person.

Outputs in --out (default data/gold/):
    manifest.jsonl                    one record per sampled sentence
    sample_summary.json               parameters, opinion lists, counts
    annotation/agreement/<name>.tsv   the same 50 sentences, one copy per annotator
    annotation/individual/<name>.tsv  each annotator's own share
    dev.tsv, test.tsv                 gold files, assembled by scotus_pos.build_gold
                                      (GOLD is "_" until annotation is done)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from . import paths
from .io import MISSING, Sentence, read_columns, write_columns

SHORT_DOC_FLAG = "short_document_review_before_annotation"
_YEAR = re.compile(r"\b(1[789]\d\d|200\d)\b")


# --------------------------------------------------------------------------- loading

def opinion_year(rec: dict) -> tuple[int | None, str]:
    """Decision year and where it came from."""
    meta = rec.get("metadata", {})
    for key in ("decision_date", "decision_date_from_text"):
        value = (meta.get(key) or {}).get("value")
        if value and str(value)[:4].isdigit():
            return int(str(value)[:4]), key
    m = _YEAR.search(rec.get("text_clean", "")[:800])
    if m:
        return int(m.group(1)), "first_year_in_header"
    return None, "unknown"


def sentence_ok(sent: dict, min_tokens: int, max_tokens: int) -> bool:
    """Keep running-prose sentences; drop headers, page markers and split fragments."""
    toks = [t["text"] for t in sent["tokens"]]
    if not (min_tokens <= len(toks) <= max_tokens):
        return False
    if "\n" in sent["text"]:          # header lines, footnote markers glued to text
        return False
    if sum(any(c.islower() for c in t) for t in toks) < 3:
        return False
    alpha = [t for t in toks if t.isalpha()]
    if alpha and sum(t.isupper() and len(t) > 1 for t in alpha) / len(alpha) > 0.5:
        return False                  # ALL-CAPS captions
    first = next((t for t in toks if any(c.isalpha() for c in t)), "")
    if first[:1].islower():           # sentence splitter cut mid-sentence
        return False
    return True


def load_opinions(path: Path, min_tokens: int, max_tokens: int, min_sentences: int) -> tuple[list[dict], Counter]:
    opinions, dropped = [], Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("duplicate_of"):
                dropped["duplicate"] += 1
                continue
            if SHORT_DOC_FLAG in rec.get("quality_flags", []):
                dropped["short_document_flag"] += 1
                continue
            seen, usable = set(), []
            for s in rec["sentences"]:
                key = " ".join(s["text"].split())
                if sentence_ok(s, min_tokens, max_tokens) and key not in seen:
                    seen.add(key)
                    usable.append(s)
            if len(usable) < min_sentences:
                dropped["too_few_usable_sentences"] += 1
                continue
            year, year_source = opinion_year(rec)
            opinions.append({
                "record_id": rec["record_id"],
                "case_group_id": rec.get("case_group_id"),
                "us_citation": (rec.get("metadata", {}).get("us_citation") or {}).get("value"),
                "year": year,
                "year_source": year_source,
                "sentences": usable,
            })
    return opinions, dropped


# --------------------------------------------------------------------------- sampling

def split_by_era(opinions: list[dict], dev_frac: float, rng: random.Random, block: int = 10) -> tuple[list[dict], list[dict]]:
    ordered = sorted(opinions, key=lambda o: (o["year"] is None, o["year"] or 0, o["record_id"]))
    dev, test = [], []
    for i in range(0, len(ordered), block):
        chunk = ordered[i:i + block]
        rng.shuffle(chunk)
        k = round(dev_frac * len(chunk))
        dev += chunk[:k]
        test += chunk[k:]
    return dev, test


def round_robin(opinions: list[dict], n: int, rng: random.Random) -> list[tuple[dict, dict]]:
    pools = []
    for op in opinions:
        sents = list(op["sentences"])
        rng.shuffle(sents)
        pools.append((op, sents))
    rng.shuffle(pools)
    picked: list[tuple[dict, dict]] = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for op, sents in pools:
            if depth < len(sents) and len(picked) < n:
                picked.append((op, sents[depth]))
                progressed = True
        if not progressed:
            raise SystemExit(f"Only {len(picked)} usable sentences available, {n} requested.")
        depth += 1
    return picked


def snake_deal(items: list[dict], names: list[str], rng: random.Random) -> dict[str, list[dict]]:
    """Deal longest-first in order 1,2,3,3,2,1,... to balance token counts."""
    items = list(items)
    rng.shuffle(items)
    items.sort(key=lambda r: -r["n_tokens"])
    out: dict[str, list[dict]] = {n: [] for n in names}
    order = names + names[::-1]
    for i, item in enumerate(items):
        out[order[i % len(order)]].append(item)
    return out


# --------------------------------------------------------------------------- writing

def record_to_sentence(rec: dict) -> Sentence:
    sent = Sentence(rec["sentence_id"], [t[0] for t in rec["tokens"]],
                    meta={"text": rec["text"], "source": f"{rec['us_citation']} ({rec['year']})",
                          "record_id": rec["record_id"]})
    sent.columns["GOLD"] = [MISSING] * len(sent)
    return sent


def annotation_started(path: Path) -> bool:
    if not path.exists():
        return False
    _, sents = read_columns(path)
    return any(t != MISSING for s in sents for t in s.columns.get("GOLD", []))


def write_annotation_file(path: Path, recs: list[dict]) -> None:
    recs = sorted(recs, key=lambda r: r["sentence_id"])
    write_columns(path, [record_to_sentence(r) for r in recs], ["GOLD"], meta_keys=("source", "text"))


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=paths.RAW_OPINIONS)
    ap.add_argument("--out", type=Path, default=paths.GOLD_DIR)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--n-opinions", type=int, default=90)
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--n-dev", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=350)
    ap.add_argument("--n-agreement", type=int, default=50)
    ap.add_argument("--annotators", default="A1,A2,A3",
                    help="comma-separated names, used as file names (default: A1,A2,A3)")
    ap.add_argument("--min-tokens", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=80,
                    help="longer sentences are skipped")
    ap.add_argument("--min-sentences", type=int, default=8,
                    help="opinions with fewer usable sentences are skipped")
    ap.add_argument("--force", action="store_true", help="overwrite annotation files that already contain tags")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.annotators.split(",") if n.strip()]
    if len(set(names)) != len(names) or len(names) < 2:
        ap.error("need at least two distinct annotator names")

    ann_dir = args.out / "annotation"
    existing = list(ann_dir.glob("*/*.tsv")) if ann_dir.exists() else []
    started = [p for p in existing if annotation_started(p)]
    if started and not args.force:
        sys.exit("Refusing to resample: these files already contain tags:\n  "
                 + "\n  ".join(map(str, started)) + "\nRe-run with --force to discard them.")

    rng = random.Random(args.seed)
    opinions, dropped = load_opinions(args.input, args.min_tokens, args.max_tokens, args.min_sentences)
    if len(opinions) < args.n_opinions:
        sys.exit(f"Only {len(opinions)} eligible opinions, {args.n_opinions} requested.")
    chosen = rng.sample(sorted(opinions, key=lambda o: o["record_id"]), args.n_opinions)
    dev_ops, test_ops = split_by_era(chosen, args.dev_frac, rng)

    records = []
    for split, ops, n in (("dev", dev_ops, args.n_dev), ("test", test_ops, args.n_test)):
        for op, s in round_robin(ops, n, rng):
            records.append({
                "sentence_id": s["sentence_id"],
                "record_id": op["record_id"],
                "case_group_id": op["case_group_id"],
                "us_citation": op["us_citation"],
                "year": op["year"],
                "year_source": op["year_source"],
                "split": split,
                "agreement": False,
                "annotators": [],
                "n_tokens": len(s["tokens"]),
                "text": " ".join(s["text"].split()),
                "tokens": [[t["text"], t["start"], t["end"]] for t in s["tokens"]],
            })

    total = args.n_dev + args.n_test
    n_agree_dev = round(args.n_agreement * args.n_dev / total)
    for split, k in (("dev", n_agree_dev), ("test", args.n_agreement - n_agree_dev)):
        for r in rng.sample([r for r in records if r["split"] == split], k):
            r["agreement"] = True
            r["annotators"] = list(names)
    shares: dict[str, list[dict]] = {n: [] for n in names}
    for split in ("dev", "test"):
        dealt = snake_deal([r for r in records if r["split"] == split and not r["agreement"]], names, rng)
        for name, recs in dealt.items():
            for r in recs:
                r["annotators"] = [name]
            shares[name] += recs

    records.sort(key=lambda r: (r["split"], r["sentence_id"]))

    # ---- write
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    agreement = [r for r in records if r["agreement"]]
    for name in names:
        write_annotation_file(ann_dir / "agreement" / f"{name}.tsv", agreement)
        write_annotation_file(ann_dir / "individual" / f"{name}.tsv", shares[name])

    def stats(rs: list[dict]) -> dict:
        return {"sentences": len(rs), "tokens": sum(r["n_tokens"] for r in rs),
                "opinions": len({r["record_id"] for r in rs})}

    summary = {
        "seed": args.seed,
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "force"},
        "eligible_opinions": len(opinions),
        "dropped_opinions": dict(dropped),
        "dev_opinions": sorted(o["record_id"] for o in dev_ops),
        "test_opinions": sorted(o["record_id"] for o in test_ops),
        "dev": stats([r for r in records if r["split"] == "dev"]),
        "test": stats([r for r in records if r["split"] == "test"]),
        "agreement": stats(agreement),
        "per_annotator_individual": {n: stats(shares[n]) for n in names},
        "per_annotator_total": {n: stats(shares[n] + agreement) for n in names},
        "years": {split: [min(o["year"] for o in ops if o["year"]), max(o["year"] for o in ops if o["year"])]
                  for split, ops in (("dev", dev_ops), ("test", test_ops))},
    }
    assert not set(summary["dev_opinions"]) & set(summary["test_opinions"])
    (args.out / "sample_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    from .build_gold import assemble
    assemble(args.out, allow_incomplete=True, quiet=True)

    print(json.dumps({k: summary[k] for k in ("eligible_opinions", "dev", "test", "agreement",
                                              "per_annotator_total", "years")}, indent=2))
    print(f"Wrote {args.out}/manifest.jsonl, annotation files for {', '.join(names)}, dev.tsv and test.tsv")


if __name__ == "__main__":
    main()
