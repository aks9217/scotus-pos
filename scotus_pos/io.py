"""The shared column file format.

One token per line, tab-separated columns (TOKEN, GOLD, then one column per
tagger or ensemble), and a blank line between sentences, as in Homework 3.
Two kinds of optional comment lines are added on top of that:

    # columns = TOKEN<TAB>GOLD<TAB>NLTK ...   (once, at the top of the file)
    # sent_id = wol-85472-s00003              (before each sentence)
    # text = Supreme Court of United States.

A comment line starts with "# " and contains " = " but no tab, so a token "#"
(which always has at least one tab after it) is never mistaken for one. Files
without a "# columns" header are read as TOKEN, GOLD, COL3, COL4, ... so plain
Homework 3 ``.pos`` files load unchanged; a one-column file is read as TOKEN.

A missing tag is written as "_".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MISSING = "_"
_COMMENT = re.compile(r"^# ([A-Za-z_]+) = (.*)$", re.S)


@dataclass
class Sentence:
    sent_id: str
    tokens: list[str]
    columns: dict[str, list[str]] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def group(self) -> str:
        """Document the sentence came from: 'wol-85472-s00003' -> 'wol-85472'."""
        return self.meta.get("record_id") or self.sent_id.rsplit("-s", 1)[0]

    def __len__(self) -> int:
        return len(self.tokens)


def _is_comment(line: str) -> re.Match | None:
    if "\t" in line and not line.startswith("# columns = "):
        return None
    return _COMMENT.match(line)


def read_columns(path: str | Path) -> tuple[list[str], list[Sentence]]:
    """Read a column file. Returns (column names excluding TOKEN, sentences)."""
    path = Path(path)
    names: list[str] | None = None
    sentences: list[Sentence] = []
    rows: list[list[str]] = []
    meta: dict[str, str] = {}

    def flush() -> None:
        nonlocal rows, meta
        if rows:
            ncol = max(len(r) for r in rows)
            cols = names[1:] if names else (["GOLD"] + [f"COL{i}" for i in range(3, ncol + 1)])[: ncol - 1]
            sid = meta.pop("sent_id", f"{path.stem}-s{len(sentences):05d}")
            sent = Sentence(sid, [r[0] for r in rows], meta=dict(meta))
            for j, name in enumerate(cols, start=1):
                sent.columns[name] = [r[j] if j < len(r) and r[j] != "" else MISSING for r in rows]
            sentences.append(sent)
        rows, meta = [], {}

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                flush()
                continue
            m = _is_comment(line)
            if m:
                key, value = m.group(1), m.group(2)
                if key == "columns":
                    names = value.split("\t")
                else:
                    if rows:  # a comment after tokens starts a new sentence
                        flush()
                    meta[key] = value
                continue
            rows.append(line.split("\t"))
    flush()
    cols = names[1:] if names else (list(sentences[0].columns) if sentences else [])
    return cols, sentences


def write_columns(path: str | Path, sentences: list[Sentence], columns: list[str],
                  meta_keys: tuple[str, ...] = ("text",)) -> None:
    """Write sentences with the given columns (TOKEN is always first)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# columns = " + "\t".join(["TOKEN", *columns]) + "\n")
        for sent in sentences:
            fh.write(f"# sent_id = {sent.sent_id}\n")
            for key in meta_keys:
                if key in sent.meta:
                    value = " ".join(str(sent.meta[key]).split())
                    fh.write(f"# {key} = {value}\n")
            for i, tok in enumerate(sent.tokens):
                cells = [tok] + [sent.columns.get(c, [MISSING] * len(sent))[i] for c in columns]
                fh.write("\t".join(cells) + "\n")
            fh.write("\n")

