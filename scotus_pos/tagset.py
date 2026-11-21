"""Penn Treebank tagset and the mapping of tagger-specific tags onto it.

Every tag (gold and predicted) passes through ``normalize_tag`` before it is
compared, so brackets, OntoNotes extras and UD-EWT extras are scored
consistently. Each mapping below is a project decision; record any change in
docs/annotation_guidelines.md.
"""

from __future__ import annotations

PTB_TAGS: frozenset[str] = frozenset(
    """CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PRP$ RB RBR RBS RP
    SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WRB # $ '' `` , . : -LRB- -RRB-""".split()
)

TAG_MAP: dict[str, str] = {
    # Bracket tags: NLTK and Homework 3 files use "(" / ")", the Treebank uses -LRB- / -RRB-.
    "(": "-LRB-", ")": "-RRB-",
    "[": "-LRB-", "]": "-RRB-", "{": "-LRB-", "}": "-RRB-",
    "-LSB-": "-LRB-", "-RSB-": "-RRB-", "-LCB-": "-LRB-", "-RCB-": "-RRB-",
    # OntoNotes 5 / UD English EWT tags that WSJ does not have (spaCy, Flair, Stanza).
    "HYPH": ":",   # WSJ tags a free-standing hyphen or dash as ":"
    "NFP": ":",    # superfluous punctuation ("***", "...")
    "AFX": "JJ",   # bound prefix ("pre", "anti") -> treated as modifier
    "ADD": "NN",   # e-mail / URL
    "XX": "FW",    # unknown / partial word
    "GW": "FW",    # "goes with" fragment of a split word
    "_SP": ":",    # whitespace token; should not occur with pretokenized input
    '""': "''",
}


def normalize_tag(tag: str) -> str:
    tag = tag.strip()
    return TAG_MAP.get(tag, tag)

