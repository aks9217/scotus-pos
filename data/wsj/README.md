# data/wsj/

The WSJ files from Homework 3 (`WSJ_POS_CORPUS_FOR_STUDENTS`), unchanged.

| File | Use |
|---|---|
| `WSJ_02-21.pos` | HMM training (`taggers.py` trains the HMM on this file only) |
| `WSJ_24.pos` | in-domain comparison for research question 1: every tagger is scored on it with `python -m scotus_pos.stats` |

Section 24 is held out from HMM training, so its score is a fair in-domain number for the HMM. Homework 3's own `main()` also trained on section 24; `taggers.py` does not.
