# scotus-pos

This repo is for Stage 1 of the Group 18 project: how well do news-trained part-of-speech taggers, and ensembles of them, work on U.S. Supreme Court opinions? So far it holds the legal gold set and the code that builds it. The taggers and ensembles will be added later.

## Layout

```
scotus_pos/          code that samples, splits and assembles the gold set
data/gold/           the gold set: dev.tsv, test.tsv and the annotation files
                     (data/gold/README.md has the tagging conventions)
data/raw/            the source opinions (not in git; see Setup)
```

## Setup

You need Python 3.10 or later. The code uses only the standard library, so there is nothing to install.

To re-sample the gold set, you also need the source corpus. Put `opinions_deduplicated.jsonl`, from the team's Hugging Face dataset, in `data/raw/`. It is kept out of git because it is 42 MB. You don't need it just to use `dev.tsv` and `test.tsv`.

## File format

Every `.tsv` file has one token per line, with tab-separated columns and a blank line between sentences, as in Homework 3. The first column is `TOKEN`, the second is `GOLD`, and taggers will later add one column each. Comment lines name the columns and give each sentence's ID and text:

```
# columns = TOKEN	GOLD
# sent_id = wol-100009-s00065
# text = Atlantic Transport Co. v. Imbrovek, 234 U.S. 52, 59, 60.
Atlantic	NNP
Transport	NNP
Co.	NNP
v.	IN
...
```

`_` means the token has no tag yet. A sentence ID such as `wol-100009-s00065` starts with the ID of its opinion (`wol-100009`), which you can use to group sentences by opinion.

## Commands

| Command | What it does |
|---|---|
| `python -m scotus_pos.sample_gold` | Samples 500 sentences from 90 opinions and splits them by opinion into dev (150) and test (350). Writes the manifest and the annotation files. Always picks the same sentences (`--seed 18`). It won't overwrite annotation files that already contain tags unless you pass `--force`. |
| `python -m scotus_pos.build_gold` | Builds `data/gold/dev.tsv` and `test.tsv` from the annotation files. It checks that every tag is a valid Penn Treebank tag and that no tokens were edited, and writes nothing if anything is untagged. `--allow-incomplete` writes a partial set instead. |

Add `--help` to either command to see all its options, such as the number of sentences, annotator names and length limits.

## Modules

| File | Contents |
|---|---|
| `sample_gold.py` | Filters out unusable sentences, splits opinions by era, samples sentences evenly across opinions, and divides the annotation work among annotators |
| `build_gold.py` | Merges the annotation files into the gold dev and test sets and checks them |
| `io.py` | `read_columns` / `write_columns` for the `.tsv` format above, and the `Sentence` class |
| `tagset.py` | The Penn Treebank tagset, and `normalize_tag`, which maps OntoNotes/UD extras (`HYPH`, `NFP`, `(`, …) onto it |
| `paths.py` | Default file locations |

## Rules for the test set

Do all tuning, weight fitting and error analysis on `dev.tsv`. Score each final system on `test.tsv` once.
