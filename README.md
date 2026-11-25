# scotus-pos

This repo is for Stage 1 of the Group 18 project: how well do news-trained part-of-speech taggers, and ensembles of them, work on U.S. Supreme Court opinions? So far it holds the legal gold set, the code that builds it, the seven taggers' output on it, and their accuracy (see Results). The ensembles will be added later.

## Layout

```
scotus_pos/          code: gold set, tagger loading, tagging runs
data/gold/           the gold set: dev.tsv, test.tsv and the annotation files
                     (data/gold/README.md has the tagging conventions)
data/tagged/         the legal dev/test and WSJ 24 files with one column per tagger
data/wsj/            WSJ sections 02-21 and 24 from Homework 3 (data/wsj/README.md)
data/raw/            the source opinions (not in git; see Setup)
models/              downloaded tagger models (not in git; see Setup)
```

## Setup

You need Python 3.10 or later. The gold-set code uses only the standard library. The taggers need some libraries and their model files (about 1.5 GB in all, mostly torch), and the Stanford tagger also needs Java 8 or later:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scotus_pos.download_models     # fills models/ and installs the spaCy models
python -m scotus_pos.taggers             # loads every tagger and tags a test sentence
```

The tagged files in `data/tagged/` are committed, so you only need this setup to re-run a tagger or add a new one. You don't need it to score or combine the existing output.

To re-sample the gold set, you also need the source corpus. Put `opinions_deduplicated.jsonl`, from the team's Hugging Face dataset, in `data/raw/`. 

## File format

Every `.tsv` file has one token per line, with tab-separated columns and a blank line between sentences, as in Homework 3. The first column is `TOKEN`, the second is `GOLD`, and the files in `data/tagged/` add one column per tagger. Comment lines name the columns and give each sentence's ID and text:

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
| `python -m scotus_pos.download_models` | Downloads the tagger models into `models/` and installs the two spaCy models. Skips anything already there. Name models (`nltk stanford spacy_sm spacy_trf flair stanza`) to fetch only those. |
| `python -m scotus_pos.taggers` | Loads every tagger and prints their tags for one legal sentence side by side. Reports which taggers fail to load. |
| `python -m scotus_pos.run_taggers` | Tags `data/gold/dev.tsv` and `test.tsv` and writes `data/tagged/dev.tsv` and `test.tsv` with one column per tagger. `--taggers flair stanza` runs only those and keeps the other columns. `--split wsj24` tags WSJ section 24 for the in-domain comparison. `--input` / `--output` tag any other column file. |
| `python -m scotus_pos.stats` | Prints each tagger's token accuracy on WSJ 24 and legal dev as a Markdown table (the Results table below). `--split` picks other files; scoring the legal test set needs `--final`. |
| `python -m scotus_pos.build_gold` | Builds `data/gold/dev.tsv` and `test.tsv` from the annotation files. It checks that every tag is a valid Penn Treebank tag and that no tokens were edited, and writes nothing if anything is untagged. `--allow-incomplete` writes a partial set instead. |

Add `--help` to any command to see all its options.

## Modules

| File | Contents |
|---|---|
| `sample_gold.py` | Filters out unusable sentences, splits opinions by era, samples sentences evenly across opinions, and divides the annotation work among annotators |
| `build_gold.py` | Merges the annotation files into the gold dev and test sets and checks them |
| `taggers.py` | Loads each pretrained tagger behind one method, `tag_sents(list of token lists) -> list of tag lists`. Taggers receive the gold tokens, so there is no re-tokenizing or alignment. Tags come back raw; pass them through `normalize_tag` before comparing. |
| `bigram_hmm_hw3.py` | The Homework 3 bigram HMM with Viterbi decoding. `taggers.py` trains it on WSJ 02-21 when it is loaded, which takes about a second |
| `download_models.py` | Fetches the model files |
| `run_taggers.py` | Adds tagger columns to a column file |
| `stats.py` | Scores the tagged files against gold |
| `io.py` | `read_columns` / `write_columns` for the `.tsv` format above, and the `Sentence` class |
| `tagset.py` | The Penn Treebank tagset, and `normalize_tag`, which maps OntoNotes/UD extras (`HYPH`, `NFP`, `(`, …) onto it |
| `paths.py` | Default file locations |

## Results

Token accuracy of each tagger, from `python -m scotus_pos.stats`. Gold and predicted tags are both passed through `normalize_tag` first. The legal test set has not been scored.

| Tagger | Trained on | WSJ 24 | Legal dev | Change |
|---|---|---:|---:|---:|
| NLTK | WSJ | 0.963 | 0.942 | −2.1 |
| Stanford | WSJ | 0.956 | 0.942 | −1.4 |
| spaCy small | OntoNotes 5 | 0.957 | 0.953 | −0.4 |
| spaCy transformer | OntoNotes 5 | 0.960 | 0.962 | +0.2 |
| Flair | OntoNotes 5 | 0.975 | 0.954 | −2.1 |
| Stanza | UD English | 0.955 | 0.956 | +0.1 |
| HMM (Homework 3) | WSJ 02-21 | 0.949 | 0.920 | −2.9 |

Tokens scored: WSJ 24 32,853, legal dev 4,427. Change is in percentage points.

- The HMM drops the most on legal text (2.9 points), then NLTK and Flair (2.1 each). spaCy transformer and Stanza hold steady.
- Legal dev is small: with about 4,400 tokens, a 95% confidence interval is roughly ±0.6 points, so differences of under a point between taggers are not reliable yet.
- OntoNotes 5 includes WSJ text, so the spaCy and Flair WSJ 24 scores may be inflated if section 24 was in their training data. Flair's 0.975 suggests it was.

## Rules for the test set

Do all tuning, weight fitting and error analysis on `dev.tsv`. Score each final system on `test.tsv` once.
