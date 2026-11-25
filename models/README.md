# models/

Downloaded tagger models. Everything here except this file is kept out of git; run

    python -m scotus_pos.download_models

to fill it. The code that loads these models is `scotus_pos/taggers.py`.

| Folder | Tagger | Size |
|---|---|---|
| `nltk_data/` | NLTK `averaged_perceptron_tagger_eng` | 7 MB |
| `stanford/` | Stanford `english-left3words-distsim.tagger` and `stanford-postagger.jar` (4.2.0) | 19 MB |
| `flair/` | Flair `flair/pos-english` | 240 MB |
| `stanza/` | Stanza English default package (tokenize, mwt, pos) | 185 MB |

The spaCy models (`en_core_web_sm`, `en_core_web_trf`) are pip packages, so they install into the Python environment instead of this folder.
