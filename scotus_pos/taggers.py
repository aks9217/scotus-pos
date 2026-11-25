"""The pretrained POS taggers, loaded behind one interface.

Every tagger takes sentences that are already tokenized (the gold
tokenization) and returns one tag per token, so no tagger re-tokenizes and
there is nothing to align afterwards:

    from scotus_pos.taggers import load_tagger
    tagger = load_tagger("NLTK")
    tagger.tag_sents([["Coe", "v.", "Errol", ",", "116", "U.S.", "517", "."]])
    # -> [["NNP", "CC", "NNP", ",", "CD", "NNP", "CD", "."]]

Tags are returned exactly as the tagger produced them (OntoNotes/UD extras
such as HYPH or NFP included). Map them with ``tagset.normalize_tag`` before
scoring or voting.

The model files live in ``models/`` (run ``python -m scotus_pos.download_models``
first). Each library is imported only when its tagger is loaded, so you only
need the libraries for the taggers you actually use.

Run ``python -m scotus_pos.taggers`` to load every tagger and tag a sample
sentence.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from .paths import MODELS_DIR, WSJ_DIR

STANFORD_DIR = MODELS_DIR / "stanford"
STANFORD_JAR = STANFORD_DIR / "stanford-postagger.jar"
STANFORD_MODEL = STANFORD_DIR / "english-left3words-distsim.tagger"
NLTK_DIR = MODELS_DIR / "nltk_data"
STANZA_DIR = MODELS_DIR / "stanza"
FLAIR_DIR = MODELS_DIR / "flair"

# Point Flair at models/ before it is imported anywhere.
os.environ.setdefault("FLAIR_CACHE_ROOT", str(FLAIR_DIR))


class Tagger:
    """Base class. ``name`` is the column the tagger writes in the .tsv files."""

    name: str = ""
    model: str = ""

    def tag_sents(self, sents: list[list[str]]) -> list[list[str]]:
        tags = self._tag_sents(sents)
        if len(tags) != len(sents):
            raise RuntimeError(f"{self.name}: got {len(tags)} sentences back, sent {len(sents)}")
        for i, (toks, out) in enumerate(zip(sents, tags)):
            if len(toks) != len(out):
                raise RuntimeError(
                    f"{self.name}: sentence {i} has {len(toks)} tokens but {len(out)} tags: {toks}"
                )
        return tags

    def _tag_sents(self, sents: list[list[str]]) -> list[list[str]]:
        raise NotImplementedError


class NLTKTagger(Tagger):
    name = "NLTK"
    model = "averaged_perceptron_tagger_eng (WSJ)"

    def __init__(self) -> None:
        import nltk
        from nltk.tag.perceptron import PerceptronTagger

        if str(NLTK_DIR) not in nltk.data.path:
            nltk.data.path.insert(0, str(NLTK_DIR))
        self._tagger = PerceptronTagger()

    def _tag_sents(self, sents):
        return [[tag for _, tag in self._tagger.tag(toks)] for toks in sents]


class StanfordTagger(Tagger):
    """Stanford's MaxentTagger, run as one Java process per call."""

    name = "STANFORD"
    model = "english-left3words-distsim.tagger (WSJ)"

    def __init__(self, memory: str = "2g") -> None:
        for f in (STANFORD_JAR, STANFORD_MODEL):
            if not f.exists():
                raise FileNotFoundError(f"{f} is missing; run python -m scotus_pos.download_models stanford")
        self.memory = memory

    def _tag_sents(self, sents):
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "input.txt"
            # One sentence per line, tokens separated by spaces (gold tokens contain no spaces).
            inp.write_text("".join(" ".join(toks) + "\n" for toks in sents), encoding="utf-8")
            cmd = [
                "java", f"-mx{self.memory}", "-cp", str(STANFORD_JAR),
                "edu.stanford.nlp.tagger.maxent.MaxentTagger",
                "-model", str(STANFORD_MODEL),
                "-textFile", str(inp),
                "-tokenize", "false",
                "-sentenceDelimiter", "newline",
                "-outputFormat", "tsv",
                "-encoding", "utf-8",
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if out.returncode != 0:
            raise RuntimeError(f"Stanford tagger failed:\n{out.stderr[-2000:]}")
        # tsv output: "token<TAB>tag" per line, blank line after each sentence.
        tags: list[list[str]] = []
        current: list[str] = []
        for line in out.stdout.splitlines():
            if line.strip():
                current.append(line.rsplit("\t", 1)[-1])
            elif current:
                tags.append(current)
                current = []
        if current:
            tags.append(current)
        return tags


class SpacyTagger(Tagger):
    """spaCy, fed a pre-built Doc so its own tokenizer never runs."""

    def __init__(self, package: str, name: str) -> None:
        import spacy

        self.name = name
        self.model = f"{package} (OntoNotes 5)"
        # Only the tagger (and the tok2vec/transformer it listens to) is needed.
        self._nlp = spacy.load(package, disable=["parser", "ner", "lemmatizer", "senter"])

    def _tag_sents(self, sents, batch_size: int = 64):
        from spacy.tokens import Doc

        docs = (Doc(self._nlp.vocab, words=toks) for toks in sents)
        return [[t.tag_ for t in doc] for doc in self._nlp.pipe(docs, batch_size=batch_size)]


class FlairTagger(Tagger):
    name = "FLAIR"
    model = "flair/pos-english (OntoNotes 5)"

    def __init__(self) -> None:
        from flair.models import SequenceTagger

        self._tagger = SequenceTagger.load("flair/pos-english")

    def _tag_sents(self, sents, batch_size: int = 32):
        from flair.data import Sentence

        flair_sents = [Sentence(list(toks)) for toks in sents]  # a list of strings = pretokenized
        self._tagger.predict(flair_sents, mini_batch_size=batch_size)
        return [[tok.get_label("pos").value for tok in s] for s in flair_sents]


class StanzaTagger(Tagger):
    name = "STANZA"
    model = "Stanza English default package, xpos (UD English)"

    def __init__(self) -> None:
        import stanza

        self._nlp = stanza.Pipeline(
            "en",
            dir=str(STANZA_DIR),
            processors="tokenize,mwt,pos",  # the English package requires mwt
            tokenize_pretokenized=True,
            download_method=None,  # never download at load time; use download_models
            logging_level="WARN",
        )

    def _tag_sents(self, sents):
        doc = self._nlp([list(toks) for toks in sents])
        # mwt may split one of our tokens into several words ("Court’s" -> "Court" + "’s").
        # Keep one tag per token: the first word's, matching the gold rule "tag by the first word".
        return [[tok.words[0].xpos for tok in s.tokens] for s in doc.sentences]


class HMMTagger(Tagger):
    """The Homework 3 bigram HMM (scotus_pos/bigram_hmm_hw3.py), trained on WSJ at load time.

    Trains on sections 02-21 only, so section 24 stays held out for the
    in-domain comparison. Training takes about a second, so nothing is cached.
    """

    name = "HMM"

    def __init__(self, train_files: list[Path] | None = None) -> None:
        from . import bigram_hmm_hw3 as hmm

        self._hmm = hmm
        train_files = train_files or [WSJ_DIR / "WSJ_02-21.pos"]
        self.model = "bigram HMM, Homework 3 (WSJ " + ", ".join(f.stem.removeprefix("WSJ_") for f in train_files) + ")"
        words: list[str] = []
        pos: list[str] = []
        for f in train_files:
            if not f.exists():
                raise FileNotFoundError(f"{f} is missing; see data/wsj/README.md")
            with f.open(encoding="utf-8") as fh:
                w, p = hmm.parseFile(fh, False)
            words += w + [""]  # "" marks a sentence boundary, as in the homework's main()
            pos += p + [""]
        likelihood, transition, self.vocabulary = hmm.countFrequencies(words, pos)
        self._emission, self._transition = hmm.getProbabilities(likelihood, transition)

    def _tag_sents(self, sents):
        return [self._hmm.transduceSentence(self._emission, self._transition, self.vocabulary, list(toks))
                for toks in sents]


# Column name -> factory. Order is the column order in the tagged files.
TAGGERS = {
    "NLTK": NLTKTagger,
    "STANFORD": StanfordTagger,
    "SPACY_SM": lambda: SpacyTagger("en_core_web_sm", "SPACY_SM"),
    "SPACY_TRF": lambda: SpacyTagger("en_core_web_trf", "SPACY_TRF"),
    "FLAIR": FlairTagger,
    "STANZA": StanzaTagger,
    "HMM": HMMTagger,
}


def load_tagger(name: str) -> Tagger:
    try:
        factory = TAGGERS[name.upper()]
    except KeyError:
        raise KeyError(f"unknown tagger {name!r}; choose from {', '.join(TAGGERS)}") from None
    return factory()


def _demo() -> None:
    """Load every tagger and tag one legal sentence, side by side."""
    sentence = "Coe v. Errol , 116 U.S. 517 , certiorari was denied under 28 U.S.C. § 1404 ( a ) .".split()
    results: dict[str, list[str]] = {}
    for name in TAGGERS:
        start = time.perf_counter()
        try:
            tagger = load_tagger(name)
            results[name] = tagger.tag_sents([sentence])[0]
            print(f"  ok    {name:<10} {time.perf_counter() - start:5.1f}s  {tagger.model}")
        except Exception as exc:  # report every tagger, even if one fails
            print(f"  FAIL  {name:<10} {type(exc).__name__}: {exc}")
    if results:
        width = max(map(len, sentence)) + 2
        print("\n" + "TOKEN".ljust(width) + "".join(n.ljust(11) for n in results))
        for i, tok in enumerate(sentence):
            print(tok.ljust(width) + "".join(results[n][i].ljust(11) for n in results))


if __name__ == "__main__":
    _demo()
