"""Download the pretrained tagger models into models/.

    python -m scotus_pos.download_models            # all of them
    python -m scotus_pos.download_models nltk stanza

Each step is skipped if its files are already there. spaCy models are pip
packages, so they install into the active Python environment rather than
models/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .paths import MODELS_DIR
from .taggers import FLAIR_DIR, NLTK_DIR, STANFORD_DIR, STANFORD_JAR, STANFORD_MODEL, STANZA_DIR

STANFORD_URL = "https://downloads.cs.stanford.edu/nlp/software/stanford-tagger-4.2.0.zip"


def nltk() -> None:
    import nltk as _nltk

    if not _nltk.download("averaged_perceptron_tagger_eng", download_dir=str(NLTK_DIR), quiet=True):
        raise RuntimeError("nltk.download failed (see the [nltk_data] message above)")


def stanford() -> None:
    if STANFORD_JAR.exists() and STANFORD_MODEL.exists():
        return
    STANFORD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "stanford-tagger.zip"
        print(f"  downloading {STANFORD_URL} (78 MB)")
        urllib.request.urlretrieve(STANFORD_URL, archive)
        # Keep only the jar and the WSJ model (the zip also has sources, javadoc, other models).
        wanted = {
            "stanford-postagger.jar": STANFORD_JAR,
            "models/english-left3words-distsim.tagger": STANFORD_MODEL,
            "models/english-left3words-distsim.tagger.props": STANFORD_MODEL.with_suffix(".tagger.props"),
        }
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                inner = member.split("/", 1)[-1]  # drop the top-level folder
                if inner in wanted:
                    with zf.open(member) as src, wanted[inner].open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    missing = [p.name for p in (STANFORD_JAR, STANFORD_MODEL) if not p.exists()]
    if missing:
        raise RuntimeError(f"Stanford zip did not contain {missing}")


def _spacy(package: str) -> None:
    import spacy.util
    from spacy.cli import download

    if not spacy.util.is_package(package):
        download(package)


def spacy_sm() -> None:
    _spacy("en_core_web_sm")


def spacy_trf() -> None:
    _spacy("en_core_web_trf")


def flair() -> None:
    os.environ.setdefault("FLAIR_CACHE_ROOT", str(FLAIR_DIR))
    from flair.models import SequenceTagger

    SequenceTagger.load("flair/pos-english")  # downloads on first load, then cached


def stanza() -> None:
    import stanza as _stanza

    _stanza.download("en", model_dir=str(STANZA_DIR), processors="tokenize,mwt,pos", logging_level="WARN")


STEPS = {
    "nltk": nltk,
    "stanford": stanford,
    "spacy_sm": spacy_sm,
    "spacy_trf": spacy_trf,
    "flair": flair,
    "stanza": stanza,
}


def _use_system_certificates() -> None:
    """python.org builds of Python on macOS ship without CA certificates, so
    HTTPS downloads fail with CERTIFICATE_VERIFY_FAILED. truststore makes Python
    use the operating system's certificate store instead."""
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", metavar="MODEL",
                    help=f"which models to fetch (default: all). Choices: {', '.join(STEPS)}")
    args = ap.parse_args()
    unknown = [m for m in args.models if m not in STEPS]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; choose from {', '.join(STEPS)}")
    _use_system_certificates()
    MODELS_DIR.mkdir(exist_ok=True)
    failed = []
    for name in args.models or STEPS:
        print(f"{name} ...")
        try:
            STEPS[name]()
            print(f"  ok")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failed.append(name)
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
