"""Default locations, relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_OPINIONS = ROOT / "data" / "raw" / "opinions_deduplicated.jsonl"
GOLD_DIR = ROOT / "data" / "gold"
MODELS_DIR = ROOT / "models"
TAGGED_DIR = ROOT / "data" / "tagged"
WSJ_DIR = ROOT / "data" / "wsj"
