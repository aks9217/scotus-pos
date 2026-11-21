"""Default locations, relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_OPINIONS = ROOT / "data" / "raw" / "opinions_deduplicated.jsonl"
GOLD_DIR = ROOT / "data" / "gold"
