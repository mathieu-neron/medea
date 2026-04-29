from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SEEDS_DIR = DATA_DIR / "seeds"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "medea.db"

CLIP_DURATION_SECONDS = 30


def ensure_dirs() -> None:
    for d in (DATA_DIR, SEEDS_DIR, RAW_DIR, FEATURES_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)
