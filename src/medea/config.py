from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SEEDS_DIR = DATA_DIR / "seeds"


def ensure_dirs() -> None:
    for d in (DATA_DIR, SEEDS_DIR):
        d.mkdir(parents=True, exist_ok=True)
