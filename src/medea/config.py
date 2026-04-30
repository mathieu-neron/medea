from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SEEDS_DIR = DATA_DIR / "seeds"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
CHROMA_DIR = DATA_DIR / "chroma"
MODELS_DIR = DATA_DIR / "models"
MLRUNS_DIR = DATA_DIR / "mlruns"
DB_PATH = DATA_DIR / "medea.db"

CLIP_DURATION_SECONDS = 30


def ensure_dirs() -> None:
    for d in (DATA_DIR, SEEDS_DIR, RAW_DIR, FEATURES_DIR, CHROMA_DIR, MODELS_DIR, MLRUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def to_repo_relative(path: Path | str) -> str:
    """Render a path as POSIX-style relative-to-repo-root if possible, else
    return the original string. Stored paths must be portable across machines —
    callers writing to SQLite or parquet should pass through this first."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(ROOT.resolve())
    except ValueError:
        # Path lives outside the repo (rare); keep the absolute form so the
        # reader can still find it on this machine.
        return str(p)
    return rel.as_posix()
