"""Idempotent parquet stores for per-clip features, keyed by ``video_id``.

Each modality writes one file under ``data/features/``. ``append_rows`` upserts
on ``video_id`` so partial re-runs don't duplicate work.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from medea.config import FEATURES_DIR

VISUAL_PARQUET = FEATURES_DIR / "visual.parquet"
AUDIO_PARQUET = FEATURES_DIR / "audio.parquet"
TEXT_PARQUET = FEATURES_DIR / "text.parquet"
METADATA_PARQUET = FEATURES_DIR / "metadata.parquet"
COMBINED_PARQUET = FEATURES_DIR / "combined.parquet"


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def existing_ids(path: Path) -> set[str]:
    df = load_table(path)
    if df.empty or "video_id" not in df.columns:
        return set()
    return set(df["video_id"].astype(str).tolist())


def append_rows(path: Path, new_rows: list[dict]) -> None:
    """Upsert ``new_rows`` into ``path``. Replaces any existing rows with the same video_id."""
    if not new_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_parquet(path)
        if "video_id" in old.columns:
            old = old[~old["video_id"].isin(new_df["video_id"])]
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_parquet(path, index=False)
