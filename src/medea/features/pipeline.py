"""Multi-modal feature fusion: per-modality L2-normalize → concatenate.

Pulls together the four per-modality parquets and produces one feature
vector per video for the downstream classifier (M6/M7).

Block ordering is fixed so downstream code can slice by modality:

    visual (512)  +  transcript (384)  +  title_desc (384)  +  scalars (1 + M)

Each embedding block is L2-normalized so cross-modal magnitudes don't
dominate the cosine geometry. The trailing scalar block (``ai_voice_prob``
plus the handcrafted metadata) keeps raw values — those will be standardized
by the classifier-side scaler in M6, where the scale is fit on the train
split, not the whole dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from medea.features.store import (
    AUDIO_PARQUET,
    COMBINED_PARQUET,
    METADATA_PARQUET,
    TEXT_PARQUET,
    VISUAL_PARQUET,
    load_table,
)
from medea.storage.db import connect

# Order is meaningful: must match downstream classifier expectations.
SCALAR_COLUMNS: tuple[str, ...] = (
    "ai_voice_prob",
    "title_len",
    "title_word_count",
    "title_caps_ratio",
    "title_excl_count",
    "title_quest_count",
    "title_emoji_count",
    "title_digit_count",
    "title_clickbait",
    "desc_len",
    "desc_url_count",
    "desc_hashtag_count",
    "view_count_log",
    "channel_video_count_observed",
    "channel_age_days",
    "channel_mean_iud_days",
)


@dataclass
class FusedDataset:
    video_ids: list[str]
    channel_ids: np.ndarray   # int (N,) — for channel-level CV split in M6
    labels: np.ndarray        # int (N,)
    X: np.ndarray             # float32 (N, D)
    block_sizes: dict[str, int]


def _l2(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


def _stack(col_of_lists: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=np.float32) for x in col_of_lists])


def build_dataset() -> FusedDataset:
    visual = load_table(VISUAL_PARQUET)
    audio = load_table(AUDIO_PARQUET)
    text = load_table(TEXT_PARQUET)
    meta = load_table(METADATA_PARQUET)

    missing = [
        name
        for name, df in (("visual", visual), ("audio", audio), ("text", text), ("metadata", meta))
        if df.empty
    ]
    if missing:
        raise RuntimeError(
            f"missing feature parquets: {missing}. "
            "run: python scripts/extract.py --modality all"
        )

    with connect() as conn:
        labels_df = pd.read_sql_query(
            "SELECT id AS video_id, channel_id, label FROM videos", conn
        )

    df = (
        labels_df
        .merge(visual.rename(columns={"embedding": "visual_emb"}), on="video_id")
        .merge(audio[["video_id", "ai_voice_prob"]], on="video_id")
        .merge(text, on="video_id")
        .merge(meta, on="video_id")
    )

    visual_block = _l2(_stack(df["visual_emb"]))
    transcript_block = _l2(_stack(df["transcript_emb"]))
    title_block = _l2(_stack(df["title_emb"]))

    scalars = df[list(SCALAR_COLUMNS)].astype(np.float32).to_numpy()
    scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)

    X = np.concatenate(
        [visual_block, transcript_block, title_block, scalars], axis=1
    ).astype(np.float32)

    return FusedDataset(
        video_ids=df["video_id"].tolist(),
        channel_ids=df["channel_id"].astype(int).to_numpy(),
        labels=df["label"].astype(int).to_numpy(),
        X=X,
        block_sizes={
            "visual": visual_block.shape[1],
            "transcript": transcript_block.shape[1],
            "title_desc": title_block.shape[1],
            "scalars": scalars.shape[1],
        },
    )


def write_combined() -> FusedDataset:
    ds = build_dataset()
    out = pd.DataFrame(
        {
            "video_id": ds.video_ids,
            "channel_id": ds.channel_ids,
            "label": ds.labels,
            "embedding": list(ds.X),
        }
    )
    COMBINED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(COMBINED_PARQUET, index=False)
    return ds
