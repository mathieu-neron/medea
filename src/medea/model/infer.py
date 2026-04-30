"""End-to-end inference: URL → 30s clip → 4-modality features → P(AI).

This is the single source of truth for inference logic. The CLI
(``scripts/predict.py``) and the FastAPI server both go through ``Predictor``
so the two interfaces can never drift.

The fused vector is built byte-for-byte the same way as
``features.pipeline.build_dataset`` — visual / transcript / title_desc each
L2-normalized, then concatenated with the raw scalar block in the same
column order as ``SCALAR_COLUMNS``. Any drift here would silently invalidate
the trained models.

Two trained models are loaded together:
  - MLP (``data/models/mlp.pt``) — primary scorer (M7, F1=0.87 channel-OOF).
  - LogReg (``data/models/logreg.joblib``) — used for feature attribution
    only. The MLP has no direct coefficients; LogReg's z·coef product is a
    reasonable linear proxy for which inputs drove the prediction.

Channel-level scalars (count, age, mean inter-upload) cannot be computed
from a single fresh URL — every training-time channel had 5–10 observed
videos, so a value of 1 / 0 / 0 here is a many-sigma input the classifier
never saw. We substitute the training-set means for those three columns so
their z-score is 0 and they contribute nothing to the prediction. The
alternative (fetching the channel's recent videos via yt-dlp) is correct
but costs ~10 extra HTTP round-trips per prediction; deferred to M9.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import torch

from medea.config import MODELS_DIR
from medea.features.metadata import channel_features, video_features
from medea.features.pipeline import SCALAR_COLUMNS
from medea.ingest.youtube import VideoMeta, download_middle_clip
from medea.model.mlp import MLP
from medea.storage.vector_db import (
    Neighbor,
    embedding_slice,
    get_collection,
    knn_query,
)

log = logging.getLogger(__name__)

LOGREG_PATH = MODELS_DIR / "logreg.joblib"
MLP_PATH = MODELS_DIR / "mlp.pt"

# Channel-derived scalars that can't be computed from a single inference URL —
# we override these with training-set means at predict-time. See module docstring.
_NEUTRALIZED_CHANNEL_COLS: tuple[str, ...] = (
    "channel_video_count_observed",
    "channel_age_days",
    "channel_mean_iud_days",
)


@dataclass
class Prediction:
    video_id: str
    title: str | None
    score: float                                 # P(AI) ∈ [0, 1]
    model: str                                   # "mlp" or "logreg"
    top_neighbors: list[Neighbor]
    modality_attribution: dict[str, float]       # share of |signed contribution|
    top_features: list[tuple[str, float]] = field(default_factory=list)


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_SHORTS_RE = re.compile(r"^/shorts/([A-Za-z0-9_-]{11})")


def extract_video_id(url_or_id: str) -> str:
    """Accept a bare 11-char video id or any common YouTube URL form."""
    s = url_or_id.strip()
    if _VIDEO_ID_RE.match(s):
        return s
    parsed = urlparse(s)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/")[0]
    if "youtube.com" in host:
        if parsed.path == "/watch":
            v = parse_qs(parsed.query).get("v", [])
            if v:
                return v[0]
        m = _SHORTS_RE.match(parsed.path)
        if m:
            return m.group(1)
    raise ValueError(f"could not extract video id from: {url_or_id!r}")


def _modality_starts(block_sizes: dict[str, int]) -> dict[str, int]:
    """Cumulative starting index per modality block. Same order as pipeline.py."""
    starts: dict[str, int] = {}
    cursor = 0
    for name in ("visual", "transcript", "title_desc", "scalars"):
        starts[name] = cursor
        cursor += block_sizes[name]
    return starts


class Predictor:
    """Loads the trained models + all four encoders and runs end-to-end inference.

    Heavy: loads CLIP, faster-whisper, wav2vec2, MiniLM, the LogReg pipeline
    and the MLP state dict. Construct once and reuse — the API server keeps
    one instance for the process lifetime.
    """

    def __init__(self, model: str = "mlp", device: str | None = None) -> None:
        if model not in ("mlp", "logreg"):
            raise ValueError(f"model must be 'mlp' or 'logreg', got {model!r}")
        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Predictor: loading encoders on %s", self.device)

        # Local imports so feature-extractor imports stay lazy (CLI users
        # who only call extract_video_id don't pay the encoder import cost).
        from medea.features.audio import AudioEncoder
        from medea.features.text import TextEncoder
        from medea.features.visual import VisualEncoder

        self.visual = VisualEncoder(device=self.device)
        self.audio = AudioEncoder(device=self.device)
        self.text = TextEncoder(device=self.device)

        log.info("Predictor: loading classifier(s)")
        self.logreg_payload = joblib.load(LOGREG_PATH)
        self.block_sizes: dict[str, int] = self.logreg_payload["block_sizes"]
        self.modality_starts = _modality_starts(self.block_sizes)

        self.mlp: MLP | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        if model == "mlp":
            payload = torch.load(MLP_PATH, map_location="cpu", weights_only=False)
            self.mlp = MLP(
                input_dim=payload["input_dim"],
                hidden=tuple(payload["hidden"]),
                dropout=payload["dropout"],
            )
            self.mlp.load_state_dict(payload["state_dict"])
            self.mlp.eval()
            self.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
            self.scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float32)

        # Cache the scalar-block means from the LogReg's StandardScaler so we
        # can neutralize untrustworthy inference-time channel scalars.
        scaler_mean = self.logreg_payload["pipeline"].named_steps["scaler"].mean_
        emb_dim = (
            self.block_sizes["visual"]
            + self.block_sizes["transcript"]
            + self.block_sizes["title_desc"]
        )
        self._scalar_block_mean = np.asarray(scaler_mean[emb_dim:], dtype=np.float32)

        self.collection = get_collection()
        log.info("Predictor ready (chroma rows=%d)", self.collection.count())

    def predict_url(self, url_or_id: str, *, k: int = 5) -> Prediction:
        video_id = extract_video_id(url_or_id)
        meta = download_middle_clip(video_id)
        if meta is None or meta.clip_path is None or not meta.clip_path.exists():
            raise RuntimeError(
                f"failed to download clip for {video_id} "
                "(too short, geo-blocked, or private)"
            )
        return self.predict_meta(meta, k=k)

    def predict_meta(self, meta: VideoMeta, *, k: int = 5) -> Prediction:
        if meta.clip_path is None:
            raise ValueError("VideoMeta.clip_path is required")
        fused = self._build_fused_vector(meta)

        score = self._score(fused)
        knn = knn_query(self.collection, fused, k=k)
        modality_attr, top_feats = self._attribute(fused)

        return Prediction(
            video_id=meta.id,
            title=meta.title,
            score=score,
            model=self.model_name,
            top_neighbors=knn.neighbors,
            modality_attribution=modality_attr,
            top_features=top_feats,
        )

    def _build_fused_vector(self, meta: VideoMeta) -> np.ndarray:
        assert meta.clip_path is not None
        visual_emb = self.visual.embed_clip(meta.clip_path)
        audio_feats = self.audio.features(meta.clip_path)
        text_feats = self.text.features(
            transcript=audio_feats.transcript,
            title=meta.title or "",
            description=meta.description or "",
        )

        scalars_dict: dict[str, float] = {"ai_voice_prob": float(audio_feats.ai_voice_prob)}
        scalars_dict.update(
            video_features(
                title=meta.title or "",
                description=meta.description or "",
                view_count=meta.view_count,
            )
        )
        # Single-video channel features → cadence stats are 0.0 by definition;
        # this is the documented inference-time gap vs ≥5-video training channels.
        scalars_dict.update(channel_features([meta.upload_date] if meta.upload_date else []))

        scalars = np.array(
            [scalars_dict[col] for col in SCALAR_COLUMNS], dtype=np.float32
        )
        scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)

        # Neutralize the channel scalars with training-set means (see module
        # docstring) so a fresh-URL count of 1 doesn't trigger a many-sigma
        # spurious "AI" signal.
        for col in _NEUTRALIZED_CHANNEL_COLS:
            idx = SCALAR_COLUMNS.index(col)
            scalars[idx] = self._scalar_block_mean[idx]

        v = _l2(visual_emb)
        t = _l2(text_feats.transcript_emb)
        h = _l2(text_feats.title_emb)
        return np.concatenate([v, t, h, scalars]).astype(np.float32)

    def _score(self, fused: np.ndarray) -> float:
        if self.model_name == "mlp":
            assert self.mlp is not None and self.scaler_mean is not None
            scale = np.where(self.scaler_scale > 0, self.scaler_scale, 1.0)
            z = (fused - self.scaler_mean) / scale
            with torch.no_grad():
                logit = self.mlp(torch.from_numpy(z.astype(np.float32)).unsqueeze(0))
            return float(torch.sigmoid(logit).item())
        pipe = self.logreg_payload["pipeline"]
        return float(pipe.predict_proba(fused.reshape(1, -1))[0, 1])

    def _attribute(
        self, fused: np.ndarray
    ) -> tuple[dict[str, float], list[tuple[str, float]]]:
        """Linear attribution via the LogReg model: contribution_i = z_i · coef_i.

        Always uses LogReg even when the active scorer is the MLP — gives a
        consistent, interpretable view of which inputs drove a similar
        decision under a linear model. Use it as a hint, not as the MLP's
        actual reasoning.
        """
        pipe = self.logreg_payload["pipeline"]
        scaler = pipe.named_steps["scaler"]
        logreg = pipe.named_steps["logreg"]
        z = scaler.transform(fused.reshape(1, -1))[0]
        contribs = z * logreg.coef_[0]

        modality_attr: dict[str, float] = {}
        for name in ("visual", "transcript", "title_desc", "scalars"):
            start = self.modality_starts[name]
            end = start + self.block_sizes[name]
            modality_attr[name] = float(np.abs(contribs[start:end]).sum())
        total = sum(modality_attr.values()) or 1.0
        modality_attr = {k: v / total for k, v in modality_attr.items()}

        names = self._feature_names()
        top_idx = np.argsort(-np.abs(contribs))[:10]
        top_feats = [(names[i], float(contribs[i])) for i in top_idx]
        return modality_attr, top_feats

    def _feature_names(self) -> list[str]:
        names: list[str] = []
        for i in range(self.block_sizes["visual"]):
            names.append(f"visual[{i}]")
        for i in range(self.block_sizes["transcript"]):
            names.append(f"transcript[{i}]")
        for i in range(self.block_sizes["title_desc"]):
            names.append(f"title_desc[{i}]")
        names.extend(SCALAR_COLUMNS)
        return names


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)
