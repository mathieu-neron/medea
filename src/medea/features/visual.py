"""CLIP visual features.

Pipeline per clip: sample N evenly-spaced frames → preprocess → encode_image →
L2-normalize each → mean-pool → re-normalize → 512-dim float32 vector.

Why mean-pool + re-normalize: averaging unit vectors gives a non-unit result
whose magnitude correlates with frame agreement; we drop that magnitude so
downstream cosine similarity is purely directional.

Model:
    ViT-B/32 from open_clip, weights ``laion2b_s34b_b79k``.
    - open_clip:  https://github.com/mlfoundations/open_clip
    - weights:    https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
      (LAION-2B English subset, 34B samples seen, 79K batch size; 512-dim
      image/text embeddings.)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

log = logging.getLogger(__name__)

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
N_FRAMES = 8
EMBED_DIM = 512


def sample_frames(clip_path: Path, n: int = N_FRAMES) -> list[Image.Image]:
    """Return n evenly-spaced frames from a video file as PIL RGB Images."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, max(0, total - 1), num=n, dtype=int)
    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


class VisualEncoder:
    """Wraps open_clip ViT-B/32 for video-clip-level embedding.

    Load once, call ``embed_clip`` per video. Holds ~150MB on disk and ~1GB VRAM.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading open_clip %s/%s on %s", CLIP_MODEL, CLIP_PRETRAINED, self.device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    @torch.no_grad()
    def embed_clip(self, clip_path: Path, n_frames: int = N_FRAMES) -> np.ndarray:
        frames = sample_frames(clip_path, n_frames)
        if not frames:
            raise RuntimeError(f"no frames sampled from {clip_path}")
        batch = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        pooled = feats.mean(dim=0)
        pooled = pooled / pooled.norm()
        return pooled.cpu().numpy().astype(np.float32)
