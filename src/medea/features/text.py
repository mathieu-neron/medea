"""Text features: transcript + title/description embeddings.

For 30s clips, transcripts cap out around 600 chars in our dataset — well
under MiniLM's 256-token limit — so we embed each in one shot. Title +
description together can exceed the limit; the model truncates internally,
which is fine: the head of the description carries most of the signal.

Empty transcripts (~25% of our clips have no detected speech) yield a zero
vector, so downstream concat shapes stay constant.

Model:
    sentence-transformers/all-MiniLM-L6-v2 — 384-dim, ~80MB.
        - card:    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
          Distilled MiniLM trained on 1B+ sentence pairs; fast on CPU, faster
          on GPU. Returns unit vectors when ``normalize_embeddings=True``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


@dataclass
class TextFeatures:
    transcript_emb: np.ndarray  # 384-dim float32, L2-normed (zeros if empty)
    title_emb: np.ndarray       # 384-dim float32, L2-normed (title + description)


class TextEncoder:
    """Sentence-transformers MiniLM-L6-v2 wrapped for batch text encoding.

    Holds ~80MB of weights. Cheap to load; even cheaper if the audio encoder
    has already pulled torch + transformers into memory upstream.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading sentence-transformers %s on %s", TEXT_MODEL, self.device)
        self.model = SentenceTransformer(TEXT_MODEL, device=self.device)

    def _encode_nonempty(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(EMBED_DIM, dtype=np.float32)
        return self._encode_nonempty([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode many strings, preserving order. Empty strings become zero rows."""
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        keep = [i for i, t in enumerate(texts) if t and t.strip()]
        if keep:
            out[keep] = self._encode_nonempty([texts[i] for i in keep])
        return out

    def features(self, *, transcript: str, title: str, description: str) -> TextFeatures:
        title = (title or "").strip()
        description = (description or "").strip()
        title_desc = f"{title}\n{description}".strip() if description else title
        return TextFeatures(
            transcript_emb=self.encode_one(transcript or ""),
            title_emb=self.encode_one(title_desc),
        )
