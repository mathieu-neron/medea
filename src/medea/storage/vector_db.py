"""Chroma persistent collection of per-video embeddings for kNN similarity search.

We store the **embedding-only** slice of the fused vector (visual + transcript +
title_desc, 1280-dim — see ``features.pipeline.SCALAR_COLUMNS`` for the trailing
scalars we deliberately omit). The handcrafted scalar block is meant for the
classifier head in M6, not for "what looks like this video?" geometry: under
cosine distance the unscaled scalars (e.g. ``view_count_log``, ``channel_age_days``)
would drown out the L2-normed embedding blocks.

Cosine space is configured at collection-creation time; once a collection
exists Chroma ignores the metadata kwarg, so changes here only apply to a
fresh ``data/chroma/`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from medea.config import CHROMA_DIR
from medea.features.pipeline import SCALAR_COLUMNS

COLLECTION_NAME = "videos"
# Block sizes from features.pipeline. Keep in sync if those change.
EMBED_DIMS = 512 + 384 + 384  # visual + transcript + title_desc
ASSERT_FUSED_DIM = EMBED_DIMS + len(SCALAR_COLUMNS)  # cross-check at runtime


@dataclass
class Neighbor:
    video_id: str
    label: int
    channel_id: int
    distance: float


@dataclass
class KnnResult:
    score: float           # cosine-weighted P(AI) ∈ [0, 1]
    neighbors: list[Neighbor]


def _client() -> chromadb.api.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection() -> Collection:
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> Collection:
    """Drop and recreate the collection — used when re-populating from scratch."""
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 — collection didn't exist
        pass
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embedding_slice(fused: np.ndarray) -> np.ndarray:
    """Drop the trailing scalar block from a fused (..., 1296) matrix or
    (1296,) vector and return only the embedding portion."""
    if fused.ndim == 1:
        if fused.shape[0] != ASSERT_FUSED_DIM:
            raise ValueError(
                f"fused vector has {fused.shape[0]} dims, expected {ASSERT_FUSED_DIM}"
            )
        return fused[:EMBED_DIMS].astype(np.float32, copy=False)
    if fused.shape[1] != ASSERT_FUSED_DIM:
        raise ValueError(
            f"fused matrix has {fused.shape[1]} cols, expected {ASSERT_FUSED_DIM}"
        )
    return fused[:, :EMBED_DIMS].astype(np.float32, copy=False)


def upsert_videos(
    collection: Collection,
    *,
    video_ids: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    channel_ids: np.ndarray,
) -> None:
    if embeddings.shape[1] != EMBED_DIMS:
        raise ValueError(
            f"expected {EMBED_DIMS}-dim embeddings, got {embeddings.shape[1]}"
        )
    metadatas = [
        {"label": int(lbl), "channel_id": int(cid)}
        for lbl, cid in zip(labels, channel_ids)
    ]
    collection.upsert(
        ids=list(video_ids),
        embeddings=embeddings.astype(np.float32).tolist(),
        metadatas=metadatas,
    )


def knn_query(
    collection: Collection,
    query_vec: np.ndarray,
    *,
    k: int = 5,
    exclude_ids: list[str] | None = None,
) -> KnnResult:
    """Cosine-weighted kNN. Returns P(AI)-style score and the neighbor list.

    Score formula: weighted mean of neighbor labels with weight = (1 - distance).
    Cosine distance is in [0, 2]; for L2-normed unit vectors of similar magnitude
    it stays close to [0, 1] in practice. Negative weights are clipped.
    """
    q = embedding_slice(query_vec) if query_vec.shape[-1] == ASSERT_FUSED_DIM else query_vec
    q = np.asarray(q, dtype=np.float32).reshape(1, -1)

    n_results = k + (len(exclude_ids) if exclude_ids else 0)
    res = collection.query(
        query_embeddings=q.tolist(),
        n_results=n_results,
        include=["metadatas", "distances"],
    )
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    neighbors: list[Neighbor] = []
    for vid, meta, dist in zip(ids, metas, dists):
        if exclude_ids and vid in exclude_ids:
            continue
        neighbors.append(
            Neighbor(
                video_id=vid,
                label=int(meta["label"]),
                channel_id=int(meta["channel_id"]),
                distance=float(dist),
            )
        )
        if len(neighbors) >= k:
            break

    if not neighbors:
        return KnnResult(score=float("nan"), neighbors=[])

    weights = np.clip([1.0 - n.distance for n in neighbors], 0.0, None)
    if weights.sum() == 0:
        score = float(np.mean([n.label for n in neighbors]))
    else:
        score = float(
            np.sum(weights * np.array([n.label for n in neighbors])) / weights.sum()
        )
    return KnnResult(score=score, neighbors=neighbors)
