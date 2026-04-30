"""Locally-generated rationale for a Prediction.

The plan's "stretch RAG" goal called for an LLM to write the rationale, but
this project is local-only — no external API integrations. Instead this
module produces a deterministic, neighbor-grounded explanation directly from
the data already on the Prediction object: neighbor labels and channels,
modality attribution, signed top-feature contributions, and the temporal
prior state.

Output is shorter and less fluent than an LLM summary would be, but it has
real upsides: zero cost, zero network, fully reproducible, and every claim
ties back to a concrete number we can point at. Useful as a confidence
check — the rationale flags when the model and the neighbor labels disagree,
which is the OOD warning sign we keep encountering.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from medea.storage.db import connect

if TYPE_CHECKING:
    from medea.model.infer import Prediction


def _channel_handles(channel_ids: list[int]) -> dict[int, str]:
    """Look up handles from SQLite for the given channel ids. Falls back to
    URL-derived names where ``handle`` is null."""
    if not channel_ids:
        return {}
    placeholders = ",".join("?" for _ in channel_ids)
    with connect() as conn:
        rows = list(
            conn.execute(
                f"SELECT id, handle, url FROM channels WHERE id IN ({placeholders})",
                channel_ids,
            )
        )
    out: dict[int, str] = {}
    for r in rows:
        if r["handle"]:
            out[int(r["id"])] = str(r["handle"])
        elif r["url"]:
            tail = str(r["url"]).rstrip("/").rsplit("/", 1)[-1]
            out[int(r["id"])] = tail
    return out


def _verdict(score: float) -> str:
    if score >= 0.85:
        return "almost certainly AI-generated"
    if score >= 0.6:
        return "likely AI-generated"
    if score >= 0.4:
        return "ambiguous — close to the decision boundary"
    if score >= 0.15:
        return "likely human-made"
    return "almost certainly human-made"


def _summarize_neighbors(prediction: "Prediction") -> tuple[str, str]:
    """Return (neighbor_summary_sentence, agreement_status).

    agreement_status is one of: "agree", "disagree", "mixed", "n/a".
    """
    neighbors = prediction.top_neighbors
    if not neighbors:
        return ("No neighbors were retrieved.", "n/a")

    n_off = sum(1 for n in neighbors if n.label == 1)
    n_ctrl = len(neighbors) - n_off
    counts = Counter((n.channel_id, n.label) for n in neighbors)
    handles = _channel_handles([cid for cid, _ in counts])

    # The most-represented (channel, label) groups, with handle suffix.
    grouped: list[str] = []
    for (cid, label), c in counts.most_common():
        tag = "offender" if label == 1 else "control"
        handle = handles.get(cid)
        suffix = f" {handle}" if handle else ""
        grouped.append(f"{c}× ch{cid} {tag}{suffix}")
    breakdown = "; ".join(grouped)

    dists = [n.distance for n in neighbors]
    dist_range = f"{min(dists):.2f}–{max(dists):.2f}"

    sentence = (
        f"Top {len(neighbors)} nearest neighbors: {breakdown}. "
        f"Cosine distances range {dist_range}."
    )

    score = prediction.score
    majority_offender = n_off > n_ctrl
    if n_off == n_ctrl:
        agreement = "mixed"
    elif (majority_offender and score >= 0.5) or (not majority_offender and score < 0.5):
        agreement = "agree"
    else:
        agreement = "disagree"
    return sentence, agreement


def _summarize_modalities(prediction: "Prediction") -> str:
    if not prediction.modality_attribution:
        return ""
    sorted_mods = sorted(
        prediction.modality_attribution.items(), key=lambda kv: -kv[1]
    )
    top_name, top_share = sorted_mods[0]
    second_name, second_share = sorted_mods[1] if len(sorted_mods) > 1 else (None, 0.0)
    parts = [f"{top_name} carries {top_share:.0%} of the linear attribution"]
    if second_name and second_share >= 0.20:
        parts.append(f"{second_name} another {second_share:.0%}")
    return ", ".join(parts) + "."


def _summarize_top_features(prediction: "Prediction", limit: int = 3) -> str:
    feats = prediction.top_features[:limit]
    if not feats:
        return ""
    pieces = []
    for name, contrib in feats:
        direction = "→AI" if contrib > 0 else "→clean"
        pieces.append(f"{name} ({contrib:+.2f} {direction})")
    return "Strongest individual features: " + "; ".join(pieces) + "."


def _temporal_prior_note(prediction: "Prediction") -> str:
    if prediction.prior_cap is None:
        return ""
    year = (prediction.upload_date or "????")[:4]
    return (
        f"Temporal prior fired: upload year {year} predates AI video, so the "
        f"raw model score {prediction.raw_score:.2f} was capped at "
        f"{prediction.prior_cap:.2f}."
    )


def _disagreement_note(prediction: "Prediction", agreement: str) -> str:
    if agreement == "disagree":
        return (
            "Caveat: the model's verdict and the labels of the nearest "
            "training neighbors point in opposite directions. This is the "
            "OOD warning sign — treat the score with extra skepticism."
        )
    if agreement == "mixed":
        return "Neighbor labels are split — this clip lives near the boundary."
    return ""


def explain(prediction: "Prediction") -> str:
    """Generate a deterministic 2-4 sentence rationale for ``prediction``."""
    parts: list[str] = []
    parts.append(
        f"P(AI) = {prediction.score:.2f} — {_verdict(prediction.score)}."
    )
    nbr_sentence, agreement = _summarize_neighbors(prediction)
    parts.append(nbr_sentence)

    mod = _summarize_modalities(prediction)
    if mod:
        parts.append(mod)

    feats = _summarize_top_features(prediction)
    if feats:
        parts.append(feats)

    prior = _temporal_prior_note(prediction)
    if prior:
        parts.append(prior)

    disagree = _disagreement_note(prediction, agreement)
    if disagree:
        parts.append(disagree)

    return " ".join(parts)
