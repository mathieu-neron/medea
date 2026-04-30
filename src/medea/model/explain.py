"""Rationale for a Prediction — full RAG via local LLM, with rule-based fallback.

Two render paths share the same retrieved context (top-k Chroma neighbors
hydrated with titles + transcript heads from SQLite + ``audio.parquet``):

* ``_render_llm`` — calls Ollama on ``localhost:11434`` for an LLM-generated
  summary. Default model: ``qwen2.5:7b-instruct-q4_K_M`` (~4.7GB, fits
  alongside CLIP/Whisper/wav2vec2/MiniLM/MLP on a 10GB GPU with margin).
  Smaller models like ``llama3.2:3b`` work but hallucinate the
  agreement/temporal flags and contradict the verdict. Override via the
  ``MEDEA_OLLAMA_MODEL`` env var.

* ``_render_rules`` — deterministic, neighbor-grounded template; runs from
  the same retrieved context. Used when Ollama isn't reachable.

``explain()`` tries the LLM first and falls back to rules with a small
"(rule-based; ollama not running)" suffix on any connection error. Both
paths read the *same* neighbors and prediction, so the difference is purely
in fluency and phrasing — not in which evidence is grounded against.

Setup for the LLM path:
    1. install Ollama from https://ollama.com/download
    2. ``ollama pull llama3.2:3b``  (or any other small instruct-tuned model)
    3. the daemon starts automatically on Windows; on other OSes ``ollama serve``

No API keys, no external services — this all talks to a local daemon.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ollama

from medea.features.store import AUDIO_PARQUET, load_table
from medea.storage.db import connect

if TYPE_CHECKING:
    from medea.model.infer import Prediction

log = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = os.environ.get(
    "MEDEA_OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"
)
MAX_TRANSCRIPT_CHARS = 240
LLM_NEIGHBORS_K = 4


# ---------------------------------------------------------------------------
# Shared retrieval — used by both render paths so the evidence is identical.
# ---------------------------------------------------------------------------

@dataclass
class NeighborContext:
    video_id: str
    label: str             # "offender" or "control"
    channel_id: int
    handle: str | None
    title: str | None
    transcript_snippet: str
    distance: float


def _channel_handles(channel_ids: list[int]) -> dict[int, str]:
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


def gather_neighbor_context(
    prediction: "Prediction", *, k: int
) -> list[NeighborContext]:
    if not prediction.top_neighbors:
        return []
    nbrs = prediction.top_neighbors[:k]
    nbr_ids = [n.video_id for n in nbrs]

    audio = load_table(AUDIO_PARQUET)
    transcripts: dict[str, str] = (
        dict(zip(audio["video_id"].astype(str), audio["transcript"].fillna("").astype(str)))
        if not audio.empty
        else {}
    )

    placeholders = ",".join("?" for _ in nbr_ids)
    with connect() as conn:
        sql_rows: dict[str, dict] = {
            r["id"]: dict(r)
            for r in conn.execute(
                f"SELECT id, title FROM videos WHERE id IN ({placeholders})", nbr_ids
            )
        }
    handles = _channel_handles([n.channel_id for n in nbrs])

    out: list[NeighborContext] = []
    for n in nbrs:
        sql = sql_rows.get(n.video_id, {})
        ts = transcripts.get(n.video_id, "").strip()
        if ts:
            snippet = ts[:MAX_TRANSCRIPT_CHARS] + ("…" if len(ts) > MAX_TRANSCRIPT_CHARS else "")
        else:
            snippet = "(no detected speech)"
        out.append(
            NeighborContext(
                video_id=n.video_id,
                label="offender" if n.label == 1 else "control",
                channel_id=n.channel_id,
                handle=handles.get(n.channel_id),
                title=sql.get("title"),
                transcript_snippet=snippet,
                distance=n.distance,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rule-based render — deterministic fallback.
# ---------------------------------------------------------------------------

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


def _summarize_neighbors_rules(prediction: "Prediction") -> tuple[str, str]:
    neighbors = prediction.top_neighbors
    if not neighbors:
        return ("No neighbors were retrieved.", "n/a")

    n_off = sum(1 for n in neighbors if n.label == 1)
    n_ctrl = len(neighbors) - n_off
    counts = Counter((n.channel_id, n.label) for n in neighbors)
    handles = _channel_handles([cid for cid, _ in counts])

    grouped: list[str] = []
    for (cid, label), c in counts.most_common():
        tag = "offender" if label == 1 else "control"
        handle = handles.get(cid)
        suffix = f" {handle}" if handle else ""
        grouped.append(f"{c}× ch{cid} {tag}{suffix}")
    breakdown = "; ".join(grouped)

    dists = [n.distance for n in neighbors]
    sentence = (
        f"Top {len(neighbors)} nearest neighbors: {breakdown}. "
        f"Cosine distances range {min(dists):.2f}–{max(dists):.2f}."
    )

    score = prediction.score
    if n_off == n_ctrl:
        agreement = "mixed"
    elif (n_off > n_ctrl and score >= 0.5) or (n_off < n_ctrl and score < 0.5):
        agreement = "agree"
    else:
        agreement = "disagree"
    return sentence, agreement


def _summarize_modalities(prediction: "Prediction") -> str:
    if not prediction.modality_attribution:
        return ""
    sorted_mods = sorted(prediction.modality_attribution.items(), key=lambda kv: -kv[1])
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
    pieces = [
        f"{name} ({contrib:+.2f} {'→AI' if contrib > 0 else '→clean'})"
        for name, contrib in feats
    ]
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


def _disagreement_note(agreement: str) -> str:
    if agreement == "disagree":
        return (
            "Caveat: the model's verdict and the labels of the nearest "
            "training neighbors point in opposite directions — OOD warning."
        )
    if agreement == "mixed":
        return "Neighbor labels are split — this clip lives near the boundary."
    return ""


def _render_rules(prediction: "Prediction") -> str:
    parts: list[str] = [
        f"P(AI) = {prediction.score:.2f} — {_verdict(prediction.score)}."
    ]
    nbr_sentence, agreement = _summarize_neighbors_rules(prediction)
    parts.append(nbr_sentence)
    for fragment in (
        _summarize_modalities(prediction),
        _summarize_top_features(prediction),
        _temporal_prior_note(prediction),
        _disagreement_note(agreement),
    ):
        if fragment:
            parts.append(fragment)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# LLM render — full RAG via local Ollama.
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "You are a media analyst who reviews short video clips and explains "
    "whether each one looks AI-generated. You will be shown the candidate "
    "clip's metadata, a labeled verdict (e.g. 'almost certainly AI-generated' "
    "or 'likely human-made'), and 3–5 nearest neighbors from a labeled "
    "training set tagged offender (AI-generated) or control (human-made).\n\n"
    "Write a concise 2–3 sentence rationale that points to specific evidence "
    "in the input: which neighbor channels match, what the modality "
    "attribution shows about which features drove the verdict, what the "
    "title and transcript heads suggest. Plain text only — no markdown, no "
    "bullet points.\n\n"
    "Hard rules:\n"
    "• Do NOT contradict the verdict or restate the score numerically.\n"
    "• Do NOT speculate about training-time behavior, biases, or what data "
    "the model was trained on. The temporal prior is purely an inference-"
    "time cap and is only relevant when an explicit 'TEMPORAL PRIOR' line "
    "appears in the input — otherwise do not mention it.\n"
    "• Do NOT invent neighbor channels, titles, or transcripts that are not "
    "shown in the input.\n"
    "• If the verdict says 'human-made' do NOT call it AI, and vice versa.\n"
    "• If the input includes a 'DISAGREEMENT' note, foreground it: explain "
    "that the model and neighbors disagree, which is an OOD warning."
)


def _agreement_status(prediction: "Prediction") -> str:
    """agree / disagree / mixed / n/a — same logic as rules path."""
    nbrs = prediction.top_neighbors
    if not nbrs:
        return "n/a"
    n_off = sum(1 for n in nbrs if n.label == 1)
    n_ctrl = len(nbrs) - n_off
    score = prediction.score
    if n_off == n_ctrl:
        return "mixed"
    if (n_off > n_ctrl and score >= 0.5) or (n_off < n_ctrl and score < 0.5):
        return "agree"
    return "disagree"


def _build_llm_user_message(
    prediction: "Prediction", neighbors: list[NeighborContext]
) -> str:
    modalities = ", ".join(
        f"{name}={share:.0%}"
        for name, share in sorted(
            prediction.modality_attribution.items(), key=lambda kv: -kv[1]
        )
    )

    lines: list[str] = [
        "CANDIDATE",
        f"  video_id: {prediction.video_id}",
        f"  title: {prediction.title!r}",
        f"  upload date: {prediction.upload_date or 'unknown'}",
        f"  VERDICT (use this — do not invent your own): "
        f"{_verdict(prediction.score)} (P(AI)={prediction.score:.2f})",
        f"  modality attribution: {modalities}",
    ]

    if prediction.prior_cap is not None:
        year = prediction.upload_date[:4] if prediction.upload_date else "?"
        lines.append(
            f"  TEMPORAL PRIOR fired: upload year {year} predates the AI-video "
            f"era; raw model score {prediction.raw_score:.2f} was capped at "
            f"{prediction.prior_cap:.2f}. Mention this as a reason."
        )

    agreement = _agreement_status(prediction)
    if agreement == "disagree":
        lines.append(
            "  DISAGREEMENT: the model's verdict and the majority of nearest "
            "neighbor labels point in opposite directions. Foreground this as "
            "an OOD warning."
        )
    elif agreement == "mixed":
        lines.append(
            "  MIXED NEIGHBORS: neighbor labels are roughly split — note this "
            "clip lives near the decision boundary."
        )

    lines += [
        "",
        f"NEIGHBORS (top {len(neighbors)} from labeled training set, by feature similarity)",
    ]
    for i, n in enumerate(neighbors, 1):
        title_repr = repr(n.title) if n.title else "(no title)"
        handle = f" {n.handle}" if n.handle else ""
        lines.append(
            f"  {i}. [{n.label}] ch{n.channel_id}{handle} d={n.distance:.2f}  title: {title_repr}"
        )
        lines.append(f"     transcript head: {n.transcript_snippet!r}")
    lines += [
        "",
        "Write the rationale now (2–3 sentences, plain text).",
    ]
    return "\n".join(lines)


def _render_llm(
    prediction: "Prediction",
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    k: int = LLM_NEIGHBORS_K,
) -> str:
    """Call local Ollama for an LLM-generated rationale.

    Raises on any failure — explain() catches and falls back. Failure modes:
    daemon not running (ConnectionError), model not pulled
    (``ollama.ResponseError`` 404), generation error.
    """
    contexts = gather_neighbor_context(prediction, k=k)
    user_msg = _build_llm_user_message(prediction, contexts)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.3, "num_predict": 220, "top_p": 0.9},
    )
    text = response["message"]["content"].strip()
    if not text:
        raise RuntimeError("ollama returned empty content")
    return text


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------

def explain(prediction: "Prediction") -> str:
    """Generate a rationale: LLM via Ollama if reachable, rules-based otherwise.

    The fallback path appends "(rule-based; ollama not running)" so the user
    knows which path produced the text.
    """
    try:
        return _render_llm(prediction)
    except Exception as e:  # noqa: BLE001 — any LLM failure → rules fallback
        log.info("LLM rationale unavailable (%s) — using rule-based fallback", e)
        return _render_rules(prediction) + " [rule-based; ollama not running]"
