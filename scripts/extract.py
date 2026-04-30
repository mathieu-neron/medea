"""Feature-extraction CLI: run all four modalities + fusion over ingested clips.

Usage:
    python scripts/extract.py                     # all modalities + fuse
    python scripts/extract.py --modality visual   # just CLIP frames
    python scripts/extract.py --modality audio    # transcript + AI-voice
    python scripts/extract.py --modality text     # transcript/title sentence-emb
    python scripts/extract.py --modality metadata # handcrafted scalars
    python scripts/extract.py --modality combined # fuse existing parquets
    python scripts/extract.py --limit 2           # smoke-test on 2 clips

Modalities run sequentially: the visual encoder is freed before the audio
models load, to fit comfortably in 10GB VRAM. Text + metadata are cheap
and run after audio.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

# Force UTF-8 on Windows consoles (cp1252 default chokes on Rich box-drawing).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import torch

from medea.config import ROOT, ensure_dirs
from medea.features.store import (
    AUDIO_PARQUET,
    METADATA_PARQUET,
    TEXT_PARQUET,
    VISUAL_PARQUET,
    append_rows,
    existing_ids,
    load_table,
)
from medea.storage.db import connect

app = typer.Typer(add_completion=False, help="Extract per-clip features into parquet.")
console = Console()

VISUAL_FLUSH_EVERY = 16
AUDIO_FLUSH_EVERY = 8
TEXT_FLUSH_EVERY = 32

MODALITIES = ("visual", "audio", "text", "metadata", "combined", "all")


def _all_clips() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    with connect() as conn:
        for r in conn.execute(
            "SELECT id, clip_path FROM videos WHERE clip_path IS NOT NULL"
        ):
            p = Path(r["clip_path"])
            if not p.is_absolute():
                p = (ROOT / p).resolve()
            if p.exists():
                rows.append((r["id"], p))
    return rows


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_visual(clips: list[tuple[str, Path]]) -> None:
    from medea.features.visual import VisualEncoder

    have = existing_ids(VISUAL_PARQUET)
    todo = [(vid, p) for vid, p in clips if vid not in have]
    console.print(
        f"[bold]visual[/bold]: {len(todo)} to embed, {len(clips) - len(todo)} cached"
    )
    if not todo:
        return

    encoder = VisualEncoder()
    pending: list[dict] = []
    ok = fail = 0
    for vid, path in todo:
        try:
            vec = encoder.embed_clip(path)
        except Exception as e:  # noqa: BLE001 — log and continue
            console.print(f"  [red]visual fail {vid}: {e}[/red]")
            fail += 1
            continue
        pending.append({"video_id": vid, "embedding": vec.tolist()})
        ok += 1
        console.print(f"  [dim]+visual[/dim] {vid}  shape={vec.shape}")
        if len(pending) >= VISUAL_FLUSH_EVERY:
            append_rows(VISUAL_PARQUET, pending)
            pending.clear()
    append_rows(VISUAL_PARQUET, pending)
    console.print(f"  [green]visual done:[/green] ok={ok} fail={fail}")

    del encoder
    _free_cuda()


def _run_audio(clips: list[tuple[str, Path]]) -> None:
    from medea.features.audio import AudioEncoder

    have = existing_ids(AUDIO_PARQUET)
    todo = [(vid, p) for vid, p in clips if vid not in have]
    console.print(
        f"[bold]audio[/bold]: {len(todo)} to process, {len(clips) - len(todo)} cached"
    )
    if not todo:
        return

    encoder = AudioEncoder()
    pending: list[dict] = []
    ok = fail = 0
    for vid, path in todo:
        try:
            feats = encoder.features(path)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]audio fail {vid}: {e}[/red]")
            fail += 1
            continue
        pending.append(
            {
                "video_id": vid,
                "transcript": feats.transcript,
                "language": feats.language,
                "ai_voice_prob": feats.ai_voice_prob,
            }
        )
        ok += 1
        console.print(
            f"  [dim]+audio[/dim] {vid}  ai={feats.ai_voice_prob:.2f} lang={feats.language} "
            f"chars={len(feats.transcript)}"
        )
        if len(pending) >= AUDIO_FLUSH_EVERY:
            append_rows(AUDIO_PARQUET, pending)
            pending.clear()
    append_rows(AUDIO_PARQUET, pending)
    console.print(f"  [green]audio done:[/green] ok={ok} fail={fail}")

    del encoder
    _free_cuda()


def _video_metadata_rows(video_ids: list[str]) -> list[dict]:
    """Load (video_id, title, description, view_count, channel_id, upload_date)
    rows from SQLite, restricted to the given video_ids."""
    if not video_ids:
        return []
    placeholders = ",".join("?" for _ in video_ids)
    with connect() as conn:
        cur = conn.execute(
            f"""
            SELECT id AS video_id, channel_id, title, description,
                   upload_date, view_count
            FROM videos
            WHERE id IN ({placeholders})
            """,
            video_ids,
        )
        return [dict(r) for r in cur.fetchall()]


def _run_text(clips: list[tuple[str, Path]]) -> None:
    from medea.features.text import TextEncoder

    have = existing_ids(TEXT_PARQUET)
    todo_ids = [vid for vid, _ in clips if vid not in have]
    console.print(
        f"[bold]text[/bold]: {len(todo_ids)} to embed, {len(clips) - len(todo_ids)} cached"
    )
    if not todo_ids:
        return

    audio_df = load_table(AUDIO_PARQUET)
    transcripts: dict[str, str] = (
        dict(zip(audio_df["video_id"].astype(str), audio_df["transcript"].fillna("")))
        if not audio_df.empty
        else {}
    )
    sql_rows = {r["video_id"]: r for r in _video_metadata_rows(todo_ids)}

    encoder = TextEncoder()
    pending: list[dict] = []
    ok = fail = 0
    for vid in todo_ids:
        meta = sql_rows.get(vid)
        if meta is None:
            console.print(f"  [red]text fail {vid}: no SQLite row[/red]")
            fail += 1
            continue
        try:
            feats = encoder.features(
                transcript=transcripts.get(vid, ""),
                title=meta["title"] or "",
                description=meta["description"] or "",
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]text fail {vid}: {e}[/red]")
            fail += 1
            continue
        pending.append(
            {
                "video_id": vid,
                "transcript_emb": feats.transcript_emb.tolist(),
                "title_emb": feats.title_emb.tolist(),
            }
        )
        ok += 1
        if len(pending) >= TEXT_FLUSH_EVERY:
            append_rows(TEXT_PARQUET, pending)
            pending.clear()
    append_rows(TEXT_PARQUET, pending)
    console.print(f"  [green]text done:[/green] ok={ok} fail={fail}")

    del encoder
    _free_cuda()


def _run_metadata(clips: list[tuple[str, Path]]) -> None:
    """Recompute metadata for every clip — channel-level features depend on
    the full set of observed uploads, so a partial recompute would be wrong.
    Cheap (regex + small math) so always do all clips."""
    from medea.features.metadata import channel_features, video_features

    target_ids = {vid for vid, _ in clips}
    with connect() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT id AS video_id, channel_id, title, description,
                       upload_date, view_count
                FROM videos
                """
            )
        ]
    rows = [r for r in rows if r["video_id"] in target_ids]

    channel_uploads: dict[int, list[str]] = {}
    for r in rows:
        channel_uploads.setdefault(r["channel_id"], []).append(r["upload_date"])
    ch_feats = {cid: channel_features(dates) for cid, dates in channel_uploads.items()}

    out: list[dict] = []
    for r in rows:
        feats = video_features(
            title=r["title"] or "",
            description=r["description"] or "",
            view_count=r["view_count"],
        )
        feats.update(ch_feats[r["channel_id"]])
        feats["video_id"] = r["video_id"]
        out.append(feats)

    # Fully replace metadata.parquet for the queried set — simpler than upsert
    # and consistent with the "always recompute" semantics above.
    if out:
        existing = load_table(METADATA_PARQUET)
        if not existing.empty:
            existing = existing[~existing["video_id"].isin([r["video_id"] for r in out])]
            new_df = pd.concat([existing, pd.DataFrame(out)], ignore_index=True)
        else:
            new_df = pd.DataFrame(out)
        METADATA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(METADATA_PARQUET, index=False)
    console.print(f"  [green]metadata done:[/green] rows={len(out)}")


def _run_combined() -> None:
    from medea.features.pipeline import write_combined

    ds = write_combined()
    console.print(
        f"  [green]combined done:[/green] X={ds.X.shape}  blocks={ds.block_sizes}"
    )


@app.command()
def main(
    modality: str = typer.Option("all", help=" | ".join(MODALITIES)),
    limit: int = typer.Option(0, help="If >0, only process the first N clips (smoke test)."),
    log_level: str = typer.Option("INFO", help="DEBUG / INFO / WARNING"),
) -> None:
    if modality not in MODALITIES:
        raise typer.BadParameter(f"modality must be one of: {MODALITIES}")

    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=False, show_path=False)],
    )

    ensure_dirs()
    clips = _all_clips()
    if limit > 0:
        clips = clips[:limit]
    console.rule(f"[bold]{len(clips)} clips queued (modality={modality})")

    if modality in ("visual", "all"):
        _run_visual(clips)
    if modality in ("audio", "all"):
        _run_audio(clips)
    if modality in ("text", "all"):
        _run_text(clips)
    if modality in ("metadata", "all"):
        _run_metadata(clips)
    if modality in ("combined", "all"):
        _run_combined()


if __name__ == "__main__":
    app()
