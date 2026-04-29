"""Feature-extraction CLI: run visual + audio features over ingested clips.

Usage:
    python scripts/extract.py                     # both modalities, all clips
    python scripts/extract.py --modality visual   # just CLIP frames
    python scripts/extract.py --modality audio    # just transcript + AI-voice
    python scripts/extract.py --limit 2           # smoke-test on 2 clips

Modalities run sequentially: the visual encoder is freed before the audio
models load, to fit comfortably in 10GB VRAM.
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

import torch

from medea.config import ROOT, ensure_dirs
from medea.features.store import (
    AUDIO_PARQUET,
    VISUAL_PARQUET,
    append_rows,
    existing_ids,
)
from medea.storage.db import connect

app = typer.Typer(add_completion=False, help="Extract per-clip features into parquet.")
console = Console()

VISUAL_FLUSH_EVERY = 16
AUDIO_FLUSH_EVERY = 8


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


@app.command()
def main(
    modality: str = typer.Option("both", help="visual | audio | both"),
    limit: int = typer.Option(0, help="If >0, only process the first N clips (smoke test)."),
    log_level: str = typer.Option("INFO", help="DEBUG / INFO / WARNING"),
) -> None:
    if modality not in ("visual", "audio", "both"):
        raise typer.BadParameter("modality must be visual | audio | both")

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

    if modality in ("visual", "both"):
        _run_visual(clips)
    if modality in ("audio", "both"):
        _run_audio(clips)


if __name__ == "__main__":
    app()
