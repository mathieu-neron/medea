"""Ingest CLI: download 30s clips for every channel in a seed file.

Usage:
    python scripts/ingest.py --seeds data/seeds/offenders.txt --label 1 --per-channel 10
    python scripts/ingest.py --seeds data/seeds/controls.txt  --label 0 --per-channel 10
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

# Rich emits unicode box-drawing chars; force UTF-8 so this works on Windows
# consoles (whose default cp1252 codepage chokes on them) without env vars.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from medea.config import RAW_DIR, ensure_dirs, to_repo_relative
from medea.ingest.youtube import download_middle_clip, list_channel_videos
from medea.storage.db import (
    connect,
    count_videos,
    init_db,
    insert_video,
    upsert_channel,
    video_exists,
)

app = typer.Typer(add_completion=False, help="Ingest YouTube channels into Medea's local store.")
console = Console()


def _read_seed_file(path: Path) -> list[str]:
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


@app.command()
def main(
    seeds: Path = typer.Option(..., exists=True, readable=True, help="Seed file with one channel URL per line."),
    label: int = typer.Option(..., min=0, max=1, help="1 = offender (AI), 0 = control (human)."),
    per_channel: int = typer.Option(10, min=1, help="How many recent videos per channel."),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG/INFO/WARNING)."),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=False, show_path=False)],
    )

    ensure_dirs()
    init_db()

    channel_urls = _read_seed_file(seeds)
    console.rule(f"[bold]Ingesting {len(channel_urls)} channels (label={label}, per_channel={per_channel})")

    new_clips = 0
    skipped = 0
    failed = 0

    for ch_url in channel_urls:
        console.print(f"[bold cyan]Channel:[/bold cyan] {ch_url}")
        try:
            entries = list_channel_videos(ch_url, per_channel)
        except Exception as e:  # noqa: BLE001 — keep going on listing failures
            console.print(f"  [red]list failed: {e}[/red]")
            failed += 1
            continue

        if not entries:
            console.print("  [yellow]no videos returned[/yellow]")
            continue

        # We don't yet know the canonical handle / channel_id until we download
        # one video. Provisionally insert with the URL as key; backfill on first
        # successful download.
        with connect() as conn:
            channel_pk = upsert_channel(conn, url=ch_url, label=label)

        for entry in entries:
            vid = entry["id"]
            with connect() as conn:
                if video_exists(conn, vid):
                    skipped += 1
                    continue

            console.print(f"  [dim]- {vid}[/dim] {entry.get('title', '')[:80]}")
            meta = download_middle_clip(vid, RAW_DIR)
            if meta is None or meta.clip_path is None:
                failed += 1
                continue

            with connect() as conn:
                # Backfill channel handle/yt id once we have it.
                channel_pk = upsert_channel(
                    conn,
                    url=ch_url,
                    label=label,
                    handle=meta.channel_handle,
                    yt_channel_id=meta.yt_channel_id,
                )
                insert_video(
                    conn,
                    video_id=meta.id,
                    channel_id=channel_pk,
                    label=label,
                    title=meta.title,
                    description=meta.description,
                    upload_date=meta.upload_date,
                    duration=meta.duration,
                    view_count=meta.view_count,
                    clip_path=to_repo_relative(meta.clip_path),
                )
            new_clips += 1

    with connect() as conn:
        total = count_videos(conn)
        offenders = count_videos(conn, label=1)
        controls = count_videos(conn, label=0)

    console.rule("[bold green]Ingest summary")
    console.print(f"new clips: [green]{new_clips}[/green]   skipped (already in db): [yellow]{skipped}[/yellow]   failed: [red]{failed}[/red]")
    console.print(f"db totals: offenders=[green]{offenders}[/green] controls=[green]{controls}[/green] total=[bold]{total}[/bold]")


if __name__ == "__main__":
    app()
