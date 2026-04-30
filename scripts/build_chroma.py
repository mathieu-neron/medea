"""Populate the Chroma collection from the fused feature matrix.

By default this resets the collection and re-inserts everything — Chroma
upserts are cheap, but resetting guarantees we don't keep stale rows for
videos that have been removed from the dataset.

Usage:
    python scripts/build_chroma.py            # reset + populate
    python scripts/build_chroma.py --no-reset # upsert in place
"""

from __future__ import annotations

import logging
import sys

import typer
from rich.console import Console
from rich.logging import RichHandler

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from medea.config import ensure_dirs
from medea.features.pipeline import build_dataset
from medea.storage.vector_db import (
    embedding_slice,
    get_collection,
    reset_collection,
    upsert_videos,
)

app = typer.Typer(add_completion=False, help="Populate Chroma from combined features.")
console = Console()


@app.command()
def main(
    reset: bool = typer.Option(True, help="Drop the existing collection before insert."),
    log_level: str = typer.Option("INFO"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=False, show_path=False)],
    )
    ensure_dirs()

    ds = build_dataset()
    embeddings = embedding_slice(ds.X)
    console.print(
        f"[bold]chroma[/bold]: {len(ds.video_ids)} videos, "
        f"embedding dim={embeddings.shape[1]} (fused {ds.X.shape[1]} - "
        f"{ds.block_sizes['scalars']} scalars)"
    )

    collection = reset_collection() if reset else get_collection()
    upsert_videos(
        collection,
        video_ids=ds.video_ids,
        embeddings=embeddings,
        labels=ds.labels,
        channel_ids=ds.channel_ids,
    )
    console.print(f"  [green]upserted[/green] {collection.count()} rows")


if __name__ == "__main__":
    app()
