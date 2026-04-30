"""Predict whether a YouTube video is AI-generated.

Usage:
    python scripts/predict.py <youtube_url_or_id>
    python scripts/predict.py <url> --model logreg
    python scripts/predict.py <url> --k 10 --json

Downloads a 30s middle clip, runs all four feature encoders, fuses the
vector exactly as in training, and reports:
  * P(AI) score
  * top-k similar known videos from Chroma (with their channel + label)
  * per-modality attribution share (from the LogReg surrogate)
  * top contributing features

Designed for one-shot manual checks; the API server in ``medea.api.server``
runs the same code path with a persistent Predictor.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from medea.model.infer import Predictor

app = typer.Typer(add_completion=False, help="Predict AI-likelihood for a YouTube URL.")
console = Console()


def _score_bar(score: float, width: int = 30) -> str:
    filled = int(round(score * width))
    bar = "█" * filled + "·" * (width - filled)
    color = "red" if score >= 0.7 else ("yellow" if score >= 0.4 else "green")
    return f"[{color}]{bar}[/{color}] {score:.3f}"


def _print_result(pred, model: str) -> None:
    console.rule(f"[bold]{pred.video_id}[/bold]")
    if pred.title:
        console.print(f"[bold]title:[/bold] {pred.title}")
    console.print(f"[bold]model:[/bold] {model}")
    console.print(f"[bold]P(AI):[/bold] {_score_bar(pred.score)}")

    nbr_table = Table(title="Top-k known neighbors", show_lines=False)
    nbr_table.add_column("video_id")
    nbr_table.add_column("label", justify="center")
    nbr_table.add_column("channel", justify="right")
    nbr_table.add_column("distance", justify="right")
    for n in pred.top_neighbors:
        lbl_str = "[red]offender[/red]" if n.label == 1 else "[green]control[/green]"
        nbr_table.add_row(n.video_id, lbl_str, str(n.channel_id), f"{n.distance:.4f}")
    console.print(nbr_table)

    mod_table = Table(title="Modality attribution (LogReg surrogate)")
    mod_table.add_column("modality")
    mod_table.add_column("share", justify="right")
    for m, share in sorted(pred.modality_attribution.items(), key=lambda kv: -kv[1]):
        mod_table.add_row(m, f"{share:.1%}")
    console.print(mod_table)

    feat_table = Table(title="Top contributing features (signed)")
    feat_table.add_column("feature")
    feat_table.add_column("contribution", justify="right")
    for name, contrib in pred.top_features:
        color = "red" if contrib > 0 else "green"
        feat_table.add_row(name, f"[{color}]{contrib:+.3f}[/{color}]")
    console.print(feat_table)


@app.command()
def main(
    url: str = typer.Argument(..., help="YouTube URL or 11-char video id."),
    model: str = typer.Option("mlp", help="mlp | logreg"),
    k: int = typer.Option(5, help="Number of neighbors to return."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of tables."),
    log_level: str = typer.Option("WARNING"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=False, show_path=False)],
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("loading models + encoders", total=None)
        predictor = Predictor(model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("downloading + scoring", total=None)
        pred = predictor.predict_url(url, k=k)

    if as_json:
        payload = {
            "video_id": pred.video_id,
            "title": pred.title,
            "score": pred.score,
            "model": pred.model,
            "top_neighbors": [asdict(n) for n in pred.top_neighbors],
            "modality_attribution": pred.modality_attribution,
            "top_features": pred.top_features,
        }
        console.print_json(data=payload)
        return

    _print_result(pred, model=model)


if __name__ == "__main__":
    app()
