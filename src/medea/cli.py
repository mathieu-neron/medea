"""Top-level `medea` CLI. Mostly a convenience wrapper over the scripts/ commands."""

from __future__ import annotations

import typer

from medea.config import ensure_dirs

app = typer.Typer(add_completion=False, help="Medea — AI YouTube video detector.")


@app.command()
def init() -> None:
    """Create the data directory tree."""
    ensure_dirs()
    typer.echo("Directories ready.")


@app.command()
def info() -> None:
    """Print environment info."""
    import torch

    typer.echo(f"torch: {torch.__version__}")
    typer.echo(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    app()
