# Medea

AI-generated YouTube video detector. Multi-modal embedding + vector search + classifier head.

Named after Medea of Greek mythology, who killed the bronze automaton Talos.

## Setup

```bash
# 1. Install uv if not already installed
pip install uv

# 2. Sync deps (first time downloads ~7GB; mostly torch + bundled CUDA libs)
uv sync

# 3. Verify GPU is visible to torch
uv run medea info
```

## Usage

Curate seed channels in `data/seeds/offenders.txt` and `data/seeds/controls.txt` (one URL per line; see `data/seeds/README.md`).

Ingest 30-second middle clips from each seed channel:

```bash
uv run python scripts/ingest.py --seeds data/seeds/offenders.txt --label 1 --per-channel 10
uv run python scripts/ingest.py --seeds data/seeds/controls.txt  --label 0 --per-channel 10
```

Output: clips at `data/raw/<id>.mp4`, metadata in `data/medea.db`. Re-runs are idempotent — already-ingested videos are skipped.

## Status

This project is being built milestone-by-milestone. See [PROGRESS.md](PROGRESS.md) for the current state and [PLAN.md](PLAN.md) for the full architecture and remaining milestones.

## Resume in a new session

```
cd C:\Users\mathi\work\medea
claude "Continue Medea from the next unfinished milestone in PROGRESS.md."
```
