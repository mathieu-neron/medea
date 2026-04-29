# Medea

AI-generated YouTube video detector. Multi-modal embedding + vector search + classifier head.

Named after Medea of Greek mythology, who killed the bronze automaton Talos.

## Setup

```bash
# 1. Install uv if not already installed
pip install uv

# 2. Sync deps (first time downloads ~3GB of torch + CUDA wheels)
uv sync

# 3. Verify GPU is visible to torch
uv run medea info
```

## Status

This project is being built milestone-by-milestone. See [PROGRESS.md](PROGRESS.md) for the current state and [PLAN.md](PLAN.md) for the full architecture and remaining milestones.

## Resume in a new session

```
cd C:\Users\mathi\work\medea
claude "Continue Medea from the next unfinished milestone in PROGRESS.md."
```
