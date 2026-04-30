# Medea

A multi-modal classifier for **detecting AI-generated YouTube videos**. URL in → P(AI) score out, plus the labeled training-set neighbors that drove the decision and an optional natural-language rationale via a local LLM.

> Named after Medea of Greek mythology, who killed the bronze automaton Talos. Fitting for a tool that hunts automated slop.

This is a learning project — small training set (~10 channels, ~94 clips), single dev machine, no cloud — but a complete end-to-end pipeline: ingest, four-modality feature extraction, vector DB, classifier head, FastAPI server, and retrieval-grounded LLM explanation.

---

## What it does

Given a YouTube URL, Medea:

1. Downloads a **30-second middle clip** of the video (yt-dlp).
2. Extracts four feature modalities and fuses them into a single 1298-dim vector:
   - **Visual** — 8 evenly-spaced frames → CLIP ViT-B/32 → mean-pooled, L2-normed (512-dim)
   - **Audio** — faster-whisper transcript + wav2vec2 anti-spoofing AI-voice probability
   - **Text** — sentence-transformers MiniLM-L6 embeddings of the transcript and `title + description` (2 × 384-dim)
   - **Metadata** — 18 handcrafted scalars: title shape (caps ratio, clickbait regex, emoji count), description URL/hashtag counts, view count, channel cadence, transcript-presence flag
3. **Retrieves** the top-k nearest neighbors from a Chroma vector DB of labeled training clips.
4. **Scores** the fused vector with a small PyTorch MLP (1298 → 128 → 64 → 1, channel-grouped CV F1 = 0.90).
5. **Caps** the score with a temporal prior on upload year (pre-2017 uploads can't be confidently labeled AI-generated since the technology didn't exist).
6. **Explains** the decision via either a deterministic rule-based rationale or — if Ollama is running — a local LLM grounded in the retrieved neighbors (full RAG, no API keys).

```
yt-dlp ─▶ 30s clip ─┬─▶ frame sampler ─▶ CLIP ─────────────┐
                    ├─▶ ffmpeg audio  ─▶ Whisper ──────────┤
                    ├─▶ ffmpeg audio  ─▶ wav2vec2 anti-spoof┤
                    ├─▶ MiniLM (transcript + title+desc) ──┤─▶ fused ─┬─▶ Chroma kNN
                    └─▶ yt-dlp meta + handcrafted scalars ─┘          ├─▶ MLP head ─▶ P(AI)
                                                                       └─▶ Ollama RAG rationale
```

## Performance

Channel-grouped `LeaveOneGroupOut` CV (10 folds, one held-out channel per fold) over the ~94-clip training set:

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| kNN (k=5, cosine-weighted) | 0.81 | 0.86 | 0.71 | 0.78 | — |
| LogisticRegression baseline | 0.81 | 0.78 | 0.82 | 0.80 | 0.93 |
| **MLP head (deployed)** | **0.90** | **0.87** | **0.93** | **0.90** | **0.96** |

Channel-grouped splits are load-bearing here — UMAP shows the 10 channels as 10 distinct clusters, so a video-level CV would mostly memorize channel identity rather than "AI-ness." See `notebooks/01_explore_embeddings.ipynb`.

---

## Requirements

| | tested with | notes |
|---|---|---|
| OS | Windows 11 | should work on Linux/macOS; `data/.bin/ffmpeg` is OS-aware |
| Python | 3.12 | <3.13 (transformers/torch wheels) |
| GPU | RTX 3080 10GB | optional but recommended; CPU-only works for inference, training is bearable |
| ffmpeg | bundled via `imageio-ffmpeg` | auto-copied to `data/.bin/`; no system install needed |
| package manager | [uv](https://docs.astral.sh/uv/) | the only required tool besides Python itself |
| **optional** | [Ollama](https://ollama.com/download) | enables the local-LLM RAG rationale path |

VRAM use during inference: CLIP (~1GB) + Whisper-small (~1.5GB) + wav2vec2 (~0.5GB) + MiniLM (~0.1GB) + MLP (negligible) ≈ 3GB total. The optional 7B Ollama model adds ~5GB; 10GB GPUs fit the whole stack comfortably.

---

## Install

```bash
# 1. Install uv (one-time)
pip install uv

# 2. Clone, then sync deps (first time downloads ~7GB; mostly torch + CUDA libs)
cd <path-to-medea-clone>
uv sync

# 3. Verify GPU is visible to torch
uv run medea info
```

The first model run will pull ~1GB of model weights from Hugging Face (CLIP, Whisper-small, wav2vec2, MiniLM) into `~/.cache/huggingface`.

### Optional: enable LLM rationale

```bash
# 1. Install Ollama from https://ollama.com/download (one-click installer; daemon
#    starts automatically on Windows).
# 2. Pull the default model (~4.7GB):
ollama pull qwen2.5:7b-instruct-q4_K_M

# Override default via env var if desired:
#   $env:MEDEA_OLLAMA_MODEL = 'llama3.2:3b'   # smaller, weaker
```

Without Ollama, the `--explain` flag falls back to a deterministic rule-based rationale tagged `[rule-based; ollama not running]`.

---

## Reproduction (full training pipeline)

End-to-end from a fresh clone, ~30 min on the reference machine. All steps are idempotent — re-runs skip work that's already done.

### 1. Curate seed channels

Edit `data/seeds/offenders.txt` (label = 1, AI-generated channels) and `data/seeds/controls.txt` (label = 0, human-made channels matched to the offenders' topical domains). 5–10 of each is enough to start. See [`data/seeds/README.md`](data/seeds/README.md) for curation guidance.

### 2. Ingest 30-second clips

```bash
uv run python scripts/ingest.py --seeds data/seeds/offenders.txt --label 1 --per-channel 10
uv run python scripts/ingest.py --seeds data/seeds/controls.txt  --label 0 --per-channel 10
```

Output: clips at `data/raw/<video_id>.mp4`, metadata in `data/medea.db` (SQLite). Channel-level dedup: each video gets ingested at most once. Per-clip disk: ~5MB at ≤720p; total ~500MB for 100 clips.

### 3. Extract features → fuse → vector DB

```bash
uv run python scripts/extract.py     # all four modalities + fused combined.parquet
uv run python scripts/build_chroma.py # populate the Chroma collection
```

Modalities run sequentially (visual freed before audio loads, etc.) to fit in 10GB VRAM. Outputs:
- `data/features/{visual,audio,text,metadata}.parquet` — per-modality
- `data/features/combined.parquet` — fused 1298-dim vectors
- `data/chroma/` — persistent Chroma collection of the 1280-dim embedding-only slice

### 4. Train classifiers

```bash
uv run python scripts/train.py       # LogReg baseline (~5 sec; mlflow-logged)
uv run python scripts/train_mlp.py   # MLP head (~2 min on GPU; mlflow-logged)
```

Both use `LeaveOneGroupOut(channel_id)` for honest channel-level cross-validation. Final models are refit on all training samples and saved to `data/models/{logreg.joblib,mlp.pt}`. mlflow runs live under `data/mlruns/`.

### 5. (Optional) Inspect separability

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_embeddings.ipynb
```

Renders `data/features/umap.png` with the 2D UMAP projection colored by label and channel. Plan calls this the "non-negotiable gating check" — if the classes don't visibly cluster, no classifier will save you.

---

## Usage

### CLI prediction

```bash
# Single URL
uv run python scripts/predict.py "https://www.youtube.com/watch?v=07AgCgpzvoM" --k 5

# With LLM rationale (requires Ollama; falls back to rules if unavailable)
uv run python scripts/predict.py "07AgCgpzvoM" --explain --k 5

# JSON output (suitable for piping)
uv run python scripts/predict.py "07AgCgpzvoM" --json --k 5

# Cross-check via the linear baseline
uv run python scripts/predict.py "07AgCgpzvoM" --model logreg --k 5
```

The bare 11-character video id, full `youtube.com/watch?v=...` URL, `youtu.be/...` short URL, and `youtube.com/shorts/...` URL are all accepted.

Output (table mode):
- **P(AI) bar** with band-coded color (green / yellow / red)
- **Top-k nearest neighbors** with their channel, label, and cosine distance
- **Modality attribution** (LogReg surrogate) — which feature blocks drove the decision
- **Top contributing features** (signed) — individual feature contributions, `→AI` / `→clean`
- **Temporal prior** notice when the upload year cap fired
- **Rationale** (when `--explain` is set)

### FastAPI server

```bash
uv run uvicorn medea.api.server:app --host 127.0.0.1 --port 8000
```

`POST /predict`:

```json
{
  "url": "https://www.youtube.com/watch?v=07AgCgpzvoM",
  "k": 5,
  "model": "mlp",
  "explain": true
}
```

`GET /health` — daemon liveness + which models are loaded.

The MLP predictor is warmed up in the FastAPI lifespan handler so the first request doesn't pay the ~60 second cold-start.

### Error analysis

```bash
uv run python scripts/error_analysis.py
```

Re-runs channel-grouped CV with both classifiers, caches OOF predictions to `data/features/oof_predictions.parquet`, prints per-channel breakdown and false-positive / false-negative tables with titles, and a deep-dive on the persistent ch61 control failure (the canonical "silent wilderness builder" case where the audio anti-spoof model misfires). Pass `--refresh` to force a full re-train.

---

## Project layout

```
medea/
├── PLAN.md                            # full architecture + milestone plan
├── PROGRESS.md                        # one-line per milestone with results
├── data/
│   ├── seeds/{offenders,controls}.txt # curate these
│   ├── raw/                           # 30s mp4 clips (gitignored)
│   ├── features/                      # parquet caches + UMAP png (gitignored)
│   ├── medea.db                       # SQLite: channels + videos (gitignored)
│   ├── chroma/                        # Chroma persistent dir (gitignored)
│   ├── models/                        # trained logreg.joblib + mlp.pt (gitignored)
│   ├── mlruns/                        # mlflow tracking dir (gitignored)
│   └── .bin/                          # bundled ffmpeg.exe (gitignored)
├── notebooks/
│   └── 01_explore_embeddings.ipynb    # UMAP separability check
├── src/medea/
│   ├── ingest/youtube.py              # yt-dlp wrapper, channel→clip download
│   ├── features/
│   │   ├── visual.py                  # CLIP frame-level mean-pool
│   │   ├── audio.py                   # ffmpeg + faster-whisper + wav2vec2
│   │   ├── text.py                    # MiniLM sentence-transformers
│   │   ├── metadata.py                # handcrafted scalars
│   │   ├── pipeline.py                # fuse → 1298-dim vector
│   │   └── store.py                   # idempotent parquet upsert
│   ├── storage/
│   │   ├── db.py                      # SQLite schema + upserts
│   │   └── vector_db.py               # Chroma wrapper + cosine kNN
│   ├── model/
│   │   ├── baseline.py                # StandardScaler → LogReg pipeline
│   │   ├── mlp.py                     # PyTorch MLP head
│   │   ├── infer.py                   # Predictor: URL → score + neighbors
│   │   └── explain.py                 # rationale (LLM-RAG + rules fallback)
│   ├── eval/metrics.py                # binary metrics + confusion fmt
│   ├── api/server.py                  # FastAPI: POST /predict, GET /health
│   └── config.py                      # paths, dirs, to_repo_relative helper
└── scripts/
    ├── ingest.py                      # CLI: ingest from seed lists
    ├── extract.py                     # CLI: features + fusion
    ├── build_chroma.py                # CLI: populate vector DB
    ├── train.py                       # CLI: LogReg baseline
    ├── train_mlp.py                   # CLI: MLP head
    ├── predict.py                     # CLI: URL → P(AI) + rationale
    └── error_analysis.py              # CLI: OOF FP/FN report
```

---

## Limitations & known caveats

- **Small training set.** ~94 clips across 10 channels. The MLP wins in-distribution but extrapolates aggressively on out-of-distribution content (e.g. a 2014 Tom Scott music video produced raw MLP=0.99 before the temporal prior kicked in). Widening the seed list is the highest-leverage next step.
- **Channel-scalar inference shim.** Three of the 18 scalars (`channel_video_count_observed`, `channel_age_days`, `channel_mean_iud_days`) can't be computed from a single fresh URL the way they were at training time, so `medea.model.infer._NEUTRALIZED_CHANNEL_COLS` substitutes the StandardScaler training mean for those columns at predict time. The right fix is to fetch the channel's recent uploads via yt-dlp at inference; deferred.
- **Temporal prior is upload-date-based.** yt-dlp returns the YouTube upload date, not the original recording date, so a 2024 re-upload of pre-2022 footage gets no cap. Correct behavior in expectation, but not always what you'd want.
- **YouTube ToS.** yt-dlp downloads sit in a gray area; this is for local research / learning use, not redistribution.

---

## Pointers

- [`PLAN.md`](PLAN.md) — full architecture, design choices, milestone plan.
- [`PROGRESS.md`](PROGRESS.md) — one line per milestone with the actual measured results.
- [`data/seeds/README.md`](data/seeds/README.md) — seed-list curation guidance.

The project was built milestone-by-milestone (M1 bootstrap → M9 error analysis + RAG stretch) — `PROGRESS.md` is the most reliable source of truth for what each piece actually achieved.
