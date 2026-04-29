# Medea — AI Video Detector MVP Plan

> Named after Medea of Greek mythology, who killed the bronze automaton Talos. Fitting for a tool that hunts automated slop.

## Context

You want to build an MVP of a YouTube AI-slop detector as a Python learning project, complementing your existing crowdsourced [RealTube](https://github.com/mathieu-neron/RealTube). Where RealTube relies on human reports, Medea will *learn* what AI-generated channels look like from a manually curated seed list and predict whether new videos are likely AI-generated.

**Goals (in priority order)**
1. Learn applied ML hands-on: embeddings, vector search, training a classifier head, evaluation.
2. Produce a working end-to-end pipeline: URL in → AI-likelihood score out.
3. Keep it small enough to iterate on a single dev machine (RTX 3080, 10GB VRAM).

**Non-goals**
- No production-grade web app, no auth, no scraping at scale.
- No training huge video models from scratch.
- No git/commits per your stated preference for local practice projects.

## Approach

The "vector DB + RAG" framing refines into **multi-modal embedding + similarity search + a small classifier head** — which is how production detection systems actually work today. RAG (LLM-explains-the-decision) becomes an optional stretch goal at the end.

**Pipeline:**

```
yt-dlp ─▶ video file ─┬─▶ frame sampler ─▶ CLIP embed ──┐
                      ├─▶ ffmpeg audio  ─▶ Whisper transcript ─▶ text embed ─┤
                      ├─▶ ffmpeg audio  ─▶ wav2vec2 anti-spoof ─────────────┤─▶ feature vec ─┬─▶ Chroma (kNN over known offenders)
                      └─▶ yt-dlp meta   ─▶ channel/title features ─────────┘                └─▶ Classifier head (sklearn → torch MLP) ─▶ P(AI)
```

The **vector DB (Chroma)** stores the labeled feature vectors and powers two things:
- kNN lookup ("is this video close to known offenders?") — your retrieval-style detection.
- Training data store for the classifier head.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Already installed; right call for ML. |
| Env mgmt | `uv` (or `venv`) | `uv` is fast and modern; venv is fine if you prefer minimal tools. |
| Ingest | `yt-dlp` (Python lib) + `ffmpeg-python` | De-facto standard. Handles metadata too. |
| Visual embed | `open_clip_torch` (ViT-B/32) | Modern CLIP fork, ~150MB, runs comfortably on 10GB. |
| Speech-to-text | `faster-whisper` (small or medium) | 3-5× faster than openai-whisper on GPU, same accuracy. |
| AI-voice detect | HF `wav2vec2` anti-spoofing model | Pre-trained, zero-train. Pick one from HF leaderboard at start. |
| Text embed | `sentence-transformers` (`all-MiniLM-L6-v2`) | 384-dim, fast, good enough. |
| Vector DB | `chromadb` (local persistent) | Zero-config, swap to Qdrant later if needed. |
| Classifier | `scikit-learn` LogReg → `torch` MLP | LogReg as baseline (you'll see it work first), MLP as the "real ML" milestone. |
| Tracking | `mlflow` (local) | Lets you compare runs without cloud setup. |
| API/CLI | `FastAPI` + `typer` | Clean Python ergonomics. |
| Storage | SQLite for metadata + parquet for feature cache | No DB to run. |

## Project layout

Create at `C:\Users\mathi\work\medea\`:

```
medea/
├── pyproject.toml
├── README.md
├── data/
│   ├── seeds/
│   │   ├── offenders.txt        # one channel URL per line (you curate)
│   │   └── controls.txt         # legitimate channels (you curate)
│   ├── raw/                     # downloaded clips (.gitignore-equivalent: keep out of any sync)
│   ├── features/                # cached parquet feature files
│   └── chroma/                  # Chroma persistent dir
├── src/medea/
│   ├── ingest/youtube.py        # yt-dlp wrapper, channel→videos→clips
│   ├── features/visual.py       # frame sampling + CLIP
│   ├── features/audio.py        # extract audio + faster-whisper + voice detector
│   ├── features/text.py         # transcript + title/desc embed
│   ├── features/metadata.py     # channel age, upload cadence, title heuristics
│   ├── features/pipeline.py     # orchestrates all 4 modalities, writes parquet
│   ├── store/vector_db.py       # Chroma collection wrapper
│   ├── model/baseline.py        # sklearn LogReg
│   ├── model/mlp.py             # torch MLP head
│   ├── model/train.py           # training loop, eval, mlflow logging
│   ├── model/infer.py           # URL → score
│   ├── eval/metrics.py          # precision/recall/F1, confusion matrix, UMAP plot
│   └── api/server.py            # FastAPI: POST /predict {url} → {score, neighbors}
├── scripts/
│   ├── ingest.py                # CLI: ingest from seed lists
│   ├── extract.py               # CLI: run feature pipeline
│   ├── train.py                 # CLI: train classifier
│   └── predict.py               # CLI: predict on a URL
└── notebooks/
    └── 01_explore_embeddings.ipynb   # UMAP, separability check
```

## Build sequence (milestones)

Each milestone is a stopping point where you have something runnable and learn something concrete.

### M1 — Bootstrap (~30 min)
- Create folder, init `uv` project, install deps.
- Write `data/seeds/offenders.txt` and `data/seeds/controls.txt` — start with **5 offender channels + 5 control channels**, you can grow later.
- **You learn:** modern Python project layout.

### M2 — Ingest (~1-2 hr)
- `src/aivd/ingest/youtube.py`: given a channel URL, list its N most recent videos via `yt-dlp`'s Python API, download a **30-second middle clip** of each (saves disk and time vs full videos), capture metadata (channel age, upload dates, title, description, view count) into SQLite.
- CLI: `python scripts/ingest.py --seeds data/seeds/offenders.txt --label 1 --per-channel 10`
- **You learn:** yt-dlp internals, SQLite basics, dataset hygiene.

### M3 — Visual + audio features (~2-3 hr)
- `features/visual.py`: sample 8 evenly-spaced frames per clip, run `open_clip_torch` ViT-B/32, mean-pool → 512-dim vector.
- `features/audio.py`: `ffmpeg` extract 16kHz mono wav, run `faster-whisper` for transcript, run a wav2vec2 anti-spoof model for AI-voice probability.
- Cache everything per video to parquet keyed by video id (idempotent re-runs).
- **You learn:** how pre-trained models actually work, embeddings, GPU memory management.

### M4 — Text + metadata features (~1 hr)
- `features/text.py`: embed transcript (chunked + mean) and title+description with sentence-transformers.
- `features/metadata.py`: channel age in days, mean inter-upload time, title length, % uppercase, presence of clickbait phrases (regex), thumbnail face-count via OpenCV (optional).
- `features/pipeline.py`: concat & L2-normalize per-modality, output single feature vector per video.
- **You learn:** multi-modal feature fusion, why normalization matters.

### M5 — Vector DB + kNN baseline (~1 hr)
- Populate Chroma with all training videos + labels.
- Notebook `01_explore_embeddings.ipynb`: UMAP-project, color by label, **eyeball separability** before training anything. This step is non-negotiable — if classes don't separate visually, no classifier will save you.
- Implement `predict_knn(url, k=5)`: embed → query Chroma → majority vote / score by neighbor labels.
- **You learn:** vector DBs, similarity search, the value of looking at your data.

### M6 — Classifier head — baseline (~1 hr)
- `model/baseline.py`: `sklearn.linear_model.LogisticRegression` on the concatenated feature vectors.
- Channel-level holdout split (NOT video-level — leak risk).
- Compute precision, recall, F1, confusion matrix, log to mlflow.
- **You learn:** why channel-level splits matter, baseline-first ML methodology.

### M7 — MLP head + proper training (~2-3 hr)
- `model/mlp.py`: small PyTorch MLP (e.g. 2 hidden layers, dropout).
- `model/train.py`: training loop, train/val curves, early stopping, mlflow.
- Compare to LogReg baseline. If MLP doesn't beat LogReg meaningfully, your features are the bottleneck — go back to M3/M4.
- **You learn:** training loops, overfitting signals, when more model isn't the answer.

### M8 — End-to-end CLI + FastAPI (~1 hr)
- `scripts/predict.py <youtube_url>` → prints `{score, top_k_neighbors, top_features}`.
- `api/server.py`: `POST /predict` with same behavior.
- **You learn:** packaging an ML model behind an interface.

### M9 — Error analysis & iteration (~ongoing)
- Look at top false positives and false negatives manually.
- Add features that fix the misses; retrain.
- This is where most of the actual learning happens.

### Stretch — RAG explanation
- Use Chroma kNN to retrieve top-3 similar known offenders, feed transcripts + your model's score to an LLM (Claude API), have it write a short "why this looks AI" explanation. *Now* it's actually RAG.

## Critical files & reused utilities

This is a greenfield project — nothing in `C:\Users\mathi\work\` to reuse (sibling folders `CV`, `leetcode-practice`, `TravelSitter` are unrelated). Key external tools:
- `yt-dlp` Python API: `yt_dlp.YoutubeDL({...}).extract_info(url, download=True)`
- `open_clip`: `open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')`
- `faster_whisper.WhisperModel('small', device='cuda', compute_type='float16')`
- `chromadb.PersistentClient(path='data/chroma')`

## Verification

After each milestone:
- **M2 ingest:** open 3 random downloaded clips, confirm content matches expected channel.
- **M3/M4 features:** assert vector shapes; spot-check 2 transcripts.
- **M5 UMAP:** offender / control clusters should be at least partially separated; if not, stop and add features.
- **M6 baseline LogReg:** precision ≥ 0.7 on holdout is the bar to clear before moving on. (Random would be 0.5 with balanced data.)
- **M7 MLP:** must beat LogReg on F1 by a non-trivial margin or features are the issue.
- **M8 end-to-end:** run `predict.py` on 3 held-out URLs (1 known offender, 1 known clean, 1 ambiguous), inspect both score and top-k neighbors.

## Risks & callouts

- **YouTube ToS:** `yt-dlp` is in a gray area. For local learning use, fine; do not redistribute downloads.
- **Class imbalance & evaluation leakage:** split by *channel*, not video. If train and test share a channel, you learn the channel, not "AI-ness."
- **AI-voice detector quality varies:** the HF anti-spoofing models were trained on specific datasets. Spot-check on a few of your clips before relying on them.
- **Disk usage:** 30s clips at 720p ≈ 5MB each. 10 channels × 10 videos = 500MB. Manageable.
- **VRAM budget on 10GB:** running CLIP + Whisper + wav2vec simultaneously will be tight — process modalities sequentially, not in parallel. Pipeline already does this.

## Resumability across sessions

Each milestone produces durable on-disk artifacts so you can pause indefinitely and resume from any milestone in a fresh Claude Code session. To make handoff frictionless:

1. **Copy this plan into the project** as `medea/PLAN.md` once the folder exists, so it's discoverable from inside the project (the `~/.claude/plans/...` filename is auto-generated and not memorable).
2. **Maintain `medea/PROGRESS.md`** — a one-line-per-milestone status file (e.g. `M2 ✅ 2026-04-30 — 100 clips ingested`). Update it as you finish each milestone. Future-you (or future-Claude) reads it first to know where to pick up.
3. **Save a memory entry** so any future Claude session in any working directory knows the project exists, where the plan lives, and how to resume.
4. **All caches are idempotent**: feature extraction skips videos already in parquet; ingestion skips video ids already in SQLite. Re-running scripts is safe and cheap.

**To resume in a new session:**
```
cd C:\Users\mathi\work\medea
claude "Continue Medea from the next unfinished milestone in PROGRESS.md."
```

## Open question to revisit later

Once M5 is done and you've eyeballed the UMAP, decide whether to:
- (a) Keep concatenated multi-modal features (current plan), or
- (b) Train per-modality classifiers and ensemble them.

(b) is more work but gives clearer signal about which modality matters most. Defer the call until you've seen M5's plot.
