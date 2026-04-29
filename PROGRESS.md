# Medea — Progress

One line per milestone. Update when finishing a milestone.

- M1 — Bootstrap: ✅ done (2026-04-28) — project layout, deps installed via `uv sync` (507 pkgs), GPU verified
- M2 — Ingest: ✅ done (2026-04-28) — 94 clips (44 offender / 50 control) at 30s middle slices, ≤720p, in `data/raw/`. SQLite at `data/medea.db` (channels + videos)
- M3 — Visual + audio features: ✅ done (2026-04-29) — `data/features/visual.parquet` (94×512 CLIP ViT-B/32 mean-pooled, L2-normed) and `data/features/audio.parquet` (faster-whisper transcript + lang + wav2vec2 ai_voice_prob). Strong class signal already: offender ai_voice median 0.86 vs control 0.37; offender mean transcript 123 chars vs control 303.
- M4 — Text + metadata features: ⏳ pending
- M5 — Vector DB + UMAP: ⏳ pending
- M6 — LogReg baseline: ⏳ pending
- M7 — MLP classifier: ⏳ pending
- M8 — Predict CLI + FastAPI: ⏳ pending
- M9 — Error analysis: ⏳ pending
- Stretch — RAG explanation: ⏳ pending

## Next action when resuming

Start M4 — Text + metadata features. See "M4" in PLAN.md. Build it together — discuss design choices, write code, test on a small batch — don't pre-write later milestones.

Inputs: `data/features/audio.parquet` (transcripts) + the videos table in `data/medea.db` (titles, descriptions, upload dates, view counts). Output: `data/features/text.parquet` (sentence-transformer embeddings of transcript and title+desc), `data/features/metadata.parquet` (channel age, mean inter-upload days, title length, %uppercase, clickbait regex hits, etc.), and a `features/pipeline.py` that concatenates + L2-normalizes per-modality into one feature vector per video.

## Notes

- Plan file: [PLAN.md](PLAN.md) (also at `~/.claude/plans/i-have-an-idea-twinkly-thunder.md`).
- Project rules: no git/commits per user preference for local practice projects.
- Hardware: RTX 3080 10GB, Python 3.12.10, Windows 11.
- ffmpeg is bundled via `imageio-ffmpeg` and copied as `data/.bin/ffmpeg(.exe)` so yt-dlp finds it; it also gets prepended to `PATH` at runtime.
