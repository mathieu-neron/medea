# Medea — Progress

One line per milestone. Update when finishing a milestone.

- M1 — Bootstrap: ✅ done (2026-04-28) — project layout, deps installed via `uv sync` (507 pkgs), GPU verified
- M2 — Ingest: ✅ done (2026-04-28) — 94 clips (44 offender / 50 control) at 30s middle slices, ≤720p, in `data/raw/`. SQLite at `data/medea.db` (channels + videos)
- M3 — Visual + audio features: ⏳ pending
- M4 — Text + metadata features: ⏳ pending
- M5 — Vector DB + UMAP: ⏳ pending
- M6 — LogReg baseline: ⏳ pending
- M7 — MLP classifier: ⏳ pending
- M8 — Predict CLI + FastAPI: ⏳ pending
- M9 — Error analysis: ⏳ pending
- Stretch — RAG explanation: ⏳ pending

## Next action when resuming

Start M3 — Visual + audio features. See "M3" in PLAN.md. Build it together — discuss design choices, write code, test on a small batch — don't pre-write later milestones.

The 94 clips in `data/raw/` are the input. Output: per-modality feature vectors cached to parquet (visual via open_clip ViT-B/32, audio via faster-whisper transcript + wav2vec2 anti-spoof).

## Notes

- Plan file: [PLAN.md](PLAN.md) (also at `~/.claude/plans/i-have-an-idea-twinkly-thunder.md`).
- Project rules: no git/commits per user preference for local practice projects.
- Hardware: RTX 3080 10GB, Python 3.12.10, Windows 11.
- ffmpeg is bundled via `imageio-ffmpeg` and copied as `data/.bin/ffmpeg(.exe)` so yt-dlp finds it; it also gets prepended to `PATH` at runtime.
