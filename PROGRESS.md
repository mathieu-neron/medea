# Medea — Progress

One line per milestone. Update when finishing a milestone.

- M1 — Bootstrap: ✅ done (2026-04-28) — project layout, deps installed via `uv sync` (507 pkgs), GPU verified
- M2 — Ingest: ✅ done (2026-04-28) — 94 clips (44 offender / 50 control) at 30s middle slices, ≤720p, in `data/raw/`. SQLite at `data/medea.db` (channels + videos)
- M3 — Visual + audio features: ✅ done (2026-04-29) — `data/features/visual.parquet` (94×512 CLIP ViT-B/32 mean-pooled, L2-normed) and `data/features/audio.parquet` (faster-whisper transcript + lang + wav2vec2 ai_voice_prob). Strong class signal already: offender ai_voice median 0.86 vs control 0.37; offender mean transcript 123 chars vs control 303.
- M4 — Text + metadata features: ✅ done (2026-04-30) — `data/features/text.parquet` (94×384 transcript + title+desc MiniLM embeddings, L2-normed, zero vec for empty transcripts) and `data/features/metadata.parquet` (94×15 handcrafted scalars: title shape/caps/clickbait/emoji, desc url+hashtag counts, view_count_log, channel age + mean inter-upload). Fused into `data/features/combined.parquet` shape (94, 1296) = 512 visual + 384 transcript + 384 title_desc + 16 scalars. New class signals: `desc_url_count` median control=3 vs offender=0; `view_count_log` median 5.1 vs 3.9.
- M5 — Vector DB + UMAP: ⏳ pending
- M6 — LogReg baseline: ⏳ pending
- M7 — MLP classifier: ⏳ pending
- M8 — Predict CLI + FastAPI: ⏳ pending
- M9 — Error analysis: ⏳ pending
- Stretch — RAG explanation: ⏳ pending

## Next action when resuming

Start M5 — Vector DB + UMAP. See "M5" in PLAN.md. Populate Chroma at `data/chroma/` with the 94 fused vectors from `data/features/combined.parquet` (1296-dim each, with `video_id` + `channel_id` + `label` metadata). Then create `notebooks/01_explore_embeddings.ipynb` to UMAP-project the combined matrix and color by label — the gating check before any classifier work. Plan calls separability "non-negotiable" here: if offender / control don't visibly separate, stop and revisit features instead of advancing to M6.

Implement `predict_knn(url, k=5)` only after the UMAP looks acceptable: embed → query Chroma → vote / score by neighbor labels.

Useful entry points:
- `medea.features.pipeline.build_dataset()` returns `FusedDataset(video_ids, channel_ids, labels, X, block_sizes)` — already merged with SQLite labels.
- `python scripts/extract.py --modality combined` rewrites `combined.parquet` if upstream parquets change.

## Notes

- Plan file: [PLAN.md](PLAN.md) (also at `~/.claude/plans/i-have-an-idea-twinkly-thunder.md`).
- Project rules: no git/commits per user preference for local practice projects.
- Hardware: RTX 3080 10GB, Python 3.12.10, Windows 11.
- ffmpeg is bundled via `imageio-ffmpeg` and copied as `data/.bin/ffmpeg(.exe)` so yt-dlp finds it; it also gets prepended to `PATH` at runtime.
