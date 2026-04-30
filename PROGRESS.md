# Medea — Progress

One line per milestone. Update when finishing a milestone.

- M1 — Bootstrap: ✅ done (2026-04-28) — project layout, deps installed via `uv sync` (507 pkgs), GPU verified
- M2 — Ingest: ✅ done (2026-04-28) — 94 clips (44 offender / 50 control) at 30s middle slices, ≤720p, in `data/raw/`. SQLite at `data/medea.db` (channels + videos)
- M3 — Visual + audio features: ✅ done (2026-04-29) — `data/features/visual.parquet` (94×512 CLIP ViT-B/32 mean-pooled, L2-normed) and `data/features/audio.parquet` (faster-whisper transcript + lang + wav2vec2 ai_voice_prob). Strong class signal already: offender ai_voice median 0.86 vs control 0.37; offender mean transcript 123 chars vs control 303.
- M4 — Text + metadata features: ✅ done (2026-04-30) — `data/features/text.parquet` (94×384 transcript + title+desc MiniLM embeddings, L2-normed, zero vec for empty transcripts) and `data/features/metadata.parquet` (94×15 handcrafted scalars: title shape/caps/clickbait/emoji, desc url+hashtag counts, view_count_log, channel age + mean inter-upload). Fused into `data/features/combined.parquet` shape (94, 1296) = 512 visual + 384 transcript + 384 title_desc + 16 scalars. New class signals: `desc_url_count` median control=3 vs offender=0; `view_count_log` median 5.1 vs 3.9.
- M5 — Vector DB + UMAP: ✅ done (2026-04-30) — Chroma persistent collection `videos` at `data/chroma/`, populated with the **1280-dim embedding-only slice** (visual+transcript+title_desc; scalars omitted to keep cosine geometry meaningful). UMAP rendered to `data/features/umap.png` and `notebooks/01_explore_embeddings.ipynb`: 10 distinct channel clusters (channel-level leakage is real — M6 holdout must split by channel), but offender/control halves are visibly separated on UMAP-1. Channel-level LOO kNN (k=5, cosine-weighted): **acc=0.81, prec=0.86, rec=0.71, F1=0.78** — already over the M6 bar of 0.7 precision before any classifier is trained. Hard cases: ch1 (offender, ASMR-shorts) → 1/9; ch20 (offender, only 5 videos) → 0/5; ch72 (control) → 6/10.
- M6 — LogReg baseline: ⏳ pending
- M7 — MLP classifier: ⏳ pending
- M8 — Predict CLI + FastAPI: ⏳ pending
- M9 — Error analysis: ⏳ pending
- Stretch — RAG explanation: ⏳ pending

## Next action when resuming

Start M6 — LogReg baseline. See "M6" in PLAN.md. Train `sklearn.linear_model.LogisticRegression` on the full 1296-dim `combined.parquet` feature matrix (visual+transcript+title_desc are L2-normed; the 16 scalars are raw — wrap in a `StandardScaler` inside a Pipeline so the scale is fit on the train split only). **Use `GroupKFold` with `channel_id` as the group** — channel-level leakage is large here (UMAP shows 10 distinct channel clusters), so video-level CV will overstate performance.

Compare against the M5 channel-level kNN baseline (acc=0.81, F1=0.78). Plan's bar to clear: precision ≥ 0.7 on the channel-level holdout. Log to mlflow under `data/mlruns/` with run name like `logreg-baseline`. Save the fitted pipeline (Pipeline → joblib) under `data/models/logreg.joblib` so M7's MLP can compete against the same artifacts.

Useful entry points:
- `medea.features.pipeline.build_dataset()` → `FusedDataset(video_ids, channel_ids, labels, X, block_sizes)` already labelled and grouped.
- `medea.storage.vector_db.knn_query(col, vec, k, exclude_ids=...)` for sanity comparisons.
- `data/features/umap.png` for visual reference; the notebook re-renders it.

## Notes

- Plan file: [PLAN.md](PLAN.md) (also at `~/.claude/plans/i-have-an-idea-twinkly-thunder.md`).
- Project rules: no git/commits per user preference for local practice projects.
- Hardware: RTX 3080 10GB, Python 3.12.10, Windows 11.
- ffmpeg is bundled via `imageio-ffmpeg` and copied as `data/.bin/ffmpeg(.exe)` so yt-dlp finds it; it also gets prepended to `PATH` at runtime.
