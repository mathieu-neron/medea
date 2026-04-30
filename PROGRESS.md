# Medea — Progress

One line per milestone. Update when finishing a milestone.

- M1 — Bootstrap: ✅ done (2026-04-28) — project layout, deps installed via `uv sync` (507 pkgs), GPU verified
- M2 — Ingest: ✅ done (2026-04-28) — 94 clips (44 offender / 50 control) at 30s middle slices, ≤720p, in `data/raw/`. SQLite at `data/medea.db` (channels + videos)
- M3 — Visual + audio features: ✅ done (2026-04-29) — `data/features/visual.parquet` (94×512 CLIP ViT-B/32 mean-pooled, L2-normed) and `data/features/audio.parquet` (faster-whisper transcript + lang + wav2vec2 ai_voice_prob). Strong class signal already: offender ai_voice median 0.86 vs control 0.37; offender mean transcript 123 chars vs control 303.
- M4 — Text + metadata features: ✅ done (2026-04-30) — `data/features/text.parquet` (94×384 transcript + title+desc MiniLM embeddings, L2-normed, zero vec for empty transcripts) and `data/features/metadata.parquet` (94×15 handcrafted scalars: title shape/caps/clickbait/emoji, desc url+hashtag counts, view_count_log, channel age + mean inter-upload). Fused into `data/features/combined.parquet` shape (94, 1296) = 512 visual + 384 transcript + 384 title_desc + 16 scalars. New class signals: `desc_url_count` median control=3 vs offender=0; `view_count_log` median 5.1 vs 3.9.
- M5 — Vector DB + UMAP: ✅ done (2026-04-30) — Chroma persistent collection `videos` at `data/chroma/`, populated with the **1280-dim embedding-only slice** (visual+transcript+title_desc; scalars omitted to keep cosine geometry meaningful). UMAP rendered to `data/features/umap.png` and `notebooks/01_explore_embeddings.ipynb`: 10 distinct channel clusters (channel-level leakage is real — M6 holdout must split by channel), but offender/control halves are visibly separated on UMAP-1. Channel-level LOO kNN (k=5, cosine-weighted): **acc=0.81, prec=0.86, rec=0.71, F1=0.78** — already over the M6 bar of 0.7 precision before any classifier is trained. Hard cases: ch1 (offender, ASMR-shorts) → 1/9; ch20 (offender, only 5 videos) → 0/5; ch72 (control) → 6/10.
- M6 — LogReg baseline: ✅ done (2026-04-30) — `StandardScaler → LogisticRegression(C=1.0, class_weight='balanced')` on the full 1296-dim fused vector, evaluated with `LeaveOneGroupOut` over `channel_id` (10 folds). Out-of-fold metrics: **acc=0.809, prec=0.783, rec=0.818, F1=0.800, ROC-AUC=0.934**. Plan's precision-≥-0.7 bar cleared. Vs M5 kNN: F1 0.78→0.80, recall 0.71→0.82 (LogReg moved the boundary toward catching more offenders), precision 0.86→0.78. Per-channel: improved on offender failures (ch1 0.11→0.44, ch20 0.00→0.40, ch72 0.60→0.70) but regressed on ch61 ctrl (0.90→0.30, mean_score 0.59 — genuine confusion, not threshold). Final pipeline saved to `data/models/logreg.joblib` (refit on all 94 samples); mlflow run logged under `data/mlruns/medea-baseline/`.
- M7 — MLP classifier: ⏳ pending
- M8 — Predict CLI + FastAPI: ⏳ pending
- M9 — Error analysis: ⏳ pending
- Stretch — RAG explanation: ⏳ pending

## Next action when resuming

Start M7 — MLP head. See "M7" in PLAN.md. Build a small PyTorch MLP (~2 hidden layers, dropout) on the same 1296-dim fused vector, evaluated with the **same `LeaveOneGroupOut(channel_id)`** protocol as M6 so the comparison is fair. Plan's gating: MLP must beat LogReg on F1 by a non-trivial margin — if it doesn't, that's a feature-quality problem, not a model problem (go back to M3/M4). Today's bar is F1 > 0.80; ROC-AUC>0.93. With only 94 samples and 1296 features, watch overfitting hard: small width (≤128), dropout ≥0.3, early stopping on a per-fold validation slice (one of the train channels), L2 weight decay.

Reuse `medea.eval.metrics.binary_metrics` and the `LeaveOneGroupOut` skeleton from `scripts/train.py`. Log to mlflow under the same `medea-baseline` experiment with run name `mlp-baseline`; save `data/models/mlp.pt`. **The hard-case channels to watch** are ch1 (off, ASMR-shorts), ch20 (off, n=5), ch61 (ctrl). LogReg flipped ch61's predictions vs kNN — see whether the MLP recovers it without breaking ch1/ch20.

Useful entry points:
- `medea.features.pipeline.build_dataset()` → `FusedDataset(video_ids, channel_ids, labels, X, block_sizes)`.
- `medea.model.baseline.make_logreg_pipeline()` and `data/models/logreg.joblib` for direct comparison.
- `python scripts/train.py` re-runs the LogReg baseline.

## Notes

- Plan file: [PLAN.md](PLAN.md) (also at `~/.claude/plans/i-have-an-idea-twinkly-thunder.md`).
- Project rules: no git/commits per user preference for local practice projects.
- Hardware: RTX 3080 10GB, Python 3.12.10, Windows 11.
- ffmpeg is bundled via `imageio-ffmpeg` and copied as `data/.bin/ffmpeg(.exe)` so yt-dlp finds it; it also gets prepended to `PATH` at runtime.
