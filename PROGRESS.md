# Medea — Progress

One line per milestone. Update when finishing a milestone.

- M1 — Bootstrap: ✅ done (2026-04-28) — project layout, deps installed via `uv sync` (507 pkgs), GPU verified
- M2 — Ingest: ✅ done (2026-04-28) — 94 clips (44 offender / 50 control) at 30s middle slices, ≤720p, in `data/raw/`. SQLite at `data/medea.db` (channels + videos)
- M3 — Visual + audio features: ✅ done (2026-04-29) — `data/features/visual.parquet` (94×512 CLIP ViT-B/32 mean-pooled, L2-normed) and `data/features/audio.parquet` (faster-whisper transcript + lang + wav2vec2 ai_voice_prob). Strong class signal already: offender ai_voice median 0.86 vs control 0.37; offender mean transcript 123 chars vs control 303.
- M4 — Text + metadata features: ✅ done (2026-04-30) — `data/features/text.parquet` (94×384 transcript + title+desc MiniLM embeddings, L2-normed, zero vec for empty transcripts) and `data/features/metadata.parquet` (94×15 handcrafted scalars: title shape/caps/clickbait/emoji, desc url+hashtag counts, view_count_log, channel age + mean inter-upload). Fused into `data/features/combined.parquet` shape (94, 1296) = 512 visual + 384 transcript + 384 title_desc + 16 scalars. New class signals: `desc_url_count` median control=3 vs offender=0; `view_count_log` median 5.1 vs 3.9.
- M5 — Vector DB + UMAP: ✅ done (2026-04-30) — Chroma persistent collection `videos` at `data/chroma/`, populated with the **1280-dim embedding-only slice** (visual+transcript+title_desc; scalars omitted to keep cosine geometry meaningful). UMAP rendered to `data/features/umap.png` and `notebooks/01_explore_embeddings.ipynb`: 10 distinct channel clusters (channel-level leakage is real — M6 holdout must split by channel), but offender/control halves are visibly separated on UMAP-1. Channel-level LOO kNN (k=5, cosine-weighted): **acc=0.81, prec=0.86, rec=0.71, F1=0.78** — already over the M6 bar of 0.7 precision before any classifier is trained. Hard cases: ch1 (offender, ASMR-shorts) → 1/9; ch20 (offender, only 5 videos) → 0/5; ch72 (control) → 6/10.
- M6 — LogReg baseline: ✅ done (2026-04-30) — `StandardScaler → LogisticRegression(C=1.0, class_weight='balanced')` on the full 1296-dim fused vector, evaluated with `LeaveOneGroupOut` over `channel_id` (10 folds). Out-of-fold metrics: **acc=0.809, prec=0.783, rec=0.818, F1=0.800, ROC-AUC=0.934**. Plan's precision-≥-0.7 bar cleared. Vs M5 kNN: F1 0.78→0.80, recall 0.71→0.82 (LogReg moved the boundary toward catching more offenders), precision 0.86→0.78. Per-channel: improved on offender failures (ch1 0.11→0.44, ch20 0.00→0.40, ch72 0.60→0.70) but regressed on ch61 ctrl (0.90→0.30, mean_score 0.59 — genuine confusion, not threshold). Final pipeline saved to `data/models/logreg.joblib` (refit on all 94 samples); mlflow run logged under `data/mlruns/medea-baseline/`.
- M7 — MLP classifier: ✅ done (2026-04-30) — small PyTorch MLP (1296→128→64→1 with LayerNorm + GELU + Dropout=0.4), AdamW(lr=1e-3, wd=1e-3), BCEWithLogitsLoss with `pos_weight=neg/pos`, batch=16, max 200 epochs. Same outer `LeaveOneGroupOut(channel_id)` as M6, plus an inner channel-level val split (1 train channel held out per fold) for early stopping (patience=25). Out-of-fold metrics: **acc=0.872, prec=0.833, rec=0.909, F1=0.870, ROC-AUC=0.933**. Plan's "non-trivial F1 margin over LogReg" bar met (0.80→0.87 = +0.07). Per-channel improvements over LogReg: ch1 0.44→0.67, ch20 0.40→0.80, ch72 0.70→0.90. **ch61 ctrl still stuck at 0.30** (mean_score 0.65) — both LogReg and MLP fail on it the same way; flagged for M9 error analysis. Final model refit on all 94 for median best epoch (150) and saved to `data/models/mlp.pt` (state_dict + scaler stats + arch config); mlflow run logged under `medea-baseline/mlp-baseline`.
- M8 — Predict CLI + FastAPI: ⏳ pending
- M9 — Error analysis: ⏳ pending
- Stretch — RAG explanation: ⏳ pending

## Next action when resuming

Start M8 — End-to-end CLI + FastAPI. See "M8" in PLAN.md. Build `scripts/predict.py <youtube_url>` that downloads a 30s middle clip (reuse `medea.ingest.youtube`), extracts all four feature modalities (reuse the encoders directly — no SQLite write), fuses them via `features.pipeline` logic, and produces `{score, top_k_neighbors, top_features}`. Then expose the same flow via `POST /predict` in `src/medea/api/server.py` (FastAPI + uvicorn, both already in deps).

Default inference model: `data/models/mlp.pt` (the M7 head — F1=0.870 vs LogReg 0.800). Top-k neighbors come from `medea.storage.vector_db.knn_query(...)` over the 1280-dim embedding-only slice — same code path as M5. "Top features" is a stretch: surface the largest |z-scaled value| × |LogReg coef| products from the M6 model for interpretability, since the MLP doesn't have direct coefficients.

Verification per PLAN.md: run on 3 held-out URLs (1 known offender, 1 known clean, 1 ambiguous) and inspect both score and neighbors. None of the 10 seed channels should be the "held-out" — pick fresh ones.

Useful entry points:
- `medea.ingest.youtube.download_clip(url, ...)` — already produces 30s middle clips at ≤720p.
- Encoders: `VisualEncoder().embed_clip(path)`, `AudioEncoder().features(path)`, `TextEncoder().features(...)`, `metadata.video_features(...)` + `channel_features([upload_date])` (channel features will be sparse for a single fresh video — that's fine).
- Models: `joblib.load('data/models/logreg.joblib')` returns `{pipeline, block_sizes, ...}`; `torch.load('data/models/mlp.pt')` returns `{state_dict, scaler_mean, scaler_scale, hidden, dropout, ...}` — rehydrate with `MLP(input_dim=, hidden=, dropout=)` then load state.

Open question (was deferred at M5): per-modality classifiers + ensemble (option b) vs single fused vector (current). M7's wins are concentrated on offender channels (ch1, ch20) and the persistent ch61 failure suggests a single fused linear/MLP boundary may be hitting a ceiling on this 10-channel set. Worth revisiting after M8 if predictions on fresh URLs feel brittle, but **don't pre-build it** — bigger seed list (M9 territory) probably matters more than ensembling.

## Notes

- Plan file: [PLAN.md](PLAN.md) (also at `~/.claude/plans/i-have-an-idea-twinkly-thunder.md`).
- Project rules: no git/commits per user preference for local practice projects.
- Hardware: RTX 3080 10GB, Python 3.12.10, Windows 11.
- ffmpeg is bundled via `imageio-ffmpeg` and copied as `data/.bin/ffmpeg(.exe)` so yt-dlp finds it; it also gets prepended to `PATH` at runtime.
