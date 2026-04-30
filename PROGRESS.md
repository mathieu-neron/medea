# Medea — Progress

One line per milestone. Update when finishing a milestone.

- M1 — Bootstrap: ✅ done (2026-04-28) — project layout, deps installed via `uv sync` (507 pkgs), GPU verified
- M2 — Ingest: ✅ done (2026-04-28) — 94 clips (44 offender / 50 control) at 30s middle slices, ≤720p, in `data/raw/`. SQLite at `data/medea.db` (channels + videos)
- M3 — Visual + audio features: ✅ done (2026-04-29) — `data/features/visual.parquet` (94×512 CLIP ViT-B/32 mean-pooled, L2-normed) and `data/features/audio.parquet` (faster-whisper transcript + lang + wav2vec2 ai_voice_prob). Strong class signal already: offender ai_voice median 0.86 vs control 0.37; offender mean transcript 123 chars vs control 303.
- M4 — Text + metadata features: ✅ done (2026-04-30) — `data/features/text.parquet` (94×384 transcript + title+desc MiniLM embeddings, L2-normed, zero vec for empty transcripts) and `data/features/metadata.parquet` (94×15 handcrafted scalars: title shape/caps/clickbait/emoji, desc url+hashtag counts, view_count_log, channel age + mean inter-upload). Fused into `data/features/combined.parquet` shape (94, 1296) = 512 visual + 384 transcript + 384 title_desc + 16 scalars. New class signals: `desc_url_count` median control=3 vs offender=0; `view_count_log` median 5.1 vs 3.9.
- M5 — Vector DB + UMAP: ✅ done (2026-04-30) — Chroma persistent collection `videos` at `data/chroma/`, populated with the **1280-dim embedding-only slice** (visual+transcript+title_desc; scalars omitted to keep cosine geometry meaningful). UMAP rendered to `data/features/umap.png` and `notebooks/01_explore_embeddings.ipynb`: 10 distinct channel clusters (channel-level leakage is real — M6 holdout must split by channel), but offender/control halves are visibly separated on UMAP-1. Channel-level LOO kNN (k=5, cosine-weighted): **acc=0.81, prec=0.86, rec=0.71, F1=0.78** — already over the M6 bar of 0.7 precision before any classifier is trained. Hard cases: ch1 (offender, ASMR-shorts) → 1/9; ch20 (offender, only 5 videos) → 0/5; ch72 (control) → 6/10.
- M6 — LogReg baseline: ✅ done (2026-04-30) — `StandardScaler → LogisticRegression(C=1.0, class_weight='balanced')` on the full 1296-dim fused vector, evaluated with `LeaveOneGroupOut` over `channel_id` (10 folds). Out-of-fold metrics: **acc=0.809, prec=0.783, rec=0.818, F1=0.800, ROC-AUC=0.934**. Plan's precision-≥-0.7 bar cleared. Vs M5 kNN: F1 0.78→0.80, recall 0.71→0.82 (LogReg moved the boundary toward catching more offenders), precision 0.86→0.78. Per-channel: improved on offender failures (ch1 0.11→0.44, ch20 0.00→0.40, ch72 0.60→0.70) but regressed on ch61 ctrl (0.90→0.30, mean_score 0.59 — genuine confusion, not threshold). Final pipeline saved to `data/models/logreg.joblib` (refit on all 94 samples); mlflow run logged under `data/mlruns/medea-baseline/`.
- M7 — MLP classifier: ✅ done (2026-04-30) — small PyTorch MLP (1296→128→64→1 with LayerNorm + GELU + Dropout=0.4), AdamW(lr=1e-3, wd=1e-3), BCEWithLogitsLoss with `pos_weight=neg/pos`, batch=16, max 200 epochs. Same outer `LeaveOneGroupOut(channel_id)` as M6, plus an inner channel-level val split (1 train channel held out per fold) for early stopping (patience=25). Out-of-fold metrics: **acc=0.872, prec=0.833, rec=0.909, F1=0.870, ROC-AUC=0.933**. Plan's "non-trivial F1 margin over LogReg" bar met (0.80→0.87 = +0.07). Per-channel improvements over LogReg: ch1 0.44→0.67, ch20 0.40→0.80, ch72 0.70→0.90. **ch61 ctrl still stuck at 0.30** (mean_score 0.65) — both LogReg and MLP fail on it the same way; flagged for M9 error analysis. Final model refit on all 94 for median best epoch (150) and saved to `data/models/mlp.pt` (state_dict + scaler stats + arch config); mlflow run logged under `medea-baseline/mlp-baseline`.
- M8 — Predict CLI + FastAPI: ✅ done (2026-04-30) — `medea.model.infer.Predictor` is the single source of truth for inference: URL → 30s clip → 4-modality features → fused vector (built byte-for-byte the same as `features.pipeline`) → score + top-k neighbors + LogReg-surrogate attribution. CLI at `scripts/predict.py <url> [--model mlp|logreg] [--k N] [--json]`; FastAPI at `medea.api.server:app` exposing `POST /predict` and `GET /health`, with the MLP predictor warmed up in the lifespan handler. **Bug found and fixed**: at inference we have only 1 video per channel, but training channels had 5–10. The three channel scalars (count, age, mean inter-upload) became multi-sigma OOD inputs that drove a spurious "AI" signal — fixed by substituting the StandardScaler's training-set mean for those columns at inference (z-score → 0, contribution → 0). Verified end-to-end on 3 in-sample clips (P(AI)=0.000/1.000/1.000 as expected) and one fresh URL (Tom Scott "Diggy Diggy Hole"). **MLP-vs-LogReg disagreement on OOD music video content**: MLP=0.997, LogReg=0.345 — the MLP, despite winning in CV, extrapolates much more aggressively. Flagged for M9.
- M9 — Error analysis: ⏳ pending
- Stretch — RAG explanation: ⏳ pending

## Next action when resuming

Start M9 — Error analysis & iteration. See "M9" in PLAN.md. The pipeline is end-to-end now; the work shifts from building to learning what the model actually got wrong and why.

**Three concrete leads from M5–M8 that should drive the first iteration:**

1. **ch61 ctrl is unfixable in current feature space.** kNN: 0.90, LogReg: 0.30, MLP: 0.30. mean_score ≈ 0.6 from both classifiers. Look at the actual videos in `data/raw/` for ch61 — what makes them look AI-y? If it's a topical-domain confound (e.g. they cover topics also covered by offenders), the answer is more controls in adjacent topics. If it's stylistic (animation, narration cadence), the visual/audio features need to discriminate it.

2. **MLP extrapolates wildly on OOD inputs.** Tom Scott "Diggy Diggy Hole" (clearly human music video): MLP=0.997, LogReg=0.345 on the same fused vector. The MLP wins in-distribution CV (F1=0.87 vs 0.80) but is much less calibrated outside the 10-channel training cone. Two reasonable responses: (a) widen the seed list (more channels, more genres) and retrain — likely the highest-leverage fix; (b) prefer LogReg as the deployed model for robustness, accept the F1 hit. Don't pre-build (b) — the seed-list expansion may make this moot.

3. **Channel-scalar inference shim is a workaround, not a fix.** `medea.model.infer._NEUTRALIZED_CHANNEL_COLS` zeroes out three columns at predict time. The right fix is to actually fetch the channel's recent video upload dates via `list_channel_videos` and compute real channel features at inference — adds ~10 HTTP round-trips per prediction but makes the inference and training distributions match.

**Recommended workflow:** run `python scripts/predict.py <url>` on each ingested video (use `--json` and a script over the SQLite ids), collect top false-positives and false-negatives, look at the clips manually, decide which features need to change. Then go back to M3/M4 if features need to change, or expand seeds + re-ingest if data needs to grow.

Useful entry points:
- `medea.model.infer.Predictor` — same path CLI/API/error analysis go through.
- `python scripts/extract.py --modality combined` — rebuilds `combined.parquet` after any feature change.
- `python scripts/train.py && python scripts/train_mlp.py` — re-runs both classifiers; mlflow runs accumulate under `data/mlruns/`.
- `python scripts/build_chroma.py` — repopulates Chroma after any feature change.

## Notes

- Plan file: [PLAN.md](PLAN.md) (also at `~/.claude/plans/i-have-an-idea-twinkly-thunder.md`).
- Project rules: no git/commits per user preference for local practice projects.
- Hardware: RTX 3080 10GB, Python 3.12.10, Windows 11.
- ffmpeg is bundled via `imageio-ffmpeg` and copied as `data/.bin/ffmpeg(.exe)` so yt-dlp finds it; it also gets prepended to `PATH` at runtime.
