# Medea — Progress

One line per milestone. Update when finishing a milestone.

- M1 — Bootstrap: ✅ done (2026-04-28) — project layout, deps installed via `uv sync` (507 pkgs), GPU verified
- M2 — Ingest: ⏳ pending
- M3 — Visual + audio features: ⏳ pending
- M4 — Text + metadata features: ⏳ pending
- M5 — Vector DB + UMAP: ⏳ pending
- M6 — LogReg baseline: ⏳ pending
- M7 — MLP classifier: ⏳ pending
- M8 — Predict CLI + FastAPI: ⏳ pending
- M9 — Error analysis: ⏳ pending
- Stretch — RAG explanation: ⏳ pending

## Next action when resuming

Start M2. See "M2 — Ingest" in PLAN.md. Build it together — discuss design choices, write code, test on a small batch — don't pre-write later milestones.

Before M2 can be tested end-to-end, populate `data/seeds/offenders.txt` and `data/seeds/controls.txt` with 5-10 channels each (see `data/seeds/README.md` for guidance).

## Notes

- Plan file: [PLAN.md](PLAN.md) (also at `~/.claude/plans/i-have-an-idea-twinkly-thunder.md`).
- Project rules: no git/commits per user preference for local practice projects.
- Hardware: RTX 3080 10GB, Python 3.12.10, Windows 11.
