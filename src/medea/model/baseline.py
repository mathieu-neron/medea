"""Baseline classifier: StandardScaler → LogisticRegression on the fused vector.

Why scale everything (including the L2-normed embedding blocks): for a linear
model with L2 regularization the per-feature gradient and per-feature penalty
must be on comparable scales, otherwise the unscaled scalar block (with values
up to ~3879) drives the loss while embedding columns (per-column std ~1/√D)
contribute almost nothing. Per-column z-scoring fixes this. The cosine
geometry that the M5 kNN relied on is intentionally discarded — it's a
different model with different invariances.

class_weight='balanced' is set defensively (44 offender / 50 control is close
enough that it barely matters; safer than letting the prior leak in).
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg_pipeline(C: float = 1.0, max_iter: int = 5000) -> Pipeline:
    # penalty defaults to L2 — explicitly setting it is deprecated in sklearn 1.8.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=max_iter,
                ),
            ),
        ]
    )
