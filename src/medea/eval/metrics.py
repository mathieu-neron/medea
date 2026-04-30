"""Binary-classification metrics shared by baseline and MLP training scripts.

Kept thin on purpose — sklearn already does the heavy lifting; this module
just packages the numbers we always want together (accuracy, precision,
recall, F1, ROC-AUC, confusion matrix) in a single dict suitable for
mlflow.log_metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    tp: int
    fp: int
    fn: int
    tn: int

    def as_log_dict(self, prefix: str = "") -> dict[str, float]:
        d = {
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}precision": self.precision,
            f"{prefix}recall": self.recall,
            f"{prefix}f1": self.f1,
            f"{prefix}tp": float(self.tp),
            f"{prefix}fp": float(self.fp),
            f"{prefix}fn": float(self.fn),
            f"{prefix}tn": float(self.tn),
        }
        if self.roc_auc is not None:
            d[f"{prefix}roc_auc"] = self.roc_auc
        return d


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> BinaryMetrics:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc: float | None = None
    if y_score is not None and len(set(y_true.tolist())) > 1:
        auc = float(roc_auc_score(y_true, y_score))
    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
    )


def format_confusion(m: BinaryMetrics) -> str:
    """Pretty 2x2 confusion matrix for terminal output."""
    return (
        "                pred 0   pred 1\n"
        f"   actual 0    {m.tn:6d}   {m.fp:6d}\n"
        f"   actual 1    {m.fn:6d}   {m.tp:6d}"
    )
