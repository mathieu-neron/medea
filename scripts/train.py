"""Train the M6 LogReg baseline with channel-grouped cross-validation.

Why channel-grouped: in M5 the UMAP showed 10 distinct channel-level clusters,
which means a video-level split would mostly memorize channel identity rather
than "AI-ness." LeaveOneGroupOut over ``channel_id`` is the honest measure —
each fold tests on exactly one held-out channel.

After CV we refit on all 94 samples and persist to ``data/models/logreg.joblib``
so M7 / M8 can load the same artifact.

Usage:
    python scripts/train.py
    python scripts/train.py --C 0.5 --no-mlflow
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import LeaveOneGroupOut

from medea.config import MLRUNS_DIR, MODELS_DIR, ensure_dirs
from medea.eval.metrics import binary_metrics, format_confusion
from medea.features.pipeline import build_dataset
from medea.model.baseline import make_logreg_pipeline

app = typer.Typer(add_completion=False, help="Train Medea's LogReg baseline.")
console = Console()


def _per_channel_table(
    *,
    channel_ids: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
) -> Table:
    table = Table(title="Channel-level held-out predictions", show_lines=False)
    table.add_column("channel", justify="right")
    table.add_column("label", justify="center")
    table.add_column("n", justify="right")
    table.add_column("acc", justify="right")
    table.add_column("mean_score", justify="right")
    for cid in sorted(set(channel_ids.tolist())):
        m = channel_ids == cid
        lbl = int(labels[m][0])
        acc = float((preds[m] == labels[m]).mean())
        mean_score = float(scores[m].mean())
        color = "green" if acc >= 0.8 else ("yellow" if acc >= 0.5 else "red")
        table.add_row(
            f"{cid}",
            "off" if lbl else "ctrl",
            f"{int(m.sum())}",
            f"[{color}]{acc:.2f}[/{color}]",
            f"{mean_score:.2f}",
        )
    return table


@app.command()
def main(
    C: float = typer.Option(1.0, help="Inverse regularization strength."),
    use_mlflow: bool = typer.Option(True, "--mlflow/--no-mlflow"),
    run_name: str = typer.Option("logreg-baseline"),
    log_level: str = typer.Option("INFO"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=False, show_path=False)],
    )
    ensure_dirs()

    ds = build_dataset()
    X = ds.X
    y = ds.labels
    groups = ds.channel_ids
    console.print(
        f"[bold]dataset[/bold]: X={X.shape}  pos={int(y.sum())}/neg={int((y==0).sum())}  "
        f"channels={len(set(groups.tolist()))}"
    )

    # Channel-grouped LOO: 10 channels → 10 folds.
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)

    oof_pred = np.full(len(y), -1, dtype=int)
    oof_score = np.full(len(y), np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        held_out = int(groups[test_idx][0])
        pipe = make_logreg_pipeline(C=C)
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        oof_pred[test_idx] = pred
        oof_score[test_idx] = proba
        console.print(
            f"  fold {fold+1}/{n_folds}  held-out=ch{held_out}  "
            f"n_test={len(test_idx)}  acc={float((pred==y[test_idx]).mean()):.2f}"
        )

    metrics = binary_metrics(y, oof_pred, oof_score)
    console.rule("[bold green]Channel-level out-of-fold results")
    console.print(
        f"acc={metrics.accuracy:.3f}  prec={metrics.precision:.3f}  "
        f"rec={metrics.recall:.3f}  f1={metrics.f1:.3f}  "
        f"roc_auc={metrics.roc_auc:.3f}"
    )
    console.print(format_confusion(metrics))
    console.print()
    console.print(_per_channel_table(
        channel_ids=groups, labels=y, preds=oof_pred, scores=oof_score
    ))

    # Reference baseline — compare against M5 kNN (recorded in PROGRESS.md).
    console.print(
        "\n[dim]M5 kNN baseline (channel-level LOO): acc=0.81 prec=0.86 "
        "rec=0.71 f1=0.78[/dim]"
    )

    # Refit on all 94 samples for the deployed artifact.
    final_pipe = make_logreg_pipeline(C=C)
    final_pipe.fit(X, y)
    model_path = MODELS_DIR / "logreg.joblib"
    payload = {
        "pipeline": final_pipe,
        "block_sizes": ds.block_sizes,
        "C": C,
        "feature_dim": int(X.shape[1]),
    }
    joblib.dump(payload, model_path)
    console.print(f"[green]saved[/green] {model_path.relative_to(Path.cwd())}")

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
        mlflow.set_experiment("medea-baseline")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model": "logreg",
                    "C": C,
                    "penalty": "l2",
                    "class_weight": "balanced",
                    "feature_dim": X.shape[1],
                    "n_samples": X.shape[0],
                    "n_channels": len(set(groups.tolist())),
                    "cv": "LeaveOneGroupOut(channel_id)",
                }
            )
            mlflow.log_metrics(metrics.as_log_dict(prefix="cv_"))
            mlflow.log_artifact(str(model_path), artifact_path="model")
        console.print(f"[green]mlflow run logged[/green] under {MLRUNS_DIR}")


if __name__ == "__main__":
    app()
