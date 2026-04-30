"""Train the M7 MLP head with channel-grouped CV and early stopping.

Same outer split as M6 (``LeaveOneGroupOut`` over ``channel_id``, 10 folds),
plus an inner channel-level val split — one train channel is held out per
fold for early stopping. ``StandardScaler`` is fit on the inner-train slice
only so val + test stay leakage-free.

Plan's gating call (M7): MLP must beat LogReg on F1 by a non-trivial margin,
otherwise features are the bottleneck and we should go back to M3/M4 rather
than tuning endlessly.

Usage:
    python scripts/train_mlp.py
    python scripts/train_mlp.py --hidden 128,64 --dropout 0.4 --no-mlflow
"""

from __future__ import annotations

import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from medea.config import MLRUNS_DIR, MODELS_DIR, ensure_dirs
from medea.eval.metrics import binary_metrics, format_confusion
from medea.features.pipeline import build_dataset
from medea.model.mlp import MLP

app = typer.Typer(add_completion=False, help="Train Medea's MLP head.")
console = Console()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_val_channel(train_channel_ids: np.ndarray, fold_idx: int) -> int:
    """Deterministic inner val pick: rotate through sorted train channels.

    A different val channel per fold spreads the noise. Picking 1/9 channels
    leaves ~8 channels (~80 samples) for inner training — small but workable.
    """
    sorted_ch = sorted(set(train_channel_ids.tolist()))
    return int(sorted_ch[fold_idx % len(sorted_ch)])


def _train_one_fold(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    hidden: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, dict, StandardScaler, MLP]:
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    pos = max(1, int((y_train == 1).sum()))
    neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)

    model = MLP(input_dim=X_train.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_s), torch.from_numpy(y_train.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val_s).to(DEVICE)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(DEVICE)

    best_val = float("inf")
    best_epoch = -1
    best_state: dict | None = None
    epochs_without_improvement = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(loss_fn(val_logits, y_val_t).item())

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test_s).to(DEVICE)
        test_logits = model(X_test_t).cpu().numpy()
    test_proba = 1.0 / (1.0 + np.exp(-test_logits))
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return test_proba, history, scaler, model


def _per_channel_table(
    *,
    channel_ids: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    knn_acc: dict[int, float],
    logreg_acc: dict[int, float],
) -> Table:
    table = Table(title="Channel-level held-out predictions vs prior baselines")
    table.add_column("ch", justify="right")
    table.add_column("label", justify="center")
    table.add_column("n", justify="right")
    table.add_column("kNN", justify="right")
    table.add_column("LogReg", justify="right")
    table.add_column("MLP", justify="right")
    table.add_column("MLP score", justify="right")
    for cid in sorted(set(channel_ids.tolist())):
        m = channel_ids == cid
        lbl = int(labels[m][0])
        acc = float((preds[m] == labels[m]).mean())
        score = float(scores[m].mean())
        color = "green" if acc >= 0.8 else ("yellow" if acc >= 0.5 else "red")
        table.add_row(
            f"{cid}",
            "off" if lbl else "ctrl",
            f"{int(m.sum())}",
            f"{knn_acc.get(cid, float('nan')):.2f}",
            f"{logreg_acc.get(cid, float('nan')):.2f}",
            f"[{color}]{acc:.2f}[/{color}]",
            f"{score:.2f}",
        )
    return table


# Captured from earlier runs so the per-channel table can show deltas without
# re-running both baselines. Kept here, not in code, because they're descriptive
# numbers from one specific dataset state — re-derive if features change.
KNN_PER_CH = {
    1: 0.111, 6: 1.00, 9: 1.00, 20: 0.00, 26: 1.00,
    47: 1.00, 61: 0.90, 72: 0.60, 83: 1.00, 94: 1.00,
}
LOGREG_PER_CH = {
    1: 0.444, 6: 1.00, 9: 1.00, 20: 0.40, 26: 1.00,
    47: 1.00, 61: 0.30, 72: 0.70, 83: 1.00, 94: 1.00,
}


def _parse_hidden(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


@app.command()
def main(
    hidden: str = typer.Option("128,64", help="Comma-sep hidden widths."),
    dropout: float = typer.Option(0.4),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-3),
    batch_size: int = typer.Option(16),
    max_epochs: int = typer.Option(200),
    patience: int = typer.Option(25),
    use_mlflow: bool = typer.Option(True, "--mlflow/--no-mlflow"),
    run_name: str = typer.Option("mlp-baseline"),
    log_level: str = typer.Option("INFO"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=False, show_path=False)],
    )
    ensure_dirs()
    _set_seed()

    hidden_t = _parse_hidden(hidden)

    ds = build_dataset()
    X = ds.X.astype(np.float32)
    y = ds.labels.astype(int)
    groups = ds.channel_ids
    console.print(
        f"[bold]dataset[/bold]: X={X.shape}  pos={int(y.sum())}/neg={int((y==0).sum())}  "
        f"channels={len(set(groups.tolist()))}  device={DEVICE}"
    )
    console.print(
        f"[bold]model[/bold]: hidden={hidden_t}  dropout={dropout}  "
        f"lr={lr}  wd={weight_decay}  batch={batch_size}  max_epochs={max_epochs}"
    )

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)

    oof_pred = np.full(len(y), -1, dtype=int)
    oof_score = np.full(len(y), np.nan, dtype=float)
    best_epochs: list[int] = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        held_out = int(groups[test_idx][0])
        train_groups = groups[train_idx]
        val_ch = _pick_val_channel(train_groups, fold)
        val_mask = train_groups == val_ch
        inner_train_idx = train_idx[~val_mask]
        inner_val_idx = train_idx[val_mask]

        proba, history, _, _ = _train_one_fold(
            X_train=X[inner_train_idx],
            y_train=y[inner_train_idx],
            X_val=X[inner_val_idx],
            y_val=y[inner_val_idx],
            X_test=X[test_idx],
            hidden=hidden_t,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
        )
        pred = (proba >= 0.5).astype(int)
        oof_pred[test_idx] = pred
        oof_score[test_idx] = proba
        best_epochs.append(history["best_epoch"])
        console.print(
            f"  fold {fold+1}/{n_folds}  test=ch{held_out}  val=ch{val_ch}  "
            f"best_epoch={history['best_epoch']:3d}  "
            f"val_loss={history['best_val_loss']:.3f}  "
            f"acc={float((pred == y[test_idx]).mean()):.2f}"
        )

    metrics = binary_metrics(y, oof_pred, oof_score)
    console.rule("[bold green]MLP channel-level out-of-fold")
    console.print(
        f"acc={metrics.accuracy:.3f}  prec={metrics.precision:.3f}  "
        f"rec={metrics.recall:.3f}  f1={metrics.f1:.3f}  "
        f"roc_auc={metrics.roc_auc:.3f}"
    )
    console.print(format_confusion(metrics))
    console.print()
    console.print(_per_channel_table(
        channel_ids=groups, labels=y, preds=oof_pred, scores=oof_score,
        knn_acc=KNN_PER_CH, logreg_acc=LOGREG_PER_CH,
    ))
    console.print(
        "\n[dim]M5 kNN: acc=0.81 prec=0.86 rec=0.71 f1=0.78 | "
        "M6 LogReg: acc=0.81 prec=0.78 rec=0.82 f1=0.80 roc_auc=0.93[/dim]"
    )

    # Final fit on all 94 — reuse the median best-epoch from CV folds and skip
    # early stopping on the deployed artifact (no val split possible here).
    median_epoch = int(np.median(best_epochs)) + 1
    console.print(f"\n[bold]final fit[/bold] on all 94 samples for {median_epoch} epochs")
    final_scaler = StandardScaler().fit(X)
    X_full_s = final_scaler.transform(X).astype(np.float32)
    pos = max(1, int((y == 1).sum()))
    neg = int((y == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)

    final_model = MLP(input_dim=X.shape[1], hidden=hidden_t, dropout=dropout).to(DEVICE)
    optim = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    full_ds = TensorDataset(
        torch.from_numpy(X_full_s), torch.from_numpy(y.astype(np.float32))
    )
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    for _ in range(median_epoch):
        final_model.train()
        for xb, yb in full_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optim.zero_grad()
            loss_fn(final_model(xb), yb).backward()
            optim.step()

    model_path = MODELS_DIR / "mlp.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "scaler_mean": final_scaler.mean_,
            "scaler_scale": final_scaler.scale_,
            "hidden": hidden_t,
            "dropout": dropout,
            "input_dim": int(X.shape[1]),
            "block_sizes": ds.block_sizes,
            "median_epoch": median_epoch,
        },
        model_path,
    )
    console.print(f"[green]saved[/green] {model_path.relative_to(Path.cwd())}")

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
        mlflow.set_experiment("medea-baseline")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model": "mlp",
                    "hidden": ",".join(str(h) for h in hidden_t),
                    "dropout": dropout,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "feature_dim": X.shape[1],
                    "n_samples": X.shape[0],
                    "n_channels": len(set(groups.tolist())),
                    "cv": "LeaveOneGroupOut(channel_id) + 1ch inner val",
                    "median_best_epoch": median_epoch,
                    "device": DEVICE,
                    "seed": SEED,
                }
            )
            mlflow.log_metrics(metrics.as_log_dict(prefix="cv_"))
            mlflow.log_artifact(str(model_path), artifact_path="model")
        console.print(f"[green]mlflow run logged[/green] under {MLRUNS_DIR}")


if __name__ == "__main__":
    app()
