"""M9 error analysis: where do our trained models actually go wrong?

Re-runs the same LeaveOneGroupOut-by-channel protocol used in M6/M7 and
captures out-of-fold predictions from *both* LogReg and MLP on every
sample. Then surfaces:

  * per-channel acc + mean score from each model
  * top false-positives (controls predicted as AI) with titles + scores
  * top false-negatives (offenders predicted as clean) with titles + scores
  * ch61 deep-dive (the persistent control failure flagged at M7)
  * a sketchy OOD signal: each video's mean kNN-distance to the *other*
    9 channels in Chroma — high distance + high disagreement between
    LogReg and MLP is where the MLP extrapolates badly.

OOF predictions are saved to ``data/features/oof_predictions.parquet`` so
later runs can read them without re-training.

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --refresh  # force re-run CV
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

from medea.config import FEATURES_DIR, ensure_dirs
from medea.eval.metrics import binary_metrics
from medea.features.pipeline import build_dataset
from medea.model.baseline import make_logreg_pipeline
from medea.model.mlp import MLP
from medea.storage.vector_db import embedding_slice, get_collection
from medea.storage.db import connect

app = typer.Typer(add_completion=False, help="Error analysis for M9.")
console = Console()

OOF_PARQUET = FEATURES_DIR / "oof_predictions.parquet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def _set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_mlp_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    *,
    hidden=(128, 64),
    dropout=0.4,
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=16,
    max_epochs=200,
    patience=25,
) -> np.ndarray:
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)

    pos = max(1, int((y_train == 1).sum()))
    neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)

    model = MLP(input_dim=X_train.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.from_numpy(Xtr), torch.from_numpy(y_train.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    Xva_t = torch.from_numpy(Xva).to(DEVICE)
    yva_t = torch.from_numpy(y_val.astype(np.float32)).to(DEVICE)

    best_val = float("inf")
    best_state = None
    no_improve = 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xva_t), yva_t).item())
        if vl < best_val - 1e-4:
            best_val = vl
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(Xte).to(DEVICE))).cpu().numpy()


def _pick_val_channel(train_channel_ids: np.ndarray, fold_idx: int) -> int:
    sorted_ch = sorted(set(train_channel_ids.tolist()))
    return int(sorted_ch[fold_idx % len(sorted_ch)])


def _run_cv(ds) -> pd.DataFrame:
    X = ds.X.astype(np.float32)
    y = ds.labels.astype(int)
    groups = ds.channel_ids
    logo = LeaveOneGroupOut()

    logreg_oof = np.full(len(y), np.nan)
    mlp_oof = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        held_out = int(groups[test_idx][0])

        # LogReg fold
        pipe = make_logreg_pipeline()
        pipe.fit(X[train_idx], y[train_idx])
        logreg_oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

        # MLP fold (with inner channel-level val for early stopping)
        train_groups = groups[train_idx]
        val_ch = _pick_val_channel(train_groups, fold)
        val_mask = train_groups == val_ch
        inner_train = train_idx[~val_mask]
        inner_val = train_idx[val_mask]
        proba = _train_mlp_fold(
            X[inner_train], y[inner_train],
            X[inner_val], y[inner_val],
            X[test_idx],
        )
        mlp_oof[test_idx] = proba
        console.print(
            f"  fold {fold+1}/10  test=ch{held_out}  "
            f"LR={float(((logreg_oof[test_idx] >= 0.5).astype(int) == y[test_idx]).mean()):.2f}  "
            f"MLP={float(((mlp_oof[test_idx] >= 0.5).astype(int) == y[test_idx]).mean()):.2f}"
        )

    return pd.DataFrame(
        {
            "video_id": ds.video_ids,
            "channel_id": ds.channel_ids.astype(int),
            "label": y,
            "logreg_oof": logreg_oof,
            "mlp_oof": mlp_oof,
        }
    )


def _load_or_run(ds, refresh: bool) -> pd.DataFrame:
    if OOF_PARQUET.exists() and not refresh:
        df = pd.read_parquet(OOF_PARQUET)
        # schema sanity check
        if set(df.columns) >= {"video_id", "channel_id", "label", "logreg_oof", "mlp_oof"} \
                and len(df) == len(ds.video_ids):
            console.print(f"[dim]loaded cached OOF predictions from {OOF_PARQUET}[/dim]")
            return df
        console.print("[yellow]cached OOF stale, rerunning CV[/yellow]")
    console.print("[bold]running channel-grouped LOGO CV (LogReg + MLP)[/bold]")
    _set_seed()
    df = _run_cv(ds)
    df.to_parquet(OOF_PARQUET, index=False)
    console.print(f"[green]saved[/green] {OOF_PARQUET}")
    return df


def _fetch_titles(video_ids: list[str]) -> dict[str, dict]:
    if not video_ids:
        return {}
    placeholders = ",".join("?" for _ in video_ids)
    with connect() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                f"""
                SELECT id, title, description, view_count, upload_date,
                       channel_id, label
                FROM videos WHERE id IN ({placeholders})
                """,
                video_ids,
            )
        ]
    return {r["id"]: r for r in rows}


def _per_channel_table(df: pd.DataFrame) -> Table:
    table = Table(title="Per-channel OOF accuracy (threshold=0.5)")
    table.add_column("ch", justify="right")
    table.add_column("label", justify="center")
    table.add_column("n", justify="right")
    table.add_column("LR acc", justify="right")
    table.add_column("MLP acc", justify="right")
    table.add_column("LR mean p", justify="right")
    table.add_column("MLP mean p", justify="right")
    for cid, g in df.groupby("channel_id"):
        lbl = int(g["label"].iloc[0])
        lr_pred = (g["logreg_oof"] >= 0.5).astype(int)
        mlp_pred = (g["mlp_oof"] >= 0.5).astype(int)
        lr_acc = float((lr_pred == g["label"]).mean())
        mlp_acc = float((mlp_pred == g["label"]).mean())
        cl = lambda v: "green" if v >= 0.8 else ("yellow" if v >= 0.5 else "red")
        table.add_row(
            f"{cid}",
            "off" if lbl else "ctrl",
            f"{len(g)}",
            f"[{cl(lr_acc)}]{lr_acc:.2f}[/]",
            f"[{cl(mlp_acc)}]{mlp_acc:.2f}[/]",
            f"{g['logreg_oof'].mean():.2f}",
            f"{g['mlp_oof'].mean():.2f}",
        )
    return table


def _error_table(df: pd.DataFrame, *, kind: str, model_col: str) -> Table:
    """kind = 'fp' (label=0, score>=0.5) or 'fn' (label=1, score<0.5)."""
    if kind == "fp":
        mask = (df["label"] == 0) & (df[model_col] >= 0.5)
        sort_by = model_col
        ascending = False
        title_kind = "False Positives (control → AI)"
    else:
        mask = (df["label"] == 1) & (df[model_col] < 0.5)
        sort_by = model_col
        ascending = True
        title_kind = "False Negatives (AI → control)"
    sub = df[mask].sort_values(sort_by, ascending=ascending).copy()
    titles = _fetch_titles(sub["video_id"].tolist())
    table = Table(title=f"{model_col} {title_kind} (n={len(sub)})", show_lines=False)
    table.add_column("video_id")
    table.add_column("ch", justify="right")
    table.add_column("score", justify="right")
    table.add_column("title")
    for _, row in sub.iterrows():
        meta = titles.get(row["video_id"], {})
        title = (meta.get("title") or "")[:75]
        table.add_row(
            row["video_id"], f"{int(row['channel_id'])}",
            f"{row[model_col]:.2f}",
            title,
        )
    return table


def _ch61_deep_dive(df: pd.DataFrame, ds) -> None:
    sub = df[df["channel_id"] == 61].sort_values("mlp_oof", ascending=False)
    titles = _fetch_titles(sub["video_id"].tolist())

    console.rule("[bold red]ch61 deep-dive (persistent control failure)")
    table = Table()
    table.add_column("video_id")
    table.add_column("LR p", justify="right")
    table.add_column("MLP p", justify="right")
    table.add_column("title")
    table.add_column("desc head", style="dim")
    for _, row in sub.iterrows():
        m = titles.get(row["video_id"], {})
        desc = (m.get("description") or "").replace("\n", " ")[:60]
        title = (m.get("title") or "")[:55]
        table.add_row(
            row["video_id"],
            f"{row['logreg_oof']:.2f}",
            f"{row['mlp_oof']:.2f}",
            title,
            desc,
        )
    console.print(table)

    # Compare ch61 audio + metadata against the rest of the controls.
    audio = pd.read_parquet(FEATURES_DIR / "audio.parquet")
    meta = pd.read_parquet(FEATURES_DIR / "metadata.parquet")
    combined = (
        audio.merge(meta, on="video_id")
        .merge(
            pd.DataFrame({
                "video_id": ds.video_ids,
                "channel_id": ds.channel_ids,
                "label": ds.labels,
            }),
            on="video_id",
        )
    )
    cols = ["ai_voice_prob", "title_len", "title_caps_ratio", "desc_url_count",
            "view_count_log", "title_word_count"]
    ch61 = combined[combined["channel_id"] == 61][cols].agg(["mean", "median"])
    other_ctrl = combined[(combined["label"] == 0) & (combined["channel_id"] != 61)][cols].agg(["mean", "median"])
    offenders = combined[combined["label"] == 1][cols].agg(["mean", "median"])

    cmp = pd.concat(
        {"ch61": ch61.loc["median"], "other_ctrl": other_ctrl.loc["median"], "offenders": offenders.loc["median"]},
        axis=1,
    )
    console.print("\n[bold]ch61 vs other controls vs offenders (medians)[/bold]")
    console.print(cmp.to_string())


def _ood_distance(df: pd.DataFrame, ds) -> Table:
    """Mean cosine distance to the 5 nearest cross-channel neighbors in Chroma —
    a rough proxy for "how far is this clip from any other channel we've seen?"."""
    col = get_collection()
    if col.count() == 0:
        console.print("[yellow]Chroma is empty; skipping OOD analysis[/yellow]")
        return Table()
    emb = embedding_slice(ds.X)
    n_results = 11  # k+1 + own-channel buffer
    dists: list[float] = []
    for i, vid in enumerate(ds.video_ids):
        own_ch = int(ds.channel_ids[i])
        res = col.query(query_embeddings=[emb[i].tolist()],
                        n_results=n_results,
                        include=["metadatas", "distances"])
        these_dists = []
        for nbr_meta, d in zip(res["metadatas"][0], res["distances"][0]):
            if int(nbr_meta["channel_id"]) == own_ch:
                continue
            these_dists.append(float(d))
            if len(these_dists) >= 5:
                break
        dists.append(float(np.mean(these_dists)) if these_dists else float("nan"))
    df = df.copy()
    df["mean_offch_dist"] = dists
    df["abs_disagreement"] = (df["mlp_oof"] - df["logreg_oof"]).abs()
    sub = df.sort_values("abs_disagreement", ascending=False).head(10)
    titles = _fetch_titles(sub["video_id"].tolist())
    table = Table(title="Top MLP-vs-LogReg disagreements (proxy OOD signal)")
    table.add_column("vid")
    table.add_column("ch", justify="right")
    table.add_column("label", justify="center")
    table.add_column("LR p", justify="right")
    table.add_column("MLP p", justify="right")
    table.add_column("|Δ|", justify="right")
    table.add_column("dist↑", justify="right")
    table.add_column("title")
    for _, row in sub.iterrows():
        m = titles.get(row["video_id"], {})
        title = (m.get("title") or "")[:55]
        table.add_row(
            row["video_id"], f"{int(row['channel_id'])}",
            "off" if int(row["label"]) else "ctrl",
            f"{row['logreg_oof']:.2f}",
            f"{row['mlp_oof']:.2f}",
            f"{row['abs_disagreement']:.2f}",
            f"{row['mean_offch_dist']:.2f}",
            title,
        )
    return table


@app.command()
def main(
    refresh: bool = typer.Option(False, help="Re-run CV from scratch."),
    log_level: str = typer.Option("WARNING"),
) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=False, show_path=False)],
    )
    ensure_dirs()
    ds = build_dataset()

    df = _load_or_run(ds, refresh=refresh)

    lr_metrics = binary_metrics(
        df["label"].to_numpy(),
        (df["logreg_oof"] >= 0.5).astype(int).to_numpy(),
        df["logreg_oof"].to_numpy(),
    )
    mlp_metrics = binary_metrics(
        df["label"].to_numpy(),
        (df["mlp_oof"] >= 0.5).astype(int).to_numpy(),
        df["mlp_oof"].to_numpy(),
    )
    console.rule("[bold]Overall OOF (channel-grouped)")
    console.print(
        f"LogReg  acc={lr_metrics.accuracy:.3f}  prec={lr_metrics.precision:.3f}  "
        f"rec={lr_metrics.recall:.3f}  f1={lr_metrics.f1:.3f}  auc={lr_metrics.roc_auc:.3f}"
    )
    console.print(
        f"MLP     acc={mlp_metrics.accuracy:.3f}  prec={mlp_metrics.precision:.3f}  "
        f"rec={mlp_metrics.recall:.3f}  f1={mlp_metrics.f1:.3f}  auc={mlp_metrics.roc_auc:.3f}"
    )

    console.print()
    console.print(_per_channel_table(df))

    console.print()
    console.print(_error_table(df, kind="fp", model_col="logreg_oof"))
    console.print(_error_table(df, kind="fn", model_col="logreg_oof"))
    console.print(_error_table(df, kind="fp", model_col="mlp_oof"))
    console.print(_error_table(df, kind="fn", model_col="mlp_oof"))

    _ch61_deep_dive(df, ds)

    console.print()
    console.print(_ood_distance(df, ds))


if __name__ == "__main__":
    app()
