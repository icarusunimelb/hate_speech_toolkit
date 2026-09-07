"""Shared pieces for the two trainers: settings, data loading, splits, a small training loop,
metrics, threshold sweep and the training summary.  Not a command-line module.

The training loop is plain PyTorch (AdamW + linear warmup/decay, validation after every epoch,
early stopping on validation F1, best epoch restored) so the toolkit needs only torch and
transformers, and what happens during training is visible in one place.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ..config import Config


# ---- settings ----------------------------------------------------------------------------------
@dataclass
class TrainSettings:
    base_model: str = "FacebookAI/roberta-large"
    max_length: int = 256
    epochs: float = 5.0
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    test_size: float = 0.15
    validation_size: float = 0.15
    seed: int = 42
    early_stopping_patience: int = 2
    threshold_step: float = 0.01


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--category", required=True, help="Category name from the config.")
    parser.add_argument("--output_dir", default=None, help="Model folder (default: <paths.models>/<category>).")
    for name, kind in [("base_model", str), ("max_length", int), ("epochs", float), ("batch_size", int),
                       ("learning_rate", float), ("test_size", float), ("validation_size", float),
                       ("seed", int), ("early_stopping_patience", int)]:
        parser.add_argument(f"--{name}", type=kind, default=None, help=f"Overrides training.{name} in the config.")


def settings_from(cfg: Config, args: argparse.Namespace | None = None) -> TrainSettings:
    s = TrainSettings()
    for k, v in cfg.section("training").items():
        if hasattr(s, k):
            setattr(s, k, type(getattr(s, k))(v))
    if args is not None:
        for k in vars(s):
            v = getattr(args, k, None)
            if v is not None:
                setattr(s, k, v)
    return s


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- data ----------------------------------------------------------------------------------------
def normalize_for_split(value: object) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip().lower()
    text = re.sub(r"^rt\s+@\w+:?\s*", "", text)
    text = re.sub(r"https?://\S+", "<url>", text)
    return re.sub(r"\s+", " ", text).strip()


def load_items(cfg: Config, category: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """items.csv and annotator_votes.csv from prepare_training_data."""
    folder = cfg.path("work", "work") / "training" / category
    items_path, votes_path = folder / "items.csv", folder / "annotator_votes.csv"
    if not items_path.exists():
        raise FileNotFoundError(f"{items_path} not found; run `python -m hst.annotation.prepare_training_data` first")
    items = pd.read_csv(items_path, dtype={"item_key": str, "text": str})
    items["text"] = items["text"].fillna("").astype(str)
    items = items[items["text"].str.strip().ne("")].copy()
    items["label"] = pd.to_numeric(items["label"], errors="coerce")
    items["norm_text"] = items["text"].map(normalize_for_split)
    if votes_path.exists():
        votes = pd.read_csv(votes_path, dtype={"item_key": str, "annotator": str})
    else:
        votes = pd.DataFrame(columns=["item_key", "annotator", "label"])
    votes["label"] = pd.to_numeric(votes["label"], errors="coerce")
    return items.reset_index(drop=True), votes


def make_splits(items: pd.DataFrame, test_size: float, validation_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Grouped (by normalised text) stratified split.  Items without a hard label train only."""
    hard = items[items["label"].isin([0, 1])]
    groups = hard.drop_duplicates("norm_text")[["norm_text", "label"]].reset_index(drop=True)
    if groups["label"].nunique() < 2:
        raise ValueError("training data must contain both classes")

    def _split(frame: pd.DataFrame, size: float, rs: int):
        strat = frame["label"] if frame["label"].value_counts().min() >= 2 else None
        return train_test_split(frame, test_size=size, random_state=rs, stratify=strat)

    rest, test_g = _split(groups, test_size, seed)
    val_rel = validation_size / max(1.0 - test_size, 1e-9)
    train_g, val_g = _split(rest, val_rel, seed + 1)
    test_norm, val_norm = set(test_g["norm_text"]), set(val_g["norm_text"])
    test = hard[hard["norm_text"].isin(test_norm)].copy()
    val = hard[hard["norm_text"].isin(val_norm)].copy()
    train = items[~items["norm_text"].isin(test_norm | val_norm)].copy()
    summary = {
        "train_items": int(len(train)), "train_hard_items": int(train["label"].isin([0, 1]).sum()),
        "train_unlabelled_items": int(train["label"].isna().sum()),
        "validation_items": int(len(val)), "validation_positive": int((val["label"] == 1).sum()),
        "test_items": int(len(test)), "test_positive": int((test["label"] == 1).sum()),
    }
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), summary


# ---- metrics -------------------------------------------------------------------------------------
def positive_probabilities(logits) -> np.ndarray:
    t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    if t.ndim == 1 or t.shape[-1] == 1:
        return torch.sigmoid(t.reshape(-1)).numpy()
    return torch.softmax(t, dim=-1).numpy()[:, 1]


def metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels).astype(int)
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out = {"threshold": float(threshold), "accuracy": float(accuracy_score(labels, preds)),
           "balanced_accuracy": float(balanced_accuracy_score(labels, preds)), "precision": float(p),
           "recall": float(r), "f1": float(f1), "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(labels))}
    if len(set(labels.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["pr_auc"] = float(average_precision_score(labels, probs))
        out["brier_score"] = float(brier_score_loss(labels, probs))
    return out


def threshold_table(labels: np.ndarray, probs: np.ndarray, step: float = 0.01) -> pd.DataFrame:
    rows = [metrics_at_threshold(labels, probs, float(round(t, 4))) for t in np.arange(step, 1.0, step)]
    return pd.DataFrame(rows)[["threshold", "accuracy", "balanced_accuracy", "precision", "recall", "f1",
                               "specificity", "tn", "fp", "fn", "tp"]]


def choose_threshold(table: pd.DataFrame) -> float:
    """F1-optimal threshold (ties broken by precision, then recall); 0.5 when nothing is positive."""
    if table.empty or table["f1"].max() <= 0:
        return 0.5
    best = table.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    return float(best["threshold"])


def label_names(category: str) -> tuple[dict, dict]:
    id2label = {0: f"not_{category}", 1: category}
    return id2label, {v: k for k, v in id2label.items()}


# ---- training loop --------------------------------------------------------------------------------
def _to_device(batch: dict, dev: torch.device) -> dict:
    return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.inference_mode()
def predict(model: torch.nn.Module, dataset: Dataset, collate: Callable, batch_size: int = 32) -> np.ndarray:
    """Positive-class probabilities for a dataset (uses the model's ``logits`` output)."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate):
        batch = _to_device(batch, dev)
        batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out.append(positive_probabilities(model(**batch).logits.detach().cpu().numpy()))
    return np.concatenate(out) if out else np.zeros(0, dtype=float)


def fit(model: torch.nn.Module, train_ds: Dataset, val_ds: Dataset, val_labels: np.ndarray, collate: Callable,
        loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor], s: TrainSettings) -> dict:
    """AdamW + linear warmup/decay; validate every epoch; early stop on validation F1 (at 0.5),
    with PR-AUC as tie-breaker; restore the best epoch.  Returns the epoch history."""
    dev = device()
    model.to(dev)
    loader = DataLoader(train_ds, batch_size=s.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(s.seed))
    n_epochs = max(1, int(np.ceil(s.epochs)))
    total_steps = max(1, len(loader) * n_epochs)
    warmup = int(s.warmup_ratio * total_steps)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": s.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=s.learning_rate)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))

    best_state, best_score, best_epoch, bad, history = None, (-1.0, -1.0), 0, 0, []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = _to_device(batch, dev)
            loss = loss_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.item()) * len(next(iter(batch.values())))
            n += len(next(iter(batch.values())))
        probs = predict(model, val_ds, collate, s.eval_batch_size)
        m = metrics_at_threshold(val_labels, probs, 0.5)
        score = (m["f1"], m.get("pr_auc", 0.0))
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_f1": m["f1"],
                        "val_precision": m["precision"], "val_recall": m["recall"], "val_pr_auc": m.get("pr_auc")})
        print(f"  epoch {epoch}: loss={total / max(n, 1):.4f} val_f1={m['f1']:.3f} val_pr_auc={m.get('pr_auc', float('nan')):.3f}")
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= s.early_stopping_patience:
                print(f"  early stopping (no improvement for {bad} epochs)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "epochs_run": len(history), "history": history}


# ---- evaluation + summary -------------------------------------------------------------------------
def evaluate_and_summarise(model: torch.nn.Module, collate: Callable, val_ds, val_frame: pd.DataFrame, test_ds,
                           test_frame: pd.DataFrame, s: TrainSettings, out_dir: Path, category: str,
                           split_summary: dict, extra: dict) -> dict:
    """Validation threshold sweep -> recommended threshold; test metrics; write CSV/JSON outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    val_probs = predict(model, val_ds, collate, s.eval_batch_size)
    val_labels = val_frame["label"].to_numpy(dtype=int)
    table = threshold_table(val_labels, val_probs, s.threshold_step)
    rec = choose_threshold(table)
    table.to_csv(out_dir / "validation_threshold_metrics.csv", index=False)

    test_probs = predict(model, test_ds, collate, s.eval_batch_size)
    test_labels = test_frame["label"].to_numpy(dtype=int)
    for name, frame, probs in (("validation", val_frame, val_probs), ("test", test_frame, test_probs)):
        pred = frame[["item_key", "text", "label"]].copy()
        pred["probability"] = probs
        pred["pred_label_recommended"] = (probs >= rec).astype(int)
        pred.to_csv(out_dir / f"{name}_predictions.csv", index=False)

    summary = {
        "category": category, "settings": asdict(s), "split": split_summary,
        "recommended_threshold": rec,
        "validation_metrics_at_recommended": metrics_at_threshold(val_labels, val_probs, rec),
        "test_metrics_at_0.5": metrics_at_threshold(test_labels, test_probs, 0.5),
        "test_metrics_at_recommended": metrics_at_threshold(test_labels, test_probs, rec),
        "threshold_table": str(out_dir / "validation_threshold_metrics.csv"),
        "device": str(device()),
        **extra,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def print_summary(summary: dict) -> None:
    t = summary["test_metrics_at_recommended"]
    print(f"recommended_threshold={summary['recommended_threshold']:.2f}  test precision={t['precision']:.3f} "
          f"recall={t['recall']:.3f} f1={t['f1']:.3f} (n={t['n']})")
