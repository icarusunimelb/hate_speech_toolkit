"""Train a plain single-label RoBERTa classifier (fallback when only one label per item exists).

Use train_multiannotator when per-annotator labels are available.  This trainer uses the hard
label from items.csv only, with inverse-frequency class weights in the cross-entropy loss.

Input:  <work>/training/<category>/items.csv (from prepare_training_data).
Output: <paths.models>/<category>/ (or --output_dir): exported model (config, weights,
        tokenizer), training_summary.json, validation_threshold_metrics.csv, predictions.

Run:  python -m hst.classifiers.train --config project.yaml --category anti_women
      [--base_model ...] [--epochs ...] [--max_length ...] [--output_dir ...] [--no_class_weights]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from ..config import Config, add_config_argument, config_from_args
from . import _common as C


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        item = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length)
        item["labels"] = self.labels[i]
        return item


def train(cfg: Config, category: str, settings: C.TrainSettings | None = None, output_dir: Path | None = None,
          class_weights: bool = True) -> dict:
    s = settings or C.settings_from(cfg)
    out_dir = Path(output_dir) if output_dir else cfg.model_dir(category)
    C.set_seed(s.seed)
    items, _ = C.load_items(cfg, category)
    train_f, val_f, test_f, split_summary = C.make_splits(items, s.test_size, s.validation_size, s.seed)
    train_f = train_f[train_f["label"].isin([0, 1])].reset_index(drop=True)   # hard labels only
    print(f"[{category}] train={len(train_f)} validation={len(val_f)} test={len(test_f)}")

    tokenizer = AutoTokenizer.from_pretrained(s.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    id2label, label2id = C.label_names(category)
    model = AutoModelForSequenceClassification.from_pretrained(s.base_model, num_labels=2, id2label=id2label,
                                                               label2id=label2id, ignore_mismatched_sizes=True)
    weights = None
    if class_weights:
        counts = train_f["label"].value_counts()
        total = float(counts.sum())
        weights = torch.tensor([total / (2 * max(counts.get(i, 1), 1)) for i in (0, 1)], dtype=torch.float)
        print(f"class weights: {weights.tolist()}")

    collate = DataCollatorWithPadding(tokenizer)
    train_ds = TextDataset(train_f, tokenizer, s.max_length)
    val_ds = TextDataset(val_f, tokenizer, s.max_length)
    test_ds = TextDataset(test_f, tokenizer, s.max_length)

    def loss_fn(m, batch):
        batch = dict(batch)
        labels = batch.pop("labels")
        logits = m(**batch).logits
        w = weights.to(logits.device) if weights is not None else None
        return torch.nn.functional.cross_entropy(logits, labels, weight=w)

    history = C.fit(model, train_ds, val_ds, val_f["label"].to_numpy(dtype=int), collate, loss_fn, s)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    summary = C.evaluate_and_summarise(model, collate, val_ds, val_f, test_ds, test_f, s, out_dir, category,
                                       split_summary, {"trainer": "single_label",
                                                       "class_weights": weights.tolist() if weights is not None else None,
                                                       "model_dir": str(out_dir), "training": history})
    C.print_summary(summary)
    print(f"model exported to {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a single-label RoBERTa classifier for one category.")
    add_config_argument(parser)
    C.add_training_arguments(parser)
    parser.add_argument("--no_class_weights", action="store_true")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    train(cfg, args.category, C.settings_from(cfg, args), args.output_dir, not args.no_class_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
