"""Train a RoBERTa classifier with a multi-annotator head (the recommended trainer).

Why: majority labels hide how much annotators disagree.  This model keeps one *consensus* head
(what the corpus will be scored with) and one small sigmoid head per annotator that learns each
annotator's own decisions.  Items on which annotators tied get no consensus label and only
supervise the annotator heads.  The annotator heads act as a regulariser that stops the encoder
from over-fitting one annotator's habits; at export time they are dropped and the consensus
model is saved as an ordinary sequence-classification model.

Input:  <work>/training/<category>/items.csv and annotator_votes.csv (from prepare_training_data).
Output: <paths.models>/<category>/ (or --output_dir) containing the exported consensus model
        (config, weights, tokenizer), training_summary.json (settings, split sizes, epoch history,
        test metrics at 0.5 and at the recommended threshold), validation_threshold_metrics.csv
        and the validation / test predictions.

Loss = weighted cross-entropy on the consensus head (items with a hard majority label; weight
grows with agreement and number of annotators) + annotator_loss_weight x masked BCE over the
annotator heads.

Run:  python -m hst.classifiers.train_multiannotator --config project.yaml --category anti_women
      [--base_model ...] [--epochs ...] [--max_length ...] [--output_dir ...]
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, RobertaForSequenceClassification, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from ..config import Config, add_config_argument, config_from_args
from . import _common as C


@dataclass
class MultiAnnotatorOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    annotator_logits: torch.FloatTensor | None = None


class MultiAnnotatorRoberta(RobertaPreTrainedModel):
    """RoBERTa encoder + consensus classification head + one logit per annotator.

    ``config.num_annotators`` and ``config.annotator_loss_weight`` must be set on the config.
    The combined loss is computed in forward() when the label tensors are given.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_annotators = int(getattr(config, "num_annotators", 1))
        self.annotator_loss_weight = float(getattr(config, "annotator_loss_weight", 0.5))
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.annotator_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.annotator_dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.annotator_out_proj = torch.nn.Linear(config.hidden_size, self.num_annotators)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, consensus_mask=None,
                consensus_weight=None, annotator_labels=None, annotator_mask=None, **kwargs):
        hidden = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        logits = self.classifier(hidden)
        rep = self.annotator_dropout(hidden[:, 0, :])
        rep = self.annotator_dropout(torch.tanh(self.annotator_dense(rep)))
        annotator_logits = self.annotator_out_proj(rep)
        loss = None
        if labels is not None:
            per_item = F.cross_entropy(logits, labels, reduction="none")
            w = (consensus_mask if consensus_mask is not None else torch.ones_like(per_item)) * \
                (consensus_weight if consensus_weight is not None else torch.ones_like(per_item))
            consensus_loss = (per_item * w).sum() / w.sum().clamp_min(1e-8) if w.sum() > 0 else logits.sum() * 0.0
            if annotator_labels is not None and annotator_mask is not None:
                bce = F.binary_cross_entropy_with_logits(annotator_logits, annotator_labels, reduction="none")
                per_ann = annotator_mask.sum(dim=1)
                item_loss = (bce * annotator_mask).sum(dim=1) / per_ann.clamp_min(1.0)
                has = per_ann.gt(0)
                annotator_loss = item_loss[has].mean() if has.any() else annotator_logits.sum() * 0.0
            else:
                annotator_loss = annotator_logits.sum() * 0.0
            loss = consensus_loss + self.annotator_loss_weight * annotator_loss
        return MultiAnnotatorOutput(loss=loss, logits=logits, annotator_logits=annotator_logits)


class MultiAnnotatorDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, annotator_cols: list[str], tokenizer, max_length: int):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].fillna(0).astype(int).to_numpy()
        self.masks = frame["consensus_mask"].astype(float).to_numpy()
        self.weights = frame["consensus_weight"].astype(float).to_numpy()
        values = frame[annotator_cols].to_numpy(dtype=float) if annotator_cols else np.full((len(frame), 1), np.nan)
        self.ann_mask = (~np.isnan(values)).astype(np.float32)
        self.ann_labels = np.nan_to_num(values, nan=0.0).astype(np.float32)
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        item = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length)
        item["labels"] = int(self.labels[i])
        item["consensus_mask"] = float(self.masks[i])
        item["consensus_weight"] = float(self.weights[i])
        item["annotator_labels"] = self.ann_labels[i]
        item["annotator_mask"] = self.ann_mask[i]
        return item


class MultiAnnotatorCollator:
    EXTRA = ["labels", "consensus_mask", "consensus_weight", "annotator_labels", "annotator_mask"]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        features = [dict(f) for f in features]
        extra = {k: [f.pop(k) for f in features] for k in self.EXTRA}
        batch = dict(self.tokenizer.pad(features, padding=True, return_tensors="pt"))
        batch["labels"] = torch.tensor(extra["labels"], dtype=torch.long)
        batch["consensus_mask"] = torch.tensor(extra["consensus_mask"], dtype=torch.float)
        batch["consensus_weight"] = torch.tensor(extra["consensus_weight"], dtype=torch.float)
        batch["annotator_labels"] = torch.tensor(np.asarray(extra["annotator_labels"]), dtype=torch.float)
        batch["annotator_mask"] = torch.tensor(np.asarray(extra["annotator_mask"]), dtype=torch.float)
        return batch


def build_item_table(items: pd.DataFrame, votes: pd.DataFrame, confidence_floor: float = 0.2,
                     full_depth: float = 3.0) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Attach one column per annotator (0/1/NaN) and the consensus mask / weight to the items."""
    annotators = sorted(votes["annotator"].dropna().unique().tolist())
    cols = [f"annotator_{i}" for i in range(len(annotators))]
    if annotators:
        wide = votes.pivot_table(index="item_key", columns="annotator", values="label", aggfunc="mean")
        wide = wide.reindex(columns=annotators).rename(columns=dict(zip(annotators, cols))).reset_index()
        out = items.merge(wide, on="item_key", how="left")
    else:
        out = items.copy()
    n = out["n_annotators"].astype(float).clip(lower=1)
    p = out["positive_votes"].astype(float) / n
    agreement = (2.0 * p - 1.0).abs()
    depth = (n / full_depth).clip(upper=1.0)
    out["consensus_weight"] = confidence_floor + (1.0 - confidence_floor) * agreement * depth
    out["consensus_mask"] = out["label"].isin([0, 1]).astype(float)
    out.loc[out["consensus_mask"] == 0, "consensus_weight"] = 0.0
    return out, cols, annotators


def export_consensus_model(model: MultiAnnotatorRoberta, tokenizer, out_dir: Path, category: str) -> None:
    config = copy.deepcopy(model.config)
    config.num_labels = 2
    config.id2label, config.label2id = C.label_names(category)
    config.architectures = ["RobertaForSequenceClassification"]
    for attr in ("num_annotators", "annotator_loss_weight"):
        if hasattr(config, attr):
            delattr(config, attr)
    std = RobertaForSequenceClassification(config)
    std.roberta.load_state_dict(model.roberta.state_dict(), strict=False)
    std.classifier.load_state_dict(model.classifier.state_dict())
    out_dir.mkdir(parents=True, exist_ok=True)
    std.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def train(cfg: Config, category: str, settings: C.TrainSettings | None = None, output_dir: Path | None = None,
          annotator_loss_weight: float = 0.5) -> dict:
    s = settings or C.settings_from(cfg)
    out_dir = Path(output_dir) if output_dir else cfg.model_dir(category)
    C.set_seed(s.seed)
    items, votes = C.load_items(cfg, category)
    items, ann_cols, annotators = build_item_table(items, votes)
    train_f, val_f, test_f, split_summary = C.make_splits(items, s.test_size, s.validation_size, s.seed)
    print(f"[{category}] train={len(train_f)} (unlabelled {int(train_f['label'].isna().sum())}) "
          f"validation={len(val_f)} test={len(test_f)} annotators={len(annotators)}")

    tokenizer = AutoTokenizer.from_pretrained(s.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    id2label, label2id = C.label_names(category)
    config = AutoConfig.from_pretrained(s.base_model, num_labels=2, id2label=id2label, label2id=label2id)
    config.num_annotators = max(1, len(ann_cols))
    config.annotator_loss_weight = float(annotator_loss_weight)
    model = MultiAnnotatorRoberta.from_pretrained(s.base_model, config=config, ignore_mismatched_sizes=True)

    collate = MultiAnnotatorCollator(tokenizer)
    train_ds = MultiAnnotatorDataset(train_f, ann_cols, tokenizer, s.max_length)
    val_ds = MultiAnnotatorDataset(val_f, ann_cols, tokenizer, s.max_length)
    test_ds = MultiAnnotatorDataset(test_f, ann_cols, tokenizer, s.max_length)

    def loss_fn(m, batch):
        return m(**batch).loss

    history = C.fit(model, train_ds, val_ds, val_f["label"].to_numpy(dtype=int), collate, loss_fn, s)
    export_consensus_model(model, tokenizer, out_dir, category)
    summary = C.evaluate_and_summarise(model, collate, val_ds, val_f, test_ds, test_f, s, out_dir, category,
                                       split_summary, {"trainer": "multiannotator", "annotators": annotators,
                                                       "annotator_loss_weight": annotator_loss_weight,
                                                       "model_dir": str(out_dir), "training": history})
    C.print_summary(summary)
    print(f"model exported to {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a multi-annotator RoBERTa classifier for one category.")
    add_config_argument(parser)
    C.add_training_arguments(parser)
    parser.add_argument("--annotator_loss_weight", type=float, default=0.5)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    train(cfg, args.category, C.settings_from(cfg, args), args.output_dir, args.annotator_loss_weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
