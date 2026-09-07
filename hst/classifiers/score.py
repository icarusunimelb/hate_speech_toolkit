"""Score the record tables with the trained classifiers.

For every platform record table (<work>/records/<platform>_records.csv.gz) and every supervised
category: the unique texts are scored once (by text_hash), and each record receives

    <category>_probability   model probability of the positive class
    <category>_label         1 if probability >= categories.<category>.threshold, else 0

plus ``n_hate_cats`` (number of positive categories) and ``any_hate``.  The table is saved back in
place and <work>/records/<platform>_scoring_summary.json is written.  A category whose columns
already exist is skipped unless --force is given.

Run:  python -m hst.classifiers.score --config project.yaml [--platform x] [--category anti_women]
      [--batch_size 64] [--max_length 256] [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import Config, add_config_argument, config_from_args
from ..schema import ANY_HATE, N_HATE, label_col, load_records, prob_col, save_records
from ._common import positive_probabilities


def load_model(model_dir: Path, base_model: str | None, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(f"model folder {model_dir} not found; train the category first")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        if not base_model:
            raise
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return model, tokenizer


@torch.inference_mode()
def score_texts(texts: list[str], model, tokenizer, device: torch.device, batch_size: int = 64,
                max_length: int = 256) -> np.ndarray:
    out = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length,
                          return_tensors="pt").to(device)
        out.append(positive_probabilities(model(**batch).logits.detach().cpu().numpy()))
    return np.concatenate(out) if out else np.zeros(0, dtype=float)


def score_platform(cfg: Config, platform: str, categories: list[str], batch_size: int = 64,
                   max_length: int | None = None, force: bool = False) -> dict:
    df = load_records(cfg, platform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = cfg.section("training").get("base_model")
    max_length = int(max_length or cfg.section("training").get("max_length", 256))
    unique = df.drop_duplicates("text_hash")[["text_hash", "text"]]
    texts = unique["text"].fillna("").astype(str).tolist()
    summary = {"platform": platform, "rows": int(len(df)), "unique_texts": int(len(unique)), "device": str(device),
               "categories": {}}
    for cat in categories:
        if label_col(cat) in df.columns and not force:
            print(f"{platform}/{cat}: already scored (use --force to redo)")
            summary["categories"][cat] = {"skipped": True}
            continue
        model_dir = cfg.model_dir(cat)
        model, tokenizer = load_model(model_dir, base_model, device)
        probs = score_texts(texts, model, tokenizer, device, batch_size, max_length)
        lookup = pd.Series(probs, index=unique["text_hash"].to_numpy())
        p = df["text_hash"].map(lookup).astype(float)
        thr = cfg.threshold(cat)
        df[prob_col(cat)] = p.round(6)
        df[label_col(cat)] = (p >= thr).astype(int)
        summary["categories"][cat] = {"model": str(model_dir), "threshold": thr,
                                      "positives": int(df[label_col(cat)].sum()),
                                      "positive_rate": round(float(df[label_col(cat)].mean()), 5)}
        print(f"{platform}/{cat}: {summary['categories'][cat]['positives']:,} positive of {len(df):,} at threshold {thr}")
        del model
    present = [label_col(c) for c in cfg.category_names if label_col(c) in df.columns]
    if present:
        df[N_HATE] = df[present].sum(axis=1).astype(int)
        df[ANY_HATE] = (df[N_HATE] > 0).astype(int)
        summary["any_hate"] = int(df[ANY_HATE].sum())
    save_records(cfg, platform, df)
    path = cfg.path("work", "work") / "records" / f"{platform}_scoring_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(cfg: Config, platforms: list[str] | None = None, categories: list[str] | None = None,
        batch_size: int = 64, max_length: int | None = None, force: bool = False) -> dict[str, dict]:
    return {p: score_platform(cfg, p, categories or cfg.category_names, batch_size, max_length, force)
            for p in (platforms or cfg.platforms)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score record tables with the trained classifiers.")
    add_config_argument(parser)
    parser.add_argument("--platform", action="append")
    parser.add_argument("--category", action="append")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(config_from_args(args), args.platform, args.category, args.batch_size, args.max_length, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
