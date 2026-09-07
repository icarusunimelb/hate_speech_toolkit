"""Validate the LLM labels against human reference labels, or draw a sample for human labelling.

Mode 1 - evaluate
    python -m hst.llm.validation --config project.yaml --reference reference.csv [--predictions llm_labels.csv]
                                 [--name round1]
    reference.csv: text_hash plus one 0/1 column per LLM category (and optionally frame_verdict).
    Writes <work>/llm/validation/validation_<name>.json and .md with, per category, precision /
    recall / F1 / accuracy / support and the confusion counts, the frame-verdict accuracy, and
    micro / macro F1 over the categories.

Mode 2 - sample
    python -m hst.llm.validation --config project.yaml --sample 300 [--seed 0] [--out sample.csv]
    Draws unique texts from the record tables, evenly across platforms; when the records already
    carry supervised hate labels, half of each platform's quota is taken from any_hate == 1 texts
    so the sample is enriched for likely positives.  Writes a CSV (and .xlsx when openpyxl is
    available) with text_hash, platform, text, empty frame_verdict and one empty column per
    category for the human coder.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, add_config_argument, config_from_args
from ..schema import ANY_HATE, label_col, load_records
from ._common import labels_path, llm_dir, load_schema, verdict_values, write_json


# ---- metrics ----------------------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = tp + fp + fn + tn
    return {"support": tp + fn, "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "accuracy": round((tp + tn) / n, 4) if n else 0.0}


def _verdict_is_positive(values: pd.Series) -> pd.Series:
    """A verdict string counts as positive unless it starts with 'not' (works for
    'supremacist_outcome' / 'supremacist' vs 'not_supremacist')."""
    s = values.astype(str).str.strip().str.lower()
    return (~s.str.startswith("not")) & s.ne("nan") & s.ne("none") & s.ne("")


def evaluate(cfg: Config, reference_csv: Path, predictions_csv: Path | None = None, name: str = "reference") -> dict[str, Any]:
    categories = cfg.llm_categories
    pred = pd.read_csv(predictions_csv or labels_path(cfg), dtype={"text_hash": str})
    ref = pd.read_csv(reference_csv, dtype={"text_hash": str})
    for frame in (pred, ref):
        frame["text_hash"] = frame["text_hash"].astype(str).str.lower()
    ref_cols = {c: (c if c in ref.columns else label_col(c)) for c in categories}
    missing = [c for c, col in ref_cols.items() if col not in ref.columns]
    if missing:
        raise ValueError(f"reference file lacks label columns for {missing}")
    if "labels_available" in pred.columns:
        pred = pred[pred["labels_available"].astype(str).str.lower().eq("true") | pred["labels_available"].eq(True)]
    joined = ref.merge(pred, on="text_hash", how="inner", suffixes=("_ref", "_pred"))
    result: dict[str, Any] = {"name": name, "reference": str(reference_csv), "reference_texts": int(len(ref)),
                              "matched_texts": int(len(joined)), "unmatched_reference_texts": int(len(ref) - len(joined)),
                              "categories": {}}
    tp = fp = fn = 0
    f1s = []
    for cat in categories:
        ref_col = ref_cols[cat] + ("_ref" if ref_cols[cat] == label_col(cat) else "")
        pred_col = label_col(cat) + ("_pred" if ref_cols[cat] == label_col(cat) else "")
        y_true = pd.to_numeric(joined[ref_col], errors="coerce").fillna(0).astype(int).to_numpy()
        y_pred = pd.to_numeric(joined[pred_col], errors="coerce").fillna(0).astype(int).to_numpy()
        m = binary_metrics(y_true, y_pred)
        result["categories"][cat] = m
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        f1s.append(m["f1"])
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    result["micro_f1"] = round(2 * micro_p * micro_r / (micro_p + micro_r), 4) if micro_p + micro_r else 0.0
    result["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
    if "frame_verdict_ref" in joined.columns and "frame_verdict_pred" in joined.columns:
        t = _verdict_is_positive(joined["frame_verdict_ref"]).to_numpy().astype(int)
        p = _verdict_is_positive(joined["frame_verdict_pred"]).to_numpy().astype(int)
        result["frame_verdict"] = binary_metrics(t, p)
    elif "frame_verdict" in ref.columns and "frame_verdict" in pred.columns:
        pass

    out_dir = llm_dir(cfg) / "validation"
    write_json(out_dir / f"validation_{name}.json", result)
    lines = [f"# LLM validation: {name}", "", f"Reference texts: {result['reference_texts']}, matched with predictions: {result['matched_texts']}", "",
             "| category | support | precision | recall | F1 | accuracy | TP | FP | FN |", "|---|---|---|---|---|---|---|---|---|"]
    for cat, m in result["categories"].items():
        lines.append(f"| {cfg.category_label(cat)} | {m['support']} | {m['precision']} | {m['recall']} | {m['f1']} | {m['accuracy']} | {m['tp']} | {m['fp']} | {m['fn']} |")
    lines += ["", f"Micro F1: {result['micro_f1']}   Macro F1: {result['macro_f1']}"]
    if "frame_verdict" in result:
        fv = result["frame_verdict"]
        lines.append(f"Frame verdict accuracy: {fv['accuracy']} (precision {fv['precision']}, recall {fv['recall']})")
    (out_dir / f"validation_{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return result


# ---- sampling ----------------------------------------------------------------------------------------
def draw_sample(cfg: Config, n: int, seed: int = 0, out: Path | None = None) -> pd.DataFrame:
    rng = random.Random(seed)
    platforms = cfg.platforms
    quota = {p: n // len(platforms) + (1 if i < n % len(platforms) else 0) for i, p in enumerate(platforms)}
    chosen = []
    seen: set[str] = set()
    for platform in platforms:
        df = load_records(cfg, platform)
        df = df[df["in_study_period"] & df["text"].fillna("").astype(str).str.strip().ne("")]
        df = df.drop_duplicates("text_hash")
        df = df[~df["text_hash"].isin(seen)]
        picks: list[pd.Series] = []
        if ANY_HATE in df.columns:
            pos = df[pd.to_numeric(df[ANY_HATE], errors="coerce").fillna(0) == 1]
            k = min(len(pos), quota[platform] // 2)
            picks += [pos.iloc[i] for i in sorted(rng.sample(range(len(pos)), k))]
        taken = {r["text_hash"] for r in picks}
        rest = df[~df["text_hash"].isin(taken)]
        k = min(len(rest), quota[platform] - len(picks))
        picks += [rest.iloc[i] for i in sorted(rng.sample(range(len(rest)), k))]
        for r in picks:
            seen.add(r["text_hash"])
            chosen.append({"text_hash": r["text_hash"], "platform": platform, "text": r["text"]})
    sample = pd.DataFrame(chosen, columns=["text_hash", "platform", "text"])
    sample["frame_verdict"] = ""
    for cat in cfg.llm_categories:
        sample[cat] = ""
    out = out or (llm_dir(cfg) / "validation" / f"human_labelling_sample_{len(sample)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, encoding="utf-8-sig")
    try:
        sample.to_excel(out.with_suffix(".xlsx"), index=False)
    except Exception:  # noqa: BLE001  (openpyxl missing)
        pass
    print(f"sample of {len(sample)} texts for human labelling -> {out}")
    return sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--reference", type=Path, help="Reference CSV (text_hash + one 0/1 column per category).")
    parser.add_argument("--predictions", type=Path, default=None, help="Defaults to <work>/llm/llm_labels.csv")
    parser.add_argument("--name", default="reference")
    parser.add_argument("--sample", type=int, default=None, help="Draw N texts for human labelling instead.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if args.sample:
        draw_sample(cfg, args.sample, args.seed, args.out)
    elif args.reference:
        evaluate(cfg, args.reference, args.predictions, args.name)
    else:
        parser.error("give --reference (evaluate) or --sample N (draw a labelling sample)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
