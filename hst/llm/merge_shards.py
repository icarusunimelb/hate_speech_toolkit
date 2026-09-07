"""Merge the LLM prediction shards into one label table and, optionally, attach it to the records.

Inputs   ``<work>/llm/predictions/shard_XX.jsonl`` (from ``hst.llm.classify_vllm``).
Outputs  ``<work>/llm/llm_labels.csv`` keyed by text_hash with
         labels_available, frame_verdict, <category>_label, <category>_confidence,
         <category>_strict (label AND confidence >= cfg.llm.strict_confidence), any_extremist,
         any_extremist_strict, n_categories;  plus ``merge_summary.json``.
         With ``--attach`` every platform record table gets llm_labelled, frame_verdict,
         <category>_label, <category>_confidence and any_extremist (NaN for unlabelled texts).

Rules    a text is labelled once: if a hash occurs in several shards or was retried, the LAST
         parse-valid record wins.  Two-step gate: when the frame verdict is negative every
         category label is forced to 0 (the prompt already requires this; contradictions are
         counted in the summary).

    python -m hst.llm.merge_shards --config project.yaml [--attach]
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, add_config_argument, config_from_args
from ..schema import ANY_EXTREMIST, FRAME_VERDICT, LLM_LABELLED, conf_col, label_col, load_records, save_records
from ._common import labels_path, llm_dir, load_schema, predictions_dir, read_jsonl, schema_categories, verdict_values, write_json


def strict_col(category: str) -> str:
    return f"{category}_strict"


def merge(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema = load_schema(cfg)
    categories = cfg.llm_categories or schema_categories(schema)
    positive, negative = verdict_values(schema)
    strict_conf = float(cfg.section("llm").get("strict_confidence", 0.8))

    latest: dict[str, dict[str, Any]] = {}
    files = sorted(predictions_dir(cfg).glob("shard_*.jsonl"))
    rows_read = Counter()
    for path in files:
        for rec in read_jsonl(path):
            h = str(rec["text_hash"]).lower()
            rows_read[path.name] += 1
            prev = latest.get(h)
            if prev is None or rec.get("parse_ok") or not prev.get("parse_ok"):
                latest[h] = rec

    out_rows = []
    contradictions = 0
    for h, rec in latest.items():
        parsed = rec.get("parsed") if rec.get("parse_ok") else None
        row: dict[str, Any] = {"text_hash": h, "labels_available": parsed is not None,
                               "frame_verdict": parsed.get("frame_verdict") if parsed else None}
        if parsed is None:
            for cat in categories:
                row[label_col(cat)] = np.nan
                row[conf_col(cat)] = np.nan
        else:
            gated = positive is not None and row["frame_verdict"] != positive
            for cat in categories:
                d = parsed.get(cat) or {}
                lab = int(bool(d.get("label")))
                if gated and lab:
                    contradictions += 1
                    lab = 0
                row[label_col(cat)] = lab
                row[conf_col(cat)] = float(d.get("confidence")) if d.get("confidence") is not None else np.nan
        out_rows.append(row)

    cols = ["text_hash", "labels_available", "frame_verdict"]
    for cat in categories:
        cols += [label_col(cat), conf_col(cat), strict_col(cat)]
    cols += ["any_extremist", "any_extremist_strict", "n_categories"]
    df = pd.DataFrame(out_rows)
    if df.empty:
        df = pd.DataFrame(columns=cols)
    else:
        for cat in categories:
            df[strict_col(cat)] = ((df[label_col(cat)] == 1) & (df[conf_col(cat)] >= strict_conf)).astype(float)
            df.loc[~df["labels_available"], strict_col(cat)] = np.nan
        label_cols = [label_col(c) for c in categories]
        n_cats = df[label_cols].sum(axis=1, min_count=1)
        df["n_categories"] = n_cats
        df["any_extremist"] = (n_cats > 0).astype(float).where(df["labels_available"])
        strict_any = df[[strict_col(c) for c in categories]].sum(axis=1, min_count=1)
        df["any_extremist_strict"] = (strict_any > 0).astype(float).where(df["labels_available"])
        df = df[cols].sort_values("text_hash").reset_index(drop=True)

    path = labels_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    available = df["labels_available"].astype(bool) if len(df) else pd.Series(dtype=bool)
    summary = {
        "prediction_files": [{"file": k, "rows": v} for k, v in sorted(rows_read.items())],
        "unique_texts": int(len(df)),
        "labels_available": int(available.sum()),
        "parse_failed": int((~available).sum()) if len(df) else 0,
        "frame_verdict_counts": df["frame_verdict"].value_counts(dropna=False).to_dict() if len(df) else {},
        "frame_label_contradictions_zeroed": contradictions,
        "positives": {c: int(df[label_col(c)].fillna(0).sum()) for c in categories} if len(df) else {},
        "positives_strict": {c: int(df[strict_col(c)].fillna(0).sum()) for c in categories} if len(df) else {},
        "any_extremist": int(df["any_extremist"].fillna(0).sum()) if len(df) else 0,
        "strict_confidence": strict_conf,
        "output": str(path),
    }
    print(f"llm labels: {len(df):,} unique texts ({summary['labels_available']:,} usable) -> {path}")
    return df, summary


def attach(cfg: Config, labels: pd.DataFrame | None = None) -> dict[str, dict[str, int]]:
    """Left-join the label table onto every platform record table and save it."""
    if labels is None:
        labels = pd.read_csv(labels_path(cfg), dtype={"text_hash": str})
    categories = cfg.llm_categories
    keep = ["text_hash", "labels_available", "frame_verdict"] + [c for cat in categories for c in (label_col(cat), conf_col(cat))] + ["any_extremist"]
    lab = labels[keep].copy()
    lab["text_hash"] = lab["text_hash"].astype(str).str.lower()
    lab = lab.drop_duplicates("text_hash")
    report: dict[str, dict[str, int]] = {}
    for platform in cfg.platforms:
        df = load_records(cfg, platform)
        drop = [c for c in df.columns if c in {LLM_LABELLED, FRAME_VERDICT, ANY_EXTREMIST, "labels_available"} or
                any(c in (label_col(cat), conf_col(cat)) for cat in categories)]
        df = df.drop(columns=drop)
        df["text_hash"] = df["text_hash"].astype(str).str.lower()
        merged = df.merge(lab, on="text_hash", how="left")
        merged[LLM_LABELLED] = merged["labels_available"].fillna(False).astype(bool)
        merged = merged.drop(columns=["labels_available"])
        for cat in categories:
            merged.loc[~merged[LLM_LABELLED], [label_col(cat), conf_col(cat)]] = np.nan
        merged.loc[~merged[LLM_LABELLED], [ANY_EXTREMIST, FRAME_VERDICT]] = [np.nan, None]
        save_records(cfg, platform, merged)
        report[platform] = {"records": int(len(merged)), "llm_labelled": int(merged[LLM_LABELLED].sum()),
                            "any_extremist": int(merged[ANY_EXTREMIST].fillna(0).sum())}
        print(f"{platform}: {report[platform]['llm_labelled']:,}/{len(merged):,} records carry LLM labels")
    return report


def run(cfg: Config, do_attach: bool = False) -> dict[str, Any]:
    df, summary = merge(cfg)
    if do_attach:
        summary["attached"] = attach(cfg, df)
    write_json(llm_dir(cfg) / "merge_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--attach", action="store_true", help="Also add the labels to every platform record table.")
    args = parser.parse_args(argv)
    run(config_from_args(args), args.attach)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
