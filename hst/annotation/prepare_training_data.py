"""Turn annotation-tool exports into training data, one folder per supervised category.

Input: every file listed under ``annotations.files`` in the project YAML.  Each is one row per
annotation with (at least) the columns

    annotator, text, hate_presence, target_group   (+ optional id; items without an id are keyed by text)

``hate_presence`` is Yes/No; ``target_group`` is the group(s) the annotator selected, either as a
plain value ("Women"), a separated list ("Women;Muslims") or the JSON an annotation tool exports
({"choices": ["Women"]}).  Extra columns are ignored.

Label rule (per category): an annotation is positive iff hate_presence == Yes AND target_group
contains the category's ``target_group`` (case-insensitive).  So hate aimed at another group is a
NEGATIVE example for this category, which is what makes the model target-specific.

Output, written to <work>/training/<category>/:

    items.csv            one row per item: item_key, text, label (0/1, empty for ties),
                         soft_label, n_annotators, positive_votes, negative_votes, source
    annotator_votes.csv  one row per item x annotator: item_key, annotator, label (0/1)
    conflicts.csv        items that received no hard label (ties, unanimity failures)
    summary.json         counts and per-annotator statistics

Run:  python -m hst.annotation.prepare_training_data --config project.yaml [--category C ...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, add_config_argument, config_from_args
from ..schema import normalize_text

REQUIRED = ["annotator", "text", "hate_presence", "target_group"]   # "id" is optional
AGGREGATIONS = ("majority", "any", "unanimous")


def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"could not decode {path}")


def clean_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def parse_target_group(value: object) -> set[str]:
    """'Women', 'Women;Muslims', '{"choices": ["Women"]}' -> {'women', ...} (lower-case)."""
    raw = clean_value(value)
    if not raw:
        return set()
    if raw[0] in "{[":
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                obj = obj.get("choices", [])
            if isinstance(obj, list):
                return {str(x).strip().lower() for x in obj if str(x).strip()}
            if isinstance(obj, str):
                raw = obj
        except json.JSONDecodeError:
            pass
    return {p.strip().lower() for p in re.split(r"[;,|]", raw) if p.strip()}


def annotation_label(hate_presence: object, target_group: object, target: str) -> float:
    """1.0 / 0.0 for a valid annotation, NaN when hate_presence is not Yes/No."""
    hate = clean_value(hate_presence).lower()
    if hate not in ("yes", "no"):
        return np.nan
    return float(hate == "yes" and target.lower() in parse_target_group(target_group))


def read_annotations(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        df = read_csv_any(Path(path))
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing annotation columns {missing}")
        df = df.copy()
        df["source"] = Path(path).name
        frames.append(df)
    if not frames:
        raise ValueError("no annotation files configured (annotations.files)")
    return pd.concat(frames, ignore_index=True)


def prepare_frame(raw: pd.DataFrame, target: str, aggregation: str = "majority",
                  min_annotations: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Core logic on an in-memory annotation frame.  Returns (items, votes, conflicts, summary)."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
    df = raw.copy()
    if "source" not in df.columns:
        df["source"] = "annotations"
    df["text"] = df["text"].map(clean_value)
    df["norm_text"] = df["text"].map(normalize_text)
    df["annotator"] = df["annotator"].map(lambda v: clean_value(v).lower())
    df["item_key"] = df["id"].map(clean_value) if "id" in df.columns else ""
    no_id = df["item_key"].eq("")
    df.loc[no_id, "item_key"] = "text:" + df.loc[no_id, "norm_text"]
    df["vote"] = [annotation_label(h, t, target) for h, t in zip(df["hate_presence"], df["target_group"])]
    valid = df[df["vote"].notna() & df["norm_text"].ne("") & df["annotator"].ne("")].copy()

    # one vote per item x annotator; a within-annotator disagreement (mean strictly between 0 and 1)
    # is dropped and counted
    per = (valid.groupby(["item_key", "annotator"], as_index=False)
           .agg(text=("text", "first"), source=("source", "first"), vote=("vote", "mean"), rows=("vote", "size")))
    conflict_mask = ~np.isclose(per["vote"], 0.0) & ~np.isclose(per["vote"], 1.0)
    duplicate_rows = int((per["rows"] > 1).sum())
    within_conflicts = int(conflict_mask.sum())
    per = per[~conflict_mask].copy()
    per["label"] = per["vote"].round().astype(int)
    votes = per[["item_key", "annotator", "label"]].sort_values(["item_key", "annotator"]).reset_index(drop=True)

    items = (per.groupby("item_key", as_index=False)
             .agg(text=("text", "first"), source=("source", "first"), n_annotators=("annotator", "nunique"),
                  positive_votes=("label", "sum")))
    items["negative_votes"] = items["n_annotators"] - items["positive_votes"]
    items["soft_label"] = (items["positive_votes"] / items["n_annotators"]).round(4)
    items = items[items["n_annotators"] >= int(min_annotations)].copy()

    label = pd.Series(np.nan, index=items.index, dtype="float")
    pos, neg, n = items["positive_votes"], items["negative_votes"], items["n_annotators"]
    if aggregation == "majority":
        label[pos > neg] = 1.0
        label[neg > pos] = 0.0
    elif aggregation == "any":
        label[pos > 0] = 1.0
        label[pos == 0] = 0.0
    else:
        label[pos == n] = 1.0
        label[neg == n] = 0.0
    items["label"] = label
    items = items[["item_key", "text", "label", "soft_label", "n_annotators", "positive_votes",
                   "negative_votes", "source"]].sort_values("item_key").reset_index(drop=True)
    conflicts = items[items["label"].isna()].copy()

    ann_stats = per.groupby("annotator")["label"].agg(["size", "mean"])
    summary = {
        "target_group": target,
        "aggregation": aggregation,
        "min_annotations": int(min_annotations),
        "annotation_rows_read": int(len(df)),
        "valid_annotation_rows": int(len(valid)),
        "duplicate_item_annotator_groups": duplicate_rows,
        "within_annotator_conflicts_dropped": within_conflicts,
        "items": int(len(items)),
        "positives": int((items["label"] == 1).sum()),
        "negatives": int((items["label"] == 0).sum()),
        "unlabelled_ties": int(len(conflicts)),
        "annotators_per_item": {str(k): int(v) for k, v in items["n_annotators"].value_counts().sort_index().items()},
        "annotator_row_counts": {k: int(v) for k, v in ann_stats["size"].items()},
        "annotator_positive_rates": {k: round(float(v), 4) for k, v in ann_stats["mean"].items()},
    }
    return items, votes, conflicts, summary


def training_dir(cfg: Config, category: str) -> Path:
    return cfg.path("work", "work") / "training" / category


def prepare_category(cfg: Config, category: str, raw: pd.DataFrame | None = None) -> dict:
    spec = cfg.categories[category]
    target = spec.get("target_group")
    if not target:
        raise ValueError(f"categories.{category}.target_group is not set")
    ann = cfg.section("annotations")
    if raw is None:
        raw = read_annotations([cfg.resolve(f) for f in ann.get("files", [])])
    items, votes, conflicts, summary = prepare_frame(
        raw, str(target), ann.get("aggregation", "majority"), int(ann.get("min_annotations", 1)))
    out = training_dir(cfg, category)
    out.mkdir(parents=True, exist_ok=True)
    items_out = items.copy()
    items_out["label"] = items_out["label"].map(lambda v: "" if pd.isna(v) else int(v))
    items_out.to_csv(out / "items.csv", index=False)
    votes.to_csv(out / "annotator_votes.csv", index=False)
    conflicts.to_csv(out / "conflicts.csv", index=False)
    summary["category"] = category
    summary["output_dir"] = str(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(cfg: Config, categories: list[str] | None = None) -> dict[str, dict]:
    files = [cfg.resolve(f) for f in cfg.section("annotations").get("files", [])]
    raw = read_annotations(files)
    results = {}
    for cat in categories or cfg.category_names:
        results[cat] = prepare_category(cfg, cat, raw)
        s = results[cat]
        print(f"{cat}: {s['items']} items, {s['positives']} positive, {s['negatives']} negative, "
              f"{s['unlabelled_ties']} ties -> {s['output_dir']}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_config_argument(parser)
    parser.add_argument("--category", action="append", help="Only this category (repeatable).")
    args = parser.parse_args(argv)
    run(config_from_args(args), args.category)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
