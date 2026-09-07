"""Inter-annotator agreement for one category's annotations.

Input: <work>/training/<category>/annotator_votes.csv (item_key, annotator, label) written by
prepare_training_data.  Output: agreement.json and agreement.md in the same folder with

    Krippendorff's alpha (nominal, any number of raters per item)
    Fleiss' kappa on the items that have the modal number of raters
    pairwise Cohen's kappa and percent agreement for annotator pairs sharing >= 10 items
    per-annotator positive rates and row counts
    unanimous / split / tie shares among multi-annotated items

Why: if annotators only agree moderately, no model trained on their labels can be expected to do
better than that on the same construct, and annotators with very different positive rates are
a sign of drifting definitions.  Run this before training and report the numbers.

Run:  python -m hst.annotation.agreement --config project.yaml [--category C ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, add_config_argument, config_from_args

BANDS = ("Interpretation bands (Landis & Koch): <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, "
         "0.61-0.80 substantial, >0.80 near-perfect.")


def krippendorff_alpha_nominal(item_votes: list[list[int]]) -> float:
    """Nominal alpha over items with >= 2 votes (coincidence-matrix form)."""
    units = [v for v in item_votes if len(v) >= 2]
    if not units:
        return float("nan")
    cats = sorted({c for v in units for c in v})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    o = np.zeros((k, k))
    for v in units:
        m = len(v)
        counts = np.zeros(k)
        for c in v:
            counts[idx[c]] += 1
        for a in range(k):
            for b in range(k):
                o[a, b] += counts[a] * (counts[a] - 1) / (m - 1) if a == b else counts[a] * counts[b] / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    d_o = sum(o[a, b] for a in range(k) for b in range(k) if a != b)
    d_e = sum(n_c[a] * n_c[b] for a in range(k) for b in range(k) if a != b) / (n - 1)
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def fleiss_kappa(matrix: np.ndarray) -> float:
    """matrix: items x categories counts, constant number of raters per item."""
    n_raters = matrix.sum(axis=1)
    if len(matrix) < 2 or len(set(n_raters.tolist())) != 1 or n_raters[0] < 2:
        return float("nan")
    r = n_raters[0]
    p_j = matrix.sum(axis=0) / matrix.sum()
    p_i = ((matrix ** 2).sum(axis=1) - r) / (r * (r - 1))
    p_bar = p_i.mean()
    p_e = (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def analyse(votes: pd.DataFrame, min_shared: int = 10) -> dict:
    d = votes[["item_key", "annotator", "label"]].dropna().copy()
    d["label"] = pd.to_numeric(d["label"], errors="coerce").round().astype(int)
    d = d.drop_duplicates(["item_key", "annotator"])
    per_item = d.groupby("item_key")["label"].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "yes"})
    per_item["no"] = per_item["n"] - per_item["yes"]
    multi = per_item[per_item["n"] >= 2]
    unanimous = (multi["yes"] == 0) | (multi["no"] == 0)
    tie = multi["yes"] == multi["no"]

    item_votes = [g["label"].tolist() for _, g in d.groupby("item_key")]
    alpha = krippendorff_alpha_nominal(item_votes)
    modal_n = int(multi["n"].mode().iloc[0]) if len(multi) else 0
    sub = multi[multi["n"] == modal_n]
    fk = fleiss_kappa(np.stack([sub["no"].to_numpy(), sub["yes"].to_numpy()], axis=1).astype(float)) if len(sub) else float("nan")

    pivot = d.pivot_table(index="item_key", columns="annotator", values="label", aggfunc="first")
    pairs, agree, total = [], 0, 0
    for a1, a2 in itertools.combinations(sorted(pivot.columns), 2):
        both = pivot[[a1, a2]].dropna()
        if len(both) < min_shared:
            continue
        x, y = both[a1].to_numpy().astype(int), both[a2].to_numpy().astype(int)
        agree += int((x == y).sum())
        total += len(both)
        pairs.append({"annotator_a": a1, "annotator_b": a2, "shared_items": int(len(both)),
                      "percent_agreement": round(float((x == y).mean()), 4),
                      "cohen_kappa": round(cohen_kappa(x, y), 4),
                      "positive_rate_a": round(float(x.mean()), 4), "positive_rate_b": round(float(y.mean()), 4)})

    def _r(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)

    return {
        "annotation_rows": int(len(d)),
        "unique_items": int(len(per_item)),
        "items_with_2plus_annotators": int(len(multi)),
        "annotators_per_item": {str(k): int(v) for k, v in per_item["n"].value_counts().sort_index().items()},
        "annotator_row_counts": {str(k): int(v) for k, v in d.groupby("annotator").size().items()},
        "annotator_positive_rates": {str(k): round(float(v), 4) for k, v in d.groupby("annotator")["label"].mean().items()},
        "krippendorff_alpha_nominal": _r(alpha),
        "fleiss_kappa": {"raters_per_item": modal_n, "items": int(len(sub)), "kappa": _r(fk)},
        "pairwise_percent_agreement_overall": round(agree / total, 4) if total else None,
        "pairwise": pairs,
        "vote_split_profile_multi_annotated": {
            "unanimous_share": _r(unanimous.mean()) if len(multi) else None,
            "split_share": _r(1 - unanimous.mean()) if len(multi) else None,
            "tie_share": _r(tie.mean()) if len(multi) else None,
        },
    }


def to_markdown(category: str, r: dict) -> str:
    prof = r["vote_split_profile_multi_annotated"]
    lines = [f"# Inter-annotator agreement: {category}", "",
             "| items (2+ raters) | raters/item | Krippendorff alpha | Fleiss kappa | pairwise % agree | unanimous share | tie share |",
             "|---|---|---|---|---|---|---|",
             f"| {r['items_with_2plus_annotators']} | {r['fleiss_kappa']['raters_per_item']} | {r['krippendorff_alpha_nominal']} | "
             f"{r['fleiss_kappa']['kappa']} | {r['pairwise_percent_agreement_overall']} | {prof['unanimous_share']} | {prof['tie_share']} |",
             "", "Per-annotator positive rates:", ""]
    for k, v in r["annotator_positive_rates"].items():
        lines.append(f"- {k}: {v} ({r['annotator_row_counts'].get(k, 0)} annotations)")
    if r["pairwise"]:
        lines += ["", "| pair | shared items | % agreement | Cohen kappa |", "|---|---|---|---|"]
        for p in r["pairwise"]:
            lines.append(f"| {p['annotator_a']} vs {p['annotator_b']} | {p['shared_items']} | {p['percent_agreement']} | {p['cohen_kappa']} |")
    lines += ["", BANDS, ""]
    return "\n".join(lines)


def run(cfg: Config, categories: list[str] | None = None) -> dict[str, dict]:
    results = {}
    for cat in categories or cfg.category_names:
        folder = cfg.path("work", "work") / "training" / cat
        votes_path = folder / "annotator_votes.csv"
        if not votes_path.exists():
            raise FileNotFoundError(f"{votes_path} not found; run prepare_training_data first")
        votes = pd.read_csv(votes_path, dtype={"item_key": str, "annotator": str})
        r = analyse(votes)
        r["category"] = cat
        (folder / "agreement.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        (folder / "agreement.md").write_text(to_markdown(cat, r), encoding="utf-8")
        print(f"{cat}: alpha={r['krippendorff_alpha_nominal']} fleiss={r['fleiss_kappa']['kappa']} "
              f"items(2+)={r['items_with_2plus_annotators']} -> {folder / 'agreement.md'}")
        results[cat] = r
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inter-annotator agreement per category.")
    add_config_argument(parser)
    parser.add_argument("--category", action="append")
    args = parser.parse_args(argv)
    run(config_from_args(args), args.category)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
