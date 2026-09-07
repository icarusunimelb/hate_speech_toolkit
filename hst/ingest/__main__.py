"""Ingest stage: raw platform exports -> one record table per platform.

    python -m hst.ingest --config project.yaml [--platform x --platform telegram]

For each platform the raw exports under ``<paths.raw>/<platform>`` are read, mapped to the
record contract (hst.schema), finalised and written to ``<paths.work>/records/<platform>_records.csv.gz``
together with ``<platform>_ingest_summary.json`` (rows, authors, content types, date coverage,
study-period count) and, for X, ``x_ingest_audit.json`` (per-export details, duplicates).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import Config, add_config_argument, config_from_args
from ..schema import finalize_records, records_path, save_records
from . import brandwatch_x, instagram, telegram
from .common import date_coverage, value_counts_dict

BUILDERS = {"x": brandwatch_x.build, "telegram": telegram.build, "instagram": instagram.build}


def summarize(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "unique_authors": int(df["author"].dropna().nunique()),
        "content_type": value_counts_dict(df["content_type"]),
        "rows_with_reply_to_author": int(df["reply_to_author"].notna().sum()),
        "rows_with_mentions": int(df["mentions"].notna().sum()),
        "in_study_period": int(df["in_study_period"].sum()),
        "coverage": date_coverage(df["day"]),
    }


def ingest_platform(cfg: Config, platform: str) -> dict[str, object]:
    out_dir = records_path(cfg, platform).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if platform == "x":
        raw, audit = brandwatch_x.build_with_audit(cfg)
        (out_dir / "x_ingest_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    else:
        raw = BUILDERS[platform](cfg)
    df = finalize_records(raw, cfg, platform)
    path = save_records(cfg, platform, df)
    summary = {"platform": platform, "output": str(path), **summarize(df)}
    (out_dir / f"{platform}_ingest_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    cov = summary["coverage"]
    print(f"{platform}: {summary['rows']:,} records, {summary['unique_authors']:,} authors, "
          f"{cov['date_min']} to {cov['date_max']} ({cov['missing_days']} missing days) -> {path}")
    return summary


def run(cfg: Config, platforms: list[str] | None = None) -> dict[str, dict[str, object]]:
    platforms = platforms or cfg.platforms
    return {p: ingest_platform(cfg, p) for p in platforms}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--platform", action="append", choices=list(BUILDERS), help="Restrict to a platform (repeatable).")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    run(cfg, args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
