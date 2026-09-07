"""X (Twitter) ingest from Brandwatch "Bulk Mentions Download" CSV exports.

Input:  every ``*.csv`` FILE found recursively under ``<paths.raw>/x``.  Each export starts with
        a few quoted label lines ("Report:", "Brand:", "From:", "To:", "Label:"), a blank line and
        then the real header.  Exports made at different times may add columns; we take the
        union of all headers.  Exports usually overlap in time, so rows are de-duplicated on
        ``Resource Id`` (falling back to ``Url``), keeping the first occurrence in export order
        (exports are ordered by their "From:" date).
Output: a frame in the record contract (see hst.schema) via ``build(cfg)``, plus an audit dict
        from ``audit(...)`` describing the exports, duplicates removed, header differences and
        date coverage.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..schema import extract_mentions, norm_handle
from .common import (blank_to_none, date_coverage, maximize_csv_field_size, parse_count,
                     parse_utc_to_local, value_counts_dict)

STATUS_URL_RE = re.compile(r"(?:twitter|x)\.com/([^/]+)/status/(\d+)", re.IGNORECASE)
CONTENT_TYPE = {"post": "post", "reply": "reply", "share": "repost", "comment": "reply"}


# ---- reading one export ------------------------------------------------------------------------
def read_preamble_and_header(path: Path) -> tuple[dict[str, str], list[str], int]:
    """Returns (preamble metadata, header row, number of rows before the header)."""
    meta: dict[str, str] = {}
    skipped = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if not row or all(not c.strip() for c in row):
                skipped += 1
                continue
            if len(row) <= 2 and row[0].strip().endswith(":"):
                meta[row[0].strip().rstrip(":")] = row[1].strip() if len(row) > 1 else ""
                skipped += 1
                continue
            return meta, [c.strip() for c in row], skipped
    raise ValueError(f"no data header found in {path}")


def read_export(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """One export -> DataFrame (all text) in file order, plus its preamble metadata."""
    maximize_csv_field_size()
    meta, header, skipped = read_preamble_and_header(path)
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for _ in range(skipped + 1):
            next(reader, None)
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            rows.append({col: (row[i] if i < len(row) else "") for i, col in enumerate(header)})
    df = pd.DataFrame(rows, columns=header, dtype=object)
    return df, meta


def export_sort_key(path: Path) -> tuple[pd.Timestamp, str]:
    meta, _, _ = read_preamble_and_header(path)
    start = pd.to_datetime(meta.get("From", ""), errors="coerce", utc=True)
    if pd.isna(start):
        start = pd.Timestamp.max.tz_localize("UTC")
    return start, str(path)


def find_exports(raw_x: Path) -> list[Path]:
    files = [p for p in raw_x.rglob("*.csv") if p.is_file()]
    return sorted(files, key=export_sort_key)


# ---- combining exports -------------------------------------------------------------------------
def combine_exports(files: list[Path]) -> tuple[pd.DataFrame, list[dict[str, object]], list[str]]:
    frames, infos = [], []
    union: list[str] = []
    for i, path in enumerate(files):
        df, meta = read_export(path)
        df["_export_order"] = i
        df["_export_file"] = str(path)
        infos.append({"file": str(path), "label": meta.get("Label", ""), "from": meta.get("From", ""),
                      "to": meta.get("To", ""), "rows": int(len(df)), "columns": int(df.shape[1] - 2),
                      "new_columns": [c for c in df.columns if c not in union and not c.startswith("_")]})
        for c in df.columns:
            if c not in union and not c.startswith("_"):
                union.append(c)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no Brandwatch export CSV files found")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for c in union:
        if c not in combined.columns:
            combined[c] = ""
    combined = combined.reindex(columns=union + ["_export_order", "_export_file"])
    return combined.fillna(""), infos, union


def dedupe(combined: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rid = combined["Resource Id"].astype(str).str.strip() if "Resource Id" in combined.columns else pd.Series("", index=combined.index)
    url = combined["Url"].astype(str).str.strip() if "Url" in combined.columns else pd.Series("", index=combined.index)
    fallback = combined["_export_order"].astype(str) + ":" + combined.index.astype(str)
    key = rid.where(rid != "", url.where(url != "", fallback))
    keep = ~key.duplicated(keep="first")
    stats = {"rows_read": int(len(combined)), "duplicates_removed": int((~keep).sum()),
             "rows_missing_resource_id": int((rid == "").sum())}
    out = combined[keep].copy()
    out["_key"] = key[keep]
    return out, stats


# ---- mapping to the record contract ---------------------------------------------------------
def to_records(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    col = lambda name: df[name] if name in df.columns else pd.Series("", index=df.index, dtype=object)  # noqa: E731
    entry = col("Thread Entry Type").astype(str).str.strip().str.lower()
    content_type = entry.map(CONTENT_TYPE)
    content_type = content_type.where(content_type.notna(), np.where(col("X Repost of").astype(str).str.strip() != "", "repost", "post"))

    reply_to = col("X Reply to").astype(str).str.extract(STATUS_URL_RE)
    reply_author = reply_to[0].map(norm_handle)
    reply_id = blank_to_none(reply_to[1])

    text = col("Full Text").astype(str)
    mentioned = col("Mentioned Authors").astype(str)
    mentions = [extract_mentions(t, "x", m) for t, m in zip(text.tolist(), mentioned.tolist())]

    out = pd.DataFrame({
        "record_id": df["_key"].astype(str),
        "content_type": content_type.astype(str),
        "author": col("Author").map(norm_handle),
        "author_name": blank_to_none(col("Full Name")),
        "timestamp": parse_utc_to_local(col("Date"), cfg.timezone),
        "text": text,
        "url": blank_to_none(col("Url")),
        "reply_to_record_id": reply_id,
        "reply_to_author": reply_author.where(reply_author.notna(), None),
        "parent_record_id": blank_to_none(col("Thread Id")),
        "parent_author": col("Thread Author").map(norm_handle),
        "mentions": pd.Series(mentions, index=df.index, dtype=object),
        "forwarded_from": None,
        "likes": parse_count(col("X Likes")),
        "reposts": parse_count(col("X Reposts")),
        "replies": parse_count(col("X Replies")),
        "views": parse_count(col("Impressions")),
        "region": blank_to_none(col("Region")),
        "city": blank_to_none(col("City")),
        "latitude": parse_count(col("Latitude")),
        "longitude": parse_count(col("Longitude")),
    }, index=df.index)
    # a repost's parent is the reposted account (Thread Author already holds it in Brandwatch)
    out["parent_author"] = out["parent_author"].where(out["parent_author"].notna(), None)
    return out.reset_index(drop=True)


def build(cfg: Config) -> pd.DataFrame:
    """<raw>/x/** exports -> record-contract frame (pre-finalize)."""
    records, _ = build_with_audit(cfg)
    return records


def build_with_audit(cfg: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_x = cfg.path("raw") / "x"
    files = find_exports(raw_x)
    if not files:
        raise FileNotFoundError(f"no Brandwatch export CSV files under {raw_x}")
    combined, infos, union = combine_exports(files)
    deduped, stats = dedupe(combined)
    records = to_records(deduped, cfg)
    audit = {
        "exports": infos,
        "union_columns": len(union),
        **stats,
        "rows_kept": int(len(records)),
        "invalid_timestamps": int(records["timestamp"].isna().sum()),
        "content_type": value_counts_dict(records["content_type"]),
        "coverage": date_coverage(records["timestamp"]),
    }
    return records, audit
