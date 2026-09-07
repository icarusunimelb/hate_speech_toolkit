"""The record table: the one data contract shared by every stage.

``ingest`` writes one record table per platform (``<work>/records/<platform>_records.csv.gz``).
``classifiers.score`` appends ``<category>_probability`` / ``<category>_label`` columns plus
``any_hate`` / ``n_hate_cats``.  ``llm.merge_shards`` appends ``<category>_label`` /
``<category>_confidence`` for the LLM categories plus ``llm_labelled``, ``frame_verdict`` and
``any_extremist``.  ``analysis.*`` only ever read this table.

Core columns (all platforms; NaN where a platform has no such field):

    platform            x | telegram | instagram
    record_id           platform-native id, unique within platform
    content_type        post | reply | repost | comment
    author              account key used for counting and as the network node (lower-case handle)
    author_name         display name (may be empty)
    timestamp           ISO-8601 local time
    day                 YYYY-MM-DD (local)
    text                the post text (raw)
    text_hash           sha256 of the whitespace-normalised, lower-cased text
    url                 permalink (may be empty)
    reply_to_record_id  id of the record this replies to (if known)
    reply_to_author     author key of the account replied to
    parent_record_id    id of the top-level post a comment/reply belongs to
    parent_author       author key of that top-level post's account
    mentions            pipe-joined lower-case handles mentioned (text + platform metadata)
    forwarded_from      source account/channel of a forward (Telegram)
    likes, reposts, replies, views   engagement counts
    region, city, latitude, longitude   self-reported location (X only)
    in_study_period     bool
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

CORE_COLUMNS = [
    "platform", "record_id", "content_type", "author", "author_name", "timestamp", "day",
    "text", "text_hash", "url",
    "reply_to_record_id", "reply_to_author", "parent_record_id", "parent_author",
    "mentions", "forwarded_from",
    "likes", "reposts", "replies", "views",
    "region", "city", "latitude", "longitude",
    "in_study_period",
]
STRING_COLUMNS = [
    "platform", "record_id", "content_type", "author", "author_name", "timestamp", "day", "text",
    "text_hash", "url", "reply_to_record_id", "reply_to_author", "parent_record_id", "parent_author",
    "mentions", "forwarded_from", "region", "city", "frame_verdict",
]
NUMERIC_COLUMNS = ["likes", "reposts", "replies", "views", "latitude", "longitude"]

ANY_HATE = "any_hate"
N_HATE = "n_hate_cats"
ANY_EXTREMIST = "any_extremist"
LLM_LABELLED = "llm_labelled"
FRAME_VERDICT = "frame_verdict"

HANDLE_RE = {
    "x": re.compile(r"@([A-Za-z0-9_]{1,15})"),
    "telegram": re.compile(r"@([A-Za-z0-9_]{4,32})"),
    "instagram": re.compile(r"@([A-Za-z0-9_.]{1,30})"),
}


def label_col(category: str) -> str:
    return f"{category}_label"


def prob_col(category: str) -> str:
    return f"{category}_probability"


def conf_col(category: str) -> str:
    return f"{category}_confidence"


def normalize_text(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def text_hash(text: object) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def norm_handle(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lstrip("@").lower()
    return s or None


def extract_mentions(text: object, platform: str, extra: object = None) -> str | None:
    """Pipe-joined, sorted, lower-case handles from the text plus an optional metadata field."""
    handles: set[str] = set()
    if isinstance(extra, str) and extra.strip():
        for part in re.split(r"[,;|\s]+", extra):
            h = norm_handle(part)
            if h:
                handles.add(h)
    if isinstance(text, str) and "@" in text:
        handles.update(m.lower().rstrip(".") for m in HANDLE_RE[platform].findall(text))
    return "|".join(sorted(handles)) if handles else None


def finalize_records(df: pd.DataFrame, cfg: Config, platform: str) -> pd.DataFrame:
    """Coerce a platform frame to the contract: column set/order, dtypes, day, hash, study flag."""
    out = df.copy()
    for c in CORE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["platform"] = platform
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    keep = ts.notna()
    out = out[keep].copy()
    ts = ts[keep]
    out["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["day"] = ts.dt.strftime("%Y-%m-%d")
    out["text"] = out["text"].fillna("").astype(str)
    out["text_hash"] = out["text"].map(text_hash)
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in STRING_COLUMNS:
        if c in out.columns:
            col = out[c].astype("object")
            out[c] = col.where(col.notna(), None)
    day = pd.to_datetime(out["day"])
    out["in_study_period"] = (day >= cfg.study_start) & (day <= cfg.study_end)
    extra = [c for c in out.columns if c not in CORE_COLUMNS]
    return out[CORE_COLUMNS + extra].reset_index(drop=True)


def records_path(cfg: Config, platform: str) -> Path:
    return cfg.path("work", "work") / "records" / f"{platform}_records.csv.gz"


def load_records(cfg: Config, platform: str, usecols=None) -> pd.DataFrame:
    path = records_path(cfg, platform)
    if not path.exists():
        raise FileNotFoundError(
            f"no record table for {platform} at {path}; run `python -m hst.ingest --config ...` first"
        )
    dtype = {c: "object" for c in STRING_COLUMNS}
    df = pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)
    if "in_study_period" in df.columns:
        df["in_study_period"] = df["in_study_period"].astype(str).str.lower().eq("true")
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_records(cfg: Config, platform: str, df: pd.DataFrame) -> Path:
    path = records_path(cfg, platform)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return path


def hate_label_columns(cfg: Config, df: pd.DataFrame) -> list[str]:
    return [label_col(c) for c in cfg.category_names if label_col(c) in df.columns]


def llm_label_columns(cfg: Config, df: pd.DataFrame) -> list[str]:
    return [label_col(c) for c in cfg.llm_categories if label_col(c) in df.columns]
