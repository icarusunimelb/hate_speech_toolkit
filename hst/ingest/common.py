"""Helpers shared by the platform ingest modules.

Covers the small, boring things every raw export needs: reading CSVs whose fields can hold
very long text, tolerating a few encodings, parsing the three timestamp formats we meet
(Brandwatch UTC, Telegram "AEST/AEDT", Instagram "Australia/Sydney"), turning count-like
strings into numbers, and auditing which days an export actually covers.
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


def maximize_csv_field_size() -> None:
    """Allow csv to read fields longer than its default 128 kB limit (long posts, raw metadata)."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    """pandas.read_csv with an encoding fallback and every column read as text."""
    kwargs.setdefault("dtype", str)
    kwargs.setdefault("keep_default_na", False)
    kwargs.setdefault("low_memory", False)
    last: Exception | None = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last = exc
    raise RuntimeError(f"could not decode {path}") from last


# ---- timestamps --------------------------------------------------------------------------------
def parse_utc_to_local(values: pd.Series, timezone: str) -> pd.Series:
    """Brandwatch dates ("2026-06-01 13:59:45.0") carry no zone and are UTC.  Returns naive local time."""
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    ts = pd.to_datetime(text, errors="coerce", utc=True)
    return ts.dt.tz_convert(timezone).dt.tz_localize(None)


_SUFFIX_RE = re.compile(r"\s+(?:[A-Z]{3,5}|[A-Za-z_]+/[A-Za-z_]+)$")


def parse_local_with_suffix(values: pd.Series) -> pd.Series:
    """Telegram ("2026-07-01 08:55:44 AEST") and Instagram ("2025-06-16T01:08:55 Australia/Sydney")
    stamps are already local time followed by a zone name.  Drop the name; return naive local time."""
    text = values.astype(str).str.strip().str.replace(_SUFFIX_RE, "", regex=True)
    return pd.to_datetime(text, errors="coerce", format="ISO8601")


# ---- numbers -----------------------------------------------------------------------------------
def parse_count(values: pd.Series) -> pd.Series:
    """'1,234' / '12' / '' -> float (NaN when blank or not a number)."""
    text = values.astype(str).str.strip().str.replace(",", "", regex=False)
    out = pd.to_numeric(text, errors="coerce")
    return out.astype(float)


# ---- date coverage -----------------------------------------------------------------------------
def contiguous_ranges(days: list[date]) -> list[dict[str, object]]:
    """[d1, d2, d4] -> [{start: d1, end: d2, days: 2}, {start: d4, end: d4, days: 1}]"""
    if not days:
        return []
    days = sorted(days)
    ranges: list[dict[str, object]] = []
    start = prev = days[0]
    for current in days[1:]:
        if (current - prev).days == 1:
            prev = current
            continue
        ranges.append({"start": start.isoformat(), "end": prev.isoformat(), "days": (prev - start).days + 1})
        start = prev = current
    ranges.append({"start": start.isoformat(), "end": prev.isoformat(), "days": (prev - start).days + 1})
    return ranges


def date_coverage(days: pd.Series) -> dict[str, object]:
    """Which days have rows, and which days inside [min, max] have none."""
    observed = sorted({d.date() if hasattr(d, "date") else d for d in pd.to_datetime(days, errors="coerce").dropna()})
    if not observed:
        return {"date_min": None, "date_max": None, "days_with_data": 0, "missing_days": 0, "missing_day_ranges": []}
    lo, hi = observed[0], observed[-1]
    expected = {lo + timedelta(days=i) for i in range((hi - lo).days + 1)}
    missing = sorted(expected - set(observed))
    return {
        "date_min": lo.isoformat(),
        "date_max": hi.isoformat(),
        "days_with_data": len(observed),
        "missing_days": len(missing),
        "missing_day_ranges": contiguous_ranges(missing),
    }


def blank_to_none(values: pd.Series) -> pd.Series:
    """Text column where '' (and NaN) become None."""
    text = values.astype("object")
    mask = text.isna() | (text.astype(str).str.strip() == "")
    return text.where(~mask, None)


def value_counts_dict(values: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in values.value_counts(dropna=False).items()}


def nan_if_blank(values: pd.Series) -> pd.Series:
    """Text column -> float NaN placeholder friendly object (keeps numpy happy in finalize)."""
    out = blank_to_none(values)
    return out.where(out.notna(), np.nan)
