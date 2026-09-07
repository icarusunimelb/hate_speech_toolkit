"""Shared loading helpers for the analysis stage.

``load_analysis_records(cfg, platform)`` reads a platform record table (see hst.schema), restricts
it to the study period, makes every label column numeric, fills in ``any_hate`` / ``n_hate_cats`` /
``llm_labelled`` / ``any_extremist`` when a stage did not write them, and adds
``circulation_adjusted_volume`` (1 + reposts for X, 1 elsewhere).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..schema import ANY_EXTREMIST, ANY_HATE, LLM_LABELLED, N_HATE, label_col, load_records

ENGAGEMENT_COLUMNS = ["likes", "reposts", "replies", "views"]


def supervised_categories(cfg: Config, df: pd.DataFrame) -> list[str]:
    return [c for c in cfg.category_names if label_col(c) in df.columns]


def llm_categories(cfg: Config, df: pd.DataFrame) -> list[str]:
    return [c for c in cfg.llm_categories if label_col(c) in df.columns]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def load_analysis_records(cfg: Config, platform: str) -> pd.DataFrame:
    df = load_records(cfg, platform)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df[df["day"].notna() & (df["day"] >= cfg.study_start) & (df["day"] <= cfg.study_end)].copy()
    df = df.reset_index(drop=True)

    hate = supervised_categories(cfg, df)
    for cat in hate:
        df[label_col(cat)] = pd.to_numeric(df[label_col(cat)], errors="coerce").fillna(0).astype(int)
    if hate:
        df[N_HATE] = df[[label_col(c) for c in hate]].sum(axis=1).astype(int)
    else:
        df[N_HATE] = 0
    df[ANY_HATE] = (df[N_HATE] > 0).astype(int)

    llm = llm_categories(cfg, df)
    for cat in llm:
        df[label_col(cat)] = pd.to_numeric(df[label_col(cat)], errors="coerce")
    if LLM_LABELLED in df.columns:
        df[LLM_LABELLED] = _as_bool(df[LLM_LABELLED])
    elif llm:
        df[LLM_LABELLED] = df[[label_col(c) for c in llm]].notna().any(axis=1)
    else:
        df[LLM_LABELLED] = False
    if llm:
        ext = df[[label_col(c) for c in llm]].sum(axis=1, min_count=1)
        df[ANY_EXTREMIST] = np.where(df[LLM_LABELLED], (ext.fillna(0) > 0).astype(float), np.nan)
    else:
        df[ANY_EXTREMIST] = np.nan

    for c in ENGAGEMENT_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    if platform == "x":
        df["circulation_adjusted_volume"] = 1.0 + df["reposts"].fillna(0)
    else:
        df["circulation_adjusted_volume"] = 1.0
    df["author"] = df["author"].astype("object").where(df["author"].notna(), None)
    return df
