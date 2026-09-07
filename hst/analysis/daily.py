"""Daily volume analysis.

Inputs:  the platform record tables (after scoring and, optionally, LLM labelling).
Outputs: <work>/analysis/daily/<platform>_daily.csv    one row per day of the study period: records,
             posts, replies/comments, active accounts, each supervised category, any_hate, each LLM
             category, any_extremist (counts and proportions), engagement sums and, for X, the
             circulation-adjusted volume (1 + reposts) of every measure
         <work>/analysis/daily/<platform>_weekly.csv   the same by week, plus active / new accounts
         <work>/analysis/daily/combined_weekly.csv     all platforms in one weekly table
         <work>/analysis/daily/daily_summary.json
         <work>/figures/daily/<platform>/*.png         all-content timeline, small multiples per label
             family (counts and proportions), circulation-adjusted versions for X, weekly accounts
         <work>/figures/daily/combined_*.png           stacked weekly volume by platform

Proportions of supervised categories are over all records of the day; proportions of LLM
categories are over the records the LLM labelled that day.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from ..config import Config, add_config_argument, config_from_args
from ..schema import ANY_EXTREMIST, ANY_HATE, LLM_LABELLED, label_col
from .common import ENGAGEMENT_COLUMNS, llm_categories, load_analysis_records, supervised_categories
from .plotting import (PLATFORM_COLOR, ROLL, add_event_markers, category_colors, category_label, fig_dir,
                       platform_label, rolling, save_csv, save_fig, save_json, style_axes, table_dir, xaxis_dates)


def week_start(days: pd.Series) -> pd.Series:
    """Monday of the week each day belongs to."""
    return days - pd.to_timedelta(days.dt.dayofweek, unit="D")


def daily_table(df: pd.DataFrame, cfg: Config, platform: str) -> pd.DataFrame:
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    days = pd.date_range(cfg.study_start, cfg.study_end, freq="D")
    d = df.assign(is_post=(df["content_type"] == "post").astype(int))
    g = d.groupby("day")
    out = pd.DataFrame(index=days)
    observed = g.size().reindex(days)
    out["records"] = observed
    out["posts"] = g["is_post"].sum()
    out["replies_or_comments"] = out["records"] - out["posts"]
    out["active_accounts"] = g["author"].nunique()
    out["llm_labelled"] = g[LLM_LABELLED].sum()
    for cat in hate:
        out[cat] = g[label_col(cat)].sum()
    out[ANY_HATE] = g[ANY_HATE].sum()
    for cat in llm:
        out[cat] = g[label_col(cat)].sum(min_count=1)
    out[ANY_EXTREMIST] = g[ANY_EXTREMIST].sum(min_count=1)
    for c in ENGAGEMENT_COLUMNS:
        if d[c].notna().any():
            out[f"{c}_sum"] = g[c].sum(min_count=1)
    if platform == "x":
        circ = d["circulation_adjusted_volume"]
        out["records_circulation_adjusted"] = g["circulation_adjusted_volume"].sum()
        for cat in hate + llm:
            out[f"{cat}_circulation_adjusted"] = (d[label_col(cat)].fillna(0) * circ).groupby(d["day"]).sum()
        out["any_hate_circulation_adjusted"] = (d[ANY_HATE] * circ).groupby(d["day"]).sum()
        out["any_extremist_circulation_adjusted"] = (d[ANY_EXTREMIST].fillna(0) * circ).groupby(d["day"]).sum()
    out["observed_day"] = observed.fillna(0).gt(0)
    count_cols = [c for c in out.columns if c != "observed_day"]
    out[count_cols] = out[count_cols].fillna(0)
    for cat in hate + [ANY_HATE]:
        out[f"{cat}_proportion"] = out[cat] / out["records"].replace(0, np.nan)
    for cat in llm + [ANY_EXTREMIST]:
        out[f"{cat}_proportion"] = out[cat] / out["llm_labelled"].replace(0, np.nan)
    out["weekday"] = out.index.dayofweek
    out.index.name = "date"
    return out.reset_index()


def weekly_table(df: pd.DataFrame, daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    d = daily.set_index("date")
    skip = {"observed_day", "weekday", "active_accounts"}
    sum_cols = [c for c in d.columns if not c.endswith("_proportion") and c not in skip]
    w = d[sum_cols].resample("W-MON", label="left", closed="left").sum(min_count=1)
    w["observed_days"] = d["observed_day"].astype(int).resample("W-MON", label="left", closed="left").sum()
    wk = week_start(df["day"])
    w["active_accounts"] = df.groupby(wk)["author"].nunique().reindex(w.index)
    first_seen = df.groupby("author")["day"].min()
    w["new_accounts"] = week_start(first_seen).value_counts().reindex(w.index).fillna(0).astype(int)
    hateful = df[df[ANY_HATE] == 1]
    w["active_accounts_with_hate"] = hateful.groupby(week_start(hateful["day"]))["author"].nunique().reindex(w.index).fillna(0).astype(int)
    first_hate = hateful.groupby("author")["day"].min()
    w["new_accounts_with_hate"] = week_start(first_hate).value_counts().reindex(w.index).fillna(0).astype(int)
    for cat in hate + [ANY_HATE]:
        w[f"{cat}_proportion"] = w[cat] / w["records"].replace(0, np.nan)
    for cat in llm + [ANY_EXTREMIST]:
        w[f"{cat}_proportion"] = w[cat] / w["llm_labelled"].replace(0, np.nan)
    w.index.name = "week_start"
    return w.reset_index()


# ---------------------------------------------------------------------------------------------
def plot_all_content(daily: pd.DataFrame, cfg: Config, platform: str, outdir) -> None:
    d = daily.set_index("date")
    panels = [("records", "all_content_daily", f"{platform_label(platform)}: posts per day", "count of posts")]
    if "records_circulation_adjusted" in d.columns:
        panels.append(("records_circulation_adjusted", "all_content_daily_circulation_adjusted",
                       "X: circulation-adjusted volume per day", "posts + reposts"))
    for col, tag, title, ylabel in panels:
        fig, ax = plt.subplots(figsize=(9.5, 3.3))
        ax.plot(d.index, d[col], color=PLATFORM_COLOR[platform], linewidth=0.7, alpha=0.35, label="daily")
        ax.plot(d.index, rolling(d[col]), color=PLATFORM_COLOR[platform], linewidth=1.8, label=f"{ROLL}-day mean")
        style_axes(ax, title, ylabel)
        ax.set_ylim(bottom=0)
        add_event_markers(ax, cfg)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        xaxis_dates(ax)
        save_fig(fig, outdir / f"{tag}.png")


def plot_small_multiples(daily: pd.DataFrame, cfg: Config, platform: str, cats: list[str], suffix: str,
                         outdir, fname: str, title: str, ylabel: str) -> None:
    cols = [f"{c}{suffix}" for c in cats if f"{c}{suffix}" in daily.columns]
    if not cols:
        return
    d = daily.set_index("date")
    colours = category_colors(cfg)
    fig, axes = plt.subplots(len(cols), 1, figsize=(9.5, 2.3 * len(cols)), sharex=True, squeeze=False)
    for ax, cat, col in zip(axes[:, 0], [c for c in cats if f"{c}{suffix}" in daily.columns], cols):
        s = d[col].astype(float)
        ax.plot(d.index, s, color=colours[cat], linewidth=0.6, alpha=0.3, label="daily")
        ax.plot(d.index, rolling(s), color=colours[cat], linewidth=1.6, label=f"{ROLL}-day mean")
        style_axes(ax, category_label(cfg, cat), ylabel)
        ax.set_ylim(bottom=0)
        if suffix == "_proportion":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        add_event_markers(ax, cfg, label=ax is axes[0, 0])
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    xaxis_dates(axes[-1, 0])
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    save_fig(fig, outdir / fname)


def plot_weekly_accounts(weekly: pd.DataFrame, platform: str, outdir) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    ax.plot(weekly["week_start"], weekly["active_accounts"], color=PLATFORM_COLOR[platform], linewidth=1.6, label="active accounts")
    ax.plot(weekly["week_start"], weekly["active_accounts_with_hate"], color="#e34948", linewidth=1.6, label="active accounts with a hate label")
    ax.plot(weekly["week_start"], weekly["new_accounts"], color=PLATFORM_COLOR[platform], linewidth=1.2, linestyle="--", label="new accounts")
    style_axes(ax, f"{platform_label(platform)}: accounts per week", "number of accounts")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    xaxis_dates(ax)
    save_fig(fig, outdir / "weekly_accounts.png")


def plot_combined(weeklies: dict[str, pd.DataFrame], cfg: Config, outdir) -> pd.DataFrame:
    frames = []
    for p, w in weeklies.items():
        ww = w.copy()
        ww.insert(0, "platform", p)
        frames.append(ww)
    comb = pd.concat(frames, ignore_index=True)
    platforms = list(weeklies)
    measures = [("records", "All content", "posts per week"), (ANY_HATE, "Any hate category", "posts per week")]
    if ANY_EXTREMIST in comb.columns:
        measures.append((ANY_EXTREMIST, "Any extremist category", "posts per week"))
    for col, title, ylabel in measures:
        piv = comb.pivot(index="week_start", columns="platform", values=col).reindex(columns=platforms).fillna(0)
        fig, ax = plt.subplots(figsize=(9.5, 3.4))
        ax.stackplot(piv.index, [piv[p] for p in platforms], labels=[platform_label(p) for p in platforms],
                     colors=[PLATFORM_COLOR[p] for p in platforms], alpha=0.9, linewidth=0.5, edgecolor="white")
        style_axes(ax, f"{title} by platform (weekly)", ylabel)
        ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=len(platforms))
        xaxis_dates(ax)
        save_fig(fig, outdir / f"combined_{col}_weekly.png")
    return comb


# ---------------------------------------------------------------------------------------------
def run(cfg: Config, platforms: list[str] | None = None) -> dict[str, pd.DataFrame]:
    platforms = platforms or cfg.platforms
    tdir = table_dir(cfg, "daily")
    dailies, weeklies, summary = {}, {}, {}
    for p in platforms:
        df = load_analysis_records(cfg, p)
        daily = daily_table(df, cfg, p)
        weekly = weekly_table(df, daily, cfg)
        save_csv(daily, tdir / f"{p}_daily.csv")
        save_csv(weekly, tdir / f"{p}_weekly.csv")
        dailies[p], weeklies[p] = daily, weekly
        outdir = fig_dir(cfg, "daily", p)
        hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
        plot_all_content(daily, cfg, p, outdir)
        plot_small_multiples(daily, cfg, p, hate + [ANY_HATE], "", outdir, "hate_categories_daily_counts.png",
                             f"{platform_label(p)}: hate categories per day", "count of posts")
        plot_small_multiples(daily, cfg, p, hate + [ANY_HATE], "_proportion", outdir, "hate_categories_daily_proportions.png",
                             f"{platform_label(p)}: hate categories, proportion of posts", "proportion of posts")
        if llm:
            plot_small_multiples(daily, cfg, p, llm + [ANY_EXTREMIST], "", outdir, "extremist_categories_daily_counts.png",
                                 f"{platform_label(p)}: extremist categories per day", "count of posts")
            plot_small_multiples(daily, cfg, p, llm + [ANY_EXTREMIST], "_proportion", outdir, "extremist_categories_daily_proportions.png",
                                 f"{platform_label(p)}: extremist categories, proportion of labelled posts", "proportion of posts")
        if p == "x":
            plot_small_multiples(daily, cfg, p, hate + [ANY_HATE], "_circulation_adjusted", outdir,
                                 "hate_categories_daily_counts_circulation_adjusted.png",
                                 "X: hate categories per day, circulation-adjusted", "posts + reposts")
            if llm:
                plot_small_multiples(daily, cfg, p, llm + [ANY_EXTREMIST], "_circulation_adjusted", outdir,
                                     "extremist_categories_daily_counts_circulation_adjusted.png",
                                     "X: extremist categories per day, circulation-adjusted", "posts + reposts")
        plot_weekly_accounts(weekly, p, outdir)
        summary[p] = {
            "days": int(len(daily)), "observed_days": int(daily["observed_day"].sum()),
            "records": int(daily["records"].sum()),
            "hate_totals": {c: int(daily[c].sum()) for c in hate + [ANY_HATE]},
            "llm_labelled_records": int(daily["llm_labelled"].sum()),
            "extremist_totals": {c: int(daily[c].sum()) for c in llm + [ANY_EXTREMIST] if c in daily.columns},
            "peak_day_records": str(daily.loc[daily["records"].idxmax(), "date"].date()),
            "peak_day_any_hate": str(daily.loc[daily[ANY_HATE].idxmax(), "date"].date()),
        }
        print(f"daily: {p} {summary[p]['records']:,} records over {summary[p]['days']} days")
    comb = plot_combined(weeklies, cfg, fig_dir(cfg, "daily"))
    save_csv(comb, tdir / "combined_weekly.csv")
    save_json(summary, tdir / "daily_summary.json")
    return dailies


def main(argv=None) -> int:
    parser = add_config_argument(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--platform", action="append", help="restrict to a platform (repeatable)")
    args = parser.parse_args(argv)
    run(config_from_args(args), args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
