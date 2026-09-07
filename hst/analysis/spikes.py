"""Spike detection on the daily series of hateful posts.

Inputs:  the platform record tables; parameters from the ``spikes`` section of the config
         (top, window_days, baseline_days, z_threshold, min_count, min_separation_days,
         volume_fraction, exclude_after_gap_days, min_baseline_observed_days, samples_per_period).
Method:  for every day, the mean and standard deviation of the previous ``baseline_days`` observed
         days give a z-score; days after a collection gap and days with a thin baseline are not
         eligible.  ``top`` spikes are chosen, half by z-score (relative anomaly) and half by absolute
         volume, at least ``min_separation_days`` apart.
Outputs: <work>/analysis/spikes/<platform>/spikes_detected.csv
         <work>/analysis/spikes/<platform>/spike_window_summary.csv   pre / spike-day / post totals
         <work>/analysis/spikes/<platform>/spike_text_samples.csv     most repeated hateful texts per period
         <work>/figures/spikes/<platform>/spike_<date>.png            posts and hateful posts around the spike
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config, add_config_argument, config_from_args
from ..schema import ANY_HATE, normalize_text
from .common import load_analysis_records
from .daily import daily_table
from .plotting import INK2, PLATFORM_COLOR, fig_dir, platform_label, save_csv, save_fig, style_axes, table_dir, xaxis_dates

DEFAULTS = {"top": 8, "window_days": 7, "baseline_days": 14, "z_threshold": 3.0, "min_count": 10,
            "min_separation_days": 15, "volume_fraction": 0.5, "exclude_after_gap_days": 7,
            "min_baseline_observed_days": 10, "samples_per_period": 10}


def spike_params(cfg: Config) -> dict:
    p = dict(DEFAULTS)
    p.update({k: v for k, v in cfg.section("spikes").items() if k in DEFAULTS})
    return p


def detect_spikes(daily: pd.DataFrame, params: dict) -> pd.DataFrame:
    d = daily.set_index("date").copy()
    observed = d["observed_day"].astype(bool)
    run, consecutive = 0, []
    for o in observed.tolist():
        run = run + 1 if o else 0
        consecutive.append(run)
    d["consecutive_observed_days"] = consecutive
    s = d[ANY_HATE].where(observed).astype(float)
    min_periods = max(3, min(params["min_baseline_observed_days"], params["baseline_days"]))
    baseline_mean = s.shift(1).rolling(params["baseline_days"], min_periods=min_periods).mean()
    baseline_std = s.shift(1).rolling(params["baseline_days"], min_periods=min_periods).std()
    baseline_observed = observed.astype(int).shift(1).rolling(params["baseline_days"], min_periods=1).sum()
    d["baseline_mean"] = baseline_mean
    d["baseline_std"] = baseline_std
    d["baseline_observed_days"] = baseline_observed.fillna(0).astype(int)
    d["z_score"] = ((s - baseline_mean) / baseline_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    eligible = observed & (d["consecutive_observed_days"] > params["exclude_after_gap_days"]) \
        & (d["baseline_observed_days"] >= params["min_baseline_observed_days"])
    cand = d[eligible & (d[ANY_HATE] >= params["min_count"]) & (d["z_score"] >= params["z_threshold"])].copy()
    if cand.empty:
        warnings.warn("no day met the z-score threshold; falling back to the largest eligible days")
        cand = d[eligible & (d[ANY_HATE] >= params["min_count"])].copy()
    by_anomaly = cand.sort_values(["z_score", ANY_HATE], ascending=False)
    by_volume = cand.sort_values([ANY_HATE, "z_score"], ascending=False)
    anomaly_rank = {date: r for r, date in enumerate(by_anomaly.index, start=1)}
    volume_rank = {date: r for r, date in enumerate(by_volume.index, start=1)}
    top = int(params["top"])
    volume_slots = min(max(int(round(top * params["volume_fraction"])), 0), top)
    selected: list[pd.Timestamp] = []
    basis: dict[pd.Timestamp, str] = {}

    def add(ranked: pd.DataFrame, slots: int, name: str) -> None:
        added = 0
        for date in ranked.index:
            if date in basis:
                basis[date] += "+" + name
                continue
            if all(abs((date - x).days) >= params["min_separation_days"] for x in selected):
                selected.append(date)
                basis[date] = name
                added += 1
            if added >= slots or len(selected) >= top:
                break

    add(by_anomaly, top - volume_slots, "relative_anomaly")
    add(by_volume, volume_slots, "absolute_volume")
    if len(selected) < top:
        add(by_anomaly, top - len(selected), "anomaly_fill")
    out = d.loc[selected].reset_index()
    out.insert(1, "selection_basis", out["date"].map(basis))
    out.insert(2, "anomaly_rank", out["date"].map(anomaly_rank))
    out.insert(3, "volume_rank", out["date"].map(volume_rank))
    keep = ["date", "selection_basis", "anomaly_rank", "volume_rank", "records", ANY_HATE, "baseline_mean",
            "baseline_std", "baseline_observed_days", "z_score", "consecutive_observed_days"]
    return out[keep].sort_values("date").reset_index(drop=True)


def sample_period_texts(df: pd.DataFrame, start, end, period: str, n: int) -> list[dict]:
    sub = df[(df["day"] >= start) & (df["day"] <= end) & (df[ANY_HATE] == 1)].copy()
    if sub.empty:
        return []
    sub["text_key"] = sub["text"].map(normalize_text)
    rows = []
    grouped = sub.groupby("text_key")
    for key, count in grouped.size().sort_values(ascending=False).head(n).items():
        g = grouped.get_group(key)
        rows.append({"period": period, "start": start.date(), "end": end.date(), "count": int(count),
                     "example_date": str(g["day"].iloc[0].date()),
                     "example_author": str(g["author"].dropna().iloc[0]) if g["author"].notna().any() else "",
                     "example_url": str(g["url"].dropna().iloc[0]) if g["url"].notna().any() else "",
                     "example_text": str(g["text"].iloc[0])})
    return rows


def spike_reports(df: pd.DataFrame, daily: pd.DataFrame, spikes: pd.DataFrame, params: dict, platform: str,
                  figdir) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = daily.set_index("date")
    summary_rows, sample_rows = [], []
    for spike in spikes.itertuples():
        day = pd.Timestamp(spike.date).normalize()
        xmin, xmax = day - pd.Timedelta(days=params["window_days"]), day + pd.Timedelta(days=params["window_days"])
        window = d.loc[(d.index >= xmin) & (d.index <= xmax)]
        fig, ax1 = plt.subplots(figsize=(9, 3.6))
        ax1.bar(window.index, window["records"], width=0.75, color="#d9d8d2", label="all posts", zorder=2)
        ax2 = ax1.twinx()
        ax2.plot(window.index, window[ANY_HATE], marker="o", color=PLATFORM_COLOR[platform], linewidth=1.5,
                 label="posts with a hate label", zorder=3)
        ax1.axvline(day, linestyle="--", linewidth=1, color=INK2, label="spike day")
        style_axes(ax1, f"{platform_label(platform)}: activity around the spike of {day.date()}", "count of posts")
        ax2.set_ylabel("count of posts with a hate label", color=INK2, fontsize=8)
        ax2.tick_params(colors=INK2, labelsize=8)
        for s in ("top",):
            ax2.spines[s].set_visible(False)
        xaxis_dates(ax1, span_days=2 * params["window_days"] + 1)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, frameon=False)
        save_fig(fig, figdir / f"spike_{day.date()}.png")
        periods = {"pre": (xmin, day - pd.Timedelta(days=1)), "spike_day": (day, day), "post": (day + pd.Timedelta(days=1), xmax)}
        for period, (s, e) in periods.items():
            pc = d.loc[(d.index >= s) & (d.index <= e)]
            row = {"spike_date": day.date(), "period": period, "start": s.date(), "end": e.date(), "days": int(len(pc))}
            for col in ["records", ANY_HATE]:
                row[f"{col}_total"] = int(pc[col].sum())
                row[f"{col}_daily_mean"] = float(pc[col].mean()) if len(pc) else 0.0
            summary_rows.append(row)
            for sample in sample_period_texts(df, s, e, period, params["samples_per_period"]):
                sample_rows.append({"spike_date": day.date(), **sample})
    sample_cols = ["spike_date", "period", "start", "end", "count", "example_date", "example_author", "example_url", "example_text"]
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows, columns=sample_cols)


def run(cfg: Config, platforms: list[str] | None = None) -> dict[str, pd.DataFrame]:
    platforms = platforms or cfg.platforms
    params = spike_params(cfg)
    results = {}
    for p in platforms:
        df = load_analysis_records(cfg, p)
        daily = daily_table(df, cfg, p)
        spikes = detect_spikes(daily, params)
        tdir, fdir = table_dir(cfg, "spikes", p), fig_dir(cfg, "spikes", p)
        save_csv(spikes, tdir / "spikes_detected.csv")
        summary, samples = spike_reports(df, daily, spikes, params, p, fdir)
        save_csv(summary, tdir / "spike_window_summary.csv")
        save_csv(samples, tdir / "spike_text_samples.csv")
        results[p] = spikes
        print(f"spikes: {p} {len(spikes)} spike day(s) -> {tdir}")
    return results


def main(argv=None) -> int:
    parser = add_config_argument(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--platform", action="append")
    args = parser.parse_args(argv)
    run(config_from_args(args), args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
