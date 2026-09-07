"""Event-based analysis: descriptive before/after windows and interrupted time series (ITS).

Inputs:  the platform record tables; the ``events`` list of the config (id, date, label);
         ``event_window_days`` (descriptive window, default 7), ``its_pre_days`` / ``its_post_days``
         (ITS fit window, default 21 / 14).

Part A, descriptive.  For each event and platform: the ``event_window_days`` before the event, the
event day, and the ``event_window_days`` after it.  Per period: posts, each category (count and
proportion), active accounts, new accounts (first appearance anywhere in the platform's records
falls inside the period) and engagement sums.  Post / pre ratios of daily means and proportion
differences are tabulated.

Part B, ITS.  On the daily series over ``its_pre_days`` before to ``its_post_days`` after the event,
a segmented regression is fitted: intercept, pre-event trend, day-of-week terms, a lag term, a
level step at the event and a post-event slope.  Counts use a negative-binomial model (Poisson if
it does not converge); proportions use a binomial GLM.  ``estimate`` is exp(step): the rate ratio
(counts) or odds ratio (proportions) at the event.  The figure shows observed values, the fitted
line and the pre-event trend extrapolated as "expected if the event had not happened".

Outputs: <work>/analysis/events/event_windows_descriptive.csv, event_pre_post_comparison.csv,
         event_its_results.csv, event_analysis_summary.json
         <work>/figures/events/<platform>/event_window_<id>_<measure>.png, event_ratio_counts.png,
         event_its_<id>_<outcome>.png
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from ..config import Config, add_config_argument, config_from_args
from ..schema import ANY_EXTREMIST, ANY_HATE, LLM_LABELLED, label_col
from .common import ENGAGEMENT_COLUMNS, llm_categories, load_analysis_records, supervised_categories
from .daily import daily_table
from .plotting import (INK, INK2, PLATFORM_COLOR, category_colors, category_label, fig_dir, platform_label,
                       save_csv, save_fig, save_json, style_axes, table_dir)

MIN_DAYS = 14          # smallest ITS window that is fitted
MIN_NONZERO = 8        # smallest number of non-zero days for a count model


def event_units(cfg: Config) -> list[dict]:
    w = int(cfg.get("event_window_days", 7))
    units = []
    for e in cfg.events().itertuples():
        d = pd.Timestamp(e.date)
        units.append({"id": str(e.id), "label": str(e.label), "date": d,
                      "pre_start": d - pd.Timedelta(days=w), "pre_end": d - pd.Timedelta(days=1),
                      "post_start": d + pd.Timedelta(days=1), "post_end": d + pd.Timedelta(days=w)})
    return units


# ============================================================================================
# Part A - descriptive windows
# ============================================================================================
def period_metrics(df: pd.DataFrame, first_seen: pd.Series, cfg: Config, start, end) -> dict:
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    sub = df[(df["day"] >= start) & (df["day"] <= end)]
    days = (end - start).days + 1
    m = {"days": days, "records": int(len(sub)), "active_accounts": int(sub["author"].nunique()),
         "llm_labelled": int(sub[LLM_LABELLED].sum()),
         "records_circulation_adjusted": float(sub["circulation_adjusted_volume"].sum())}
    accounts = sub["author"].dropna().unique()
    fs = first_seen.reindex(accounts)
    m["new_accounts"] = int(((fs >= start) & (fs <= end)).sum())
    for cat in hate + [ANY_HATE]:
        m[cat] = int(sub[label_col(cat) if cat != ANY_HATE else ANY_HATE].sum())
        m[f"{cat}_proportion"] = m[cat] / m["records"] if m["records"] else np.nan
    for cat in llm + [ANY_EXTREMIST]:
        col = label_col(cat) if cat != ANY_EXTREMIST else ANY_EXTREMIST
        m[cat] = int(sub[col].fillna(0).sum())
        m[f"{cat}_proportion"] = m[cat] / m["llm_labelled"] if m["llm_labelled"] else np.nan
    for c in ENGAGEMENT_COLUMNS:
        if sub[c].notna().any():
            m[f"{c}_sum"] = float(sub[c].sum())
    return m


def descriptive_windows(records: dict[str, pd.DataFrame], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, comp_rows = [], []
    for p, df in records.items():
        hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
        first_seen = df.groupby("author")["day"].min()
        for u in event_units(cfg):
            per = {}
            for period, (s, e) in {"pre": (u["pre_start"], u["pre_end"]), "event": (u["date"], u["date"]),
                                   "post": (u["post_start"], u["post_end"])}.items():
                m = period_metrics(df, first_seen, cfg, s, e)
                per[period] = m
                rows.append({"platform": p, "event_id": u["id"], "event_label": u["label"], "period": period,
                             "start": s.date(), "end": e.date(), **m})
            pre, ev, post = per["pre"], per["event"], per["post"]
            c = {"platform": p, "event_id": u["id"], "event_label": u["label"]}
            measures = ["records", "active_accounts", "new_accounts"] + hate + [ANY_HATE] + llm + [ANY_EXTREMIST]
            measures += [k for k in pre if k.endswith("_sum")]
            for meas in measures:
                pre_v, ev_v, post_v = pre.get(meas, np.nan), ev.get(meas, np.nan), post.get(meas, np.nan)
                c[f"{meas}_pre_daily"] = pre_v / pre["days"] if pd.notna(pre_v) else np.nan
                c[f"{meas}_event_daily"] = ev_v / ev["days"] if pd.notna(ev_v) else np.nan
                c[f"{meas}_post_daily"] = post_v / post["days"] if pd.notna(post_v) else np.nan
                c[f"{meas}_post_pre_ratio"] = (c[f"{meas}_post_daily"] / c[f"{meas}_pre_daily"]
                                               if pd.notna(pre_v) and pre_v > 0 else np.nan)
            for meas in hate + [ANY_HATE] + llm + [ANY_EXTREMIST]:
                c[f"{meas}_proportion_pre"] = pre[f"{meas}_proportion"]
                c[f"{meas}_proportion_event"] = ev[f"{meas}_proportion"]
                c[f"{meas}_proportion_post"] = post[f"{meas}_proportion"]
                c[f"{meas}_proportion_diff_pp"] = (post[f"{meas}_proportion"] - pre[f"{meas}_proportion"]) * 100
            comp_rows.append(c)
    return pd.DataFrame(rows), pd.DataFrame(comp_rows)


def plot_event_windows(records: dict[str, pd.DataFrame], cfg: Config) -> None:
    measures = [("records", "all posts"), (ANY_HATE, "posts with a hate label"), (ANY_EXTREMIST, "posts with an extremist label")]
    for u in event_units(cfg):
        for p, df in records.items():
            outdir = fig_dir(cfg, "events", p)
            sub = df[(df["day"] >= u["pre_start"]) & (df["day"] <= u["post_end"])]
            days = pd.date_range(u["pre_start"], u["post_end"], freq="D")
            for meas, title in measures:
                if meas == "records":
                    s = sub.groupby("day").size()
                elif meas == ANY_HATE:
                    s = sub.groupby("day")[ANY_HATE].sum()
                else:
                    if sub[ANY_EXTREMIST].notna().sum() == 0:
                        continue
                    s = sub.groupby("day")[ANY_EXTREMIST].sum(min_count=1)
                s = s.reindex(days).fillna(0)
                colours = [PLATFORM_COLOR[p] if d == u["date"] else "#c9d3df" for d in days]
                fig, ax = plt.subplots(figsize=(7.5, 3.0))
                ax.bar(days, s.to_numpy(), color=colours, width=0.8, edgecolor="white", linewidth=0.5)
                style_axes(ax, f"{u['id']} {platform_label(p)}: {title}", "count of posts")
                for bound in (u["date"] - pd.Timedelta(hours=12), u["date"] + pd.Timedelta(hours=12)):
                    ax.axvline(bound, color="#8a8984", linestyle=(0, (4, 3)), linewidth=0.9, zorder=3)
                ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
                ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
                ax.tick_params(axis="x", labelsize=7, rotation=45)
                save_fig(fig, outdir / f"event_window_{u['id']}_{meas}.png")


def plot_ratio_heatmap(comp: pd.DataFrame, cfg: Config, platforms: list[str]) -> None:
    units = [u["id"] for u in event_units(cfg)]
    for p in platforms:
        sub = comp[comp["platform"] == p].set_index("event_id").reindex(units)
        measures = [c[: -len("_post_pre_ratio")] for c in sub.columns if c.endswith("_post_pre_ratio")]
        measures = [m for m in measures if not m.endswith("_sum")]
        if not measures:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.log2(sub[[f"{m}_post_pre_ratio" for m in measures]].astype(float).to_numpy())
        fig, ax = plt.subplots(figsize=(1.2 + 0.6 * len(units), 0.42 * len(measures) + 1.5))
        im = ax.imshow(mat.T, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-2.5, vmax=2.5), aspect="auto")
        ax.set_xticks(range(len(units)), units, fontsize=7, rotation=45)
        ax.set_yticks(range(len(measures)), [category_label(cfg, m) if m not in ("records", "active_accounts", "new_accounts")
                                             else m.replace("_", " ") for m in measures], fontsize=7)
        ax.set_title(f"{platform_label(p)}: daily mean after / before each event", fontsize=9, loc="left", color=INK)
        for (r, c), v in np.ndenumerate(mat.T):
            if np.isfinite(v):
                ax.text(c, r, f"{2 ** v:.2f}", ha="center", va="center", fontsize=6, color=INK if abs(v) < 1.4 else "white")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("ratio (log scale, white = no change)", fontsize=7)
        save_fig(fig, fig_dir(cfg, "events", p) / "event_ratio_counts.png", dpi=200)


# ============================================================================================
# Part B - interrupted time series with a counterfactual
# ============================================================================================
def _design(dates: pd.Series, event: pd.Timestamp, lag: np.ndarray | None) -> pd.DataFrame:
    t = (dates - dates.min()).dt.days.to_numpy() / 7.0
    X = pd.DataFrame({"const": 1.0, "time": t,
                      "post": (dates >= event).astype(float).to_numpy(),
                      "time_after": np.clip((dates - event).dt.days.to_numpy(), 0, None) / 7.0}, index=dates.to_numpy())
    for wd in range(1, 7):
        X[f"weekday_{wd}"] = (dates.dt.dayofweek == wd).astype(float).to_numpy()
    if lag is not None:
        X["lag"] = lag
    keep = [c for c in X.columns if c == "const" or X[c].nunique() > 1]
    return X[keep]


def _counterfactual(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    Xc["post"] = 0.0
    if "time_after" in Xc.columns:
        Xc["time_after"] = 0.0
    return Xc


def _fit_count(y: np.ndarray, X: pd.DataFrame):
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=300, method="bfgs")
            if r.mle_retvals.get("converged", False) and np.isfinite(r.params).all() and np.isfinite(r.bse).all():
                return r, "negative_binomial"
        except Exception:
            pass
        try:
            return sm.GLM(y, X, family=sm.families.Poisson()).fit(), "poisson"
        except Exception:
            return None, "failed"


def _fit_proportion(succ: np.ndarray, trials: np.ndarray, X: pd.DataFrame):
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        endog = np.column_stack([succ, np.maximum(trials - succ, 0)])
        try:
            return sm.GLM(endog, X, family=sm.families.Binomial()).fit(), "binomial"
        except Exception:
            return None, "failed"


def _step_row(res, family: str) -> dict:
    ci = res.conf_int()
    lo, hi = float(ci.loc["post"].iloc[0]), float(ci.loc["post"].iloc[1])
    return {"model": family, "estimate": float(np.exp(res.params["post"])), "ci_low": float(np.exp(lo)),
            "ci_high": float(np.exp(hi)), "p_value": float(res.pvalues["post"])}


def _draw(dates, obs, fitted, cf, event, colour, title, ylabel, path, as_percent=False) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.scatter(dates, obs, s=11, color=colour, alpha=0.45, label="observed", zorder=2)
    ax.plot(dates, cf, color="#8a8984", linestyle=(0, (4, 3)), linewidth=1.5, label="expected if no event", zorder=3)
    ax.plot(dates, fitted, color=colour, linewidth=1.9, label="fitted", zorder=4)
    ax.axvline(event, color=INK2, linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
    ax.set_ylim(bottom=0)
    if as_percent:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    style_axes(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    save_fig(fig, path)


def its_for_event(daily: pd.DataFrame, cfg: Config, platform: str, unit: dict, hate: list[str], llm: list[str],
                  figdir, draw: bool = True) -> list[dict]:
    pre, post = int(cfg.get("its_pre_days", 21)), int(cfg.get("its_post_days", 14))
    event = unit["date"]
    w_s, w_e = max(event - pd.Timedelta(days=pre), cfg.study_start), min(event + pd.Timedelta(days=post), cfg.study_end)
    d = daily[(daily["date"] >= w_s) & (daily["date"] <= w_e)].copy()
    d = d[d["observed_day"].astype(bool) | (d["records"] > 0)] if d["observed_day"].sum() < len(d) * 0.5 else d
    dates = d["date"].reset_index(drop=True)
    meta = {"platform": platform, "event_id": unit["id"], "event_label": unit["label"], "n_days": int(len(d)),
            "window_start": w_s.date(), "window_end": w_e.date()}
    if len(d) < MIN_DAYS or dates.min() > event or dates.max() < event:
        return [{**meta, "outcome": "all", "outcome_type": "count", "model": "skipped: window too short",
                 "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}]
    colours = category_colors(cfg)
    rows = []
    count_outcomes = [("records", "all", "all posts", PLATFORM_COLOR[platform]),
                      (ANY_HATE, ANY_HATE, "posts with a hate label", PLATFORM_COLOR[platform])]
    if llm:
        count_outcomes.append((ANY_EXTREMIST, ANY_EXTREMIST, "posts with an extremist label", PLATFORM_COLOR[platform]))
    count_outcomes += [(c, c, category_label(cfg, c), colours[c]) for c in hate + llm]
    for col, name, title, colour in count_outcomes:
        y = d[col].astype(float).to_numpy()
        lag = np.log1p(np.concatenate([[y[0]], y[:-1]]))
        if (y > 0).sum() < MIN_NONZERO:
            rows.append({**meta, "outcome": name, "outcome_type": "count", "model": "skipped: too few non-zero days",
                         "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan})
            continue
        X = _design(dates, event, lag)
        if "post" not in X.columns:
            continue
        res, family = _fit_count(y, X)
        if res is None:
            rows.append({**meta, "outcome": name, "outcome_type": "count", "model": "failed",
                         "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan})
            continue
        rows.append({**meta, "outcome": name, "outcome_type": "count", **_step_row(res, family)})
        if draw and (name in ("all", ANY_HATE, ANY_EXTREMIST) or name in hate):
            _draw(dates, y, np.asarray(res.predict(X)), np.asarray(res.predict(_counterfactual(X))), event, colour,
                  f"{unit['id']} {platform_label(platform)}: {title}", "count of posts",
                  figdir / f"event_its_{unit['id']}_{name}.png")
    prop_outcomes = [(c, "records") for c in hate + [ANY_HATE]] + [(c, "llm_labelled") for c in llm + [ANY_EXTREMIST]]
    for col, den in prop_outcomes:
        if col not in d.columns:
            continue
        succ, trials = d[col].astype(float).to_numpy(), d[den].astype(float).to_numpy()
        ok = trials > 0
        if ok.sum() < MIN_DAYS or (succ[ok] > 0).sum() < MIN_NONZERO:
            rows.append({**meta, "outcome": col, "outcome_type": "proportion", "model": "skipped: too few non-zero days",
                         "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan})
            continue
        prop = np.divide(succ, trials, out=np.full_like(succ, np.nan), where=ok)
        lag = np.concatenate([[np.nanmean(prop[:3])], prop[:-1]])
        lag = np.where(np.isfinite(lag), lag, np.nanmean(prop))
        X = _design(dates[ok], event, lag[ok])
        if "post" not in X.columns:
            continue
        res, family = _fit_proportion(succ[ok], trials[ok], X)
        if res is None or not np.isfinite(res.params).all():
            continue
        rows.append({**meta, "outcome": col, "outcome_type": "proportion", **_step_row(res, family)})
        if draw and col in hate:
            _draw(dates[ok], prop[ok], np.asarray(res.predict(X)), np.asarray(res.predict(_counterfactual(X))), event,
                  colours.get(col, PLATFORM_COLOR[platform]), f"{unit['id']} {platform_label(platform)}: {category_label(cfg, col)}, proportion of posts",
                  "proportion of posts", figdir / f"event_its_{unit['id']}_{col}_proportion.png", as_percent=True)
    return rows


def run(cfg: Config, platforms: list[str] | None = None) -> pd.DataFrame:
    platforms = platforms or cfg.platforms
    tdir = table_dir(cfg, "events")
    records = {p: load_analysis_records(cfg, p) for p in platforms}
    windows, comp = descriptive_windows(records, cfg)
    save_csv(windows, tdir / "event_windows_descriptive.csv")
    save_csv(comp, tdir / "event_pre_post_comparison.csv")
    plot_event_windows(records, cfg)
    if len(comp):
        plot_ratio_heatmap(comp, cfg, platforms)
    rows = []
    for p, df in records.items():
        daily = daily_table(df, cfg, p)
        hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
        figdir = fig_dir(cfg, "events", p)
        for u in event_units(cfg):
            rows += its_for_event(daily, cfg, p, u, hate, llm, figdir)
    cols = ["platform", "event_id", "event_label", "outcome", "outcome_type", "model", "estimate", "ci_low", "ci_high",
            "p_value", "n_days", "window_start", "window_end"]
    its = pd.DataFrame(rows, columns=cols)
    save_csv(its, tdir / "event_its_results.csv")
    save_json({"event_window_days": int(cfg.get("event_window_days", 7)), "its_pre_days": int(cfg.get("its_pre_days", 21)),
               "its_post_days": int(cfg.get("its_post_days", 14)),
               "estimate": "exp(level step at the event): rate ratio for counts, odds ratio for proportions",
               "new_account_definition": "first record of the account anywhere in the platform's records falls inside the period",
               "events": [{"id": u["id"], "date": str(u["date"].date()), "label": u["label"]} for u in event_units(cfg)],
               "its_rows": int(len(its))}, tdir / "event_analysis_summary.json")
    print(f"events: {len(comp)} event x platform comparisons, {len(its)} ITS rows -> {tdir}")
    return its


def main(argv=None) -> int:
    parser = add_config_argument(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--platform", action="append")
    args = parser.parse_args(argv)
    run(config_from_args(args), args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
