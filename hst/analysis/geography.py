"""Geographical analysis of X records (the only platform with self-reported location).

Inputs:  the X record table (region / city are profile-derived, not verified geolocation); the
         ``geography`` section of the config: population_csv (State,population; a built-in
         approximate table is used otherwise), min_users (suppress state cells with fewer accounts, default 1 = no suppression),
         min_city_users (default 20).
Method:  region -> canonical state name; each account is assigned ONE state (its most frequent,
         ties broken by the most recent record) and every record of the account inherits it.
Outputs: <work>/analysis/geography/state_summary.csv          accounts, records, circulation-adjusted
             volume, each category (count, proportion, per 1,000 accounts, per 100,000 residents)
         <work>/analysis/geography/city_summary.csv
         <work>/analysis/geography/location_audit.json, population_used.csv
         <work>/figures/geography/geo_accounts.png, geo_records.png, geo_circulation_adjusted_volume.png,
             geo_any_hate_proportion.png, geo_any_hate_per_1000_accounts.png,
             geo_any_hate_per_100k_population.png, geo_heatmap_proportion.png, geo_heatmap_per100k.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config, add_config_argument, config_from_args
from ..schema import ANY_EXTREMIST, ANY_HATE, LLM_LABELLED, label_col
from .common import llm_categories, load_analysis_records, supervised_categories
from .plotting import INK, INK2, PALETTE, SEQ_CMAP, category_label, fig_dir, save_csv, save_fig, save_json, table_dir

STATE_ORDER = ["New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia",
               "Tasmania", "Australian Capital Territory", "Northern Territory"]
STATE_ABBR = {"New South Wales": "NSW", "Victoria": "VIC", "Queensland": "QLD", "Western Australia": "WA",
              "South Australia": "SA", "Tasmania": "TAS", "Australian Capital Territory": "ACT", "Northern Territory": "NT"}
STATE_ALIASES = {k.lower(): k for k in STATE_ORDER}
STATE_ALIASES.update({"nsw": "New South Wales", "vic": "Victoria", "qld": "Queensland", "wa": "Western Australia",
                      "sa": "South Australia", "tas": "Tasmania", "act": "Australian Capital Territory", "nt": "Northern Territory"})
FALLBACK_POPULATION = {"New South Wales": 8444200, "Victoria": 6959200, "Queensland": 5560500, "Western Australia": 2951600,
                       "South Australia": 1878200, "Tasmania": 575700, "Australian Capital Territory": 475600, "Northern Territory": 252500}
UNKNOWN = "Unknown"


def canonical_state(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = str(value).strip().lower()
    return STATE_ALIASES.get(key)


def load_population(cfg: Config) -> tuple[dict, str]:
    path = cfg.section("geography").get("population_csv")
    if path:
        pop = pd.read_csv(cfg.resolve(path))
        return {str(s): float(p) for s, p in zip(pop["State"], pd.to_numeric(pop["population"]))}, f"user_provided:{path}"
    print("geography: using built-in approximate state populations; set geography.population_csv for reporting")
    return dict(FALLBACK_POPULATION), "built_in_approximate"


def account_state(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["state"].notna()].sort_values("day")
    counts = d.groupby(["author", "state"]).size().rename("n").reset_index()
    last = d.groupby(["author", "state"])["day"].max().rename("last_day").reset_index()
    counts = counts.merge(last, on=["author", "state"]).sort_values(["author", "n", "last_day"], ascending=[True, False, False])
    chosen = counts.drop_duplicates("author")[["author", "state"]].rename(columns={"state": "account_state"})
    n_states = counts.groupby("author")["state"].nunique().rename("n_states_seen")
    return chosen.merge(n_states, left_on="author", right_index=True)


def state_tables(df: pd.DataFrame, cfg: Config, population: dict, min_users: int) -> pd.DataFrame:
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    g = df.groupby("account_state")
    t = pd.DataFrame({"accounts": g["author"].nunique(), "records": g.size(),
                      "circulation_adjusted_volume": g["circulation_adjusted_volume"].sum(),
                      "llm_labelled": g[LLM_LABELLED].sum(), ANY_HATE: g[ANY_HATE].sum(), ANY_EXTREMIST: g[ANY_EXTREMIST].sum(min_count=1)})
    t["accounts_with_hate"] = df[df[ANY_HATE] == 1].groupby("account_state")["author"].nunique()
    t["accounts_with_hate"] = t["accounts_with_hate"].fillna(0).astype(int)
    for c in hate:
        t[c] = g[label_col(c)].sum()
        t[f"{c}_proportion"] = t[c] / t["records"].replace(0, np.nan)
    for c in llm:
        t[c] = g[label_col(c)].sum(min_count=1)
        t[f"{c}_proportion"] = t[c] / t["llm_labelled"].replace(0, np.nan)
    t["any_hate_proportion"] = t[ANY_HATE] / t["records"].replace(0, np.nan)
    t["any_extremist_proportion"] = t[ANY_EXTREMIST] / t["llm_labelled"].replace(0, np.nan)
    t["records_per_account"] = t["records"] / t["accounts"]
    for c in hate + llm + [ANY_HATE, ANY_EXTREMIST]:
        t[f"{c}_per_1000_accounts"] = t[c] / t["accounts"] * 1000
    t["population"] = pd.Series({s: population.get(s, np.nan) for s in t.index})
    for c in ["records", "accounts", "circulation_adjusted_volume"] + hate + llm + [ANY_HATE, ANY_EXTREMIST]:
        t[f"{c}_per_100k_population"] = t[c] / (t["population"] / 100000)
    located = t.index != UNKNOWN
    t["proportion_of_located_accounts"] = t["accounts"] / t.loc[located, "accounts"].sum()
    t["proportion_of_located_records"] = t["records"] / t.loc[located, "records"].sum()
    t["suppressed_small_cell"] = t["accounts"] < min_users
    t = t.reindex([s for s in STATE_ORDER if s in t.index] + [s for s in t.index if s not in STATE_ORDER])
    t.index.name = "state"
    return t.reset_index()


def barh(ax, labels, values, colour, title, xlabel):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colour, height=0.65)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.tick_params(labelsize=7)
    integer_like = all(float(v).is_integer() for v in values if np.isfinite(v))
    for yi, v in zip(y, values):
        if np.isfinite(v):
            ax.text(v, yi, f" {v:,.0f}" if integer_like else f" {v:,.1f}", va="center", fontsize=7, color=INK2)


def plot_bars(t: pd.DataFrame, figdir) -> None:
    s = t[(t["state"] != UNKNOWN) & (~t["suppressed_small_cell"])]
    labels = [STATE_ABBR.get(x, x) for x in s["state"]]
    panels = [("accounts", s["accounts"], PALETTE[0], "X: accounts by state", "number of accounts"),
              ("records", s["records"], PALETTE[0], "X: posts by state", "number of posts"),
              ("circulation_adjusted_volume", s["circulation_adjusted_volume"], PALETTE[0], "X: circulation-adjusted volume by state", "posts + reposts"),
              ("any_hate_proportion", s["any_hate_proportion"] * 100, PALETTE[1], "X: proportion of posts with a hate label, by state", "% of posts"),
              ("any_hate_per_1000_accounts", s["any_hate_per_1000_accounts"], PALETTE[1], "X: posts with a hate label per 1,000 accounts, by state", "per 1,000 accounts"),
              ("any_hate_per_100k_population", s["any_hate_per_100k_population"], PALETTE[1], "X: posts with a hate label per 100,000 residents, by state", "per 100,000 residents")]
    for tag, values, colour, title, xlabel in panels:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        barh(ax, labels, values.to_numpy(dtype=float), colour, title, xlabel)
        save_fig(fig, figdir / f"geo_{tag}.png", dpi=200)


def plot_heatmaps(t: pd.DataFrame, cfg: Config, cats: list[str], figdir) -> None:
    s = t[(t["state"] != UNKNOWN) & (~t["suppressed_small_cell"])]
    labels = [STATE_ABBR.get(x, x) for x in s["state"]]
    panels = [("proportion", [f"{c}_proportion" for c in cats], 100, "% of posts", "X: proportion of posts in each category, by state"),
              ("per100k", [f"{c}_per_100k_population" for c in cats], 1, "posts per 100,000 residents", "X: posts per 100,000 residents, by category and state")]
    for key, cols, scale, cbar, title in panels:
        cols = [c for c in cols if c in s.columns]
        if not cols or s.empty:
            continue
        mat = s[cols].to_numpy(dtype=float) * scale
        fig, ax = plt.subplots(figsize=(2 + 0.9 * len(cols), 1.5 + 0.4 * len(labels)))
        vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) and np.nanmax(mat) > 0 else 1
        im = ax.imshow(mat, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xticks(range(len(cols)), [category_label(cfg, c) for c in cats if f"{c}_{'proportion' if key == 'proportion' else 'per_100k_population'}" in s.columns], fontsize=6.5, rotation=40, ha="right")
        for (r, c), v in np.ndenumerate(mat):
            if np.isfinite(v):
                ax.text(c, r, f"{v:.1f}" if v < 1000 else f"{v:,.0f}", ha="center", va="center", fontsize=6, color=INK if v < 0.6 * vmax else "white")
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(cbar, fontsize=7)
        save_fig(fig, figdir / f"geo_heatmap_{key}.png", dpi=200)


def run(cfg: Config) -> pd.DataFrame | None:
    if "x" not in cfg.platforms:
        print("geography: only X carries location; nothing to do")
        return None
    geo = cfg.section("geography")
    min_users, min_city_users = int(geo.get("min_users", 1)), int(geo.get("min_city_users", 20))
    tdir, fdir = table_dir(cfg, "geography"), fig_dir(cfg, "geography")
    df = load_analysis_records(cfg, "x")
    df = df[df["author"].notna()].copy()
    df["state"] = df["region"].map(canonical_state)
    df["city"] = df["city"].astype("object").where(df["city"].notna(), None)
    population, pop_source = load_population(cfg)
    pd.DataFrame({"State": list(population), "population": list(population.values()), "source": pop_source}).to_csv(tdir / "population_used.csv", index=False)
    us = account_state(df)
    df = df.merge(us, on="author", how="left")
    df["account_state"] = df["account_state"].fillna(UNKNOWN)
    audit = {"records": int(len(df)), "accounts": int(df["author"].nunique()),
             "records_with_state_field": int(df["state"].notna().sum()), "records_with_city_field": int(df["city"].notna().sum()),
             "accounts_with_any_state": int(us["author"].nunique()), "accounts_with_multiple_states": int((us["n_states_seen"] > 1).sum()),
             "records_assigned_a_state": int((df["account_state"] != UNKNOWN).sum()),
             "account_state_rule": "most frequent state across the account's records; ties -> state of the most recent record",
             "location_type": "profile-derived region and city, not verified geolocation",
             "population_source": pop_source, "min_users_for_cell": min_users}
    save_json(audit, tdir / "location_audit.json")
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    t = state_tables(df, cfg, population, min_users)
    save_csv(t, tdir / "state_summary.csv")
    plot_bars(t, fdir)
    plot_heatmaps(t, cfg, hate + llm, fdir)
    c = df[df["city"].notna()]
    if len(c):
        gc = c.groupby(["account_state", "city"])
        city = pd.DataFrame({"accounts": gc["author"].nunique(), "records": gc.size(), ANY_HATE: gc[ANY_HATE].sum(),
                             "circulation_adjusted_volume": gc["circulation_adjusted_volume"].sum()})
        for cat in hate:
            city[cat] = gc[label_col(cat)].sum()
        city["any_hate_proportion"] = city[ANY_HATE] / city["records"].replace(0, np.nan)
        city["any_hate_per_1000_accounts"] = city[ANY_HATE] / city["accounts"] * 1000
        city = city.reset_index().rename(columns={"account_state": "state"})
        city["meets_min_accounts"] = city["accounts"] >= min_city_users
        city = city.sort_values("records", ascending=False)
    else:
        city = pd.DataFrame(columns=["state", "city", "accounts", "records", ANY_HATE, "any_hate_proportion", "meets_min_accounts"])
    save_csv(city, tdir / "city_summary.csv")
    print(f"geography: {audit['records_assigned_a_state']:,} of {audit['records']:,} X records assigned a state -> {tdir}")
    return t


def main(argv=None) -> int:
    parser = add_config_argument(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    run(config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
