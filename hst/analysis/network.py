"""Interaction networks per platform.

Inputs:  the platform record tables; the ``network`` section of the config (min_degree, max_nodes,
         optional before_after windows).
Edges (directed, weighted by the number of interactions):
    author -> account replied to            (reply_to_author)
    author -> author of the parent post     (parent_author; comments and reposts)
    author -> mentioned account             (mentions)
    author -> forwarded-from account        (forwarded_from; Telegram)
Per platform, over the study period: communities (Louvain, seeded), size / density / reciprocity /
modularity / degree centralisation, per-account centralities and target profile (main target = the
category most of the account's labelled posts fall in), community content profiles, specialist vs
generalist accounts, overlap between hate targets.

Drawing.  The connected backbone (2-core, top ``max_nodes`` by PageRank) is laid out with a
force-directed algorithm: DrL through python-igraph when available (seeded), otherwise the
Fruchterman-Reingold spring layout from networkx.  A compaction step then pulls every community
towards the centre and a rein-in step compresses the few far-flung accounts, so communities read as
distinct clusters that fill the frame.  Position and orientation are illustrative only.
When ``before_after`` windows are configured, the two snapshots are drawn from ONE shared reference
layout (the union of both windows) with the same community colours, so they can be compared.

Outputs: <work>/analysis/network/<platform>/edges.csv.gz, metrics.json, nodes.csv,
         community_profiles.csv, specialist_generalist.csv, category_overlap.csv, before_after_metrics.csv
         <work>/figures/network/<platform>/network_by_community.png, network_by_main_target.png,
         community_by_category_heatmap.png, hate_category_overlap_heatmap.png,
         hate_category_count_distribution.png, hate_category_main_target_distribution.png,
         network_before.png, network_after.png
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import Config, add_config_argument, config_from_args
from ..schema import ANY_EXTREMIST, ANY_HATE, LLM_LABELLED, label_col
from .common import llm_categories, load_analysis_records, supervised_categories
from .plotting import (INK, INK2, NEUTRAL, PALETTE, SEQ_CMAP, category_colors, category_label, fig_dir, platform_label,
                       save_csv, save_fig, save_json, table_dir)

COMMUNITY_PALETTE = PALETTE[:8]
MIN_ACCOUNT_RECORDS = 5       # records needed for an account-level target profile
DOMINANT_SHARE = 0.5          # proportion of an account's hate labels needed for a "main target"
BETWEENNESS_SAMPLE = 400
MULTI = "multiple targets"
NONE = "no hate label"
FEW = "too few records"


# ============================================================================================
# edges and graphs
# ============================================================================================
def build_edges(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    src = df["author"]
    ok = src.notna()
    parts = []

    def add(mask, target: pd.Series, kind: str) -> None:
        m = ok & mask & target.notna() & (target != src)
        if m.any():
            parts.append(pd.DataFrame({"source": src[m], "target": target[m], "edge_type": kind, "record_id": df.loc[m, "record_id"]}))

    add(df["content_type"].isin(["reply", "comment", "post", "repost"]), df["reply_to_author"], "reply")
    parent = df["parent_author"].where(df["parent_author"] != df["reply_to_author"].fillna(""))
    add(df["content_type"].isin(["comment", "repost", "reply"]), parent, "parent")
    if "forwarded_from" in df.columns:
        add(pd.Series(True, index=df.index), df["forwarded_from"], "forward")
    men = df.loc[ok & df["mentions"].notna(), ["record_id", "author", "reply_to_author", "mentions"]]
    if len(men):
        ex = men.assign(target=men["mentions"].astype(str).str.split("|")).explode("target")
        ex = ex[(ex["target"].notna()) & (ex["target"] != "") & (ex["target"] != ex["author"])
                & (ex["target"] != ex["reply_to_author"].fillna(""))]
        parts.append(pd.DataFrame({"source": ex["author"], "target": ex["target"], "edge_type": "mention", "record_id": ex["record_id"]}))
    cols = ["source", "target", "edge_type", "record_id"]
    edges = pd.concat([p for p in parts if len(p)], ignore_index=True).drop_duplicates() if parts else pd.DataFrame(columns=cols)
    meta = ["record_id", "day", ANY_HATE, ANY_EXTREMIST] + [label_col(c) for c in hate + llm]
    edges = edges.merge(df[meta], on="record_id", how="left")
    return edges


def make_graph(edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    if len(edges):
        w = edges.groupby(["source", "target"]).size().rename("weight").reset_index()
        G.add_weighted_edges_from(w.itertuples(index=False, name=None))
        G.remove_edges_from(nx.selfloop_edges(G))
    return G


def undirected(G: nx.DiGraph) -> nx.Graph:
    U = G.to_undirected()
    for u, v, d in U.edges(data=True):
        d["weight"] = G.get_edge_data(u, v, {}).get("weight", 0) + G.get_edge_data(v, u, {}).get("weight", 0)
    return U


def degree_centralization(U: nx.Graph) -> float:
    n = U.number_of_nodes()
    if n < 3:
        return float("nan")
    deg = np.array([d for _, d in U.degree()], dtype=float)
    return float((deg.max() - deg).sum() / ((n - 1) * (n - 2)))


def communities_and_modularity(U: nx.Graph, seed: int = 42) -> tuple[dict, float]:
    if U.number_of_edges() == 0:
        return {n: i for i, n in enumerate(U.nodes())}, float("nan")
    comms = sorted(nx.community.louvain_communities(U, weight="weight", seed=seed), key=len, reverse=True)
    part = {n: i for i, c in enumerate(comms) for n in c}
    return part, float(nx.community.modularity(U, comms, weight="weight"))


# ============================================================================================
# account and community profiles
# ============================================================================================
def account_profile(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    g = df.groupby("author")
    prof = pd.DataFrame({"records": g.size(), "llm_labelled": g[LLM_LABELLED].sum(),
                         ANY_HATE: g[ANY_HATE].sum(), ANY_EXTREMIST: g[ANY_EXTREMIST].sum(min_count=1)})
    for c in hate:
        prof[c] = g[label_col(c)].sum()
        prof[f"{c}_proportion"] = prof[c] / prof["records"].replace(0, np.nan)
    for c in llm:
        prof[c] = g[label_col(c)].sum(min_count=1)
        prof[f"{c}_proportion"] = prof[c] / prof["llm_labelled"].replace(0, np.nan)
    prof["any_hate_proportion"] = prof[ANY_HATE] / prof["records"].replace(0, np.nan)
    prof["any_extremist_proportion"] = prof[ANY_EXTREMIST] / prof["llm_labelled"].replace(0, np.nan)
    counts = prof[hate].fillna(0) if hate else pd.DataFrame(index=prof.index)
    total = counts.sum(axis=1) if hate else pd.Series(0, index=prof.index)
    prof["hate_label_total"] = total
    prof["n_hate_categories"] = (counts > 0).sum(axis=1) if hate else 0
    if hate:
        top_cat = counts.idxmax(axis=1)
        top_share = counts.max(axis=1) / total.replace(0, np.nan)
    else:
        top_cat, top_share = pd.Series(None, index=prof.index, dtype="object"), pd.Series(np.nan, index=prof.index)
    eligible = prof["records"] >= MIN_ACCOUNT_RECORDS
    prof["top_category"] = top_cat.where(total > 0)
    prof["top_category_proportion"] = top_share
    prof["profile_eligible"] = eligible
    prof["main_target"] = np.where(~eligible, FEW, np.where(total == 0, NONE, np.where(top_share >= DOMINANT_SHARE, top_cat, MULTI)))
    return prof


def analyse_graph(edges: pd.DataFrame, df: pd.DataFrame, cfg: Config, compute_betweenness: bool = True) -> tuple[dict, pd.DataFrame, nx.DiGraph]:
    G = make_graph(edges)
    U = undirected(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    metrics = {"nodes": n, "edges": m, "interactions": int(len(edges)),
               "density": float(nx.density(G)) if n > 1 else float("nan"),
               "reciprocity": float(nx.overall_reciprocity(G)) if m else float("nan")}
    if n == 0:
        return metrics, pd.DataFrame(), G
    wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    metrics["components"] = len(wcc)
    metrics["largest_component_nodes"] = len(wcc[0])
    metrics["largest_component_proportion"] = len(wcc[0]) / n
    part, q = communities_and_modularity(U)
    sizes = pd.Series(part).value_counts()
    metrics["modularity"] = q
    metrics["communities"] = int(len(sizes))
    metrics["communities_size_ge5"] = int((sizes >= 5).sum())
    metrics["community_sizes_top10"] = [int(x) for x in sizes.head(10).tolist()]
    metrics["degree_centralization"] = degree_centralization(U)
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    btw = {}
    if compute_betweenness and n > 2:
        lcc = U.subgraph(wcc[0])
        k = min(BETWEENNESS_SAMPLE, lcc.number_of_nodes())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            btw = nx.betweenness_centrality(lcc, k=k if k < lcc.number_of_nodes() else None, seed=42)
    part_coef = {}
    for node in U.nodes():
        nbrs = list(U.neighbors(node))
        part_coef[node] = float(np.mean([part[x] != part[node] for x in nbrs])) if nbrs else np.nan
    prof = account_profile(df, cfg)
    hate_in = edges[edges[ANY_HATE] == 1].groupby("target").size()
    nodes_list = list(G.nodes())
    indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())
    nodes = pd.DataFrame({
        "node": nodes_list,
        "in_degree": [indeg[x] for x in nodes_list], "out_degree": [outdeg[x] for x in nodes_list],
        "pagerank": [pr[x] for x in nodes_list], "betweenness": [btw.get(x, np.nan) for x in nodes_list],
        "participation_coefficient": [part_coef.get(x, np.nan) for x in nodes_list],
        "community": [part[x] for x in nodes_list],
        "hateful_in_degree": [int(hate_in.get(x, 0)) for x in nodes_list],
    })
    nodes["role"] = np.select([(nodes["out_degree"] > 0) & (nodes["in_degree"] == 0), (nodes["out_degree"] > 0) & (nodes["in_degree"] > 0),
                               (nodes["in_degree"] > 0) & (nodes["out_degree"] == 0)], ["spreader", "intermediary", "recipient"], "isolated")
    nodes = nodes.merge(prof, left_on="node", right_index=True, how="left")
    nodes["main_target"] = nodes["main_target"].fillna(FEW)
    nodes["profile_eligible"] = nodes["profile_eligible"].astype("boolean").fillna(False).astype(bool)
    nodes = nodes.sort_values(["pagerank", "in_degree"], ascending=False).reset_index(drop=True)
    metrics["accounts_posting"] = int(df["author"].nunique())
    metrics["accounts_in_network"] = int(nodes["records"].notna().sum())
    return metrics, nodes, G


def community_profiles(nodes: pd.DataFrame, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    comm = nodes.set_index("node")["community"]
    d = df[df["author"].isin(comm.index)].copy()
    d["community"] = d["author"].map(comm)
    g = d.groupby("community")
    prof = pd.DataFrame({"member_nodes": nodes.groupby("community").size(), "accounts": g["author"].nunique(),
                         "records": g.size(), "llm_labelled": g[LLM_LABELLED].sum(),
                         ANY_HATE: g[ANY_HATE].sum(), ANY_EXTREMIST: g[ANY_EXTREMIST].sum(min_count=1)})
    for c in hate:
        prof[c] = g[label_col(c)].sum()
        prof[f"{c}_proportion"] = prof[c] / prof["records"].replace(0, np.nan)
    for c in llm:
        prof[c] = g[label_col(c)].sum(min_count=1)
        prof[f"{c}_proportion"] = prof[c] / prof["llm_labelled"].replace(0, np.nan)
    prof["any_hate_proportion"] = prof[ANY_HATE] / prof["records"].replace(0, np.nan)
    prof["any_extremist_proportion"] = prof[ANY_EXTREMIST] / prof["llm_labelled"].replace(0, np.nan)
    prof["top_account_by_pagerank"] = nodes.sort_values("pagerank", ascending=False).groupby("community")["node"].first()
    elig = nodes[nodes["profile_eligible"]]
    if len(elig):
        comp = elig.groupby(["community", "main_target"]).size().unstack(fill_value=0)
        prof = prof.join(comp.add_prefix("members_"), how="left")
    prof = prof.sort_values(["records", "member_nodes"], ascending=False)
    prof.index.name = "community"
    prof = prof.reset_index()
    prof.insert(1, "community_rank", range(1, len(prof) + 1))
    return prof


def specialist_generalist(nodes: pd.DataFrame, hate: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = nodes[nodes["profile_eligible"]].copy()
    hate_acc = acc[acc["hate_label_total"] > 0]
    dist = hate_acc["n_hate_categories"].value_counts().sort_index()
    summary = pd.DataFrame({"measure": [f"accounts with {int(k)} hate categor{'y' if k == 1 else 'ies'}" for k in dist.index],
                            "n_categories": [int(k) for k in dist.index], "accounts": dist.to_numpy()})
    summary["proportion"] = summary["accounts"] / max(len(hate_acc), 1)
    summary["denominator"] = f"accounts with at least one hate label (n={len(hate_acc)})"
    vc = acc["main_target"].value_counts()
    extra = pd.DataFrame({"measure": ["main target: " + str(k) for k in vc.index], "n_categories": np.nan, "accounts": vc.to_numpy()})
    extra["proportion"] = extra["accounts"] / max(len(acc), 1)
    extra["denominator"] = f"accounts with at least {MIN_ACCOUNT_RECORDS} records (n={len(acc)})"
    extra["main_target"] = list(vc.index)
    summary = pd.concat([summary, extra], ignore_index=True)
    rows = []
    pos = hate_acc[hate] > 0 if hate else pd.DataFrame()
    for a in hate:
        for b in hate:
            inter, union = int((pos[a] & pos[b]).sum()), int((pos[a] | pos[b]).sum())
            rows.append({"category_a": a, "category_b": b, "accounts_both": inter, "accounts_either": union,
                         "jaccard_index": inter / union if union else np.nan,
                         "proportion_of_a_also_b": inter / max(int(pos[a].sum()), 1)})
    return summary, pd.DataFrame(rows, columns=["category_a", "category_b", "accounts_both", "accounts_either", "jaccard_index", "proportion_of_a_also_b"])


# ============================================================================================
# layout
# ============================================================================================
def igraph_layout(U: nx.Graph, seed: int = 42):
    """DrL through python-igraph (seeded), Fruchterman-Reingold as fallback; None when igraph is missing."""
    try:
        import random

        import igraph as ig
    except Exception:
        return None
    rng = random.Random(seed)
    random.seed(seed)
    try:
        ig.set_random_number_generator(random)
    except Exception:
        pass
    nl = list(U.nodes())
    idx = {n: i for i, n in enumerate(nl)}
    g = ig.Graph(n=len(nl), edges=[(idx[u], idx[v]) for u, v in U.edges()], directed=False)
    w = [float(U[u][v].get("weight", 1.0)) for u, v in U.edges()]
    try:
        lay = g.layout_drl(weights=w) if len(nl) >= 3 else g.layout_fruchterman_reingold(weights=w, niter=200, seed=None)
    except Exception:
        lay = g.layout_fruchterman_reingold(weights=w, niter=500)
    xy = np.asarray(lay.coords, dtype=float)
    del rng
    return {nl[i]: xy[i] for i in range(len(nl))}


def spring_layout(U: nx.Graph, seed: int = 42) -> dict:
    return {n: np.asarray(p, dtype=float) for n, p in nx.spring_layout(U, seed=seed, weight="weight", iterations=80).items()}


def compact_communities(pos: dict, node_comm: dict, shrink: float = 0.35) -> dict:
    """Pull each community's centroid towards the global centre (0 = collapse, 1 = unchanged), keeping shapes."""
    members = defaultdict(list)
    for n, c in node_comm.items():
        members[c].append(n)
    gc = np.mean([pos[n] for n in pos], axis=0)
    cen = {c: np.mean([pos[n] for n in mem], axis=0) for c, mem in members.items()}
    return {n: gc + (cen[node_comm[n]] - gc) * shrink + (pos[n] - cen[node_comm[n]]) for n in pos}


def rein_in_outliers(pos: dict, knee_q: float = 0.88, excess: float = 0.2) -> dict:
    """Compress only the distance beyond the ``knee_q`` quantile, radially; every node is kept."""
    nl = list(pos)
    pts = np.array([pos[n] for n in nl], dtype=float)
    gc = pts.mean(axis=0)
    v = pts - gc
    d = np.hypot(v[:, 0], v[:, 1])
    knee = float(np.quantile(d, knee_q))
    scale = np.ones_like(d)
    far = d > knee
    scale[far] = (knee + (d[far] - knee) * excess) / np.maximum(d[far], 1e-9)
    out = gc + v * scale[:, None]
    return {nl[i]: out[i] for i in range(len(nl))}


def layout_backbone(G: nx.DiGraph, meta: pd.DataFrame, cfg: Config):
    """Backbone extraction + layout.  Returns (U, pos, disp, drawn) or None."""
    if G.number_of_nodes() == 0:
        return None
    net = cfg.section("network")
    min_degree, max_nodes = int(net.get("min_degree", 2)), int(net.get("max_nodes", 2500))
    wcc = max(nx.weakly_connected_components(G), key=len)
    U0 = G.subgraph(wcc).to_undirected()
    U0 = U0.subgraph([n for n, d in U0.degree() if d >= min_degree]).copy()
    if U0.number_of_nodes() == 0:
        return None
    core = nx.k_core(U0, k=2)
    base = core if core.number_of_nodes() >= 50 else U0
    if base.number_of_nodes() > max_nodes:
        idx = [n for n in base.nodes() if n in meta.index]
        keep = meta.loc[idx].sort_values("pagerank", ascending=False).head(max_nodes).index
        base = base.subgraph(keep)
    if base.number_of_edges() == 0:
        return None
    U = base.subgraph(max(nx.connected_components(base), key=len)).copy()
    comm = meta["community"]
    present = pd.Series([comm.get(n) for n in U.nodes()]).value_counts()
    keep = [c for c in present.index if present[c] >= 10][:len(COMMUNITY_PALETTE)] or list(present.index[:len(COMMUNITY_PALETTE)])
    keep_set = set(keep)
    U = U.subgraph([n for n in U.nodes() if comm.get(n) in keep_set])
    if U.number_of_nodes() == 0 or U.number_of_edges() == 0:
        return None
    U = U.subgraph(max(nx.connected_components(U), key=len)).copy()
    drawn = [c for c in keep if c in {comm.get(n) for n in U.nodes()}]
    disp = {c: j for j, c in enumerate(drawn)}
    node_comm = {n: comm.get(n, -1) for n in U.nodes()}
    pos = igraph_layout(U)
    if pos is None:
        pos = spring_layout(U)
    pos = compact_communities(pos, node_comm, shrink=0.35)
    pos = rein_in_outliers(pos)
    return U, pos, disp, drawn


def crop_limits(pos: dict, U: nx.Graph, min_nodes_for_crop: int = 300) -> tuple:
    """Axis limits: the full extent for small graphs; for large graphs the 4-96 % quantile box, so a
    handful of far-flung accounts does not shrink the rest of the picture."""
    pts = np.array([pos[n] for n in U.nodes()])
    q = (0.04, 0.96) if len(pts) >= min_nodes_for_crop else (0.0, 1.0)
    qx, qy = np.quantile(pts[:, 0], q), np.quantile(pts[:, 1], q)
    mx, my = (qx[1] - qx[0]) * 0.08 + 1e-9, (qy[1] - qy[0]) * 0.08 + 1e-9
    return ((qx[0] - mx, qx[1] + mx), (qy[0] - my, qy[1] + my))


def render(U: nx.Graph, pos: dict, sizes, cols, handles, title: str, path, lim=None) -> None:
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#111110")
    ax.set_facecolor("#111110")
    ew = np.array([d.get("weight", 1) for _, _, d in U.edges(data=True)], dtype=float)
    widths = 0.15 + 0.4 * np.log1p(ew) / max(np.log1p(ew.max()) if len(ew) else 1.0, 1e-9)
    nx.draw_networkx_edges(U, pos, ax=ax, edge_color="#9fb2c9", alpha=0.08, width=widths)
    nx.draw_networkx_nodes(U, pos, ax=ax, nodelist=list(U.nodes()), node_size=sizes, node_color=cols, linewidths=0)
    lim = lim or crop_limits(pos, U)
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    if handles:
        ax.legend(handles=handles, loc="lower left", frameon=False, labelcolor="white", fontsize=9, ncol=2)
    ax.set_title(title, color="white", fontsize=12, loc="left")
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="#111110", bbox_inches="tight")
    plt.close(fig)


def _sizes(meta: pd.DataFrame, nodes_list) -> np.ndarray:
    prs = meta.reindex(nodes_list)["pagerank"].fillna(0).to_numpy(dtype=float)
    return 3 + 70 * np.sqrt(prs / max(prs.max(), 1e-12))


def draw_network(G: nx.DiGraph, nodes: pd.DataFrame, cfg: Config, colour_by: str, title: str, path, hate: list[str]) -> bool:
    meta = nodes.set_index("node")
    bb = layout_backbone(G, meta, cfg)
    if bb is None:
        return False
    U, pos, disp, drawn = bb
    comm = meta["community"]
    if colour_by == "community":
        cols = [COMMUNITY_PALETTE[disp[comm[n]]] for n in U.nodes()]
        handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COMMUNITY_PALETTE[disp[c]], markeredgecolor="none",
                          markersize=8, label=f"Community {disp[c] + 1}") for c in drawn]
    else:
        colours = category_colors(cfg)

        def col(n):
            t = meta.loc[n, "main_target"]
            if t in hate:
                return colours[t]
            return "#f5f4ef" if t == MULTI else "#55554f"
        cols = [col(n) for n in U.nodes()]
        handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colours[c], markeredgecolor="none", markersize=8,
                          label=category_label(cfg, c)) for c in hate]
        handles += [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#f5f4ef", markeredgecolor="none", markersize=8, label=MULTI),
                    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#55554f", markeredgecolor="none", markersize=8, label="no hate label or too few records")]
    render(U, pos, _sizes(meta, list(U.nodes())), cols, handles, title, path)
    return True


def shared_reference_layout(g_before: nx.DiGraph, g_after: nx.DiGraph, meta: pd.DataFrame | None = None,
                            cfg: Config | None = None) -> tuple[dict, dict]:
    """One layout for two snapshots: positions and communities are computed on the union graph, so an
    account present in both windows sits at the same place with the same colour in both panels.
    Returns (positions, community) for the union backbone (or the whole union when no meta is given)."""
    union = nx.compose(g_before, g_after)
    if meta is None:
        U = undirected(union)
        part, _ = communities_and_modularity(U)
        pos = igraph_layout(U) or spring_layout(U)
        pos = rein_in_outliers(compact_communities(pos, part)) if len(pos) > 2 else pos
        return pos, part
    bb = layout_backbone(union, meta, cfg)
    if bb is None:
        return {}, {}
    U, pos, disp, drawn = bb
    comm = meta["community"]
    return pos, {n: comm.get(n) for n in U.nodes()}


def plot_before_after(platform: str, edges: pd.DataFrame, nodes: pd.DataFrame, cfg: Config, figdir) -> None:
    ba = cfg.section("network").get("before_after") or {}
    if not ba:
        return
    windows = [("before", ba.get("before", {})), ("after", ba.get("after", {}))]
    graphs = {}
    for name, w in windows:
        s, e = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        graphs[name] = make_graph(edges[(edges["day"] >= s) & (edges["day"] <= e)])
    if any(g.number_of_nodes() == 0 for g in graphs.values()):
        return
    meta = nodes.set_index("node")
    bb = layout_backbone(nx.compose(graphs["before"], graphs["after"]), meta, cfg)
    if bb is None:
        return
    Uref, pos, disp, drawn = bb
    ref_nodes = set(Uref.nodes())
    lim = crop_limits(pos, Uref)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COMMUNITY_PALETTE[disp[c]], markeredgecolor="none",
                      markersize=8, label=f"Community {disp[c] + 1}") for c in drawn]
    for name, w in windows:
        Uw = graphs[name].subgraph([n for n in graphs[name].nodes() if n in ref_nodes]).to_undirected()
        if Uw.number_of_nodes() == 0:
            continue
        cols = [COMMUNITY_PALETTE[disp[meta.loc[n, "community"]]] for n in Uw.nodes()]
        label = w.get("label", name)
        title = f"{platform_label(platform)}: interaction network, {label} ({w['start']} to {w['end']})"
        render(Uw, {n: pos[n] for n in Uw.nodes()}, _sizes(meta, list(Uw.nodes())), cols, handles, title, figdir / f"network_{name}.png", lim=lim)


# ============================================================================================
# other figures
# ============================================================================================
def plot_cluster_heatmap(prof: pd.DataFrame, cfg: Config, platform: str, cats: list[str], path, top_k: int = 12, min_records: int = 20) -> None:
    p = prof[prof["records"] >= min_records].head(top_k)
    cols = [f"{c}_proportion" for c in cats if f"{c}_proportion" in p.columns]
    if p.empty or not cols:
        return
    mat = p[cols].to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(1.5 + 0.9 * len(cols), 0.45 * len(p) + 2))
    vmax = max(np.nanmax(mat), 1) if np.isfinite(np.nanmax(mat)) else 1
    im = ax.imshow(mat, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(p)), [f"Community {int(r)} ({int(a)} accounts, {int(s):,} posts)" for r, a, s in zip(p["community_rank"], p["accounts"], p["records"])], fontsize=7)
    ax.set_xticks(range(len(cols)), [category_label(cfg, c) for c in cats if f"{c}_proportion" in p.columns], fontsize=7, rotation=35, ha="right")
    for (r, c), v in np.ndenumerate(mat):
        if np.isfinite(v):
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=6, color=INK if v < 0.6 * vmax else "white")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("% of the community's posts", fontsize=7)
    ax.set_title(f"{platform_label(platform)}: category composition of each community's posts", fontsize=10, loc="left", color=INK)
    save_fig(fig, path, dpi=200)


def plot_overlap(overlap: pd.DataFrame, summary: pd.DataFrame, cfg: Config, platform: str, hate: list[str], figdir) -> None:
    colours = category_colors(cfg)
    labels = [category_label(cfg, c) for c in hate]
    if len(hate) and len(overlap):
        mat = overlap.pivot(index="category_a", columns="category_b", values="proportion_of_a_also_b").reindex(index=hate, columns=hate).to_numpy(dtype=float) * 100
        fig, ax = plt.subplots(figsize=(1.8 + 0.9 * len(hate), 1.5 + 0.7 * len(hate)))
        im = ax.imshow(mat, cmap=SEQ_CMAP, vmin=0, vmax=100)
        ax.set_xticks(range(len(hate)), labels, fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(len(hate)), labels, fontsize=7)
        for (r, c), v in np.ndenumerate(mat):
            if np.isfinite(v):
                ax.text(c, r, f"{v:.0f}%", ha="center", va="center", fontsize=7, color=INK if v < 60 else "white")
        ax.set_title(f"{platform_label(platform)}: accounts with a given target that also have another", fontsize=8.5, loc="left", color=INK)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label("% of accounts (row) also in column", fontsize=7)
        save_fig(fig, figdir / "hate_category_overlap_heatmap.png", dpi=200)
    s = summary[summary["n_categories"].notna()].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if len(s):
        xs = s["n_categories"].astype(int).to_numpy()
        ax.bar(xs, s["proportion"] * 100, color=PALETTE[0], width=0.7)
        for x, v, n in zip(xs, s["proportion"] * 100, s["accounts"]):
            ax.text(x, v + 1, f"{v:.0f}%\n(n={int(n):,})", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(range(1, max(len(hate), 1) + 1))
        ax.set_ylim(0, max(s["proportion"].max() * 100 + 12, 10))
    style_axes(ax, f"{platform_label(platform)}: number of hate categories per account", "% of accounts with a hate label",
               "number of hate categories per account")
    save_fig(fig, figdir / "hate_category_count_distribution.png", dpi=200)
    order = hate + [MULTI]
    mt = summary[summary["measure"].astype(str).str.startswith("main target: ")].copy()
    mt = mt.set_index("main_target").reindex(order) if "main_target" in mt.columns else pd.DataFrame(index=order)
    vals = (mt["proportion"].astype(float) * 100).to_numpy() if "proportion" in mt.columns else np.full(len(order), np.nan)
    accs = mt["accounts"].astype(float).to_numpy() if "accounts" in mt.columns else np.full(len(order), np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = list(range(len(order)))
    ax.bar(xs, np.nan_to_num(vals), color=[colours.get(c, NEUTRAL) for c in order], width=0.7)
    for x, v, n in zip(xs, vals, accs):
        if np.isfinite(v) and np.isfinite(n):
            ax.text(x, v + 0.6, f"{v:.0f}%\n(n={int(n):,})", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(xs, [category_label(cfg, c) if c in hate else "Multiple targets" for c in order], fontsize=7, rotation=25, ha="right")
    top = np.nanmax(vals) if len(vals) and np.isfinite(np.nanmax(vals)) else 10
    ax.set_ylim(0, top * 1.25 + 2)
    style_axes(ax, f"{platform_label(platform)}: main target of hate per account", "% of accounts")
    save_fig(fig, figdir / "hate_category_main_target_distribution.png", dpi=200)


def style_axes(ax, title, ylabel, xlabel=None):
    from .plotting import style_axes as _style
    _style(ax, title, ylabel, xlabel)


# ============================================================================================
def run_platform(cfg: Config, platform: str) -> dict:
    df = load_analysis_records(cfg, platform)
    df = df[df["author"].notna()].copy()
    hate, llm = supervised_categories(cfg, df), llm_categories(cfg, df)
    tdir, fdir = table_dir(cfg, "network", platform), fig_dir(cfg, "network", platform)
    edges = build_edges(df, cfg)
    edges.to_csv(tdir / "edges.csv.gz", index=False, compression="gzip")
    metrics, nodes, G = analyse_graph(edges, df, cfg)
    metrics["edge_types"] = {str(k): int(v) for k, v in edges["edge_type"].value_counts().items()} if len(edges) else {}
    metrics["hateful_records"] = int(df[ANY_HATE].sum())
    save_json(metrics, tdir / "metrics.json")
    if nodes.empty:
        print(f"network: {platform} has no interactions")
        return metrics
    save_csv(nodes, tdir / "nodes.csv")
    prof = community_profiles(nodes, df, cfg)
    save_csv(prof, tdir / "community_profiles.csv")
    spec, overlap = specialist_generalist(nodes, hate)
    save_csv(spec, tdir / "specialist_generalist.csv")
    save_csv(overlap, tdir / "category_overlap.csv")
    plot_cluster_heatmap(prof, cfg, platform, hate + llm, fdir / "community_by_category_heatmap.png")
    plot_overlap(overlap, spec, cfg, platform, hate, fdir)
    draw_network(G, nodes, cfg, "community", f"{platform_label(platform)}: interaction network, accounts coloured by community",
                 fdir / "network_by_community.png", hate)
    draw_network(G, nodes, cfg, "target", f"{platform_label(platform)}: interaction network, accounts coloured by main target of hate",
                 fdir / "network_by_main_target.png", hate)
    ba = cfg.section("network").get("before_after") or {}
    if ba:
        rows = []
        for name in ("before", "after"):
            w = ba.get(name) or {}
            s, e = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            dw = df[(df["day"] >= s) & (df["day"] <= e)]
            ew = edges[(edges["day"] >= s) & (edges["day"] <= e)]
            m, _, _ = analyse_graph(ew, dw, cfg, compute_betweenness=False)
            m.pop("community_sizes_top10", None)
            rows.append({"comparison": f"{platform_label(platform)}: {w.get('label', name)}", "window": name,
                         "start": str(s.date()), "end": str(e.date()), "days": (e - s).days + 1, "records": int(len(dw)),
                         "hateful_records": int(dw[ANY_HATE].sum()),
                         "hateful_proportion": float(dw[ANY_HATE].mean()) if len(dw) else np.nan, **m})
        save_csv(pd.DataFrame(rows), tdir / "before_after_metrics.csv")
        plot_before_after(platform, edges, nodes, cfg, fdir)
    print(f"network: {platform} {metrics['nodes']} accounts, {metrics['edges']} ties, {metrics['communities']} communities -> {tdir}")
    return metrics


def run(cfg: Config, platforms: list[str] | None = None) -> dict:
    platforms = platforms or cfg.platforms
    out = {p: run_platform(cfg, p) for p in platforms}
    rows = []
    for p in platforms:
        f = table_dir(cfg, "network", p) / "before_after_metrics.csv"
        if f.exists():
            rows.append(pd.read_csv(f).assign(platform=p))
    if rows:
        save_csv(pd.concat(rows, ignore_index=True), table_dir(cfg, "network") / "before_after_all_platforms.csv")
    save_json({"parameters": cfg.section("network"), "platforms": out}, table_dir(cfg, "network") / "network_summary.json")
    return out


def main(argv=None) -> int:
    parser = add_config_argument(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--platform", action="append")
    args = parser.parse_args(argv)
    run(config_from_args(args), args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
