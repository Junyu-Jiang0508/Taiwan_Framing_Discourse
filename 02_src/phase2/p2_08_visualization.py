#!/usr/bin/env python3
"""p2_08 — Network figures.

Main figure: 3-camp panel with
  * unified layout (computed once on the union graph) so node positions are
    directly comparable across camps;
  * node color encoding substantive role from the sub-network audit
    (shared infrastructure vs differentiating subnet);
  * CI-disjoint edges (from p2_13) highlighted in red.

Differentiating sub-network figure: 4-node ({L2-02, L2-04, L2-05, L2-08})
× 3-camp panel with the same CI-disjoint highlighting — this is the
"money figure" for sub-claim 1.

L2-07 ego-network figure: 3-camp panel restricted to L2-07 and its
immediate neighbors in each camp, illustrating how the "Solidarity/
Vision" hub is wired differently across camps despite being shared
infrastructure.

Permutation null figure: histograms of pairwise Hubert γ under the
camp-label permutation null (p2_12) with observed γ overlaid.

Appendix: 6-grid camp × genre (unchanged design, kept for robustness).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.io import read_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402

try:
    from fa2 import ForceAtlas2
except ImportError:
    ForceAtlas2 = None  # type: ignore


# Substantive node roles from sub-network audit (p2_14).
# Shared infrastructure: Jaccard >= 0.66 across all camps in cross-camp matching.
# Differentiating: nodes carrying the bulk of CI-disjoint edges.
SHARED_NODES = {"L2-01", "L2-03", "L2-06", "L2-07"}
DIFFERENTIATING_NODES = {"L2-02", "L2-04", "L2-05", "L2-08"}

COLOR_SHARED = "#4C78A8"        # blue: shared infrastructure
COLOR_DIFFERENTIATING = "#F58518"  # orange: differentiating subnet
COLOR_OTHER = "#BAB0AC"         # gray fallback
COLOR_EDGE_DEFAULT = "#888888"
COLOR_EDGE_DISJOINT = "#D62728"  # red: CI-disjoint edge

L2_LABELS_EN = {
    "L2-01": "Subjectivity",
    "L2-02": "Diff. Identity",
    "L2-03": "Intl. Legitimacy",
    "L2-04": "Narrative Recon.",
    "L2-05": "Shared Threat",
    "L2-06": "National Pride",
    "L2-07": "Solidarity/Vision",
    "L2-08": "Democratic Values",
}


def node_color(n: str) -> str:
    if n in SHARED_NODES:
        return COLOR_SHARED
    if n in DIFFERENTIATING_NODES:
        return COLOR_DIFFERENTIATING
    return COLOR_OTHER


def build_union_graph(camp_graphs: dict[str, nx.Graph]) -> nx.Graph:
    """Union of all camps' positive graphs; edge weight = max abs NPMI seen."""
    U = nx.Graph()
    for G in camp_graphs.values():
        for n, d in G.nodes(data=True):
            if n not in U:
                U.add_node(n, **d)
        for u, v, d in G.edges(data=True):
            w = abs(float(d.get("npmi_median", 0.0)))
            if U.has_edge(u, v):
                if w > U[u][v]["weight"]:
                    U[u][v]["weight"] = w
            else:
                U.add_edge(u, v, weight=w)
    return U


def compute_unified_layout(union: nx.Graph) -> dict:
    if ForceAtlas2 is not None and len(union) > 0:
        fa2 = ForceAtlas2(outboundAttractionDistribution=True, lin_log_mode=True, verbose=False)
        return fa2.forceatlas2_networkx_layout(union, pos=None, iterations=200)
    return nx.spring_layout(union, seed=42, weight="weight", k=1.2)


def disjoint_edges_for_camp(diff_df: pd.DataFrame, camp: str) -> set[tuple[str, str]]:
    """Edges (unordered) that are CI-disjoint in at least one pairwise comparison
    involving `camp`."""
    rows = diff_df[
        ((diff_df["camp_a"] == camp) | (diff_df["camp_b"] == camp))
        & (diff_df["ci_disjoint"] == True)  # noqa: E712
    ]
    return {tuple(sorted([a, b])) for a, b in zip(rows["l2_a"], rows["l2_b"])}


def draw_panel(
    ax,
    G: nx.Graph,
    pos: dict,
    disjoint_edges: set[tuple[str, str]],
    title: str,
    *,
    node_subset: set[str] | None = None,
    show_labels: bool = True,
) -> None:
    """Draw one camp panel using shared `pos`. CI-disjoint edges in red."""
    H = G.subgraph(node_subset).copy() if node_subset is not None else G.copy()
    if H.number_of_nodes() == 0:
        ax.set_title(f"{title} (empty)")
        ax.axis("off")
        return

    nodes = list(H.nodes())
    sizes = [550 for _ in nodes]
    colors = [node_color(n) for n in nodes]

    # Edge styling: split by CI-disjoint vs not
    edges = list(H.edges(data=True))
    disj, normal = [], []
    for u, v, d in edges:
        key = tuple(sorted([u, v]))
        w = abs(float(d.get("npmi_median", 0.0)))
        width = max(0.6, 6.0 * (w**1.3))
        if key in disjoint_edges:
            disj.append((u, v, width))
        else:
            normal.append((u, v, width))

    nx.draw_networkx_edges(
        H, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in normal],
        width=[w for _, _, w in normal],
        edge_color=COLOR_EDGE_DEFAULT, alpha=0.45,
    )
    nx.draw_networkx_edges(
        H, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in disj],
        width=[w for _, _, w in disj],
        edge_color=COLOR_EDGE_DISJOINT, alpha=0.95,
    )
    nx.draw_networkx_nodes(
        H, pos, ax=ax, nodelist=nodes, node_size=sizes,
        node_color=colors, edgecolors="white", linewidths=1.4,
    )
    if show_labels:
        nx.draw_networkx_labels(H, pos, ax=ax, font_size=7, font_weight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def add_legend(fig, *, include_disjoint: bool = True) -> None:
    handles = [
        mpatches.Patch(color=COLOR_SHARED, label="Shared infrastructure nodes\n{L2-01, L2-03, L2-06, L2-07}"),
        mpatches.Patch(color=COLOR_DIFFERENTIATING, label="Differentiating nodes\n{L2-02, L2-04, L2-05, L2-08}"),
    ]
    if include_disjoint:
        handles.append(mpatches.Patch(color=COLOR_EDGE_DISJOINT, label="CI-disjoint edge (p2_13)"))
        handles.append(mpatches.Patch(color=COLOR_EDGE_DEFAULT, label="Other significant edge (FDR)"))
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8,
    )


def add_label_table(fig) -> None:
    """Small text block listing L2 code → short English label."""
    lines = [f"{k}: {v}" for k, v in L2_LABELS_EN.items()]
    text = "   ".join(lines[:4]) + "\n" + "   ".join(lines[4:])
    fig.text(0.5, -0.08, text, ha="center", va="top", fontsize=7, family="monospace")


def draw_ego_panel(
    ax,
    G: nx.Graph,
    pos: dict,
    disjoint_edges: set[tuple[str, str]],
    title: str,
    ego: str,
) -> None:
    """Draw the ego sub-network for `ego` (ego + its immediate neighbors)."""
    if ego not in G:
        ax.set_title(f"{title} ({ego} absent)")
        ax.axis("off")
        return
    neighbors = set(G.neighbors(ego))
    node_subset = {ego} | neighbors
    draw_panel(ax, G, pos, disjoint_edges, title, node_subset=node_subset)


def draw_permutation_null(null_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    """Histogram of null γ per camp pair with observed γ overlaid."""
    pairs = [("DPP", "KMT"), ("DPP", "TPP"), ("KMT", "TPP")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (a, b) in zip(axes, pairs):
        key = f"gamma_{a.lower()}_{b.lower()}"
        if null_df.empty or key not in null_df.columns:
            ax.set_title(f"{a}–{b} (no data)")
            ax.axis("off")
            continue
        vals = null_df[key].values.astype(float)
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=40, color=COLOR_SHARED, alpha=0.7, edgecolor="white")

        row = summary_df[summary_df["test"] == key]
        if len(row):
            r = row.iloc[0]
            obs = float(r["observed_gamma"])
            q05 = float(r["null_q05"])
            q95 = float(r["null_q95"])
            p_val = float(r["p_value"])
            ax.axvline(q05, color="#555555", linestyle=":", linewidth=1, label="null 5/95%")
            ax.axvline(q95, color="#555555", linestyle=":", linewidth=1)
            ax.axvline(obs, color=COLOR_EDGE_DISJOINT, linestyle="-", linewidth=2,
                       label=f"observed γ={obs:.3f}\np={p_val:.3f}")
            ax.legend(loc="upper left", fontsize=7, frameon=False)
        ax.set_title(f"{a}–{b}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hubert γ")
        ax.set_ylabel("count")
    fig.suptitle(
        "Camp-label permutation null vs observed cross-camp γ (p2_12)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    fig_dir = art / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = art / "manifests" / "p2_08.json"
    expected = {"corpus_content_hash": cfg.corpus_content_hash}
    if should_skip(manifest_path, expected, force):
        print("p2_08: skip (manifest match)")
        return

    net_camp = cfg.scheme_dir("networks", "camp")

    # Load per-camp graphs
    camp_graphs: dict[str, nx.Graph] = {}
    for camp in cfg.camps:
        gp = net_camp / f"{camp}_positive.graphml"
        if gp.is_file():
            camp_graphs[camp] = nx.read_graphml(gp)

    # Unified layout from union graph (all camps' edges combined)
    union = build_union_graph(camp_graphs)
    pos = compute_unified_layout(union)

    # CI-disjoint edges per camp (from p2_13)
    diff_path = art / "cross_camp" / "edge_diff_pairwise.parquet"
    diff_df = read_parquet(diff_path) if diff_path.is_file() else pd.DataFrame(
        columns=["l2_a", "l2_b", "camp_a", "camp_b", "ci_disjoint"]
    )
    disjoint_by_camp = {c: disjoint_edges_for_camp(diff_df, c) for c in cfg.camps}

    # === Main figure: 3-camp full network, shared layout ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, camp in enumerate(cfg.camps):
        G = camp_graphs.get(camp)
        if G is None:
            axes[i].set_title(f"{camp} (missing)")
            axes[i].axis("off")
            continue
        draw_panel(axes[i], G, pos, disjoint_by_camp[camp], camp)
    fig.suptitle(
        "Co-articulation networks by camp (unified layout; CI-disjoint edges in red)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    add_legend(fig)
    add_label_table(fig)
    main_path = fig_dir / "main_3camp_pooled.pdf"
    fig.savefig(main_path, bbox_inches="tight")
    plt.close(fig)

    # === Differentiating sub-network figure ===
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, camp in enumerate(cfg.camps):
        G = camp_graphs.get(camp)
        if G is None:
            axes[i].set_title(f"{camp} (missing)")
            axes[i].axis("off")
            continue
        draw_panel(
            axes[i], G, pos, disjoint_by_camp[camp], camp,
            node_subset=DIFFERENTIATING_NODES,
        )
    fig.suptitle(
        "Differentiating sub-network: {L2-02, L2-04, L2-05, L2-08} × 3 camps",
        fontsize=12, fontweight="bold", y=1.03,
    )
    add_legend(fig)
    diff_path_fig = fig_dir / "differentiating_subnet_3camp.pdf"
    fig.savefig(diff_path_fig, bbox_inches="tight")
    plt.close(fig)

    # === L2-07 ego network across camps ===
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, camp in enumerate(cfg.camps):
        G = camp_graphs.get(camp)
        if G is None:
            axes[i].set_title(f"{camp} (missing)")
            axes[i].axis("off")
            continue
        draw_ego_panel(axes[i], G, pos, disjoint_by_camp[camp], camp, ego="L2-07")
    fig.suptitle(
        "L2-07 (Solidarity/Vision) ego network × 3 camps",
        fontsize=12, fontweight="bold", y=1.03,
    )
    add_legend(fig)
    ego07_path = fig_dir / "ego_l2_07_3camp.pdf"
    fig.savefig(ego07_path, bbox_inches="tight")
    plt.close(fig)

    # === Permutation null distribution (p2_12) ===
    null_df_path = art / "cross_camp" / "permutation_null_distribution.parquet"
    perm_summary_path = art / "cross_camp" / "permutation_summary.parquet"
    null_df = read_parquet(null_df_path) if null_df_path.is_file() else pd.DataFrame()
    perm_summary_df = read_parquet(perm_summary_path) if perm_summary_path.is_file() else pd.DataFrame()
    perm_null_fig_path = fig_dir / "permutation_null_distribution.pdf"
    draw_permutation_null(null_df, perm_summary_df, perm_null_fig_path)

    # === Appendix: 6-grid camp × genre (unified layout reused) ===
    net_cg = cfg.scheme_dir("networks", "camp_genre")
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    for i, camp in enumerate(cfg.camps):
        for j, genre in enumerate(cfg.genres_main):
            ax = axes[i, j]
            key = f"{camp}_{genre}"
            gp = net_cg / f"{key}_positive.graphml"
            if not gp.is_file():
                ax.set_title(f"{key} (missing)")
                ax.axis("off")
                continue
            G = nx.read_graphml(gp)
            # Reuse unified layout for the subset of nodes present
            sub_pos = {n: pos[n] for n in G.nodes() if n in pos}
            draw_panel(ax, G, sub_pos, disjoint_by_camp.get(camp, set()), key)
    fig.suptitle("Appendix: camp × genre networks (unified layout)", fontsize=12, fontweight="bold", y=1.01)
    add_legend(fig)
    app_path = fig_dir / "appendix_6grid_camp_genre.pdf"
    fig.savefig(app_path, bbox_inches="tight")
    plt.close(fig)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([net_camp, net_cg])})
    write_summary(
        art,
        "p2_08",
        params={
            "layout": "fa2" if ForceAtlas2 else "spring",
            "unified_layout": True,
            "ci_disjoint_highlight": True,
            "node_color_scheme": "substantive_role",
        },
        outputs=[
            str(main_path),
            str(diff_path_fig),
            str(ego07_path),
            str(perm_null_fig_path),
            str(app_path),
        ],
        stats={
            "n_disjoint_edges_total": int(diff_df["ci_disjoint"].sum()) if len(diff_df) else 0,
            "disjoint_by_camp": {c: len(s) for c, s in disjoint_by_camp.items()},
        },
        notes=[
            "Unified layout via union graph (FA2 if available, else spring).",
            "Node color: blue=shared {01,03,06,07}, orange=differentiating {02,04,05,08}.",
            "Red edges = CI-disjoint vs at least one other camp (p2_13).",
            "Differentiating sub-network figure is the primary money figure for sub-claim 1.",
        ],
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_08 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
