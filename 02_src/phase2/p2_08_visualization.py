#!/usr/bin/env python3
"""p2_08 — Network figures (main 3-camp pooled + appendix 6-grid)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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


def layout_graph(G: nx.Graph):
    if ForceAtlas2 is not None and len(G) > 0:
        fa2 = ForceAtlas2(outboundAttractionDistribution=True, lin_log_mode=True)
        pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=50)
    else:
        pos = nx.spring_layout(G, seed=42)
    return pos


def draw_network(ax, G: nx.Graph, comm_map: dict, top_edges: int = 50):
    if len(G) == 0:
        ax.set_title("empty")
        return
    edges = sorted(G.edges(data=True), key=lambda e: abs(float(e[2].get("npmi_median", 0))), reverse=True)
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in edges[:top_edges]:
        H.add_edge(u, v, **d)
    pos = layout_graph(H)
    nodes = list(H.nodes())
    sizes = []
    for n in nodes:
        mp = float(H.nodes[n].get("marginal_p", 0.1))
        sizes.append(max(80, 400 * np.log1p(mp * 100)))
    colors = [comm_map.get(n, 0) for n in nodes]
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=sizes, node_color=colors, cmap=plt.cm.Set2)
    widths = [max(0.5, 3 * abs(float(H[u][v].get("npmi_median", 0.5)))) for u, v in H.edges()]
    nx.draw_networkx_edges(H, pos, ax=ax, width=widths, alpha=0.6)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=7)
    ax.axis("off")


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

    # Main: 3 camp pooled
    net_camp = cfg.scheme_dir("networks", "camp")
    part_camp = read_parquet(cfg.scheme_dir("partitions", "camp") / "consensus_partition.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, camp in enumerate(cfg.camps):
        gp = net_camp / f"{camp}_positive.graphml"
        if not gp.is_file():
            axes[i].set_title(f"{camp} (missing)")
            continue
        G = nx.read_graphml(gp)
        comm = dict(zip(part_camp[part_camp["camp"] == camp]["node"], part_camp[part_camp["camp"] == camp]["community_id"]))
        axes[i].set_title(camp)
        draw_network(axes[i], G, comm)
    fig.tight_layout()
    main_path = fig_dir / "main_3camp_pooled.pdf"
    fig.savefig(main_path, bbox_inches="tight")
    plt.close(fig)

    # Appendix: 6-grid camp × genre
    net_cg = cfg.scheme_dir("networks", "camp_genre")
    part_cg = read_parquet(cfg.scheme_dir("partitions", "camp_genre") / "consensus_partition.parquet")
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for i, camp in enumerate(cfg.camps):
        for j, genre in enumerate(cfg.genres_main):
            ax = axes[i, j]
            key = f"{camp}_{genre}"
            gp = net_cg / f"{key}_positive.graphml"
            ax.set_title(key)
            if not gp.is_file():
                continue
            G = nx.read_graphml(gp)
            sub = part_cg[(part_cg["camp"] == camp) & (part_cg["genre"] == genre)]
            comm = dict(zip(sub["node"], sub["community_id"]))
            draw_network(ax, G, comm)
    fig.tight_layout()
    app_path = fig_dir / "appendix_6grid_camp_genre.pdf"
    fig.savefig(app_path, bbox_inches="tight")
    plt.close(fig)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([net_camp, net_cg])})
    write_summary(
        art / "p2_08_summary.md",
        "p2_08",
        params={"layout": "fa2" if ForceAtlas2 else "spring"},
        outputs=[str(main_path), str(app_path)],
        stats={},
        notes=["Gephi manual polish optional; reproducible chain via this script."],
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_08 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
