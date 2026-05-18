#!/usr/bin/env python3
"""p2_05 — Leiden community detection + consensus partition."""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx
import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.leiden_consensus import (  # noqa: E402
    coassignment_matrix,
    community_stability_scores,
    consensus_partition,
    run_leiden_partition,
)
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.stratum_meta import stratum_row_fields  # noqa: E402
from utils.summary import write_summary  # noqa: E402


def _communities_summary(cons: dict, nodes: List[str]) -> Dict[int, List[str]]:
    by_c: Dict[int, List[str]] = defaultdict(list)
    for node in nodes:
        by_c[int(cons[node])].append(node)
    return dict(by_c)


def run(
    cfg: Phase2Config,
    force: bool = False,
    scheme_filter: str | None = None,
    window_suffix: str | int | None = None,
) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    lcfg = cfg.raw["leiden"]
    gammas = lcfg["gammas"]
    seeds_per = int(lcfg["seeds_per_gamma"])
    seed_base = int(lcfg["seed_base"])
    threshold = float(lcfg["consensus_threshold"])
    consensus_gamma = float(lcfg["consensus_gamma"])
    stab_min = float(lcfg["stability_coassignment_min"])

    manifest_suffix = "" if window_suffix is None else f"_n{window_suffix}"
    manifest_path = art / "manifests" / f"p2_05{manifest_suffix}.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "leiden": str(lcfg),
        "window_suffix": str(window_suffix),
    }
    if should_skip(manifest_path, expected, force):
        print(f"p2_05{manifest_suffix}: skip (manifest match)")
        return

    raw_rows = []
    consensus_rows = []
    stability_rows = []
    input_paths = []
    stratum_partition_stats = []

    for scheme in cfg.schemes:
        if scheme_filter and scheme.name != scheme_filter:
            continue
        net_dir = cfg.scheme_dir("networks", scheme.name, window_suffix)
        part_dir = cfg.scheme_dir("partitions", scheme.name, window_suffix)
        part_dir.mkdir(parents=True, exist_ok=True)

        for graph_path in sorted(net_dir.glob("*_positive.graphml")):
            key_str = graph_path.name.replace("_positive.graphml", "")
            input_paths.append(graph_path)
            G = nx.read_graphml(graph_path)
            nodes = sorted(G.nodes())
            if len(nodes) < 2:
                continue

            import igraph as ig

            idx = {n: i for i, n in enumerate(nodes)}
            edges = []
            weights = []
            for u, v, d in G.edges(data=True):
                edges.append((idx[u], idx[v]))
                weights.append(float(d.get("npmi_median", 1.0)))
            g = ig.Graph(n=len(nodes), edges=edges, directed=False)
            g.vs["name"] = nodes
            if weights:
                g.es["weight"] = weights

            partitions = []
            for gi, gamma in enumerate(gammas):
                for si in range(seeds_per):
                    seed = seed_base + gi * seeds_per + si
                    part = run_leiden_partition(g, float(gamma), seed)
                    partitions.append(part)
                    for node, cid in part.items():
                        row = {
                            "scheme": scheme.name,
                            "gamma": gamma,
                            "seed": seed,
                            "node": node,
                            "community_id": cid,
                            **stratum_row_fields(scheme.name, key_str, scheme.groupby),
                        }
                        raw_rows.append(row)

            C = coassignment_matrix(partitions, nodes)
            cons = consensus_partition(C, nodes, threshold, consensus_gamma, seed_base)
            stab_df = community_stability_scores(cons, partitions, nodes, stab_min)
            comm_map = _communities_summary(cons, nodes)
            n_stable = int(stab_df["stable"].sum()) if not stab_df.empty and "stable" in stab_df.columns else 0
            stratum_partition_stats.append({
                "scheme": scheme.name,
                "stratum_key": key_str,
                "n_communities": len(comm_map),
                "n_stable_communities": n_stable,
                "communities": {str(k): sorted(v) for k, v in comm_map.items()},
            })
            for node, cid in cons.items():
                row = {
                    "scheme": scheme.name,
                    "node": node,
                    "community_id": cid,
                    **stratum_row_fields(scheme.name, key_str, scheme.groupby),
                }
                consensus_rows.append(row)
            if not stab_df.empty:
                stab_df = stab_df.copy()
                stab_df["scheme"] = scheme.name
                meta = stratum_row_fields(scheme.name, key_str, scheme.groupby)
                for mk, mv in meta.items():
                    stab_df[mk] = mv
                stability_rows.append(stab_df)

    raw_df = pd.DataFrame(raw_rows)
    cons_df = pd.DataFrame(consensus_rows)
    stab_all = pd.concat(stability_rows, ignore_index=True) if stability_rows else pd.DataFrame()

    for scheme in cfg.schemes:
        if scheme_filter and scheme.name != scheme_filter:
            continue
        part_dir = cfg.scheme_dir("partitions", scheme.name, window_suffix)
        sub_raw = raw_df[raw_df["scheme"] == scheme.name] if not raw_df.empty else raw_df
        sub_cons = cons_df[cons_df["scheme"] == scheme.name] if not cons_df.empty else cons_df
        sub_stab = stab_all[stab_all["scheme"] == scheme.name] if not stab_all.empty else stab_all
        sub_raw.to_parquet(part_dir / "partitions_raw.parquet", index=False)
        sub_cons.to_parquet(part_dir / "consensus_partition.parquet", index=False)
        sub_stab.to_parquet(part_dir / "community_stability.parquet", index=False)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash(input_paths)})
    if window_suffix is None:
        write_summary(
            art,
            "p2_05",
            params={"gammas": gammas, "seeds_per_gamma": seeds_per, "consensus_gamma": consensus_gamma},
            outputs=[str(cfg.scheme_dir("partitions", s.name)) for s in cfg.schemes],
            stats={
                "n_raw_partitions": len(raw_rows),
                "n_consensus_nodes": len(consensus_rows),
                "stratum_partitions": stratum_partition_stats,
            },
            notes=[
                "RBConfigurationVertexPartition; edge weight=npmi_median.",
                f"stability(C)=mean co_assignment_freq(i,j) for i!=j in C across {len(gammas)*seeds_per} partitions.",
                f"consensus_gamma={consensus_gamma} (median of sweep; post-threshold graph).",
            ],
            elapsed_sec=time.perf_counter() - t0,
        )
    print(f"p2_05{manifest_suffix} done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
