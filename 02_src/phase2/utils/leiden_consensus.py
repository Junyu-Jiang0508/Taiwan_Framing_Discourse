"""Leiden community detection and Lancichinetti-Fortunato consensus."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import igraph as ig
    import leidenalg
except ImportError:
    ig = None  # type: ignore
    leidenalg = None  # type: ignore


def _require_igraph():
    if ig is None or leidenalg is None:
        raise ImportError("python-igraph and leidenalg are required for p2_05")


def graph_from_nx_path(G_nx) -> "ig.Graph":
    import networkx as nx

    G = nx.read_graphml(G_nx)
    nodes = sorted(G.nodes())
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
    return g


def run_leiden_partition(
    g: "ig.Graph",
    gamma: float,
    seed: int,
) -> Dict[str, int]:
    _require_igraph()
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=gamma,
        seed=seed,
    )
    names = g.vs["name"]
    return {names[i]: int(part.membership[i]) for i in range(len(names))}


def coassignment_matrix(partitions: List[Dict[str, int]], nodes: List[str]) -> np.ndarray:
    n = len(nodes)
    C = np.zeros((n, n), dtype=float)
    if not partitions:
        return C
    idx = {nodes[i]: i for i in range(n)}
    for part in partitions:
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i >= j:
                    continue
                if part.get(ni) == part.get(nj):
                    C[idx[ni], idx[nj]] += 1
                    C[idx[nj], idx[ni]] += 1
    C /= len(partitions)
    return C


def consensus_partition(
    C: np.ndarray,
    nodes: List[str],
    threshold: float,
    gamma: float,
    seed: int,
) -> Dict[str, int]:
    _require_igraph()
    n = len(nodes)
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= threshold:
                edges.append((i, j))
                weights.append(float(C[i, j]))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.vs["name"] = nodes
    if weights:
        g.es["weight"] = weights
    return run_leiden_partition(g, gamma, seed)


def community_stability_scores(
    consensus: Dict[str, int],
    partitions: List[Dict[str, int]],
    nodes: List[str],
    min_coassignment: float,
) -> pd.DataFrame:
    """stability(C) = mean co_assignment_freq(i,j) for i!=j in C."""
    n_parts = len(partitions)
    if n_parts == 0:
        return pd.DataFrame()

    # Precompute co-assignment freq per pair across partitions
    pair_freq: Dict[Tuple[str, str], float] = {}
    for part in partitions:
        for a, b in combinations(nodes, 2):
            key = (a, b) if a < b else (b, a)
            if part.get(a) == part.get(b):
                pair_freq[key] = pair_freq.get(key, 0) + 1
    for k in pair_freq:
        pair_freq[k] /= n_parts

    comm_to_nodes: Dict[int, List[str]] = {}
    for node, cid in consensus.items():
        comm_to_nodes.setdefault(cid, []).append(node)

    rows = []
    for cid, members in comm_to_nodes.items():
        if len(members) < 2:
            rows.append({
                "community_id": cid,
                "stability": 1.0 if len(members) == 1 else float("nan"),
                "stable": len(members) == 1,
                "n_members": len(members),
            })
            continue
        vals = []
        for a, b in combinations(members, 2):
            key = (a, b) if a < b else (b, a)
            vals.append(pair_freq.get(key, 0.0))
        stab = float(np.mean(vals))
        rows.append({
            "community_id": cid,
            "stability": stab,
            "stable": stab > min_coassignment,
            "n_members": len(members),
        })
    return pd.DataFrame(rows)
