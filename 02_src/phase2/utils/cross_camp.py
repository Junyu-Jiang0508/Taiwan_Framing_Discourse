"""Cross-camp network comparison utilities (Hubert gamma, adjacency builders)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def hubert_gamma(a: np.ndarray, b: np.ndarray) -> float:
    n = a.shape[0]
    if n < 2:
        return float("nan")
    triu_i, triu_j = np.triu_indices(n, k=1)
    x = a[triu_i, triu_j]
    y = b[triu_i, triu_j]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def qap_test(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> Tuple[float, float]:
    obs = hubert_gamma(a, b)
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        bp = b[np.ix_(perm, perm)]
        null.append(hubert_gamma(a, bp))
    null = np.array([x for x in null if not np.isnan(x)])
    if len(null) == 0 or np.isnan(obs):
        return obs, float("nan")
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    return obs, p


def adjacency_from_npmi(npmi_df: pd.DataFrame, nodes: List[str]) -> np.ndarray:
    """Build weighted adjacency from full NPMI table (missing pairs → 0.0)."""
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    mat = np.zeros((n, n))
    if npmi_df.empty:
        return mat
    for _, r in npmi_df.iterrows():
        a, b = str(r["l2_a"]), str(r["l2_b"])
        if a in idx and b in idx:
            w = float(r["npmi"]) if pd.notna(r.get("npmi")) else 0.0
            i, j = idx[a], idx[b]
            mat[i, j] = w
            mat[j, i] = w
    return mat


def adjacency_from_graph(path: Path, nodes: List[str]) -> np.ndarray:
    G = nx.read_graphml(path)
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}
    mat = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            w = float(d.get("npmi_median", 0))
            i, j = idx[u], idx[v]
            mat[i, j] = w
            mat[j, i] = w
    return mat


def empirical_p_value(observed: float, null: np.ndarray, side: str = "lower") -> float:
    """One-sided empirical p-value from permutation null."""
    null = null[~np.isnan(null)]
    if len(null) == 0 or np.isnan(observed):
        return float("nan")
    if side == "lower":
        return float((np.sum(null <= observed) + 1) / (len(null) + 1))
    if side == "upper":
        return float((np.sum(null >= observed) + 1) / (len(null) + 1))
    raise ValueError(f"Unknown side: {side}")
