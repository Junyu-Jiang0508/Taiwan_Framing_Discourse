#!/usr/bin/env python3
"""p2_06 — Cross-camp community matching (Jaccard) and QAP appendix."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.io import read_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def communities_from_partition(cons: pd.DataFrame, camp: str) -> Dict[int, Set[str]]:
    sub = cons[cons["camp"] == camp]
    out: Dict[int, Set[str]] = {}
    for _, r in sub.iterrows():
        out.setdefault(int(r["community_id"]), set()).add(str(r["node"]))
    return out


def top_k_communities(comm_map: Dict[int, Set[str]], k: int) -> List[Tuple[int, Set[str]]]:
    ranked = sorted(comm_map.items(), key=lambda x: len(x[1]), reverse=True)
    return ranked[:k]


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


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    ccfg = cfg.raw["cross_camp"]
    scheme_name = ccfg.get("primary_network_scheme", "camp")
    top_k = int(ccfg["top_k_communities"])
    n_perm = int(ccfg["qap_permutations"])

    manifest_path = art / "manifests" / "p2_06.json"
    expected = {"corpus_content_hash": cfg.corpus_content_hash}
    if should_skip(manifest_path, expected, force):
        print("p2_06: skip (manifest match)")
        return

    part_dir = cfg.scheme_dir("partitions", scheme_name)
    cons_path = part_dir / "consensus_partition.parquet"
    cons = read_parquet(cons_path) if cons_path.is_file() else pd.DataFrame()
    net_dir = cfg.scheme_dir("networks", scheme_name)
    out_dir = art / "cross_camp"
    out_dir.mkdir(parents=True, exist_ok=True)

    camps = cfg.camps
    if cons.empty or "camp" not in cons.columns:
        pd.DataFrame().to_parquet(out_dir / "community_table.parquet", index=False)
        pd.DataFrame().to_parquet(out_dir / "qap_results.parquet", index=False)
        write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([cons_path])})
        write_summary(
            art / "p2_06_summary.md", "p2_06",
            params={"scheme": scheme_name}, outputs=[str(out_dir)],
            stats={"skipped": "no consensus partition"}, elapsed_sec=time.perf_counter() - t0,
        )
        print("p2_06 skipped (no partitions)")
        return

    comm_by_camp = {c: top_k_communities(communities_from_partition(cons, c), top_k) for c in camps}

    match_rows = []
    for dpp_id, dpp_set in comm_by_camp.get("DPP", []):
        best_kmt, best_kmt_j = None, -1.0
        best_tpp, best_tpp_j = None, -1.0
        for kmt_id, kmt_set in comm_by_camp.get("KMT", []):
            j = jaccard(dpp_set, kmt_set)
            if j > best_kmt_j:
                best_kmt_j, best_kmt = j, kmt_id
        for tpp_id, tpp_set in comm_by_camp.get("TPP", []):
            j = jaccard(dpp_set, tpp_set)
            if j > best_tpp_j:
                best_tpp_j, best_tpp = j, tpp_id
        match_rows.append({
            "dpp_community_id": dpp_id,
            "dpp_l2_set": "|".join(sorted(dpp_set)),
            "kmt_community_id": best_kmt,
            "kmt_jaccard": best_kmt_j,
            "tpp_community_id": best_tpp,
            "tpp_jaccard": best_tpp_j,
        })

    pd.DataFrame(match_rows).to_parquet(out_dir / "community_table.parquet", index=False)

    # QAP on shared L2 nodes (signed graphs)
    all_nodes = sorted({str(r["node"]) for _, r in cons.iterrows()})
    if len(all_nodes) >= 2:
        pair_labels = [("DPP", "KMT"), ("DPP", "TPP"), ("KMT", "TPP")]
        qap_rows = []
        mats = {}
        for camp in camps:
            gp = net_dir / f"{camp}_signed.graphml"
            if gp.is_file():
                mats[camp] = adjacency_from_graph(gp, all_nodes)
        for a, b in pair_labels:
            if a in mats and b in mats:
                gamma, p = qap_test(mats[a], mats[b], n_perm, seed=42)
                qap_rows.append({"camp_a": a, "camp_b": b, "hubert_gamma": gamma, "p_value": p, "n_perm": n_perm})
        pd.DataFrame(qap_rows).to_parquet(out_dir / "qap_results.parquet", index=False)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([part_dir / "consensus_partition.parquet"])})
    write_summary(
        art / "p2_06_summary.md",
        "p2_06",
        params={"scheme": scheme_name, "top_k": top_k, "qap_permutations": n_perm},
        outputs=[str(out_dir / "community_table.parquet"), str(out_dir / "qap_results.parquet")],
        stats={"n_matched_triples": len(match_rows)},
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_06 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
