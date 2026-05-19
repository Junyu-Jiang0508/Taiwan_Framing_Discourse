#!/usr/bin/env python3
"""p2_14 — Sub-network restricted analysis (differentiating vs shared infrastructure).

Node grouping is exploratory: communities with cross-camp Jaccard ≥ threshold against
both KMT and TPP are labeled shared; remaining nodes form the differentiating sub-network.
No Leiden rerun on 4-node subgraphs — reports edge weights, ranks, and Spearman ρ.
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.cross_camp import adjacency_from_graph, hubert_gamma  # noqa: E402
from utils.io import read_parquet, write_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402

CAMP_PAIRS = [("DPP", "KMT"), ("DPP", "TPP"), ("KMT", "TPP")]


def _communities_from_partition(cons: pd.DataFrame, camp: str) -> Dict[int, Set[str]]:
    sub = cons[cons["camp"] == camp]
    out: Dict[int, Set[str]] = {}
    for _, r in sub.iterrows():
        out.setdefault(int(r["community_id"]), set()).add(str(r["node"]))
    return out


def _best_cross_camp_jaccard(
    jaccard_df: pd.DataFrame,
    camp_a: str,
    comm_a: int,
    camp_b: str,
) -> float:
    sub = jaccard_df[
        (jaccard_df["camp_a"] == camp_a)
        & (jaccard_df["community_a"] == comm_a)
        & (jaccard_df["camp_b"] == camp_b)
    ]
    if sub.empty:
        return 0.0
    return float(sub["jaccard"].max())


def select_subnet_nodes(
    cons: pd.DataFrame,
    jaccard_df: pd.DataFrame,
    threshold: float,
) -> Tuple[Set[str], Set[str], List[Dict]]:
    all_nodes = sorted({str(r["node"]) for _, r in cons.iterrows()})
    dpp_comms = _communities_from_partition(cons, "DPP")
    shared_nodes: Set[str] = set()
    audit_rows = []

    for comm_id, nodes in dpp_comms.items():
        best_kmt = _best_cross_camp_jaccard(jaccard_df, "DPP", comm_id, "KMT")
        best_tpp = _best_cross_camp_jaccard(jaccard_df, "DPP", comm_id, "TPP")
        is_shared = best_kmt >= threshold and best_tpp >= threshold
        if is_shared:
            shared_nodes |= nodes
        audit_rows.append({
            "dpp_community_id": comm_id,
            "l2_set": "|".join(sorted(nodes)),
            "best_kmt_jaccard": best_kmt,
            "best_tpp_jaccard": best_tpp,
            "category": "shared" if is_shared else "differentiating",
        })

    differentiating_nodes = set(all_nodes) - shared_nodes
    return shared_nodes, differentiating_nodes, audit_rows


def _subnet_edges(nodes: List[str]) -> List[Tuple[str, str]]:
    return list(combinations(sorted(nodes), 2))


def _edge_weights_and_ranks(
    mats: Dict[str, np.ndarray],
    node_list: List[str],
    camps: List[str],
) -> pd.DataFrame:
    idx = {n: i for i, n in enumerate(node_list)}
    rows = []
    for l2_a, l2_b in _subnet_edges(node_list):
        for camp in camps:
            mat = mats[camp]
            w = float(mat[idx[l2_a], idx[l2_b]])
            rows.append({"camp": camp, "l2_a": l2_a, "l2_b": l2_b, "npmi_weight": w})
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("camp")["npmi_weight"].rank(ascending=False, method="average")
    return df


def _spearman_for_subnet(
    mats: Dict[str, np.ndarray],
    node_list: List[str],
    camp_a: str,
    camp_b: str,
) -> float:
    idx = {n: i for i, n in enumerate(node_list)}
    edges = _subnet_edges(node_list)
    if len(edges) < 2:
        return float("nan")
    wa = [float(mats[camp_a][idx[a], idx[b]]) for a, b in edges]
    wb = [float(mats[camp_b][idx[a], idx[b]]) for a, b in edges]
    if np.std(wa) == 0 or np.std(wb) == 0:
        return float("nan")
    rho, _ = spearmanr(wa, wb)
    return float(rho) if rho == rho else float("nan")


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    ccfg = cfg.raw["cross_camp"]
    scheme_name = ccfg.get("primary_network_scheme", "camp")
    threshold = float(ccfg.get("shared_jaccard_threshold", 0.66))
    camps = cfg.camps

    out_dir = art / "subnetwork"
    audit_path = out_dir / "node_selection_audit.parquet"
    diff_path = out_dir / "differentiating_subnet_stats.parquet"
    shared_path = out_dir / "shared_subnet_stats.parquet"
    rank_path = out_dir / "rank_correlation_comparison.parquet"

    part_dir = cfg.scheme_dir("partitions", scheme_name)
    net_dir = cfg.scheme_dir("networks", scheme_name)
    cross_dir = art / "cross_camp"
    cons_path = part_dir / "consensus_partition.parquet"
    jacc_path = cross_dir / "jaccard_all_pairs.parquet"
    qap_path = cross_dir / "qap_results.parquet"

    manifest_path = art / "manifests" / "p2_14.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "edge_selection": cfg.edge_selection,
        "shared_jaccard_threshold": threshold,
    }
    if should_skip(manifest_path, expected, force):
        print("p2_14: skip (manifest match)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cons = read_parquet(cons_path) if cons_path.is_file() else pd.DataFrame()
    jaccard_df = read_parquet(jacc_path) if jacc_path.is_file() else pd.DataFrame()
    qap = read_parquet(qap_path) if qap_path.is_file() else pd.DataFrame()

    if cons.empty or jaccard_df.empty:
        for p in (audit_path, diff_path, shared_path, rank_path):
            write_parquet(pd.DataFrame(), p)
        write_manifest(
            manifest_path,
            {**expected, "inputs_hash": inputs_hash([cons_path, jacc_path, qap_path])},
        )
        print("p2_14 skipped (missing upstream data)")
        return

    shared_nodes, diff_nodes, audit_rows = select_subnet_nodes(cons, jaccard_df, threshold)
    write_parquet(pd.DataFrame(audit_rows), audit_path)

    all_nodes = sorted({str(r["node"]) for _, r in cons.iterrows()})
    mats = {}
    for camp in camps:
        gp = net_dir / f"{camp}_signed.graphml"
        if gp.is_file():
            mats[camp] = adjacency_from_graph(gp, all_nodes)

    diff_list = sorted(diff_nodes)
    shared_list = sorted(shared_nodes)
    diff_stats = _edge_weights_and_ranks(mats, diff_list, camps) if diff_list and mats else pd.DataFrame()
    shared_stats = _edge_weights_and_ranks(mats, shared_list, camps) if shared_list and mats else pd.DataFrame()
    write_parquet(diff_stats, diff_path)
    write_parquet(shared_stats, shared_path)

    rank_rows = []
    for a, b in CAMP_PAIRS:
        if a in mats and b in mats:
            full_gamma = float("nan")
            if not qap.empty:
                match = qap[(qap["camp_a"] == a) & (qap["camp_b"] == b)]
                if len(match):
                    full_gamma = float(match.iloc[0]["hubert_gamma"])
            else:
                full_gamma = hubert_gamma(mats[a], mats[b])
            rank_rows.append({
                "scope": "full",
                "camp_a": a,
                "camp_b": b,
                "metric": "hubert_gamma",
                "value": full_gamma,
            })
            if diff_list:
                rank_rows.append({
                    "scope": "differentiating",
                    "camp_a": a,
                    "camp_b": b,
                    "metric": "spearman_rho",
                    "value": _spearman_for_subnet(mats, diff_list, a, b),
                })
            if shared_list:
                rank_rows.append({
                    "scope": "shared",
                    "camp_a": a,
                    "camp_b": b,
                    "metric": "spearman_rho",
                    "value": _spearman_for_subnet(mats, shared_list, a, b),
                })

    write_parquet(pd.DataFrame(rank_rows), rank_path)

    write_manifest(
        manifest_path,
        {**expected, "inputs_hash": inputs_hash([cons_path, jacc_path, qap_path])},
    )
    write_summary(
        art,
        "p2_14",
        params={"shared_jaccard_threshold": threshold, "scheme": scheme_name},
        outputs=[str(audit_path), str(diff_path), str(shared_path), str(rank_path)],
        stats={
            "shared_nodes": sorted(shared_nodes),
            "differentiating_nodes": sorted(diff_nodes),
            "rank_correlation_comparison": rank_rows,
        },
        notes=[
            "Node selection is exploratory (Jaccard ≥ threshold vs both KMT and TPP); "
            "informs structured analysis, not a confirmatory test on the same data.",
        ],
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_14 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
