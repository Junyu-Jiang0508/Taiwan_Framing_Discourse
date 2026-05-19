#!/usr/bin/env python3
"""p2_06 — Cross-camp community matching (Jaccard) and QAP appendix.

QAP γ measures against-random distinctness, not between-camp distinctness;
the latter is tested by p2_12_camp_permutation.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.cross_camp import adjacency_from_graph, qap_test  # noqa: E402
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


def all_pair_jaccard(
    camps: List[str],
    comm_by_camp: Dict[str, List[Tuple[int, Set[str]]]],
) -> pd.DataFrame:
    rows = []
    for camp_a in camps:
        for camp_b in camps:
            for id_a, set_a in comm_by_camp.get(camp_a, []):
                for id_b, set_b in comm_by_camp.get(camp_b, []):
                    rows.append({
                        "camp_a": camp_a,
                        "community_a": id_a,
                        "l2_set_a": "|".join(sorted(set_a)),
                        "camp_b": camp_b,
                        "community_b": id_b,
                        "l2_set_b": "|".join(sorted(set_b)),
                        "jaccard": jaccard(set_a, set_b),
                    })
    return pd.DataFrame(rows)


def community_category(kmt_j: float, tpp_j: float, threshold: float) -> str:
    kmt_ok = kmt_j >= threshold
    tpp_ok = tpp_j >= threshold
    if kmt_ok and tpp_ok:
        return "shared"
    if kmt_ok or tpp_ok:
        return "partial"
    return "distinct"


def stability_lookup(stab: pd.DataFrame, camp: str, community_id: Optional[int]) -> Optional[bool]:
    if stab.empty or community_id is None:
        return None
    sub = stab[(stab["camp"] == camp) & (stab["community_id"] == community_id)]
    if sub.empty:
        return None
    return bool(sub.iloc[0]["stable"])


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    ccfg = cfg.raw["cross_camp"]
    scheme_name = ccfg.get("primary_network_scheme", "camp")
    top_k = int(ccfg["top_k_communities"])
    n_perm = int(ccfg["qap_permutations"])
    jaccard_threshold = float(ccfg.get("shared_jaccard_threshold", 0.66))

    manifest_path = art / "manifests" / "p2_06.json"
    expected = {"corpus_content_hash": cfg.corpus_content_hash, "edge_selection": cfg.edge_selection}
    if should_skip(manifest_path, expected, force):
        print("p2_06: skip (manifest match)")
        return

    part_dir = cfg.scheme_dir("partitions", scheme_name)
    cons_path = part_dir / "consensus_partition.parquet"
    stab_path = part_dir / "community_stability.parquet"
    cons = read_parquet(cons_path) if cons_path.is_file() else pd.DataFrame()
    stab = read_parquet(stab_path) if stab_path.is_file() else pd.DataFrame()
    net_dir = cfg.scheme_dir("networks", scheme_name)
    out_dir = art / "cross_camp"
    out_dir.mkdir(parents=True, exist_ok=True)

    camps = cfg.camps
    if cons.empty or "camp" not in cons.columns:
        pd.DataFrame().to_parquet(out_dir / "community_table.parquet", index=False)
        pd.DataFrame().to_parquet(out_dir / "qap_results.parquet", index=False)
        write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([cons_path])})
        write_summary(
            art, "p2_06",
            params={"scheme": scheme_name}, outputs=[str(out_dir)],
            stats={"skipped": "no consensus partition"}, elapsed_sec=time.perf_counter() - t0,
        )
        print("p2_06 skipped (no partitions)")
        return

    comm_by_camp_full = {c: list(communities_from_partition(cons, c).items()) for c in camps}
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
            "category": community_category(best_kmt_j, best_tpp_j, jaccard_threshold),
            "dpp_is_stable": stability_lookup(stab, "DPP", dpp_id),
            "kmt_is_stable": stability_lookup(stab, "KMT", best_kmt),
            "tpp_is_stable": stability_lookup(stab, "TPP", best_tpp),
        })

    pd.DataFrame(match_rows).to_parquet(out_dir / "community_table.parquet", index=False)
    jaccard_df = all_pair_jaccard(camps, {c: comm_by_camp_full[c] for c in camps})
    jaccard_df.to_parquet(out_dir / "jaccard_all_pairs.parquet", index=False)

    all_nodes = sorted({str(r["node"]) for _, r in cons.iterrows()})
    qap_rows = []
    if len(all_nodes) >= 2:
        pair_labels = [("DPP", "KMT"), ("DPP", "TPP"), ("KMT", "TPP")]
        mats = {}
        for camp in camps:
            gp = net_dir / f"{camp}_signed.graphml"
            if gp.is_file():
                mats[camp] = adjacency_from_graph(gp, all_nodes)
        for a, b in pair_labels:
            if a in mats and b in mats:
                gamma, p = qap_test(mats[a], mats[b], n_perm, seed=42)
                qap_rows.append({
                    "camp_a": a,
                    "camp_b": b,
                    "hubert_gamma": gamma,
                    "p_value": p,
                    "n_perm": n_perm,
                    "n_nodes": len(all_nodes),
                })
        pd.DataFrame(qap_rows).to_parquet(out_dir / "qap_results.parquet", index=False)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([cons_path, stab_path])})
    write_summary(
        art,
        "p2_06",
        params={"scheme": scheme_name, "top_k": top_k, "qap_permutations": n_perm},
        outputs=[
            str(out_dir / "community_table.parquet"),
            str(out_dir / "jaccard_all_pairs.parquet"),
            str(out_dir / "qap_results.parquet"),
        ],
        stats={
            "n_matched_triples": len(match_rows),
            "community_table": match_rows,
            "qap_results": qap_rows,
            "jaccard_all_pairs_n": len(jaccard_df),
        },
        notes=[
            "Report Hubert gamma as effect size; p-values on 28 edge pairs have limited power.",
            "QAP γ measures against-random distinctness, not between-camp distinctness; "
            "the latter is tested by p2_12_camp_permutation.py.",
        ],
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_06 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
