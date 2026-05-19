#!/usr/bin/env python3
"""p2_12 — Camp-level permutation test for cross-camp Hubert γ.

H0: camp labels are independent of co-articulation structure. Doc IDs are shuffled
among camps (preserving per-camp document counts); windows inherit shuffled camp via
doc_id. Each permutation recomputes NPMI on the full 28-edge table (no FDR inside
permutations — the test targets the γ distribution, not individual edge significance).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.cross_camp import adjacency_from_npmi, empirical_p_value, hubert_gamma  # noqa: E402
from utils.io import deserialize_l2_column, read_parquet, tuple_to_l2_set, write_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.npmi import compute_npmi_table  # noqa: E402
from utils.summary import write_summary  # noqa: E402

CAMP_PAIRS = [("DPP", "KMT"), ("DPP", "TPP"), ("KMT", "TPP")]


def _build_shuffled_camp_map(
    doc_ids_by_camp: Dict[str, List[str]],
    camps: List[str],
    rng: np.random.Generator,
) -> Dict[str, str]:
    slot_labels: List[str] = []
    all_doc_ids: List[str] = []
    for camp in camps:
        docs = doc_ids_by_camp[camp]
        slot_labels.extend([camp] * len(docs))
        all_doc_ids.extend(docs)
    perm_docs = list(all_doc_ids)
    rng.shuffle(perm_docs)
    return dict(zip(perm_docs, slot_labels))


def _gamma_for_camps(
    windows: pd.DataFrame,
    camp_map: Dict[str, str],
    camps: List[str],
    nodes: List[str],
    min_marginal: int,
) -> Dict[str, float]:
    windows = windows.copy()
    windows["camp_shuffled"] = windows["doc_id"].map(camp_map)
    mats = {}
    for camp in camps:
        sub = windows[windows["camp_shuffled"] == camp]
        npmi_df, _ = compute_npmi_table(sub, min_marginal)
        mats[camp] = adjacency_from_npmi(npmi_df, nodes)
    out = {}
    for a, b in CAMP_PAIRS:
        key = f"gamma_{a.lower()}_{b.lower()}"
        out[key] = hubert_gamma(mats[a], mats[b])
    return out


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    ccfg = cfg.raw["cross_camp"]
    n_perm = int(ccfg.get("camp_permutation_n", 1000))
    seed_base = int(ccfg.get("camp_permutation_seed_base", 42))
    min_marginal = int(cfg.raw["npmi"]["min_marginal_count"])
    camps = cfg.camps

    windows_path = cfg.windows_parquet_path()
    cross_dir = art / "cross_camp"
    qap_path = cross_dir / "qap_results.parquet"
    null_path = cross_dir / "permutation_null_distribution.parquet"
    summary_path = cross_dir / "permutation_summary.parquet"

    manifest_path = art / "manifests" / "p2_12.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "n_permutations": n_perm,
        "seed_base": seed_base,
        "min_marginal_count": min_marginal,
    }
    if should_skip(manifest_path, expected, force):
        print("p2_12: skip (manifest match)")
        return

    cross_dir.mkdir(parents=True, exist_ok=True)
    windows = deserialize_l2_column(read_parquet(windows_path))
    if windows.empty:
        write_parquet(pd.DataFrame(), null_path)
        write_parquet(pd.DataFrame(), summary_path)
        write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([windows_path, qap_path])})
        print("p2_12 skipped (no windows)")
        return

    all_nodes = sorted({lab for ls in windows["l2_set"] for lab in tuple_to_l2_set(ls)})
    doc_ids_by_camp: Dict[str, List[str]] = {}
    for camp in camps:
        doc_ids_by_camp[camp] = sorted(windows.loc[windows["camp"] == camp, "doc_id"].unique())

    null_rows = []
    for i in range(n_perm):
        rng = np.random.default_rng(seed_base + i)
        camp_map = _build_shuffled_camp_map(doc_ids_by_camp, camps, rng)
        gammas = _gamma_for_camps(windows, camp_map, camps, all_nodes, min_marginal)
        null_rows.append({"perm_id": i, **gammas})

    null_df = pd.DataFrame(null_rows)
    write_parquet(null_df, null_path)

    qap = read_parquet(qap_path) if qap_path.is_file() else pd.DataFrame()
    summary_rows = []
    observed_gammas = []
    null_means = []

    for a, b in CAMP_PAIRS:
        key = f"gamma_{a.lower()}_{b.lower()}"
        null_vals = null_df[key].values.astype(float)
        obs = float("nan")
        if not qap.empty:
            match = qap[(qap["camp_a"] == a) & (qap["camp_b"] == b)]
            if len(match):
                obs = float(match.iloc[0]["hubert_gamma"])
        p_val = empirical_p_value(obs, null_vals, side="lower")
        summary_rows.append({
            "test": key,
            "camp_a": a,
            "camp_b": b,
            "observed_gamma": obs,
            "null_mean": float(np.nanmean(null_vals)),
            "null_q05": float(np.nanquantile(null_vals, 0.05)),
            "null_q50": float(np.nanquantile(null_vals, 0.50)),
            "null_q95": float(np.nanquantile(null_vals, 0.95)),
            "p_value": p_val,
            "n_permutations": n_perm,
        })
        if not np.isnan(obs):
            observed_gammas.append(obs)
        null_means.append(float(np.nanmean(null_vals)))

    if observed_gammas and null_means:
        obs_joint = float(np.mean(observed_gammas))
        null_joint = null_df[[f"gamma_{a.lower()}_{b.lower()}" for a, b in CAMP_PAIRS]].mean(axis=1).values
        summary_rows.append({
            "test": "joint_mean_gamma",
            "camp_a": None,
            "camp_b": None,
            "observed_gamma": obs_joint,
            "null_mean": float(np.nanmean(null_joint)),
            "null_q05": float(np.nanquantile(null_joint, 0.05)),
            "null_q50": float(np.nanquantile(null_joint, 0.50)),
            "null_q95": float(np.nanquantile(null_joint, 0.95)),
            "p_value": empirical_p_value(obs_joint, null_joint, side="lower"),
            "n_permutations": n_perm,
        })

    summary_df = pd.DataFrame(summary_rows)
    write_parquet(summary_df, summary_path)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([windows_path, qap_path])})
    write_summary(
        art,
        "p2_12",
        params={"n_permutations": n_perm, "seed_base": seed_base, "min_marginal_count": min_marginal},
        outputs=[str(null_path), str(summary_path)],
        stats={"permutation_summary": summary_rows},
        notes=[
            "Permutation shuffles doc_id→camp assignments (preserving counts); windows inherit via doc_id.",
            "Full 28-edge NPMI table used inside permutations (no FDR) — test targets γ, not edge significance.",
        ],
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_12 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
