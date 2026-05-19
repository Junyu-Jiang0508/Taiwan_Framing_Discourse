#!/usr/bin/env python3
"""p2_13 — Per-edge cross-camp CI overlap (conservative differentiation test).

We use non-overlapping 95% bootstrap CIs as a conservative test of cross-camp
edge difference. Reads camp-stratified bootstrap output from p2_03 (npmi_by_camp).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.io import read_parquet, write_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402

CAMP_PAIRS = [("DPP", "KMT"), ("DPP", "TPP"), ("KMT", "TPP")]


def _ci_disjoint(row_a: pd.Series, row_b: pd.Series) -> bool:
    return bool(row_a["npmi_upper"] < row_b["npmi_lower"] or row_b["npmi_upper"] < row_a["npmi_lower"])


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    bcfg = cfg.raw["bootstrap"]
    ci_alpha = float(bcfg["ci_alpha"])
    n_resamples = int(bcfg["n_resamples"])

    boot_path = cfg.scheme_dir("npmi", "camp") / "npmi_bootstrap.parquet"
    cross_dir = art / "cross_camp"
    edge_path = cross_dir / "edge_diff_pairwise.parquet"
    node_path = cross_dir / "diff_by_l2_node.parquet"

    manifest_path = art / "manifests" / "p2_13.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "edge_selection": cfg.edge_selection,
        "bootstrap_n_resamples": n_resamples,
        "ci_alpha": ci_alpha,
    }
    if should_skip(manifest_path, expected, force):
        print("p2_13: skip (manifest match)")
        return

    cross_dir.mkdir(parents=True, exist_ok=True)
    boot = read_parquet(boot_path) if boot_path.is_file() else pd.DataFrame()
    if boot.empty:
        write_parquet(pd.DataFrame(), edge_path)
        write_parquet(pd.DataFrame(), node_path)
        write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([boot_path])})
        print("p2_13 skipped (no bootstrap data)")
        return

    if "scheme" in boot.columns:
        boot = boot[boot["scheme"] == "camp"].copy()

    edge_rows = []
    for camp_a, camp_b in CAMP_PAIRS:
        sub_a = boot[boot["camp"] == camp_a].set_index(["l2_a", "l2_b"])
        sub_b = boot[boot["camp"] == camp_b].set_index(["l2_a", "l2_b"])
        common = sub_a.index.intersection(sub_b.index)
        for l2_a, l2_b in common:
            ra = sub_a.loc[(l2_a, l2_b)]
            rb = sub_b.loc[(l2_a, l2_b)]
            if isinstance(ra, pd.DataFrame):
                ra = ra.iloc[0]
            if isinstance(rb, pd.DataFrame):
                rb = rb.iloc[0]
            disjoint = _ci_disjoint(ra, rb)
            direction = None
            if disjoint:
                direction = camp_a if ra["npmi_median"] > rb["npmi_median"] else camp_b
            edge_rows.append({
                "l2_a": l2_a,
                "l2_b": l2_b,
                "camp_a": camp_a,
                "camp_b": camp_b,
                "ci_disjoint": disjoint,
                "direction": direction,
                f"npmi_median_{camp_a.lower()}": float(ra["npmi_median"]),
                f"npmi_median_{camp_b.lower()}": float(rb["npmi_median"]),
                f"npmi_lower_{camp_a.lower()}": float(ra["npmi_lower"]),
                f"npmi_upper_{camp_a.lower()}": float(ra["npmi_upper"]),
                f"npmi_lower_{camp_b.lower()}": float(rb["npmi_lower"]),
                f"npmi_upper_{camp_b.lower()}": float(rb["npmi_upper"]),
            })

    edge_df = pd.DataFrame(edge_rows)
    write_parquet(edge_df, edge_path)

    node_rows = []
    if not edge_df.empty:
        diff_edges = edge_df[edge_df["ci_disjoint"] == True]  # noqa: E712
        all_nodes = sorted(set(edge_df["l2_a"]) | set(edge_df["l2_b"]))
        for camp_a, camp_b in CAMP_PAIRS:
            pair_diff = diff_edges[(diff_edges["camp_a"] == camp_a) & (diff_edges["camp_b"] == camp_b)]
            for node in all_nodes:
                n_incident = int(
                    ((pair_diff["l2_a"] == node) | (pair_diff["l2_b"] == node)).sum()
                )
                node_rows.append({
                    "l2_node": node,
                    "camp_a": camp_a,
                    "camp_b": camp_b,
                    "n_differentiating_edges": n_incident,
                })

    write_parquet(pd.DataFrame(node_rows), node_path)

    n_disjoint = int(edge_df["ci_disjoint"].sum()) if not edge_df.empty else 0
    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([boot_path])})
    write_summary(
        art,
        "p2_13",
        params={"ci_alpha": ci_alpha, "bootstrap_n_resamples": n_resamples},
        outputs=[str(edge_path), str(node_path)],
        stats={"n_edges_tested": len(edge_df), "n_ci_disjoint": n_disjoint},
        notes=[
            "Non-overlapping 95% bootstrap CIs used as conservative cross-camp edge difference test.",
        ],
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_13 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
