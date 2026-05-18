#!/usr/bin/env python3
"""p2_11 — Temporal NPMI edge stability across election time buckets."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.io import read_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402

TIME_BUCKETS = ["pre_registration", "post_registration", "campaign", "election_plus"]


def _ci_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    """True if bootstrap CIs are disjoint (no overlap)."""
    if any(pd.isna(x) for x in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return hi_a < lo_b or hi_b < lo_a


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    scheme_name = "camp_time"
    if scheme_name not in [s.name for s in cfg.schemes]:
        print("p2_11: camp_time scheme not configured")
        return

    manifest_path = art / "manifests" / "p2_11.json"
    expected = {"corpus_content_hash": cfg.corpus_content_hash}
    if should_skip(manifest_path, expected, force):
        print("p2_11: skip (manifest match)")
        return

    boot_path = cfg.scheme_dir("npmi", scheme_name) / "npmi_bootstrap.parquet"
    boot = read_parquet(boot_path) if boot_path.is_file() else pd.DataFrame()
    if boot.empty:
        print("p2_11: no camp_time bootstrap data")
        return

    rows = []
    for (camp, l2_a, l2_b), grp in boot.groupby(["camp", "l2_a", "l2_b"], sort=True):
        by_time = {str(r["time_bucket"]): r for _, r in grp.iterrows()}
        present = [b for b in TIME_BUCKETS if b in by_time]
        for i, b1 in enumerate(present):
            for b2 in present[i + 1 :]:
                r1, r2 = by_time[b1], by_time[b2]
                disjoint = _ci_overlap(
                    float(r1["npmi_lower"]), float(r1["npmi_upper"]),
                    float(r2["npmi_lower"]), float(r2["npmi_upper"]),
                )
                rows.append({
                    "camp": camp,
                    "l2_a": l2_a,
                    "l2_b": l2_b,
                    "bucket_a": b1,
                    "bucket_b": b2,
                    "npmi_median_a": float(r1["npmi_median"]),
                    "npmi_median_b": float(r2["npmi_median"]),
                    "ci_disjoint": disjoint,
                    "fdr_a": bool(r1.get("fdr_significant", False)),
                    "fdr_b": bool(r2.get("fdr_significant", False)),
                })

    out_df = pd.DataFrame(rows)
    out_path = art / "temporal_edge_stability.parquet"
    out_df.to_parquet(out_path, index=False)

    n_disjoint = int(out_df["ci_disjoint"].sum()) if not out_df.empty else 0
    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([boot_path])})
    write_summary(
        art,
        "p2_11",
        params={"scheme": scheme_name, "time_buckets": TIME_BUCKETS},
        outputs=[str(out_path)],
        stats={
            "n_pairwise_comparisons": len(out_df),
            "n_ci_disjoint": n_disjoint,
            "disjoint_rate": round(n_disjoint / len(out_df), 4) if len(out_df) else 0,
        },
        notes=["ci_disjoint=True means 95% bootstrap CIs do not overlap between time buckets."],
        elapsed_sec=time.perf_counter() - t0,
    )
    print(f"p2_11 done: {len(out_df)} comparisons, {n_disjoint} disjoint")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
