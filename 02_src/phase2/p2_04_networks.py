#!/usr/bin/env python3
"""p2_04 — Network construction (GraphML per stratum, dual schemes)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.camp_genre import format_stratum_key  # noqa: E402
from utils.config import Phase2Config  # noqa: E402
from utils.io import deserialize_l2_column, read_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.networks import build_graphs  # noqa: E402
from utils.summary import write_summary  # noqa: E402


def run(
    cfg: Phase2Config,
    force: bool = False,
    scheme_filter: str | None = None,
) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    labeled = deserialize_l2_column(read_parquet(art / "labeled_units.parquet"))
    manifest_path = art / "manifests" / "p2_04.json"
    expected = {"corpus_content_hash": cfg.corpus_content_hash}
    if should_skip(manifest_path, expected, force):
        print("p2_04: skip (manifest match)")
        return

    built = []
    input_paths = []

    for scheme in cfg.schemes:
        if scheme_filter and scheme.name != scheme_filter:
            continue
        npmi_dir = cfg.scheme_dir("npmi", scheme.name)
        net_dir = cfg.scheme_dir("networks", scheme.name)
        point_path = npmi_dir / "npmi_point.parquet"
        boot_path = npmi_dir / "npmi_bootstrap.parquet"
        point = read_parquet(point_path)
        boot = read_parquet(boot_path)
        input_paths.extend([point_path, boot_path])
        if point.empty or boot.empty:
            continue
        for keys, grp_p in point.groupby(scheme.groupby, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            stratum = dict(zip(scheme.groupby, keys))
            key_str = format_stratum_key(keys, scheme.groupby)
            boot_sub = boot
            for k, v in stratum.items():
                boot_sub = boot_sub[boot_sub[k] == v]
            point_sub = grp_p
            filt = {k: v for k, v in stratum.items()}
            labeled_sub = labeled.copy()
            for k, v in filt.items():
                labeled_sub = labeled_sub[labeled_sub[k] == v]
            if scheme.name == "camp" and "genre" not in scheme.groupby:
                labeled_sub = labeled_sub[labeled_sub["genre"].isin(cfg.genres_main)]
            build_graphs(point_sub, boot_sub, labeled_sub, filt, key_str, net_dir)
            built.append(f"{scheme.name}/{key_str}")

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash(input_paths)})
    write_summary(
        art / "p2_04_summary.md",
        "p2_04",
        params={"schemes": [s.name for s in cfg.schemes]},
        outputs=built,
        stats={"n_graphs": len(built) * 2},
        notes=["No post-hoc genre aggregation; camp scheme uses independent NPMI estimates."],
        elapsed_sec=time.perf_counter() - t0,
    )
    print(f"p2_04 done: {len(built)} strata")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
