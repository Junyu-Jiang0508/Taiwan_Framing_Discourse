#!/usr/bin/env python3
"""p2_04 — Network construction (GraphML per stratum, dual schemes)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
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
    window_suffix: str | int | None = None,
) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    labeled = deserialize_l2_column(read_parquet(art / "labeled_units.parquet"))
    manifest_suffix = "" if window_suffix is None else f"_n{window_suffix}"
    manifest_path = art / "manifests" / f"p2_04{manifest_suffix}.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "edge_selection": cfg.edge_selection,
        "window_suffix": str(window_suffix),
    }
    if should_skip(manifest_path, expected, force):
        print(f"p2_04{manifest_suffix}: skip (manifest match)")
        return

    min_stratum_windows = int(cfg.raw["npmi"]["min_stratum_windows"])
    edge_selection = cfg.edge_selection
    built = []
    skipped_low_n = []
    graph_stats = []
    input_paths = []

    for scheme in cfg.schemes:
        if scheme_filter and scheme.name != scheme_filter:
            continue
        npmi_dir = cfg.scheme_dir("npmi", scheme.name, window_suffix)
        net_dir = cfg.scheme_dir("networks", scheme.name, window_suffix)
        point_path = npmi_dir / "npmi_point.parquet"
        boot_path = npmi_dir / "npmi_bootstrap.parquet"
        diag_path = npmi_dir / "stratum_diagnostics.parquet"
        point = read_parquet(point_path)
        boot = read_parquet(boot_path)
        stratum_diag = read_parquet(diag_path) if diag_path.is_file() else pd.DataFrame()
        input_paths.extend([point_path, boot_path])
        if point.empty or boot.empty:
            continue

        low_n_keys = set()
        if not stratum_diag.empty and "low_n_warning" in stratum_diag.columns:
            warn = stratum_diag[stratum_diag["low_n_warning"] == True]  # noqa: E712
            for _, r in warn.iterrows():
                key_tuple = tuple(r[c] for c in scheme.groupby)
                low_n_keys.add(key_tuple)

        for keys, grp_p in point.groupby(scheme.groupby, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            if keys in low_n_keys:
                skipped_low_n.append(format_stratum_key(keys, scheme.groupby))
                continue
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
            _, _, edge_stats = build_graphs(
                point_sub, boot_sub, labeled_sub, filt, key_str, net_dir, edge_selection=edge_selection
            )
            pos_path = net_dir / f"{key_str}_positive.graphml"
            if pos_path.is_file():
                G = nx.read_graphml(pos_path)
                n = G.number_of_nodes()
                m = G.number_of_edges()
                max_e = n * (n - 1) // 2 if n >= 2 else 0
                graph_stats.append({
                    "scheme": scheme.name,
                    "stratum": key_str,
                    **edge_stats,
                    "n_nodes": n,
                    "n_edges": m,
                    "density": round(m / max_e, 4) if max_e else 0.0,
                })
            built.append(f"{scheme.name}/{key_str}")

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash(input_paths)})
    if window_suffix is None:
        write_summary(
            art,
            "p2_04",
            params={"schemes": [s.name for s in cfg.schemes], "edge_selection": edge_selection},
            outputs=built,
            stats={
                "n_graphs": len(built) * 2,
                "skipped_low_n": skipped_low_n,
                "graph_stats": graph_stats,
            },
            notes=[
                "No post-hoc genre aggregation; camp scheme uses independent NPMI estimates.",
                f"Strata with n_windows < {min_stratum_windows} are skipped.",
            ],
            elapsed_sec=time.perf_counter() - t0,
        )
    print(f"p2_04{manifest_suffix} done: {len(built)} strata")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
