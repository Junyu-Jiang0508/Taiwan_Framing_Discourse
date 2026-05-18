#!/usr/bin/env python3
"""p2_03 — Document-cluster bootstrap CI for NPMI."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from joblib import Parallel, delayed

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.bootstrap import bootstrap_npmi  # noqa: E402
from utils.config import Phase2Config, StratificationScheme  # noqa: E402
from utils.io import read_parquet, write_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.multiple_testing import apply_fdr_to_edges  # noqa: E402
from utils.summary import write_summary  # noqa: E402


def _bootstrap_stratum(
    grp: pd.DataFrame,
    stratum_cols: dict,
    n_resamples: int,
    seed_base: int,
    min_marginal: int,
    ci_alpha: float,
    fdr_alpha: float,
):
    doc_ids = grp["doc_id"].unique()
    boot_df, diag_df = bootstrap_npmi(
        grp, doc_ids, n_resamples, seed_base, min_marginal, ci_alpha
    )
    if not boot_df.empty:
        apply_fdr_to_edges(boot_df, fdr_alpha)
    for k, v in stratum_cols.items():
        if not boot_df.empty:
            boot_df[k] = v
        if not diag_df.empty:
            diag_df[k] = v
    return boot_df, diag_df


def run_scheme(
    windows: pd.DataFrame,
    scheme: StratificationScheme,
    out_dir: Path,
    n_resamples: int,
    seed_base: int,
    min_marginal: int,
    ci_alpha: float,
    fdr_alpha: float,
    n_jobs: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    boot_parts = []
    diag_parts = []

    groups = list(windows.groupby(scheme.groupby, sort=True))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_stratum)(
            grp,
            dict(zip(scheme.groupby, keys if isinstance(keys, tuple) else (keys,))),
            n_resamples,
            seed_base,
            min_marginal,
            ci_alpha,
            fdr_alpha,
        )
        for keys, grp in groups
    )
    for (keys, _), (b, d) in zip(groups, results):
        if not b.empty:
            b["scheme"] = scheme.name
            boot_parts.append(b)
        if not d.empty:
            d["scheme"] = scheme.name
            diag_parts.append(d)

    boot_all = pd.concat(boot_parts, ignore_index=True) if boot_parts else pd.DataFrame()
    diag_all = pd.concat(diag_parts, ignore_index=True) if diag_parts else pd.DataFrame()
    sort_cols = scheme.groupby + ["l2_a", "l2_b"]
    write_parquet(boot_all, out_dir / "npmi_bootstrap.parquet", sort_by=sort_cols)
    write_parquet(diag_all, out_dir / "bootstrap_diagnostics.parquet", sort_by=sort_cols)
    return len(boot_all)


def run(
    cfg: Phase2Config,
    mode: str = "full",
    force: bool = False,
    scheme_filter: str | None = None,
    window_suffix: str | int | None = None,
) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    win_path = cfg.windows_parquet_path(window_suffix)
    n_resamples = cfg.bootstrap_n_for_mode(mode)
    bcfg = cfg.raw["bootstrap"]
    manifest_suffix = "" if window_suffix is None else f"_n{window_suffix}"
    manifest_path = art / "manifests" / f"p2_03_{mode}{manifest_suffix}.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "bootstrap_n": n_resamples,
        "mode": mode,
        "window_suffix": str(window_suffix),
        "fdr_alpha": cfg.fdr_alpha,
    }
    if should_skip(manifest_path, expected, force):
        print(f"p2_03 ({mode}){manifest_suffix}: skip (manifest match)")
        return

    windows = read_parquet(win_path)
    n_jobs = int(bcfg["n_jobs"])
    min_marginal = int(cfg.raw["npmi"]["min_marginal_count"])
    ci_alpha = float(bcfg["ci_alpha"])
    seed_base = int(bcfg["seed_base"])
    fdr_alpha = cfg.fdr_alpha
    counts = {}

    for scheme in cfg.schemes:
        if scheme_filter and scheme.name != scheme_filter:
            continue
        out_dir = cfg.scheme_dir("npmi", scheme.name, window_suffix)
        counts[scheme.name] = run_scheme(
            windows, scheme, out_dir, n_resamples, seed_base, min_marginal, ci_alpha, fdr_alpha, n_jobs
        )

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([win_path])})
    if mode == "full" and window_suffix is None:
        write_summary(
            art,
            "p2_03",
            params={
                "n_resamples": n_resamples,
                "ci_alpha": ci_alpha,
                "n_jobs": n_jobs,
                "fdr_alpha": fdr_alpha,
                "edge_selection": cfg.edge_selection,
            },
            outputs=[str(cfg.scheme_dir("npmi", s.name) / "npmi_bootstrap.parquet") for s in cfg.schemes],
            stats=counts,
            notes=["Resample unit = document (cluster bootstrap). BH-FDR applied per stratum."],
            elapsed_sec=time.perf_counter() - t0,
        )
    print(f"p2_03 ({mode}){manifest_suffix} done: B={n_resamples}")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), mode="smoke", force="--force" in sys.argv)
