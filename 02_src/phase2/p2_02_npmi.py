#!/usr/bin/env python3
"""p2_02 — NPMI point estimates (dual stratification schemes)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config, StratificationScheme  # noqa: E402
from utils.io import read_parquet, write_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.npmi import compute_npmi_table, stratum_diagnostics  # noqa: E402
from utils.summary import write_summary  # noqa: E402

EMPTY_L2_DENOM_NOTE = (
    "Empty-L2 windows are retained in the denominator to estimate P(A) as the "
    "unconditional probability of label presence across all articulatory sites in "
    "the stratum, consistent with the theoretical claim that absence of articulation "
    "is itself meaningful."
)


def run_scheme(
    windows: pd.DataFrame,
    scheme: StratificationScheme,
    out_dir: Path,
    min_marginal: int,
    min_stratum_windows: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    npmi_rows = []
    diag_rows = []
    excl_rows = []

    for keys, grp in windows.groupby(scheme.groupby, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stratum = dict(zip(scheme.groupby, keys))
        diag = stratum_diagnostics(grp)
        diag.update(stratum)
        diag["scheme"] = scheme.name
        if diag["n_windows"] < min_stratum_windows:
            diag["low_n_warning"] = True
        diag_rows.append(diag)

        npmi_df, excl_df = compute_npmi_table(grp, min_marginal)
        if npmi_df.empty:
            continue
        for k, v in stratum.items():
            npmi_df[k] = v
            if not excl_df.empty:
                excl_df[k] = v
        npmi_df["scheme"] = scheme.name
        npmi_rows.append(npmi_df)
        if not excl_df.empty:
            excl_df["scheme"] = scheme.name
            excl_rows.append(excl_df)

    npmi_all = pd.concat(npmi_rows, ignore_index=True) if npmi_rows else pd.DataFrame()
    diag_all = pd.DataFrame(diag_rows)
    excl_all = pd.concat(excl_rows, ignore_index=True) if excl_rows else pd.DataFrame()

    sort_cols = scheme.groupby + ["l2_a", "l2_b"]
    write_parquet(npmi_all, out_dir / "npmi_point.parquet", sort_by=sort_cols)
    write_parquet(diag_all, out_dir / "stratum_diagnostics.parquet", sort_by=scheme.groupby)
    write_parquet(excl_all, out_dir / "excluded_labels.parquet", sort_by=scheme.groupby + ["l2"])
    return {"n_edges": len(npmi_all), "n_strata": len(diag_rows)}


def run(
    cfg: Phase2Config,
    force: bool = False,
    scheme_filter: str | None = None,
) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    win_path = art / "windows" / f"windows_n{cfg.raw['windows']['default_n']}.parquet"
    manifest_path = art / "manifests" / "p2_02.json"
    npmi_cfg = cfg.raw["npmi"]
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "npmi": str(npmi_cfg),
    }
    if should_skip(manifest_path, expected, force):
        print("p2_02: skip (manifest match)")
        return

    windows = read_parquet(win_path)
    min_marginal = int(npmi_cfg["min_marginal_count"])
    min_stratum = int(npmi_cfg["min_stratum_windows"])
    stats = {}

    for scheme in cfg.schemes:
        if scheme_filter and scheme.name != scheme_filter:
            continue
        out_dir = cfg.scheme_dir("npmi", scheme.name)
        stats[scheme.name] = run_scheme(windows, scheme, out_dir, min_marginal, min_stratum)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([win_path])})
    write_summary(
        art / "p2_02_summary.md",
        "p2_02",
        params={"min_marginal_count": min_marginal, "schemes": [s.name for s in cfg.schemes]},
        outputs=[str(cfg.scheme_dir("npmi", s.name)) for s in cfg.schemes],
        stats=stats,
        notes=[
            EMPTY_L2_DENOM_NOTE,
            "Pair counts exclude is_empty_l2 windows; marginals include all windows.",
            "Step=1 window overlap inflates counts equally in P(A), P(B), P(AB) — ratio cancels in NPMI.",
        ],
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_02 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
