#!/usr/bin/env python3
"""p2_10 — Compare NPMI rankings across window sizes (robustness)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from scipy.stats import spearmanr

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.io import read_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.networks import edge_is_significant  # noqa: E402
from utils.summary import write_summary  # noqa: E402


def _suffix_label(n) -> str:
    return "document" if n == "document" else str(n)


def _load_boot(cfg: Phase2Config, scheme: str, window_suffix) -> pd.DataFrame:
    path = cfg.scheme_dir("npmi", scheme, window_suffix) / "npmi_bootstrap.parquet"
    return read_parquet(path) if path.is_file() else pd.DataFrame()


def compare_stratum(
    ref: pd.DataFrame,
    alt: pd.DataFrame,
    groupby: List[str],
    stratum_keys: tuple,
    edge_selection: str,
) -> dict:
    ref_sub = ref
    alt_sub = alt
    for col, val in zip(groupby, stratum_keys):
        ref_sub = ref_sub[ref_sub[col] == val]
        alt_sub = alt_sub[alt_sub[col] == val]
    if ref_sub.empty or alt_sub.empty:
        return {}
    merged = ref_sub.merge(
        alt_sub,
        on=["l2_a", "l2_b"],
        suffixes=("_ref", "_alt"),
        how="inner",
    )
    if merged.empty:
        return {}

    def _sig_pairs(df: pd.DataFrame, suffix: str) -> set:
        pairs = set()
        for _, r in df.iterrows():
            row = {
                "ci_excludes_zero": r.get(f"ci_excludes_zero{suffix}", False),
                "fdr_significant": r.get(f"fdr_significant{suffix}", False),
            }
            if edge_is_significant(pd.Series(row), edge_selection):
                pairs.add((r["l2_a"], r["l2_b"]))
        return pairs

    rho, _ = spearmanr(merged["npmi_median_ref"], merged["npmi_median_alt"])
    sig_ref = _sig_pairs(merged, "_ref")
    sig_alt = _sig_pairs(merged, "_alt")
    j_edges = len(sig_ref & sig_alt) / len(sig_ref | sig_alt) if (sig_ref | sig_alt) else 1.0
    return {
        "spearman_rho": float(rho) if rho == rho else float("nan"),
        "n_pairs": len(merged),
        "sig_edge_jaccard": round(j_edges, 4),
        "n_sig_ref": len(sig_ref),
        "n_sig_alt": len(sig_alt),
    }


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    wcfg = cfg.raw["windows"]
    default_n = wcfg["default_n"]
    robustness_ns = list(wcfg.get("robustness_ns", []))
    manifest_path = art / "manifests" / "p2_10.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "robustness_ns": str(robustness_ns),
        "edge_selection": cfg.edge_selection,
    }
    if should_skip(manifest_path, expected, force):
        print("p2_10: skip (manifest match)")
        return

    ref_suffix = None
    rows = []
    schemes = [s for s in cfg.schemes if s.name in ("camp", "camp_genre")]

    for scheme in schemes:
        ref_boot = _load_boot(cfg, scheme.name, ref_suffix)
        if ref_boot.empty:
            continue
        strata = ref_boot.groupby(scheme.groupby).groups.keys()
        for alt_n in robustness_ns:
            if alt_n == default_n:
                continue
            alt_suffix = alt_n
            alt_boot = _load_boot(cfg, scheme.name, alt_suffix)
            if alt_boot.empty:
                continue
            for keys in strata:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                stats = compare_stratum(
                    ref_boot, alt_boot, scheme.groupby, keys, cfg.edge_selection
                )
                if stats:
                    row = {
                        "scheme": scheme.name,
                        "ref_window": _suffix_label(default_n),
                        "alt_window": _suffix_label(alt_n),
                        **{c: keys[i] for i, c in enumerate(scheme.groupby)},
                        **stats,
                    }
                    rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = art / "robustness_summary.parquet"
    out_df.to_parquet(out_path, index=False)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([out_path])})
    write_summary(
        art,
        "p2_10",
        params={"robustness_ns": robustness_ns, "ref_n": default_n},
        outputs=[str(out_path)],
        stats={"n_comparisons": len(out_df), "summary": out_df.to_dict(orient="records") if len(out_df) else []},
        elapsed_sec=time.perf_counter() - t0,
    )
    print(f"p2_10 done: {len(out_df)} comparisons")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
