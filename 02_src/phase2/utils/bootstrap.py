"""Document-cluster bootstrap for NPMI."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .npmi import compute_npmi_table


def bootstrap_npmi(
    windows: pd.DataFrame,
    doc_ids: np.ndarray,
    n_resamples: int,
    seed_base: int,
    min_marginal_count: int,
    ci_alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster bootstrap: resample documents with replacement, filter windows, recompute NPMI."""
    if len(windows) == 0 or len(doc_ids) == 0:
        return pd.DataFrame(), pd.DataFrame()

    unique_docs = np.array(sorted(set(doc_ids)))
    n_docs = len(unique_docs)
    doc_to_idx = {d: i for i, d in enumerate(unique_docs)}
    window_doc_idx = windows["doc_id"].map(doc_to_idx).values

    # Collect all pair keys from point estimate universe
    point_df, _ = compute_npmi_table(windows, min_marginal_count)
    if point_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pair_keys = list(zip(point_df["l2_a"], point_df["l2_b"]))
    samples = np.full((n_resamples, len(pair_keys)), np.nan, dtype=float)
    effective_counts = np.zeros(len(pair_keys), dtype=int)

    lo_q = ci_alpha / 2
    hi_q = 1 - ci_alpha / 2

    for i in range(n_resamples):
        rng = np.random.default_rng(seed_base + i)
        drawn = rng.choice(n_docs, size=n_docs, replace=True)
        # Map drawn doc indices to window mask
        allowed = set(drawn.tolist())
        mask = np.isin(window_doc_idx, list(allowed))
        sub = windows.loc[mask]
        if len(sub) == 0:
            continue
        res_df, _ = compute_npmi_table(sub, min_marginal_count)
        if res_df.empty:
            continue
        lookup = {
            (r["l2_a"], r["l2_b"]): r["npmi"]
            for _, r in res_df.iterrows()
        }
        for j, key in enumerate(pair_keys):
            val = lookup.get(key, float("nan"))
            if key in lookup and not (isinstance(val, float) and np.isnan(val)):
                effective_counts[j] += 1
            samples[i, j] = val

    rows = []
    diag_rows = []
    for j, (a, b) in enumerate(pair_keys):
        col = samples[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            med = lo = hi = float("nan")
            ci_excl = False
        else:
            med = float(np.median(valid))
            lo = float(np.quantile(valid, lo_q))
            hi = float(np.quantile(valid, hi_q))
            ci_excl = (lo > 0) or (hi < 0)
        rows.append({
            "l2_a": a, "l2_b": b,
            "npmi_median": med,
            "npmi_lower": lo,
            "npmi_upper": hi,
            "ci_excludes_zero": ci_excl,
            "n_resamples": n_resamples,
        })
        diag_rows.append({
            "l2_a": a, "l2_b": b,
            "bootstrap_mean": float(np.nanmean(col)) if len(valid) else float("nan"),
            "bootstrap_std": float(np.nanstd(col)) if len(valid) else float("nan"),
            "effective_resample_count": int(effective_counts[j]),
        })

    return pd.DataFrame(rows), pd.DataFrame(diag_rows)
