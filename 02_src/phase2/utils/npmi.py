"""Log-space NPMI estimation from window tables."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Set, Tuple

import numpy as np
import pandas as pd

from .io import tuple_to_l2_set


def count_label_presence(windows: pd.DataFrame) -> Tuple[int, Dict[str, int], Dict[Tuple[str, str], int]]:
    """Return N_windows, marginal counts, pair counts (excluding empty-l2 windows for pairs)."""
    n_total = len(windows)
    marg: Dict[str, int] = {}
    pairs: Dict[Tuple[str, str], int] = {}
    for _, row in windows.iterrows():
        l2s = tuple_to_l2_set(row["l2_set"])
        for lab in l2s:
            marg[lab] = marg.get(lab, 0) + 1
        if row.get("is_empty_l2", len(l2s) == 0):
            continue
        labs = sorted(l2s)
        for a, b in combinations(labs, 2):
            key = (a, b)
            pairs[key] = pairs.get(key, 0) + 1
    return n_total, marg, pairs


def compute_npmi_table(
    windows: pd.DataFrame,
    min_marginal_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Point NPMI for one stratum. Returns (npmi_df, excluded_labels_df)."""
    n_windows, marg, pair_counts = count_label_presence(windows)
    if n_windows == 0:
        return pd.DataFrame(), pd.DataFrame()

    valid_labels = {lab for lab, c in marg.items() if c >= min_marginal_count}
    excluded = [
        {"l2": lab, "count": marg[lab], "reason": "below_min_marginal_count"}
        for lab, c in marg.items()
        if lab not in valid_labels
    ]
    excluded_df = pd.DataFrame(excluded)

    log_n = math.log(n_windows)
    log_marg = {lab: math.log(marg[lab]) - log_n for lab in valid_labels}

    rows = []
    for (a, b), cab in pair_counts.items():
        if a not in valid_labels or b not in valid_labels:
            continue
        if cab == 0:
            rows.append({
                "l2_a": a, "l2_b": b,
                "p_a": math.exp(log_marg[a]),
                "p_b": math.exp(log_marg[b]),
                "p_ab": 0.0,
                "npmi": float("nan"),
                "n_windows": n_windows,
            })
            continue
        log_pab = math.log(cab) - log_n
        npmi = log_pab - log_marg[a] - log_marg[b]
        rows.append({
            "l2_a": a, "l2_b": b,
            "p_a": math.exp(log_marg[a]),
            "p_b": math.exp(log_marg[b]),
            "p_ab": math.exp(log_pab),
            "npmi": npmi,
            "n_windows": n_windows,
        })

    return pd.DataFrame(rows), excluded_df


def stratum_diagnostics(windows: pd.DataFrame) -> Dict[str, Any]:
    n = len(windows)
    empty = int(windows["is_empty_l2"].sum()) if "is_empty_l2" in windows.columns else 0
    all_l2: Set[str] = set()
    for ls in windows["l2_set"]:
        all_l2 |= set(tuple_to_l2_set(ls))
    return {
        "n_windows": n,
        "n_empty_l2_windows": empty,
        "n_unique_l2": len(all_l2),
    }
