"""Multiple-testing helpers for Phase 2 NPMI bootstrap edges."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def bootstrap_two_sided_p(samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value: H0 NPMI = 0."""
    valid = samples[~np.isnan(samples)]
    if len(valid) == 0:
        return 1.0
    p_le = float(np.mean(valid <= 0))
    p_ge = float(np.mean(valid >= 0))
    return min(1.0, 2.0 * min(p_le, p_ge))


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return (q_values, reject) for BH-FDR at level alpha."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev_q = min(prev_q, val)
        q[order[i]] = prev_q
    reject = q <= alpha
    return q, reject


def apply_fdr_to_edges(boot_df, alpha: float, p_col: str = "p_value") -> None:
    """Add q_value and fdr_significant columns in-place (per input frame = one stratum)."""
    if boot_df.empty or p_col not in boot_df.columns:
        boot_df["q_value"] = np.nan
        boot_df["fdr_significant"] = False
        return
    q, rej = benjamini_hochberg(boot_df[p_col].values, alpha=alpha)
    boot_df["q_value"] = q
    boot_df["fdr_significant"] = rej
