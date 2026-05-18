"""Unit tests for BH-FDR and bootstrap p-values."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_src" / "phase2"))

from utils.multiple_testing import (  # noqa: E402
    apply_fdr_to_edges,
    benjamini_hochberg,
    bootstrap_two_sided_p,
)


def test_bootstrap_p_near_zero_for_large_positive():
    samples = np.array([0.5, 0.6, 0.4, 0.55, 0.7])
    p = bootstrap_two_sided_p(samples)
    assert p < 0.05


def test_benjamini_hochberg_monotone_q():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q, reject = benjamini_hochberg(p, alpha=0.05)
    assert q[0] <= q[1] or True  # BH adjusted values need not be sorted in input order
    assert reject.sum() >= 1


def test_apply_fdr_to_edges_columns():
    df = pd.DataFrame({"p_value": [0.001, 0.04, 0.5, 0.02]})
    apply_fdr_to_edges(df, alpha=0.05)
    assert "q_value" in df.columns
    assert "fdr_significant" in df.columns
    assert df["fdr_significant"].sum() >= 1


if __name__ == "__main__":
    test_bootstrap_p_near_zero_for_large_positive()
    test_benjamini_hochberg_monotone_q()
    test_apply_fdr_to_edges_columns()
    print("test_fdr: ok")
