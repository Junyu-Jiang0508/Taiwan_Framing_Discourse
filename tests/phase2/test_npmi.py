"""Unit tests for phase 2 NPMI utilities."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_src" / "phase2"))

from utils.io import l2_set_to_tuple  # noqa: E402
from utils.npmi import compute_npmi_table  # noqa: E402


def test_npmi_symmetric_off_diagonal():
    rows = [
        {"l2_set": l2_set_to_tuple({"L2-01", "L2-02"}), "is_empty_l2": False},
        {"l2_set": l2_set_to_tuple({"L2-01", "L2-02"}), "is_empty_l2": False},
        {"l2_set": l2_set_to_tuple({"L2-01"}), "is_empty_l2": False},
        {"l2_set": l2_set_to_tuple(frozenset()), "is_empty_l2": True},
    ]
    w = pd.DataFrame(rows)
    npmi, _ = compute_npmi_table(w, min_marginal_count=1)
    assert not npmi.empty
    ab = npmi[(npmi["l2_a"] == "L2-01") & (npmi["l2_b"] == "L2-02")]
    assert len(ab) == 1
    assert ab.iloc[0]["npmi"] > 0


def test_canonical_pair_order():  # noqa: D103
    rows = [{"l2_set": l2_set_to_tuple({"L2-02", "L2-01"}), "is_empty_l2": False}] * 5
    w = pd.DataFrame(rows)
    npmi, _ = compute_npmi_table(w, min_marginal_count=1)
    for _, r in npmi.iterrows():
        assert r["l2_a"] < r["l2_b"]


if __name__ == "__main__":
    test_npmi_symmetric_off_diagonal()
    test_canonical_pair_order()
    print("test_npmi: ok")
