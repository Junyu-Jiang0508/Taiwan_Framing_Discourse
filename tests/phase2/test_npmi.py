"""Unit tests for phase 2 NPMI utilities."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_src" / "phase2"))

from utils.io import l2_set_to_tuple  # noqa: E402
from utils.npmi import compute_npmi_table  # noqa: E402


def _npmi_from_probs(p_a: float, p_b: float, p_ab: float) -> float:
    pmi = math.log(p_ab) - math.log(p_a) - math.log(p_b)
    return pmi / (-math.log(p_ab))


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


def test_npmi_bounded_not_pmi():
    """NPMI must lie in [-1, 1]; raw PMI can exceed 1."""
    rows = []
    # 100 windows: 5 with both labels (p_ab = p_a = 0.05), 15 with only L2-08
    for _ in range(5):
        rows.append({"l2_set": l2_set_to_tuple({"L2-04", "L2-08"}), "is_empty_l2": False})
    for _ in range(15):
        rows.append({"l2_set": l2_set_to_tuple({"L2-08"}), "is_empty_l2": False})
    for _ in range(80):
        rows.append({"l2_set": l2_set_to_tuple(frozenset()), "is_empty_l2": True})
    w = pd.DataFrame(rows)
    npmi, _ = compute_npmi_table(w, min_marginal_count=1)
    row = npmi[(npmi["l2_a"] == "L2-04") & (npmi["l2_b"] == "L2-08")].iloc[0]
    expected = _npmi_from_probs(row["p_a"], row["p_b"], row["p_ab"])
    assert row["npmi"] == expected
    assert -1 <= row["npmi"] <= 1
    pmi = math.log(row["p_ab"]) - math.log(row["p_a"]) - math.log(row["p_b"])
    assert pmi > 1  # PMI inflation on sparse marginals; NPMI should not


if __name__ == "__main__":
    test_npmi_symmetric_off_diagonal()
    test_canonical_pair_order()
    print("test_npmi: ok")
