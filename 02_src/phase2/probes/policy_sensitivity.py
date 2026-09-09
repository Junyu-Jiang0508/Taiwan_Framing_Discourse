#!/usr/bin/env python3
"""Price the boilerplate policy choice: drop_residue vs drop_all vs no filter.

`drop_residue` keeps the prefix tier (a real sentence carrying a scrape prefix);
`drop_all` removes it too. This compares the three point-NPMI solutions so the
cost of keeping ~1.9k real sentences is a number rather than an argument.

Usage:
    python3 probes/policy_sensitivity.py NOFILTER_DIR DROP_RESIDUE_DIR DROP_ALL_DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

EDGE_KEY = ["camp", "l2_a", "l2_b"]
COH_PAIRS = [("L2-04", "L2-06"), ("L2-04", "L2-07"), ("L2-04", "L2-08"),
             ("L2-06", "L2-07"), ("L2-06", "L2-08"), ("L2-07", "L2-08")]


def _point(root: Path) -> pd.DataFrame:
    p = root / "npmi_by_camp" / "npmi_point.parquet"
    return pd.read_parquet(p)[EDGE_KEY + ["npmi", "n_windows"]]


def _pair_stats(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    m = a.merge(b, on=EDGE_KEY, suffixes=("_x", "_y"))
    ok = m["npmi_x"].notna() & m["npmi_y"].notna()
    d = (m["npmi_y"] - m["npmi_x"]).abs()
    coh = m.merge(pd.DataFrame(COH_PAIRS, columns=["l2_a", "l2_b"]), on=["l2_a", "l2_b"])
    return {
        "对比": f"{name_a} → {name_b}",
        "n 对": int(ok.sum()),
        "Pearson": pearsonr(m.loc[ok, "npmi_x"], m.loc[ok, "npmi_y"])[0],
        "Spearman": spearmanr(m.loc[ok, "npmi_x"], m.loc[ok, "npmi_y"])[0],
        "平均|Δ|": d.mean(),
        "最大|Δ|": d.max(),
        "凝聚簇平均|Δ|": (coh["npmi_y"] - coh["npmi_x"]).abs().mean(),
        "窗口数": int(m["n_windows_y"].groupby(m["camp"]).first().sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nofilter", type=Path)
    ap.add_argument("drop_residue", type=Path)
    ap.add_argument("drop_all", type=Path)
    args = ap.parse_args()

    nf, dr, da = _point(args.nofilter), _point(args.drop_residue), _point(args.drop_all)
    rows = [
        _pair_stats(nf, dr, "无过滤", "drop_residue"),
        _pair_stats(nf, da, "无过滤", "drop_all"),
        _pair_stats(dr, da, "drop_residue", "drop_all"),
    ]
    t = pd.DataFrame(rows).set_index("对比")
    print(t.round(4).to_string())
    print()
    print("读法：drop_residue→drop_all 那一行就是多删 1,874 个 prefix 层真句子买到的差别。")


if __name__ == "__main__":
    main()
