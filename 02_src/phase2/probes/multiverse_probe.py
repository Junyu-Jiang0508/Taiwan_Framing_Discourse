#!/usr/bin/env python3
"""Multiverse over the analysis chain: same corpus, same labels, 40 specifications per pair.

The point is not that one specification is right. It is that a relation whose sign flips
across defensible specifications is a product of the specification, not a finding.

Crossed factors
  L2 annotator : DeepSeek (8 independent binary calls) | GPT (one multi-label call)
  unit         : sentence | document
  threshold    : none | co-occurrence n_ab >= 20 before testing
  estimand     : raw OR | lift | degree-preserving null | conditional OR (L1=GPT) | conditional OR (L1=DS)

At document level the null fixes each document's number of distinct codes and each code's
document margin (the within-document design has nothing left to permute).

Usage: python multiverse_probe.py [--reps N] [--out DIR]
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from corpus_composition_probe import load  # noqa: E402
from tier1_reestimation import PAIR_NAMES, PAIRS, ROOT, _curveball, _fdsm_counts  # noqa: E402

OUT = f"{ROOT}/14_multiverse"
II = np.array([a for a, _ in PAIRS])
JJ = np.array([b for _, b in PAIRS])


def _counts(m: np.ndarray) -> tuple[np.ndarray, ...]:
    m = m.astype(np.int64)          # int8 matmul silently overflows at these counts
    n = len(m)
    C = m.T @ m
    c1 = m.sum(0)
    n11 = C[II, JJ]
    n1a, n1b = c1[II], c1[JJ]
    return n, n11, n1a - n11, n1b - n11, n - n1a - n1b + n11


def raw_or(m):
    n, n11, n10, n01, n00 = _counts(m)
    return ((n11 + .5) * (n00 + .5)) / ((n10 + .5) * (n01 + .5))


def lift(m):
    n, n11, n10, n01, _ = _counts(m)
    return (n11 / n) / np.maximum(((n11 + n10) / n) * ((n11 + n01) / n), 1e-12)


def doc_level_null(m: np.ndarray, reps: int, seed: int) -> np.ndarray:
    """Fixed-margin bipartite null on the document x code matrix."""
    rng = np.random.default_rng(seed)
    free = m[(m.sum(1) > 0) & (m.sum(1) < m.shape[1])]
    fixed = m[(m.sum(1) == 0) | (m.sum(1) == m.shape[1])]
    Cf = (fixed.astype(np.int64).T @ fixed.astype(np.int64))[II, JJ] if len(fixed) else np.zeros(len(PAIRS))
    out = np.empty((reps, len(PAIRS)))
    for r in range(reps):
        Mr = _curveball(free.copy().astype(np.int8), rng, 5 * len(free) + 10).astype(np.int64)
        out[r] = (Mr.T @ Mr)[II, JJ] + Cf
    return out


def conditional_or(d: pd.DataFrame, codes: list[str], l1col: str) -> np.ndarray:
    dd = d.dropna(subset=[l1col]).copy()
    dd["z_len"] = (dd["len"] - dd["len"].mean()) / dd["len"].std()
    L1 = pd.get_dummies(dd[l1col], prefix="L1", drop_first=True).astype(float)
    OUTL = pd.get_dummies(dd.outlet, prefix="o", drop_first=True).astype(float)
    res = []
    for i, j in PAIRS:
        a, b = codes[i], codes[j]
        X = sm.add_constant(pd.concat([dd[[a]].astype(float),
                                       dd[[c for c in codes if c not in (a, b)]].astype(float),
                                       L1, OUTL, dd[["z_len"]]], axis=1))
        try:
            m = sm.GLM(dd[b].astype(float), X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": dd.doc_id})
            res.append((float(np.exp(m.params[a])), float(m.pvalues[a])))
        except Exception:
            res.append((np.nan, np.nan))
    return np.array(res)


def main(reps: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    base = load()
    # the second L1 annotator is a factor in its own right (demonstration D3 of the paper)
    ds_l1 = pd.read_csv(f"{os.path.dirname(os.path.dirname(ROOT))}/L1_Deepseek/l1_results.csv",
                        usecols=["unit_id", "L1_label"]).rename(columns={"L1_label": "L1_ds"})
    base = base.merge(ds_l1.drop_duplicates("unit_id"), on="unit_id", how="left")
    print(f"L1_ds coverage {base.L1_ds.notna().mean():.3f}", flush=True)
    rows = []
    for ann in ["ds", "gpt"]:
        d = base.copy()
        col = "l2_set_ds" if ann == "ds" else "l2_set_gpt"
        codes = [f"c{k:02d}" for k in range(1, 9)]
        for k in range(1, 9):
            d[f"c{k:02d}"] = d[col].fillna("").str.contains(f"L2-0{k}").astype(int)
        for unit in ["sentence", "document"]:
            if unit == "sentence":
                m = d[codes].to_numpy(np.int8)
                n11 = _counts(m)[1]
                o, nl = _fdsm_counts(d, reps, 11)
                null_mean = nl.mean(0)
                fdsm = np.where(null_mean > 0, o / np.maximum(null_mean, 1e-9), np.nan)
                fdsm_p = np.minimum(1.0, 2 * np.minimum(((nl >= o).sum(0) + 1) / (reps + 1),
                                                        ((nl <= o).sum(0) + 1) / (reps + 1)))
                frame = d
            else:
                g = d.groupby("doc_id")[codes].max()
                m = g.to_numpy(np.int8)
                n11 = _counts(m)[1]
                nl = doc_level_null(m, reps, 12)
                o = (m.astype(np.int64).T @ m.astype(np.int64))[II, JJ]
                null_mean = nl.mean(0)
                fdsm = np.where(null_mean > 0, o / np.maximum(null_mean, 1e-9), np.nan)
                fdsm_p = np.minimum(1.0, 2 * np.minimum(((nl >= o).sum(0) + 1) / (reps + 1),
                                                        ((nl <= o).sum(0) + 1) / (reps + 1)))
                frame = None
            ro, li = raw_or(m), lift(m)
            specs = {"raw_OR": (ro, None), "lift": (li, None), "null_model": (fdsm, fdsm_p)}
            if unit == "sentence":
                for l1 in ["L1_gpt", "L1_ds"]:
                    if l1 == "L1_ds" and "L1_ds" not in frame.columns:
                        continue
                    c = conditional_or(frame, codes, l1)
                    specs[f"conditional_OR({l1})"] = (c[:, 0], c[:, 1])
            for est, (val, pv) in specs.items():
                for thr in [False, True]:
                    keep = (n11 >= 20) if thr else np.ones(len(PAIRS), bool)
                    for k, name in enumerate(PAIR_NAMES):
                        rows.append({"pair": name, "annotator": ann, "unit": unit, "estimand": est,
                                     "threshold20": thr, "n_ab": int(n11[k]),
                                     "value": None if not keep[k] else (None if np.isnan(val[k]) else round(float(val[k]), 4)),
                                     "p": None if (pv is None or not keep[k] or np.isnan(pv[k])) else round(float(pv[k]), 4),
                                     "dropped_by_threshold": bool(not keep[k])})
            print(f"  done {ann} / {unit}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/multiverse_specs.csv", index=False)
    live = t[~t.dropped_by_threshold & t.value.notna()]
    summ = live.groupby("pair").apply(lambda g: pd.Series({
        "n_specs": len(g),
        "pct_positive": round((g.value > 1).mean(), 3),
        "sign_stable": bool((g.value > 1).all() or (g.value < 1).all()),
    }))
    summ.to_csv(f"{OUT}/multiverse_pair_summary.csv")
    print(summ.sort_values("pct_positive").to_string())
    print(f"\npairs with the same sign in every specification: {int(summ.sign_stable.sum())}/28")


if __name__ == "__main__":
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 300
    main(reps)
