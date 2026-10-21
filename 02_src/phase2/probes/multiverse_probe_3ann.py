#!/usr/bin/env python3
"""Multiverse with a third annotator: the 2-annotator multiverse (multiverse_probe.py) plus Claude.

Crossed factors
  L2 annotator : DeepSeek (8 binary calls) | GPT (one multi-label call) | Claude (one call, per-code decisions)
  unit         : sentence | document
  threshold    : none | co-occurrence n_ab >= 20 before testing
  estimand     : raw OR | lift | degree-preserving null | conditional OR with L1 from GPT / DS / Claude

Claude labels: 06_claude_reannotation/eval/full/analysis_frame_with_claude.csv (full run, L1 v9.1 / L2 v10.1).
Output: 15_multiverse_3ann/ (the 2-annotator results in 14_multiverse/ are left untouched).

Usage: python multiverse_probe_3ann.py [--reps N] [--final]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from corpus_composition_probe import load  # noqa: E402
from multiverse_probe import II, JJ, _counts, conditional_or, doc_level_null, lift, raw_or  # noqa: E402
from tier1_reestimation import PAIR_NAMES, PAIRS, ROOT, _fdsm_counts  # noqa: E402

REPO = os.path.dirname(os.path.dirname(ROOT))
# --final: Claude = main configuration (3-run majority, L2-07 code-specific); default: single run (historical)
FINAL = "--final" in sys.argv
OUT = f"{ROOT}/15_multiverse_3ann_final" if FINAL else f"{ROOT}/15_multiverse_3ann"
CLAUDE = (f"{REPO}/06_claude_reannotation/eval/full/analysis_frame_with_claude_final.csv" if FINAL
          else f"{REPO}/06_claude_reannotation/eval/full/analysis_frame_with_claude.csv")
L2COL = {"ds": "l2_set_ds", "gpt": "l2_set_gpt", "claude": "l2_set_claude"}


def main(reps: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    base = load()
    ds_l1 = pd.read_csv(f"{REPO}/L1_Deepseek/l1_results.csv", usecols=["unit_id", "L1_label"]).rename(
        columns={"L1_label": "L1_ds"})
    cl = pd.read_csv(CLAUDE, usecols=["unit_id", "L1_claude", "L2_claude"], keep_default_na=False).drop_duplicates("unit_id")
    cl["l2_set_claude"] = cl.L2_claude.str.replace("|", "+", regex=False)
    base = base.merge(ds_l1.drop_duplicates("unit_id"), on="unit_id", how="left")
    base = base.merge(cl[["unit_id", "L1_claude", "l2_set_claude"]], on="unit_id", how="left")
    assert base.l2_set_claude.notna().all(), "Claude labels missing for some analysis-frame rows"
    rows = []
    codes = [f"c{k:02d}" for k in range(1, 9)]
    for ann, col in L2COL.items():
        d = base.copy()
        for k in range(1, 9):
            d[f"c{k:02d}"] = d[col].fillna("").str.contains(f"L2-0{k}").astype(int)
        for unit in ["sentence", "document"]:
            if unit == "sentence":
                m = d[codes].to_numpy(np.int8)
                n11 = _counts(m)[1]
                o, nl = _fdsm_counts(d, reps, 11)
            else:
                m = d.groupby("doc_id")[codes].max().to_numpy(np.int8)
                n11 = _counts(m)[1]
                nl = doc_level_null(m, reps, 12)
                o = (m.astype(np.int64).T @ m.astype(np.int64))[II, JJ]
            null_mean = nl.mean(0)
            fdsm = np.where(null_mean > 0, o / np.maximum(null_mean, 1e-9), np.nan)
            fdsm_p = np.minimum(1.0, 2 * np.minimum(((nl >= o).sum(0) + 1) / (reps + 1),
                                                    ((nl <= o).sum(0) + 1) / (reps + 1)))
            specs = {"raw_OR": (raw_or(m), None), "lift": (lift(m), None), "null_model": (fdsm, fdsm_p)}
            if unit == "sentence":
                for l1 in ["L1_gpt", "L1_ds", "L1_claude"]:
                    c = conditional_or(d, codes, l1)
                    specs[f"conditional_OR({l1})"] = (c[:, 0], c[:, 1])
            for est, (val, pv) in specs.items():
                for thr in [False, True]:
                    keep = (n11 >= 20) if thr else np.ones(len(PAIRS), bool)
                    for k, name in enumerate(PAIR_NAMES):
                        rows.append({"pair": name, "annotator": ann, "unit": unit, "estimand": est,
                                     "threshold20": thr, "n_ab": int(n11[k]),
                                     "value": None if not keep[k] or np.isnan(val[k]) else round(float(val[k]), 4),
                                     "p": None if (pv is None or not keep[k] or np.isnan(pv[k])) else round(float(pv[k]), 4),
                                     "dropped_by_threshold": bool(not keep[k])})
            print(f"  done {ann} / {unit}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/multiverse_specs_3ann.csv", index=False)
    live = t[~t.dropped_by_threshold & t.value.notna()]

    def summarise(g):
        return pd.Series({"n_specs": len(g), "pct_positive": round((g.value > 1).mean(), 3),
                          "sign_stable": bool((g.value > 1).all() or (g.value < 1).all())})
    two = live[live.annotator.isin(["ds", "gpt"]) & ~live.estimand.eq("conditional_OR(L1_claude)")]
    s2 = two.groupby("pair").apply(summarise).add_suffix("_2ann")
    s3 = live.groupby("pair").apply(summarise).add_suffix("_3ann")
    sc = live[live.annotator == "claude"].groupby("pair").apply(summarise).add_suffix("_claude_only")
    summ = pd.concat([s2, s3, sc], axis=1)
    summ.to_csv(f"{OUT}/multiverse_pair_summary_3ann.csv")
    print(summ.sort_values("pct_positive_3ann").to_string())
    print(f"\nsign-stable pairs: 2 annotators {int(summ.sign_stable_2ann.sum())}/28 | "
          f"3 annotators {int(summ.sign_stable_3ann.sum())}/28 | Claude alone {int(summ.sign_stable_claude_only.sum())}/28")


if __name__ == "__main__":
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 300
    main(reps)
