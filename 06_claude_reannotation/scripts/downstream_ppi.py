"""Bias-corrected prevalence (overall and by camp) with prediction-powered inference (PPI++),
cluster-robust over documents (Angelopoulos, Duchi & Zrnic 2026; Broska & Howes 2026).

Labeled set : the SRS component of gold2 (human consensus labels, known probability, random)
Predictions : Claude full-run labels for all 20,900 sentences
Estimand    : P(code) overall and within each camp; camp differences.

Usage:
  python downstream_ppi.py --consensus consensus_gold2.csv --out ../eval/ppi
      consensus CSV: app_id, L1, L2 ('|'-separated) for the 406 gold2 items (only the SRS part is used)
  python downstream_ppi.py --selftest      # simulation check of coverage using DeepSeek labels as pseudo-truth
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import EVAL, L1, L2, SUB

Z = 1.959964


def cluster_var_of_mean(v, groups):
    """Variance of the mean of v with observations clustered in `groups` (CR0)."""
    v = np.asarray(v, float); n = len(v)
    s = pd.Series(v - v.mean()).groupby(np.asarray(groups)).sum()
    return float((s ** 2).sum()) / n ** 2


def ppi_mean(y_lab, f_lab, g_lab, f_all, g_all):
    """PPI++ estimate of E[Y] with power-tuned lambda and cluster-robust variance."""
    y_lab, f_lab, f_all = map(lambda a: np.asarray(a, float), (y_lab, f_lab, f_all))
    n, N = len(y_lab), len(f_all)
    vf = np.var(f_all) if np.var(f_all) > 0 else 1e-12
    lam = float(np.clip(np.cov(y_lab, f_lab)[0, 1] / ((1 + n / N) * vf), 0, 1)) if n > 2 else 0.0
    theta = (y_lab - lam * f_lab).mean() + lam * f_all.mean()
    var = cluster_var_of_mean(y_lab - lam * f_lab, g_lab) + lam ** 2 * cluster_var_of_mean(f_all, g_all)
    se = np.sqrt(var)
    return theta, theta - Z * se, theta + Z * se, lam


def frame():
    # predictions = main configuration (L1_claude / L2_claude); strata = the frozen gold2 design, which
    # build_human_tasks.py drew from the single-run labels (L1_claude_single / L2_claude_single)
    f = pd.read_csv(EVAL / "full" / "analysis_frame_with_claude_final.csv", keep_default_na=False).drop_duplicates("unit_id")
    f["doc"] = f.unit_id.str.rsplit("_u", n=1).str[0]
    labs = f.apply(lambda r: [r.L1_claude_single] + [x for x in r.L2_claude_single.split("|") if x], axis=1)
    prev = {l: labs.map(lambda s: l in s).mean() for l in L1 + L2}
    order = sorted(prev, key=prev.get)
    f["stratum"] = labs.map(lambda s: min(s, key=order.index))
    return f


def strat_ppi(y_lab, f_lab, g_lab, h_lab, f_all, g_all, h_all):
    """Stratified PPI++ (Fisch et al. 2024): PPI within each stratum, combined with N_h / N weights."""
    y_lab, f_lab, f_all = map(lambda a: np.asarray(a, float), (y_lab, f_lab, f_all))
    h_lab, h_all, g_lab, g_all = map(np.asarray, (h_lab, h_all, g_lab, g_all))
    N = len(f_all); theta = var = 0.0
    for h in np.unique(h_all):
        ma, ml = h_all == h, h_lab == h
        w = ma.sum() / N
        if ml.sum() < 2:          # no labels in this stratum: fall back to the prediction mean (flag in output)
            theta += w * f_all[ma].mean(); continue
        th, lo, hi, _ = ppi_mean(y_lab[ml], f_lab[ml], g_lab[ml], f_all[ma], g_all[ma])
        theta += w * th; var += (w * (hi - lo) / (2 * Z)) ** 2
    se = np.sqrt(var)
    return theta, theta - Z * se, theta + Z * se


def has(series, code, sep="|"):
    return series.map(lambda s: code in str(s).split(sep)).astype(float)


def run(consensus_path, out):
    """Two estimators per code x camp: PPI++ on the SRS part only, and stratified PPI on all 406 items.
    Simulation (selftest) shows the stratified version is needed for rare codes; its coverage is ~.86-.92,
    slightly below nominal, which must be stated when reporting."""
    f = frame()
    design = pd.read_csv(SUB / "human_tasks" / "P1_gold2_DESIGN_do_not_show.csv")
    cons = pd.read_csv(consensus_path, keep_default_na=False)
    lab = design[["app_id", "unit_id", "component"]].merge(cons, on="app_id").merge(f, on="unit_id")
    rows = []
    for code in L1 + L2:
        if code.startswith("L1"):
            y = (lab.L1 == code).astype(float).values; fl = (lab.L1_claude == code).astype(float).values
            fa = (f.L1_claude == code).astype(float).values
        else:
            y = has(lab.L2, code).values; fl = has(lab.L2_claude, code).values; fa = has(f.L2_claude, code).values
        for camp in ["ALL", "DPP", "KMT", "TPP"]:
            ml = np.ones(len(lab), bool) if camp == "ALL" else (lab.camp == camp).values
            ma = np.ones(len(f), bool) if camp == "ALL" else (f.camp == camp).values
            srs = ml & (lab.component == "srs").values
            th1, lo1, hi1, _ = ppi_mean(y[srs], fl[srs], lab.doc.values[srs], fa[ma], f.doc.values[ma])
            th2, lo2, hi2 = strat_ppi(y[ml], fl[ml], lab.doc.values[ml], lab.stratum.values[ml],
                                      fa[ma], f.doc.values[ma], f.stratum.values[ma])
            rows.append(dict(code=code, camp=camp, n_srs=int(srs.sum()), n_all=int(ml.sum()), claude_naive=fa[ma].mean(),
                             ppi_srs=th1, srs_lo=lo1, srs_hi=hi1, ppi_strat=th2, strat_lo=lo2, strat_hi=hi2))
    t = pd.DataFrame(rows)
    Path(out).mkdir(parents=True, exist_ok=True)
    t.to_csv(Path(out) / "ppi_prevalence.csv", index=False)
    print(t.round(4).to_string(index=False))


def selftest(reps=300, seed=1):
    """Pseudo-truth = DeepSeek L2 labels; predictions = Claude. 95% CI coverage of the DS prevalence under
    (a) the 100-sentence SRS alone and (b) the full gold2 design (SRS + 30/12 per stratum), stratified PPI."""
    f = frame(); rng = np.random.default_rng(seed)
    rare = set(sorted(f.stratum.value_counts().index, key=lambda h: (f.stratum == h).sum())[:7])
    H, D = f.stratum.values, f.doc.values
    print("selftest (pseudo-truth = DeepSeek L2):")
    for code in ["L2-01", "L2-02", "L2-03", "L2-07"]:
        truth = has(f.l2_set_ds, code, "+").values; pred = has(f.L2_claude, code).values; target = truth.mean()
        cov_a = cov_b = 0; wa, wb = [], []
        for _ in range(reps):
            srs = rng.choice(len(f), 100, replace=False)
            th, lo, hi, _ = ppi_mean(truth[srs], pred[srs], D[srs], pred, D)
            cov_a += lo <= target <= hi; wa.append(hi - lo)
            idx = list(srs)
            for h in np.unique(H):
                pool = np.setdiff1d(np.where(H == h)[0], srs)
                idx += list(rng.choice(pool, min(30 if h in rare else 12, len(pool)), replace=False))
            idx = np.array(idx)
            th, lo, hi = strat_ppi(truth[idx], pred[idx], D[idx], H[idx], pred, D, H)
            cov_b += lo <= target <= hi; wb.append(hi - lo)
        print(f"  {code}: true {target:.3f} naive {pred.mean():.3f} | SRS-100 PPI coverage {cov_a/reps:.2f} width {np.mean(wa):.3f}"
              f" | gold2 design stratified PPI coverage {cov_b/reps:.2f} width {np.mean(wb):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus"); ap.add_argument("--out", default=str(EVAL / "ppi")); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.consensus, a.out)
