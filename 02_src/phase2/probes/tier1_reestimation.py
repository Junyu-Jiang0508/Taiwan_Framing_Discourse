#!/usr/bin/env python3
"""Tier-1 re-estimation: fixes that need no new annotation.

A  fdsm       degree-preserving bipartite null for all 28 L2 pairs (pooled / camp / genre)
B  layera     Layer A as load + composition, with cluster bootstrap CIs
C  mundlak    within-document vs between-document decomposition of the L1->L2 coupling
D  threshold  re-apply co-occurrence thresholds before FDR in p2_03, then recheck p2_13
E  campperm   camp permutation test on the Layer A composition matrix (p2_12 spec)

Usage: python tier1_reestimation.py [A|B|C|D|E|all] [--reps N]
"""
from __future__ import annotations

import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/Junyu/work/Taiwan_Framing_Discourse/03_outputs/01_results_labelings"
FRAME = f"{ROOT}/10_merged_master/analysis_frame_n20901.csv"
OUT = f"{ROOT}/11_tier1_reestimation"
CODES = [f"L2-0{k}" for k in range(1, 9)]
PAIRS = list(itertools.combinations(range(8), 2))
PAIR_NAMES = [f"{a+1:02d}-{b+1:02d}" for a, b in PAIRS]


def load() -> pd.DataFrame:
    d = pd.read_csv(FRAME)
    for k in range(1, 9):
        d[f"c{k:02d}"] = d.l2_set_ds.fillna("").str.contains(f"L2-0{k}").astype(int)
    d = d.drop_duplicates(["doc_id", "sent_idx"]).sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)
    d["len"] = d.sent_text.str.len()
    d["lenq"] = pd.qcut(d["len"], 4, labels=False)
    d["load"] = d[[f"c{k:02d}" for k in range(1, 9)]].sum(axis=1)
    return d


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for rank, i in enumerate(order[::-1]):
        running = min(running, p[i] * n / (n - rank))
        q[i] = running
    return q


# ---------------------------------------------------------------- A: FDSM null

def _curveball(M: np.ndarray, rng: np.random.Generator, n_swap: int) -> np.ndarray:
    n = M.shape[0]
    for _ in range(n_swap):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        a, b = M[i], M[j]
        diff = np.flatnonzero(a != b)
        if len(diff) < 2:
            continue
        k = int(a[diff].sum())
        if k == 0 or k == len(diff):
            continue
        perm = rng.permutation(diff)
        a[diff] = 0
        b[diff] = 0
        a[perm[:k]] = 1
        b[perm[k:]] = 1
    return M


def _fdsm_counts(df: pd.DataFrame, reps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Observed and null co-occurrence counts, permuting within document.

    The null fixes each sentence's label load (row sums) and each document's
    per-code count (column sums), so sentence length enters only through load.
    Rows carrying 0 or 8 codes are invariant under the swap and are added once.
    """
    cols = [f"c{k:02d}" for k in range(1, 9)]
    rng = np.random.default_rng(seed)
    obs = np.zeros(len(PAIRS))
    null = np.zeros((reps, len(PAIRS)))
    for _, g in df.groupby("doc_id", sort=False):
        M = g[cols].to_numpy(np.int8)
        rs = M.sum(1)
        fixed = M[(rs == 0) | (rs == 8)]
        free = M[(rs > 0) & (rs < 8)]
        Cf = fixed.T @ fixed
        C = M.T @ M
        for p, (x, y) in enumerate(PAIRS):
            obs[p] += C[x, y]
        if len(free) < 2:
            for r in range(reps):
                for p, (x, y) in enumerate(PAIRS):
                    null[r, p] += C[x, y]
            continue
        for r in range(reps):
            Mr = _curveball(free.copy(), rng, 5 * len(free) + 10)
            Cr = Mr.T @ Mr + Cf
            for p, (x, y) in enumerate(PAIRS):
                null[r, p] += Cr[x, y]
    return obs, null


def _fdsm_table(df: pd.DataFrame, reps: int, seed: int, label: str) -> pd.DataFrame:
    obs, null = _fdsm_counts(df, reps, seed)
    mu, sd = null.mean(0), null.std(0)
    ge = (null >= obs).sum(0)
    le = (null <= obs).sum(0)
    p = np.minimum(1.0, 2 * np.minimum((ge + 1) / (reps + 1), (le + 1) / (reps + 1)))
    return pd.DataFrame({
        "stratum": label,
        "pair": PAIR_NAMES,
        "n_ab": obs.astype(int),
        "null_mean": mu.round(2),
        "ratio": np.where(mu > 0, obs / np.where(mu == 0, np.nan, mu), np.nan).round(3),
        "z": np.where(sd > 0, (obs - mu) / np.where(sd == 0, np.nan, sd), np.nan).round(2),
        "p_perm": p.round(4),
        "q_bh": bh(p).round(4),
        "n_sent": len(df),
    })


def step_a(d: pd.DataFrame, reps: int) -> None:
    out = [_fdsm_table(d, reps, 11, "pooled")]
    for camp in ["DPP", "KMT", "TPP"]:
        out.append(_fdsm_table(d[d.camp == camp], reps, 12, f"camp={camp}"))
    for genre in ["news", "debate"]:
        out.append(_fdsm_table(d[d.genre == genre], reps, 13, f"genre={genre}"))
    t = pd.concat(out, ignore_index=True)
    t.to_csv(f"{OUT}/A_fdsm_28pairs.csv", index=False)
    print(t[t.stratum == "pooled"].to_string(index=False))


# ------------------------------------------------------- B: Layer A composition

def step_b(d: pd.DataFrame, reps: int) -> None:
    cols = [f"c{k:02d}" for k in range(1, 9)]
    docs = d.doc_id.unique()
    idx = d.groupby("doc_id").indices
    rng = np.random.default_rng(21)

    def comp_matrix(x: pd.DataFrame) -> pd.DataFrame:
        tot = x[cols].sum()
        base = tot / tot.sum()
        rows = {}
        for l1, g in x.groupby("L1_gpt"):
            s = g[cols].sum()
            rows[l1] = (s / s.sum()) / base if s.sum() > 0 else pd.Series(np.nan, index=cols)
        return pd.DataFrame(rows).T

    point = comp_matrix(d)
    boots = []
    for _ in range(reps):
        pick = rng.choice(docs, len(docs))
        boots.append(comp_matrix(d.iloc[np.concatenate([idx[k] for k in pick])]))
    lo = pd.concat(boots).groupby(level=0).quantile(0.025)
    hi = pd.concat(boots).groupby(level=0).quantile(0.975)

    load = d.groupby("L1_gpt").agg(n=("load", "size"), load=("load", "mean"),
                                   any_code=("load", lambda s: (s > 0).mean()),
                                   chars=("len", "mean"))
    load["load_ratio"] = load["load"] / d.load.mean()

    recs = []
    for l1 in point.index:
        for c in cols:
            recs.append({"L1": l1, "L2": "L2-" + c[1:], "n_l1": int(load.loc[l1, "n"]),
                         "load": round(load.loc[l1, "load"], 3),
                         "load_ratio": round(load.loc[l1, "load_ratio"], 3),
                         "raw_lift": round(d[d.L1_gpt == l1][c].mean() / d[c].mean(), 3),
                         "comp_ratio": round(point.loc[l1, c], 3),
                         "comp_lo": round(lo.loc[l1, c], 3), "comp_hi": round(hi.loc[l1, c], 3),
                         "sig": not (lo.loc[l1, c] <= 1 <= hi.loc[l1, c])})
    t = pd.DataFrame(recs)
    t.to_csv(f"{OUT}/B_layerA_load_composition.csv", index=False)
    load.round(3).to_csv(f"{OUT}/B_layerA_load_by_l1.csv")
    print(t.pivot(index="L1", columns="L2", values="comp_ratio").to_string())


# ------------------------------------------------------------- C: Mundlak split

def step_c(d: pd.DataFrame) -> None:
    import statsmodels.api as sm

    l1s = sorted(d.L1_gpt.dropna().unique())
    ref = "L1-06"
    work = d.dropna(subset=["L1_gpt"]).copy()
    for l1 in l1s:
        work[f"d_{l1}"] = (work.L1_gpt == l1).astype(float)
        work[f"m_{l1}"] = work.groupby("doc_id")[f"d_{l1}"].transform("mean")
    work["z_len"] = (work["len"] - work["len"].mean()) / work["len"].std()
    keep = [l for l in l1s if l != ref]
    X = work[[f"d_{l}" for l in keep] + [f"m_{l}" for l in keep] + ["z_len"]].copy()
    X = pd.concat([X, pd.get_dummies(work.camp, prefix="camp", drop_first=True).astype(float),
                   pd.get_dummies(work.genre, prefix="g", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    recs = []
    for k in range(1, 9):
        y = work[f"c{k:02d}"].astype(float)
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                            cov_kwds={"groups": work.doc_id})
        for l in keep:
            w, b_ctx = m.params[f"d_{l}"], m.params[f"m_{l}"]
            recs.append({"L2": f"L2-0{k}", "L1": l,
                         "within_OR": round(float(np.exp(w)), 3),
                         "within_p": round(float(m.pvalues[f"d_{l}"]), 4),
                         "contextual_OR": round(float(np.exp(b_ctx)), 3),
                         "contextual_p": round(float(m.pvalues[f"m_{l}"]), 4),
                         "between_OR": round(float(np.exp(w + b_ctx)), 3)})
    t = pd.DataFrame(recs)
    t.to_csv(f"{OUT}/C_mundlak_within_between.csv", index=False)
    print(t[t.L1.isin(["L1-01", "L1-03"])].to_string(index=False))


# ------------------------------------------- D: thresholds before FDR, p2_13 recheck

def step_d(min_ab: int = 20, min_marg: int = 30) -> None:
    rows, recheck = [], []
    for run in ["06_phase2_coarticulation", "06_phase2_coarticulation_deepseek"]:
        for dd in sorted(glob.glob(f"{ROOT}/{run}/npmi_by_*")):
            pt, bt = f"{dd}/npmi_point.parquet", f"{dd}/npmi_bootstrap.parquet"
            if not (os.path.exists(pt) and os.path.exists(bt)):
                continue
            p, b = pd.read_parquet(pt), pd.read_parquet(bt)
            m = p.merge(b, on=[c for c in p.columns if c in b.columns])
            m["n_ab"] = (m.p_ab * m.n_windows).round()
            m["n_a"] = (m.p_a * m.n_windows).round()
            m["n_b"] = (m.p_b * m.n_windows).round()
            m["passes"] = (m.n_ab >= min_ab) & (m.n_a >= min_marg) & (m.n_b >= min_marg)
            strata = [c for c in ["camp", "genre", "time_bucket"] if c in m.columns]
            m["q_new"] = np.nan
            for _, g in m[m.passes].groupby(strata) if strata else [((), m[m.passes])]:
                m.loc[g.index, "q_new"] = bh(g.p_value.to_numpy())
            m["sig_new"] = m.q_new.notna() & (m.q_new < 0.05)
            rows.append(pd.DataFrame([{
                "run": run, "stratum_dir": os.path.basename(dd),
                "edges": len(m), "sig_old": int(m.fdr_significant.sum()),
                "sig_old_below_threshold": int((m.fdr_significant & ~m.passes).sum()),
                "sig_new": int(m.sig_new.sum()),
            }]))
            if os.path.basename(dd) == "npmi_by_camp":
                recheck.append((run, m))
    pd.concat(rows, ignore_index=True).to_csv(f"{OUT}/D_fdr_threshold_audit.csv", index=False)

    out13 = []
    for run, m in recheck:
        for a, bcamp in [("DPP", "KMT"), ("DPP", "TPP"), ("KMT", "TPP")]:
            A = m[m.camp == a].set_index(["l2_a", "l2_b"])
            B = m[m.camp == bcamp].set_index(["l2_a", "l2_b"])
            for key in A.index.intersection(B.index):
                ra, rb = A.loc[key], B.loc[key]
                disj = bool(ra.npmi_upper < rb.npmi_lower or rb.npmi_upper < ra.npmi_lower)
                out13.append({"run": run, "pair": f"{a}-{bcamp}", "edge": f"{key[0]}~{key[1]}",
                              "ci_disjoint": disj, "n_ab_a": int(ra.n_ab), "n_ab_b": int(rb.n_ab),
                              "both_pass": bool(ra.passes and rb.passes)})
    t = pd.DataFrame(out13)
    t.to_csv(f"{OUT}/D_p2_13_recheck.csv", index=False)
    print(t[t.ci_disjoint].groupby(["run", "both_pass"]).size().to_string())


# ------------------------------------------------------- E: camp permutation test

def step_e(d: pd.DataFrame, reps: int) -> None:
    cols = [f"c{k:02d}" for k in range(1, 9)]
    work = d.dropna(subset=["L1_gpt", "camp"]).copy()
    doc_camp = work.groupby("doc_id").camp.first()

    def stat(camp_of_doc: pd.Series, block: list[str]) -> float:
        x = work.assign(cc=work.doc_id.map(camp_of_doc))
        mats = {}
        for camp, g in x.groupby("cc"):
            tot = g[block].sum()
            base = tot / tot.sum()
            rows = {}
            for l1, gg in g.groupby("L1_gpt"):
                s = gg[block].sum()
                rows[l1] = (s / s.sum()) / base if s.sum() > 0 else pd.Series(np.nan, index=block)
            mats[camp] = pd.DataFrame(rows).T.reindex(sorted(work.L1_gpt.unique()))
        tot_d = 0.0
        for a, b in itertools.combinations(sorted(mats), 2):
            tot_d += np.nanmean(np.abs(mats[a].to_numpy() - mats[b].to_numpy()))
        return tot_d

    rng = np.random.default_rng(31)
    docs = doc_camp.index.to_numpy()
    res = []
    for name, block in [("cohesion", ["c04", "c06", "c07", "c08"]),
                        ("boundary", ["c01", "c02", "c03", "c05"]),
                        ("all8", cols)]:
        obs = stat(doc_camp, block)
        null = []
        for _ in range(reps):
            shuffled = pd.Series(rng.permutation(doc_camp.to_numpy()), index=docs)
            null.append(stat(shuffled, block))
        null = np.array(null)
        res.append({"block": name, "observed": round(obs, 4),
                    "null_mean": round(null.mean(), 4), "null_sd": round(null.std(), 4),
                    "p_perm": round((1 + (null >= obs).sum()) / (1 + reps), 4), "reps": reps})
    t = pd.DataFrame(res)
    t.to_csv(f"{OUT}/E_camp_permutation_layerA.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 500
    frame = load()
    print(f"N={len(frame)} docs={frame.doc_id.nunique()} reps={reps}", flush=True)
    if which in ("A", "all"):
        step_a(frame, reps)
    if which in ("B", "all"):
        step_b(frame, min(reps, 500))
    if which in ("C", "all"):
        step_c(frame)
    if which in ("D", "all"):
        step_d()
    if which in ("E", "all"):
        step_e(frame, min(reps, 500))
