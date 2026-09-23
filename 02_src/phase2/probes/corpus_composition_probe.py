#!/usr/bin/env python3
"""Corpus-composition probe: who supplies the identity grammar.

The corpus is 95% news retrieved by candidate name from three outlets, so `camp`
is a retrieval key rather than a speaker. This probe splits the corpus by voice
(own party speaker / other party speaker / reporter prose), by role, and by
outlet, and re-estimates every quantity the paper rests on inside each slice.

F  fdsm_strata   28 pairs under the degree-preserving null, per slice
G  camp_outlet   28 pairs by camp within a single outlet (China Times)
H  composition   Layer A load + composition, own voice vs reporter prose
I  campperm      Layer A camp permutation within outlet and within own voice
J  mundlak       within/between decomposition on the own-voice subset

Usage: python corpus_composition_probe.py [F|G|H|I|J|all] [--reps N]
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tier1_reestimation import OUT, ROOT, _fdsm_table, bh  # noqa: E402

GPT_RUN = f"{ROOT}/01_results_datasets/Run_20260407_221407/final_results.csv"
FRAME = f"{ROOT}/10_merged_master/analysis_frame_n20901.csv"
CODES = [f"c{k:02d}" for k in range(1, 9)]


def load() -> pd.DataFrame:
    meta = pd.read_csv(GPT_RUN, usecols=["unit_id", "role", "speakers", "targets"])
    d = pd.read_csv(FRAME).merge(meta, on="unit_id", how="left")
    d = d.drop_duplicates(["doc_id", "sent_idx"]).sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)
    for k in range(1, 9):
        d[f"c{k:02d}"] = d.l2_set_ds.fillna("").str.contains(f"L2-0{k}").astype(int)
    d["len"] = d.sent_text.str.len()
    d["load"] = d[CODES].sum(axis=1)

    def parties(s: object) -> list[str]:
        try:
            v = ast.literal_eval(s) if isinstance(s, str) else []
        except (ValueError, SyntaxError):
            v = []
        return sorted({x.get("party") for x in v if isinstance(x, dict) and x.get("party")})

    d["sp_parties"] = d.speakers.apply(parties)
    d["attributed"] = d.sp_parties.str.len() > 0
    d["own_voice"] = [c in p for c, p in zip(d.camp, d.sp_parties)]
    d["voice"] = np.where(d.own_voice, "own", np.where(d.attributed, "cross", "reporter"))
    # The files named "chinatimes" are Central News Agency copy: every url in the raw
    # data is www.cna.com.tw. Canonical labels are used everywhere downstream.
    d["outlet"] = (d.source.str.extract(r"_(chinatimes|ltn|tvbs)_")[0]
                   .replace({"chinatimes": "cna"}).fillna("debate_other"))
    d["outlet_role"] = d.outlet.map({"cna": "wire", "ltn": "green", "tvbs": "blue"}).fillna("other")
    return d


def composition(x: pd.DataFrame) -> pd.DataFrame:
    tot = x[CODES].sum()
    base = tot / tot.sum()
    rows = {}
    for l1, g in x.groupby("L1_gpt"):
        s = g[CODES].sum()
        rows[l1] = (s / s.sum()) / base if s.sum() > 0 else pd.Series(np.nan, index=CODES)
    return pd.DataFrame(rows).T


def step_f(d: pd.DataFrame, reps: int) -> None:
    slices = {
        "own": d[d.voice == "own"], "cross": d[d.voice == "cross"], "reporter": d[d.voice == "reporter"],
        "role=claim": d[d.role == "claim"], "role=context": d[d.role == "context"],
        "outlet=cna(wire)": d[d.outlet == "cna"], "outlet=ltn(green)": d[d.outlet == "ltn"],
        "outlet=tvbs(blue)": d[d.outlet == "tvbs"],
    }
    t = pd.concat([_fdsm_table(s, reps, 17, k) for k, s in slices.items()], ignore_index=True)
    t.to_csv(f"{OUT}/F_corpus_strata_fdsm.csv", index=False)
    print(t.pivot(index="pair", columns="stratum", values="ratio").round(3).to_string())


def step_g(d: pd.DataFrame, reps: int) -> None:
    out = [_fdsm_table(d[d.voice == "own"], reps, 23, "own_voice_all")]
    for camp in ["DPP", "KMT", "TPP"]:
        out.append(_fdsm_table(d[(d.outlet == "cna") & (d.camp == camp)], reps, 24, f"cna×{camp}"))
        out.append(_fdsm_table(d[(d.voice == "own") & (d.camp == camp)], reps, 25, f"own×{camp}"))
    t = pd.concat(out, ignore_index=True)
    t.to_csv(f"{OUT}/G_camp_within_outlet_fdsm.csv", index=False)
    print(t[t.stratum == "own_voice_all"].sort_values("ratio", ascending=False).to_string(index=False))


def step_h(d: pd.DataFrame, reps: int) -> None:
    rows = []
    for name, sub in [("own", d[d.voice == "own"]), ("reporter", d[d.voice == "reporter"]),
                      ("cross", d[d.voice == "cross"])]:
        point = composition(sub)
        docs = sub.doc_id.unique()
        idx = sub.groupby("doc_id").indices
        rng = np.random.default_rng(41)
        boots = [composition(sub.iloc[np.concatenate([idx[k] for k in rng.choice(docs, len(docs))])])
                 for _ in range(reps)]
        lo = pd.concat(boots).groupby(level=0).quantile(0.025)
        hi = pd.concat(boots).groupby(level=0).quantile(0.975)
        load = sub.groupby("L1_gpt").load.agg(["size", "mean"])
        for l1 in point.index:
            for c in CODES:
                rows.append({"voice": name, "L1": l1, "L2": "L2-" + c[1:], "n_l1": int(load.loc[l1, "size"]),
                             "load": round(load.loc[l1, "mean"], 3), "comp": round(point.loc[l1, c], 3),
                             "lo": round(lo.loc[l1, c], 3), "hi": round(hi.loc[l1, c], 3),
                             "sig": not (lo.loc[l1, c] <= 1 <= hi.loc[l1, c])})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/H_composition_by_voice.csv", index=False)
    print(t.pivot_table(index=["voice", "L1"], columns="L2", values="comp").round(2).to_string())


def step_i(d: pd.DataFrame, reps: int) -> None:
    import itertools

    def stat(x: pd.DataFrame, camp_of_doc: pd.Series, block: list[str]) -> float:
        y = x.assign(cc=x.doc_id.map(camp_of_doc))
        mats = {}
        for camp, g in y.groupby("cc"):
            tot = g[block].sum()
            base = tot / tot.sum()
            rows = {}
            for l1, gg in g.groupby("L1_gpt"):
                s = gg[block].sum()
                rows[l1] = (s / s.sum()) / base if s.sum() > 0 else pd.Series(np.nan, index=block)
            mats[camp] = pd.DataFrame(rows).T.reindex(sorted(x.L1_gpt.dropna().unique()))
        return sum(np.nanmean(np.abs(mats[a].to_numpy() - mats[b].to_numpy()))
                   for a, b in itertools.combinations(sorted(mats), 2))

    rng = np.random.default_rng(51)
    res = []
    for sname, sub in [("cna", d[d.outlet == "cna"]), ("own_voice", d[d.voice == "own"]),
                       ("cna_own", d[(d.outlet == "cna") & (d.voice == "own")])]:
        sub = sub.dropna(subset=["L1_gpt", "camp"])
        doc_camp = sub.groupby("doc_id").camp.first()
        docs = doc_camp.index.to_numpy()
        for bname, block in [("cohesion", ["c04", "c06", "c07", "c08"]),
                             ("boundary", ["c01", "c02", "c03", "c05"]), ("all8", CODES)]:
            obs = stat(sub, doc_camp, block)
            null = np.array([stat(sub, pd.Series(rng.permutation(doc_camp.to_numpy()), index=docs), block)
                             for _ in range(reps)])
            res.append({"slice": sname, "block": bname, "n": len(sub), "observed": round(obs, 4),
                        "null_mean": round(null.mean(), 4),
                        "p_perm": round((1 + (null >= obs).sum()) / (1 + reps), 4)})
    t = pd.DataFrame(res)
    t.to_csv(f"{OUT}/I_camp_permutation_by_slice.csv", index=False)
    print(t.to_string(index=False))


def step_j(d: pd.DataFrame) -> None:
    import statsmodels.api as sm

    rows = []
    for name, sub in [("own", d[d.voice == "own"]), ("reporter", d[d.voice == "reporter"])]:
        work = sub.dropna(subset=["L1_gpt"]).copy()
        l1s = sorted(work.L1_gpt.unique())
        keep = [l for l in l1s if l != "L1-06"]
        for l1 in l1s:
            work[f"d_{l1}"] = (work.L1_gpt == l1).astype(float)
            work[f"m_{l1}"] = work.groupby("doc_id")[f"d_{l1}"].transform("mean")
        work["z_len"] = (work["len"] - work["len"].mean()) / work["len"].std()
        X = work[[f"d_{l}" for l in keep] + [f"m_{l}" for l in keep] + ["z_len"]].copy()
        X = pd.concat([X, pd.get_dummies(work.camp, prefix="camp", drop_first=True).astype(float),
                       pd.get_dummies(work.outlet, prefix="o", drop_first=True).astype(float)], axis=1)
        X = sm.add_constant(X)
        for k in range(1, 9):
            m = sm.GLM(work[f"c{k:02d}"].astype(float), X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": work.doc_id})
            for l in keep:
                rows.append({"voice": name, "L2": f"L2-0{k}", "L1": l,
                             "within_OR": round(float(np.exp(m.params[f"d_{l}"])), 3),
                             "within_p": round(float(m.pvalues[f"d_{l}"]), 4),
                             "contextual_OR": round(float(np.exp(m.params[f"m_{l}"])), 3),
                             "contextual_p": round(float(m.pvalues[f"m_{l}"]), 4)})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/J_mundlak_by_voice.csv", index=False)
    print(t[t.L1.isin(["L1-01", "L1-03"])].to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 500
    frame = load()
    print(f"N={len(frame)} own={int((frame.voice=='own').sum())} "
          f"cross={int((frame.voice=='cross').sum())} reporter={int((frame.voice=='reporter').sum())}", flush=True)
    if which in ("F", "all"):
        step_f(frame, reps)
    if which in ("G", "all"):
        step_g(frame, reps)
    if which in ("H", "all"):
        step_h(frame, min(reps, 400))
    if which in ("I", "all"):
        step_i(frame, min(reps, 400))
    if which in ("J", "all"):
        step_j(frame)
