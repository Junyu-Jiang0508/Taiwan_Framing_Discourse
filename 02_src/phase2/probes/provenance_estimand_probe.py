#!/usr/bin/env python3
"""Provenance, estimand and single-label competition checks (2026-09-22).

R  provenance  where the documents actually come from (the "chinatimes" files are CNA)
S  estimand    the same relations under three estimands, which disagree in sign
T  l1compete   what single-label L1 assignment does to the economy x democracy result
U  signifiers  signifier-level camp differences, threat referents, the "peace" chains
V  selection   whether the relevance filter manufactures negative association (Berkson)
W  partisan    green (LTN) vs blue (TVBS) against the CNA wire baseline, on load-adjusted shares
X  robustness  tries to kill W: topic-mix standardisation, the other L2 annotator, and by period
Y  estimands   the master table: every pair and Layer A cell under all estimands at once
Z  dsl1        Layer A recomputed on the independent DeepSeek L1 labels
P  peace       the "peace" signifier: same word, same rate, different chain by camp
C  classify    the 143 economy+democracy sentences, grouped by what they actually are

Usage: python provenance_estimand_probe.py [R|S|T|U|V|W|X|Y|Z|P|C|all]
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from corpus_composition_probe import CODES, load  # noqa: E402
from tier1_reestimation import PAIR_NAMES, PAIRS, ROOT  # noqa: E402

PAIRS_IDX = PAIRS
II_ = np.array([a for a, _ in PAIRS])
JJ_ = np.array([b for _, b in PAIRS])

OUT = f"{ROOT}/13_provenance_estimand"
RAW = f"{os.path.dirname(ROOT)}/../01_data/01_raw_datasets/01_news_datasets"
ECON = "經濟|经济|GDP|薪資|薪资|物價|物价|通膨|產業|产业|半導體|半导体|台積電|出口|經貿|经贸|投資|投资"
SIGNIFIERS = {"台灣": "台灣|臺灣", "中華民國": "中華民國", "中華民族類": "中華民族|炎黃|兩岸一家",
              "主權": "主權", "民主": "民主", "和平": "和平", "戰爭": "戰爭|開戰",
              "九二共識": "九二共識", "台獨": "台獨"}


def step_r(d: pd.DataFrame) -> None:
    rows = []
    for f in sorted(glob.glob(f"{RAW}/*.csv")):
        r = pd.read_csv(f, usecols=["url", "content"])
        dom = r.url.astype(str).str.extract(r"https?://([^/]+)")[0].value_counts()
        rows.append({"file": os.path.basename(f), "n_rows": len(r), "top_domain": dom.index[0],
                     "top_share": round(dom.iloc[0] / len(r), 4),
                     "cna_dateline": round(r.content.astype(str).str.contains(r"[（(]中央社記者.{1,12}電[）)]").mean(), 3)})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/R_document_provenance.csv", index=False)
    print(t.to_string(index=False))
    g = d.groupby("sent_text").outlet.nunique()
    print("identical sentences appearing under >1 outlet:", int((g > 1).sum()))


def _fdsm_ratio() -> pd.Series:
    a = pd.read_csv(f"{ROOT}/11_tier1_reestimation/A_fdsm_28pairs.csv").query("stratum == 'pooled'")
    return a.set_index("pair").ratio


def step_s(d: pd.DataFrame) -> None:
    """Three estimands for the same 28 pairs, plus the Layer A cells."""
    fd = _fdsm_ratio()
    rows = []
    for (i, j), name in zip(PAIRS, PAIR_NAMES):
        x, y = d[CODES[i]], d[CODES[j]]
        t = np.array([[((1 - x) & (1 - y)).sum(), ((1 - x) & y).sum()],
                      [(x & (1 - y)).sum(), (x & y).sum()]]) + 0.5
        rows.append({"pair": name, "raw_OR": round(t[1, 1] * t[0, 0] / (t[1, 0] * t[0, 1]), 3),
                     "fdsm_ratio": fd.get(name, np.nan)})
    t = pd.DataFrame(rows)
    t["sign_agree"] = ((t.raw_OR - 1) * (t.fdsm_ratio - 1)) > 0
    t.to_csv(f"{OUT}/S_estimand_28pairs.csv", index=False)
    print(t.to_string(index=False))
    print("pairs where the two estimands disagree in sign:", int((~t.sign_agree).sum()), "/", len(t))

    # Layer A: raw P(code|domain) against the load-adjusted composition
    dd = d.dropna(subset=["L1_gpt"])
    raw = dd.groupby("L1_gpt")[CODES].mean()
    base = dd[CODES].mean()
    comp_num = dd.groupby("L1_gpt")[CODES].sum()
    comp = comp_num.div(comp_num.sum(axis=1), axis=0).div(dd[CODES].sum() / dd[CODES].sum().sum(), axis=1)
    out = pd.concat([(raw / base).round(3).add_suffix("_rawlift"), comp.round(3).add_suffix("_composition")], axis=1)
    out.to_csv(f"{OUT}/S_layerA_raw_vs_composition.csv")
    print("\nraw lift vs composition, selected cells:")
    for l1, c in [("L1-03", "c08"), ("L1-01", "c08"), ("L1-01", "c06"), ("L1-03", "c07")]:
        print(f"  {l1} x L2-{c[1:]}: raw P = {raw.loc[l1, c]*100:.1f}% (base {base[c]*100:.1f}%, "
              f"lift {raw.loc[l1, c]/base[c]:.2f})  composition {comp.loc[l1, c]:.2f}")

    # conditional: nodewise logistic, doc-clustered
    dd = dd.copy()
    dd["z_len"] = (dd["len"] - dd["len"].mean()) / dd["len"].std()
    L1 = pd.get_dummies(dd.L1_gpt, prefix="L1", drop_first=True).astype(float)
    OUTL = pd.get_dummies(dd.outlet, prefix="o", drop_first=True).astype(float)
    rows = []
    for y, x in [("c07", "c06"), ("c08", "c06"), ("c07", "c04"), ("c07", "c02"), ("c08", "c02"), ("c06", "c04")]:
        spec = {"raw": [], "+len": [dd[["z_len"]]], "+L1": [L1],
                "+other_codes": [dd[[c for c in CODES if c not in (y, x)]].astype(float)],
                "full": [L1, dd[["z_len"]], OUTL, dd[[c for c in CODES if c not in (y, x)]].astype(float)]}
        r = {"pair": f"{x[1:]}->{y[1:]}", "fdsm": fd.get(f"{min(x,y)[1:]}-{max(x,y)[1:]}", np.nan)}
        for k, extra in spec.items():
            X = sm.add_constant(pd.concat([dd[[x]].astype(float)] + extra, axis=1))
            m = sm.GLM(dd[y].astype(float), X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": dd.doc_id})
            r[k] = round(float(np.exp(m.params[x])), 2)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/S_conditional_vs_null.csv", index=False)
    print("\n", t.to_string(index=False))


def step_t(d: pd.DataFrame) -> None:
    dd = d.dropna(subset=["L1_gpt"])
    e = dd.sent_text.str.contains(ECON)
    dm = dd.sent_text.str.contains("民主")
    t = pd.crosstab(dd[e].L1_gpt, dm[e], normalize="columns").round(3)
    t.columns = ["econ_no_democracy", "econ_and_democracy"]
    t.to_csv(f"{OUT}/T_L1_of_econ_democracy_sentences.csv")
    print(t.to_string())
    x = dd[e & dm]
    prof = {k: round(x.sent_text.str.contains(p).mean(), 3) for k, p in
            {"和平/戰爭": "和平|戰爭|战争", "安全/國安": "安全|國安|国安", "中國": "中國|中共|對岸|北京",
             "美國/盟友": "美國|美国|盟友|理念相近"}.items()}
    print(f"n = {len(x)} economy+democracy sentences; profile {prof}")
    x[["unit_id", "camp", "L1_gpt", "sent_text"]].to_csv(f"{OUT}/T_econ_democracy_sentences.csv", index=False)


def step_u(d: pd.DataFrame) -> None:
    rows = []
    for k, p in SIGNIFIERS.items():
        h = d.sent_text.str.contains(p)
        r = {"signifier": k, "n": int(h.sum()), "mean_load": round(d.load[h].mean(), 2),
             "any_code": round((d.load[h] > 0).mean(), 2)}
        r.update({f"pct_{c}": round(100 * h[d.camp == c].mean(), 2) for c in ["DPP", "KMT", "TPP"]})
        r.update({f"code_{c[1:]}": round(100 * d.loc[h, c].mean(), 1) for c in CODES})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/U_signifiers_by_camp.csv", index=False)
    print(t[["signifier", "n", "mean_load", "pct_DPP", "pct_KMT", "pct_TPP"]].to_string(index=False))
    threat = d[d.c05 == 1]
    ref = {"中國/中共": "中國|中共|對岸|北京|解放軍", "民進黨/綠營": "民進黨|綠營|執政黨", "戰爭": "戰爭|開戰"}
    tr = pd.DataFrame({k: threat.groupby("camp").sent_text.apply(lambda s: s.str.contains(p).mean())
                       for k, p in ref.items()}).round(3)
    tr.to_csv(f"{OUT}/U_threat_referent_by_camp.csv")
    print("\nthreat referents inside L2-05:\n", tr.to_string())
    pc = d[d.sent_text.str.contains("和平")].groupby("camp")[CODES].mean().round(3)
    pc.to_csv(f"{OUT}/U_peace_chains_by_camp.csv")
    print("\ncodes carried by 和平 sentences:\n", pc[["c05", "c07", "c08", "c06"]].to_string())


def step_v(d: pd.DataFrame) -> None:
    """Lexical association in the unfiltered sentence pool vs in the analysis corpus."""
    def pool_stats(texts: pd.Series) -> dict:
        s = pd.Series([x for t in texts.drop_duplicates() for x in re.split(r"(?<=[。！？!?])", str(t))
                       if len(x.strip()) >= 8]).drop_duplicates()
        e, m = s.str.contains(ECON), s.str.contains("民主")
        return {"n": len(s), "n_econ": int(e.sum()), "n_demo": int(m.sum()), "both": int((e & m).sum())}

    agg = {"n": 0, "n_econ": 0, "n_demo": 0, "both": 0}
    for f in sorted(glob.glob(f"{RAW}/*.csv")):
        st = pool_stats(pd.read_csv(f, usecols=["content"]).content.dropna().astype(str))
        for k in agg:
            agg[k] += st[k]

    def orr(n, a, b, ab):
        return round((ab * (n - a - b + ab)) / max((a - ab) * (b - ab), 1), 3)

    e, m = d.sent_text.str.contains(ECON), d.sent_text.str.contains("民主")
    t = pd.DataFrame([
        {"pool": "raw unfiltered sentences", **agg, "OR": orr(agg["n"], agg["n_econ"], agg["n_demo"], agg["both"])},
        {"pool": "analysis corpus", "n": len(d), "n_econ": int(e.sum()), "n_demo": int(m.sum()),
         "both": int((e & m).sum()), "OR": orr(len(d), e.sum(), m.sum(), (e & m).sum())}])
    t.to_csv(f"{OUT}/V_selection_berkson_check.csv", index=False)
    print(t.to_string(index=False))
    print("both ORs > 1 -> the relevance filter is not manufacturing the economy/democracy negative relation")


def step_w(d: pd.DataFrame) -> None:
    """Partisan emphasis: LTN is the green outlet, TVBS the blue one, CNA the wire baseline.

    TVBS sentences carry much less identity content (mean load 0.30 vs LTN 0.78), so raw prevalence
    would report every code as "lower in blue". The comparison therefore runs on load-adjusted
    shares: of all the identity coding an outlet does, what share goes to each code.
    """
    d = d[d.outlet.isin(["cna", "ltn", "tvbs"])].copy()
    d["camp_outlet"] = d.outlet.map({"cna": "CNA_wire", "ltn": "LTN_green", "tvbs": "TVBS_blue"})
    share = lambda x: x[CODES].sum() / x[CODES].sum().sum()
    base = share(d)
    tab = pd.DataFrame({o: share(g) / base for o, g in d.groupby("camp_outlet")})

    two = d[d.camp_outlet != "CNA_wire"]
    docs = two.groupby("doc_id").camp_outlet.first()
    rng = np.random.default_rng(7)

    def gap(x):
        c = {o: share(v) for o, v in x.groupby("camp_outlet")}
        return (c["LTN_green"] - c["TVBS_blue"]) / base

    obs = gap(two)
    null = pd.DataFrame([gap(two.assign(camp_outlet=two.doc_id.map(
        pd.Series(rng.permutation(docs.to_numpy()), index=docs.index)))) for _ in range(1000)])
    tab["green_minus_blue"] = obs.round(3)
    tab["p_perm"] = (((null.abs() >= obs.abs()).sum() + 1) / 1001).round(4)
    tab.round(3).to_csv(f"{OUT}/W_partisan_emphasis.csv")
    print(tab.round(3).to_string())

    # lexical corroboration, anchored on sentences that mention Taiwan
    tw = d[d.sent_text.str.contains("台灣|臺灣")]
    lex = pd.DataFrame({k: tw.groupby("camp_outlet").sent_text.apply(lambda s: s.str.contains(v).mean())
                        for k, v in SIGNIFIERS.items()}).round(3)
    lex.to_csv(f"{OUT}/W_lexical_anchor_taiwan.csv")
    print("\nsignifier rates inside Taiwan-mentioning sentences:\n", lex.to_string())

    # what the blue outlet cannot support: edge-level estimation
    tv = d[d.camp_outlet == "TVBS_blue"]
    co = pd.Series({f"{a+1:02d}-{b+1:02d}": int((tv[CODES[a]] & tv[CODES[b]]).sum())
                    for a, b in PAIRS_IDX}).sort_values(ascending=False)
    co.to_csv(f"{OUT}/W_tvbs_edge_feasibility.csv")
    print(f"\nTVBS: n={len(tv)}, sentences with any code={int((tv.load>0).sum())}, "
          f"pairs with n_ab>=20: {int((co>=20).sum())}/28 (max {int(co.iloc[0])})")


def _share(x, w=None):
    s = (x[CODES].multiply(w, axis=0) if w is not None else x[CODES]).sum()
    return s / s.sum()


def _perm_gap(two, fn, reps, seed, base):
    docs = two.groupby("doc_id").camp_outlet.first()
    rng = np.random.default_rng(seed)

    def gap(x):
        c = {o: fn(v) for o, v in x.groupby("camp_outlet")}
        return (c["LTN_green"] - c["TVBS_blue"]) / base

    obs = gap(two)
    null = pd.DataFrame([gap(two.assign(camp_outlet=two.doc_id.map(
        pd.Series(rng.permutation(docs.to_numpy()), index=docs.index)))) for _ in range(reps)])
    return obs, (((null.abs() >= obs.abs()).sum() + 1) / (reps + 1))


def step_x(d: pd.DataFrame, reps: int = 500) -> None:
    """The partisan gaps are only worth reporting if they survive these three."""
    d = d[d.outlet.isin(["cna", "ltn", "tvbs"])].copy()
    d["camp_outlet"] = d.outlet.map({"cna": "CNA_wire", "ltn": "LTN_green", "tvbs": "TVBS_blue"})
    base = _share(d)
    two = d[d.camp_outlet != "CNA_wire"]
    out = {}

    # 1. topic mix: blue covers a different mix of issues, so reweight every outlet to the corpus mix
    mix = pd.crosstab(d.L1_gpt, d.camp_outlet, normalize="columns")
    target = d.L1_gpt.value_counts(normalize=True)

    def std_share(x):
        obs = x.L1_gpt.value_counts(normalize=True)
        return _share(x, x.L1_gpt.map((target / obs).reindex(obs.index)).fillna(0.0))

    o1, p1 = _perm_gap(two.dropna(subset=["L1_gpt"]), std_share, reps, 3, base)
    out["standardised_gap"], out["standardised_p"] = o1.round(3), p1.round(3)

    # 2. the other annotator
    g = d.copy()
    for k in range(1, 9):
        g[f"c{k:02d}"] = g.l2_set_gpt.fillna("").str.contains(f"L2-0{k}").astype(int)
    base_g = _share(g)
    o2, p2 = _perm_gap(g[g.camp_outlet != "CNA_wire"], _share, reps, 4, base_g)
    out["gpt_labels_gap"], out["gpt_labels_p"] = o2.round(3), p2.round(3)

    t = pd.DataFrame(out)
    t["survives_both"] = (t.standardised_p < .05) & (t.gpt_labels_p < .05) &                          (np.sign(t.standardised_gap) == np.sign(t.gpt_labels_gap))
    t.to_csv(f"{OUT}/X_partisan_robustness.csv")
    print(mix.round(3).to_string(), "\n")
    print(t.to_string())
    print("codes surviving topic-standardisation AND the annotator swap:", list(t.index[t.survives_both]))

    # 3. by period
    d["period"] = pd.to_datetime(d.date, errors="coerce").dt.to_period("M").astype(str)
    d["phase"] = np.where(d.period <= "2023-08", "a_early", np.where(d.period <= "2023-11", "b_autumn", "c_final"))
    rows = []
    for ph, gg in d[d.camp_outlet != "CNA_wire"].groupby("phase"):
        c = {o: _share(v) for o, v in gg.groupby("camp_outlet")}
        if len(c) == 2:
            rows.append(pd.Series((c["LTN_green"] - c["TVBS_blue"]) / base, name=ph))
    ph = pd.DataFrame(rows).T.round(3)
    ph.to_csv(f"{OUT}/X_partisan_by_period.csv")
    print("\ngreen-blue gap by campaign phase:\n", ph.to_string())


def step_y(d: pd.DataFrame) -> None:
    """P0-1: one table, every estimand, so a claim can never again rest on an unstated choice."""
    fd = _fdsm_ratio()
    rows = []
    for (i, j), name in zip(PAIRS, PAIR_NAMES):
        x, y = d[CODES[i]], d[CODES[j]]
        n = len(d)
        t = np.array([[((1 - x) & (1 - y)).sum(), ((1 - x) & y).sum()],
                      [(x & (1 - y)).sum(), (x & y).sum()]]) + 0.5
        lift = ((x & y).mean()) / max(x.mean() * y.mean(), 1e-12)
        rows.append({"pair": name, "n_ab": int((x & y).sum()),
                     "raw_OR": round(t[1, 1] * t[0, 0] / (t[1, 0] * t[0, 1]), 3),
                     "lift": round(lift, 3), "fdsm_ratio": fd.get(name, np.nan)})
    t = pd.DataFrame(rows)
    dd = d.dropna(subset=["L1_gpt"]).copy()
    dd["z_len"] = (dd["len"] - dd["len"].mean()) / dd["len"].std()
    L1 = pd.get_dummies(dd.L1_gpt, prefix="L1", drop_first=True).astype(float)
    OUTL = pd.get_dummies(dd.outlet, prefix="o", drop_first=True).astype(float)
    cond = []
    for (i, j) in PAIRS:
        a, b = CODES[i], CODES[j]
        X = sm.add_constant(pd.concat([dd[[a]].astype(float), dd[[c for c in CODES if c not in (a, b)]].astype(float),
                                       L1, OUTL, dd[["z_len"]]], axis=1))
        m = sm.GLM(dd[b].astype(float), X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": dd.doc_id})
        ci = m.conf_int().loc[a]
        cond.append((round(float(np.exp(m.params[a])), 3), float(np.exp(ci[0])), float(np.exp(ci[1])),
                     float(m.pvalues[a])))
    t["conditional_OR"] = [c[0] for c in cond]
    t["cond_CI"] = [f"[{c[1]:.2f},{c[2]:.2f}]" for c in cond]
    t["cond_sig"] = [c[3] < .05 for c in cond]

    # document-cluster bootstrap for the two estimands that have no analytic interval here
    M = d[CODES].to_numpy(float)
    docs = pd.factorize(d.doc_id)[0]
    idx = [np.flatnonzero(docs == k) for k in range(docs.max() + 1)]
    rng = np.random.default_rng(0)
    boot_or, boot_lift = [], []
    for _ in range(400):
        pick = rng.integers(0, len(idx), len(idx))
        m = M[np.concatenate([idx[k] for k in pick])]
        n = len(m)
        C = m.T @ m
        c1 = m.sum(0)
        n11 = C[II_, JJ_]
        n1a, n1b = c1[II_], c1[JJ_]
        n10, n01, n00 = n1a - n11, n1b - n11, n - n1a - n1b + n11
        boot_or.append(((n11 + .5) * (n00 + .5)) / ((n10 + .5) * (n01 + .5)))
        boot_lift.append((n11 / n) / np.maximum((n1a / n) * (n1b / n), 1e-12))
    for name, B in [("raw_OR", np.array(boot_or)), ("lift", np.array(boot_lift))]:
        lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
        t[name + "_CI"] = [f"[{a:.2f},{b:.2f}]" for a, b in zip(lo, hi)]
        t[name + "_sig"] = (lo > 1) | (hi < 1)
    t["fdsm_sig"] = pd.read_csv(f"{ROOT}/11_tier1_reestimation/A_fdsm_28pairs.csv").query(
        "stratum == 'pooled'").q_bh.to_numpy() < .05

    def verdict(r):
        sig = [np.sign(r[v] - 1) for v, f in [("raw_OR", "raw_OR_sig"), ("lift", "lift_sig"),
                                              ("fdsm_ratio", "fdsm_sig"), ("conditional_OR", "cond_sig")] if r[f]]
        if not sig:
            return "no estimand significant"
        if len(sig) == 4 and all(v > 0 for v in sig):
            return "robust positive (4/4)"
        if len(sig) == 4 and all(v < 0 for v in sig):
            return "robust negative (4/4)"
        if len(set(sig)) > 1:
            return "SIGN CONFLICT"
        return f"partial, same sign ({len(sig)}/4)"

    t["verdict"] = t.apply(verdict, axis=1)
    t.to_csv(f"{OUT}/Y_estimand_master_pairs.csv", index=False)
    print(t[["pair", "n_ab", "raw_OR", "raw_OR_CI", "lift", "fdsm_ratio", "conditional_OR", "cond_CI", "verdict"]].to_string(index=False))
    print("\n", t.verdict.value_counts().to_string())


def step_p(d: pd.DataFrame) -> None:
    """A signifier used at the same rate by every camp, but wired to different things."""
    d = d[d.outlet.isin(["cna", "ltn", "tvbs"])].dropna(subset=["L1_gpt"]).copy()
    d["peace"] = d.sent_text.str.contains("和平").astype(float)
    d["z_len"] = (d["len"] - d["len"].mean()) / d["len"].std()
    d["c08_gpt"] = d.l2_set_gpt.fillna("").str.contains("L2-08").astype(float)
    d["lex_demo"] = d.sent_text.str.contains("民主").astype(float)
    desc = d.groupby("camp").apply(lambda g: pd.Series({
        "n_peace": int(g.peace.sum()),
        "P_demo_given_peace": round(g[g.peace == 1].c08.mean(), 3),
        "P_demo_given_not": round(g[g.peace == 0].c08.mean(), 3)}))
    print(desc.to_string())
    X = pd.concat([d[["peace", "z_len"]],
                   pd.get_dummies(d.camp, prefix="camp", drop_first=True).astype(float),
                   pd.get_dummies(d.outlet, prefix="o", drop_first=True).astype(float),
                   pd.get_dummies(d.L1_gpt, prefix="L1", drop_first=True).astype(float)], axis=1)
    for c in [c for c in X.columns if c.startswith("camp_")]:
        X["peace_x_" + c] = X.peace * X[c]
    X = sm.add_constant(X)
    rows = []
    for name, y in [("L2-08 (DeepSeek)", d.c08.astype(float)), ("L2-08 (GPT)", d.c08_gpt),
                    ("lexical 民主", d.lex_demo)]:
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.doc_id})
        for k in [c for c in X.columns if c.startswith("peace")]:
            rows.append({"outcome": name, "term": k, "OR": round(float(np.exp(m.params[k])), 3),
                         "p": round(float(m.pvalues[k]), 4)})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/P_peace_chain_by_camp.csv", index=False)
    print(t.pivot(index="term", columns="outcome", values=["OR", "p"]).to_string())
    print("note: on the probability scale the three outlets are nearly identical "
          "(CNA .251 / green .246 / blue .237); the outlet contrast is a baseline artefact of the ratio scale.")


def step_c(d: pd.DataFrame) -> None:
    """What the economy+democracy sentences are, read one by one and grouped."""
    ECON_RE = ECON
    x = d[d.sent_text.str.contains(ECON_RE) & d.sent_text.str.contains("民主")].copy()

    def cat(v: str) -> str:
        if re.search(r"一覽|一次看|政見比一比", v) and re.search(r"\d{4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2} \d{2}:\d{2}", v):
            return "G page furniture"
        if "經濟民主連合" in v or "经济民主连合" in v:
            return "F lexical false positive (org name)"
        if re.search(r"四大支柱|4大支柱|經濟安全是國家安全|提升經濟安全", v) or \
           (re.search(r"強化國防|提高國防|增加國防", v) and re.search(r"強化經濟|提升經濟|經濟實力", v)):
            return "A Lai's four-pillar formula"
        if re.search(r"拚經濟", v) and re.search(r"護民主", v):
            return "B four-slogan list"
        if re.search(r"中國|中共|北京|紅色供應鏈|介選|脅迫|服貿|九二共識", v) and \
           re.search(r"經濟|經貿|產業|供應鏈", v) and re.search(r"威脅|脅迫|介入|破壞|分化|侵|武力|撤資|去風險", v):
            return "E China threatens both"
        if re.search(r"民主陣營|民主夥伴|民主國家|民主同盟|理念相近|民主世界", v):
            return "D democratic-alliance + trade"
        if re.search(r"模範生|優等生|成就|肯定|繁榮|成長|進步|品牌價值", v):
            return "C achievement list"
        return "H other"

    x["category"] = x.sent_text.apply(cat)
    x["is_enumeration"] = x.sent_text.str.count("、") >= 2
    x["mentions_lai"] = x.sent_text.str.contains("賴清德|赖清德|蕭美琴")
    x[["unit_id", "camp", "L1_gpt", "category", "is_enumeration", "mentions_lai", "sent_text"]].to_csv(
        f"{OUT}/C_econ_democracy_classified.csv", index=False)
    print(x.category.value_counts().sort_index().to_string())
    print(f"\nn={len(x)}; enumerations (>=2 list commas): {x.is_enumeration.mean():.0%}; "
          f"mention Lai/Hsiao: {x.mentions_lai.mean():.0%}, of which {int((x.mentions_lai & x.camp.isin(['KMT','TPP'])).sum())} "
          f"sit in blue/white retrieval sets")


def step_z(d: pd.DataFrame) -> None:
    ds = pd.read_csv(f"{os.path.dirname(os.path.dirname(ROOT))}/L1_Deepseek/l1_results.csv",
                     usecols=["unit_id", "L1_label"]).rename(columns={"L1_label": "L1_ds"})
    d = d.merge(ds.drop_duplicates("unit_id"), on="unit_id", how="left")
    both = d.L1_ds.notna() & d.L1_gpt.notna()
    print(f"DS L1 coverage {d.L1_ds.notna().mean():.3f}, agreement with GPT where both label: "
          f"{(d.L1_gpt == d.L1_ds)[both].mean():.3f}")
    rows = []
    for col in ["L1_gpt", "L1_ds"]:
        g = d.dropna(subset=[col])
        base = g[CODES].sum() / g[CODES].sum().sum()
        s = g.groupby(col)[CODES].sum()
        comp = s.div(s.sum(axis=1), axis=0) / base
        raw = g.groupby(col)[CODES].mean()
        for l1 in comp.index:
            for c in CODES:
                rows.append({"annotator": col, "L1": l1, "L2": "L2-" + c[1:], "n": int((g[col] == l1).sum()),
                             "composition": round(comp.loc[l1, c], 3), "raw_pct": round(raw.loc[l1, c] * 100, 1)})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/Z_layerA_two_L1_annotators.csv", index=False)
    w = t.pivot_table(index=["L1", "L2"], columns="annotator", values="composition")
    key = [("L1-01", "L2-08"), ("L1-01", "L2-06"), ("L1-01", "L2-03"), ("L1-03", "L2-05"),
           ("L1-03", "L2-07"), ("L1-03", "L2-08"), ("L1-07", "L2-07")]
    print(w.loc[[k for k in key if k in w.index]].round(3).to_string())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = load()
    for key, fn in [("R", step_r), ("S", step_s), ("T", step_t), ("U", step_u), ("V", step_v), ("W", step_w),
                    ("X", step_x), ("Y", step_y), ("Z", step_z), ("P", step_p), ("C", step_c)]:
        if which in (key, "all"):
            print(f"\n===== {key} =====")
            fn(d)
