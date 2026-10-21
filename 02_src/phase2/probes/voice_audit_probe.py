#!/usr/bin/env python3
"""Voice audit: what the `speakers` split in corpus_composition_probe actually measures.

`speakers` is built in s01_preprocess.find_speakers by substring-matching 23 strings
(six candidate names, three party names, three camp colours) anywhere in the sentence.
It records who is *named*, not who is *speaking*. This probe re-cuts the corpus so that
naming and speaking are separated, then re-runs the tests the paper plan rested on.

K  validity   what each published voice bucket contains (role, quotes, name+say-verb)
L  campperm   Layer A camp permutation on the corrected slices, plus a size-matched check
M  edges      degree-preserving null on the corrected slices; own-vs-reporter contrasts
N  layera     Layer A composition by outlet and by camp, cluster-bootstrap CIs
O  lexical    the econ x democracy result with the literal string 民主 as outcome (no LLM on the L2 side)
P  codebook   which pairs the v11 codebook routes apart, against their observed pooled sign
Q  emphasis   L2 prevalence by outlet (document-level permutation) and by camp

Slices (news only unless stated):
  S1 debate                     candidate speech, the only unmediated voice in the corpus
  S2 own named + speaking       an own-camp name followed within 12 chars by a say-verb
  S3 own named, not speaking    an own-camp name, no such say-verb
  S4 no name, speech-marked     pronoun + say-verb, or a 「」 quotation
  S5 no name, plain
  cross                         names only other camps

Usage: python voice_audit_probe.py [K|L|M|N|O|P|Q|all] [--reps N]
"""
from __future__ import annotations

import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "utils"))
import config as C  # noqa: E402
from corpus_composition_probe import CODES, composition, load  # noqa: E402
from tier1_reestimation import ROOT, _fdsm_table, bh  # noqa: E402

OUT = f"{ROOT}/12_voice_audit"
SAY = C.SAY_VERBS_RE.pattern
CAMP_NAMES = {p: "|".join(sorted([k for k, v in C.ACTOR_PARTY_MAP.items() if v == p], key=len, reverse=True))
              for p in ["DPP", "KMT", "TPP"]}
CANDIDATES = {"賴清德", "赖清德", "蕭美琴", "萧美琴", "侯友宜", "侯友谊", "趙少康", "赵少康", "柯文哲", "吳欣盈", "吴欣盈"}
PARTIES = {"民進黨", "民进党", "國民黨", "国民党", "民眾黨", "民众党"}
BLOCKS = {"cohesion": ["c04", "c06", "c07", "c08"], "boundary": ["c01", "c02", "c03", "c05"], "all8": CODES}


def frame() -> pd.DataFrame:
    d = load()
    d["corner_quote"] = d.sent_text.str.contains("「", regex=False)
    d["pron_say"] = d.sent_text.str.contains(rf"(?:他|她|其|對方|兩人|雙方)[^。；;]{{0,6}}?{SAY}")
    d["own_say"] = [bool(CAMP_NAMES.get(c)) and re.search(rf"(?:{CAMP_NAMES[c]})[^。；;「」]{{0,12}}?{SAY}", t) is not None
                    for c, t in zip(d.camp, d.sent_text)]
    d["slice"] = np.select(
        [d.genre == "debate",
         (d.genre == "news") & d.own_say,
         (d.genre == "news") & (d.voice == "own"),
         d.voice == "cross",
         (d.voice == "reporter") & (d.pron_say | d.corner_quote)],
        ["S1 debate", "S2 own named+speaking", "S3 own named, not speaking", "cross", "S4 no name, speech-marked"],
        default="S5 no name, plain")
    return d


def step_k(d: pd.DataFrame) -> None:
    def trigger(camp: str, text: str) -> str:
        ks = [k for k, v in C.ACTOR_PARTY_MAP.items() if v == camp and k in text]
        if any(k in CANDIDATES for k in ks):
            return "candidate"
        if any(k in PARTIES for k in ks):
            return "party_name"
        return "camp_colour" if ks else "none"

    own = d[d.voice == "own"]
    rows = []
    for v, g in d.groupby("voice"):
        rows.append({"voice": v, "n": len(g), "debate": int((g.genre == "debate").sum()),
                     "role_claim": round((g.role == "claim").mean(), 3), "role_quote": int((g.role == "quote").sum()),
                     "corner_quote": round(g.corner_quote.mean(), 3), "pron_say": round(g.pron_say.mean(), 3),
                     "own_name_say": round(g.own_say.mean(), 3),
                     "multi_party_named": round((g.sp_parties.str.len() > 1).mean(), 3)})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/K_voice_bucket_contents.csv", index=False)
    print(t.to_string(index=False))
    trig = pd.Series([trigger(c, s) for c, s in zip(own.camp, own.sent_text)]).value_counts(normalize=True).round(3)
    print("own-bucket trigger:", trig.to_dict())
    share = pd.crosstab(d.outlet, d.camp, values=(d.voice == "own"), aggfunc="mean").round(3)
    share.to_csv(f"{OUT}/K_own_named_share_outlet_camp.csv")
    print(share.to_string())
    print(d.slice.value_counts().sort_index().to_string())


def _stat(l1i: np.ndarray, M: np.ndarray, camp_i: np.ndarray, n_l1: int, n_camp: int) -> float:
    """Vectorised form of corpus_composition_probe.step_i's statistic."""
    S = np.zeros((n_camp, n_l1, M.shape[1]))
    np.add.at(S, (camp_i, l1i), M)
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = S.sum(1)
        base = tot / tot.sum(1, keepdims=True)
        rs = S.sum(2, keepdims=True)
        comp = (S / rs) / base[:, None, :]
    comp[np.broadcast_to(rs == 0, comp.shape)] = np.nan
    return float(sum(np.nanmean(np.abs(comp[a] - comp[b])) for a, b in itertools.combinations(range(n_camp), 2)))


def camp_perm(sub: pd.DataFrame, l1s: list[str], reps: int, seed: int) -> dict[str, dict[str, float]]:
    sub = sub.dropna(subset=["L1_gpt", "camp"]).reset_index(drop=True)
    camps = sorted(sub.camp.unique())
    doc_camp = sub.groupby("doc_id").camp.first()
    dpos = pd.Index(doc_camp.index).get_indexer(sub.doc_id)
    l1i = pd.Index(l1s).get_indexer(sub.L1_gpt)
    code = {c: i for i, c in enumerate(camps)}
    rng = np.random.default_rng(seed)
    perms = [pd.Series(rng.permutation(doc_camp.to_numpy())).map(code).to_numpy()[dpos] for _ in range(reps)]
    obs_camp = doc_camp.map(code).to_numpy()[dpos]
    res = {}
    for name, block in BLOCKS.items():
        M = sub[block].to_numpy(float)
        obs = _stat(l1i, M, obs_camp, len(l1s), len(camps))
        null = np.array([_stat(l1i, M, p, len(l1s), len(camps)) for p in perms])
        res[name] = {"observed": obs, "null_mean": null.mean(), "z": (obs - null.mean()) / null.std(),
                     "p_perm": (1 + (null >= obs).sum()) / (1 + reps)}
    return res


def step_l(d: pd.DataFrame, reps: int) -> None:
    l1s = sorted(d.L1_gpt.dropna().unique())
    slices = {"pooled": d, "own (published)": d[d.voice == "own"], "reporter (published)": d[d.voice == "reporter"], "attributed own+cross (published)": d[d.voice != "reporter"],
              "cna(wire)": d[d.outlet == "cna"], "cna ∩ own": d[(d.outlet == "cna") & (d.voice == "own")]}
    slices.update({s: g for s, g in d.groupby("slice")})
    rows = []
    for name, sub in slices.items():
        for block, v in camp_perm(sub, l1s, reps, 51).items():
            rows.append({"slice": name, "block": block, "n": len(sub), "n_docs": sub.doc_id.nunique(),
                         "observed": round(v["observed"], 4), "null_mean": round(v["null_mean"], 4),
                         "z": round(v["z"], 2), "p_perm": round(v["p_perm"], 4)})
    t = pd.DataFrame(rows)
    t["q_bh_all_rows"] = bh(t.p_perm.to_numpy()).round(4)
    t.to_csv(f"{OUT}/L_camp_permutation_corrected_slices.csv", index=False)
    print(t.pivot(index="slice", columns="block", values="p_perm").to_string())

    # S2 and S3 are both news sentences naming the own camp; they differ only in whether the actor speaks.
    s2, s3 = d[d.slice == "S2 own named+speaking"], d[d.slice == "S3 own named, not speaking"]
    draws = [camp_perm(s3.sample(len(s2), random_state=300 + k), l1s, 300, 400 + k) for k in range(30)]
    m = pd.DataFrame([{"block": b, "z": r[b]["z"]} for r in draws for b in r])
    m = m.groupby("block").z.describe(percentiles=[.25, .5, .75]).round(2)
    m.to_csv(f"{OUT}/L_size_matched_S3_to_S2.csv")
    print(m.to_string())


def step_m(d: pd.DataFrame, reps: int) -> None:
    t = pd.concat([_fdsm_table(g, reps, 17, s) for s, g in d.groupby("slice")], ignore_index=True)
    t.to_csv(f"{OUT}/M_fdsm_corrected_slices.csv", index=False)
    keep = ["01-02", "02-08", "04-08", "07-08", "06-07", "04-07", "06-08", "04-06", "05-06"]
    print(t[t.pair.isin(keep)].pivot(index="pair", columns="stratum", values="ratio").round(3).to_string())

    # A direct test of the difference, instead of comparing one significant with one non-significant ratio.
    # The null-model s.d. stands in for the sampling s.d. of each ratio, so read the contrast as approximate.
    f = pd.read_csv(f"{ROOT}/11_tier1_reestimation/F_corpus_strata_fdsm.csv")
    f["se"] = np.where(f.z != 0, (f.n_ab - f.null_mean) / f.z, np.nan) / f.null_mean
    # the saved table has carried two label vintages; match on meaning, not on spelling
    pick = lambda pats: f[f.stratum.str.contains("|".join(pats), case=False, regex=True)]
    a = pick(["own", "自述"]).drop_duplicates("pair").set_index("pair")
    b = pick(["reporter", "no speaker", "记者"]).drop_duplicates("pair").set_index("pair")
    z = (a.ratio - b.ratio) / np.sqrt(a.se ** 2 + b.se ** 2)
    p = pd.Series(2 * norm.sf(z.abs()), index=z.index)
    c = pd.DataFrame({"own_named": a.ratio, "no_name": b.ratio, "diff": (a.ratio - b.ratio).round(3),
                      "z": z.round(2), "p": p.round(4), "q_bh": bh(p.to_numpy()).round(4)}).sort_values("p")
    c.to_csv(f"{OUT}/M_own_vs_noname_contrast.csv")
    print(c.head(8).to_string())


def step_n(d: pd.DataFrame, reps: int) -> None:
    cells = [("L1-01", "c08"), ("L1-01", "c06"), ("L1-01", "c03"), ("L1-03", "c05"), ("L1-03", "c07"),
             ("L1-03", "c08"), ("L1-07", "c07")]
    d = d.dropna(subset=["L1_gpt"])
    slices = {"cna(wire)": d[d.outlet == "cna"], "ltn(green)": d[d.outlet == "ltn"], "tvbs(blue)": d[d.outlet == "tvbs"],
              "DPP": d[d.camp == "DPP"], "KMT": d[d.camp == "KMT"], "TPP": d[d.camp == "TPP"],
              "ltn ∩ DPP": d[(d.outlet == "ltn") & (d.camp == "DPP")],
              "cna ∩ KMT": d[(d.outlet == "cna") & (d.camp == "KMT")]}
    rows = []
    for name, sub in slices.items():
        sub = sub.reset_index(drop=True)
        point = composition(sub)
        docs, idx = sub.doc_id.unique(), sub.groupby("doc_id").indices
        rng = np.random.default_rng(41)
        boots = pd.concat([composition(sub.iloc[np.concatenate([idx[k] for k in rng.choice(docs, len(docs))])])
                           for _ in range(reps)])
        lo, hi = boots.groupby(level=0).quantile(0.025), boots.groupby(level=0).quantile(0.975)
        for l1, c in cells:
            rows.append({"slice": name, "cell": f"{l1}×L2-{c[1:]}", "n_l1": int((sub.L1_gpt == l1).sum()),
                         "comp": round(point.loc[l1, c], 3), "lo": round(lo.loc[l1, c], 3), "hi": round(hi.loc[l1, c], 3),
                         "excludes_1": not (lo.loc[l1, c] <= 1 <= hi.loc[l1, c])})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/N_layerA_by_outlet_camp.csv", index=False)
    print(t.pivot(index="cell", columns="slice", values="comp").to_string())


def step_o(d: pd.DataFrame) -> None:
    import statsmodels.api as sm

    d = d.dropna(subset=["L1_gpt"]).copy()
    d["lex_demo"] = d.sent_text.str.contains("民主", regex=False).astype(int)
    lift = d.groupby("L1_gpt").lex_demo.mean() / d.lex_demo.mean()
    by_outlet = {o: g[g.L1_gpt == "L1-01"].lex_demo.mean() / g.lex_demo.mean()
                 for o, g in d.groupby("outlet") if o in ("cna", "ltn")}
    d["econ"] = (d.L1_gpt == "L1-01").astype(float)
    d["econ_m"] = d.groupby("doc_id").econ.transform("mean")
    d["z_len"] = (d["len"] - d["len"].mean()) / d["len"].std()
    X = sm.add_constant(pd.concat([d[["econ", "econ_m", "z_len"]],
                                   pd.get_dummies(d.outlet, prefix="o", drop_first=True).astype(float)], axis=1))
    rows = [{"quantity": f"lift_lexical_民主|{k}", "value": round(v, 3)} for k, v in lift.items()]
    rows += [{"quantity": f"lift_lexical_民主|L1-01|{k}", "value": round(v, 3)} for k, v in by_outlet.items()]
    for y in ["lex_demo", "c08"]:
        m = sm.GLM(d[y].astype(float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.doc_id})
        rows += [{"quantity": f"{y}|within_OR", "value": round(float(np.exp(m.params.econ)), 3), "p": round(float(m.pvalues.econ), 4)},
                 {"quantity": f"{y}|between_OR", "value": round(float(np.exp(m.params.econ_m)), 3), "p": round(float(m.pvalues.econ_m), 4)}]
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/O_lexical_democracy_check.csv", index=False)
    print(t.to_string(index=False))


def step_p() -> None:
    guide = pd.read_csv(f"{ROOT}/../../l2_binary_corpus_out-20260705T020641Z-3-001/l2_binary_corpus_out/"
                        "02_annotation_guide_label2_v11.csv")
    pooled = pd.read_csv(f"{ROOT}/11_tier1_reestimation/A_fdsm_28pairs.csv").query("stratum == 'pooled'").set_index("pair")
    rows = []
    for _, r in guide.iterrows():
        for col in ["exclusion_criteria", "boundary_test"]:
            for clause in re.split(r"(?=（\d）)|(?=【)", str(r[col])):
                for other in set(re.findall(r"L2-0\d", clause)) - {r.label_id}:
                    a, b = sorted([r.label_id[-2:], other[-2:]])
                    rows.append({"pair": f"{a}-{b}", "code": r.label_id, "field": col,
                                 "rule": "permit_colabel" if re.search(r"並標|可同時", clause) else "route_away",
                                 "clause": clause.strip()[:80]})
    t = pd.DataFrame(rows).drop_duplicates(["pair", "code", "field", "rule"])
    t["pooled_ratio"] = t.pair.map(pooled.ratio)
    t["pooled_q"] = t.pair.map(pooled.q_bh)
    t.to_csv(f"{OUT}/P_codebook_rules_vs_edges.csv", index=False)
    routed = t[t.rule == "route_away"].drop_duplicates("pair")
    sign = np.where(routed.pooled_q >= .05, "null", np.where(routed.pooled_ratio > 1, "positive", "negative"))
    print("pairs with a route-away rule:", len(routed), pd.Series(sign).value_counts().to_dict())


def step_q(d: pd.DataFrame, reps: int) -> None:
    two = d[d.outlet.isin(["cna", "ltn"])]   # wire vs green; tvbs is handled in the partisan probe
    docs = two.groupby("doc_id").outlet.first()

    def gap(x: pd.DataFrame) -> pd.Series:
        p = x.groupby("outlet")[CODES].mean()
        return p.loc["cna"] - p.loc["ltn"]

    obs = gap(two)
    rng = np.random.default_rng(5)
    null = pd.DataFrame([gap(two.assign(outlet=two.doc_id.map(pd.Series(rng.permutation(docs.to_numpy()), index=docs.index))))
                         for _ in range(reps)])
    t = two.groupby("outlet")[CODES].mean().T
    t["diff_pp"] = (obs * 100).round(2)
    t["p_docperm"] = (((null.abs() >= obs.abs()).sum() + 1) / (reps + 1)).round(4)
    t = t.join(d.groupby("camp")[CODES].mean().T.add_prefix("camp_"))
    t.round(4).to_csv(f"{OUT}/Q_prevalence_by_outlet_camp.csv")
    print(t.round(4).to_string())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1000
    d = frame()
    print(f"N={len(d)}", flush=True)
    if which in ("K", "all"):
        step_k(d)
    if which in ("L", "all"):
        step_l(d, reps)
    if which in ("M", "all"):
        step_m(d, min(reps, 500))
    if which in ("N", "all"):
        step_n(d, min(reps, 400))
    if which in ("O", "all"):
        step_o(d)
    if which in ("P", "all"):
        step_p()
    if which in ("Q", "all"):
        step_q(d, min(reps, 500))
