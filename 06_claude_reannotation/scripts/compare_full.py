"""Full-corpus comparison of Claude vs GPT vs DeepSeek on the analysis frame (N = 20,901 rows).

Prevalence, label load, pairwise agreement, camp-uniformity of the Claude-vs-DS shift,
outlet label load, and a consistency check on gold sentences that are in the corpus.

Usage: python compare_full.py --run full_low --out ../eval/full
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ANALYSIS_FRAME, DS_L1_RUN, EVAL, L1, L2, RUNS, read_jsonl


def kappa_bin(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def kappa_multi(a, b, labels):
    a, b = np.asarray(a, object), np.asarray(b, object)
    po = (a == b).mean(); pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return (po - pe) / (1 - pe)


def sets(col, sep):
    return col.fillna("").map(lambda s: frozenset(x for x in str(s).split(sep) if x in L2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_low")
    ap.add_argument("--out", default=str(EVAL / "full"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    af = pd.read_csv(ANALYSIS_FRAME)
    c1 = {r["item_id"]: r["parsed"] for r in read_jsonl(RUNS / a.run / "L1.jsonl") if r.get("ok")}
    c2 = {r["item_id"]: r["parsed"] for r in read_jsonl(RUNS / a.run / "L2.jsonl") if r.get("ok")}
    ds1 = pd.read_csv(DS_L1_RUN, usecols=["unit_id", "L1_label"]).drop_duplicates("unit_id").set_index("unit_id")["L1_label"]

    af["L1_claude"] = af.unit_id.map(lambda u: c1[u]["L1"])
    af["L1_conf"] = af.unit_id.map(lambda u: c1[u]["confidence"])
    af["L1_sec"] = af.unit_id.map(lambda u: c1[u]["L1_secondary"])
    af["L1_ds"] = af.unit_id.map(ds1).fillna("")
    S = {"Claude": af.unit_id.map(lambda u: frozenset(l for l in L2 if c2[u][l])),
         "GPT": sets(af.l2_set_gpt, "+"), "DeepSeek": sets(af.l2_set_ds, "+")}
    outlet = af.source.str.extract(r"_(chinatimes|ltn|tvbs)_")[0].map({"chinatimes": "中央社", "ltn": "自由", "tvbs": "TVBS"})
    af["outlet"] = outlet.fillna("辩论/其他")

    rep = [f"# 全量对照：Claude（{a.run}）vs GPT vs DeepSeek\n\n分析框架 N = {len(af):,} 行（{af.unit_id.nunique():,} 个句子）。"
           "GPT 的 L2 是 v10 联合多选；DeepSeek 的 L2 是 v11 逐码二元；Claude 是 v10.1 一次调用逐码判定。\n"]

    # 1. L1 distribution
    rep.append("\n## 1. L1 分布\n\n| 类 | Claude | GPT | DeepSeek（无 fewshot，含弃权） |\n|---|---|---|---|\n")
    for l in L1:
        rep.append(f"| {l} | {(af.L1_claude == l).mean():.1%} | {(af.L1_gpt == l).mean():.1%} | {(af.L1_ds == l).mean():.1%} |\n")
    rep.append(f"| 弃权/空 | 0.0% | {(af.L1_gpt.fillna('') == '').mean():.1%} | {(af.L1_ds == '').mean():.1%} |\n")
    rep.append(f"\nClaude L1 置信度：high {(af.L1_conf == 'high').mean():.1%} / medium {(af.L1_conf == 'medium').mean():.1%} / "
               f"low {(af.L1_conf == 'low').mean():.1%}；给出次要框架的句子 {af.L1_sec.notna().mean():.1%}。\n")

    # 2. L2 prevalence and load
    rep.append("\n## 2. L2 出现率与标签负荷\n\n| 码 | Claude | GPT | DeepSeek |\n|---|---|---|---|\n")
    for l in L2:
        rep.append("| " + l + " | " + " | ".join(f"{S[k].map(lambda s: l in s).mean():.1%}" for k in S) + " |\n")
    rep.append("| 至少一个码 | " + " | ".join(f"{(S[k].map(len) > 0).mean():.1%}" for k in S) + " |\n")
    rep.append("| 平均码数/句 | " + " | ".join(f"{S[k].map(len).mean():.3f}" for k in S) + " |\n")

    # 3. agreement
    rep.append("\n## 3. 两两一致性\n\n")
    m = af.L1_ds != ""
    rep.append(f"L1：Claude–GPT 一致 {(af.L1_claude == af.L1_gpt).mean():.1%}（κ {kappa_multi(af.L1_claude, af.L1_gpt, L1):.2f}）；"
               f"Claude–DeepSeek {(af.L1_claude[m] == af.L1_ds[m]).mean():.1%}（κ {kappa_multi(af.L1_claude[m], af.L1_ds[m], L1):.2f}，DS 有标签的句子）；"
               f"GPT–DeepSeek {(af.L1_gpt[m] == af.L1_ds[m]).mean():.1%}（κ {kappa_multi(af.L1_gpt[m], af.L1_ds[m], L1):.2f}）。\n\n")
    rep.append("L2 逐码 κ：\n\n| 码 | Claude–GPT | Claude–DeepSeek | GPT–DeepSeek |\n|---|---|---|---|\n")
    for l in L2:
        v = {k: S[k].map(lambda s: l in s) for k in S}
        rep.append(f"| {l} | {kappa_bin(v['Claude'], v['GPT']):.2f} | {kappa_bin(v['Claude'], v['DeepSeek']):.2f} | "
                   f"{kappa_bin(v['GPT'], v['DeepSeek']):.2f} |\n")

    # 4. camp uniformity of the shift
    rep.append("\n## 4. 换标注器后的出现率变化是否在三阵营间均匀\n\n"
               "差值 = Claude − 对照标注器（百分点）。若三阵营差值接近，NPMI/配方类的跨阵营比较受影响小；"
               "差值的阵营间极差越大，越要警惕。\n\n| 码 | 对照 | DPP | KMT | TPP | 极差 |\n|---|---|---|---|---|---|\n")
    for l in L2:
        for ref in ["DeepSeek", "GPT"]:
            d = []
            for camp in ["DPP", "KMT", "TPP"]:
                mm = af.camp == camp
                d.append(100 * (S["Claude"][mm].map(lambda s: l in s).mean() - S[ref][mm].map(lambda s: l in s).mean()))
            rep.append(f"| {l} | {ref} | {d[0]:+.1f} | {d[1]:+.1f} | {d[2]:+.1f} | {max(d) - min(d):.1f} |\n")

    # 5. outlet label load
    rep.append("\n## 5. 媒体的标签负荷（平均 L2 码数/句）\n\n| 来源 | 句数 | Claude | GPT | DeepSeek |\n|---|---|---|---|---|\n")
    for o, grp in af.groupby("outlet"):
        rep.append(f"| {o} | {len(grp):,} | " + " | ".join(f"{S[k][grp.index].map(len).mean():.3f}" for k in S) + " |\n")

    # 6. gold sentences inside the corpus: consistency with the pilot/holdout runs
    key = pd.read_csv(Path(__file__).resolve().parents[1] / "input" / "gold_all_key.csv", keep_default_na=False)
    key = key[(key.unit_id != "") & key.unit_id.isin(c1)]
    ev = []
    for run_name, kfile in [("pilot_low", "pilot_100_key.csv"), ("holdout_low", "holdout_39_key.csv")]:
        k = pd.read_csv(Path(__file__).resolve().parents[1] / "input" / kfile, keep_default_na=False)
        k = k[k.unit_id.isin(c1)]
        r1 = {r["item_id"]: r["parsed"] for r in read_jsonl(RUNS / run_name / "L1.jsonl") if r.get("ok")}
        l2_run = "pilot_low_v101" if run_name == "pilot_low" else "holdout_low"
        r2 = {r["item_id"]: r["parsed"] for r in read_jsonl(RUNS / l2_run / "L2.jsonl") if r.get("ok")}
        for rr in k.itertuples():
            if rr.item_id in r1 and rr.item_id in r2:
                ev.append((rr.unit_id, r1[rr.item_id]["L1"], frozenset(l for l in L2 if r2[rr.item_id][l])))
    if ev:
        l1_same = np.mean([c1[u]["L1"] == p for u, p, _ in ev])
        l2_same = np.mean([frozenset(l for l in L2 if c2[u][l]) == s for u, _, s in ev])
        rep.append(f"\n## 6. 一致性复核\n\n金标准句里有 {len(ev)} 句也在全量语料中，被独立标了两次（试点/留出集一次、全量一次，同协议）："
                   f"L1 两次一致 {l1_same:.1%}，L2 整组一致 {l2_same:.1%}。\n")

    (out / "compare_report.md").write_text("".join(rep), encoding="utf-8")
    af.assign(L2_claude=S["Claude"].map(lambda s: "|".join(sorted(s)))).drop(columns=["sent_text"]).to_csv(
        out / "analysis_frame_with_claude.csv", index=False)
    print("".join(rep))


if __name__ == "__main__":
    main()
