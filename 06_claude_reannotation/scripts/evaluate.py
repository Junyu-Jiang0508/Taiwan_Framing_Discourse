"""Evaluate Claude runs against the original gold (label1_set_v2), next to GPT / DeepSeek / ZXH on the same items.

Usage:
  python evaluate.py --key ../input/pilot_100_key.csv --runs pilot_low pilot_medium pilot_high --out ../eval/pilot
  python evaluate.py ... --stability pilot_medium:stab_rerun_medium pilot_medium:stab_rev_medium
  python evaluate.py ... --defs defs_medium

Writes metrics_L1.csv, metrics_L2.csv, summary.csv, usage.csv, error_cases_<run>.md and report.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import EVAL, L1, L2, RUNS, read_jsonl

B = 2000
THRESH = 0.80
rng = np.random.default_rng(20261018)


# ---------------------------------------------------------------- loading
def load_run(run, layer):
    rows = read_jsonl(RUNS / run / f"{layer}.jsonl")
    out = {}
    for r in rows:
        if r.get("ok"):
            out[r["item_id"]] = r          # last ok record wins
    return out


def l1_of(rec):
    return rec["parsed"]["L1"] if rec else ""


def l2_of(rec):
    return frozenset(l for l in L2 if rec and rec["parsed"].get(l) is True)


def split_l2(s):
    return frozenset(x for x in str(s).split("|") if x in L2) if isinstance(s, str) else frozenset()


# ---------------------------------------------------------------- metrics
def prf(y, p):
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    P = tp / (tp + fp) if tp + fp else np.nan
    R = tp / (tp + fn) if tp + fn else np.nan
    F = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return P, R, F


def kappa_bin(y, p):
    n = len(y)
    po = ((y & p) | (~y & ~p)).sum() / n
    pe = (y.mean() * p.mean()) + ((1 - y.mean()) * (1 - p.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def kappa_multi(y, p, labels):
    n = len(y)
    po = (y == p).mean()
    pe = sum((y == l).mean() * (p == l).mean() for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def boot_ci(fn, idx_n):
    vals = []
    for _ in range(B):
        i = rng.integers(0, idx_n, idx_n)
        v = fn(i)
        if not np.isnan(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def l1_table(gold, pred):
    g = np.asarray(gold, dtype=object); p = np.asarray(pred, dtype=object)
    rows = []
    for lab in L1:
        y, q = g == lab, p == lab
        P, R, F = prf(y, q)
        lo, hi = boot_ci(lambda i: prf(y[i], q[i])[2], len(y))
        rows.append(dict(label=lab, support=int(y.sum()), n_pred=int(q.sum()), P=P, R=R, F1=F, F1_lo=lo, F1_hi=hi))
    t = pd.DataFrame(rows)
    acc = float((g == p).mean())
    macro = float(t.F1.mean())
    weighted = float(np.nansum(t.F1 * t.support) / t.support.sum())
    kap = kappa_multi(g, p, L1)
    return t, dict(acc=acc, macro_F1=macro, weighted_F1=weighted, kappa=kap)


def l2_table(gold_sets, pred_sets):
    rows = []
    TP = FP = FN = 0
    for lab in L2:
        y = np.array([lab in s for s in gold_sets]); q = np.array([lab in s for s in pred_sets])
        P, R, F = prf(y, q)
        lo, hi = boot_ci(lambda i: prf(y[i], q[i])[2], len(y))
        TP += int((y & q).sum()); FP += int((~y & q).sum()); FN += int((y & ~q).sum())
        rows.append(dict(label=lab, support=int(y.sum()), n_pred=int(q.sum()), P=P, R=R, F1=F, F1_lo=lo, F1_hi=hi,
                         kappa=kappa_bin(y, q), prev_pred=float(q.mean())))
    t = pd.DataFrame(rows)
    micro = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else np.nan
    exact = float(np.mean([a == b for a, b in zip(gold_sets, pred_sets)]))
    return t, dict(micro_F1=micro, macro_F1=float(t.F1.mean()), exact_set_match=exact)


def macro_l1(g, p):
    return np.nanmean([prf(g == l, p == l)[2] for l in L1])


def macro_l2(G, P):
    return np.nanmean([prf(G[:, j], P[:, j])[2] for j in range(len(L2))])


def paired_diff(fn_a, fn_b, n):
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        a, b = fn_a(i), fn_b(i)
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- report pieces
def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def side_by_side(tables, labels):
    names = list(tables)
    lines = ["| 标签 | 支持数 | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for lab in labels:
        sup = tables[names[0]].set_index("label").at[lab, "support"]
        cells = []
        for n in names:
            r = tables[n].set_index("label").loc[lab]
            mark = " ✓" if r.F1 >= THRESH else ""
            cells.append(f"{fmt(r.F1)}{mark}")
        lines.append(f"| {lab} | {sup} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def usage_summary(run, layer):
    rows = read_jsonl(RUNS / run / f"{layer}.jsonl")
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return None
    u = pd.DataFrame([r["usage"] or {} for r in ok])
    det = pd.DataFrame([(r["usage"] or {}).get("output_tokens_details") or {} for r in ok])
    return dict(run=run, layer=layer, n_ok=len(ok), n_fail=len(rows) - len(ok),
                in_tok=float(u.get("input_tokens", pd.Series([0])).mean()),
                cache_read=float(u.get("cache_read_input_tokens", pd.Series([0])).mean()),
                cache_write=float(u.get("cache_creation_input_tokens", pd.Series([0])).mean()),
                out_tok=float(u.get("output_tokens", pd.Series([0])).mean()),
                thinking_tok=float(det.get("thinking_tokens", pd.Series([0])).mean()) if len(det.columns) else np.nan,
                sec=float(np.mean([r["duration_ms"] or 0 for r in ok]) / 1000),
                cost_usd_per_call=float(np.mean([r["cost_usd"] or 0 for r in ok])))


def error_cases(key, run, recs1, recs2, path):
    out = [f"# 错例清单：{run}\n", "每条：金标准 vs Claude；L1 附理由，L2 附证据片段。\n"]
    for lab in L1:
        fn_ = key[(key.gold_L1 == lab) & (key.c_L1 != lab)]
        fp_ = key[(key.gold_L1 != lab) & (key.c_L1 == lab)]
        out.append(f"\n## {lab}：漏判 {len(fn_)}，误判 {len(fp_)}\n")
        for kind, df in [("漏判", fn_), ("误判", fp_)]:
            for r in df.itertuples():
                rec = recs1.get(r.item_id)
                reason = rec["parsed"].get("reason", "") if rec else ""
                out.append(f"- [{kind}] {r.item_id} 金标准 {r.gold_L1} / Claude {r.c_L1}（{reason}）：{r.sentence}\n")
    for lab in L2:
        g = key.gold_set.map(lambda s: lab in s); c = key.c_set.map(lambda s: lab in s)
        fn_, fp_ = key[g & ~c], key[~g & c]
        out.append(f"\n## {lab}：漏判 {len(fn_)}，误判 {len(fp_)}\n")
        for kind, df in [("漏判", fn_), ("误判", fp_)]:
            for r in df.itertuples():
                rec = recs2.get(r.item_id)
                ev = rec["parsed"].get(f"{lab}_ev", "") if rec else ""
                out.append(f"- [{kind}] {r.item_id} 金标准 {'|'.join(sorted(r.gold_set)) or '∅'} / "
                           f"Claude {'|'.join(sorted(r.c_set)) or '∅'}（证据：{ev}）：{r.sentence}\n")
    Path(path).write_text("".join(out), encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stability", nargs="*", default=[])
    ap.add_argument("--defs", default=None)
    ap.add_argument("--errors-for", default=None, help="run to write error cases for (default: last run)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(a.key, keep_default_na=False)
    key["gold_set"] = key.gold_L2.map(split_l2)
    n = len(key)
    rep = [f"# 试点评估报告\n\n- 评估集：`{Path(a.key).name}`，{n} 句（原版金标准，已剔除与提示词重叠的句子）\n"
           f"- 阈值：每个标签 F1 ≥ {THRESH}（✓ = 达标）；95% CI 为 bootstrap（{B} 次）\n"]

    # annotators: Claude runs + baselines (on items where available)
    l1_preds, l2_preds, recs = {}, {}, {}
    for run in a.runs:
        r1, r2 = load_run(run, "L1"), load_run(run, "L2")
        recs[run] = (r1, r2)
        l1_preds[f"Claude·{run}"] = key.item_id.map(lambda i: l1_of(r1.get(i)))
        l2_preds[f"Claude·{run}"] = key.item_id.map(lambda i: l2_of(r2.get(i)))
        miss = (~key.item_id.isin(r1)).sum(), (~key.item_id.isin(r2)).sum()
        rep.append(f"- {run}：缺失 L1 {miss[0]} 条、L2 {miss[1]} 条（按判错计）\n")
    base_mask = key.unit_id != ""
    for name, c1, c2 in [("GPT", "gpt_L1", "gpt_L2"), ("DeepSeek", "ds_L1", "ds_L2"), ("ZXH(人工)", "zxh_L1", "zxh_L2")]:
        l1_preds[name] = key[c1]
        l2_preds[name] = key[c2].map(split_l2)

    t1, s1, t2, s2 = {}, {}, {}, {}
    for name in l1_preds:
        m = base_mask if name in ("GPT", "DeepSeek") else pd.Series(True, index=key.index)
        t1[name], s1[name] = l1_table(key.gold_L1[m].values, l1_preds[name][m].values)
        t2[name], s2[name] = l2_table(list(key.gold_set[m]), list(l2_preds[name][m]))
    if (~base_mask).any():
        rep.append(f"- GPT/DeepSeek 只在能对上语料的 {int(base_mask.sum())} 句上计算\n")

    pd.concat([t.assign(annotator=k) for k, t in t1.items()]).to_csv(out / "metrics_L1.csv", index=False)
    pd.concat([t.assign(annotator=k) for k, t in t2.items()]).to_csv(out / "metrics_L2.csv", index=False)

    rep.append("\n## 1. 逐标签 F1\n\n### L1（单标签）\n\n" + side_by_side(t1, L1) + "\n")
    rep.append("\n### L2（多标签）\n\n" + side_by_side(t2, L2) + "\n")

    summ = []
    for name in l1_preds:
        n_pass = int((t1[name].F1 >= THRESH).sum() + (t2[name].F1 >= THRESH).sum())
        summ.append(dict(annotator=name, L1_acc=s1[name]["acc"], L1_macro_F1=s1[name]["macro_F1"],
                         L1_kappa=s1[name]["kappa"], L2_micro_F1=s2[name]["micro_F1"],
                         L2_macro_F1=s2[name]["macro_F1"], L2_exact=s2[name]["exact_set_match"],
                         labels_passing=f"{n_pass}/15"))
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    rep.append("\n## 2. 总体指标\n\n| 标注者 | L1 准确率 | L1 macro-F1 | L1 κ | L2 micro-F1 | L2 macro-F1 | L2 整组一致 | 达标标签数 |\n"
               "|---|---|---|---|---|---|---|---|\n")
    for r in summ.itertuples():
        rep.append(f"| {r.annotator} | {fmt(r.L1_acc)} | {fmt(r.L1_macro_F1)} | {fmt(r.L1_kappa)} | "
                   f"{fmt(r.L2_micro_F1)} | {fmt(r.L2_macro_F1)} | {fmt(r.L2_exact)} | {r.labels_passing} |\n")

    # per-label CI and precision/recall for each Claude run
    for run in a.runs:
        name = f"Claude·{run}"
        rep.append(f"\n### {name}：逐标签 P / R / F1 [95% CI]\n\n| 标签 | 支持 | 预测数 | P | R | F1 [CI] |\n|---|---|---|---|---|---|\n")
        for t in (t1[name], t2[name]):
            for r in t.itertuples():
                rep.append(f"| {r.label} | {r.support} | {r.n_pred} | {fmt(r.P)} | {fmt(r.R)} | "
                           f"{fmt(r.F1)} [{fmt(r.F1_lo)}, {fmt(r.F1_hi)}] |\n")

    # paired comparison vs baselines on matched items
    rep.append("\n## 3. 配对比较（同一批句子，bootstrap；差值 = Claude − 基线）\n\n| Claude 配置 | 对比 | L1 macro-F1 差 [CI] | L2 macro-F1 差 [CI] |\n|---|---|---|---|\n")
    m = base_mask.values
    G1 = key.gold_L1.values[m]
    G2 = np.array([[l in s for l in L2] for s in key.gold_set[m]])
    for run in a.runs:
        c = f"Claude·{run}"
        C1 = l1_preds[c].values[m]; C2 = np.array([[l in s for l in L2] for s in l2_preds[c][m]])
        for bname in ["GPT", "DeepSeek"]:
            B1 = l1_preds[bname].values[m]; B2 = np.array([[l in s for l in L2] for s in l2_preds[bname][m]])
            d1 = paired_diff(lambda i: macro_l1(G1[i], C1[i]), lambda i: macro_l1(G1[i], B1[i]), m.sum())
            d2 = paired_diff(lambda i: macro_l2(G2[i], C2[i]), lambda i: macro_l2(G2[i], B2[i]), m.sum())
            rep.append(f"| {run} | {bname} | {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}] | "
                       f"{d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}] |\n")

    # sensitivity subsets
    rep.append("\n## 4. 敏感性\n")
    sub = key.in_hard_pool.astype(str).str.lower() != "true"
    rep.append(f"\n**剔除 hard-case 池句（GPT 生产提示词里见过的，{int((~sub).sum())} 句）后的 macro-F1：**\n\n"
               "| 标注者 | L1 macro-F1 | L2 macro-F1 |\n|---|---|---|\n")
    for name in l1_preds:
        mm = (sub & (base_mask if name in ("GPT", "DeepSeek") else True)).values
        _, a1 = l1_table(key.gold_L1.values[mm], l1_preds[name].values[mm])
        _, a2 = l2_table(list(key.gold_set[mm]), list(l2_preds[name][mm]))
        rep.append(f"| {name} | {fmt(a1['macro_F1'])} | {fmt(a2['macro_F1'])} |\n")
    mot = key.v91_motivating.astype(str).str.lower() == "true"
    if mot.any():
        rep.append(f"\n**剔除促成 v9.1 修订的 {int(mot.sum())} 句（金标准 L1-07 而 GPT、DS 都判 L1-02）后的 L1-02 / L1-07 F1：**\n\n"
                   "| 标注者 | L1-02 | L1-07 |\n|---|---|---|\n")
        for name in l1_preds:
            mm = (~mot & (base_mask if name in ("GPT", "DeepSeek") else True)).values
            tt, _ = l1_table(key.gold_L1.values[mm], l1_preds[name].values[mm])
            tt = tt.set_index("label")
            rep.append(f"| {name} | {fmt(tt.at['L1-02', 'F1'])} | {fmt(tt.at['L1-07', 'F1'])} |\n")

    # prevalence vs gold
    rep.append("\n## 5. 各码预测出现率（金标准是富集样本，只作相对比较）\n\n| 标签 | 金标准 | "
               + " | ".join(l2_preds) + " |\n|---|---|" + "---|" * len(l2_preds) + "\n")
    for lab in L2:
        gp = key.gold_set.map(lambda s: lab in s).mean()
        cells = [fmt(l2_preds[nm].map(lambda s: lab in s).mean()) for nm in l2_preds]
        rep.append(f"| {lab} | {fmt(gp)} | " + " | ".join(cells) + " |\n")

    # usage
    us = [x for run in a.runs for layer in ("L1", "L2") if (x := usage_summary(run, layer))]
    if us:
        u = pd.DataFrame(us); u.to_csv(out / "usage.csv", index=False)
        rep.append("\n## 6. 额度与耗时（每次调用平均；cost 为 API 等价价格，仅供换算，订阅不另收费）\n\n"
                   "| 配置 | 层 | 成功/失败 | 输入 | 缓存读 | 缓存写 | 输出 | 其中思考 | 秒 | API 等价 $ |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in u.itertuples():
            rep.append(f"| {r.run} | {r.layer} | {r.n_ok}/{r.n_fail} | {r.in_tok:.0f} | {r.cache_read:.0f} | "
                       f"{r.cache_write:.0f} | {r.out_tok:.0f} | {fmt(r.thinking_tok, 0)} | {r.sec:.1f} | {r.cost_usd_per_call:.4f} |\n")
        rep.append("\n全量外推（20,900 句 × 2 层 = 41,800 次调用）：\n\n| 配置 | 输出 token 合计 | API 等价 $ | 4 并发墙钟（小时） |\n|---|---|---|---|\n")
        for run in a.runs:
            uu = u[u.run == run]
            if len(uu) == 2:
                out_tok = 20900 * uu.out_tok.sum(); cost = 20900 * uu.cost_usd_per_call.sum()
                hours = 20900 * uu.sec.sum() / 4 / 3600
                rep.append(f"| {run} | {out_tok/1e6:.1f}M | {cost:,.0f} | {hours:.1f} |\n")

    # stability
    if a.stability:
        rep.append("\n## 7. 稳定性（同一批句子两次运行的一致率）\n\n| 对比 | 句数 | L1 一致 | L2 整组一致 | L2 逐码一致（最低码） |\n|---|---|---|---|---|\n")
        for pair in a.stability:
            ra, rb = pair.split(":")
            a1, a2 = load_run(ra, "L1"), load_run(ra, "L2")
            b1, b2 = load_run(rb, "L1"), load_run(rb, "L2")
            ids = sorted(set(b1) & set(a1) & set(a2) & set(b2))
            if not ids:
                continue
            l1a = np.mean([l1_of(a1[i]) == l1_of(b1[i]) for i in ids])
            ex = np.mean([l2_of(a2[i]) == l2_of(b2[i]) for i in ids])
            per = {lab: np.mean([(lab in l2_of(a2[i])) == (lab in l2_of(b2[i])) for i in ids]) for lab in L2}
            worst = min(per, key=per.get)
            rep.append(f"| {ra} vs {rb} | {len(ids)} | {l1a:.2f} | {ex:.2f} | {per[worst]:.2f}（{worst}） |\n")

    # definition recovery
    if a.defs:
        d1, d2 = load_run(a.defs, "L1"), load_run(a.defs, "L2")
        rep.append(f"\n## 8. 定义回收测试（把每个码的定义本身当作句子来标；Halterman & Keith 的无标签行为测试）\n\n")
        ok1 = [(lab, l1_of(d1.get(f'DEF_{lab}'))) for lab in L1]
        ok2 = [(lab, sorted(l2_of(d2.get(f'DEF_{lab}')))) for lab in L2]
        rep.append("L1：" + "，".join(f"{lab}→{p}{'' if p == lab else ' ✗'}" for lab, p in ok1) + "\n\n")
        rep.append("L2：" + "，".join(f"{lab}→{'|'.join(p) or '∅'}{'' if lab in p else ' ✗'}" for lab, p in ok2) + "\n")

    # error cases for one run
    er = a.errors_for or a.runs[-1]
    r1, r2 = recs[er]
    key["c_L1"] = l1_preds[f"Claude·{er}"]; key["c_set"] = l2_preds[f"Claude·{er}"]
    error_cases(key, er, r1, r2, out / f"error_cases_{er}.md")
    rep.append(f"\n错例清单见 `error_cases_{er}.md`。\n")
    (out / "report.md").write_text("".join(rep), encoding="utf-8")
    print("".join(rep))


if __name__ == "__main__":
    main()
