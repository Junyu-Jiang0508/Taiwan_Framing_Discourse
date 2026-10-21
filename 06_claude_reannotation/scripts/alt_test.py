"""Alternative Annotator Test (Calderon, Reichart & Dror 2025, ACL) for Claude vs human coders,
plus human-human reliability and (for gold2) design-weighted, document-clustered F1.

Inputs: one CSV per human coder exported from the annotation app (columns id, your_L1, your_L2),
and the Claude labels (looked up by unit_id through the package design / key files).

Usage:
  # full alt-test on the new gold (needs >= 3 coders)
  python alt_test.py --package gold2 --coders a.csv b.csv c.csv --eps 0.10 0.15 0.20 --out ../eval/gold2
  # single-reference variant on the existing gold (consensus gold vs one third-party coder)
  python alt_test.py --package gold_v2 --out ../eval/alt_single
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import EVAL, INPUT, L1, L2, RUNS, SUB, read_jsonl

rng = np.random.default_rng(20261020)


def sset(s):
    return frozenset(x for x in str(s).replace(",", "|").split("|") if x in L2)


def agree_l1(a, b):
    return float(a == b)


def agree_l2_jacc(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def agree_code(code):
    return lambda a, b: float((code in a) == (code in b))


def bh(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank, i in enumerate(order[::-1], 1):
        k = n - rank + 1
        prev = min(prev, p[i] * n / k); adj[i] = prev
    return adj


def alt_test(humans, llm, agree, eps, alpha=0.05):
    """humans: dict name -> list of labels (aligned); llm: list. Returns winning rate, AP, per-human detail."""
    names = list(humans); m = len(names); n = len(llm)
    pvals, adv, detail = [], [], []
    for j in names:
        others = [k for k in names if k != j]
        s_f = np.array([np.mean([agree(llm[i], humans[k][i]) for k in others]) for i in range(n)])
        s_h = np.array([np.mean([agree(humans[j][i], humans[k][i]) for k in others]) for i in range(n)])
        d = s_h - s_f
        se = d.std(ddof=1) / np.sqrt(n)
        t = (d.mean() - eps) / se if se > 0 else (-np.inf if d.mean() < eps else np.inf)
        pvals.append(stats.t.cdf(t, n - 1))
        adv.append(np.mean(s_f >= s_h))
        detail.append(dict(human=j, llm_align=s_f.mean(), human_align=s_h.mean(), diff=d.mean()))
    rej = bh(pvals) < alpha
    return dict(winning_rate=rej.mean(), passed=rej.mean() >= 0.5, advantage_prob=float(np.mean(adv)),
                detail=pd.DataFrame(detail).assign(p=pvals, rejected=rej))


def kappa_bin(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def weighted_f1(y, p, w, clusters, B=2000):
    """Horvitz-Thompson weighted F1 with a cluster (document) bootstrap CI."""
    y, p, w = np.asarray(y, bool), np.asarray(p, bool), np.asarray(w, float)

    def f(idx):
        tp = (w[idx] * (y[idx] & p[idx])).sum(); fp = (w[idx] * (~y[idx] & p[idx])).sum()
        fn = (w[idx] * (y[idx] & ~p[idx])).sum()
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    cl = pd.Series(clusters); groups = [np.where(cl.values == c)[0] for c in cl.unique()]
    boots = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        boots.append(f(idx))
    boots = [b for b in boots if not np.isnan(b)]
    return f(np.arange(len(y))), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def load_claude_by_unit_maj3(config="primary"):
    """primary (pre-registered 2026-10-20) = Run_claude_final_*: 3-run majority, L2-07 from the code-specific prompt.
    joint = Run_claude_maj3_*: 3-run majority with L2-07 from the joint call (sensitivity)."""
    from common import ROOT
    base = ROOT / "03_outputs" / "01_results_labelings" / "01_results_datasets"
    ptr = base / ("claude_final_run.txt" if config == "primary" else "claude_maj3_run.txt")
    target = ROOT / ptr.read_text().strip()
    target = target / "final_results.csv" if target.is_dir() else target
    df = pd.read_csv(target, keep_default_na=False, encoding="utf-8-sig")
    df = df[df.claude_annotated.astype(str).str.lower().eq("true")].drop_duplicates("unit_id")
    return dict(zip(df.unit_id, df.L1_label)), {u: sset(v) for u, v in zip(df.unit_id, df.L2_labels)}


def load_claude_by_unit(config="primary"):
    """primary = full_low; secondary = full_low with L2-07 replaced by the code-specific run (pre-registered)."""
    c1 = {r["item_id"]: r["parsed"]["L1"] for r in read_jsonl(RUNS / "full_low" / "L1.jsonl") if r.get("ok")}
    c2 = {r["item_id"]: set(l for l in L2 if r["parsed"][l]) for r in read_jsonl(RUNS / "full_low" / "L2.jsonl") if r.get("ok")}
    if config == "secondary":
        sp = {r["item_id"]: r["parsed"]["value"] for r in read_jsonl(RUNS / "full_code_L2-07" / "L2-07.jsonl") if r.get("ok")}
        for u, v in sp.items():
            (c2[u].add if v else c2[u].discard)("L2-07")
    return c1, {u: frozenset(v) for u, v in c2.items()}


def read_coder(path):
    df = pd.read_csv(path, keep_default_na=False)
    return df.set_index(df["id"].astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, choices=["gold2", "review", "gold_v2"])
    ap.add_argument("--coders", nargs="*", default=[])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="primary", choices=["primary", "joint"])
    ap.add_argument("--single-run", action="store_true", help="sensitivity: use run 1 (full_low) instead of the 3-run majority")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = [f"# Alt-test 报告（{a.package}）\n\n"
           "ε 的参考（Calderon et al. 2025）：众包 .10，熟练编码员 .15，专家 .20。通过 = 赢率 ω ≥ .5（BH-FDR 校正后）。\n"]

    if a.package == "gold_v2":
        # single-reference variant: consensus gold as reference, ZXH as the human, Claude final protocol
        tri = pd.read_csv(EVAL / "alt_single_input.csv", keep_default_na=False)
        g1, z1, c1 = tri.gold_L1, tri.zxh_L1, tri.c_L1
        g2, z2, c2 = tri.gold_L2.map(sset), tri.zxh_L2.map(sset), tri.c_L2.map(sset)
        rep.append(f"\n单一参考变体，n = {len(tri)}：以双人共识金标准为参照，比较第三方 ZXH 与 Claude。\n\n"
                   "| 指标 | ZXH | Claude | 差 | " + " | ".join(f"ε={e:.2f}" for e in a.eps) + " |\n|---|---|---|---|" + "---|" * len(a.eps) + "\n")
        for name, fz, fc in [("L1 准确", [agree_l1(x, y) for x, y in zip(z1, g1)], [agree_l1(x, y) for x, y in zip(c1, g1)]),
                             ("L2 Jaccard", [agree_l2_jacc(x, y) for x, y in zip(z2, g2)], [agree_l2_jacc(x, y) for x, y in zip(c2, g2)])]:
            d = np.array(fz) - np.array(fc); se = d.std(ddof=1) / np.sqrt(len(d))
            res = ["✓" if stats.t.cdf((d.mean() - e) / se, len(d) - 1) < 0.05 else "✗" for e in a.eps]
            rep.append(f"| {name} | {np.mean(fz):.3f} | {np.mean(fc):.3f} | {d.mean():+.3f} | " + " | ".join(res) + " |\n")
        (out / "report.md").write_text("".join(rep), encoding="utf-8"); print("".join(rep)); return

    assert len(a.coders) >= 3 or a.package == "review", "full alt-test needs >= 3 human coders"
    if a.package == "gold2":
        design = pd.read_csv(SUB / "human_tasks" / "P1_gold2_DESIGN_do_not_show.csv")
        c1, c2 = load_claude_by_unit("primary") if a.single_run else load_claude_by_unit_maj3(a.config)
        ids = design.app_id.astype(int).tolist()
        llm1 = [c1[u] for u in design.unit_id]; llm2 = [c2[u] for u in design.unit_id]
    else:
        keyr = pd.read_csv(SUB / "human_tasks" / "P2_review_gold_KEY_do_not_show.csv")
        ids = keyr.app_id.astype(int).tolist(); llm1 = llm2 = None

    coders = {Path(p).stem: read_coder(p) for p in a.coders}
    H1 = {n: [c.at[i, "your_L1"] for i in ids] for n, c in coders.items()}
    H2 = {n: [sset(c.at[i, "your_L2"]) for i in ids] for n, c in coders.items()}

    # human-human reliability per code
    names = list(coders)
    rep.append("\n## 人–人一致（逐对 κ）\n\n| 码 | " + " | ".join(f"{x}–{y}" for i, x in enumerate(names) for y in names[i + 1:]) + " |\n|---|"
               + "---|" * (len(names) * (len(names) - 1) // 2) + "\n")
    for code in L1 + L2:
        cells = []
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                if code.startswith("L1"):
                    cells.append(f"{kappa_bin([v == code for v in H1[x]], [v == code for v in H1[y]]):.2f}")
                else:
                    cells.append(f"{kappa_bin([code in v for v in H2[x]], [code in v for v in H2[y]]):.2f}")
        rep.append(f"| {code} | " + " | ".join(cells) + " |\n")

    if llm1 is not None:
        rep.append("\n## Alt-test：Claude 能否替代其中一名人工\n\n| 指标 | " + " | ".join(f"ε={e:.2f}" for e in a.eps) + " | 优势概率 ρ |\n|---|"
                   + "---|" * (len(a.eps) + 1) + "\n")
        tests = [("L1（准确）", H1, llm1, agree_l1), ("L2（Jaccard）", H2, llm2, agree_l2_jacc)]
        tests += [(code, H1, llm1, (lambda c: (lambda x, y: float((x == c) == (y == c))))(code)) for code in L1]
        tests += [(code, H2, llm2, agree_code(code)) for code in L2]
        for name, H, llm, f in tests:
            res = [alt_test(H, llm, f, e) for e in a.eps]
            rep.append(f"| {name} | " + " | ".join(f"{'✓' if r['passed'] else '✗'} ω={r['winning_rate']:.2f}" for r in res)
                       + f" | {res[0]['advantage_prob']:.2f} |\n")

        # design-weighted F1 against the majority of the human coders, clustered by document
        w = 1 / design.prob.values
        docs = design.unit_id.str.rsplit("_u", n=1).str[0].values
        maj1 = [pd.Series([H1[n][i] for n in names]).mode().iloc[0] for i in range(len(ids))]
        maj2 = [frozenset(l for l in L2 if sum(l in H2[n][i] for n in names) * 2 > len(names)) for i in range(len(ids))]
        rep.append("\n## 加权 F1（对人工多数票；按抽样概率加权、按文档聚类 bootstrap）\n\n| 码 | 加权 F1 [95% CI] |\n|---|---|\n")
        for code in L1:
            f, lo, hi = weighted_f1([m == code for m in maj1], [c == code for c in llm1], w, docs)
            rep.append(f"| {code} | {f:.2f} [{lo:.2f}, {hi:.2f}] |\n")
        for code in L2:
            f, lo, hi = weighted_f1([code in m for m in maj2], [code in c for c in llm2], w, docs)
            rep.append(f"| {code} | {f:.2f} [{lo:.2f}, {hi:.2f}] |\n")
    (out / "report.md").write_text("".join(rep), encoding="utf-8")
    print("".join(rep))


if __name__ == "__main__":
    main()
