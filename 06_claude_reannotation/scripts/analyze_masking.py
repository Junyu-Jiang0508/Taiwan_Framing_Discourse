"""P4 party-cue test (Vallejo Vera & Driggers 2025): do labels change when candidate / party / camp names are masked?

Compares, on the same 300 sentences (100 per camp):
  masked run  vs full run   -> change due to masking + run-to-run noise
  control run vs full run   -> run-to-run noise only (same unmasked input)
The excess of the first over the second is the masking effect. Reported per code, overall and by camp,
plus the direction (does masking raise or lower the prevalence of a code, and differently by camp?).

Usage: python analyze_masking.py --out ../eval/masking
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ANALYSIS_FRAME, L1, L2, RUNS, read_jsonl


def load(run, layer):
    return {r["item_id"]: r["parsed"] for r in read_jsonl(RUNS / run / f"{layer}.jsonl") if r.get("ok")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    camp = pd.read_csv(ANALYSIS_FRAME, usecols=["unit_id", "camp"]).drop_duplicates("unit_id").set_index("unit_id").camp
    F1, F2 = load("full_low", "L1"), load("full_low", "L2")
    M1, M2 = load("p4_mask", "L1"), load("p4_mask", "L2")
    C1, C2 = load("p4_mask_ctrl", "L1"), load("p4_mask_ctrl", "L2")
    ids = sorted(set(M1) & set(M2) & set(C1) & set(C2))
    rows = []
    for u in ids:
        r = {"unit_id": u, "camp": camp[u], "L1_full": F1[u]["L1"], "L1_mask": M1[u]["L1"], "L1_ctrl": C1[u]["L1"]}
        for l in L2:
            r[f"{l}_full"], r[f"{l}_mask"], r[f"{l}_ctrl"] = F2[u][l], M2[u][l], C2[u][l]
        rows.append(r)
    t = pd.DataFrame(rows)
    rep = [f"# P4 政党线索检验（{len(t)} 句，每阵营约 100 句）\n\n"
           "「遮蔽后改变」= 遮蔽版 vs 全量原标签；「重跑改变」= 未遮蔽重跑 vs 全量原标签（纯随机噪声基线）。"
           "超出部分才归因于遮蔽。\n\n| 指标 | 遮蔽后改变 | 重跑改变 | 超出 |\n|---|---|---|---|\n"]
    m = (t.L1_mask != t.L1_full).mean(); c = (t.L1_ctrl != t.L1_full).mean()
    rep.append(f"| L1 主框架 | {m:.1%} | {c:.1%} | {m - c:+.1%} |\n")
    for l in L2:
        m = (t[f"{l}_mask"] != t[f"{l}_full"]).mean(); c = (t[f"{l}_ctrl"] != t[f"{l}_full"]).mean()
        rep.append(f"| {l} | {m:.1%} | {c:.1%} | {m - c:+.1%} |\n")
    rep.append("\n## 方向：遮蔽前后的出现率（按阵营；百分点差 = 遮蔽 − 未遮蔽重跑）\n\n| 码 | DPP | KMT | TPP |\n|---|---|---|---|\n")
    for l in L1:
        cells = [f"{100 * ((g.L1_mask == l).mean() - (g.L1_ctrl == l).mean()):+.1f}" for _, g in t.groupby("camp")]
        rep.append(f"| {l} | " + " | ".join(cells) + " |\n")
    for l in L2:
        cells = [f"{100 * (g[f'{l}_mask'].mean() - g[f'{l}_ctrl'].mean()):+.1f}" for _, g in t.groupby("camp")]
        rep.append(f"| {l} | " + " | ".join(cells) + " |\n")
    rep.append("\n读法：若某码在三个阵营的差值方向相反或大小悬殊，说明 Claude 用政党线索调整了判断，跨阵营比较需要警惕；"
               "差值都在重跑噪声范围内（±2–3pp），则政党线索的影响可以忽略。每阵营只有约 100 句，±5pp 以内的差异不宜解读。\n")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t.to_csv(Path(a.out) / "masking_merged.csv", index=False)
    (Path(a.out) / "report.md").write_text("".join(rep), encoding="utf-8")
    print("".join(rep))


if __name__ == "__main__":
    main()
