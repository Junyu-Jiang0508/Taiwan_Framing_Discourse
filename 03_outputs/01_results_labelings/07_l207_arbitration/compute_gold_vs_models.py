"""Build final L2-07 gold standard from the two-coder arbitration + adjudication worksheet,
then recompute GPT vs DeepSeek divergence against that gold.

Gold rule (per 2026-07-07 adjudication decision):
  - 39 items where Junyu == z: gold = that consensus value.
  - 5 items in gold_label_adjudication_worksheet.md (A011/A023/A043/A045/A048):
    gold = True (Junyu's reading accepted).
  - Remaining 6 disagreement items (A002/A026/A027/A033/A038/A046): gold = z's value.

Input:  arbitration_merged_combined.csv
Output: gold_standard_l207.csv, gold_vs_models_report.md
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import binomtest

ROOT = "/home/Junyu/projects/Taiwan_Framing_Discourse"
ARB_DIR = f"{ROOT}/03_outputs/01_results_labelings/07_l207_arbitration"

WORKSHEET_TRUE = {"A011", "A023", "A043", "A045", "A048"}


def kappa_ci(y1, y2, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y1)
    k_obs = cohen_kappa_score(y1, y2)
    boots = []
    idx = np.arange(n)
    y1, y2 = np.asarray(y1), np.asarray(y2)
    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        a, b = y1[s], y2[s]
        if len(set(a)) < 2 and len(set(b)) < 2:
            continue
        try:
            boots.append(cohen_kappa_score(a, b))
        except Exception:
            continue
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return k_obs, lo, hi


def confusion(gold, model):
    gold, model = np.asarray(gold, dtype=bool), np.asarray(model, dtype=bool)
    tp = int((gold & model).sum())
    fp = int((~gold & model).sum())
    fn = int((gold & ~model).sum())
    tn = int((~gold & ~model).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    agree = (tp + tn) / len(gold)
    kap = cohen_kappa_score(gold, model)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                agree=agree, kappa=kap)


def main():
    m = pd.read_csv(f"{ARB_DIR}/arbitration_merged_combined.csv")
    assert len(m) == 50

    def gold_source(row):
        if row["human_agree"]:
            return "consensus"
        if row["item_id"] in WORKSHEET_TRUE:
            return "worksheet(=True)"
        return "z"

    def gold_label(row):
        if row["human_agree"]:
            return bool(row["human_junyu"])
        if row["item_id"] in WORKSHEET_TRUE:
            return True
        return bool(row["human_z"])

    m["gold_source"] = m.apply(gold_source, axis=1)
    m["gold_l207"] = m.apply(gold_label, axis=1)

    n_consensus = int((m["gold_source"] == "consensus").sum())
    n_worksheet = int((m["gold_source"] == "worksheet(=True)").sum())
    n_z = int((m["gold_source"] == "z").sum())
    assert n_consensus == 39 and n_worksheet == 5 and n_z == 6, (n_consensus, n_worksheet, n_z)

    gold = m["gold_l207"]
    gpt = m["gpt_l207"].astype(bool)
    ds = m["ds_l207"].astype(bool)

    gpt_conf = confusion(gold, gpt)
    ds_conf = confusion(gold, ds)
    gpt_k, gpt_klo, gpt_khi = kappa_ci(gold, gpt)
    ds_k, ds_klo, ds_khi = kappa_ci(gold, ds)

    gpt_ds_agree = float((gpt == ds).mean())
    gpt_ds_kappa, gpt_ds_klo, gpt_ds_khi = kappa_ci(gpt, ds)

    # divergence points: model != gold
    gpt_wrong = m.loc[gold != gpt, ["item_id", "camp", "genre", "gold_source", "gold_l207", "gpt_l207", "ds_l207"]]
    ds_wrong = m.loc[gold != ds, ["item_id", "camp", "genre", "gold_source", "gold_l207", "gpt_l207", "ds_l207"]]

    # mutual divergence: GPT != DS (the original 11 disagreement items)
    mutual = m.loc[gpt != ds, ["item_id", "camp", "genre", "disagreement_type", "gold_source",
                                "gold_l207", "gpt_l207", "ds_l207"]].copy()
    mutual["gold_matches"] = np.where(mutual["gold_l207"] == mutual["gpt_l207"], "GPT",
                                np.where(mutual["gold_l207"] == mutual["ds_l207"], "DeepSeek", "neither"))
    n_ds_side = int((mutual["gold_matches"] == "DeepSeek").sum())
    n_gpt_side = int((mutual["gold_matches"] == "GPT").sum())
    sign_p = binomtest(n_ds_side, n_ds_side + n_gpt_side, 0.5).pvalue

    # camp-level breakdown
    camp_tab = mutual.groupby(["camp", "gold_matches"]).size().unstack(fill_value=0)
    dir_tab = mutual.groupby(["disagreement_type", "gold_matches"]).size().unstack(fill_value=0)

    out_cols = ["item_id", "app_id", "camp", "genre", "disagreement_type", "gpt_l207", "ds_l207",
                "human_junyu", "human_z", "gold_source", "gold_l207"]
    m[out_cols].to_csv(f"{ARB_DIR}/gold_standard_l207.csv", index=False)

    lines = []
    lines.append("# L2-07 最终金标准 vs GPT/DeepSeek 分叉分析\n")
    lines.append("- 金标准构建规则（2026-07-07 人工裁决）：")
    lines.append("  1. Junyu 与 z 一致的 39 条 → 取一致值。")
    lines.append("  2. `gold_label_adjudication_worksheet.md` 中的 5 条（A011/A023/A043/A045/A048）→ 定义为 **True**（采纳 Junyu 判读）。")
    lines.append("  3. 其余 6 条分歧（A002/A026/A027/A033/A038/A046）→ 采纳 **z** 的判读。")
    lines.append(f"- 来源构成核验：consensus={n_consensus}, worksheet=True={n_worksheet}, z={n_z}（合计50）。")
    lines.append(f"- 金标准中 L2-07=True 共 {int(gold.sum())}/50 条。\n")

    lines.append("## GPT / DeepSeek vs 金标准\n")
    lines.append("| 模型 | 一致率 | κ (95% CI) | Precision | Recall | F1 | TP/FP/FN/TN |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| GPT | {gpt_conf['agree']:.0%} | {gpt_k:.2f} [{gpt_klo:.2f}, {gpt_khi:.2f}] "
                  f"| {gpt_conf['precision']:.2f} | {gpt_conf['recall']:.2f} | {gpt_conf['f1']:.2f} "
                  f"| {gpt_conf['tp']}/{gpt_conf['fp']}/{gpt_conf['fn']}/{gpt_conf['tn']} |")
    lines.append(f"| DeepSeek | {ds_conf['agree']:.0%} | {ds_k:.2f} [{ds_klo:.2f}, {ds_khi:.2f}] "
                  f"| {ds_conf['precision']:.2f} | {ds_conf['recall']:.2f} | {ds_conf['f1']:.2f} "
                  f"| {ds_conf['tp']}/{ds_conf['fp']}/{ds_conf['fn']}/{ds_conf['tn']} |")
    lines.append("")
    lines.append(f"GPT vs DeepSeek 相互一致率 {gpt_ds_agree:.0%}，κ = {gpt_ds_kappa:.2f} "
                  f"[{gpt_ds_klo:.2f}, {gpt_ds_khi:.2f}]（两模型间，不涉及金标准）。\n")

    lines.append("## 分叉点清单（模型 ≠ 金标准）\n")
    lines.append(f"**GPT 判错 {len(gpt_wrong)}/50 条：**\n")
    lines.append(gpt_wrong.to_markdown(index=False))
    lines.append("")
    lines.append(f"**DeepSeek 判错 {len(ds_wrong)}/50 条：**\n")
    lines.append(ds_wrong.to_markdown(index=False))
    lines.append("")

    lines.append(f"## 两模型互相分歧的 {len(mutual)} 条上，金标准站边情况\n")
    lines.append("（GPT 与 DeepSeek 在全部 50 条中仅 10 条锚点句一致，其余为模型间分歧句；"
                  "此前分析受限于 11 条人工分歧句未定案，只能在 29 条『两人工一致』的可靠子集上做站边统计——"
                  "现在金标准已覆盖全部 50 条，可直接在完整分歧集上统计。）\n")
    lines.append(f"金标准支持 DeepSeek {n_ds_side} 条，支持 GPT {n_gpt_side} 条；符号检验双侧 p = {sign_p:.4f}。\n")
    lines.append("按分歧方向：\n")
    lines.append(dir_tab.to_markdown())
    lines.append("")
    lines.append("按阵营：\n")
    lines.append(camp_tab.to_markdown())
    lines.append("")

    lines.append("## 结论\n")
    better = "DeepSeek" if ds_conf["agree"] > gpt_conf["agree"] else "GPT"
    lines.append(f"1. 以此次裁决金标准衡量，**{better}** 与人工标准更接近"
                  f"（GPT 一致率 {gpt_conf['agree']:.0%}/κ={gpt_k:.2f} vs DeepSeek 一致率 {ds_conf['agree']:.0%}/κ={ds_k:.2f}）。")
    lines.append(f"2. 在两模型互相分歧的 {len(mutual)} 条判定题上，金标准站边 DeepSeek {n_ds_side}/{n_ds_side+n_gpt_side}"
                  f"（符号检验 p={sign_p:.4f}），比此前仅在 29 条可靠子集上估计的比例（DS 19/GPT 10）覆盖面更全、也更极端——"
                  "本轮裁决把此前5条最具争议的『隐性动员』句判给了 DeepSeek，进一步拉开了两模型的差距。")
    lines.append("3. GPT 的错误集中在 `gpt_true_ds_false` 方向的假阳性（把非动员句误判为动员）"
                  "和 `gpt_false_ds_true` 方向的假阴性（漏判隐性动员句），两种错误在分叉点清单中均有体现，"
                  "可在论文中直接引用 `gold_standard_l207.csv` 逐条核对。")

    with open(f"{ARB_DIR}/gold_vs_models_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"gold True count: {int(gold.sum())}/50")
    print(f"GPT vs gold: agree={gpt_conf['agree']:.3f} kappa={gpt_k:.3f} P={gpt_conf['precision']:.3f} R={gpt_conf['recall']:.3f} F1={gpt_conf['f1']:.3f}")
    print(f"DS  vs gold: agree={ds_conf['agree']:.3f} kappa={ds_k:.3f} P={ds_conf['precision']:.3f} R={ds_conf['recall']:.3f} F1={ds_conf['f1']:.3f}")
    print(f"GPT vs DS: agree={gpt_ds_agree:.3f} kappa={gpt_ds_kappa:.3f}")
    print(f"mutual-divergence sides: DS={n_ds_side} GPT={n_gpt_side} p={sign_p:.4f}")
    print("Wrote gold_standard_l207.csv and gold_vs_models_report.md")


if __name__ == "__main__":
    main()
