#!/usr/bin/env python3
"""Before/after comparison for the p2_00 scrape-residue filter.

Every Phase 2 estimate is computed on the corpus that p2_00 freezes, so adding
the filter moves all of them. This probe quantifies by how much: it reads a
snapshot of the pre-filter artifacts and the current post-filter artifacts and
reports the deltas that matter for the existing claims — edge NPMI, the FDR
edge set, Leiden partitions, cross-camp QAP gamma, the camp-contrast
permutation p-values, and the six cohesion-block edges.

Usage:
    python3 probes/boilerplate_filter_delta.py BASELINE_DIR CURRENT_DIR [--label GPT]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score

#: The cohesion block from cohesion_next_steps.md, plus its two "false" edges.
COH_PAIRS = [
    ("L2-04", "L2-06"), ("L2-04", "L2-07"), ("L2-04", "L2-08"),
    ("L2-06", "L2-07"), ("L2-06", "L2-08"), ("L2-07", "L2-08"),
]
EDGE_KEY = ["camp", "l2_a", "l2_b"]


def _read(root: Path, rel: str) -> Optional[pd.DataFrame]:
    p = root / rel
    return pd.read_parquet(p) if p.is_file() else None


def _fmt(x, nd: int = 3) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def corpus_section(base: Path, cur: Path) -> list[str]:
    out = ["## 1. 语料规模", ""]
    b = _read(base, "labeled_units.parquet")
    c = _read(cur, "labeled_units.parquet")
    if b is None or c is None:
        return out + ["labeled_units.parquet 缺失。", ""]
    out += [
        f"句数 {len(b)} → {len(c)}（{len(c) - len(b):+d}，{100 * (len(c) - len(b)) / len(b):+.2f}%）",
        "",
        "| camp | genre | before | after | Δ |",
        "| --- | --- | --- | --- | --- |",
    ]
    tb = b.groupby(["camp", "genre"]).size()
    tc = c.groupby(["camp", "genre"]).size()
    for k in sorted(set(tb.index) | set(tc.index)):
        nb, nc = int(tb.get(k, 0)), int(tc.get(k, 0))
        out.append(f"| {k[0]} | {k[1]} | {nb} | {nc} | {nc - nb:+d} |")
    wb = _read(base, "windows/windows_n3.parquet")
    wc = _read(cur, "windows/windows_n3.parquet")
    if wb is not None and wc is not None:
        out += [
            "",
            f"n=3 窗口 {len(wb)} → {len(wc)}（{len(wc) - len(wb):+d}）；"
            f"空 L2 窗口率 {100 * wb['is_empty_l2'].mean():.2f}% → "
            f"{100 * wc['is_empty_l2'].mean():.2f}%",
        ]
    if "boiler_tier" in c.columns:
        keep = c["boiler_tier"].value_counts().to_dict()
        out += ["", f"保留行的 boiler_tier 分布：{keep}（prefix = 真句 + 抓取前缀，按设计保留）"]
    return out + [""]


def edges_section(base: Path, cur: Path, scheme: str = "camp") -> list[str]:
    out = [f"## 2. 边估计（scheme={scheme}, n=3）", ""]
    pb = _read(base, f"npmi_by_{scheme}/npmi_point.parquet")
    pc = _read(cur, f"npmi_by_{scheme}/npmi_point.parquet")
    bb = _read(base, f"npmi_by_{scheme}/npmi_bootstrap.parquet")
    bc = _read(cur, f"npmi_by_{scheme}/npmi_bootstrap.parquet")
    if pb is None or pc is None:
        return out + ["npmi_point.parquet 缺失。", ""]

    m = pb.merge(pc, on=EDGE_KEY, suffixes=("_before", "_after"))
    out += [
        "点估计 NPMI 的一致性（每格 = 一个 camp × 一对码）：",
        "",
        "| camp | n 对 | Pearson | Spearman | 平均\\|Δ\\| | 最大\\|Δ\\| | 最大变动边 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for camp, g in list(m.groupby("camp")) + [("ALL", m)]:
        d = (g["npmi_after"] - g["npmi_before"]).abs()
        ok = g["npmi_before"].notna() & g["npmi_after"].notna()
        r = pearsonr(g.loc[ok, "npmi_before"], g.loc[ok, "npmi_after"])[0] if ok.sum() > 2 else None
        rho = spearmanr(g.loc[ok, "npmi_before"], g.loc[ok, "npmi_after"])[0] if ok.sum() > 2 else None
        top = g.loc[d.idxmax()] if len(d) and d.notna().any() else None
        top_s = (
            f"{top['l2_a'][-2:]}-{top['l2_b'][-2:]} ({top['npmi_before']:.3f}→{top['npmi_after']:.3f})"
            if top is not None else "—"
        )
        out.append(
            f"| {camp} | {int(ok.sum())} | {_fmt(r)} | {_fmt(rho)} | "
            f"{_fmt(d.mean())} | {_fmt(d.max())} | {top_s} |"
        )

    if bb is not None and bc is not None:
        out += ["", "FDR 显著边集合的变化：", "",
                "| camp | before | after | 交集 | Jaccard | 新增 | 消失 |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        for camp in sorted(set(bb["camp"]) | set(bc["camp"])):
            sb = {(r.l2_a, r.l2_b) for r in bb[(bb.camp == camp) & bb.fdr_significant].itertuples()}
            sc = {(r.l2_a, r.l2_b) for r in bc[(bc.camp == camp) & bc.fdr_significant].itertuples()}
            inter, union = sb & sc, sb | sc
            gained = ",".join(f"{a[-2:]}-{b[-2:]}" for a, b in sorted(sc - sb)) or "—"
            lost = ",".join(f"{a[-2:]}-{b[-2:]}" for a, b in sorted(sb - sc)) or "—"
            j = len(inter) / len(union) if union else float("nan")
            out.append(
                f"| {camp} | {len(sb)} | {len(sc)} | {len(inter)} | {_fmt(j)} | {gained} | {lost} |"
            )
    return out + [""]


def cohesion_section(base: Path, cur: Path) -> list[str]:
    out = ["## 3. 凝聚簇六条边（cohesion_next_steps.md §2.4 的对象）", ""]
    bb = _read(base, "npmi_by_camp/npmi_bootstrap.parquet")
    bc = _read(cur, "npmi_by_camp/npmi_bootstrap.parquet")
    if bb is None or bc is None:
        return out + ["npmi_bootstrap.parquet 缺失。", ""]
    sel = pd.DataFrame(COH_PAIRS, columns=["l2_a", "l2_b"])
    b = bb.merge(sel, on=["l2_a", "l2_b"])
    c = bc.merge(sel, on=["l2_a", "l2_b"])
    m = b.merge(c, on=EDGE_KEY, suffixes=("_b", "_a"))
    out += ["| camp | edge | NPMI before [CI] | NPMI after [CI] | Δ | FDR before→after |",
            "| --- | --- | --- | --- | --- | --- |"]
    for r in m.sort_values(["camp", "l2_a", "l2_b"]).itertuples():
        d = r.npmi_median_a - r.npmi_median_b
        out.append(
            f"| {r.camp} | {r.l2_a[-2:]}-{r.l2_b[-2:]} "
            f"| {_fmt(r.npmi_median_b)} [{_fmt(r.npmi_lower_b)}, {_fmt(r.npmi_upper_b)}] "
            f"| {_fmt(r.npmi_median_a)} [{_fmt(r.npmi_lower_a)}, {_fmt(r.npmi_upper_a)}] "
            f"| {d:+.3f} | {bool(r.fdr_significant_b)}→{bool(r.fdr_significant_a)} |"
        )
    return out + [""]


def partition_section(base: Path, cur: Path) -> list[str]:
    out = ["## 4. Leiden 共识划分", ""]
    b = _read(base, "partitions_by_camp/consensus_partition.parquet")
    c = _read(cur, "partitions_by_camp/consensus_partition.parquet")
    if b is None or c is None:
        return out + ["consensus_partition.parquet 缺失。", ""]
    out += ["| camp | ARI(before, after) | before | after |", "| --- | --- | --- | --- |"]
    for camp in sorted(set(b["camp"].dropna()) | set(c["camp"].dropna())):
        gb = b[b.camp == camp].set_index("node")["community_id"]
        gc = c[c.camp == camp].set_index("node")["community_id"]
        nodes = sorted(set(gb.index) & set(gc.index))
        ari = adjusted_rand_score(gb.loc[nodes], gc.loc[nodes]) if len(nodes) > 1 else float("nan")
        fmt = lambda s: "/".join(  # noqa: E731
            "+".join(n[-2:] for n in sorted(s[s == k].index)) for k in sorted(s.unique())
        )
        out.append(f"| {camp} | {_fmt(ari)} | {fmt(gb)} | {fmt(gc)} |")
    return out + [""]


def qap_section(base: Path, cur: Path) -> list[str]:
    out = ["## 5. 跨阵营结构（p2_06 QAP / p2_12 阵营置换）", ""]
    qb, qc = _read(base, "cross_camp/qap_results.parquet"), _read(cur, "cross_camp/qap_results.parquet")
    if qb is not None and qc is not None:
        m = qb.merge(qc, on=["camp_a", "camp_b"], suffixes=("_b", "_a"))
        out += ["| 阵营对 | γ before | γ after | Δ | p before | p after |",
                "| --- | --- | --- | --- | --- | --- |"]
        for r in m.itertuples():
            out.append(
                f"| {r.camp_a}–{r.camp_b} | {_fmt(r.hubert_gamma_b)} | {_fmt(r.hubert_gamma_a)} "
                f"| {r.hubert_gamma_a - r.hubert_gamma_b:+.3f} "
                f"| {_fmt(r.p_value_b, 4)} | {_fmt(r.p_value_a, 4)} |"
            )
        out.append("")
    pb, pc = (_read(base, "cross_camp/permutation_summary.parquet"),
              _read(cur, "cross_camp/permutation_summary.parquet"))
    if pb is not None and pc is not None:
        m = pb.merge(pc, on="test", suffixes=("_b", "_a"))
        out += ["p2_12 阵营置换（observed γ vs 零模型）：", "",
                "| test | obs before | obs after | null_mean before | null_mean after | p before | p after |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in m.itertuples():
            out.append(
                f"| {r.test} | {_fmt(r.observed_gamma_b)} | {_fmt(r.observed_gamma_a)} "
                f"| {_fmt(r.null_mean_b)} | {_fmt(r.null_mean_a)} "
                f"| {_fmt(r.p_value_b, 4)} | {_fmt(r.p_value_a, 4)} |"
            )
    return out + [""]


def subnet_section(base: Path, cur: Path) -> list[str]:
    out = ["## 6. 差异化子网（p2_14）", ""]
    b = _read(base, "subnetwork/differentiating_subnet_stats.parquet")
    c = _read(cur, "subnetwork/differentiating_subnet_stats.parquet")
    if b is None or c is None:
        return out + ["differentiating_subnet_stats.parquet 缺失。", ""]
    # indicator column renamed: itertuples mangles leading-underscore names.
    m = b.merge(c, on=EDGE_KEY, suffixes=("_b", "_a"), how="outer", indicator="side")
    # Only the moved rows are informative; DS keeps all 28 pairs per camp.
    moved = (
        (m["side"] != "both")
        | (m["rank_b"] != m["rank_a"])
        | ((m["npmi_weight_a"] - m["npmi_weight_b"]).abs() > 0.01)
    )
    out += [
        f"共 {len(m)} 条边，其中 {int(moved.sum())} 条发生 rank 变化或 |Δw|>0.01；未变动的省略。",
        "",
        "| camp | edge | w before | w after | rank before→after | 状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in m[moved].sort_values(["camp", "l2_a", "l2_b"]).itertuples():
        state = {"both": "两侧都有", "left_only": "仅 before", "right_only": "仅 after"}[r.side]
        out.append(
            f"| {r.camp} | {str(r.l2_a)[-2:]}-{str(r.l2_b)[-2:]} | {_fmt(r.npmi_weight_b)} "
            f"| {_fmt(r.npmi_weight_a)} | {_fmt(r.rank_b, 0)}→{_fmt(r.rank_a, 0)} | {state} |"
        )
    return out + [""]


def robustness_section(base: Path, cur: Path) -> list[str]:
    out = ["## 7. 窗口稳健性（p2_10）与时间稳定性（p2_11）", ""]
    b, c = _read(base, "robustness_summary.parquet"), _read(cur, "robustness_summary.parquet")
    if b is not None and c is not None:
        key = ["scheme", "ref_window", "alt_window", "camp", "genre"]
        # outer: window sizes added to robustness_ns exist only on the after side.
        m = b.merge(c, on=key, suffixes=("_b", "_a"), how="outer")
        agg = m.groupby("alt_window").agg(
            rho_before=("spearman_rho_b", "mean"), rho_after=("spearman_rho_a", "mean"),
            jac_before=("sig_edge_jaccard_b", "mean"), jac_after=("sig_edge_jaccard_a", "mean"),
            n=("camp", "size"))
        out += ["| alt window | 平均 ρ before | after | 平均显著边 Jaccard before | after | n 层 |",
                "| --- | --- | --- | --- | --- | --- |"]
        for w, r in agg.iterrows():
            out.append(f"| n={w} | {_fmt(r.rho_before)} | {_fmt(r.rho_after)} | "
                       f"{_fmt(r.jac_before)} | {_fmt(r.jac_after)} | {int(r.n)} |")
        out.append("")
    tb, tc = _read(base, "temporal_edge_stability.parquet"), _read(cur, "temporal_edge_stability.parquet")
    if tb is not None and tc is not None:
        out += [
            f"p2_11 桶间比较对数 {len(tb)} → {len(tc)}；CI 不重叠比例 "
            f"{100 * tb['ci_disjoint'].mean():.1f}% → {100 * tc['ci_disjoint'].mean():.1f}%",
            "",
        ]
    return out


def build_report(base: Path, cur: Path, label: str) -> str:
    lines = [
        f"# 抓取残留过滤前后对比（{label}）",
        "",
        f"- before: `{base}`（p2_00 无过滤）",
        f"- after:  `{cur}`（p2_00 `boilerplate_policy: drop_residue`）",
        "",
    ]
    for fn in (corpus_section, edges_section, cohesion_section,
               partition_section, qap_section, subnet_section, robustness_section):
        lines += fn(base, cur)
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("current", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = build_report(args.baseline, args.current, args.label or args.current.name)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"→ {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
