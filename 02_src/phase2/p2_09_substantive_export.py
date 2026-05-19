#!/usr/bin/env python3
"""p2_09 — Aggregate substantive Phase 2 results for interpretation."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.io import read_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402

INTERPRETIVE_NOTES = [
    "Shared infrastructure cluster ({L2-06, L2-07}) is isomorphic across all three camps (Jaccard=1.0, stable).",
    "Differentiation concentrates in the {L2-02, L2-04, L2-05, L2-08} sub-network.",
    "KMT and TPP show near-isomorphic articulation grammar (γ≈0.949); DPP is the outlier.",
    "Community partition is supplementary to edge-level analysis given 8-node saturation.",
]


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_empty_\n"
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows)_\n")
    return "\n".join(lines) + "\n"


def _partition_block(cons: pd.DataFrame, stab: pd.DataFrame, scheme: str) -> str:
    lines = [f"### Scheme: `{scheme}`\n"]
    if cons.empty:
        lines.append("_No consensus partition._\n")
        return "\n".join(lines)
    for sk, sub in cons.groupby("stratum_key", sort=True):
        lines.append(f"#### Stratum `{sk}`\n")
        comms: Dict[int, List[str]] = {}
        for _, r in sub.iterrows():
            comms.setdefault(int(r["community_id"]), []).append(str(r["node"]))
        for cid, nodes in sorted(comms.items()):
            stable = ""
            if not stab.empty:
                match = stab[(stab["stratum_key"] == sk) & (stab["community_id"] == cid)]
                if len(match):
                    st = match.iloc[0]
                    stable = f" (stability={st['stability']:.3f}, stable={st['stable']})"
            lines.append(f"- Community {cid}{stable}: {', '.join(sorted(nodes))}")
        lines.append("")
    return "\n".join(lines)


def _graph_diagnostics(cfg: Phase2Config, scheme: str) -> pd.DataFrame:
    rows = []
    net_dir = cfg.scheme_dir("networks", scheme)
    diag_path = cfg.scheme_dir("npmi", scheme) / "stratum_diagnostics.parquet"
    diag = read_parquet(diag_path) if diag_path.is_file() else pd.DataFrame()
    for gp in sorted(net_dir.glob("*_positive.graphml")):
        key = gp.name.replace("_positive.graphml", "")
        G = nx.read_graphml(gp)
        n, m = G.number_of_nodes(), G.number_of_edges()
        max_e = n * (n - 1) // 2 if n >= 2 else 0
        row: Dict[str, Any] = {
            "scheme": scheme,
            "stratum": key,
            "n_nodes": n,
            "n_edges": m,
            "max_edges": max_e,
            "density": round(m / max_e, 4) if max_e else 0.0,
        }
        if not diag.empty:
            if scheme == "camp":
                dsub = diag[diag["camp"] == key]
            elif scheme == "camp_genre":
                parts = key.split("_", 1)
                if len(parts) == 2:
                    dsub = diag[(diag["camp"] == parts[0]) & (diag["genre"] == parts[1])]
                else:
                    dsub = pd.DataFrame()
            elif scheme == "camp_time":
                parts = key.split("_", 1)
                if len(parts) == 2:
                    dsub = diag[(diag["camp"] == parts[0]) & (diag["time_bucket"] == parts[1])]
                else:
                    dsub = pd.DataFrame()
            else:
                dsub = pd.DataFrame()
            if len(dsub):
                r0 = dsub.iloc[0]
                row["n_windows"] = int(r0["n_windows"])
                row["empty_l2_rate"] = round(r0["n_empty_l2_windows"] / r0["n_windows"], 4)
                row["low_n_warning"] = bool(r0.get("low_n_warning", False))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_differentiating_long_table(
    diff_stats: pd.DataFrame,
    edge_diff: pd.DataFrame,
    boot: pd.DataFrame,
) -> pd.DataFrame:
    if diff_stats.empty:
        return pd.DataFrame()
    long_df = diff_stats[["camp", "l2_a", "l2_b", "npmi_weight", "rank"]].rename(
        columns={"npmi_weight": "npmi_median"}
    ).copy()
    long_df["npmi_lower"] = float("nan")
    long_df["npmi_upper"] = float("nan")
    long_df["ci_disjoint"] = False

    if not boot.empty:
        if "scheme" in boot.columns:
            boot = boot[boot["scheme"] == "camp"]
        boot_idx = boot.set_index(["camp", "l2_a", "l2_b"])
        for i, row in long_df.iterrows():
            key = (row["camp"], row["l2_a"], row["l2_b"])
            if key in boot_idx.index:
                b = boot_idx.loc[key]
                if isinstance(b, pd.DataFrame):
                    b = b.iloc[0]
                long_df.at[i, "npmi_lower"] = float(b["npmi_lower"])
                long_df.at[i, "npmi_upper"] = float(b["npmi_upper"])

    if not edge_diff.empty:
        disjoint_keys = set()
        for _, r in edge_diff[edge_diff["ci_disjoint"] == True].iterrows():  # noqa: E712
            disjoint_keys.add((r["camp_a"], r["l2_a"], r["l2_b"]))
            disjoint_keys.add((r["camp_b"], r["l2_a"], r["l2_b"]))
        for i, row in long_df.iterrows():
            if (row["camp"], row["l2_a"], row["l2_b"]) in disjoint_keys:
                long_df.at[i, "ci_disjoint"] = True
    return long_df


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    manifest_path = art / "manifests" / "p2_09.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "edge_selection": cfg.edge_selection,
        "export_version": "subclaim1_v2",
    }
    if should_skip(manifest_path, expected, force):
        print("p2_09: skip (manifest match)")
        return

    out_md = art / "substantive_results.md"
    long_rows: List[Dict[str, Any]] = []
    md_parts = [
        "# Phase 2 substantive results\n",
        f"_Edge selection: `{cfg.edge_selection}`; FDR α={cfg.fdr_alpha}_\n",
        "## Interpretive notes\n",
    ]
    for note in INTERPRETIVE_NOTES:
        md_parts.append(f"- {note}\n")
    md_parts.append("\n")

    cross_dir = art / "cross_camp"
    subnet_dir = art / "subnetwork"
    comm = read_parquet(cross_dir / "community_table.parquet") if (cross_dir / "community_table.parquet").is_file() else pd.DataFrame()
    jacc = read_parquet(cross_dir / "jaccard_all_pairs.parquet") if (cross_dir / "jaccard_all_pairs.parquet").is_file() else pd.DataFrame()
    qap = read_parquet(cross_dir / "qap_results.parquet") if (cross_dir / "qap_results.parquet").is_file() else pd.DataFrame()
    perm_summary = read_parquet(cross_dir / "permutation_summary.parquet") if (cross_dir / "permutation_summary.parquet").is_file() else pd.DataFrame()
    edge_diff = read_parquet(cross_dir / "edge_diff_pairwise.parquet") if (cross_dir / "edge_diff_pairwise.parquet").is_file() else pd.DataFrame()
    diff_by_node = read_parquet(cross_dir / "diff_by_l2_node.parquet") if (cross_dir / "diff_by_l2_node.parquet").is_file() else pd.DataFrame()
    node_audit = read_parquet(subnet_dir / "node_selection_audit.parquet") if (subnet_dir / "node_selection_audit.parquet").is_file() else pd.DataFrame()
    diff_stats = read_parquet(subnet_dir / "differentiating_subnet_stats.parquet") if (subnet_dir / "differentiating_subnet_stats.parquet").is_file() else pd.DataFrame()
    shared_stats = read_parquet(subnet_dir / "shared_subnet_stats.parquet") if (subnet_dir / "shared_subnet_stats.parquet").is_file() else pd.DataFrame()
    rank_cmp = read_parquet(subnet_dir / "rank_correlation_comparison.parquet") if (subnet_dir / "rank_correlation_comparison.parquet").is_file() else pd.DataFrame()
    boot_path = cfg.scheme_dir("npmi", "camp") / "npmi_bootstrap.parquet"
    boot = read_parquet(boot_path) if boot_path.is_file() else pd.DataFrame()

    md_parts.append("## Shared infrastructure\n\n")
    if not comm.empty and "category" in comm.columns:
        shared_comm = comm[comm["category"] == "shared"]
        md_parts.append(_md_table(shared_comm))
    else:
        md_parts.append("_No shared-infrastructure classification available._\n")

    md_parts.append("## Differentiating edges\n\n")
    md_parts.append("_Conservative test: non-overlapping 95% bootstrap CIs._\n\n")
    if not edge_diff.empty:
        disjoint = edge_diff[edge_diff["ci_disjoint"] == True]  # noqa: E712
        md_parts.append(f"**{len(disjoint)}** of {len(edge_diff)} camp-pair edge tests have disjoint CIs.\n\n")
        md_parts.append("### CI-disjoint edges\n\n")
        md_parts.append(_md_table(disjoint))
    if not diff_by_node.empty:
        md_parts.append("### Differentiating edges by L2 node\n\n")
        md_parts.append(_md_table(diff_by_node.sort_values("n_differentiating_edges", ascending=False)))

    md_parts.append("## Permutation test\n\n")
    md_parts.append("_H₀: camp labels independent of co-articulation structure (doc-level shuffle)._\n\n")
    md_parts.append(_md_table(perm_summary))

    md_parts.append("## Sub-network analysis\n\n")
    if not node_audit.empty:
        md_parts.append("### Node selection audit\n\n")
        md_parts.append(_md_table(node_audit))
    md_parts.append("### Rank correlation comparison (full vs sub-networks)\n\n")
    md_parts.append(_md_table(rank_cmp))
    if not shared_stats.empty:
        md_parts.append("### Shared infrastructure sub-network edge stats\n\n")
        md_parts.append(_md_table(shared_stats))

    long_table = _build_differentiating_long_table(diff_stats, edge_diff, boot)
    long_table_path = art / "differentiating_subnet_long.parquet"
    if not long_table.empty:
        long_table.to_parquet(long_table_path, index=False)
        md_parts.append("## Differentiating sub-network (figure-ready)\n\n")
        md_parts.append(_md_table(long_table))

    md_parts.append("## Cross-camp community matching (top-k greedy)\n\n")
    md_parts.append(_md_table(comm))
    md_parts.append("## Cross-camp Jaccard (all community pairs)\n\n")
    md_parts.append(_md_table(jacc, max_rows=30))
    md_parts.append("## QAP (Hubert γ)\n\n")
    md_parts.append(_md_table(qap))
    md_parts.append(
        "\n_QAP γ measures against-random distinctness; between-camp distinctness "
        "is tested by the camp permutation test above._\n\n"
    )

    for scheme in ["camp", "camp_genre", "camp_time"]:
        if scheme not in [s.name for s in cfg.schemes]:
            continue
        part_dir = cfg.scheme_dir("partitions", scheme)
        cons_path = part_dir / "consensus_partition.parquet"
        stab_path = part_dir / "community_stability.parquet"
        cons = read_parquet(cons_path) if cons_path.is_file() else pd.DataFrame()
        stab = read_parquet(stab_path) if stab_path.is_file() else pd.DataFrame()
        md_parts.append(_partition_block(cons, stab, scheme))
        gdiag = _graph_diagnostics(cfg, scheme)
        md_parts.append(f"### Graph diagnostics (`{scheme}`)\n\n")
        md_parts.append(_md_table(gdiag))
        if not gdiag.empty and "low_n_warning" in gdiag.columns:
            warned = gdiag[gdiag["low_n_warning"] == True]  # noqa: E712
            if len(warned):
                md_parts.append("**Low-n strata (excluded from networks when n_windows < min_stratum_windows):**\n")
                for _, r in warned.iterrows():
                    md_parts.append(f"- `{r['stratum']}`: n_windows={r.get('n_windows', '?')}\n")
        for _, r in gdiag.iterrows():
            long_rows.append({"section": "graph_diagnostics", **r.to_dict()})

    for section, df in [
        ("community_table", comm),
        ("qap", qap),
        ("permutation_summary", perm_summary),
        ("edge_diff_pairwise", edge_diff),
        ("diff_by_l2_node", diff_by_node),
        ("node_selection_audit", node_audit),
        ("rank_correlation_comparison", rank_cmp),
    ]:
        if not df.empty:
            for _, r in df.iterrows():
                long_rows.append({"section": section, **r.to_dict()})
    if not long_table.empty:
        for _, r in long_table.iterrows():
            long_rows.append({"section": "differentiating_subnet_long", **r.to_dict()})

    out_md.write_text("".join(md_parts), encoding="utf-8")
    pd.DataFrame(long_rows).to_parquet(art / "substantive_results.parquet", index=False)

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([out_md])})
    write_summary(
        art,
        "p2_09",
        params={"edge_selection": cfg.edge_selection, "export_version": "subclaim1_v2"},
        outputs=[str(out_md), str(art / "substantive_results.parquet")],
        stats={"n_long_rows": len(long_rows)},
        notes=INTERPRETIVE_NOTES,
        elapsed_sec=time.perf_counter() - t0,
    )
    print("p2_09 done")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
