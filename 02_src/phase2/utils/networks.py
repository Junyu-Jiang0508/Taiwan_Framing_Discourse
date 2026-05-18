"""Build NetworkX graphs from NPMI + bootstrap tables."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd


def l1_distribution_for_nodes(
    labeled: pd.DataFrame,
    nodes: List[str],
    stratum_filter: Dict[str, Any],
) -> Dict[str, str]:
    from .io import tuple_to_l2_set

    sub = labeled.copy()
    for k, v in stratum_filter.items():
        if k in sub.columns:
            sub = sub[sub[k] == v]
    out = {}
    for l2 in nodes:
        mask = sub["l2_set"].map(lambda x: l2 in tuple_to_l2_set(x))
        rows = sub.loc[mask]
        if len(rows) == 0:
            out[l2] = json.dumps({})
            continue
        ctr = Counter(rows["L1"].fillna("").astype(str))
        ctr.pop("", None)
        out[l2] = json.dumps(dict(ctr))
    return out


def edge_is_significant(row: pd.Series, edge_selection: str) -> bool:
    ci = bool(row.get("ci_excludes_zero", False))
    fdr = bool(row.get("fdr_significant", False))
    if edge_selection == "fdr":
        return fdr
    if edge_selection == "both":
        return ci and fdr
    return ci


def build_graphs(
    npmi_point: pd.DataFrame,
    npmi_boot: pd.DataFrame,
    labeled: pd.DataFrame,
    stratum_filter: Dict[str, Any],
    stratum_key: str,
    out_dir: Path,
    edge_selection: str = "ci_excludes_zero",
) -> Tuple[Path, Path, Dict[str, int]]:
    boot_cols = [
        "l2_a", "l2_b", "npmi_median", "npmi_lower", "npmi_upper",
        "ci_excludes_zero", "p_value", "q_value", "fdr_significant",
    ]
    boot_cols = [c for c in boot_cols if c in npmi_boot.columns]
    merged = npmi_point.merge(npmi_boot[boot_cols], on=["l2_a", "l2_b"], how="inner")
    sig_mask = merged.apply(lambda r: edge_is_significant(r, edge_selection), axis=1)
    sig = merged[sig_mask]
    n_ci = int((merged["ci_excludes_zero"] == True).sum()) if "ci_excludes_zero" in merged.columns else 0  # noqa: E712
    n_fdr = int((merged["fdr_significant"] == True).sum()) if "fdr_significant" in merged.columns else 0  # noqa: E712
    stats = {"n_sig_ci": n_ci, "n_sig_fdr": n_fdr, "n_sig_selected": len(sig)}
    nodes = sorted(set(sig["l2_a"]) | set(sig["l2_b"]))
    l1_dist = l1_distribution_for_nodes(labeled, nodes, stratum_filter)

    pos_path = out_dir / f"{stratum_key}_positive.graphml"
    signed_path = out_dir / f"{stratum_key}_signed.graphml"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, edge_df, positive_only in [
        (pos_path, sig[sig["npmi_median"] > 0], True),
        (signed_path, sig, False),
    ]:
        G = nx.Graph()
        marg_lookup = {}
        for _, r in npmi_point.iterrows():
            marg_lookup[r["l2_a"]] = r["p_a"]
            marg_lookup[r["l2_b"]] = r["p_b"]
        for n in nodes:
            G.add_node(
                n,
                l2=n,
                marginal_p=float(marg_lookup.get(n, 0)),
                l1_distribution=l1_dist.get(n, "{}"),
            )
        for _, r in edge_df.iterrows():
            if positive_only and r["npmi_median"] <= 0:
                continue
            sign = "+" if r["npmi_median"] > 0 else "-"
            G.add_edge(
                r["l2_a"],
                r["l2_b"],
                npmi_median=float(r["npmi_median"]),
                npmi_lower=float(r["npmi_lower"]),
                npmi_upper=float(r["npmi_upper"]),
                p_ab=float(r["p_ab"]),
                n_windows_supporting=int(r.get("n_windows", 0)),
                sign=sign,
            )
        nx.write_graphml(G, path)

    return pos_path, signed_path, stats
