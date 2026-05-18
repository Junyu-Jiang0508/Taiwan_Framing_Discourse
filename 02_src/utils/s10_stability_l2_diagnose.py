#!/usr/bin/env python3
"""Post-hoc L2 stability diagnostics from a stability triplicate CSV.

Reads ``stability_triplicate_sample.csv`` (three L2 pipe-joined columns per row) and
writes four tables for manuscript / sanity checks:

1. Label stability profile (per L2 tag, conditional on ≥1 appearance among non-empty rows)
2. Row decomposition (consensus vs swing; cohort column; Jaccard NaN when consensus empty) + ``table2_metadata.json``
3. Label co-fluctuation matrix (joint swing vs union swing denominator)
4. Empty-L2-row L1 distribution (rows with all three L2 runs empty)

Run from repo root or ``02_src``::

    python 02_src/s10_stability_l2_diagnose.py \\
        --triplicate-csv 03_outputs/.../stability_triplicate_sample.csv \\
        --out-dir 03_outputs/.../stability_run/diagnostics_l2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from s10_corpus_label_audit import jaccard, parse_l2_cell  # noqa: E402
except ImportError:  # pragma: no cover — running outside 02_src on PYTHONPATH

    def parse_l2_cell(s: Any) -> List[str]:
        if pd.isna(s) or str(s).strip() == "":
            return []
        return [x.strip() for x in str(s).split("|") if x.strip()]

    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        u = len(a | b)
        if u == 0:
            return 1.0
        return len(a & b) / u


L2_CANONICAL: Tuple[str, ...] = tuple(f"L2-0{i}" for i in range(1, 9))


def _json_safe(o: Any) -> Any:
    """JSON-serializable tree (NaN/Inf → null for strict JSON)."""
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating,)):
        x = float(o)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def _l2_set(cell) -> Set[str]:
    return set(parse_l2_cell(cell))


def _row_all_l2_empty(s0: Set[str], s1: Set[str], s2: Set[str]) -> bool:
    return not s0 and not s1 and not s2


def _min_triplet_jaccard(s0: Set[str], s1: Set[str], s2: Set[str]) -> float:
    return float(
        min(
            jaccard(s0, s1),
            jaccard(s0, s2),
            jaccard(s1, s2),
        )
    )


def _consensus_swing(
    s0: Set[str], s1: Set[str], s2: Set[str]
) -> Tuple[Set[str], Set[str]]:
    counts: Counter[str] = Counter()
    for s in (s0, s1, s2):
        for lab in s:
            counts[lab] += 1
    consensus = {lab for lab, c in counts.items() if c >= 2}
    swing = {lab for lab, c in counts.items() if c == 1}
    return consensus, swing


def table1_label_stability(
    df_ne: pd.DataFrame,
    col_r1: str,
    col_r2: str,
    col_r3: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    rows_out: List[Dict[str, object]] = []
    for lab in labels:
        n_present = 0
        c3 = c2 = c1 = 0
        for _, row in df_ne.iterrows():
            s0, s1, s2 = _l2_set(row[col_r1]), _l2_set(row[col_r2]), _l2_set(row[col_r3])
            hits = int(lab in s0) + int(lab in s1) + int(lab in s2)
            if hits == 0:
                continue
            n_present += 1
            if hits == 3:
                c3 += 1
            elif hits == 2:
                c2 += 1
            else:
                c1 += 1
        rate3 = c3 / n_present if n_present else float("nan")
        rate2 = c2 / n_present if n_present else float("nan")
        rate1 = c1 / n_present if n_present else float("nan")
        rows_out.append(
            {
                "L2_label": lab,
                "n_rows_present": n_present,
                "rate_3of3": rate3,
                "rate_2of3": rate2,
                "rate_1of3": rate1,
            }
        )
    return pd.DataFrame(rows_out)


def table2_row_decomposition(
    df_ne: pd.DataFrame,
    col_r1: str,
    col_r2: str,
    col_r3: str,
    id_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Per-row consensus/swing. ``jaccard_consensus_only`` is NaN when consensus is empty
    (union nonempty but every label appears in exactly one run)—avoids misleading 1.0.
    """
    out_rows: List[Dict[str, object]] = []
    n_swings: List[int] = []
    cohort_no_nc: List[int] = []
    cohort_has_nc: List[int] = []
    j_has: List[float] = []
    for _, row in df_ne.iterrows():
        s0, s1, s2 = _l2_set(row[col_r1]), _l2_set(row[col_r2]), _l2_set(row[col_r3])
        consensus, swing = _consensus_swing(s0, s1, s2)
        has_consensus = bool(consensus)
        if has_consensus:
            t0, t1, t2 = s0 & consensus, s1 & consensus, s2 & consensus
            jc: float = _min_triplet_jaccard(t0, t1, t2)
            j_has.append(jc)
            cohort_has_nc.append(len(swing))
        else:
            jc = float("nan")
            cohort_no_nc.append(len(swing))
        n_swings.append(len(swing))
        keys = {}
        for c in id_cols:
            if c in row.index:
                keys[c] = row[c]
        out_rows.append(
            {
                **keys,
                "cohort": "has_consensus" if has_consensus else "no_consensus",
                "consensus_labels": "|".join(sorted(consensus)),
                "swing_labels": "|".join(sorted(swing)),
                "n_swing": len(swing),
                "jaccard_consensus_only": jc,
            }
        )
    tbl = pd.DataFrame(out_rows)
    n_no = len(cohort_no_nc)
    n_yes = len(cohort_has_nc)
    swing_hist_all = dict(Counter(n_swings))
    swing_hist_no = dict(Counter(cohort_no_nc)) if cohort_no_nc else {}
    swing_hist_yes = dict(Counter(cohort_has_nc)) if cohort_has_nc else {}
    mean_j_has = float(np.nanmean(j_has)) if j_has else float("nan")
    meta = {
        "n_rows_non_empty_l2_union": int(len(df_ne)),
        "cohorts": {
            "all_three_l2_empty": {
                "n_rows": None,
                "share_of_sample": None,
                "mean_jaccard_consensus_only": None,
                "note": "Filled in run_diagnose; rows omitted from table2_row_decomposition.csv.",
            },
            "nonempty_union_no_consensus": {
                "n_rows": n_no,
                "share_of_sample": None,
                "share_of_nonempty_l2_union": None,
                "mean_n_swing": float(np.mean(cohort_no_nc)) if cohort_no_nc else float("nan"),
                "mean_jaccard_consensus_only": None,
                "n_swing_distribution": {str(k): v for k, v in sorted(swing_hist_no.items())},
                "note": "Per-row jaccard_consensus_only is NaN; majority vote undefined for L2.",
            },
            "nonempty_union_has_consensus": {
                "n_rows": n_yes,
                "share_of_sample": None,
                "share_of_nonempty_l2_union": None,
                "mean_n_swing": float(np.mean(cohort_has_nc)) if cohort_has_nc else float("nan"),
                "mean_jaccard_consensus_only": mean_j_has,
                "n_swing_distribution": {str(k): v for k, v in sorted(swing_hist_yes.items())},
            },
        },
        "mean_jaccard_consensus_only_among_has_consensus_rows": mean_j_has,
        "mean_n_swing_non_empty_union_all": float(np.mean(n_swings)) if n_swings else float("nan"),
        "n_swing_distribution_non_empty_union_all": {
            str(k): v for k, v in sorted(swing_hist_all.items())
        },
    }
    return tbl, meta


def table3_cofluctuation(
    df_ne: pd.DataFrame,
    col_r1: str,
    col_r2: str,
    col_r3: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    n = len(labels)
    mat = np.full((n, n), np.nan, dtype=float)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    swing_sets: List[Set[str]] = []
    for _, row in df_ne.iterrows():
        s0, s1, s2 = _l2_set(row[col_r1]), _l2_set(row[col_r2]), _l2_set(row[col_r3])
        _, swing = _consensus_swing(s0, s1, s2)
        swing_sets.append(swing)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            both = sum(1 for sw in swing_sets if li in sw and lj in sw)
            either = sum(1 for sw in swing_sets if li in sw or lj in sw)
            if either == 0:
                mat[i, j] = float("nan")
            else:
                mat[i, j] = both / either
    return pd.DataFrame(mat, index=list(labels), columns=list(labels))


def table4_empty_l2_l1(
    df_empty: pd.DataFrame,
    l1_cols: Sequence[str],
) -> pd.DataFrame:
    l1_series = []
    for _, row in df_empty.iterrows():
        chosen = ""
        if "L1_label" in row.index and str(row["L1_label"]).strip():
            chosen = str(row["L1_label"]).strip()
        else:
            votes = [
                str(row[c]).strip()
                for c in l1_cols
                if c in row.index and str(row[c]).strip()
            ]
            if votes:
                chosen = Counter(votes).most_common(1)[0][0]
        l1_series.append(chosen)
    s = pd.Series(l1_series, name="L1_for_diagnostic")
    counts = s.value_counts(dropna=False).rename_axis("L1").reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum() if len(df_empty) else 0.0
    return counts


def run_diagnose(triplicate_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(triplicate_csv, encoding="utf-8-sig")
    for c in ("L2_run1", "L2_run2", "L2_run3"):
        if c not in df.columns:
            raise SystemExit(f"Missing column {c!r} in {triplicate_csv}")

    col_r1, col_r2, col_r3 = "L2_run1", "L2_run2", "L2_run3"
    mask_empty = df.apply(
        lambda r: _row_all_l2_empty(
            _l2_set(r[col_r1]), _l2_set(r[col_r2]), _l2_set(r[col_r3])
        ),
        axis=1,
    )
    df_ne = df.loc[~mask_empty].copy()
    df_empty = df.loc[mask_empty].copy()

    id_cols = [c for c in ("id", "doc_id", "unit_id") if c in df.columns]

    t1 = table1_label_stability(df_ne, col_r1, col_r2, col_r3, L2_CANONICAL)
    t1.to_csv(out_dir / "table1_label_stability_profile.csv", index=False, encoding="utf-8-sig")

    t2, meta2 = table2_row_decomposition(df_ne, col_r1, col_r2, col_r3, id_cols)
    n_total = int(len(df))
    n_ne = int(len(df_ne))
    n_empty = int(mask_empty.sum())
    meta2["cohorts"]["all_three_l2_empty"]["n_rows"] = n_empty
    meta2["cohorts"]["all_three_l2_empty"]["share_of_sample"] = (
        n_empty / n_total if n_total else None
    )
    nc = meta2["cohorts"]["nonempty_union_no_consensus"]
    hc = meta2["cohorts"]["nonempty_union_has_consensus"]
    nc["share_of_sample"] = nc["n_rows"] / n_total if n_total else None
    nc["share_of_nonempty_l2_union"] = nc["n_rows"] / n_ne if n_ne else None
    hc["share_of_sample"] = hc["n_rows"] / n_total if n_total else None
    hc["share_of_nonempty_l2_union"] = hc["n_rows"] / n_ne if n_ne else None
    t2.to_csv(out_dir / "table2_row_decomposition.csv", index=False, encoding="utf-8-sig")
    table2_meta_safe = _json_safe(meta2)
    (out_dir / "table2_metadata.json").write_text(
        json.dumps(table2_meta_safe, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    t3 = table3_cofluctuation(df_ne, col_r1, col_r2, col_r3, L2_CANONICAL)
    t3.to_csv(out_dir / "table3_label_cofluctuation.csv", encoding="utf-8-sig")

    l1_run_cols = [c for c in ("L1_run1", "L1_run2", "L1_run3") if c in df.columns]
    t4 = table4_empty_l2_l1(df_empty, l1_run_cols)
    t4.to_csv(out_dir / "table4_empty_l2_l1_distribution.csv", index=False, encoding="utf-8-sig")

    observed_labels = sorted(
        {
            x
            for _, row in df.iterrows()
            for x in (
                _l2_set(row[col_r1])
                | _l2_set(row[col_r2])
                | _l2_set(row[col_r3])
            )
        }
    )
    extra_in_data = [x for x in observed_labels if x not in set(L2_CANONICAL)]

    summary = {
        "input_csv": str(triplicate_csv.resolve()),
        "n_total_rows": int(len(df)),
        "n_rows_all_three_l2_empty": int(mask_empty.sum()),
        "n_rows_nonempty_l2_union": int(len(df_ne)),
        "canonical_l2_labels": list(L2_CANONICAL),
        "extra_l2_labels_in_data": extra_in_data,
        "table2": table2_meta_safe,
        "table2_metadata_file": "table2_metadata.json",
    }
    (out_dir / "diagnostics_summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="L2 stability triplicate diagnostics (four tables).")
    ap.add_argument(
        "--triplicate-csv",
        type=Path,
        required=True,
        help="stability_triplicate_sample.csv from s10 stability run",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for table1–4 CSVs + diagnostics_summary.json",
    )
    args = ap.parse_args()
    run_diagnose(args.triplicate_csv.resolve(), args.out_dir.resolve())
    print(json.dumps({"wrote": str(args.out_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
