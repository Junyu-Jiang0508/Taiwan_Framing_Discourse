#!/usr/bin/env python3
"""Stage 10 — Corpus label quality audit & distribution diagnostics (pre–descriptive stats).

This stage is **not** substantive analysis. It checks that full-corpus labeling behavior
is comparable to the validation set before interpreting F1 or running descriptive stats.

Workflow (per study protocol)::

    1. Distribution consistency — L1/L2 prevalence vs validation; chi-square; >10pp L1 gaps → flag + sample CSV
    2. Empty L2 rate — corpus vs validation band (~18–20%)
    3. Slices — by candidate / source / time period; flag dominant-L1 or skewed L2
    4. L1×L2 joint — cross-tab; theoretical + empirical low-prior cells; sample for review
    5. Stability (optional API) — 100 rows × 3 runs, temp=0.2; L1 agreement; L2 Jaccard
    6. Descriptive tables — only after gates pass (or ``--force-descriptive``)

Usage::

    python s10_corpus_label_audit.py audit [--corpus-csv ...] [--validation-csv ...] --out-dir ...
    python s10_corpus_label_audit.py stability --corpus-csv ... --out-dir ...
    python s10_corpus_label_audit.py describe --corpus-csv ... --out-dir ...  # step 6 only

Each ``audit`` run also writes ``run_manifest.json``, ``METHODS_AUDIT.md``, and
``tables_for_manuscript/`` (rounded tables + reproducibility metadata).

Defaults resolve ``latest_merged_deduped_run.txt`` / ``latest_corpus_run.txt`` and validation
``latest_run.txt`` like ``s08_validation_analyze.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency
except ImportError:
    chi2_contingency = None  # type: ignore

from dotenv import load_dotenv

from framing_batch_core import (
    CORPUS_RESULTS_BASE,
    VALIDATION_OUTPUT_BASE,
    TaiwanBatchManager,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Theoretical L1×L2 pairs to always flag for manual review (extend via --low-prior-pairs-csv).
DEFAULT_LOW_PRIOR_L1_L2: List[Tuple[str, str]] = [
    ("L1-01", "L2-01"),
]

_LATEST_RUN_FILE = "latest_run.txt"
_LATEST_CORPUS_RUN_FILE = "latest_corpus_run.txt"
_LATEST_MERGED_DEDUPED_RUN_FILE = "latest_merged_deduped_run.txt"


# ═══════════════════════════════════════════════════════════════════════════
#  Path resolution (align with s08 / framing_batch_core)
# ═══════════════════════════════════════════════════════════════════════════


def resolve_validation_final_results(base_dir: Path) -> Optional[Path]:
    output_base = base_dir / VALIDATION_OUTPUT_BASE
    pointer = output_base / _LATEST_RUN_FILE
    legacy = output_base / "Validation_v5" / "final_validation_results.csv"
    if pointer.exists():
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        cand = run_dir / "final_results.csv"
        if cand.is_file():
            return cand
    if legacy.is_file():
        return legacy
    return None


def resolve_corpus_final_results(base_dir: Path) -> Optional[Path]:
    output_base = base_dir / CORPUS_RESULTS_BASE
    for ptr_name in (_LATEST_MERGED_DEDUPED_RUN_FILE, _LATEST_CORPUS_RUN_FILE):
        pointer = output_base / ptr_name
        if not pointer.exists():
            continue
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        cand = run_dir / "final_results.csv"
        if cand.is_file():
            return cand
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Parsing & inference (align with s07 / s08)
# ═══════════════════════════════════════════════════════════════════════════


def parse_l2_cell(s: Any) -> List[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def l2_is_empty(s: Any) -> bool:
    return len(parse_l2_cell(s)) == 0


def normalize_l1(s: Any) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip()


def infer_candidate_party(source_blob: str) -> Tuple[str, str]:
    s = str(source_blob).lower()
    if "01_ho" in s or "hou" in s or "侯" in source_blob:
        return "侯友宜", "KMT"
    if "02_ke" in s or "柯" in source_blob:
        return "柯文哲", "TPP"
    if "03_lai" in s or "lai" in s or "赖" in source_blob or "賴" in source_blob:
        return "赖清德", "DPP"
    return "Unknown", "Unknown"


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate" not in out.columns:
        blob = None
        for col in ("source", "_source_file", "source_file"):
            if col in out.columns:
                blob = out[col].astype(str)
                break
        if blob is not None:
            cp = blob.apply(lambda x: pd.Series(infer_candidate_party(x)))
            out["candidate"] = cp[0]
            out["party"] = cp[1]
        else:
            out["candidate"] = "Unknown"
            out["party"] = "Unknown"
    if "party" not in out.columns:
        out["party"] = "Unknown"
    if "source_type" not in out.columns and "_source_file" in out.columns:
        def _st(sf: Any) -> str:
            x = str(sf).lower()
            if "03_x_datasets" in x or "x_datasets" in x or "twitter" in x:
                return "social_media"
            if "conference" in x or "debate" in x:
                return "conference"
            return "news"

        out["source_type"] = out["_source_file"].map(_st)
    elif "source_type" not in out.columns:
        out["source_type"] = "unknown"
    if "date" not in out.columns:
        out["date"] = ""
    out["_period"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M").astype(str)
    out.loc[out["_period"] == "NaT", "_period"] = "_nodate"
    return out


def valid_l1_mask(df: pd.DataFrame) -> pd.Series:
    s = df["L1_label"].map(normalize_l1)
    return s != ""


def chi2_homogeneity(counts_a: np.ndarray, counts_b: np.ndarray) -> Tuple[float, float, str]:
    """2-row contingency chi-square; returns chi2, pvalue, note."""
    if chi2_contingency is None:
        return float("nan"), float("nan"), "scipy not installed; skipped"
    table = np.vstack([counts_a.astype(float), counts_b.astype(float)])
    col_sum = table.sum(axis=0)
    keep = col_sum > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        return float("nan"), float("nan"), "need ≥2 labels with nonzero counts; skipped"
    try:
        chi2, p, _, _ = chi2_contingency(table)
        return float(chi2), float(p), ""
    except Exception as e:
        return float("nan"), float("nan"), str(e)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return len(a & b) / u


def _sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def _gather_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
    }
    try:
        import scipy

        env["scipy"] = scipy.__version__
    except ImportError:
        env["scipy"] = None
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            env["git_commit"] = r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return env


def _format_p_for_manuscript(p: float) -> str:
    if p != p:  # NaN
        return "NA"
    if p < 1e-8:
        return f"<1e-8 (exact {p:.2e})"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _round_prop(x: float, ndp: int = 3) -> float:
    return round(float(x), ndp)


def normalize_source_raw_key(key: str) -> str:
    """Strip trailing .csv from source keys for display/manuscript tables."""
    s = str(key).strip()
    if s.lower().endswith(".csv"):
        return s[:-4]
    return s


def load_low_prior_pairs(path: Optional[Path]) -> List[Tuple[str, str]]:
    pairs = list(DEFAULT_LOW_PRIOR_L1_L2)
    if path is None or not path.is_file():
        return pairs
    extra = pd.read_csv(path, encoding="utf-8-sig")
    if not {"L1", "L2"}.issubset(extra.columns):
        logger.warning("low-prior CSV needs L1,L2 columns; ignoring %s", path)
        return pairs
    for _, r in extra.iterrows():
        pairs.append((str(r["L1"]).strip(), str(r["L2"]).strip()))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  Audit steps 1–4
# ═══════════════════════════════════════════════════════════════════════════


def l1_l2_distribution_tables(
    df: pd.DataFrame, prefix: str, out_dir: Path
) -> Dict[str, Any]:
    m = valid_l1_mask(df)
    sub = df.loc[m]
    n = len(sub)
    l1_vc = sub["L1_label"].astype(str).str.strip().value_counts()
    l1_rows = [
        {"label": lab, "count": int(c), "proportion": float(c / n) if n else 0.0}
        for lab, c in l1_vc.items()
    ]
    pd.DataFrame(l1_rows).to_csv(out_dir / f"{prefix}_L1_frequency.csv", index=False, encoding="utf-8-sig")

    l2_binary: Dict[str, int] = {}
    l2_instance = 0
    for s in sub["L2_labels"]:
        labs = parse_l2_cell(s)
        l2_instance += len(labs)
        for lab in labs:
            l2_binary[lab] = l2_binary.get(lab, 0) + 1
    nrows = n
    l2_rows = [
        {
            "label": lab,
            "row_count_with_label": c,
            "row_prevalence": float(c / nrows) if nrows else 0.0,
        }
        for lab, c in sorted(l2_binary.items(), key=lambda x: -x[1])
    ]
    pd.DataFrame(l2_rows).to_csv(out_dir / f"{prefix}_L2_row_prevalence.csv", index=False, encoding="utf-8-sig")

    empty_rate = float(sub["L2_labels"].map(l2_is_empty).mean()) if n else 0.0
    return {
        "n_rows_l1_nonempty": int(n),
        "l1_counts": {str(k): int(v) for k, v in l1_vc.items()},
        "l2_row_prevalence": {r["label"]: r["row_prevalence"] for r in l2_rows},
        "empty_l2_rate": empty_rate,
        "l2_instances_per_row_mean": float(l2_instance / n) if n else 0.0,
    }


def compare_distributions(
    corpus_stats: Dict[str, Any],
    val_stats: Dict[str, Any],
    out_dir: Path,
    l1_pp_threshold: float,
) -> Dict[str, Any]:
    """Chi-square + per-label proportion gaps for L1 and L2 row-prevalence."""
    c_l1 = corpus_stats["l1_counts"]
    v_l1 = val_stats["l1_counts"]
    labels = sorted(set(c_l1) | set(v_l1))
    nc, nv = corpus_stats["n_rows_l1_nonempty"], val_stats["n_rows_l1_nonempty"]
    ca = np.array([c_l1.get(l, 0) for l in labels], dtype=float)
    va = np.array([v_l1.get(l, 0) for l in labels], dtype=float)
    chi2, p, note = chi2_homogeneity(va, ca)
    comp_l1 = []
    large_gaps = []
    for lab in labels:
        pv = va[labels.index(lab)] / nv if nv else 0.0
        pc = ca[labels.index(lab)] / nc if nc else 0.0
        diff_pp = (pc - pv) * 100.0
        row = {
            "L1": lab,
            "validation_prop": pv,
            "corpus_prop": pc,
            "diff_percentage_points": diff_pp,
            "validation_count": int(va[labels.index(lab)]),
            "corpus_count": int(ca[labels.index(lab)]),
        }
        comp_l1.append(row)
        if abs(diff_pp) > l1_pp_threshold:
            large_gaps.append(lab)
    pd.DataFrame(comp_l1).to_csv(out_dir / "compare_L1_validation_vs_corpus.csv", index=False, encoding="utf-8-sig")

    c_l2 = corpus_stats["l2_row_prevalence"]
    v_l2 = val_stats["l2_row_prevalence"]
    l2_labels = sorted(set(c_l2) | set(v_l2))
    va2 = np.array([v_l2.get(l, 0) * nv for l in l2_labels], dtype=float)
    ca2 = np.array([c_l2.get(l, 0) * nc for l in l2_labels], dtype=float)
    chi2_l2, p_l2, note_l2 = chi2_homogeneity(va2, ca2)
    comp_l2 = []
    for lab in l2_labels:
        comp_l2.append(
            {
                "L2": lab,
                "validation_row_prev": v_l2.get(lab, 0.0),
                "corpus_row_prev": c_l2.get(lab, 0.0),
                "diff_pp": (c_l2.get(lab, 0.0) - v_l2.get(lab, 0.0)) * 100.0,
            }
        )
    pd.DataFrame(comp_l2).to_csv(out_dir / "compare_L2_row_prev_validation_vs_corpus.csv", index=False, encoding="utf-8-sig")

    return {
        "L1_chi2": chi2,
        "L1_chi2_p": p,
        "L1_chi2_note": note,
        "L1_labels_over_pp_threshold": large_gaps,
        "L2_chi2": chi2_l2,
        "L2_chi2_p": p_l2,
        "L2_chi2_note": note_l2,
        "validation_empty_l2_rate": val_stats["empty_l2_rate"],
        "corpus_empty_l2_rate": corpus_stats["empty_l2_rate"],
    }


def sample_corpus_by_l1(
    corpus_df: pd.DataFrame, l1_label: str, n: int, seed: int, out_path: Path
) -> int:
    m = valid_l1_mask(corpus_df) & (corpus_df["L1_label"].astype(str).str.strip() == l1_label)
    sub = corpus_df.loc[m]
    if sub.empty:
        return 0
    k = min(n, len(sub))
    sampled = sub.sample(n=k, random_state=seed)
    sampled.to_csv(out_path, index=False, encoding="utf-8-sig")
    return len(sampled)


def slice_diagnostics(
    corpus_df: pd.DataFrame,
    out_dir: Path,
    l1_dominance: float,
    l2_skew_ratio: float,
    sample_n: int,
    seed: int,
) -> Dict[str, Any]:
    """Per candidate / source / source_type / period / raw source: L1 & L2 prevalence."""
    df = add_derived_columns(corpus_df)
    m = valid_l1_mask(df)
    work = df.loc[m].copy()
    anomalies: List[Dict[str, Any]] = []
    slice_specs = [
        ("candidate", "candidate"),
        ("source_raw", "source"),
        ("source_type", "source_type"),
        ("period", "_period"),
    ]

    rows_out: List[Dict[str, Any]] = []
    for slice_name, col in slice_specs:
        if col not in work.columns:
            continue
        for key, g in work.groupby(col, dropna=False):
            n = len(g)
            if n < 20:
                continue
            l1_vc = g["L1_label"].astype(str).str.strip().value_counts(normalize=True)
            top_l1 = l1_vc.index[0] if len(l1_vc) else ""
            top_share = float(l1_vc.iloc[0]) if len(l1_vc) else 0.0
            l2_prev: Dict[str, float] = {}
            for _, r in g.iterrows():
                for lab in parse_l2_cell(r["L2_labels"]):
                    l2_prev[lab] = l2_prev.get(lab, 0) + 1
            l2_prev = {k: v / n for k, v in l2_prev.items()}
            max_l2 = max(l2_prev.values()) if l2_prev else 0.0
            key_str = str(key)
            key_norm = normalize_source_raw_key(key_str) if slice_name == "source_raw" else key_str
            row = {
                "slice": slice_name,
                "key": key_str,
                "key_normalized": key_norm,
                "n": n,
                "top_L1": top_l1,
                "top_L1_share": top_share,
                "max_L2_row_prev": max_l2,
            }
            rows_out.append(row)
            reasons = []
            if top_share >= l1_dominance:
                reasons.append("dominant_L1")
            corpus_max = max(l2_prev.values()) if l2_prev else 0.0
            if l2_prev and corpus_max > 0:
                rest = [v for k, v in l2_prev.items() if v < corpus_max]
                if rest and corpus_max / (sum(rest) / len(rest) + 1e-9) > l2_skew_ratio:
                    reasons.append("skewed_L2_vs_slice_mean")
            if reasons:
                anomalies.append({**row, "flags": "|".join(reasons)})
                sp = out_dir / f"sample_slice_{slice_name}_{_safe_fname(key_norm)}.csv"
                k_take = min(sample_n, len(g))
                g.sample(n=k_take, random_state=seed).to_csv(sp, index=False, encoding="utf-8-sig")

    slice_df = pd.DataFrame(rows_out)
    slice_df.to_csv(out_dir / "slice_prevalence_summary.csv", index=False, encoding="utf-8-sig")
    return {"slice_anomalies": anomalies, "n_slice_rows": len(rows_out), "slice_df": slice_df}


def _safe_fname(x: Any) -> str:
    s = str(x)[:80]
    for ch in ' /\\:*?"<>|\n\r\t':
        s = s.replace(ch, "_")
    return s or "unknown"


def joint_l1_l2_audit(
    corpus_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    out_dir: Path,
    low_prior_pairs: List[Tuple[str, str]],
    empirical_zero_val_min_corpus: int,
    sample_joint_n: int,
    seed: int,
) -> Dict[str, Any]:
    m = valid_l1_mask(corpus_df)
    csub = corpus_df.loc[m]
    records = []
    for _, r in csub.iterrows():
        l1 = str(r["L1_label"]).strip()
        for l2 in parse_l2_cell(r["L2_labels"]):
            records.append((l1, l2))
    if not records:
        return {
            "low_prior_cells": [],
            "joint_note": "no L1/L2 pairs in corpus",
            "n_flagged_unique": 0,
            "flag_df": pd.DataFrame(columns=["L1", "L2", "reason", "corpus_pair_count", "validation_pair_count"]),
        }
    j_corpus = pd.crosstab(
        pd.Series([x[0] for x in records], name="L1"),
        pd.Series([x[1] for x in records], name="L2"),
    )
    j_corpus.to_csv(out_dir / "joint_L1_x_L2_corpus_crosstab_counts.csv", encoding="utf-8-sig")

    val_joint: Optional[pd.DataFrame] = None
    if val_df is not None:
        mv = valid_l1_mask(val_df)
        vsub = val_df.loc[mv]
        vr = []
        for _, r in vsub.iterrows():
            l1 = str(r["L1_label"]).strip()
            for l2 in parse_l2_cell(r["L2_labels"]):
                vr.append((l1, l2))
        if vr:
            val_joint = pd.crosstab(
                pd.Series([x[0] for x in vr], name="L1"),
                pd.Series([x[1] for x in vr], name="L2"),
            )
            val_joint.to_csv(out_dir / "joint_L1_x_L2_validation_crosstab_counts.csv", encoding="utf-8-sig")

    flagged: List[Dict[str, Any]] = []
    for l1, l2 in low_prior_pairs:
        cc = int(j_corpus.loc[l1, l2]) if l1 in j_corpus.index and l2 in j_corpus.columns else 0
        vc = (
            int(val_joint.loc[l1, l2])
            if val_joint is not None and l1 in val_joint.index and l2 in val_joint.columns
            else None
        )
        flagged.append(
            {
                "L1": l1,
                "L2": l2,
                "reason": "theoretical_low_prior",
                "corpus_pair_count": cc,
                "validation_pair_count": vc,
            }
        )

    if val_joint is not None:
        for l1 in j_corpus.index:
            for l2 in j_corpus.columns:
                cc = int(j_corpus.loc[l1, l2])
                vc = int(val_joint.loc[l1, l2]) if l1 in val_joint.index and l2 in val_joint.columns else 0
                if vc == 0 and cc >= empirical_zero_val_min_corpus:
                    flagged.append(
                        {
                            "L1": l1,
                            "L2": l2,
                            "reason": "zero_in_validation_high_in_corpus",
                            "corpus_pair_count": cc,
                            "validation_pair_count": vc,
                        }
                    )

    flag_df = pd.DataFrame(flagged).drop_duplicates(subset=["L1", "L2", "reason"])
    flag_df.to_csv(out_dir / "joint_flagged_low_prior_cells.csv", index=False, encoding="utf-8-sig")

    rng = random.Random(seed)
    joint_samples: List[pd.DataFrame] = []
    for _, row in flag_df.iterrows():
        l1, l2 = row["L1"], row["L2"]
        mask = m & (corpus_df["L1_label"].astype(str).str.strip() == l1)
        mask &= corpus_df["L2_labels"].map(lambda s: l2 in parse_l2_cell(s))
        hit = corpus_df.loc[mask]
        if len(hit) == 0:
            continue
        take = hit.sample(n=min(sample_joint_n, len(hit)), random_state=rng.randint(0, 2**31 - 1))
        joint_samples.append(take)
    if joint_samples:
        pd.concat(joint_samples, ignore_index=True).to_csv(
            out_dir / "sample_joint_low_prior_rows.csv", index=False, encoding="utf-8-sig"
        )

    return {
        "low_prior_cells": flagged[:200],
        "n_flagged_unique": len(flag_df),
        "flag_df": flag_df,
    }


def empty_l2_audit(c_stats: Dict[str, Any], v_stats: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    band_lo, band_hi = 0.18, 0.20
    ce, ve = c_stats["empty_l2_rate"], v_stats["empty_l2_rate"]
    warn_high = ce > 0.30 and ce > ve + 0.10
    warn_low = ce < 0.10 and ce < ve - 0.08
    note = []
    if ve < band_lo or ve > band_hi:
        note.append(f"validation empty-L2 rate {ve:.1%} outside typical {band_lo:.0%}-{band_hi:.0%} band")
    if warn_high:
        note.append("corpus empty L2 much higher than validation — check sources / over-caution")
    if warn_low:
        note.append("corpus empty L2 much lower than validation — possible over-labeling of L2")
    summary = {
        "corpus_empty_l2_rate": ce,
        "validation_empty_l2_rate": ve,
        "warn_high_vs_validation": warn_high,
        "warn_low_vs_validation": warn_low,
        "notes": note,
    }
    with open(out_dir / "empty_l2_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Publication-oriented bundle: manifest, methods text, rounded tables
# ═══════════════════════════════════════════════════════════════════════════


def write_run_manifest(
    out_dir: Path,
    *,
    command: str,
    argv: List[str],
    corpus_path: Path,
    validation_path: Optional[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    started = extra.get("started_utc") if extra else None
    manifest: Dict[str, Any] = {
        "schema": "s10_run_manifest_v1",
        "command": command,
        "argv": argv,
        "cwd": os.getcwd(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _gather_environment(),
        "inputs": {
            "corpus_csv": str(corpus_path),
            "corpus_sha256": _sha256_file(corpus_path),
            "validation_csv": str(validation_path) if validation_path else None,
            "validation_sha256": _sha256_file(validation_path) if validation_path else None,
        },
    }
    if started:
        manifest["started_utc"] = started
    if extra:
        for k, v in extra.items():
            if k != "started_utc" and k not in manifest:
                manifest[k] = v
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_methods_audit_md(
    out_dir: Path,
    *,
    corpus_path: Path,
    validation_path: Path,
    thresholds: Dict[str, Any],
    scipy_available: bool,
) -> None:
    lines = [
        "# Label audit — methods (Stage 10)",
        "",
        "## What is being compared",
        "",
        "Unless you pass alternate CSVs, **both** inputs are `final_results.csv` files produced by the batch "
        "annotation pipeline (model predictions). The validation file is **not** automatically merged with "
        "human reference labels. Any statement about “validation set distribution” in this audit refers to "
        "**model output on the validation sample**, not gold-standard prevalence. To compare corpus to human "
        "gold, supply a reference CSV and extend the pipeline or merge externally, then re-run audit.",
        "",
        "## Row inclusion",
        "",
        "- **L1-distribution rows**: units with non-empty `L1_label` after stripping whitespace.",
        "- **Empty L2**: `L2_labels` parses to zero tokens (empty string, NaN, or no pipe-separated codes).",
        "",
        "## Step 1 — Distribution consistency",
        "",
        "- **L1**: Multinomial counts by label; **Pearson χ² test of homogeneity** on a 2×K table "
        "(row 1 = validation counts, row 2 = corpus counts). Columns with zero total counts are dropped. "
        "If fewer than two nonempty columns remain, the test is skipped (`note` in machine-readable outputs).",
        f"- **L1 practical flag**: any label whose corpus percentage minus validation percentage exceeds "
        f"**±{thresholds['l1_pp_threshold']:.1f} percentage points** triggers a supplemental random sample CSV.",
        "- **L2**: For each L2 code, **row prevalence** = fraction of rows (with valid L1) where that code "
        "appears in the pipe-separated `L2_labels` list. χ² homogeneity uses implied counts "
        "(prevalence × N) per dataset.",
        "",
        "## Step 2 — Empty L2 rate",
        "",
        "- Rates are **row fractions** among L1-valid rows.",
        "- Heuristic warnings compare corpus vs validation and vs an informal 18–20% band on validation only.",
        "",
        "## Step 3 — Slice diagnostics",
        "",
        "- Slices: `candidate` (inferred from `source` / `_source_file` if missing), raw `source`, "
        "`source_type`, calendar **month** from `date` (`_nodate` if unparsable).",
        f"- **dominant_L1**: top L1 share ≥ **{thresholds['l1_dominance']:.2f}** within the slice.",
        f"- **skewed_L2_vs_slice_mean**: let *m* = max label row-prevalence among L2 codes in the slice, "
        f"and *μ* = mean of the remaining labels’ prevalences (empty slice of L2 codes skips). Flag if "
        f"*m* / (*μ* + ε) > **{thresholds['l2_skew_ratio']:.1f}** (ε = 1e-9). This is a **screening heuristic**, "
        "not a formal test.",
        "- `key_normalized` on `source_raw` strips a trailing `.csv` for readability.",
        "",
        "## Step 4 — L1×L2 joint table",
        "",
        "- Each **(L1, L2) pair** from the same row expands to one count in the crosstab (multi-L2 rows "
        "contribute multiple cells).",
        f"- **Theoretical low-prior list**: configurable; default includes (L1-01, L2-01). "
        f"**Empirical flag**: validation joint count = 0 and corpus joint count ≥ "
        f"**{int(thresholds['empirical_joint_min_corpus'])}**.",
        "",
        "## Step 5 — Stability (subcommand `stability`)",
        "",
        "- Triplicate **Chat Completions** (not Batch API), same prompts as parallel corpus labeling: "
        "one L1 call + one L2 call per replicate, user-specified `temperature` (default 0.2).",
        "",
        "## Step 6 — Descriptive tables",
        "",
        "- Written under `06_descriptive/` when gates allow or when forced; see `run_manifest.json` for flags.",
        "",
        "## Output layout (this run)",
        "",
        "| Path | Role |",
        "|------|------|",
        "| `run_manifest.json` | Reproducibility: argv, hashes, versions |",
        "| `METHODS_AUDIT.md` | This document |",
        "| `tables_for_manuscript/` | Rounded CSVs for supplementary tables |",
        "| `corpus_*`, `validation_*` | Full-precision internal frequency tables |",
        "| `06_descriptive/` | Full-corpus descriptive exports (if emitted) |",
        "| `compare_*.csv` | Machine-precision comparison |",
        "| `joint_L1_x_L2_*_crosstab_counts.csv` | L1×L2 count matrices |",
        "",
        "## Software",
        "",
        f"- **scipy** for χ²: {'available' if scipy_available else 'not installed (tests skipped in notes)'}",
        "",
    ]
    (out_dir / "METHODS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def export_tables_for_manuscript(
    out_dir: Path,
    *,
    cmp_res: Dict[str, Any],
    c_stats: Dict[str, Any],
    v_stats: Dict[str, Any],
    slice_df: pd.DataFrame,
    flag_df: pd.DataFrame,
    empty_res: Dict[str, Any],
) -> None:
    tdir = out_dir / "tables_for_manuscript"
    tdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "dataset": "validation",
                "N_rows_L1_valid": int(v_stats["n_rows_l1_nonempty"]),
            },
            {
                "dataset": "corpus",
                "N_rows_L1_valid": int(c_stats["n_rows_l1_nonempty"]),
            },
        ]
    ).to_csv(tdir / "Table_sample_sizes.csv", index=False, encoding="utf-8-sig")

    l1_path = out_dir / "compare_L1_validation_vs_corpus.csv"
    if l1_path.is_file():
        l1 = pd.read_csv(l1_path, encoding="utf-8-sig")
        l1_out = pd.DataFrame(
            {
                "L1": l1["L1"],
                "n_validation": l1["validation_count"].astype(int),
                "pct_validation": (l1["validation_prop"] * 100.0).round(1),
                "n_corpus": l1["corpus_count"].astype(int),
                "pct_corpus": (l1["corpus_prop"] * 100.0).round(1),
                "diff_pp_corpus_minus_validation": l1["diff_percentage_points"].round(1),
            }
        )
        l1_out.to_csv(tdir / "Table_L1_validation_vs_corpus_rounded.csv", index=False, encoding="utf-8-sig")

    l2_path = out_dir / "compare_L2_row_prev_validation_vs_corpus.csv"
    if l2_path.is_file():
        l2 = pd.read_csv(l2_path, encoding="utf-8-sig")
        l2_out = pd.DataFrame(
            {
                "L2": l2["L2"],
                "row_prev_validation_pct": (l2["validation_row_prev"] * 100.0).round(1),
                "row_prev_corpus_pct": (l2["corpus_row_prev"] * 100.0).round(1),
                "diff_pp_corpus_minus_validation": l2["diff_pp"].round(1),
            }
        )
        l2_out.to_csv(tdir / "Table_L2_row_prevalence_rounded.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "metric": "empty_L2_rate_validation",
                "value_pct": round(float(empty_res["validation_empty_l2_rate"]) * 100, 1),
                "flag": "",
            },
            {
                "metric": "empty_L2_rate_corpus",
                "value_pct": round(float(empty_res["corpus_empty_l2_rate"]) * 100, 1),
                "flag": "",
            },
            {
                "metric": "warn_high_corpus_empty_L2",
                "value_pct": "",
                "flag": bool(empty_res["warn_high_vs_validation"]),
            },
            {
                "metric": "warn_low_corpus_empty_L2",
                "value_pct": "",
                "flag": bool(empty_res["warn_low_vs_validation"]),
            },
        ]
    ).to_csv(tdir / "Table_empty_L2_summary.csv", index=False, encoding="utf-8-sig")

    def _p_disp(key: str) -> str:
        p = cmp_res.get(key)
        if p is None or (isinstance(p, float) and p != p):
            return "NA"
        return _format_p_for_manuscript(float(p))

    chi_rows = [
        {
            "test": "L1_homogeneity_validation_vs_corpus",
            "chi2": round(float(cmp_res["L1_chi2"]), 3) if cmp_res.get("L1_chi2") == cmp_res.get("L1_chi2") else None,
            "p_value_display": _p_disp("L1_chi2_p"),
            "p_value_exact": cmp_res.get("L1_chi2_p"),
            "note": cmp_res.get("L1_chi2_note") or "",
        },
        {
            "test": "L2_row_prevalence_homogeneity",
            "chi2": round(float(cmp_res["L2_chi2"]), 3) if cmp_res.get("L2_chi2") == cmp_res.get("L2_chi2") else None,
            "p_value_display": _p_disp("L2_chi2_p"),
            "p_value_exact": cmp_res.get("L2_chi2_p"),
            "note": cmp_res.get("L2_chi2_note") or "",
        },
    ]
    pd.DataFrame(chi_rows).to_csv(tdir / "Table_chi_square_tests.csv", index=False, encoding="utf-8-sig")

    if not slice_df.empty:
        sm = slice_df.copy()
        sm = sm[~((sm["slice"] == "candidate") & (sm["key"].astype(str) == "Unknown"))]
        sm["top_L1_share_pct"] = (sm["top_L1_share"] * 100.0).round(1)
        sm["max_L2_row_prev_pct"] = (sm["max_L2_row_prev"] * 100.0).round(1)
        cols = [
            c
            for c in ("slice", "key", "key_normalized", "n", "top_L1", "top_L1_share_pct", "max_L2_row_prev_pct")
            if c in sm.columns
        ]
        sm[cols].to_csv(tdir / "Table_slice_prevalence_rounded.csv", index=False, encoding="utf-8-sig")

    if not flag_df.empty:
        flag_df.to_csv(tdir / "Table_joint_flagged_low_prior_cells.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "artifact": "corpus_L1_frequency.csv / 06_descriptive/L1_frequency_full.csv",
                "note": "Same population (L1-valid corpus rows); manuscript table uses rounded compare + descriptive prop.",
            },
            {
                "artifact": "corpus_L2_row_prevalence.csv vs 06_descriptive/L2_frequency_full.csv",
                "note": "Row prevalence vs L2 token instance counts — different aggregations; do not merge without renaming.",
            },
        ]
    ).to_csv(tdir / "README_duplicate_artifacts.csv", index=False, encoding="utf-8-sig")


def export_descriptive(corpus_df: pd.DataFrame, out_dir: Path) -> None:
    desc = out_dir / "06_descriptive"
    desc.mkdir(parents=True, exist_ok=True)
    df = add_derived_columns(corpus_df)
    m = valid_l1_mask(df)
    sub = df.loc[m]
    n = len(sub)

    l1_vc = sub["L1_label"].astype(str).str.strip().value_counts()
    pd.DataFrame({"L1": l1_vc.index, "count": l1_vc.values, "prop": l1_vc.values / n}).to_csv(
        desc / "L1_frequency_full.csv", index=False, encoding="utf-8-sig"
    )

    l2_counts: Dict[str, int] = {}
    for s in sub["L2_labels"]:
        for lab in parse_l2_cell(s):
            l2_counts[lab] = l2_counts.get(lab, 0) + 1
    pd.DataFrame(
        [{"L2": k, "instance_count": v, "row_prev": v / n} for k, v in sorted(l2_counts.items(), key=lambda x: -x[1])]
    ).to_csv(desc / "L2_frequency_full.csv", index=False, encoding="utf-8-sig")

    ct = pd.crosstab(sub["L1_label"].astype(str).str.strip(), sub["candidate"])
    ct.to_csv(desc / "L1_x_candidate_crosstab.csv", encoding="utf-8-sig")

    labels = sorted(l2_counts.keys())
    co = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for s in sub["L2_labels"]:
        labs = sorted(set(parse_l2_cell(s)))
        for a, b in combinations(labs, 2):
            if a in co.index and b in co.columns:
                co.loc[a, b] += 1
                co.loc[b, a] += 1
    co.to_csv(desc / "L2_cooccurrence_matrix.csv", encoding="utf-8-sig")

    sub2 = sub.copy()
    sub2["_p"] = pd.to_datetime(sub2["date"], errors="coerce").dt.to_period("M").astype(str)
    sub2.loc[sub2["_p"] == "NaT", "_p"] = "_nodate"
    l1_trend = sub2.groupby("_p")["L1_label"].value_counts(normalize=True).unstack(fill_value=0)
    l1_trend.to_csv(desc / "L1_trend_by_month.csv", encoding="utf-8-sig")

    l2_rows = []
    for p, g in sub2.groupby("_p"):
        gn = len(g)
        prev: Dict[str, float] = {}
        for _, r in g.iterrows():
            for lab in parse_l2_cell(r["L2_labels"]):
                prev[lab] = prev.get(lab, 0) + 1
        for lab, c in prev.items():
            l2_rows.append({"period": p, "L2": lab, "row_prev": c / gn if gn else 0.0, "count": c})
    pd.DataFrame(l2_rows).to_csv(desc / "L2_trend_by_month_long.csv", index=False, encoding="utf-8-sig")


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5 — Stability (sync API)
# ═══════════════════════════════════════════════════════════════════════════


def _defaults_batch(base_dir: Path) -> dict:
    return dict(
        l1=str(base_dir / "01_data/05_labels_guidance/01_annotation_guide_label1_v9.csv"),
        l2=str(base_dir / "01_data/05_labels_guidance/02_annotation_guide_label2_v10.csv"),
        l1_fs=str(base_dir / "01_data/05_labels_guidance/03_fewshot_L1_v9.csv"),
        l2_fs=str(base_dir / "01_data/05_labels_guidance/03_fewshot_L2_v10.csv"),
        hard=str(base_dir / "01_data/05_labels_guidance/04_hard_case_pool.csv"),
    )


def _parse_completion_json(manager: TaiwanBatchManager, text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "json_decode", "raw": text[:500]}


def _call_l1_l2_once(
    manager: TaiwanBatchManager,
    row: pd.Series,
    temperature: float,
    run_ix: int,
) -> Tuple[Optional[str], Set[str], str]:
    """One parallel-style L1+L2 call (same user text; two completions). Returns (l1, l2_set, err)."""
    if pd.isna(row.get("sentence")) or str(row["sentence"]).strip() == "":
        return None, set(), "empty_sentence"
    cid = f"stab_{uuid.uuid4().hex}"
    rng_l1 = TaiwanBatchManager._rng_for_request_id(manager.hard_pool_seed, f"{cid}:L1:r{run_ix}")
    rng_l2 = TaiwanBatchManager._rng_for_request_id(manager.hard_pool_seed, f"{cid}:L2:r{run_ix}")
    sys_l1 = manager._build_l1_system_prompt(rng_l1)
    sys_l2 = manager._build_l2_system_prompt(rng_l2)
    user_plain = TaiwanBatchManager._build_user_text(row)
    err_parts = []
    l1_val: Optional[str] = None
    l2_set: Set[str] = set()
    valid_l1 = set(manager.l1_guide["label_id"].astype(str)) if not manager.l1_guide.empty else set()
    valid_l2 = set(manager.l2_guide["label_id"].astype(str)) if not manager.l2_guide.empty else set()

    for phase, sys_p, key in (
        ("L1", sys_l1, "L1"),
        ("L2", sys_l2, "L2"),
    ):
        try:
            resp = manager.client.chat.completions.create(
                model=manager.model,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_plain},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            data = _parse_completion_json(manager, content)
            if data.get("error"):
                err_parts.append(f"{phase}:{data.get('error')}")
                continue
            if phase == "L1":
                raw = str(data.get("L1", "")).strip()
                l1_val = manager._normalize_l1_code(raw)
                if valid_l1 and l1_val not in valid_l1:
                    err_parts.append("L1_invalid")
                    l1_val = None
            else:
                l2_pred = data.get("L2", [])
                if isinstance(l2_pred, list):
                    for x in l2_pred:
                        t = str(x).strip()
                        if t and (not valid_l2 or t in valid_l2):
                            l2_set.add(t)
                else:
                    err_parts.append("L2_not_list")
        except Exception as e:
            err_parts.append(f"{phase}_exc:{e}")
    return l1_val, l2_set, "|".join(err_parts)


def cmd_stability(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set"); sys.exit(1)
    base_dir = Path(os.getcwd())
    defs = _defaults_batch(base_dir)
    corpus_path = Path(args.corpus_csv).resolve()
    df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    if "id" not in df.columns:
        df["id"] = df.index
    m = valid_l1_mask(df) & df["sentence"].astype(str).str.strip().ne("")
    pool = df.loc[m]
    n = min(args.n_sample, len(pool))
    if n < 10:
        raise SystemExit(f"Too few rows for stability test after filter: {n}")
    sampled = pool.sample(n=n, random_state=args.seed)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    argv_snapshot = sys.argv[:]

    manager = TaiwanBatchManager(api_key, model=args.model)
    manager.load_guides(defs["l1"], defs["l2"], defs["l1_fs"], defs["l2_fs"])
    manager.load_hard_pool(
        args.hard_pool or defs["hard"],
        n_l1=args.hard_pool_n_l1,
        n_l2=args.hard_pool_n_l2,
        seed=args.hard_pool_seed,
        enabled=not args.no_hard_pool,
    )

    runs_l1: List[List[Optional[str]]] = [[] for _ in range(3)]
    runs_l2: List[List[Set[str]]] = [[] for _ in range(3)]
    errors: List[str] = []
    t0 = time.time()
    for i, (_, row) in enumerate(sampled.iterrows()):
        for r in range(3):
            l1, l2s, err = _call_l1_l2_once(manager, row, args.temperature, r)
            runs_l1[r].append(l1)
            runs_l2[r].append(l2s)
            if err:
                errors.append(f"id={row.get('id')} run{r}:{err}")
        if (i + 1) % 10 == 0:
            logger.info("stability %s/%s rows (%.1fs)", i + 1, n, time.time() - t0)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    l1_agree = sum(
        1
        for a, b, c in zip(runs_l1[0], runs_l1[1], runs_l1[2])
        if a and a == b == c
    )
    l1_any = sum(1 for a, b, c in zip(runs_l1[0], runs_l1[1], runs_l1[2]) if a or b or c)
    l1_rate = l1_agree / n if n else 0.0

    jacc_mins = []
    for i in range(n):
        s0, s1, s2 = runs_l2[0][i], runs_l2[1][i], runs_l2[2][i]
        jm = min(jaccard(s0, s1), jaccard(s0, s2), jaccard(s1, s2))
        jacc_mins.append(jm)
    jacc_pass = sum(1 for j in jacc_mins if j >= args.l2_jaccard_threshold) / n if n else 0.0
    jacc_mean = float(np.mean(jacc_mins)) if jacc_mins else 0.0

    summary = {
        "n_sample": n,
        "temperature": args.temperature,
        "l1_full_agreement_rate": l1_rate,
        "l1_full_agreement_count": int(l1_agree),
        "l2_min_pairwise_jaccard_mean": jacc_mean,
        "l2_rows_min_jacc_ge_threshold_rate": jacc_pass,
        "l2_jaccard_threshold": args.l2_jaccard_threshold,
        "recommend_majority_vote_l1": l1_rate < 0.85,
        "recommend_majority_vote_l2": jacc_pass < 0.80,
        "protocol_note": "L1: target >=0.90 single pass OK; <0.85 suggest 3-run majority. "
        "L2: min pairwise Jaccard per row vs threshold 0.80.",
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out_rows = sampled.reset_index(drop=True).copy()
    for r in range(3):
        out_rows[f"L1_run{r + 1}"] = runs_l1[r]
        out_rows[f"L2_run{r + 1}"] = ["|".join(sorted(s)) for s in runs_l2[r]]
    out_rows["_min_jaccard_L2"] = jacc_mins
    out_rows.to_csv(out_dir / "stability_triplicate_sample.csv", index=False, encoding="utf-8-sig")
    if errors:
        (out_dir / "stability_errors.txt").write_text("\n".join(errors[:500]), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "stability_summary.json")
    write_run_manifest(
        out_dir,
        command="stability",
        argv=argv_snapshot,
        corpus_path=corpus_path,
        validation_path=None,
        extra={
            "started_utc": started_utc,
            "openai_model": args.model,
            "n_sample": n,
            "temperature": args.temperature,
            "l2_jaccard_threshold": args.l2_jaccard_threshold,
            "stability_summary": summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_describe(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    export_descriptive(df, out_dir)
    logger.info("Descriptive tables under %s/06_descriptive", out_dir)


def cmd_audit(args: argparse.Namespace) -> None:
    base_dir = Path(os.getcwd())
    corpus_path = Path(args.corpus_csv).resolve() if args.corpus_csv else None
    val_path = Path(args.validation_csv).resolve() if args.validation_csv else None
    if corpus_path is None:
        r = resolve_corpus_final_results(base_dir)
        if not r:
            raise SystemExit("Could not resolve corpus final_results.csv; pass --corpus-csv")
        corpus_path = r
    if val_path is None:
        r = resolve_validation_final_results(base_dir)
        if not r:
            raise SystemExit("Could not resolve validation final_results.csv; pass --validation-csv")
        val_path = r

    corpus_df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    val_df = pd.read_csv(val_path, encoding="utf-8-sig")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    argv_snapshot = sys.argv[:]

    low_prior = load_low_prior_pairs(Path(args.low_prior_pairs_csv) if args.low_prior_pairs_csv else None)

    c_stats = l1_l2_distribution_tables(corpus_df, "corpus", out_dir)
    v_stats = l1_l2_distribution_tables(val_df, "validation", out_dir)
    cmp_res = compare_distributions(c_stats, v_stats, out_dir, args.l1_pp_threshold)
    empty_res = empty_l2_audit(c_stats, v_stats, out_dir)
    slice_res = slice_diagnostics(
        corpus_df,
        out_dir,
        l1_dominance=args.l1_dominance,
        l2_skew_ratio=args.l2_skew_ratio,
        sample_n=args.slice_sample_n,
        seed=args.seed,
    )
    joint_res = joint_l1_l2_audit(
        corpus_df,
        val_df,
        out_dir,
        low_prior,
        args.empirical_joint_min_corpus,
        args.joint_sample_n,
        args.seed,
    )

    for lab in cmp_res["L1_labels_over_pp_threshold"]:
        p = out_dir / f"sample_supplement_L1_{lab}.csv"
        sample_corpus_by_l1(corpus_df, lab, args.supplement_n, args.seed, p)

    gates = {
        "step1_L1_chi2_p": cmp_res["L1_chi2_p"],
        "step1_L2_chi2_p": cmp_res["L2_chi2_p"],
        "step1_L1_abs_pp_over_threshold": cmp_res["L1_labels_over_pp_threshold"],
        "step2_empty_l2_warn_high": empty_res["warn_high_vs_validation"],
        "step2_empty_l2_warn_low": empty_res["warn_low_vs_validation"],
        "step3_slice_anomaly_count": len(slice_res["slice_anomalies"]),
        "step4_joint_flagged_cells": joint_res.get("n_flagged_unique", 0),
    }
    gates["pass_suggested"] = (
        not cmp_res["L1_labels_over_pp_threshold"]
        and not empty_res["warn_high_vs_validation"]
        and not empty_res["warn_low_vs_validation"]
        and len(slice_res["slice_anomalies"]) <= args.max_slice_anomalies
        and joint_res.get("n_flagged_unique", 0) <= args.max_joint_flags
    )

    stability_path = (
        Path(args.stability_summary).resolve()
        if args.stability_summary
        else None
    )
    step5_ok = None
    if stability_path is not None:
        if stability_path.is_file():
            st = json.loads(stability_path.read_text(encoding="utf-8"))
            l1_ok = float(st.get("l1_full_agreement_rate", 0)) >= 0.85
            l2_ok = float(st.get("l2_rows_min_jacc_ge_threshold_rate", 0)) >= 0.80
            step5_ok = l1_ok and l2_ok
            gates["step5_stability_from_file"] = str(stability_path)
            gates["step5_pass"] = step5_ok
        else:
            gates["pass_suggested"] = False
            gates["step5_stability_from_file"] = str(stability_path)
            gates["step5_pass"] = None
            gates["step5_note"] = "stability_summary path not found — run `stability` subcommand first"
    else:
        gates["step5_stability_from_file"] = None
        gates["step5_pass"] = None

    allow_desc = args.force_descriptive or (
        gates["pass_suggested"] and (gates["step5_pass"] is True or args.allow_descriptive_without_stability)
    )

    audit_report = {
        "corpus_csv": str(corpus_path),
        "validation_csv": str(val_path),
        "gates": gates,
        "compare": {k: v for k, v in cmp_res.items() if k != "L1_labels_over_pp_threshold"},
        "empty_l2": empty_res,
        "slice_anomalies_head": slice_res["slice_anomalies"][:50],
        "joint": {
            k: v
            for k, v in joint_res.items()
            if k not in ("low_prior_cells", "flag_df")
        },
        "publication_bundle": {
            "run_manifest": "run_manifest.json",
            "methods": "METHODS_AUDIT.md",
            "manuscript_tables": "tables_for_manuscript/",
            "comparison_basis": "Model predictions in both CSVs (final_results); not human gold unless you merged separately.",
        },
        "descriptive_emitted": allow_desc,
    }
    (out_dir / "audit_summary.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    md_lines = [
        "# Corpus label audit (Stage 10)",
        "",
        f"- Corpus: `{corpus_path}`",
        f"- Validation: `{val_path}`",
        "",
        "## Gate summary",
        "",
        f"- **Suggested pass** (automated heuristics only): `{gates['pass_suggested']}`",
        "- L1 labels with |Δpp| > threshold vs validation: "
        + (", ".join(cmp_res["L1_labels_over_pp_threshold"]) or "(none)"),
        f"- L1 χ² p-value (display): {_format_p_for_manuscript(float(cmp_res['L1_chi2_p'])) if cmp_res.get('L1_chi2_p') == cmp_res.get('L1_chi2_p') else 'NA'}",
        f"- L2 row-prevalence χ² p-value (display): {_format_p_for_manuscript(float(cmp_res['L2_chi2_p'])) if cmp_res.get('L2_chi2_p') == cmp_res.get('L2_chi2_p') else 'NA'}",
        f"- Empty L2: corpus {c_stats['empty_l2_rate']:.1%}, validation {v_stats['empty_l2_rate']:.1%}",
        f"- Slice anomalies flagged: {len(slice_res['slice_anomalies'])}",
        f"- Joint low-prior cells: {joint_res.get('n_flagged_unique', 0)}",
        "",
        "Interpretation is left to the researcher; any failed gate should be investigated before descriptive stats.",
        "",
        "## Files",
        "",
        "- `run_manifest.json` — argv, input SHA-256, environment",
        "- `METHODS_AUDIT.md` — definitions and comparison basis",
        "- `tables_for_manuscript/` — rounded CSVs for supplementary tables",
        "- `compare_L1_validation_vs_corpus.csv`, `compare_L2_row_prev_validation_vs_corpus.csv` (full precision)",
        "- `slice_prevalence_summary.csv`, `joint_L1_x_L2_*_crosstab_counts.csv`",
        "- `audit_summary.json`",
        "",
    ]
    (out_dir / "audit_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    thresholds_md = {
        "l1_pp_threshold": float(args.l1_pp_threshold),
        "l1_dominance": float(args.l1_dominance),
        "l2_skew_ratio": float(args.l2_skew_ratio),
        "empirical_joint_min_corpus": int(args.empirical_joint_min_corpus),
    }
    write_run_manifest(
        out_dir,
        command="audit",
        argv=argv_snapshot,
        corpus_path=corpus_path,
        validation_path=val_path,
        extra={
            "started_utc": started_utc,
            "thresholds": thresholds_md,
            "gates": gates,
            "force_descriptive": args.force_descriptive,
            "allow_descriptive_without_stability": args.allow_descriptive_without_stability,
            "descriptive_emitted": allow_desc,
        },
    )
    write_methods_audit_md(
        out_dir,
        corpus_path=corpus_path,
        validation_path=val_path,
        thresholds=thresholds_md,
        scipy_available=chi2_contingency is not None,
    )
    export_tables_for_manuscript(
        out_dir,
        cmp_res=cmp_res,
        c_stats=c_stats,
        v_stats=v_stats,
        slice_df=slice_res.get("slice_df", pd.DataFrame()),
        flag_df=joint_res.get("flag_df", pd.DataFrame()),
        empty_res=empty_res,
    )

    if allow_desc:
        export_descriptive(corpus_df, out_dir)
        logger.info("Wrote descriptive tables under %s/06_descriptive", out_dir)
    else:
        logger.info(
            "Skipped descriptive export (06_descriptive). "
            "Fix gates or pass --force-descriptive / --allow-descriptive-without-stability."
        )

    print(f"Audit complete → {out_dir}")


def main() -> None:
    base_dir = Path(os.getcwd())
    defs = _defaults_batch(base_dir)
    ap = argparse.ArgumentParser(
        description="Stage 10: Corpus label quality audit before descriptive statistics.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_a = sub.add_parser("audit", help="Steps 1–4 (+6 if gates pass)")
    p_a.add_argument("--corpus-csv", default=None, help="Full corpus final_results.csv (default: latest corpus run)")
    p_a.add_argument("--validation-csv", default=None, help="Validation final_results.csv (default: latest validation run)")
    p_a.add_argument("--out-dir", required=True, help="Output directory for all audit artifacts")
    p_a.add_argument("--l1-pp-threshold", type=float, default=10.0, help="|percentage points| L1 gap → supplement sample")
    p_a.add_argument("--supplement-n", type=int, default=75, help="Rows to sample per over-threshold L1 (50–100)")
    p_a.add_argument("--l1-dominance", type=float, default=0.75, help="Flag slice if top L1 share ≥ this")
    p_a.add_argument("--l2-skew-ratio", type=float, default=4.0, help="Heuristic: max L2 prev / mean(other) in slice")
    p_a.add_argument("--slice-sample-n", type=int, default=25, help="Random rows per anomalous slice")
    p_a.add_argument("--joint-sample-n", type=int, default=25, help="Max rows sampled per flagged L1×L2 cell")
    p_a.add_argument("--empirical-joint-min-corpus", type=int, default=20, help="Corpus joint count flag if validation=0")
    p_a.add_argument("--low-prior-pairs-csv", default=None, help="Optional CSV with L1,L2 columns")
    p_a.add_argument("--seed", type=int, default=42)
    p_a.add_argument(
        "--stability-summary",
        default=None,
        help="Path to stability_summary.json from `stability`; required for step5_pass in gates",
    )
    p_a.add_argument(
        "--allow-descriptive-without-stability",
        action="store_true",
        help="Emit 06_descriptive if steps 1–4 pass even when stability JSON absent",
    )
    p_a.add_argument("--force-descriptive", action="store_true", help="Always write 06_descriptive")
    p_a.add_argument(
        "--max-slice-anomalies",
        type=int,
        default=9999,
        help="Gate: fail if more than this many slice rows are flagged (default: effectively off)",
    )
    p_a.add_argument(
        "--max-joint-flags",
        type=int,
        default=9999,
        help="Gate: fail if more than this many joint cells flagged (default: effectively off)",
    )
    p_a.set_defaults(func=cmd_audit)

    p_d = sub.add_parser("describe", help="Step 6 only: descriptive tables → 06_descriptive/")
    p_d.add_argument("--corpus-csv", required=True)
    p_d.add_argument("--out-dir", required=True)
    p_d.set_defaults(func=cmd_describe)

    p_s = sub.add_parser("stability", help="Step 5: triplicate labeling via Chat Completions API")
    p_s.add_argument("--corpus-csv", default=None)
    p_s.add_argument("--out-dir", required=True)
    p_s.add_argument("--n-sample", type=int, default=100)
    p_s.add_argument("--temperature", type=float, default=0.2)
    p_s.add_argument("--l2-jaccard-threshold", type=float, default=0.80)
    p_s.add_argument("--model", default="gpt-5.1")
    p_s.add_argument("--seed", type=int, default=42)
    p_s.add_argument("--sleep-s", type=float, default=0.15, help="Pause between rows (rate limits)")
    p_s.add_argument("--no-hard-pool", action="store_true")
    p_s.add_argument("--hard-pool", default=defs["hard"])
    p_s.add_argument("--hard-pool-n-l1", type=int, default=4)
    p_s.add_argument("--hard-pool-n-l2", type=int, default=5)
    p_s.add_argument("--hard-pool-seed", type=int, default=None)
    p_s.set_defaults(func=cmd_stability)

    args = ap.parse_args()
    if args.command == "stability" and not args.corpus_csv:
        r = resolve_corpus_final_results(base_dir)
        if not r:
            raise SystemExit("pass --corpus-csv or ensure latest corpus run exists")
        args.corpus_csv = str(r)

    args.func(args)


if __name__ == "__main__":
    main()
