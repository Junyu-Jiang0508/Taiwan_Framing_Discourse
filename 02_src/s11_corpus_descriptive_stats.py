#!/usr/bin/env python3
"""Stage 11 — Manuscript-oriented descriptive statistics (sentence-level units).

Produces Tables 1–5 and Figures 1–4 per study protocol (no significance tests;
optional Cramér's V effect size only). Reuses parsing / candidate inference from
``s10_corpus_label_audit.py``.

Usage::

    cd /path/to/project && python 02_src/s11_corpus_descriptive_stats.py \\
        --corpus-csv 03_outputs/.../final_results.csv \\
        --out-dir 03_outputs/01_results_labelings/05_descriptive_stats/my_run_001 \\
        [--time-bin week|biweek] [--figure4]

Defaults for ``--corpus-csv`` resolve ``latest_merged_deduped_run.txt`` /
``latest_corpus_run.txt`` like Stage 10.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Load s10 helpers (module filename starts with a digit) ─────────────────
_S10 = None


def _s10():
    global _S10
    if _S10 is None:
        p = Path(__file__).resolve().parent / "s10_corpus_label_audit.py"
        spec = importlib.util.spec_from_file_location("s10_corpus_label_audit", p)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load s10_corpus_label_audit.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _S10 = mod
    return _S10


def _load_label_orders(base_dir: Path) -> Tuple[List[str], List[str]]:
    l1_path = base_dir / "01_data/05_labels_guidance/01_annotation_guide_label1_v9.csv"
    l2_path = base_dir / "01_data/05_labels_guidance/02_annotation_guide_label2_v10.csv"
    l1_ids = pd.read_csv(l1_path, encoding="utf-8-sig")["label_id"].astype(str).tolist()
    l2_ids = pd.read_csv(l2_path, encoding="utf-8-sig")["label_id"].astype(str).tolist()
    return l1_ids, l2_ids


def _resolve_corpus_csv(base_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).resolve()
    s10 = _s10()
    r = s10.resolve_corpus_final_results(base_dir)
    if not r:
        raise SystemExit(
            "Could not resolve corpus final_results.csv; pass --corpus-csv or set latest pointer."
        )
    return r


def source_outlet(raw: Any) -> str:
    """Map raw ``source`` / file key to a short outlet family for Table 1 (English labels)."""
    s0 = str(raw) if raw is not None and not (isinstance(raw, float) and np.isnan(raw)) else ""
    sl = s0.lower()
    if "chinatimes" in sl:
        return "China Times"
    if "ltn" in sl or "_ltn" in sl:
        return "LTN (Liberty Times)"
    if "tvbs" in sl:
        return "TVBS"
    if "辩论" in s0 or "debate" in sl:
        return "Televised debate"
    if "meeting" in sl:
        return "Policy forum (televised)"
    if "x_datasets" in sl:
        return "X (social media)"
    if "专访" in s0:
        return "Candidate interview"
    if s0.strip() == "":
        return "(blank)"
    return s0


def election_phase(dt: pd.Timestamp) -> str:
    """
    2024 Taiwan presidential election phases.
    Legal anchors:
      - CEC registration: Nov 20–24, 2023
      - Official campaign period (§41 PERLA): Dec 16, 2023 – Jan 12, 2024
      - Blackout day: Jan 12, 2024
      - Election day: Jan 13, 2024
    """
    if pd.isna(dt):
        return "No valid date"
    d = dt.normalize()
    if d < pd.Timestamp("2023-07-01"):
        return "Primary season"
    if d < pd.Timestamp("2023-11-25"):   # after CEC registration closes (Nov 24)
        return "Nomination period"
    if d < pd.Timestamp("2023-12-16"):   # before legal campaign period starts
        return "Post-registration interregnum"
    if d < pd.Timestamp("2024-01-12"):   # legal campaign period
        return "General campaign"
    if d <= pd.Timestamp("2024-01-13"):  # blackout + election day
        return "Final sprint / Election day"
    return "Post-election"


def sentence_char_len(s: Any) -> int:
    if pd.isna(s):
        return 0
    return len(str(s))


def _configure_matplotlib_fonts() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Microsoft JhengHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "Source Han Sans TC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# Internal corpus keys (from s10 inference) → English labels for tables / figures
CANDIDATE_EN: Dict[str, str] = {
    "赖清德": "Lai (DPP)",
    "侯友宜": "Hou (KMT)",
    "柯文哲": "Ko (TPP)",
}


def candidate_figure_label(name: str) -> str:
    return CANDIDATE_EN.get(str(name), str(name))


def candidate_table_slug(name: str) -> str:
    """Short ASCII slug for CSV column prefixes."""
    return {"赖清德": "Lai", "侯友宜": "Hou", "柯文哲": "Ko"}.get(str(name), str(name))


def _fmt_pct(x: float) -> float:
    return round(float(x) * 100.0, 1)


def _cramers_v(contingency: np.ndarray) -> Optional[float]:
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return None
    contingency = np.asarray(contingency, dtype=float)
    if contingency.size == 0 or contingency.sum() == 0:
        return None
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum()
    r, c = contingency.shape
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(chi2 / denom))


def table1_corpus_profile(df: pd.DataFrame) -> pd.DataFrame:
    s10 = _s10()
    work = df.copy()
    work["_outlet"] = work["source"].map(source_outlet)
    if "candidate" not in work.columns:
        work = s10.add_derived_columns(work)
    work["_dt"] = pd.to_datetime(work["date"], errors="coerce")
    work["_phase"] = work["_dt"].map(election_phase)
    rows = []
    for (outlet, cand, ph), g in work.groupby(["_outlet", "candidate", "_phase"], dropna=False):
        n = len(g)
        mean_len = g["sentence"].map(sentence_char_len).mean() if n else 0.0
        l2_nonempty = (~g["L2_labels"].map(s10.l2_is_empty)).mean() if n else 0.0
        cand_disp = CANDIDATE_EN.get(str(cand), str(cand))
        rows.append(
            {
                "source_outlet": outlet,
                "candidate": cand_disp,
                "election_phase": ph,
                "n_units": n,
                "mean_sentence_chars": round(float(mean_len), 2),
                "l2_nonempty_rate_pct": _fmt_pct(l2_nonempty),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["source_outlet", "candidate", "election_phase"]
    )


def table2_l1_by_candidate(
    df: pd.DataFrame, l1_order: Sequence[str], candidates: Sequence[str]
) -> pd.DataFrame:
    s10 = _s10()
    work = s10.add_derived_columns(df.copy())
    m = s10.valid_l1_mask(work) & work["candidate"].isin(list(candidates))
    sub = work.loc[m].copy()
    sub["L1"] = sub["L1_label"].astype(str).str.strip()
    out = pd.DataFrame({"L1": list(l1_order)})
    for cand in candidates:
        slug = candidate_table_slug(cand)
        g = sub[sub["candidate"] == cand]
        n = len(g)
        vc = g["L1"].value_counts()
        counts = [int(vc.get(lab, 0)) for lab in l1_order]
        out[f"n_{slug}"] = counts
        out[f"pct_within_{slug}_pct"] = [_fmt_pct(c / n) if n else 0.0 for c in counts]
    out["n_total_three_candidates"] = out[[f"n_{candidate_table_slug(c)}" for c in candidates]].sum(axis=1)
    return out


def figure1_l1_bars(df: pd.DataFrame, l1_order: Sequence[str], candidates: Sequence[str], out_path: Path) -> None:
    _configure_matplotlib_fonts()
    import matplotlib.pyplot as plt

    s10 = _s10()
    work = s10.add_derived_columns(df.copy())
    m = s10.valid_l1_mask(work) & work["candidate"].isin(list(candidates))
    sub = work.loc[m]
    mat = []
    for cand in candidates:
        g = sub[sub["candidate"] == cand]
        n = len(g)
        row = []
        for lab in l1_order:
            row.append((g["L1_label"].astype(str).str.strip() == lab).sum() / n if n else 0.0)
        mat.append(row)
    arr = np.array(mat).T  # shape (n_l1, n_cand)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(candidates))
    bottoms = np.zeros(len(candidates))
    tab_colors = plt.cm.tab10.colors
    for i, lab in enumerate(l1_order):
        ax.bar(x, arr[i], bottom=bottoms, label=lab, color=tab_colors[i % 10])
        bottoms += arr[i]
    ax.set_xticks(x)
    ax.set_xticklabels([candidate_figure_label(c) for c in candidates])
    ax.set_ylabel("Share (within candidate)")
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="L1 code")
    ax.set_title("Figure 1. L1 distribution by candidate (stacked proportions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _daily_l1_shares_for_trend(
    df: pd.DataFrame,
    l1_focus: Sequence[str],
    candidates: Sequence[str],
) -> pd.DataFrame:
    s10 = _s10()
    work = s10.add_derived_columns(df.copy())
    m = s10.valid_l1_mask(work) & work["candidate"].isin(list(candidates))
    sub = work.loc[m].copy()
    sub["L1"] = sub["L1_label"].astype(str).str.strip()
    sub["_day"] = pd.to_datetime(sub["date"], errors="coerce").dt.normalize()
    sub = sub[sub["_day"].notna()]
    rows = []
    for day, g in sub.groupby("_day"):
        for cand in candidates:
            gc = g[g["candidate"] == cand]
            n = len(gc)
            if n == 0:
                continue
            for lab in l1_focus:
                rows.append(
                    {
                        "day": day,
                        "candidate": cand,
                        "L1": lab,
                        "share": float((gc["L1"] == lab).sum() / n),
                        "n": n,
                    }
                )
    return pd.DataFrame(rows)


def _apply_7day_ma(daily_long: pd.DataFrame, label_key: str) -> pd.DataFrame:
    if daily_long.empty:
        return daily_long
    out_parts = []
    for (cand, lab), g in daily_long.groupby(["candidate", label_key]):
        g = g.sort_values("day").set_index("day")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        s = g["share"].reindex(idx).fillna(0.0)
        smooth = s.rolling(window=7, min_periods=1).mean()
        out_parts.append(
            pd.DataFrame(
                {
                    "day": smooth.index,
                    "candidate": cand,
                    label_key: lab,
                    "share_smooth": smooth.values,
                }
            )
        )
    return pd.concat(out_parts, ignore_index=True)


def _bin_time_series(daily_ma: pd.DataFrame, rule: str, label_key: str) -> pd.DataFrame:
    """Aggregate daily smoothed shares to week / biweek (period end label)."""
    if daily_ma.empty:
        return daily_ma
    daily_ma = daily_ma.copy()
    daily_ma["_p"] = daily_ma["day"].dt.to_period("W-SUN" if rule == "week" else "2W-SUN")
    agg = (
        daily_ma.groupby(["_p", "candidate", label_key], as_index=False)["share_smooth"]
        .mean()
        .rename(columns={"_p": "period"})
    )
    agg["period_end"] = agg["period"].astype(str)
    return agg


def figure2_l1_trend_panels(
    df: pd.DataFrame,
    l1_panels: Sequence[str],
    candidates: Sequence[str],
    out_path: Path,
    time_bin: str,
) -> pd.DataFrame:
    _configure_matplotlib_fonts()
    import matplotlib.pyplot as plt

    l1_en = {
        "L1-01": "Economic",
        "L1-03": "Conflict & security",
        "L1-06": "Governance",
        "L1-07": "Sentiment / emotion",
    }

    daily = _daily_l1_shares_for_trend(df, l1_panels, candidates)
    ma = _apply_7day_ma(daily, "L1")
    binned = _bin_time_series(ma, "week" if time_bin == "week" else "biweek", "L1")
    binned.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    axes_flat = axes.flatten()
    styles = ["-", "--", "-."]
    for ax, lab in zip(axes_flat, l1_panels):
        for i, cand in enumerate(candidates):
            sub = binned[(binned["L1"] == lab) & (binned["candidate"] == cand)].sort_values("period")
            if sub.empty:
                continue
            x = np.arange(len(sub))
            ax.plot(
                x,
                sub["share_smooth"],
                linestyle=styles[i % len(styles)],
                label=candidate_figure_label(cand),
                linewidth=1.4,
            )
        ax.set_title(f"{lab}: {l1_en.get(lab, lab)}")
        ax.set_ylabel("Smoothed share")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        tick_step = max(1, len(sub) // 6)
        ax.set_xticks(np.arange(0, len(sub), tick_step))
    axes_flat[0].legend(fontsize=8, title="Candidate")
    for ax in axes_flat[-2:]:
        ax.set_xlabel("Time bin index (see companion CSV for period labels)")
    bin_word = "weekly" if time_bin == "week" else "biweekly"
    fig.suptitle(
        f"Figure 2. L1 share over time ({bin_word} bins; 7-day MA of daily shares, then bin mean)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return binned


def table3_l2_prevalence_nonempty(
    df: pd.DataFrame, l2_order: Sequence[str], candidates: Sequence[str]
) -> pd.DataFrame:
    s10 = _s10()
    work = s10.add_derived_columns(df.copy())
    m = s10.valid_l1_mask(work) & work["candidate"].isin(list(candidates))
    sub = work.loc[m].copy()
    rows = []
    for cand in candidates:
        g = sub[sub["candidate"] == cand]
        nonempty = g[~g["L2_labels"].map(s10.l2_is_empty)]
        denom = len(nonempty)
        for lab in l2_order:
            hit = sum(lab in s10.parse_l2_cell(r["L2_labels"]) for _, r in nonempty.iterrows())
            rows.append(
                {
                    "candidate": CANDIDATE_EN.get(str(cand), str(cand)),
                    "L2": lab,
                    "n_nonempty_l2_units": denom,
                    "n_with_label": int(hit),
                    "prevalence_pct": _fmt_pct(hit / denom) if denom else 0.0,
                }
            )
    return pd.DataFrame(rows)


def l2_cooccurrence_matrix(df: pd.DataFrame, l2_order: Sequence[str]) -> pd.DataFrame:
    s10 = _s10()
    work = s10.add_derived_columns(df.copy())
    m = s10.valid_l1_mask(work)
    sub = work.loc[m]
    labels = list(l2_order)
    M = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for _, r in sub.iterrows():
        labs = sorted(set(s10.parse_l2_cell(r["L2_labels"])) & set(labels))
        for a in labs:
            M.loc[a, a] += 1
        for a, b in combinations(labs, 2):
            M.loc[a, b] += 1
            M.loc[b, a] += 1
    return M


def figure3_l2_heatmap(M: pd.DataFrame, out_path: Path) -> None:
    _configure_matplotlib_fonts()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    arr = M.values.astype(float)
    im = ax.imshow(arr, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(M.columns)))
    ax.set_yticks(np.arange(len(M.index)))
    ax.set_xticklabels(M.columns, rotation=45, ha="right")
    ax.set_yticklabels(M.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")
    ax.set_title("Figure 3. L2 co-occurrence (counts; diagonal = units with label)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def table5_l1_l2_prevalence(df: pd.DataFrame, l1_order: Sequence[str], l2_order: Sequence[str]) -> pd.DataFrame:
    s10 = _s10()
    work = s10.add_derived_columns(df.copy())
    m = s10.valid_l1_mask(work)
    sub = work.loc[m].copy()
    sub["L1"] = sub["L1_label"].astype(str).str.strip()
    rows = []
    for l1 in l1_order:
        g = sub[sub["L1"] == l1]
        n = len(g)
        row: Dict[str, Any] = {"L1": l1, "n_units_l1_row": n}
        for l2 in l2_order:
            if n == 0:
                row[f"prev_L2_{l2}_pct"] = 0.0
            else:
                hit = sum(l2 in s10.parse_l2_cell(r["L2_labels"]) for _, r in g.iterrows())
                row[f"prev_L2_{l2}_pct"] = _fmt_pct(hit / n)
        rows.append(row)
    return pd.DataFrame(rows)


def figure4_l2_trends(
    df: pd.DataFrame,
    l2_focus: Sequence[str],
    candidates: Sequence[str],
    out_path: Path,
    time_bin: str,
) -> None:
    _configure_matplotlib_fonts()
    import matplotlib.pyplot as plt

    s10 = _s10()
    work = s10.add_derived_columns(df.copy())
    m = s10.valid_l1_mask(work) & work["candidate"].isin(list(candidates))
    sub = work.loc[m].copy()
    sub["_day"] = pd.to_datetime(sub["date"], errors="coerce").dt.normalize()
    sub = sub[sub["_day"].notna()]
    daily_rows = []
    for day, g in sub.groupby("_day"):
        for cand in candidates:
            gc = g[g["candidate"] == cand]
            n = len(gc)
            if n == 0:
                continue
            for lab in l2_focus:
                hit = sum(lab in s10.parse_l2_cell(r["L2_labels"]) for _, r in gc.iterrows())
                daily_rows.append(
                    {"day": day, "candidate": cand, "L2": lab, "share": hit / n, "n": n}
                )
    daily_long = pd.DataFrame(daily_rows)
    ma = _apply_7day_ma(daily_long, "L2") if not daily_long.empty else daily_long
    binned = _bin_time_series(ma, "week" if time_bin == "week" else "biweek", "L2")
    binned.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, len(l2_focus), figsize=(5 * len(l2_focus), 3.8), squeeze=False)
    styles = ["-", "--", "-."]
    for j, (ax, lab) in enumerate(zip(axes[0], l2_focus)):
        for i, cand in enumerate(candidates):
            subp = binned[(binned["L2"] == lab) & (binned["candidate"] == cand)].sort_values("period")
            if subp.empty:
                continue
            x = np.arange(len(subp))
            ax.plot(
                x,
                subp["share_smooth"],
                linestyle=styles[i % len(styles)],
                label=candidate_figure_label(cand),
                linewidth=1.4,
            )
        l2_en = {"L2-05": "Shared threat", "L2-07": "Solidarity & mobilization"}
        ax.set_title(f"{lab}: {l2_en.get(lab, lab)}")
        if j == 0:
            ax.set_ylabel("Smoothed share (row prevalence)")
        ax.set_xlabel("Time bin index (see companion CSV)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8, title="Candidate")
    bin_word = "weekly" if time_bin == "week" else "biweekly"
    fig.suptitle(
        f"Figure 4. L2 prevalence over time ({bin_word} bins; 7-day MA, then bin mean)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_captions(
    path: Path,
    *,
    n_all: int,
    n_l1_valid: int,
    n_l2_nonempty: int,
    candidates: Sequence[str],
    time_bin: str,
) -> None:
    cand_s = ", ".join(CANDIDATE_EN.get(c, c) for c in candidates)
    bin_word = "weekly" if time_bin == "week" else "biweekly"
    text = f"""# Figure and table notes (draft captions)

**Unit of analysis:** sentence-level unit (same as the corpus segmentation and labeling pipeline).

- **N (all rows in corpus CSV):** {n_all}
- **N with non-empty L1:** {n_l1_valid} (subset used for L1 distributions, time trends, and L1×L2 joint prevalence)
- **N with non-empty L2 (among L1-valid rows):** {n_l2_nonempty} (Table 3 denominators are per-candidate non-empty-L2 counts, see table)
- **Candidate subset:** {cand_s} (rows with other inferred candidates, e.g. Unknown, excluded from Tables 2–3 and figures)

## Table 1 — Corpus profile

By mapped outlet family, candidate (English label), and election phase (**Primary season** / **Nomination period** / **General campaign** / **Final sprint** / **Post-election / other** / **No valid date**): number of units, mean sentence length (characters), and L2 non-empty rate (%). Phase boundaries: 2023-07-01, 2023-11-20 (CEC presidential registration window start), 2024-01-01, 2024-01-13 (polling day). Post-election includes news after polling day and rows without a parseable date.

## Table 2 & Figure 1 — L1 frequency by candidate

Raw counts for seven L1 categories plus **column percentages** (within-candidate L1 composition, not row percentages). Figure 1: stacked bars by candidate, fill = L1 code.

## Figure 2 — L1 over time

Panels for L1-01, L1-03, L1-06, L1-07. Daily shares → **7-day moving average** → mean of smoothed values within each **{bin_word}** bin; line style distinguishes candidates. Companion CSV lists period labels.

## Table 3 — L2 prevalence (multi-label)

Among units with **non-empty L2** for that candidate, the percentage of units carrying each L2 code. With multi-label rows, prevalences can sum to more than 100%.

## Table 4 & Figure 3 — L2 co-occurrence

8×8 matrix of **absolute** co-occurrence counts: diagonal = number of units containing that label; off-diagonal = units containing both labels. No φ or correlation coefficients in this descriptive stage.

## Table 5 — L1×L2 joint prevalence

Rows = L1, columns = L2; cell = among units with L1 = *i*, the percentage that also carry L2 = *j*.

## Figure 4 (optional) — Core L2 time trends

L2-05 and L2-07 (by default) over time by candidate; same smoothing and binning as Figure 2.

---

**Reporting:** percentages rounded to one decimal. **No** χ² or other significance tests at this descriptive stage.
"""
    path.write_text(text, encoding="utf-8")


def write_manifest(
    out_dir: Path,
    corpus_path: Path,
    argv: List[str],
    extra: Dict[str, Any],
) -> None:
    s10 = _s10()
    manifest: Dict[str, Any] = {
        "schema": "s11_descriptive_run_v1",
        "argv": argv,
        "cwd": os.getcwd(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "environment": s10._gather_environment(),
        "inputs": {"corpus_csv": str(corpus_path), "corpus_sha256": s10._sha256_file(corpus_path)},
        **extra,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    base_dir = Path(os.getcwd()).resolve()
    ap = argparse.ArgumentParser(description="Stage 11: Descriptive stats for manuscript tables/figures.")
    ap.add_argument("--corpus-csv", default=None, help="final_results.csv (default: latest corpus pointer)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--time-bin", choices=["week", "biweek"], default="week", help="Figure 2/4 time aggregation after MA")
    ap.add_argument(
        "--figure4",
        action="store_true",
        help="Emit Figure 4 (L2-05 / L2-07 time trends by candidate)",
    )
    args = ap.parse_args()

    corpus_path = _resolve_corpus_csv(base_dir, args.corpus_csv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    l1_order, l2_order = _load_label_orders(base_dir)
    candidates = ["赖清德", "侯友宜", "柯文哲"]

    df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    s10 = _s10()
    n_all = len(df)
    work = s10.add_derived_columns(df.copy())
    m_l1 = s10.valid_l1_mask(work)
    n_l1_valid = int(m_l1.sum())
    n_l2_nonempty = int((~work.loc[m_l1, "L2_labels"].map(s10.l2_is_empty)).sum())

    t1 = table1_corpus_profile(df)
    t1.to_csv(tables / "table1_corpus_profile.csv", index=False, encoding="utf-8-sig")

    t2 = table2_l1_by_candidate(df, l1_order, candidates)
    t2.to_csv(tables / "table2_L1_freq_by_candidate.csv", index=False, encoding="utf-8-sig")

    fig1 = figures / "fig1_L1_stacked_by_candidate.png"
    figure1_l1_bars(df, l1_order, candidates, fig1)

    l1_panels = ["L1-01", "L1-03", "L1-06", "L1-07"]
    figure2_l1_trend_panels(df, l1_panels, candidates, figures / "fig2_L1_time_trends.png", args.time_bin)

    t3 = table3_l2_prevalence_nonempty(df, l2_order, candidates)
    t3.to_csv(tables / "table3_L2_prevalence_nonempty_by_candidate.csv", index=False, encoding="utf-8-sig")

    M = l2_cooccurrence_matrix(df, l2_order)
    M.to_csv(tables / "table4_L2_cooccurrence_counts.csv", encoding="utf-8-sig")
    figure3_l2_heatmap(M, figures / "fig3_L2_cooccurrence_heatmap.png")

    t5 = table5_l1_l2_prevalence(df, l1_order, l2_order)
    t5.to_csv(tables / "table5_L1_L2_prevalence_joint.csv", index=False, encoding="utf-8-sig")

    if args.figure4:
        figure4_l2_trends(df, ["L2-05", "L2-07"], candidates, figures / "fig4_L2_core_trends.png", args.time_bin)

    # Cramér's V: L1 × candidate (no p-value)
    ct = pd.crosstab(
        work.loc[m_l1, "L1_label"].astype(str).str.strip(),
        work.loc[m_l1, "candidate"],
    )
    cand_cols = [c for c in candidates if c in ct.columns]
    ct_sub = ct.reindex(index=[x for x in l1_order if x in ct.index], columns=cand_cols, fill_value=0)
    cram = _cramers_v(ct_sub.values)
    effect = {"L1_by_candidate_cramers_v": cram, "note": "Effect size only; no χ² p-value reported here."}
    (out_dir / "effect_sizes.json").write_text(json.dumps(effect, ensure_ascii=False, indent=2), encoding="utf-8")

    write_captions(
        out_dir / "CAPTIONS.md",
        n_all=n_all,
        n_l1_valid=n_l1_valid,
        n_l2_nonempty=n_l2_nonempty,
        candidates=candidates,
        time_bin=args.time_bin,
    )

    write_manifest(
        out_dir,
        corpus_path,
        sys.argv[:],
        extra={
            "n_all_units": n_all,
            "n_l1_valid": n_l1_valid,
            "n_l2_nonempty_among_l1_valid": n_l2_nonempty,
            "time_bin": args.time_bin,
            "figure4_emitted": bool(args.figure4),
        },
    )
    logger.info("Wrote outputs under %s", out_dir)


if __name__ == "__main__":
    main()
