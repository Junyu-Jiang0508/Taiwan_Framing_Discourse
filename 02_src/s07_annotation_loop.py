"""Stage 7 — Iterative annotation analysis, gap sampling, merging.

Consolidates eight former scripts (06–13) into a single CLI with sub-commands:

* ``analyze``     — L1/L2 distribution + gap report
* ``sample``      — Smart gap-based sampling (Tier A/B/C or targeted round)
* ``prepare``     — Format sampled candidates for annotation input
* ``compare``     — Compare target_l1 vs actual annotated L1
* ``merge``       — Merge annotation batches + gap status

Usage::

    python s07_annotation_loop.py analyze  --input <annotations.csv>
    python s07_annotation_loop.py sample   --stats <stats.json> --tier C
    python s07_annotation_loop.py prepare  --input <candidates.csv> --output <ready.csv>
    python s07_annotation_loop.py compare  --anno <anno.csv> --cand <candidates.csv>
    python s07_annotation_loop.py merge    <csv1> <csv2> [<csv3> ...] --output <merged.csv>
"""
from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import L1_SAMPLE_KEYWORDS, L2_SAMPLE_KEYWORDS


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_refined_datasets(refined_dir: str = "01_data/04_refined_datasets") -> pd.DataFrame:
    """Load all *_units.csv from refined datasets (excl. CONTEXT/DROPPED)."""
    root = Path(refined_dir)
    all_data: list[pd.DataFrame] = []
    for dataset in ("01_news_datasets", "02_conference_datasets", "03_X_datasets"):
        ddir = root / dataset
        if not ddir.exists():
            continue
        for csv_file in ddir.glob("*_units.csv"):
            if "_CONTEXT" in csv_file.name or "_DROPPED" in csv_file.name:
                continue
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig")
                df["source_file"] = csv_file.stem
                all_data.append(df)
                print(f"Loaded: {csv_file.name} ({len(df)})")
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Total refined samples: {len(combined)}")
        return combined
    return pd.DataFrame()


def _parse_l2(s) -> list[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def compute_gaps(
    df: pd.DataFrame,
    l1_target: int = 100,
    l2_target: int = 75,
) -> dict:
    l1_counts = df["L1_label"].value_counts()
    all_l2: list[str] = []
    for l2_str in df["L2_labels"].dropna():
        all_l2.extend(_parse_l2(l2_str))
    l2_counts = Counter(all_l2)

    l1_gaps = {f"L1-{i:02d}": max(0, l1_target - int(l1_counts.get(f"L1-{i:02d}", 0)))
               for i in range(1, 10)}
    l1_gaps = {k: v for k, v in l1_gaps.items() if v > 0}
    l2_gaps = {f"L2-{i:02d}": max(0, l2_target - int(l2_counts.get(f"L2-{i:02d}", 0)))
               for i in range(1, 16)}
    l2_gaps = {k: v for k, v in l2_gaps.items() if v > 0}
    avg_l2 = len(all_l2) / len(df) if len(df) else 1.8
    return {
        "l1_distribution": {k: int(v) for k, v in l1_counts.items()},
        "l2_distribution": {k: int(v) for k, v in l2_counts.items()},
        "l2_avg_per_sample": float(avg_l2),
        "l1_gaps": l1_gaps,
        "l2_gaps": l2_gaps,
        "total_l1_gap": sum(l1_gaps.values()),
        "total_l2_gap": sum(l2_gaps.values()),
    }


def print_gaps(gaps: dict) -> int:
    for label, gap in sorted(gaps["l1_gaps"].items(), key=lambda x: x[1], reverse=True):
        print(f"  L1 {label} gap={gap}")
    print(f"L1 total_gap={gaps['total_l1_gap']}")
    for label, gap in sorted(gaps["l2_gaps"].items(), key=lambda x: x[1], reverse=True):
        print(f"  L2 {label} gap={gap}")
    print(f"L2 total_gap={gaps['total_l2_gap']}")
    est = int(gaps["total_l2_gap"] / gaps["l2_avg_per_sample"]) if gaps["l2_avg_per_sample"] else gaps["total_l2_gap"]
    rec = max(gaps["total_l1_gap"], est)
    print(f"est_by_L2~{est} recommend+{rec}")
    return rec


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-command: analyze
# ═══════════════════════════════════════════════════════════════════════════

def cmd_analyze(args):
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    print(f"samples={len(df)}")
    error_count = df["error"].notna().sum() if "error" in df.columns else 0
    if error_count:
        print(f"errors={error_count}")

    l1_counts = df["L1_label"].value_counts()
    print("L1:")
    for label, count in l1_counts.items():
        if pd.notna(label):
            print(f"  {label} {count} {count / len(df) * 100:.1f}%")

    all_l2: list[str] = []
    l2_combinations: list[tuple] = []
    for l2_str in df["L2_labels"].dropna():
        labels = _parse_l2(l2_str)
        if labels:
            all_l2.extend(labels)
            l2_combinations.append(tuple(sorted(labels)))
    l2_counts = Counter(all_l2)
    print(f"L2 instances={len(all_l2)} avg/sample={len(all_l2) / len(df):.2f}")
    for label, count in sorted(l2_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label} {count} {count / len(df) * 100:.1f}%")
    combo_counts = Counter(l2_combinations)
    print("L2 top combos:")
    for combo, count in combo_counts.most_common(10):
        print(f"  {'+'.join(combo) if combo else 'None'} {count}")

    if "confidence" in df.columns:
        print("confidence:")
        for conf, count in df["confidence"].value_counts().items():
            if pd.notna(conf):
                print(f"  {conf} {count}")
    if "candidate" in df.columns:
        print("L1 x candidate:\n", pd.crosstab(df["L1_label"], df["candidate"], margins=True))

    gaps = compute_gaps(df, l1_target=args.l1_target, l2_target=args.l2_target)
    print_gaps(gaps)

    summary = {
        "total_samples": len(df),
        "error_count": int(error_count),
        "l1_distribution": {k: int(v) for k, v in l1_counts.items()},
        "l2_distribution": dict(l2_counts),
        "l2_avg_per_sample": len(all_l2) / len(df) if len(df) else 0,
        **gaps,
    }
    out_json = args.input.replace(".csv", "_statistics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_json}")


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-command: sample
# ═══════════════════════════════════════════════════════════════════════════

def _score_sample(row, l2_gaps: dict) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: list[str] = []
    text = str(row.get("sentence", row.get("text", "")))

    if "semantic_similarity" in row:
        sim = row["semantic_similarity"]
        if pd.notna(sim) and 0.45 <= float(sim) <= 0.60:
            score += 2.0; reasons.append("boundary_semantic")
    if "nli_score" in row:
        nli = row["nli_score"]
        if pd.notna(nli) and 0.05 <= float(nli) <= 0.20:
            score += 1.5; reasons.append("boundary_nli")

    for l2_label, keywords in L2_SAMPLE_KEYWORDS.items():
        if l2_label in l2_gaps and any(kw in text for kw in keywords):
            score += 1.0; reasons.append(f"l2_{l2_label}")

    if row.get("speakers"):
        score += 1.0; reasons.append("has_speaker")
    if row.get("targets"):
        score += 0.8; reasons.append("has_target")
    if row.get("proc_symbolic_flag"):
        score += 0.5; reasons.append("procedural")
    role = row.get("role", "")
    if role == "quote":
        score += 1.5; reasons.append("quote")
    elif role == "claim":
        score += 1.0; reasons.append("claim")
    if 50 <= len(text) <= 200:
        score += 0.5; reasons.append("good_length")
    return score, reasons


def cmd_sample(args):
    with open(args.stats, "r", encoding="utf-8") as f:
        stats = json.load(f)

    l1_gaps = stats.get("l1_gaps", {})
    l2_gaps = stats.get("l2_gaps", {})
    if not l1_gaps:
        print("No L1 gaps — nothing to sample."); return

    print(f"Loading refined datasets…")
    df = load_refined_datasets(args.refined)
    if df.empty:
        print("No data."); return

    sampled: list[pd.DataFrame] = []
    for label, quota in sorted(l1_gaps.items(), key=lambda x: x[1], reverse=True):
        print(f"\nSampling for {label}: need {quota}")
        keywords = L1_SAMPLE_KEYWORDS.get(label, [])
        if keywords:
            mask = df["sentence"].str.contains("|".join(keywords), case=False, na=False)
            candidates = df[mask].copy()
            if len(candidates) < quota * 3:
                rem = df[~mask]
                if len(rem):
                    candidates = pd.concat([candidates, rem.sample(n=min(quota * 3 - len(candidates), len(rem)))],
                                           ignore_index=True)
        else:
            candidates = df.sample(n=min(quota * 5, len(df)))

        if candidates.empty:
            print(f"  No candidates for {label}"); continue

        candidates["sample_score"] = candidates.apply(lambda r: _score_sample(r, l2_gaps)[0], axis=1)
        candidates["score_reasons"] = candidates.apply(lambda r: "|".join(_score_sample(r, l2_gaps)[1]), axis=1)
        candidates = candidates.sort_values("sample_score", ascending=False)

        oversample = 2.0 if quota <= 20 else (1.5 if quota <= 50 else 1.3)
        target_count = int(quota * oversample)
        selected, used_src, used_dt = [], [], []
        max_src = 3 if quota <= 50 else (7 if quota <= 80 else 10)
        max_dt = 2 if quota <= 50 else (4 if quota <= 80 else 5)
        for _, row in candidates.iterrows():
            if len(selected) >= target_count:
                break
            src = row.get("source_file", "")
            dt = row.get("date", "")
            if src and used_src.count(src) >= max_src:
                continue
            if dt and used_dt.count(dt) >= max_dt:
                continue
            selected.append(row)
            used_src.append(src)
            used_dt.append(dt)

        sel_df = pd.DataFrame(selected)
        sel_df["target_l1"] = label
        sel_df["sampling_reason"] = "gap_filling"
        sampled.append(sel_df)
        print(f"  Selected {len(selected)} ({len(selected)/quota:.1f}x quota)")

    if not sampled:
        print("No candidates."); return
    result = pd.concat(sampled, ignore_index=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"candidates_{args.tag}.csv"
    cols = [c for c in [
        "unit_id", "sentence", "role", "prev", "next", "speakers", "targets",
        "proc_symbolic_flag", "event_phase", "source", "date",
        "target_l1", "sample_score", "score_reasons", "sampling_reason",
    ] if c in result.columns]
    result[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nWrote {out_csv} ({len(result)} rows)")

    for label in result["target_l1"].unique():
        sub = result[result["target_l1"] == label]
        sub[cols].to_csv(out_dir / f"candidates_{label}_{args.tag}.csv", index=False, encoding="utf-8-sig")
        print(f"  {label}: {len(sub)}")

    summary = {
        "tag": args.tag, "total_candidates": len(result),
        "l1_gaps": l1_gaps, "l2_gaps": l2_gaps,
        "candidates_by_l1": result["target_l1"].value_counts().to_dict(),
    }
    sp = out_dir / f"sampling_summary_{args.tag}.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {sp}")


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-command: prepare
# ═══════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    print(f"candidates n={len(df)}")

    if "unit_id" in df.columns:
        df["id"] = df["unit_id"]
    else:
        df["id"] = range(1, len(df) + 1)

    def _infer_cp(source):
        source = str(source).lower()
        if "01_ho" in source or "hou" in source or "侯" in source:
            return "侯友宜", "KMT"
        if "02_ke" in source or "柯" in source:
            return "柯文哲", "TPP"
        if "03_lai" in source or "lai" in source or "赖" in source or "賴" in source:
            return "赖清德", "DPP"
        return "Unknown", "Unknown"

    if "candidate" not in df.columns or "party" not in df.columns:
        df[["candidate", "party"]] = df["source"].apply(lambda x: pd.Series(_infer_cp(x)))

    def _infer_st(source):
        source = str(source).lower()
        if "x_datasets" in source or "twitter" in source:
            return "social_media"
        if "meeting" in source or "conference" in source:
            return "conference"
        return "news"

    if "source_type" not in df.columns:
        df["source_type"] = df["source"].apply(_infer_st)
    if "date" not in df.columns:
        df["date"] = ""

    req = ["id", "candidate", "party", "source_type", "date", "sentence"]
    opt = ["role", "prev", "next", "speakers", "targets", "proc_symbolic_flag",
           "event_phase", "target_l1", "sample_score", "score_reasons", "unit_id", "source"]
    out_cols = req + [c for c in opt if c in df.columns]
    df[out_cols].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"wrote {args.output}")


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-command: compare
# ═══════════════════════════════════════════════════════════════════════════

def cmd_compare(args):
    df_anno = pd.read_csv(args.anno, encoding="utf-8-sig").drop_duplicates(subset="id", keep="first")
    df_cand = pd.read_csv(args.cand, encoding="utf-8-sig").drop_duplicates(subset="id", keep="first")
    merge_cols = ["id", "target_l1"]
    if "sample_score" in df_cand.columns:
        merge_cols.append("sample_score")
    if "score_reasons" in df_cand.columns:
        merge_cols.append("score_reasons")
    df = df_anno.merge(df_cand[merge_cols], on="id", how="inner")
    print(f"merged n={len(df)}")

    if "target_l1" not in df.columns or "L1_label" not in df.columns:
        print("Missing columns."); return

    df["match"] = df["target_l1"] == df["L1_label"]
    mc = int(df["match"].sum())
    print(f"L1 match {mc}/{len(df)} {mc / len(df) * 100:.1f}%")
    for t in sorted(df["target_l1"].unique()):
        tdf = df[df["target_l1"] == t]
        m = int(tdf["match"].sum())
        print(f"  target {t} n={len(tdf)} match={m} {m / len(tdf) * 100 if len(tdf) else 0:.1f}%")

    print("crosstab:\n", pd.crosstab(df["target_l1"], df["L1_label"], margins=True))

    mismatch_df = df[~df["match"]].copy()
    if len(mismatch_df):
        out_m = args.anno.replace(".csv", "_mismatch_analysis.csv")
        cols = [c for c in ["id", "target_l1", "L1_label", "confidence", "sentence",
                            "L1_reasoning", "L2_labels", "candidate", "party", "source_type",
                            "sample_score", "score_reasons"] if c in mismatch_df.columns]
        mismatch_df[cols].to_csv(out_m, index=False, encoding="utf-8-sig")
        print(f"wrote {out_m}")

    summary = {
        "total_samples": len(df), "matches": mc, "mismatches": len(df) - mc,
        "match_rate": float(mc / len(df) * 100) if len(df) else 0,
    }
    sp = args.anno.replace(".csv", "_comparison_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"wrote {sp}")


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-command: merge
# ═══════════════════════════════════════════════════════════════════════════

def cmd_merge(args):
    batches: list[pd.DataFrame] = []
    seen_ids: set = set()
    for i, path in enumerate(args.csvs):
        b = pd.read_csv(path, encoding="utf-8-sig").drop_duplicates(subset="id", keep="first")
        b["batch"] = Path(path).stem
        before = len(b)
        b = b[~b["id"].isin(seen_ids)]
        if before != len(b):
            print(f"  {Path(path).name}: dropped {before - len(b)} overlap ids (earlier batch wins)")
        seen_ids.update(b["id"])
        batches.append(b)
        print(f"Batch {i + 1}: {path} → {len(b)} rows")

    merged = pd.concat(batches, ignore_index=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"merged n={len(merged)} → {args.output}")

    gaps = compute_gaps(merged, l1_target=args.l1_target, l2_target=args.l2_target)
    print_gaps(gaps)

    summary = {
        "total_samples": len(merged),
        "batch_counts": {Path(p).stem: int(len(b)) for p, b in zip(args.csvs, batches)},
        **gaps,
    }
    stats_path = args.output.replace(".csv", "_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"wrote {stats_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 7: Iterative annotation loop — analyze, sample, prepare, compare, merge"
    )
    ap.add_argument("--l1-target", type=int, default=100, help="Per-label L1 sample target")
    ap.add_argument("--l2-target", type=int, default=75, help="Per-label L2 sample target")
    sub = ap.add_subparsers(dest="command", required=True)

    # analyze
    p_a = sub.add_parser("analyze", help="L1/L2 distribution + gap report")
    p_a.add_argument("--input", required=True, help="Annotations CSV")

    # sample
    p_s = sub.add_parser("sample", help="Gap-based smart sampling")
    p_s.add_argument("--stats", required=True, help="Statistics JSON (from analyze)")
    p_s.add_argument("--refined", default="01_data/04_refined_datasets", help="Refined datasets root")
    p_s.add_argument("--output-dir", default="01_data/06_manual_sets/02_sampling_candidates")
    p_s.add_argument("--tag", default="round", help="Tag for output filenames (e.g. tierC, round2, round3)")

    # prepare
    p_p = sub.add_parser("prepare", help="Format candidates for annotation")
    p_p.add_argument("--input", required=True, help="Candidates CSV")
    p_p.add_argument("--output", required=True, help="Output ready CSV")

    # compare
    p_c = sub.add_parser("compare", help="Compare target_l1 vs actual L1")
    p_c.add_argument("--anno", required=True, help="Annotations CSV")
    p_c.add_argument("--cand", required=True, help="Candidates CSV (with target_l1)")

    # merge
    p_m = sub.add_parser("merge", help="Merge annotation batches (earlier batch wins on id overlap)")
    p_m.add_argument("csvs", nargs="+", help="CSV files to merge (priority order: first wins)")
    p_m.add_argument("--output", required=True, help="Output merged CSV")

    args = ap.parse_args()
    dispatch = {
        "analyze": cmd_analyze,
        "sample": cmd_sample,
        "prepare": cmd_prepare,
        "compare": cmd_compare,
        "merge": cmd_merge,
    }
    try:
        dispatch[args.command](args)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
