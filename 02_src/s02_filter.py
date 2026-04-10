"""Stage 2 — Apply TextFilterStack then refine REVIEW subset.

Merges the former ``02_wrangling_units.py`` (BERT / NLI filtering) and
``03_post_filter_review.py`` (REVIEW re-classification into KEEP / CONTEXT / DROP).

Usage::

    python s02_filter.py filter            # run semantic + NLI filtering
    python s02_filter.py post-review       # refine REVIEW samples
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import pandas as pd

from config import (
    ACTOR_TERMS,
    DATASETS,
    FILTERED_DIR,
    INST_TERMS,
    PROC_SYMBOLIC_TERMS,
    PROCESSED_DIR,
    REFINED_DIR,
    SCARCE_L1_L2,
    STRONG_BLACKLIST,
    TAIWAN_MARKERS,
    has_actor,
    has_blacklist,
    has_inst,
    has_proc_symbolic,
    has_scarce,
    has_taiwan_context,
)

# Dynamically load TextFilterStack from utils_wrangling_bert
_spec = importlib.util.spec_from_file_location(
    "wrangling_bert", Path(__file__).resolve().parent / "utils_wrangling_bert.py"
)
_wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wr)
TextFilterStack = _wr.TextFilterStack


# ═══════════════════════════════════════════════════════════════════════════
#  Part A — Semantic + NLI Filtering  (formerly 02_wrangling_units.py)
# ═══════════════════════════════════════════════════════════════════════════

def process_units_file(
    input_file: str,
    output_file: str,
    filter_stack: TextFilterStack,
    save_all_categories: bool = True,
):
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    print(f"  in n={len(df)} roles={df['role'].value_counts().to_dict()}")

    result_df, stats = filter_stack.filter_pipeline(
        df, text_column="sentence", use_bm25=False, use_semantic=True, use_nli=True,
    )

    keep_df = result_df[result_df["final_status"] == "KEEP"]
    drop_df = result_df[result_df["final_status"] == "DROP"]
    review_df = result_df[result_df["final_status"] == "REVIEW"]
    print(f"  out KEEP={len(keep_df)} DROP={len(drop_df)} REVIEW={len(review_df)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    keep_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    if save_all_categories:
        base = os.path.splitext(output_file)[0]
        if len(drop_df):
            drop_df.to_csv(f"{base}_DROP.csv", index=False, encoding="utf-8-sig")
        if len(review_df):
            review_df.to_csv(f"{base}_REVIEW.csv", index=False, encoding="utf-8-sig")

    stats.update(input_file=input_file, output_file=output_file, original_count=len(df))
    return stats


def _numpy_to_native(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_native(i) for i in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return obj


def batch_filter():
    filter_stack = TextFilterStack()
    all_stats: list[dict] = []
    all_files: list[tuple[str, str]] = []
    for dataset in DATASETS:
        in_dir = os.path.join(PROCESSED_DIR, dataset)
        out_dir = os.path.join(FILTERED_DIR, dataset)
        if not os.path.exists(in_dir):
            continue
        import glob
        for f in glob.glob(os.path.join(in_dir, "*_units.csv")):
            all_files.append((f, out_dir))

    print(f"files n={len(all_files)}")
    start_time = time.time()
    file_times: list[float] = []

    for i, (input_file, out_dir) in enumerate(all_files):
        filename = os.path.basename(input_file)
        output_file = os.path.join(out_dir, filename)
        t0 = time.time()
        print(f"[{i + 1}/{len(all_files)}] {filename}")
        try:
            stats = process_units_file(input_file, output_file, filter_stack)
            dt = time.time() - t0
            file_times.append(dt)
            all_stats.append(stats)
            print(f"  ok {dt:.1f}s")
        except Exception as e:
            print(f"  err {e}")
            import traceback; traceback.print_exc()

    stats_output = os.path.join(FILTERED_DIR, "_filtering_stats.json")
    os.makedirs(os.path.dirname(stats_output), exist_ok=True)
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(_numpy_to_native(all_stats), f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    h, m, s = int(total_time // 3600), int((total_time % 3600) // 60), int(total_time % 60)
    print(f"done files={len(all_stats)}/{len(all_files)} time={h}h{m}m{s}s")
    if file_times:
        print(f"avg_s_per_file={sum(file_times) / len(file_times):.1f}")
    total_orig = sum(s.get("original_count", 0) for s in all_stats)
    total_keep = sum(s.get("final_KEEP", 0) for s in all_stats)
    total_drop = sum(s.get("final_DROP", 0) for s in all_stats)
    total_review = sum(s.get("final_REVIEW", 0) for s in all_stats)
    if total_orig:
        print(
            f"summary orig={total_orig} KEEP={total_keep} DROP={total_drop} "
            f"REVIEW={total_review} retain_pct={total_keep / total_orig * 100:.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Part B — Post-filter REVIEW refinement  (formerly 03_post_filter_review.py)
# ═══════════════════════════════════════════════════════════════════════════

def classify_review(row: pd.Series) -> str:
    """REVIEW_KEEP / REVIEW_CONTEXT / REVIEW_DROP."""
    text = str(row.get("sentence", ""))
    try:
        s = float(row.get("semantic_similarity", 0.0))
    except (ValueError, TypeError):
        s = 0.0
    try:
        m = float(row.get("nli_score", 0.0))
    except (ValueError, TypeError):
        m = 0.0

    _actor = has_actor(text)
    _inst = has_inst(text)

    if has_blacklist(text) and not (_actor or _inst):
        return "REVIEW_DROP"
    if ("美国" in text or "美國" in text) and "选举" in text and not has_taiwan_context(text):
        if not (_actor or _inst):
            return "REVIEW_DROP"

    if s >= 0.40:
        return "REVIEW_KEEP"
    if _actor or _inst:
        return "REVIEW_KEEP"
    if m >= 0.10:
        return "REVIEW_KEEP"
    if has_scarce(text):
        return "REVIEW_KEEP"

    if has_proc_symbolic(text):
        return "REVIEW_CONTEXT"
    if 0.30 <= s < 0.40 and (_actor or _inst):
        return "REVIEW_CONTEXT"
    if 0.0 < m < 0.10:
        return "REVIEW_CONTEXT"

    return "REVIEW_DROP"


def process_review_file(review_file: str, main_file: str, output_dir: str) -> dict:
    print(f"\nProcessing: {os.path.basename(review_file)}")
    review_df = pd.read_csv(review_file, encoding="utf-8-sig")
    review_count = len(review_df)
    print(f"  REVIEW samples: {review_count}")

    if os.path.exists(main_file):
        main_df = pd.read_csv(main_file, encoding="utf-8-sig")
        keep_count = len(main_df)
    else:
        main_df, keep_count = pd.DataFrame(), 0

    review_df["refined_status"] = review_df.apply(classify_review, axis=1)
    refined_counts = review_df["refined_status"].value_counts().to_dict()
    print(f"  REVIEW classification: {refined_counts}")

    review_keep = review_df[review_df["refined_status"] == "REVIEW_KEEP"].drop(columns=["refined_status"])
    review_context = review_df[review_df["refined_status"] == "REVIEW_CONTEXT"]
    review_drop = review_df[review_df["refined_status"] == "REVIEW_DROP"]

    final_main = pd.concat([main_df, review_keep], ignore_index=True)
    base_name = os.path.basename(main_file).replace(".csv", "")
    os.makedirs(output_dir, exist_ok=True)

    final_main.to_csv(os.path.join(output_dir, f"{base_name}.csv"), index=False, encoding="utf-8-sig")
    if len(review_context):
        review_context.to_csv(os.path.join(output_dir, f"{base_name}_CONTEXT.csv"),
                              index=False, encoding="utf-8-sig")
    if len(review_drop):
        review_drop.to_csv(os.path.join(output_dir, f"{base_name}_DROPPED_FROM_REVIEW.csv"),
                           index=False, encoding="utf-8-sig")

    print(f"  Final: KEEP+REVIEW_KEEP={len(final_main)}, CONTEXT={len(review_context)}, DROP={len(review_drop)}")
    return {
        "review_file": review_file, "main_file": main_file,
        "original_keep": keep_count, "original_review": review_count,
        "review_keep_added": len(review_keep), "review_context": len(review_context),
        "review_dropped": len(review_drop), "final_main_count": len(final_main),
        "addition_rate": len(review_keep) / review_count if review_count else 0,
    }


def batch_post_review():
    import glob
    all_stats: list[dict] = []
    review_files: list[tuple[str, str]] = []
    for dataset in DATASETS:
        dpath = os.path.join(FILTERED_DIR, dataset)
        if os.path.exists(dpath):
            for f in glob.glob(os.path.join(dpath, "*_REVIEW.csv")):
                review_files.append((f, dataset))

    print(f"Found {len(review_files)} REVIEW files\n{'=' * 60}")
    for review_file, dataset in review_files:
        base = os.path.basename(review_file).replace("_REVIEW.csv", ".csv")
        main_file = os.path.join(os.path.dirname(review_file), base)
        out_dir = os.path.join(REFINED_DIR, dataset)
        try:
            stats = process_review_file(review_file, main_file, out_dir)
            all_stats.append(stats)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()

    stats_out = os.path.join(REFINED_DIR, "_post_filtering_stats.json")
    os.makedirs(os.path.dirname(stats_out), exist_ok=True)
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    total_keep = sum(s["original_keep"] for s in all_stats)
    total_review = sum(s["original_review"] for s in all_stats)
    total_added = sum(s["review_keep_added"] for s in all_stats)
    print(f"\n{'=' * 60}\nOverall: KEEP={total_keep} REVIEW={total_review} added={total_added}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Stage 2: Filter units (semantic/NLI) + refine REVIEW")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("filter", help="Run semantic + NLI filtering on all units CSVs")
    sub.add_parser("post-review", help="Refine REVIEW subset into KEEP / CONTEXT / DROP")
    args = ap.parse_args()
    if args.command == "filter":
        batch_filter()
    else:
        batch_post_review()


if __name__ == "__main__":
    main()
