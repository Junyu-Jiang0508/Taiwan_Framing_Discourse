#!/usr/bin/env python3
"""Stage 9 — Export rows dropped during cleaning/dedup with reasons for auditing."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_clean_module():
    spec = importlib.util.spec_from_file_location("crc", ROOT / "s03_clean_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pick_refined(defaults: List[str]) -> Path:
    for d in defaults:
        p = (ROOT / d).resolve()
        if p.is_dir():
            return p
    raise SystemExit(
        "找不到 refined 输入目录，请用 --refined 指定（试过: " + ", ".join(defaults) + "）"
    )


def collect_news_drops(crc, df0: pd.DataFrame, rel: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in df0.iterrows():
        new_s, why = crc.clean_news_sentence(r["sentence"])
        if new_s is None:
            d = r.to_dict()
            d["drop_reason"] = why
            d["_source_file"] = rel
            rows.append(d)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_debate_drops(crc, df0: pd.DataFrame, filename: str, rel: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if filename not in crc.DEBATE_DEBATE_FILES:
        return pd.DataFrame()
    for _, r in df0.iterrows():
        if crc.is_hu_yuanhui_procedural(str(r["sentence"])):
            d = r.to_dict()
            d["drop_reason"] = "moderator_procedural_hu_yuanhui"
            d["_source_file"] = rel
            rows.append(d)
    work = df0.loc[
        ~df0["sentence"].astype(str).map(crc.is_hu_yuanhui_procedural)
    ].copy()
    for _, r in work.iterrows():
        s = str(r.get("sentence", "")).strip()
        if len(s) >= 15:
            continue
        if crc.DEBATE_FILLER_DROP.match(s):
            d = r.to_dict()
            d["drop_reason"] = "debate_short_filler_transition"
            d["_source_file"] = rel
            rows.append(d)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_merged_from_cleaned(cleaned_root: Path) -> pd.DataFrame:
    paths = sorted(
        p
        for p in cleaned_root.rglob("*_units.csv")
        if p.is_file() and p.name != "merged_deduped.csv"
    )
    if not paths:
        raise SystemExit(f"在 {cleaned_root} 下未找到 *_units.csv")
    parts = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
    return pd.concat(parts, ignore_index=True)


def dedup_loser_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Rows with non-empty unit_hash that are not the kept first row per hash after tier sort."""
    if merged.empty or "unit_hash" not in merged.columns:
        return pd.DataFrame()
    work = merged.copy()
    work["_dedup_key"] = work["unit_hash"].astype(str)
    has_h = work["_dedup_key"].str.len() > 0
    sub = work.loc[has_h].copy()
    sub = sub.sort_values(["_dedup_key", "_corpus_tier"], ascending=[True, False])
    sub["_keep"] = ~sub.duplicated("_dedup_key", keep="first")
    losers = sub.loc[~sub["_keep"]].copy()
    if losers.empty:
        out = losers.drop(columns=["_dedup_key", "_keep"], errors="ignore")
        out["drop_reason"] = []
        return out
    winners = sub.loc[sub["_keep"]].drop_duplicates("_dedup_key", keep="first")
    widx = winners.set_index("_dedup_key")
    w_sf = widx["_source_file"].astype(str)
    w_src = widx["source"].astype(str) if "source" in widx.columns else pd.Series("", index=widx.index)
    w_tier = widx["_corpus_tier"].astype(str) if "_corpus_tier" in widx.columns else pd.Series("", index=widx.index)
    w_uid = widx["unit_id"].astype(str) if "unit_id" in widx.columns else pd.Series("", index=widx.index)

    losers["drop_reason"] = "cross_corpus_dedup_lower_priority_or_tiebreak_file_order"
    losers["winner__source_file"] = losers["_dedup_key"].map(w_sf)
    losers["winner_source"] = losers["_dedup_key"].map(w_src)
    losers["winner__corpus_tier"] = losers["_dedup_key"].map(w_tier)
    losers["winner_unit_id"] = losers["_dedup_key"].map(w_uid)
    return losers.drop(columns=["_dedup_key", "_keep"], errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export dropped corpus rows with reasons.")
    ap.add_argument(
        "--refined",
        type=Path,
        default=None,
        help="Refined datasets root (default: first existing among 04/03_refined_datasets)",
    )
    ap.add_argument(
        "--cleaned",
        type=Path,
        default=Path("01_data/04_cleaned_datasets"),
        help="Cleaned output root (for reading per-file CSVs + writing reports)",
    )
    ap.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Match 04_clean_refined_corpus.py --skip-semantic for X gate",
    )
    ap.add_argument(
        "--x-sim-threshold",
        type=float,
        default=None,
        help="Override X cosine threshold (default: from 04_clean_refined_corpus)",
    )
    args = ap.parse_args()
    crc = _load_clean_module()
    x_thr = (
        float(args.x_sim_threshold)
        if args.x_sim_threshold is not None
        else float(crc.X_CONTEXT_SIM_THRESHOLD)
    )

    refined = (
        args.refined.resolve()
        if args.refined is not None
        else _pick_refined(["01_data/04_refined_datasets", "01_data/03_refined_datasets"])
    )
    cleaned_root = (ROOT / args.cleaned).resolve() if not args.cleaned.is_absolute() else args.cleaned.resolve()
    if not cleaned_root.is_dir():
        raise SystemExit(f"cleaned 目录不存在: {cleaned_root}")

    csv_files = sorted(p for p in refined.rglob("*_units.csv") if p.is_file())
    if not csv_files:
        raise SystemExit(f"在 {refined} 下未找到 *_units.csv")

    stack = crc.TextFilterStack()
    debate_paths = [
        refined / "02_conference_datasets" / n
        for n in ("Debate_President_units.csv", "vice_president_units.csv")
    ]
    debate_paths = [p.resolve() for p in debate_paths if p.is_file()]
    debate_cache: Dict[Path, Tuple[pd.DataFrame, Any, pd.DataFrame]] = {}
    cleaned_for_seed: List[pd.DataFrame] = []
    for p in debate_paths:
        df0 = crc._ensure_required_columns(crc._load_csv(p))
        dfc, st, srev = crc.clean_debate_dataframe(df0, p.name)
        debate_cache[p] = (dfc, st, srev)
        cleaned_for_seed.append(dfc)
    seed_texts = crc.build_debate_seed_texts(cleaned_for_seed, stack)

    cleaning_parts: List[pd.DataFrame] = []
    for path in csv_files:
        rel = str(path.relative_to(refined)).replace("\\", "/")
        rp = path.resolve()
        df0 = crc._ensure_required_columns(crc._load_csv(path))

        if rp in debate_cache:
            drops = collect_debate_drops(crc, df0, path.name, rel)
        elif "01_news_datasets" in path.parts:
            drops = collect_news_drops(crc, df0, rel)
        elif "03_X_datasets" in path.parts:
            _, _, drops_df = crc.clean_x_dataframe(
                df0, seed_texts, stack, x_thr, args.skip_semantic
            )
            drops = drops_df.copy() if len(drops_df) else pd.DataFrame()
            if len(drops):
                drops["_source_file"] = rel
        elif "02_conference_datasets" in path.parts:
            drops = collect_debate_drops(crc, df0, path.name, rel)
        else:
            drops = pd.DataFrame()

        if len(drops):
            cleaning_parts.append(drops)

    cleaning_all = (
        pd.concat(cleaning_parts, ignore_index=True)
        if cleaning_parts
        else pd.DataFrame()
    )
    out_clean = cleaned_root / "_dropped_cleaning.csv"
    cleaning_all.to_csv(out_clean, index=False, encoding="utf-8-sig")

    merged = build_merged_from_cleaned(cleaned_root)
    dedup_losers = dedup_loser_table(merged)
    out_dedup = cleaned_root / "_dropped_dedup.csv"
    dedup_losers.to_csv(out_dedup, index=False, encoding="utf-8-sig")

    summary_path = cleaned_root / "_dropped_summary.json"
    summary = {
        "refined_input": str(refined),
        "cleaned_root": str(cleaned_root),
        "skip_semantic": args.skip_semantic,
        "x_sim_threshold": x_thr,
        "dropped_cleaning_rows": len(cleaning_all),
        "dropped_cleaning_by_reason": (
            cleaning_all["drop_reason"].value_counts().to_dict()
            if len(cleaning_all) and "drop_reason" in cleaning_all.columns
            else {}
        ),
        "dropped_dedup_rows": len(dedup_losers),
        "files_written": {
            "cleaning": str(out_clean),
            "dedup": str(out_dedup),
        },
    }
    import json

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {out_clean} ({len(cleaning_all)} rows), "
        f"{out_dedup} ({len(dedup_losers)} rows), {summary_path}"
    )


if __name__ == "__main__":
    main()
