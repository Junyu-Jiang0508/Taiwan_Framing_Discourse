#!/usr/bin/env python3
"""p2_01 — Sliding window construction."""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.io import (  # noqa: E402
    deserialize_l2_column,
    l2_set_to_tuple,
    read_parquet,
    union_l2_sets,
    write_parquet,
)
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402
from utils.time_buckets import window_time_bucket  # noqa: E402


def build_windows_for_doc(
    doc_df: pd.DataFrame,
    n: int,
    step: int,
) -> list:
    doc_df = doc_df.sort_values("sent_idx")
    sents = doc_df.to_dict("records")
    n_sents = len(sents)
    if n_sents == 0:
        return []
    n_eff = n if n != "document" else n_sents
    if isinstance(n_eff, str):
        n_eff = n_sents
    n_eff = min(int(n_eff), n_sents)
    truncated = n_eff < (n if isinstance(n, int) else n_sents)
    camp = sents[0]["camp"]
    genre = sents[0]["genre"]
    doc_id = sents[0]["doc_id"]
    rows = []
    if n == "document":
        starts = [0]
    else:
        starts = list(range(0, n_sents - n_eff + 1, step))
    for start in starts:
        chunk = sents[start : start + n_eff]
        l2_union = union_l2_sets(r["l2_set"] for r in chunk)
        sent_dates = [r.get("date", "") for r in chunk]
        rows.append({
            "window_id": f"{doc_id}_w{start:04d}",
            "doc_id": doc_id,
            "sent_idx_start": chunk[0]["sent_idx"],
            "sent_idx_end": chunk[-1]["sent_idx"],
            "camp": camp,
            "genre": genre,
            "time_bucket": window_time_bucket(sent_dates),
            "l2_set": l2_set_to_tuple(l2_union),
            "is_empty_l2": len(l2_union) == 0,
            "is_truncated": truncated,
        })
    return rows


def run(cfg: Phase2Config, force: bool = False, doc_ids: list | None = None) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    labeled_path = art / "labeled_units.parquet"
    manifest_path = art / "manifests" / "p2_01.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "windows_config": str(cfg.raw["windows"]),
    }
    if should_skip(manifest_path, expected, force):
        print("p2_01: skip (manifest match)")
        return

    labeled = deserialize_l2_column(read_parquet(labeled_path))
    if doc_ids:
        labeled = labeled[labeled["doc_id"].isin(doc_ids)]

    wcfg = cfg.raw["windows"]
    default_n = wcfg["default_n"]
    step = wcfg["step"]
    robustness = wcfg.get("robustness_ns", [])

    win_dir = art / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    def _run_n(n_val, suffix: str):
        all_rows = []
        for doc_id, grp in labeled.groupby("doc_id", sort=True):
            all_rows.extend(build_windows_for_doc(grp, n_val, step))
        wdf = pd.DataFrame(all_rows)
        out = win_dir / f"windows_n{suffix}.parquet"
        write_parquet(wdf, out, sort_by=["camp", "genre", "doc_id", "sent_idx_start"])
        outputs.append(str(out))
        return wdf

    main_df = _run_n(default_n, str(default_n))
    for n_val in robustness:
        if n_val == default_n:
            continue
        suf = "document" if n_val == "document" else str(n_val)
        _run_n(n_val, suf)

    trunc_rate = float(main_df["is_truncated"].mean()) if len(main_df) else 0.0
    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([labeled_path])})
    write_summary(
        art,
        "p2_01",
        params={"default_n": default_n, "step": step, "robustness_ns": robustness},
        outputs=outputs,
        stats={
            "n_windows": len(main_df),
            "truncated_rate": f"{trunc_rate:.4%}",
            "empty_l2_rate": f"{float(main_df['is_empty_l2'].mean()):.4%}" if len(main_df) else "0",
        },
        notes=["Sliding step=1; no cross-document windows."],
        elapsed_sec=time.perf_counter() - t0,
    )
    print(f"p2_01 done: {len(main_df)} windows")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
