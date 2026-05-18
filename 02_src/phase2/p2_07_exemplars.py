#!/usr/bin/env python3
"""p2_07 — Qualitative exemplar sampling for close reading."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import Phase2Config  # noqa: E402
from utils.io import deserialize_l2_column, read_parquet, tuple_to_l2_set  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402


from utils.time_buckets import election_time_bucket  # noqa: E402


def run(cfg: Phase2Config, force: bool = False) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    scheme_name = cfg.raw["cross_camp"].get("primary_network_scheme", "camp")
    manifest_path = art / "manifests" / "p2_07.json"
    expected = {"corpus_content_hash": cfg.corpus_content_hash}
    if should_skip(manifest_path, expected, force):
        print("p2_07: skip (manifest match)")
        return

    cons_path = cfg.scheme_dir("partitions", scheme_name) / "consensus_partition.parquet"
    cons = read_parquet(cons_path) if cons_path.is_file() else pd.DataFrame()
    windows = deserialize_l2_column(read_parquet(art / "windows" / f"windows_n{cfg.raw['windows']['default_n']}.parquet"))
    labeled = deserialize_l2_column(read_parquet(art / "labeled_units.parquet"))

    out_dir = art / "exemplars"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    if cons.empty or "camp" not in cons.columns:
        write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([art / "labeled_units.parquet"])})
        write_summary(
            art, "p2_07",
            params={"scheme": scheme_name}, outputs=[str(out_dir)],
            stats={"skipped": "no consensus"}, elapsed_sec=time.perf_counter() - t0,
        )
        print("p2_07 skipped (no partitions)")
        return

    for camp in cfg.camps:
        camp_cons = cons[cons["camp"] == camp]
        for comm_id, nodes_df in camp_cons.groupby("community_id"):
            comm_l2 = set(nodes_df["node"].astype(str))
            if len(comm_l2) < 2:
                continue
            sub_w = windows[windows["camp"] == camp].copy()
            sub_w["_overlap"] = sub_w["l2_set"].map(lambda s: len(tuple_to_l2_set(s) & comm_l2))
            cand = sub_w[sub_w["_overlap"] >= 2]
            if cand.empty:
                continue

            # Join text from labeled units
            text_map = labeled.groupby(["doc_id", "sent_idx"])["sent_text"].first().to_dict()

            def full_text(row):
                parts = []
                for si in range(int(row["sent_idx_start"]), int(row["sent_idx_end"]) + 1):
                    t = text_map.get((row["doc_id"], si), "")
                    parts.append(f"[{si}] {t}")
                return " ".join(parts)

            cand = cand.copy()
            cand["full_text_with_sentence_marks"] = cand.apply(full_text, axis=1)
            cand["l2_set_str"] = cand["l2_set"].map(lambda s: "|".join(sorted(tuple_to_l2_set(s))))
            cand["community_id"] = comm_id
            cand["time_bucket"] = cand.apply(
                lambda r: election_time_bucket(
                    labeled[(labeled["doc_id"] == r["doc_id"]) & (labeled["sent_idx"] == r["sent_idx_start"])]["date"].iloc[0]
                    if len(labeled[(labeled["doc_id"] == r["doc_id"])]) else ""
                ),
                axis=1,
            )
            cand["sampling_weight"] = cand["_overlap"].astype(float) + np.random.default_rng(42).random(len(cand)) * 0.01
            cand = cand.sort_values("sampling_weight", ascending=False).head(30)

            out_cols = [
                "window_id", "doc_id", "source", "date", "full_text_with_sentence_marks",
                "l2_set_str", "community_id", "sampling_weight",
            ]
            for c in out_cols:
                if c not in cand.columns:
                    cand[c] = ""
            cand["interpretive_notes"] = ""
            # source/date from labeled
            meta = labeled.groupby("doc_id").agg({"source": "first", "date": "first"}).reset_index()
            cand = cand.merge(meta, on="doc_id", how="left", suffixes=("", "_m"))
            if "source_m" in cand.columns:
                cand["source"] = cand["source_m"].fillna(cand.get("source", ""))

            path = out_dir / f"{camp}_community_{comm_id}.csv"
            cand[[c for c in out_cols if c in cand.columns] + ["interpretive_notes"]].to_csv(
                path, index=False, encoding="utf-8-sig"
            )
            n_written += 1

    write_manifest(manifest_path, {**expected, "inputs_hash": inputs_hash([art / "labeled_units.parquet"])})
    write_summary(
        art,
        "p2_07",
        params={"scheme": scheme_name, "top_n": 30},
        outputs=[str(out_dir)],
        stats={"n_exemplar_files": n_written},
        elapsed_sec=time.perf_counter() - t0,
    )
    print(f"p2_07 done: {n_written} exemplar files")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
