#!/usr/bin/env python3
"""p2_00 — Freeze wide-format labeled_units.parquet from final_results.csv."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils import boilerplate  # noqa: E402
from utils.camp_genre import (  # noqa: E402
    derive_camp,
    derive_sent_idx,
    infer_genre,
    parse_l2_cell,
    parse_speaker_parties,
)
from utils.config import Phase2Config  # noqa: E402
from utils.io import l2_set_to_tuple, serialize_l2_column, write_parquet  # noqa: E402
from utils.manifest import inputs_hash, should_skip, write_manifest  # noqa: E402
from utils.summary import write_summary  # noqa: E402
from utils.time_buckets import election_time_bucket  # noqa: E402

UNKNOWN_RATE_HALT = 0.02


def run(cfg: Phase2Config, force: bool = False, doc_ids: list | None = None) -> None:
    t0 = time.perf_counter()
    art = cfg.artifacts_root
    art.mkdir(parents=True, exist_ok=True)
    manifest_path = art / "manifests" / "p2_00.json"
    expected = {
        "corpus_content_hash": cfg.corpus_content_hash,
        "boilerplate_policy": cfg.boilerplate_policy,
    }
    if should_skip(manifest_path, expected, force):
        print("p2_00: skip (manifest match)")
        return

    df = pd.read_csv(cfg.corpus_csv, encoding="utf-8-sig")
    if doc_ids:
        df = df[df["doc_id"].isin(doc_ids)].copy()
    genres_main = set(cfg.genres_main)
    boiler = boilerplate.annotate(df["sentence"].fillna(""))

    rows = []
    social_rows = []
    multi_party = []

    for doc_id, grp in df.groupby("doc_id", sort=False):
        grp = grp.copy()
        grp["_fallback_idx"] = range(len(grp))
        for src_idx, row in grp.iterrows():
            genre = infer_genre(row)
            camp = derive_camp(row, genre)
            sent_idx = derive_sent_idx(row, int(row["_fallback_idx"]))
            l2_set = frozenset(parse_l2_cell(row.get("L2_labels")))
            parties = parse_speaker_parties(row.get("speakers"))
            date_val = row.get("date", "")
            rec = {
                "doc_id": row["doc_id"],
                "sent_idx": sent_idx,
                "sent_text": row.get("sentence", ""),
                "L1": str(row.get("L1_label", "") or "").strip(),
                "l2_set": l2_set_to_tuple(l2_set),
                "camp": camp,
                "genre": genre,
                "date": date_val,
                "time_bucket": election_time_bucket(date_val),
                "source": row.get("source", ""),
                "boiler_kind": boiler.at[src_idx, "boiler_kind"],
                "boiler_tier": boiler.at[src_idx, "boiler_tier"],
            }
            if len(parties) >= 2:
                multi_party.append({
                    **rec,
                    "speaker_parties": sorted(parties),
                })
            if genre == "social":
                social_rows.append(rec)
            elif genre in genres_main:
                rows.append(rec)

    labeled = pd.DataFrame(rows)
    labeled = labeled.sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)

    diag_dir = art / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Scrape-residue filter. Audited on the pre-filter main-genre set so the
    # per-pattern label-load profile that justifies the tiers stays visible.
    bp_policy = cfg.boilerplate_policy
    n_l2 = labeled["l2_set"].map(len)
    boiler_audit = {
        "policy": bp_policy,
        "n_rows_before": len(labeled),
        "patterns": boilerplate.audit(labeled["sent_text"], n_l2=n_l2),
    }
    to_drop = boilerplate.drop_mask(labeled["boiler_tier"], bp_policy)
    boiler_audit["n_dropped"] = int(to_drop.sum())
    boiler_audit["dropped_rate"] = float(to_drop.mean()) if len(labeled) else 0.0
    boiler_audit["dropped_by_kind"] = (
        labeled.loc[to_drop, "boiler_kind"].value_counts().to_dict()
    )
    boiler_audit["dropped_by_genre"] = (
        labeled.loc[to_drop].groupby("genre").size().to_dict()
    )
    if to_drop.any():
        labeled = labeled[~to_drop].reset_index(drop=True)
        social_rows = [
            r for r in social_rows
            if not boilerplate.drops(r["boiler_tier"], bp_policy)
        ]
    boiler_audit["n_rows_after"] = len(labeled)
    (diag_dir / "boilerplate_audit.json").write_text(
        json.dumps(boiler_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"p2_00 boilerplate ({bp_policy}): dropped {boiler_audit['n_dropped']} "
        f"/ {boiler_audit['n_rows_before']} rows "
        f"({100 * boiler_audit['dropped_rate']:.2f}%)"
    )

    # Unknown camp audit (news)
    news = labeled[labeled["genre"] == "news"]
    n_news = len(news)
    n_unk = int((news["camp"] == "Unknown").sum()) if n_news else 0
    unk_rate = n_unk / n_news if n_news else 0.0
    audit = {
        "news_rows": n_news,
        "news_unknown_camp": n_unk,
        "news_unknown_rate": unk_rate,
        "halt_threshold": UNKNOWN_RATE_HALT,
    }
    if unk_rate >= UNKNOWN_RATE_HALT:
        (diag_dir / "unknown_camp_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        raise SystemExit(
            f"p2_00 gate: news Unknown camp rate {unk_rate:.2%} >= {UNKNOWN_RATE_HALT:.0%}. "
            "Fix source→camp mapping before continuing."
        )

    policy = cfg.raw["strata"].get("unknown_camp_policy", "drop_with_audit")
    audit["unknown_camp_policy"] = policy
    if policy == "drop_with_audit":
        n_before = len(labeled)
        labeled = labeled[labeled["camp"] != "Unknown"].reset_index(drop=True)
        audit["dropped_unknown_rows"] = n_before - len(labeled)
    (diag_dir / "unknown_camp_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    out_main = art / "labeled_units.parquet"
    write_parquet(labeled, out_main, sort_by=["doc_id", "sent_idx"])

    if social_rows:
        social_df = pd.DataFrame(social_rows)
        write_parquet(
            social_df,
            art / "labeled_units_social_appendix.parquet",
            sort_by=["doc_id", "sent_idx"],
        )

    multi_party = [r for r in multi_party if not boilerplate.drops(r["boiler_tier"], bp_policy)]
    if multi_party:
        mp = pd.DataFrame(multi_party)
        write_parquet(mp, diag_dir / "multi_party_sentences.parquet", sort_by=["doc_id", "sent_idx"])

    write_manifest(manifest_path, {
        **expected,
        "corpus_file_hash": cfg.corpus_file_hash,
        "inputs_hash": inputs_hash([cfg.corpus_csv]),
        "corpus_csv": str(cfg.corpus_csv),
        "n_rows_main": len(labeled),
        "n_rows_social": len(social_rows),
        "n_multi_party": len(multi_party),
        "n_boilerplate_dropped": boiler_audit["n_dropped"],
    })

    elapsed = time.perf_counter() - t0
    write_summary(
        art,
        "p2_00",
        params={
            "corpus_csv": str(cfg.corpus_csv),
            "corpus_hash": cfg.corpus_content_hash[:16],
            "boilerplate_policy": bp_policy,
        },
        outputs=[
            str(out_main),
            str(diag_dir / "unknown_camp_audit.json"),
            str(diag_dir / "boilerplate_audit.json"),
        ],
        stats={
            "n_sentences_main": len(labeled),
            "stratum_counts": labeled.groupby(["camp", "genre"]).size().to_dict(),
            "news_unknown_rate": f"{unk_rate:.4%}",
            "n_multi_party_sentences": len(multi_party),
            "n_boilerplate_dropped": boiler_audit["n_dropped"],
            "boilerplate_dropped_rate": f"{boiler_audit['dropped_rate']:.4%}",
        },
        notes=[
            f"Scrape-residue filter '{bp_policy}' applied before the camp gate; "
            "prefix-tier rows (real sentence + scrape prefix) are kept and flagged "
            "in boiler_kind / boiler_tier.",
            "Dropping rows makes previously non-adjacent sentences adjacent in p2_01 "
            "windows; window spans are contiguous in position, not in original sent_idx.",
        ],
        elapsed_sec=elapsed,
    )
    print(f"p2_00 done: {len(labeled)} sentences → {out_main}")


if __name__ == "__main__":
    from utils.config import load_config

    run(load_config(), force="--force" in sys.argv)
