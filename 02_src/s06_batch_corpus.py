#!/usr/bin/env python3
"""Stage 6 — OpenAI Batch annotation for the full corpus.

Supports three scopes via sub-commands:

* ``full``           — all CSVs under cleaned datasets root
* ``merged``         — single merged_deduped.csv
* ``resume-submit``  — resume an interrupted merged-deduped submit

Usage::

    python s06_batch_corpus.py full submit --input 01_data/04_cleaned_datasets
    python s06_batch_corpus.py full retrieve
    python s06_batch_corpus.py merged submit [--input path/to/merged_deduped.csv]
    python s06_batch_corpus.py merged retrieve
    python s06_batch_corpus.py resume-submit [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from framing_batch_core import (
    TaiwanBatchManager,
    create_corpus_run_dir,
    create_merged_deduped_corpus_run_dir,
    resolve_corpus_run_dir,
    resolve_merged_deduped_corpus_run_dir,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DEFAULT_CORPUS_INPUT = "01_data/04_cleaned_datasets"
DEFAULT_MERGED_DEDUPED = "01_data/04_cleaned_datasets/merged_deduped.csv"


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _defaults(base_dir: Path):
    return dict(
        l1=str(base_dir / "01_data/05_labels_guidance/01_annotation_guide_label1_v9.csv"),
        l2=str(base_dir / "01_data/05_labels_guidance/02_annotation_guide_label2_v10.csv"),
        l1_fs=str(base_dir / "01_data/05_labels_guidance/03_fewshot_L1_v9.csv"),
        l2_fs=str(base_dir / "01_data/05_labels_guidance/03_fewshot_L2_v10.csv"),
        hard=str(base_dir / "01_data/05_labels_guidance/04_hard_case_pool.csv"),
    )


def _build_manager(args, defs: dict) -> TaiwanBatchManager:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set"); sys.exit(1)
    manager = TaiwanBatchManager(api_key, model=args.model)
    l1 = getattr(args, "l1", defs["l1"])
    l2 = getattr(args, "l2", defs["l2"])
    l1_fs = getattr(args, "l1_fewshot", defs["l1_fs"])
    l2_fs = getattr(args, "l2_fewshot", defs["l2_fs"])
    manager.load_guides(l1, l2, l1_fs, l2_fs)
    manager.load_hard_pool(
        getattr(args, "hard_pool", defs["hard"]),
        n_l1=args.hard_pool_n_l1,
        n_l2=args.hard_pool_n_l2,
        seed=getattr(args, "hard_pool_seed", None),
        enabled=not getattr(args, "no_hard_pool", False),
    )
    return manager


def _parallel_submit_retrieve(args, manager, run_dir, batch_l1, batch_l2, ref_csv):
    """Common parallel submit/retrieve logic shared by full & merged scopes."""
    ids_l1 = run_dir / "l1_parallel_batch_ids.json"
    ids_l2 = run_dir / "l2_parallel_batch_ids.json"
    results_l1 = str(run_dir / "results_l1.csv")
    final_csv = str(run_dir / "final_results.csv")
    chunk_lines = getattr(args, "jsonl_chunk_lines", 50)
    max_conc = getattr(args, "max_concurrent", 3)

    if args.action == "submit":
        l1_ids = manager.submit_files(
            batch_l1, f"{args.scope.title()} parallel L1", lines_per_chunk=chunk_lines,
            max_concurrent=max_conc,
        )
        l2_ids = manager.submit_files(
            batch_l2, f"{args.scope.title()} parallel L2", lines_per_chunk=chunk_lines,
            max_concurrent=max_conc,
        )
        if not l1_ids or not l2_ids:
            logger.error("Missing batch IDs."); sys.exit(1)
        ids_l1.write_text(json.dumps(l1_ids, indent=2), encoding="utf-8")
        ids_l2.write_text(json.dumps(l2_ids, indent=2), encoding="utf-8")
        logger.info("Wrote %s and %s", ids_l1, ids_l2)
    else:
        if not Path(ref_csv).exists():
            logger.error("Missing %s — run submit first", ref_csv); sys.exit(1)
        if not ids_l1.exists() or not ids_l2.exists():
            logger.error("Missing batch-id files under %s", run_dir); sys.exit(1)
        with open(ids_l1, encoding="utf-8") as f:
            l1_ids = json.load(f)
        with open(ids_l2, encoding="utf-8") as f:
            l2_ids = json.load(f)
        manager.process_l1_results(l1_ids, ref_csv, results_l1)
        manager.process_l2_results(l2_ids, results_l1, final_csv)


# ═══════════════════════════════════════════════════════════════════════════
#  Scope: full  (from 05_corpus_parallel_async)
# ═══════════════════════════════════════════════════════════════════════════

def _full(args, base_dir: Path, defs: dict):
    manager = _build_manager(args, defs)
    if args.action == "submit":
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            logger.error("Input missing: %s", input_path); sys.exit(1)
        td = Path(tempfile.mkdtemp(prefix="corpus_parallel_"))
        try:
            tp_ref = td / "ref_parallel.csv"
            tp_l1 = td / "batch_l1_parallel.jsonl"
            tp_l2 = td / "batch_l2_parallel.jsonl"
            n = manager.prepare_parallel_l1_l2(str(input_path), str(tp_l1), str(tp_l2), str(tp_ref))
            if n == 0:
                logger.error("Built 0 requests."); sys.exit(1)
            run_dir = create_corpus_run_dir(base_dir)
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"input": str(input_path), "n_requests": n, "model": args.model},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shutil.move(str(tp_ref), run_dir / "ref_parallel.csv")
            shutil.move(str(tp_l1), run_dir / "batch_l1_parallel.jsonl")
            shutil.move(str(tp_l2), run_dir / "batch_l2_parallel.jsonl")
        finally:
            shutil.rmtree(td, ignore_errors=True)
    else:
        try:
            run_dir = resolve_corpus_run_dir(base_dir)
        except FileNotFoundError as e:
            logger.error("%s", e); sys.exit(1)

    _parallel_submit_retrieve(
        args, manager, run_dir,
        str(run_dir / "batch_l1_parallel.jsonl"),
        str(run_dir / "batch_l2_parallel.jsonl"),
        str(run_dir / "ref_parallel.csv"),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Scope: merged  (from 16_corpus_merged_deduped_async)
# ═══════════════════════════════════════════════════════════════════════════

def _merged(args, base_dir: Path, defs: dict):
    manager = _build_manager(args, defs)
    if args.action == "submit":
        input_path = Path(args.input).resolve()
        if not input_path.is_file() or input_path.suffix.lower() != ".csv":
            logger.error("Input must be an existing CSV: %s", input_path); sys.exit(1)
        td = Path(tempfile.mkdtemp(prefix="merged_deduped_parallel_"))
        try:
            tp_ref = td / "ref_parallel.csv"
            tp_l1 = td / "batch_l1_parallel.jsonl"
            tp_l2 = td / "batch_l2_parallel.jsonl"
            n = manager.prepare_parallel_l1_l2(str(input_path), str(tp_l1), str(tp_l2), str(tp_ref))
            if n == 0:
                logger.error("Built 0 requests."); sys.exit(1)
            run_dir = create_merged_deduped_corpus_run_dir(base_dir)
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"script": "s06_batch_corpus.py merged", "input": str(input_path),
                            "n_requests": n, "model": args.model}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shutil.move(str(tp_ref), run_dir / "ref_parallel.csv")
            shutil.move(str(tp_l1), run_dir / "batch_l1_parallel.jsonl")
            shutil.move(str(tp_l2), run_dir / "batch_l2_parallel.jsonl")
        finally:
            shutil.rmtree(td, ignore_errors=True)
    else:
        try:
            run_dir = resolve_merged_deduped_corpus_run_dir(base_dir)
        except FileNotFoundError as e:
            logger.error("%s", e); sys.exit(1)

    _parallel_submit_retrieve(
        args, manager, run_dir,
        str(run_dir / "batch_l1_parallel.jsonl"),
        str(run_dir / "batch_l2_parallel.jsonl"),
        str(run_dir / "ref_parallel.csv"),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Scope: resume-submit  (from 16_resume_submit)
# ═══════════════════════════════════════════════════════════════════════════

PART_RE = re.compile(r"part (\d+)$")


def _list_all_batches(client):
    from openai import OpenAI
    all_b, after = [], None
    while True:
        kw: dict = {"limit": 100}
        if after:
            kw["after"] = after
        page = client.batches.list(**kw)
        if not page.data:
            break
        all_b.extend(page.data)
        after = page.data[-1].id
        if len(page.data) < 100:
            break
    return all_b


def _collect_existing(all_batches, prefix: str) -> dict[int, str]:
    parts: dict[int, list] = defaultdict(list)
    for b in all_batches:
        desc = (b.metadata or {}).get("description", "")
        if not desc.startswith(prefix):
            continue
        m = PART_RE.search(desc)
        if not m:
            continue
        parts[int(m.group(1))].append(b)

    rank = {"completed": 0, "in_progress": 1, "finalizing": 2, "validating": 3}
    result: dict[int, str] = {}
    for num, batches in parts.items():
        good = [b for b in batches if b.status in rank]
        if good:
            good.sort(key=lambda b: rank.get(b.status, 99))
            result[num] = good[0].id
    return result


def _submit_remaining(manager, part_files, existing, phase_desc, max_concurrent):
    ids_by_part: dict[int, str] = dict(existing)
    active_ids: list[str] = []
    n = len(part_files)
    abort = False
    for i, f_path in enumerate(part_files):
        part_num = i + 1
        if part_num in ids_by_part:
            continue
        if abort:
            break
        active_ids = manager._wait_for_batch_slots(active_ids, max_concurrent)
        while True:
            try:
                logger.info("[%s/%s] upload %s", part_num, n, os.path.basename(f_path))
                with open(f_path, "rb") as f:
                    file_obj = manager.client.files.create(file=f, purpose="batch")
                batch_job = manager.client.batches.create(
                    input_file_id=file_obj.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": f"{phase_desc} part {part_num}"},
                )
                outcome = manager._poll_batch_after_create(batch_job.id)
                if outcome == "retry_enqueued":
                    logger.warning("Enqueued-token cap; sleeping 10m then retrying")
                    time.sleep(600); continue
                if outcome == "fatal":
                    abort = True; break
                ids_by_part[part_num] = batch_job.id
                active_ids.append(batch_job.id)
                time.sleep(2); break
            except Exception as e:
                err_msg = str(e).lower()
                if "enqueued" in err_msg and "token" in err_msg:
                    time.sleep(600); continue
                elif "rate limit" in err_msg:
                    time.sleep(300); continue
                else:
                    logger.error("submit %s: %s", f_path, e)
                    abort = True; break
    return [ids_by_part[k] for k in sorted(ids_by_part)]


def _resume_submit(args, base_dir: Path, defs: dict):
    from openai import OpenAI
    manager = _build_manager(args, defs)

    try:
        run_dir = resolve_merged_deduped_corpus_run_dir(base_dir)
    except FileNotFoundError as e:
        logger.error("%s", e); sys.exit(1)
    logger.info("Resuming run: %s", run_dir)

    ids_l1_path = run_dir / "l1_parallel_batch_ids.json"
    ids_l2_path = run_dir / "l2_parallel_batch_ids.json"
    batch_l2 = run_dir / "batch_l2_parallel.jsonl"

    if ids_l1_path.exists() and ids_l2_path.exists():
        logger.info("Both batch-ID files already exist — use 'retrieve' instead."); sys.exit(0)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_batches = _list_all_batches(client)
    logger.info("Found %d total batches in account", len(all_batches))
    max_conc = getattr(args, "max_concurrent", 3)

    l1_existing = _collect_existing(all_batches, "Merged deduped parallel L1")
    l1_parts = sorted(run_dir.glob("batch_l1_parallel_part*.jsonl"), key=lambda p: p.name)
    l1_todo = [i + 1 for i in range(len(l1_parts)) if (i + 1) not in l1_existing]
    logger.info("L1: %d parts, %d submitted, %d remaining", len(l1_parts), len(l1_existing), len(l1_todo))

    if getattr(args, "dry_run", False):
        logger.info("[DRY RUN] Would submit %d L1 parts, then L2", len(l1_todo)); sys.exit(0)

    l1_ids = _submit_remaining(manager, [str(p) for p in l1_parts], l1_existing,
                               "Merged deduped parallel L1", max_conc)
    ids_l1_path.write_text(json.dumps(l1_ids, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d IDs)", ids_l1_path, len(l1_ids))

    l2_existing = _collect_existing(all_batches, "Merged deduped parallel L2")
    l2_parts = sorted(run_dir.glob("batch_l2_parallel_part*.jsonl"), key=lambda p: p.name)
    if not l2_parts:
        l2_parts_str = manager._split_jsonl(str(batch_l2), lines_per_chunk=50)
        l2_parts = [Path(p) for p in l2_parts_str]
    l2_ids = _submit_remaining(manager, [str(p) for p in l2_parts], l2_existing,
                               "Merged deduped parallel L2", max_conc)
    ids_l2_path.write_text(json.dumps(l2_ids, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d IDs)", ids_l2_path, len(l2_ids))
    logger.info("Resume complete. Run: python s06_batch_corpus.py merged retrieve")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    base_dir = Path(os.getcwd())
    defs = _defaults(base_dir)

    ap = argparse.ArgumentParser(description="Stage 6: Corpus batch annotation (OpenAI)")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--no-hard-pool", action="store_true")
    ap.add_argument("--hard-pool", default=defs["hard"])
    ap.add_argument("--hard-pool-n-l1", type=int, default=4)
    ap.add_argument("--hard-pool-n-l2", type=int, default=5)
    ap.add_argument("--hard-pool-seed", type=int, default=None)
    ap.add_argument("--jsonl-chunk-lines", type=int, default=50)
    ap.add_argument("--max-concurrent", type=int, default=3)

    sub = ap.add_subparsers(dest="scope", required=True)

    # full
    p_full = sub.add_parser("full", help="All CSVs under cleaned datasets root")
    sp_full = p_full.add_subparsers(dest="action", required=True)
    fs = sp_full.add_parser("submit")
    fs.add_argument("--input", default=str(base_dir / DEFAULT_CORPUS_INPUT))
    fs.add_argument("--l1", default=defs["l1"])
    fs.add_argument("--l2", default=defs["l2"])
    fs.add_argument("--l1-fewshot", default=defs["l1_fs"])
    fs.add_argument("--l2-fewshot", default=defs["l2_fs"])
    fr = sp_full.add_parser("retrieve")
    fr.add_argument("--l1", default=defs["l1"])
    fr.add_argument("--l2", default=defs["l2"])
    fr.add_argument("--l1-fewshot", default=defs["l1_fs"])
    fr.add_argument("--l2-fewshot", default=defs["l2_fs"])

    # merged
    p_m = sub.add_parser("merged", help="Single merged_deduped.csv")
    sp_m = p_m.add_subparsers(dest="action", required=True)
    ms = sp_m.add_parser("submit")
    ms.add_argument("--input", default=str(base_dir / DEFAULT_MERGED_DEDUPED))
    ms.add_argument("--l1", default=defs["l1"])
    ms.add_argument("--l2", default=defs["l2"])
    ms.add_argument("--l1-fewshot", default=defs["l1_fs"])
    ms.add_argument("--l2-fewshot", default=defs["l2_fs"])
    mr = sp_m.add_parser("retrieve")
    mr.add_argument("--l1", default=defs["l1"])
    mr.add_argument("--l2", default=defs["l2"])
    mr.add_argument("--l1-fewshot", default=defs["l1_fs"])
    mr.add_argument("--l2-fewshot", default=defs["l2_fs"])

    # resume-submit
    p_r = sub.add_parser("resume-submit", help="Resume interrupted merged-deduped submit")
    p_r.add_argument("--dry-run", action="store_true")
    p_r.add_argument("--l1", default=defs["l1"])
    p_r.add_argument("--l2", default=defs["l2"])
    p_r.add_argument("--l1-fewshot", default=defs["l1_fs"])
    p_r.add_argument("--l2-fewshot", default=defs["l2_fs"])

    args = ap.parse_args()
    dispatch = {"full": _full, "merged": _merged, "resume-submit": _resume_submit}
    dispatch[args.scope](args, base_dir, defs)


if __name__ == "__main__":
    main()
