#!/usr/bin/env python3
"""Stage 5 — OpenAI Batch annotation for validation sets.

Unified entry point for three strategies:

* ``parallel``   — L1 + L2 in parallel  (no L1→L2 conditioning in prompts)
* ``l1-only``    — L1 phase only
* ``l2-only``    — L2 phase conditioned on prior L1 results

Each strategy supports ``submit`` and ``retrieve`` sub-commands.

Usage::

    python s05_batch_validate.py parallel submit --input <csv>
    python s05_batch_validate.py parallel retrieve
    python s05_batch_validate.py l1-only submit --input <csv>
    python s05_batch_validate.py l1-only retrieve
    python s05_batch_validate.py l2-only submit
    python s05_batch_validate.py l2-only retrieve
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from framing_batch_core import (
    TaiwanBatchManager,
    create_run_dir,
    resolve_run_dir,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    manager = TaiwanBatchManager(api_key, model=args.model)
    l1 = getattr(args, "l1", None) or defs["l1"]
    l2 = getattr(args, "l2", None) or defs["l2"]
    l1_fs = getattr(args, "l1_fewshot", None) or defs["l1_fs"]
    l2_fs = getattr(args, "l2_fewshot", None) or defs["l2_fs"]
    manager.load_guides(l1, l2, l1_fs, l2_fs)
    manager.load_hard_pool(
        getattr(args, "hard_pool", defs["hard"]),
        n_l1=args.hard_pool_n_l1,
        n_l2=args.hard_pool_n_l2,
        seed=getattr(args, "hard_pool_seed", None),
        enabled=not getattr(args, "no_hard_pool", False),
    )
    return manager


# ═══════════════════════════════════════════════════════════════════════════
#  Parallel L1+L2  (from 05_validation_parallel_async)
# ═══════════════════════════════════════════════════════════════════════════

def _parallel(args, base_dir: Path, defs: dict):
    manager = _build_manager(args, defs)

    if args.action == "submit":
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            logger.error("Input missing: %s", input_path); sys.exit(1)
        td = Path(tempfile.mkdtemp(prefix="parallel_batch_"))
        try:
            tp_ref = td / "ref_parallel.csv"
            tp_l1 = td / "batch_l1_parallel.jsonl"
            tp_l2 = td / "batch_l2_parallel.jsonl"
            n = manager.prepare_parallel_l1_l2(str(input_path), str(tp_l1), str(tp_l2), str(tp_ref))
            if n == 0:
                logger.error("Built 0 parallel requests."); sys.exit(1)
            run_dir = create_run_dir(base_dir)
            shutil.move(str(tp_ref), run_dir / "ref_parallel.csv")
            shutil.move(str(tp_l1), run_dir / "batch_l1_parallel.jsonl")
            shutil.move(str(tp_l2), run_dir / "batch_l2_parallel.jsonl")
        finally:
            shutil.rmtree(td, ignore_errors=True)
    else:
        try:
            run_dir = resolve_run_dir(base_dir)
        except FileNotFoundError as e:
            logger.error("%s", e); sys.exit(1)

    ref_csv = str(run_dir / "ref_parallel.csv")
    ids_l1 = run_dir / "l1_parallel_batch_ids.json"
    ids_l2 = run_dir / "l2_parallel_batch_ids.json"
    results_l1 = str(run_dir / "results_l1.csv")
    final_csv = str(run_dir / "final_results.csv")

    if args.action == "submit":
        l1_ids = manager.submit_files(str(run_dir / "batch_l1_parallel.jsonl"), "Parallel L1")
        l2_ids = manager.submit_files(str(run_dir / "batch_l2_parallel.jsonl"), "Parallel L2")
        if not l1_ids or not l2_ids:
            logger.error("Missing batch IDs."); sys.exit(1)
        ids_l1.write_text(json.dumps(l1_ids, indent=2), encoding="utf-8")
        ids_l2.write_text(json.dumps(l2_ids, indent=2), encoding="utf-8")
        logger.info("Wrote %s and %s", ids_l1, ids_l2)
    else:
        if not Path(ref_csv).exists():
            logger.error("Missing %s — not a parallel run?", ref_csv); sys.exit(1)
        with open(ids_l1, encoding="utf-8") as f:
            l1_ids = json.load(f)
        with open(ids_l2, encoding="utf-8") as f:
            l2_ids = json.load(f)
        manager.process_l1_results(l1_ids, ref_csv, results_l1)
        manager.process_l2_results(l2_ids, results_l1, final_csv)


# ═══════════════════════════════════════════════════════════════════════════
#  L1-only  (from 05_validation_l1_async)
# ═══════════════════════════════════════════════════════════════════════════

def _l1_only(args, base_dir: Path, defs: dict):
    manager = _build_manager(args, defs)
    manager.load_guides(getattr(args, "l1", defs["l1"]), None,
                        getattr(args, "l1_fewshot", defs["l1_fs"]), None)

    if args.action == "submit":
        run_dir = create_run_dir(base_dir)
    else:
        try:
            run_dir = resolve_run_dir(base_dir)
        except FileNotFoundError as e:
            logger.error("%s", e); return

    batch_jsonl = str(run_dir / "batch_l1.jsonl")
    ref_csv = str(run_dir / "ref_l1.csv")
    results_csv = str(run_dir / "results_l1.csv")

    if args.action == "submit":
        n = manager.prepare_l1_batch_from_input(args.input, batch_jsonl, ref_csv)
        if n == 0:
            logger.error("Built 0 L1 requests."); sys.exit(1)
        batch_ids = manager.submit_files(batch_jsonl, "L1 Phase")
        if not batch_ids:
            logger.error("No batch IDs returned."); sys.exit(1)
        (run_dir / "l1_batch_ids.json").write_text(
            json.dumps(batch_ids, indent=2), encoding="utf-8"
        )
    else:
        ids_file = run_dir / "l1_batch_ids.json"
        if not ids_file.exists():
            logger.error("Missing %s", ids_file); return
        with open(ids_file, encoding="utf-8") as f:
            batch_ids = json.load(f)
        manager.process_l1_results(batch_ids, ref_csv, results_csv)


# ═══════════════════════════════════════════════════════════════════════════
#  L2-only  (from 05_validation_l2_async)
# ═══════════════════════════════════════════════════════════════════════════

def _l2_only(args, base_dir: Path, defs: dict):
    manager = _build_manager(args, defs)
    manager.load_guides(None, getattr(args, "l2", defs["l2"]),
                        None, getattr(args, "l2_fewshot", defs["l2_fs"]))

    try:
        run_dir = resolve_run_dir(base_dir)
    except FileNotFoundError as e:
        logger.error("%s", e); return

    results_l1 = run_dir / "results_l1.csv"
    batch_jsonl = str(run_dir / "batch_l2.jsonl")
    ref_l2 = str(run_dir / "ref_l2.csv")
    final_csv = str(run_dir / "final_results.csv")

    if args.action == "submit":
        if not results_l1.exists():
            logger.error("Missing %s — run 'l1-only retrieve' first", results_l1)
            sys.exit(1)
        n = manager.prepare_l2_batch_after_l1(str(results_l1), batch_jsonl, ref_l2)
        if n == 0:
            logger.error("Built 0 L2 requests."); sys.exit(1)
        batch_ids = manager.submit_files(batch_jsonl, "L2 Phase")
        if not batch_ids:
            logger.error("No batch IDs returned."); sys.exit(1)
        (run_dir / "l2_batch_ids.json").write_text(
            json.dumps(batch_ids, indent=2), encoding="utf-8"
        )
    else:
        ids_file = run_dir / "l2_batch_ids.json"
        if not ids_file.exists():
            logger.error("Missing %s", ids_file); return
        with open(ids_file, encoding="utf-8") as f:
            batch_ids = json.load(f)
        manager.process_l2_results(batch_ids, ref_l2, final_csv)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    base_dir = Path(os.getcwd())
    defs = _defaults(base_dir)

    ap = argparse.ArgumentParser(description="Stage 5: Validation batch annotation (OpenAI)")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--no-hard-pool", action="store_true")
    ap.add_argument("--hard-pool", default=defs["hard"])
    ap.add_argument("--hard-pool-n-l1", type=int, default=4)
    ap.add_argument("--hard-pool-n-l2", type=int, default=5)
    ap.add_argument("--hard-pool-seed", type=int, default=None)

    sub = ap.add_subparsers(dest="strategy", required=True)

    # parallel
    p_par = sub.add_parser("parallel", help="L1+L2 in parallel (no L1→L2 conditioning)")
    sp_par = p_par.add_subparsers(dest="action", required=True)
    ps = sp_par.add_parser("submit")
    ps.add_argument("--input", required=True)
    ps.add_argument("--l1", default=defs["l1"])
    ps.add_argument("--l2", default=defs["l2"])
    ps.add_argument("--l1-fewshot", default=defs["l1_fs"])
    ps.add_argument("--l2-fewshot", default=defs["l2_fs"])
    pr = sp_par.add_parser("retrieve")
    pr.add_argument("--l1", default=defs["l1"])
    pr.add_argument("--l2", default=defs["l2"])
    pr.add_argument("--l1-fewshot", default=defs["l1_fs"])
    pr.add_argument("--l2-fewshot", default=defs["l2_fs"])

    # l1-only
    p_l1 = sub.add_parser("l1-only", help="L1 phase only")
    sp_l1 = p_l1.add_subparsers(dest="action", required=True)
    l1s = sp_l1.add_parser("submit")
    l1s.add_argument("--input", required=True)
    l1s.add_argument("--l1", default=defs["l1"])
    l1s.add_argument("--l1-fewshot", default=defs["l1_fs"])
    l1r = sp_l1.add_parser("retrieve")
    l1r.add_argument("--l1", default=defs["l1"])
    l1r.add_argument("--l1-fewshot", default=defs["l1_fs"])

    # l2-only
    p_l2 = sub.add_parser("l2-only", help="L2 phase (needs prior L1 results)")
    sp_l2 = p_l2.add_subparsers(dest="action", required=True)
    l2s = sp_l2.add_parser("submit")
    l2s.add_argument("--l2", default=defs["l2"])
    l2s.add_argument("--l2-fewshot", default=defs["l2_fs"])
    l2r = sp_l2.add_parser("retrieve")
    l2r.add_argument("--l2", default=defs["l2"])
    l2r.add_argument("--l2-fewshot", default=defs["l2_fs"])

    args = ap.parse_args()

    dispatch = {"parallel": _parallel, "l1-only": _l1_only, "l2-only": _l2_only}
    dispatch[args.strategy](args, base_dir, defs)


if __name__ == "__main__":
    main()
