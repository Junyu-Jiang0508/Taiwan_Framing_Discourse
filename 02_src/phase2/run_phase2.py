#!/usr/bin/env python3
"""Run Phase 2 analysis DAG."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE2_DIR))

from utils.config import load_config  # noqa: E402
from utils.smoke import select_smoke_doc_ids_from_corpus  # noqa: E402

import p2_00_freeze  # noqa: E402
import p2_01_windows  # noqa: E402
import p2_02_npmi  # noqa: E402
import p2_03_bootstrap  # noqa: E402
import p2_04_networks  # noqa: E402
import p2_05_leiden  # noqa: E402
import p2_06_cross_camp  # noqa: E402
import p2_07_exemplars  # noqa: E402
import p2_08_visualization  # noqa: E402

MODULES = {
    "p2_00": p2_00_freeze,
    "p2_01": p2_01_windows,
    "p2_02": p2_02_npmi,
    "p2_03": p2_03_bootstrap,
    "p2_04": p2_04_networks,
    "p2_05": p2_05_leiden,
    "p2_06": p2_06_cross_camp,
    "p2_07": p2_07_exemplars,
    "p2_08": p2_08_visualization,
}

ORDER = ["p2_00", "p2_01", "p2_02", "p2_03", "p2_04", "p2_05", "p2_06", "p2_07", "p2_08"]


def run_pipeline(
    mode: str = "full",
    only: str | None = None,
    scheme: str | None = None,
    force: bool = False,
    config_path: Path | None = None,
) -> None:
    cfg = load_config(config_path)
    if mode == "smoke":
        cfg.raw["npmi"]["min_marginal_count"] = 1
        cfg.raw["npmi"]["min_stratum_windows"] = 1
    cfg.artifacts_root.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_root / "manifests").mkdir(parents=True, exist_ok=True)

    doc_ids = list(cfg.raw["smoke"].get("doc_ids") or []) if mode == "smoke" else None
    if mode == "smoke" and not doc_ids:
        doc_ids = select_smoke_doc_ids_from_corpus(cfg.corpus_csv)
        print(f"Smoke doc_ids: {doc_ids}")

    modules = [only] if only else ORDER
    if only and only not in MODULES:
        raise SystemExit(f"Unknown module: {only}")

    for name in modules:
        print(f"\n=== {name} (mode={mode}) ===")
        if name == "p2_00":
            p2_00_freeze.run(cfg, force=force, doc_ids=doc_ids if mode == "smoke" else None)
        elif name == "p2_01":
            p2_01_windows.run(cfg, force=force)
        elif name == "p2_02":
            p2_02_npmi.run(cfg, force=force, scheme_filter=scheme)
        elif name == "p2_03":
            p2_03_bootstrap.run(cfg, mode=mode, force=force, scheme_filter=scheme)
        elif name == "p2_04":
            p2_04_networks.run(cfg, force=force, scheme_filter=scheme)
        elif name == "p2_05":
            p2_05_leiden.run(cfg, force=force, scheme_filter=scheme)
        elif name == "p2_06":
            p2_06_cross_camp.run(cfg, force=force)
        elif name == "p2_07":
            p2_07_exemplars.run(cfg, force=force)
        elif name == "p2_08":
            p2_08_visualization.run(cfg, force=force)

    print(f"\nPhase 2 pipeline complete (mode={mode}).")


def main():
    ap = argparse.ArgumentParser(description="Phase 2 analysis pipeline")
    ap.add_argument("--mode", choices=["smoke", "dev", "full"], default="full")
    ap.add_argument("--only", type=str, default=None, help="Run single module e.g. p2_03")
    ap.add_argument("--scheme", type=str, default=None, help="Limit p2_02-05 to scheme name")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    run_pipeline(
        mode=args.mode,
        only=args.only,
        scheme=args.scheme,
        force=args.force,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
