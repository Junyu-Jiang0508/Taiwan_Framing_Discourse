"""Tests for corpus hash and manifests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_src" / "phase2"))

from utils.config import compute_corpus_hash, resolve_corpus_csv  # noqa: E402


def test_corpus_hash_stable():
    ptr = "03_outputs/01_results_labelings/01_results_datasets/latest_merged_deduped_run.txt"
    csv = resolve_corpus_csv(ROOT, ptr)
    h1 = compute_corpus_hash(csv)
    h2 = compute_corpus_hash(csv)
    assert h1 == h2
    assert len(h1) == 64


if __name__ == "__main__":
    test_corpus_hash_stable()
    print("test_manifest: ok")
