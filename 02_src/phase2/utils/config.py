"""Load phase2_config.yaml, resolve paths, corpus content hash."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PHASE2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE2_ROOT.parents[1]
DEFAULT_CONFIG = PHASE2_ROOT / "phase2_config.yaml"


@dataclass
class StratificationScheme:
    name: str
    groupby: List[str]


@dataclass
class Phase2Config:
    raw: Dict[str, Any]
    repo_root: Path
    artifacts_root: Path
    corpus_csv: Path
    corpus_content_hash: str
    schemes: List[StratificationScheme] = field(default_factory=list)

    @property
    def camps(self) -> List[str]:
        return list(self.raw["strata"]["camps"])

    @property
    def genres_main(self) -> List[str]:
        return list(self.raw["strata"]["genres_main"])

    def scheme_dir(self, prefix: str, scheme_name: str) -> Path:
        return self.artifacts_root / f"{prefix}_by_{scheme_name}"

    def bootstrap_n_for_mode(self, mode: str) -> int:
        b = self.raw["bootstrap"]
        if mode == "smoke":
            return int(self.raw["smoke"]["bootstrap_n"])
        if mode == "dev":
            return int(b["n_resamples_dev"])
        return int(b["n_resamples"])


def resolve_corpus_csv(repo_root: Path, pointer_rel: str) -> Path:
    pointer = repo_root / pointer_rel
    if not pointer.is_file():
        raise FileNotFoundError(f"Corpus pointer not found: {pointer}")
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    csv_path = run_dir / "final_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"final_results.csv not found: {csv_path}")
    return csv_path.resolve()


def compute_corpus_hash(csv_path: Path) -> str:
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(
    config_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    corpus_csv: Optional[Path] = None,
) -> Phase2Config:
    repo_root = (repo_root or REPO_ROOT).resolve()
    cfg_path = (config_path or DEFAULT_CONFIG).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if corpus_csv is None:
        corpus_csv = resolve_corpus_csv(repo_root, raw["paths"]["corpus_pointer"])
    else:
        corpus_csv = Path(corpus_csv).resolve()

    artifacts = repo_root / raw["paths"]["artifacts_root"]
    schemes = [
        StratificationScheme(name=s["name"], groupby=list(s["groupby"]))
        for s in raw["strata"]["stratification_schemes"]
    ]
    return Phase2Config(
        raw=raw,
        repo_root=repo_root,
        artifacts_root=artifacts,
        corpus_csv=corpus_csv,
        corpus_content_hash=compute_corpus_hash(corpus_csv),
        schemes=schemes,
    )


def config_subset_hash(cfg: Phase2Config, keys: List[str]) -> str:
    subset = {}
    for k in keys:
        parts = k.split(".")
        node = cfg.raw
        for p in parts:
            node = node[p]
        subset[k] = node
    return hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()
