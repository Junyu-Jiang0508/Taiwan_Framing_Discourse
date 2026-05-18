"""Idempotency manifests per module."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs_hash(paths: List[Path], extra: Optional[Dict[str, Any]] = None) -> str:
    parts = []
    for p in sorted(paths, key=lambda x: str(x)):
        if p.is_file():
            parts.append(f"{p}:{file_sha256(p)}")
        else:
            parts.append(f"{p}:missing")
    if extra:
        parts.append(json.dumps(extra, sort_keys=True))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**data, "written_at": datetime.now(timezone.utc).isoformat()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def should_skip(manifest_path: Path, expected: Dict[str, Any], force: bool) -> bool:
    if force:
        return False
    existing = load_manifest(manifest_path)
    if not existing:
        return False
    for k, v in expected.items():
        if existing.get(k) != v:
            return False
    return True
