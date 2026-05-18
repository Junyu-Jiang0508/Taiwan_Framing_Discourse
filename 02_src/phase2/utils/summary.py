"""Single rolling Phase 2 run log (one markdown file under artifacts_root)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PHASE2_SUMMARY_FILENAME = "phase2_summary.md"


def phase2_summary_path(artifacts_root: Path) -> Path:
    return artifacts_root / PHASE2_SUMMARY_FILENAME


def _format_section(
    section: str,
    *,
    params: Dict[str, Any],
    outputs: List[str],
    stats: Dict[str, Any],
    notes: Optional[List[str]] = None,
    elapsed_sec: Optional[float] = None,
) -> str:
    lines = [
        f"## {section}",
        "",
        f"_Updated: {datetime.now(timezone.utc).isoformat()}_",
        "",
    ]
    if params:
        lines.append("**Parameters**")
        for k, v in params.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    if outputs:
        lines.append("**Outputs**")
        for o in outputs:
            lines.append(f"- `{o}`")
        lines.append("")
    if stats or elapsed_sec is not None:
        lines.append("**Stats**")
        for k, v in stats.items():
            lines.append(f"- {k}: {v}")
        if elapsed_sec is not None:
            lines.append(f"- elapsed_sec: {elapsed_sec:.2f}")
        lines.append("")
    if notes:
        lines.append("**Notes**")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _upsert_section(body: str, section: str, block: str) -> str:
    pattern = re.compile(
        rf"^## {re.escape(section)}\s*$.*?(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    body = pattern.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body:
        body = "# Phase 2 pipeline log"
    return body + "\n\n" + block


def write_summary(
    artifacts_root: Path,
    section: str,
    *,
    params: Dict[str, Any],
    outputs: List[str],
    stats: Dict[str, Any],
    notes: Optional[List[str]] = None,
    elapsed_sec: Optional[float] = None,
) -> None:
    """Append or replace one section in artifacts_root/phase2_summary.md."""
    path = phase2_summary_path(artifacts_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        body = path.read_text(encoding="utf-8")
    else:
        body = "# Phase 2 pipeline log\n"
    block = _format_section(
        section,
        params=params,
        outputs=outputs,
        stats=stats,
        notes=notes,
        elapsed_sec=elapsed_sec,
    )
    path.write_text(_upsert_section(body, section, block), encoding="utf-8")
