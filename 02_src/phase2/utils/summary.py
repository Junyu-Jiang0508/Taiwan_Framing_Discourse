"""Auto-generate module summary markdown."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_summary(
    path: Path,
    module: str,
    *,
    params: Dict[str, Any],
    outputs: List[str],
    stats: Dict[str, Any],
    notes: Optional[List[str]] = None,
    elapsed_sec: Optional[float] = None,
) -> None:
    lines = [
        f"# {module} summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Parameters",
        "",
    ]
    for k, v in params.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.extend(["", "## Outputs", ""])
    for o in outputs:
        lines.append(f"- `{o}`")
    lines.extend(["", "## Key statistics", ""])
    for k, v in stats.items():
        lines.append(f"- **{k}**: {v}")
    if elapsed_sec is not None:
        lines.append(f"- **elapsed_sec**: {elapsed_sec:.2f}")
    if notes:
        lines.extend(["", "## Notes", ""])
        for n in notes:
            lines.append(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
