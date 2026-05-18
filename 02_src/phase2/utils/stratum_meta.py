"""Parse stratum keys into row metadata for partitions and graphs."""
from __future__ import annotations

from typing import Any, Dict, List


def stratum_row_fields(scheme_name: str, key_str: str, groupby: List[str]) -> Dict[str, Any]:
    parts = key_str.split("_")
    row: Dict[str, Any] = {"stratum_key": key_str}
    if scheme_name == "camp_genre" and len(groupby) == 2 and len(parts) >= 2:
        row["camp"] = parts[0]
        row["genre"] = parts[1]
    elif scheme_name == "camp_time" and len(groupby) == 2 and len(parts) >= 2:
        row["camp"] = parts[0]
        row["time_bucket"] = "_".join(parts[1:])
    elif scheme_name == "camp" and len(groupby) == 1:
        row["camp"] = key_str
        row["genre"] = None
    else:
        for i, col in enumerate(groupby):
            row[col] = parts[i] if i < len(parts) else None
    return row
