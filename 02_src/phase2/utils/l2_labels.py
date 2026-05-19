"""L2 v10 codebook labels (DPM scheme) for Phase 2 reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

_CODEBOOK_PATH = (
    Path(__file__).resolve().parents[3]
    / "01_data/05_labels_guidance/02_annotation_guide_label2_v10.csv"
)

_L2_LABELS: Dict[str, Dict[str, str]] | None = None


def load_l2_labels() -> Dict[str, Dict[str, str]]:
    """Return {label_id: {cn, en, core_definition}} from v10 codebook."""
    global _L2_LABELS
    if _L2_LABELS is not None:
        return _L2_LABELS
    df = pd.read_csv(_CODEBOOK_PATH)
    _L2_LABELS = {
        str(row["label_id"]): {
            "cn": str(row["label_cn"]),
            "en": str(row["label_en"]),
            "core_definition": str(row["core_definition"]),
        }
        for _, row in df.iterrows()
    }
    return _L2_LABELS


def l2_display(label_id: str) -> str:
    """Format as L2-XX (中文名 / English name)."""
    info = load_l2_labels().get(label_id)
    if not info:
        return label_id
    return f"{label_id} ({info['cn']} / {info['en']})"


def l2_set_display(l2_ids: list[str] | set[str]) -> str:
    return ", ".join(l2_display(x) for x in sorted(l2_ids))
