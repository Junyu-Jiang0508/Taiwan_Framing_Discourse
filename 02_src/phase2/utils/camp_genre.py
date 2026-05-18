"""Camp and genre derivation for phase 2 freeze."""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Set, Tuple

import pandas as pd

_SENT_IDX_RE = re.compile(r"_u(\d+)$", re.IGNORECASE)


def parse_l2_cell(s: Any) -> List[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def infer_candidate_party(source_blob: str) -> Tuple[str, str]:
    s = str(source_blob).lower()
    if "01_ho" in s or "hou" in s or "侯" in str(source_blob):
        return "侯友宜", "KMT"
    if "02_ke" in s or "柯" in str(source_blob):
        return "柯文哲", "TPP"
    if "03_lai" in s or "lai" in s or "赖" in str(source_blob) or "賴" in str(source_blob):
        return "赖清德", "DPP"
    return "Unknown", "Unknown"


def infer_genre(row: pd.Series) -> str:
    sf = str(row.get("_source_file", row.get("source_file", ""))).lower()
    src = str(row.get("source", "")).lower()
    blob = sf + " " + src
    if "03_x_datasets" in blob or "x_datasets" in blob or "twitter" in blob:
        return "social"
    if "conference" in blob or "debate" in blob or "辩论" in str(row.get("source", "")):
        return "debate"
    return "news"


def parse_speaker_parties(speakers: Any) -> Set[str]:
    if pd.isna(speakers) or not str(speakers).strip():
        return set()
    try:
        arr = json.loads(speakers) if isinstance(speakers, str) else speakers
    except (json.JSONDecodeError, TypeError):
        return set()
    if not isinstance(arr, list):
        return set()
    parties = set()
    for item in arr:
        if isinstance(item, dict) and item.get("party"):
            parties.add(str(item["party"]).strip())
    return parties


def primary_speaker_party(speakers: Any) -> str:
    if pd.isna(speakers) or not str(speakers).strip():
        return "Unknown"
    try:
        arr = json.loads(speakers) if isinstance(speakers, str) else speakers
    except (json.JSONDecodeError, TypeError):
        return "Unknown"
    if not isinstance(arr, list):
        return "Unknown"
    for item in arr:
        if isinstance(item, dict) and item.get("party"):
            p = str(item["party"]).strip()
            if p:
                return p
    return "Unknown"


def derive_camp(row: pd.Series, genre: str) -> str:
    if genre == "debate":
        camp = primary_speaker_party(row.get("speakers"))
        if camp != "Unknown":
            return camp
    blob = str(row.get("source", "")) + " " + str(row.get("_source_file", ""))
    _, party = infer_candidate_party(blob)
    return party


def derive_sent_idx(row: pd.Series, fallback: int) -> int:
    uid = str(row.get("unit_id", ""))
    m = _SENT_IDX_RE.search(uid)
    if m:
        return int(m.group(1))
    return fallback


def stratum_key_from_row(row: pd.Series, groupby: List[str]) -> Tuple:
    return tuple(row[k] for k in groupby)


def format_stratum_key(key: Tuple, groupby: List[str]) -> str:
    if len(groupby) == 1:
        return str(key[0])
    return "_".join(str(x) for x in key)
