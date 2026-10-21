"""Shared paths, label sets and helpers for the Claude re-annotation sub-project."""
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "06_claude_reannotation"
PROTOCOL = SUB / "protocol"
INPUT = SUB / "input"
RUNS = SUB / "runs"
EVAL = SUB / "eval"

# Neutral working directory for `claude -p`: outside the repo, so the project's
# auto-memory and any CLAUDE.md are never injected into an annotation call.
NEUTRAL_CWD = Path.home() / "scratch" / "claude_annot_cwd"

GUIDES = ROOT / "01_data" / "05_labels_guidance"
L1_GUIDE = GUIDES / "01_annotation_guide_label1_v9.csv"
L2_GUIDE = GUIDES / "02_annotation_guide_label2_v10.csv"
L1_FEWSHOT = GUIDES / "03_fewshot_L1_v9.csv"
L2_FEWSHOT = GUIDES / "03_fewshot_L2_v10.csv"
HARD_POOL = GUIDES / "04_hard_case_pool.csv"

GOLD = ROOT / "01_data" / "06_validation_sets" / "04_validation_set_v2" / "label1_set_v2.csv"
ZXH = ROOT / "01_data" / "06_validation_sets" / "05_validation_set" / "01_ZXH.csv"
ANALYSIS_FRAME = ROOT / "03_outputs" / "01_results_labelings" / "10_merged_master" / "analysis_frame_n20901.csv"
MERGED_DEDUPED = ROOT / "01_data" / "04_cleaned_datasets" / "merged_deduped.csv"
GPT_RUN = ROOT / "03_outputs" / "01_results_labelings" / "01_results_datasets" / "Run_20260407_221407" / "final_results.csv"
DS_L2_RUN = ROOT / "03_outputs" / "01_results_labelings" / "01_results_datasets" / "Run_deepseek_l2binary" / "final_results.csv"
DS_L1_RUN = ROOT / "L1_Deepseek" / "l1_results.csv"

L1 = [f"L1-0{i}" for i in range(1, 8)]
L2 = [f"L2-0{i}" for i in range(1, 9)]


def norm(s):
    """Normalize text for matching: NFKC, drop whitespace, punctuation and emoji."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\s​﻿]+", "", s)
    return re.sub(r"[^\w一-鿿]", "", s)


def parse_l2(v):
    """'L2-01|L2-05' / 'L2-01, L2-05' / NaN -> frozenset of valid L2 codes."""
    if not isinstance(v, str) or not v.strip():
        return frozenset()
    return frozenset(t.strip() for t in re.split(r"[|,;\s]+", v) if t.strip() in L2)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def overlaps(text_a, text_b, window=10, step=1):
    """True if any `window`-char slice of normalized text_a occurs in normalized text_b.

    Window 10 / step 1 catches codebook and few-shot examples paraphrased from a gold sentence
    (a 12-char window at step 4 missed several, e.g. gold 264, 248, 483).
    """
    a, b = norm(text_a), norm(text_b)
    if len(a) < window:
        return False
    return any(a[i:i + window] in b for i in range(0, len(a) - window + 1, step))


def match_gold_to_units(gold, corpus):
    """Map each gold sentence to a corpus unit_id: exact normalized match, then containment, then fuzzy >= .90.

    `corpus` needs columns unit_id, sentence. Returns two lists (unit_ids, how).
    """
    key2uid = {}
    for k, u in zip(corpus["sentence"].map(norm), corpus["unit_id"]):
        key2uid.setdefault(k, u)
    keys = list(key2uid)
    uids, how = [], []
    for k in gold["sentence"].map(norm):
        if k in key2uid:
            uids.append(key2uid[k]); how.append("exact"); continue
        cand = [kk for kk in keys if len(k) > 15 and (k in kk or kk in k)
                and min(len(k), len(kk)) / max(len(k), len(kk)) > 0.7]
        if cand:
            uids.append(key2uid[cand[0]]); how.append("contain"); continue
        best, br = None, 0.0
        for kk in keys:
            if abs(len(kk) - len(k)) > 0.2 * max(len(k), 1):
                continue
            if SequenceMatcher(None, k, kk).quick_ratio() >= 0.9:
                r = SequenceMatcher(None, k, kk).ratio()
                if r > br:
                    best, br = kk, r
        if best is not None and br >= 0.9:
            uids.append(key2uid[best]); how.append("fuzzy"); continue
        uids.append(None); how.append("none")
    return uids, how


def user_text(prev, sentence, nxt):
    """Same user-message layout as the GPT/DeepSeek pipelines (framing_batch_core._build_user_text)."""
    prev = (prev or "").strip()
    nxt = (nxt or "").strip()
    sentence = (sentence or "").strip()
    if not prev and not nxt:
        return f"請標註以下文本：\n\n{sentence}"
    return (
        f"【前文（僅供語境參考，不標註）】{prev}\n"
        f"【本句（請標註）】{sentence}\n"
        f"【後文（僅供語境參考，不標註）】{nxt}"
    )


def clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()
