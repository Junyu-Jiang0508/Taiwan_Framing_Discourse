"""Stage 1 — Raw datasets → document normalisation → sentence-level analysis units.

Merges the former ``00_preprocess_to_units.py`` (document→unit pipeline) and
``01_convert_units_to_csv.py`` (JSONL→CSV format conversion).

Usage::

    python s01_preprocess.py                       # default dirs
    python s01_preprocess.py --raw … --output …    # custom paths
    python s01_preprocess.py convert-csv            # batch JSONL→CSV only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata as ud
from collections import Counter
from pathlib import Path

import pandas as pd

from config import (
    ACTOR_PARTY_MAP,
    CLOSERS,
    COMBO_ENDS,
    ELLIPSIS_ENDS,
    ENDERS,
    HTML_RE,
    INST_CATEGORY_MAP,
    OPENERS,
    PAIRS,
    PROC_SYMBOLIC_TERMS,
    PROMO_PATTERNS,
    QUOTE_RE,
    REQ_COLS,
    SAY_VERBS_RE,
    URL_RE,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Column normalisation
# ═══════════════════════════════════════════════════════════════════════════

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapper = {
        "pub_time": "date", "pubDate": "date", "publish_time": "date",
        "正文": "content", "text": "content", "body": "content",
        "链接": "url", "link": "url",
        "Tweet Date": "date", "Tweet Content": "content",
    }
    df = df.rename(columns={k: v for k, v in mapper.items() if k in df.columns})
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = ""

    def _parse_dt(x):
        try:
            return pd.to_datetime(x).tz_localize(None)
        except Exception:
            return pd.NaT

    df["date_std"] = df["date"].apply(_parse_dt)
    df["date_parse_ok"] = df["date_std"].notna()

    if df["doc_id"].eq("").all():
        def _mk_id(row):
            h = hashlib.md5(
                (str(row.get("url", "")) + str(row.get("title", ""))
                 + str(row.get("content", ""))[:100]).encode("utf-8")
            ).hexdigest()[:12]
            return f'{row.get("source", "UNK")}_{h}'
        df["doc_id"] = df.apply(_mk_id, axis=1)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Text cleaning helpers
# ═══════════════════════════════════════════════════════════════════════════

def light_clean(text: str) -> tuple[str, dict]:
    meta = {"url_removed": 0, "html_removed": 0, "promo_trim": 0, "space_norm": 0}
    if not isinstance(text, str) or not text.strip():
        return "", meta
    t = text
    urls = URL_RE.findall(t)
    meta["url_removed"] = len(urls)
    t = URL_RE.sub("<URL>", t)
    htmls = HTML_RE.findall(t)
    meta["html_removed"] = len(htmls)
    t = HTML_RE.sub("<HTML>", t)
    for pat in PROMO_PATTERNS:
        t2 = re.sub(pat, "", t)
        if t2 != t:
            meta["promo_trim"] += 1
            t = t2
    t2 = re.sub(r"\s+", " ", t).strip()
    if t2 != t:
        meta["space_norm"] += 1
    return t2, meta


def nfkc(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return ud.normalize("NFKC", text)


def make_dual_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for t in df["content"].fillna(""):
        cleaned, meta = light_clean(t)
        normed = nfkc(cleaned)
        results.append((t, cleaned, normed, meta))
    df[["content_orig", "content_clean", "content_norm", "clean_meta"]] = pd.DataFrame(
        results, index=df.index
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Sentence segmentation
# ═══════════════════════════════════════════════════════════════════════════

def _protected_spans(text: str):
    stack = []
    spans = []
    for i, ch in enumerate(text):
        if ch in OPENERS:
            stack.append((ch, i))
        elif ch in CLOSERS:
            for k in range(len(stack) - 1, -1, -1):
                op, pos = stack[k]
                if PAIRS.get(op) == ch:
                    spans.append((pos, i))
                    stack = stack[:k]
                    break
    spans.sort()
    return spans


def _in_spans(idx, spans):
    for s, e in spans:
        if s <= idx <= e:
            return True
        if idx < s:
            return False
    return False


def split_sentences(text: str, max_len_for_secondary: int = 180) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    spans = _protected_spans(text)
    n = len(text)
    i = 0
    pieces: list[str] = []
    start = 0

    def _is_ender_at(pos):
        ch2 = text[pos:pos + 2]
        ch3 = text[pos:pos + 3]
        if ch2 in COMBO_ENDS or ch2 in ELLIPSIS_ENDS:
            return 2
        if ch3 == "...":
            return 3
        if text[pos] in ENDERS:
            return 1
        return 0

    while i < n:
        if _in_spans(i, spans):
            i += 1
            continue
        step = _is_ender_at(i)
        if step:
            j = i + step
            while j < n and (text[j] in CLOSERS or text[j].isspace()):
                j += 1
            seg = text[start:j].strip()
            if seg:
                pieces.append(seg)
            start = j
            i = j
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)

    merged = []
    for seg in pieces:
        if len(seg) < 6 and merged:
            merged[-1] = (merged[-1] + " " + seg).strip()
        else:
            merged.append(seg)

    connector = re.compile(
        r"(但是|然而|不过|不過|因为|因為|因此|所以|同时|同時|以及|并且|並且|或者|或是|例如)"
    )
    final: list[str] = []
    for seg in merged:
        if len(seg) <= max_len_for_secondary:
            final.append(seg)
            continue
        loc_spans = _protected_spans(seg)
        tmp: list[str] = []
        last = 0
        for m in re.finditer(r"[；;]", seg):
            if not _in_spans(m.start(), loc_spans):
                cut = seg[last:m.end()].strip()
                if cut:
                    tmp.append(cut)
                last = m.end()
        tail2 = seg[last:].strip()
        if tail2:
            tmp.append(tail2)

        sec: list[str] = []
        for chunk in tmp:
            if len(chunk) <= max_len_for_secondary:
                sec.append(chunk)
                continue
            loc2 = _protected_spans(chunk)
            spl = False
            for m2 in connector.finditer(chunk):
                if not _in_spans(m2.start(), loc2):
                    left = chunk[:m2.start()].strip()
                    right = chunk[m2.start():].strip()
                    if left:
                        sec.append(left)
                    if right:
                        sec.append(right)
                    spl = True
                    break
            if not spl:
                sec.append(chunk)
        final.extend(sec)
    return final


# ═══════════════════════════════════════════════════════════════════════════
#  Unit extraction & enrichment
# ═══════════════════════════════════════════════════════════════════════════

def classify_unit(sent: str) -> str:
    if QUOTE_RE.search(sent):
        return "quote"
    if SAY_VERBS_RE.search(sent):
        return "claim"
    return "context"


def extract_units(doc_row) -> list[dict]:
    sents = split_sentences(doc_row["content_norm"])
    merged_sents: list[str] = []
    i = 0
    while i < len(sents):
        current = sents[i]
        if i + 1 < len(sents):
            next_sent = sents[i + 1]
            if (len(current) <= 20
                    and SAY_VERBS_RE.search(current)
                    and re.search(r"[：:，,]?$", current)
                    and re.match(r'^["\'『「]', next_sent)):
                merged_sents.append(current + " " + next_sent)
                i += 2
                continue
        merged_sents.append(current)
        i += 1

    units = []
    for idx, s in enumerate(merged_sents):
        units.append({
            "doc_id": doc_row["doc_id"],
            "unit_id": f'{doc_row["doc_id"]}_u{idx:03d}',
            "text": s,
            "role": classify_unit(s),
            "prev": merged_sents[idx - 1] if idx > 0 else "",
            "next": merged_sents[idx + 1] if idx + 1 < len(merged_sents) else "",
        })
    return units


def find_speakers(text: str) -> list[dict]:
    hits, seen = [], set()
    for k, party in ACTOR_PARTY_MAP.items():
        if k in text and k not in seen:
            hits.append({"name": k, "party": party})
            seen.add(k)
    return hits


def find_targets(text: str) -> list[str]:
    targets, seen = [], set()
    for k, cat in INST_CATEGORY_MAP.items():
        if k in text and k not in seen:
            targets.append(f"{k}|{cat}")
            seen.add(k)
    return targets


def enrich_roles(units: list[dict]) -> list[dict]:
    for u in units:
        u["speakers"] = find_speakers(u["text"])
        u["targets"] = find_targets(u["text"])
    return units


def add_event_flags(u: dict) -> dict:
    txt = u["text"]
    u["proc_symbolic_flag"] = any(k in txt for k in PROC_SYMBOLIC_TERMS)
    if any(k in txt for k in ["政见", "辩论", "政見", "辯論"]):
        u["event_phase"] = "debate/manifesto"
    elif any(k in txt for k in ["表决", "三读", "版本", "協商", "协商", "表決", "三讀"]):
        u["event_phase"] = "legislative"
    elif any(k in txt for k in ["造势", "拜票", "扫街", "掃街", "造勢"]):
        u["event_phase"] = "campaigning"
    else:
        u["event_phase"] = ""
    return u


# ═══════════════════════════════════════════════════════════════════════════
#  Deduplication & export
# ═══════════════════════════════════════════════════════════════════════════

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def mark_duplicates(df_docs: pd.DataFrame, units: list[dict]):
    df_docs["doc_hash"] = df_docs["content_norm"].apply(_md5)
    df_docs["doc_dup_flag"] = df_docs["doc_hash"].duplicated(keep="first")
    seen: set[str] = set()
    for u in units:
        h = _md5(u["text"])
        u["unit_hash"] = h
        u["unit_dup_flag"] = h in seen
        seen.add(h)
    return df_docs, units


def to_llm_jsonl(units: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in units:
            rec = {
                "doc_id": u["doc_id"],
                "unit_id": u["unit_id"],
                "text": u["text"],
                "role": u["role"],
                "prev": u["prev"],
                "next": u["next"],
                "speakers": u.get("speakers", []),
                "targets": u.get("targets", []),
                "proc_symbolic_flag": u.get("proc_symbolic_flag", False),
                "event_phase": u.get("event_phase", ""),
                "unit_hash": u.get("unit_hash", ""),
                "unit_dup_flag": u.get("unit_dup_flag", False),
                "source_meta": {
                    "source": u.get("source", ""),
                    "date": u.get("date", ""),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_to_units(df_raw: pd.DataFrame, source_name: str):
    df = normalize_cols(df_raw)
    df["source"] = df["source"].replace("", source_name)
    df = make_dual_text_cols(df)

    df_docs_out = df[[
        "doc_id", "source", "title", "url", "date", "date_std", "date_parse_ok",
        "content_orig", "content_clean", "content_norm", "clean_meta",
    ]].copy()

    all_units: list[dict] = []
    for _, row in df.iterrows():
        if not row["content_norm"] or len(row["content_norm"].strip()) < 10:
            continue
        us = extract_units(row)
        us = enrich_roles(us)
        us = [add_event_flags(u) for u in us]
        for u in us:
            u["source"] = row["source"]
            u["date"] = str(row["date_std"]) if pd.notna(row["date_std"]) else str(row["date"])
        all_units.extend(us)

    df_docs_out, all_units = mark_duplicates(df_docs_out, all_units)
    return df_docs_out, all_units


def quick_report(df_docs, units, source_name: str = ""):
    print(f"\n{'=' * 60}\nSource: {source_name}\n{'=' * 60}")
    print(f"Documents: {len(df_docs)}")
    print(f"Units: {len(units)}")
    print(f"Doc duplicate rate: {df_docs['doc_dup_flag'].mean():.2%}")
    c_role = Counter(u["role"] for u in units)
    print("Unit type distribution:")
    for role, count in c_role.items():
        print(f"  {role}: {count} ({count / len(units):.2%})")
    proc_count = sum(u["proc_symbolic_flag"] for u in units)
    print(f"Procedural/symbolic: {proc_count} ({proc_count / len(units):.2%})")
    speaker_count = sum(1 for u in units if u["speakers"])
    print(f"With speakers: {speaker_count} ({speaker_count / len(units):.2%})")


# ═══════════════════════════════════════════════════════════════════════════
#  Conference / debate TXT parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_conference_txt(txt_path: str, source_name: str) -> pd.DataFrame:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records: list[dict] = []
    current_speaker, current_content = "", []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 20 and not any(p in line for p in ["。", "！", "？", "，"]):
            if current_content:
                records.append({"speaker": current_speaker, "content": " ".join(current_content)})
                current_content = []
            current_speaker = line
        else:
            current_content.append(line)
    if current_content:
        records.append({"speaker": current_speaker, "content": " ".join(current_content)})
    df = pd.DataFrame(records)
    df["source"] = source_name
    df["title"] = Path(txt_path).stem
    df["date"] = ""
    df["url"] = ""
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  JSONL → CSV conversion  (formerly 01_convert_units_to_csv.py)
# ═══════════════════════════════════════════════════════════════════════════

def convert_jsonl_to_csv(jsonl_path: str, csv_path: str) -> int:
    units = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            unit = json.loads(line)
            row = {
                "doc_id": unit["doc_id"],
                "unit_id": unit["unit_id"],
                "sentence": unit["text"],
                "role": unit["role"],
                "prev": unit["prev"],
                "next": unit["next"],
                "speakers": json.dumps(unit["speakers"], ensure_ascii=False),
                "targets": json.dumps(unit["targets"], ensure_ascii=False),
                "proc_symbolic_flag": unit["proc_symbolic_flag"],
                "event_phase": unit["event_phase"],
                "unit_hash": unit["unit_hash"],
                "unit_dup_flag": unit["unit_dup_flag"],
                "source": unit["source_meta"]["source"],
                "date": unit["source_meta"]["date"],
            }
            units.append(row)
    df = pd.DataFrame(units)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return len(df)


# ═══════════════════════════════════════════════════════════════════════════
#  Batch runners
# ═══════════════════════════════════════════════════════════════════════════

def _read_csv_multienc(csv_file):
    for enc in ("utf-8", "gbk", "gb2312", "gb18030", "big5"):
        try:
            return pd.read_csv(csv_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    return None


def process_all_datasets(raw_dir: str, output_dir: str):
    raw_path, output_path = Path(raw_dir), Path(output_dir)
    for sub in ("01_news_datasets", "02_conference_datasets", "03_X_datasets"):
        (output_path / sub).mkdir(parents=True, exist_ok=True)

    for sub, ext, parser in [
        ("01_news_datasets", "*.csv", None),
        ("03_X_datasets", "*.csv", None),
    ]:
        print(f"\n{'=' * 80}\nProcessing {sub}\n{'=' * 80}")
        for f in (raw_path / sub).glob(ext):
            print(f"\nProcessing: {f.name}")
            try:
                df_raw = _read_csv_multienc(f)
                if df_raw is None:
                    print("  Failed to read with any encoding, skipping")
                    continue
                df_docs, units = preprocess_to_units(df_raw, f.stem)
                df_docs.to_csv(output_path / sub / f"{f.stem}_docs.csv", index=False, encoding="utf-8")
                to_llm_jsonl(units, str(output_path / sub / f"{f.stem}_units.jsonl"))
                quick_report(df_docs, units, f.stem)
            except Exception as e:
                print(f"Error: {e}")

    print(f"\n{'=' * 80}\nProcessing 02_conference_datasets\n{'=' * 80}")
    conf_dir = raw_path / "02_conference_datasets"
    for txt_file in conf_dir.glob("*.txt"):
        print(f"\nProcessing: {txt_file.name}")
        try:
            df_raw = parse_conference_txt(str(txt_file), txt_file.stem)
            df_docs, units = preprocess_to_units(df_raw, txt_file.stem)
            df_docs.to_csv(output_path / "02_conference_datasets" / f"{txt_file.stem}_docs.csv",
                           index=False, encoding="utf-8")
            to_llm_jsonl(units, str(output_path / "02_conference_datasets" / f"{txt_file.stem}_units.jsonl"))
            quick_report(df_docs, units, txt_file.stem)
        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'=' * 80}\nAll processing complete!\n{'=' * 80}")


def batch_convert_jsonl_to_csv():
    from config import PROCESSED_DIR, DATASETS
    base_dir = Path(PROCESSED_DIR)
    total_units, total_files = 0, 0
    for dataset in DATASETS:
        dataset_dir = base_dir / dataset
        if not dataset_dir.exists():
            continue
        for jsonl_file in dataset_dir.glob("*_units.jsonl"):
            csv_file = jsonl_file.with_suffix(".csv")
            try:
                n = convert_jsonl_to_csv(str(jsonl_file), str(csv_file))
                print(f"ok {jsonl_file.name} -> {csv_file.name} ({n})")
                total_units += n
                total_files += 1
            except Exception as e:
                print(f"fail {jsonl_file.name}: {e}")
    print(f"done files={total_files} units={total_units}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Stage 1: Raw data → sentence-level units")
    sub = ap.add_subparsers(dest="command")

    p_proc = sub.add_parser("preprocess", help="Full preprocessing pipeline")
    p_proc.add_argument("--raw", default="01_data/01_raw_datasets")
    p_proc.add_argument("--output", default="01_data/02_processed_datasets")

    sub.add_parser("convert-csv", help="Batch-convert JSONL units to CSV")

    args = ap.parse_args()
    if args.command == "convert-csv":
        batch_convert_jsonl_to_csv()
    else:
        process_all_datasets(
            args.raw if hasattr(args, "raw") else "01_data/01_raw_datasets",
            args.output if hasattr(args, "output") else "01_data/02_processed_datasets",
        )


if __name__ == "__main__":
    main()
