import pandas as pd
import unicodedata as ud
import re
import hashlib
import json
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

REQ_COLS = ["doc_id", "source", "date", "title", "content", "url"]
OPTIONAL = ["section", "author", "candidate_hint"]

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")

PROMO_PATTERNS = [
    r"（.+?）(?:下載|订阅|訂閱|更多內容|點我|点我|延伸閱讀).{0,40}$",
    r"^\s*（(?:記者|记者).+?攝）\s*",
    r"透過\s*Google\s*News\s*$",
    r"編輯：.{2,10}\）\d{7}$",
]

QUOTE_RE = re.compile(r'["\'『](.+?)["\'』]')
SAY_VERBS = r"(表示|說|说|稱|称|指出|批評|批评|強調|强调|回應|回应|呼籲|呼吁|主張|主张|承諾|承诺|反對|反对|支持|宣稱|宣称|認為|认为|提到|提及|痛批|抨擊|譴責|赞扬|讚揚)"

# Paired symbols for sentence segmentation protection
PAIRS = {}
PAIRS["\u201c"] = "\u201d"  # " "
PAIRS["\u2018"] = "\u2019"  # ' '
PAIRS["\u300e"] = "\u300f"  # 『 』
PAIRS["\u300c"] = "\u300d"  # 「 」
PAIRS["\uff08"] = "\uff09"  # （ ）
PAIRS["("] = ")"
PAIRS["\u300a"] = "\u300b"  # 《 》
PAIRS["\u3008"] = "\u3009"  # 〈 〉
PAIRS["\u3010"] = "\u3011"  # 【 】
PAIRS["\u3014"] = "\u3015"  # 〔 〕
PAIRS["["] = "]"
PAIRS["{"] = "}"

OPENERS = set(PAIRS.keys())
CLOSERS = set(PAIRS.values())

# Strong sentence boundaries
ENDERS = set(list("。！？!?；;"))
COMBO_ENDS = {"！？", "？！"}
ELLIPSIS = {"……", "..."}

ACTOR = {
    "赖清德": "DPP", "賴清德": "DPP",
    "萧美琴": "DPP", "蕭美琴": "DPP",
    "侯友宜": "KMT", "侯友谊": "KMT",
    "赵少康": "KMT", "趙少康": "KMT",
    "柯文哲": "TPP",
    "吴欣盈": "TPP", "吳欣盈": "TPP",
    "民进党": "DPP", "民進黨": "DPP",
    "国民党": "KMT", "國民黨": "KMT",
    "民众党": "TPP", "民眾黨": "TPP",
    "绿营": "DPP", "綠營": "DPP",
    "蓝营": "KMT", "藍營": "KMT",
    "白营": "TPP", "白營": "TPP",
}

INST = {
    "政见发表会": "procedure", "政見發表會": "procedure",
    "辩论": "procedure", "辯論": "procedure",
    "立法院": "institution",
    "表决": "procedure", "表決": "procedure",
    "ECFA": "policy",
    "修宪": "constitutional", "修憲": "constitutional",
    "核能": "energy",
    "社宅": "housing",
    "两岸": "cross-strait", "兩岸": "cross-strait",
    "国家认同": "identity", "國家認同": "identity",
    "国旗": "symbol", "國旗": "symbol",
    "三读": "procedure", "三讀": "procedure",
    "院会": "procedure", "院會": "procedure",
}

PROC_SYMBOLIC = [
    "政见发表会", "政見發表會", "辩论", "辯論",
    "三读", "三讀", "表决", "表決", "院会", "院會",
    "造势", "造勢", "扫街", "掃街", "拜票",
    "号次", "號次",
    "升旗", "挥舞国旗", "揮舞國旗", "挥舞國旗",
    "集会", "集會", "游行", "遊行", "动员", "動員",
    "协商", "協商", "版本", "草案"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, standardize dates, generate doc_id"""
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
        except:
            return pd.NaT
    
    df["date_std"] = df["date"].apply(_parse_dt)
    df["date_parse_ok"] = df["date_std"].notna()
    
    if df["doc_id"].eq("").all():
        def _mk_id(row):
            h = hashlib.md5(
                (str(row.get("url", "")) + str(row.get("title", "")) + 
                 str(row.get("content", ""))[:100]).encode("utf-8")
            ).hexdigest()[:12]
            return f'{row.get("source", "UNK")}_{h}'
        df["doc_id"] = df.apply(_mk_id, axis=1)
    
    return df

def light_clean(text: str) -> tuple[str, dict]:
    """Replace URLs/HTML with placeholders, remove promo patterns, normalize whitespace"""
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
    t = t2
    
    return t, meta

def nfkc(text: str) -> str:
    """Apply NFKC normalization"""
    if not isinstance(text, str):
        return ""
    return ud.normalize("NFKC", text)

def make_dual_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Generate original, cleaned, and normalized text columns"""
    results = []
    for t in df["content"].fillna(""):
        cleaned, meta = light_clean(t)
        normed = nfkc(cleaned)
        results.append((t, cleaned, normed, meta))
    
    df[["content_orig", "content_clean", "content_norm", "clean_meta"]] = pd.DataFrame(
        results, index=df.index
    )
    return df

def _protected_spans(text: str):
    """Return list of (start, end) spans that should not be split (paired symbols)"""
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
    """Check if idx is within any protected span"""
    for s, e in spans:
        if s <= idx <= e:
            return True
        if idx < s:
            return False
    return False

def split_sentences(text: str, max_len_for_secondary=180) -> list[str]:
    """
    Advanced Chinese sentence segmentation:
    - Strong terminators: 。！？!?；;
    - Protected regions: paired quotes/brackets are not split
    - Combo terminators and ellipsis as single boundary
    - Secondary split for overly long sentences
    - Returns list of sentences (whitespace trimmed, end punctuation kept)
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    spans = _protected_spans(text)
    n = len(text)
    i = 0
    pieces = []
    start = 0
    
    def _is_ender_at(i):
        ch = text[i]
        ch2 = text[i:i+2]
        ch3 = text[i:i+3]
        if ch2 in COMBO_ENDS or ch2 in ELLIPSIS:
            return 2
        if ch3 == "...":
            return 3
        if ch in ENDERS:
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
    
    # Merge short fragments
    merged = []
    for seg in pieces:
        if len(seg) < 6 and merged:
            merged[-1] = (merged[-1] + " " + seg).strip()
        else:
            merged.append(seg)
    
    # Secondary split for overly long sentences
    final = []
    connector = re.compile(r"(但是|然而|不过|不過|因为|因為|因此|所以|同时|同時|以及|并且|並且|或者|或是|例如)")
    for seg in merged:
        if len(seg) <= max_len_for_secondary:
            final.append(seg)
            continue
        
        loc_spans = _protected_spans(seg)
        tmp = []
        last = 0
        for m in re.finditer(r"[；;]", seg):
            if not _in_spans(m.start(), loc_spans):
                cut = seg[last:m.end()].strip()
                if cut:
                    tmp.append(cut)
                last = m.end()
        tail = seg[last:].strip()
        if tail:
            tmp.append(tail)
        
        sec = []
        for chunk in tmp:
            if len(chunk) <= max_len_for_secondary:
                sec.append(chunk)
                continue
            loc2 = _protected_spans(chunk)
            last2 = 0
            spl = False
            for m in connector.finditer(chunk):
                if not _in_spans(m.start(), loc2):
                    left = chunk[last2:m.start()].strip()
                    right = chunk[m.start():].strip()
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

def classify_unit(sent: str) -> str:
    """Classify sentence type: quote/claim/context"""
    if QUOTE_RE.search(sent):
        return "quote"
    if re.search(SAY_VERBS, sent):
        return "claim"
    return "context"

def extract_units(doc_row) -> list[dict]:
    """
    Extract annotation units from document with quote merging:
    - If a short sentence contains SAY_VERBS and next sentence starts with quote,
      merge them as a single quote unit
    """
    sents = split_sentences(doc_row["content_norm"])
    
    # Merge lead-in phrases with quotes
    merged_sents = []
    i = 0
    while i < len(sents):
        current = sents[i]
        
        # Check if this is a short lead-in with SAY_VERBS followed by a quote
        if i + 1 < len(sents):
            next_sent = sents[i + 1]
            # Pattern: short sentence (<=20 chars) with SAY_VERBS ending with punctuation/colon
            # followed by sentence starting with quote mark
            if (len(current) <= 20 and 
                re.search(SAY_VERBS, current) and
                re.search(r"[：:，,]?$", current) and
                re.match(r'^["\'『「]', next_sent)):
                # Merge current and next
                merged_sents.append(current + " " + next_sent)
                i += 2
                continue
        
        merged_sents.append(current)
        i += 1
    
    # Build units with prev/next context
    units = []
    for i, s in enumerate(merged_sents):
        units.append({
            "doc_id": doc_row["doc_id"],
            "unit_id": f'{doc_row["doc_id"]}_u{i:03d}',
            "text": s,
            "role": classify_unit(s),
            "prev": merged_sents[i - 1] if i > 0 else "",
            "next": merged_sents[i + 1] if i + 1 < len(merged_sents) else ""
        })
    
    return units

def find_speakers(text: str) -> list[dict]:
    """Identify speakers in text"""
    hits = []
    seen = set()
    for k, party in ACTOR.items():
        if k in text and k not in seen:
            hits.append({"name": k, "party": party})
            seen.add(k)
    return hits

def find_targets(text: str) -> list[str]:
    """Identify policy/institutional targets in text"""
    targets = []
    seen = set()
    for k, cat in INST.items():
        if k in text and k not in seen:
            targets.append(f"{k}|{cat}")
            seen.add(k)
    return targets

def enrich_roles(units: list[dict]) -> list[dict]:
    """Enrich units with speaker and target information"""
    for u in units:
        u["speakers"] = find_speakers(u["text"])
        u["targets"] = find_targets(u["text"])
    return units

def add_event_flags(u: dict) -> dict:
    """Add event and procedural flags"""
    txt = u["text"]
    u["proc_symbolic_flag"] = any(k in txt for k in PROC_SYMBOLIC)
    
    if any(k in txt for k in ["政见", "辩论", "政見", "辯論"]):
        u["event_phase"] = "debate/manifesto"
    elif any(k in txt for k in ["表决", "三读", "版本", "協商", "协商", "表決", "三讀"]):
        u["event_phase"] = "legislative"
    elif any(k in txt for k in ["造势", "拜票", "扫街", "掃街", "造勢"]):
        u["event_phase"] = "campaigning"
    else:
        u["event_phase"] = ""
    
    return u

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def mark_duplicates(df_docs: pd.DataFrame, units: list[dict]):
    """Mark duplicates without removing them"""
    df_docs["doc_hash"] = df_docs["content_norm"].apply(lambda x: md5(x))
    df_docs["doc_dup_flag"] = df_docs["doc_hash"].duplicated(keep="first")
    
    seen = set()
    for u in units:
        h = md5(u["text"])
        u["unit_hash"] = h
        if h in seen:
            u["unit_dup_flag"] = True
        else:
            u["unit_dup_flag"] = False
            seen.add(h)
    
    return df_docs, units

def to_llm_jsonl(units: list[dict], path: str):
    """Export units to LLM-ready JSONL format"""
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
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def preprocess_to_units(df_raw: pd.DataFrame, source_name: str):
    """Main preprocessing pipeline: documents to units"""
    df = normalize_cols(df_raw)
    df["source"] = df["source"].replace("", source_name)
    df = make_dual_text_cols(df)
    
    df_docs_out = df[[
        "doc_id", "source", "title", "url", "date", "date_std", "date_parse_ok",
        "content_orig", "content_clean", "content_norm", "clean_meta"
    ]].copy()
    
    all_units = []
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

def quick_report(df_docs, units, source_name=""):
    """Generate quality report"""
    print(f"\n{'='*60}")
    print(f"Source: {source_name}")
    print(f"{'='*60}")
    print(f"Documents: {len(df_docs)}")
    print(f"Units: {len(units)}")
    print(f"Doc duplicate rate: {df_docs['doc_dup_flag'].mean():.2%}")
    
    c_role = Counter(u["role"] for u in units)
    print(f"\nUnit type distribution:")
    for role, count in c_role.items():
        print(f"  {role}: {count} ({count/len(units):.2%})")
    
    proc_count = sum(u["proc_symbolic_flag"] for u in units)
    print(f"\nProcedural/symbolic flags: {proc_count} ({proc_count/len(units):.2%})")
    
    speaker_count = sum(1 for u in units if u["speakers"])
    print(f"Units with speakers: {speaker_count} ({speaker_count/len(units):.2%})")
    
    target_count = sum(1 for u in units if u["targets"])
    print(f"Units with targets: {target_count} ({target_count/len(units):.2%})")

def parse_conference_txt(txt_path: str, source_name: str) -> pd.DataFrame:
    """Parse conference/debate TXT file to DataFrame"""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    records = []
    current_speaker = ""
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if len(line) < 20 and not any(p in line for p in ["。", "！", "？", "，"]):
            if current_content:
                records.append({
                    "speaker": current_speaker,
                    "content": " ".join(current_content)
                })
                current_content = []
            current_speaker = line
        else:
            current_content.append(line)
    
    if current_content:
        records.append({
            "speaker": current_speaker,
            "content": " ".join(current_content)
        })
    
    df = pd.DataFrame(records)
    df["source"] = source_name
    df["title"] = Path(txt_path).stem
    df["date"] = ""
    df["url"] = ""
    df = df.rename(columns={"content": "content"})
    
    return df

def process_all_datasets(raw_dir: str, output_dir: str):
    """Batch process all datasets"""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    (output_path / "01_news_datasets").mkdir(parents=True, exist_ok=True)
    (output_path / "02_conference_datasets").mkdir(parents=True, exist_ok=True)
    (output_path / "03_X_datasets").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Processing news datasets")
    print("="*80)
    
    news_dir = raw_path / "01_news_datasets"
    for csv_file in news_dir.glob("*.csv"):
        print(f"\nProcessing: {csv_file.name}")
        
        try:
            # Try multiple encodings
            df_raw = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']:
                try:
                    df_raw = pd.read_csv(csv_file, encoding=encoding)
                    if encoding != 'utf-8':
                        print(f"  Read with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df_raw is None:
                print(f"  Failed to read with any encoding, skipping")
                continue
            
            source_name = csv_file.stem
            
            df_docs, units = preprocess_to_units(df_raw, source_name)
            
            doc_out_path = output_path / "01_news_datasets" / f"{csv_file.stem}_docs.csv"
            df_docs.to_csv(doc_out_path, index=False, encoding="utf-8")
            
            unit_out_path = output_path / "01_news_datasets" / f"{csv_file.stem}_units.jsonl"
            to_llm_jsonl(units, str(unit_out_path))
            
            quick_report(df_docs, units, source_name)
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*80)
    print("Processing conference datasets")
    print("="*80)
    
    conf_dir = raw_path / "02_conference_datasets"
    for txt_file in conf_dir.glob("*.txt"):
        print(f"\nProcessing: {txt_file.name}")
        
        try:
            source_name = txt_file.stem
            df_raw = parse_conference_txt(str(txt_file), source_name)
            
            df_docs, units = preprocess_to_units(df_raw, source_name)
            
            doc_out_path = output_path / "02_conference_datasets" / f"{txt_file.stem}_docs.csv"
            df_docs.to_csv(doc_out_path, index=False, encoding="utf-8")
            
            unit_out_path = output_path / "02_conference_datasets" / f"{txt_file.stem}_units.jsonl"
            to_llm_jsonl(units, str(unit_out_path))
            
            quick_report(df_docs, units, source_name)
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*80)
    print("Processing X (Twitter) datasets")
    print("="*80)
    
    x_dir = raw_path / "03_X_datasets"
    for csv_file in x_dir.glob("*.csv"):
        print(f"\nProcessing: {csv_file.name}")
        
        try:
            # Try multiple encodings
            df_raw = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']:
                try:
                    df_raw = pd.read_csv(csv_file, encoding=encoding)
                    if encoding != 'utf-8':
                        print(f"  Read with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df_raw is None:
                print(f"  Failed to read with any encoding, skipping")
                continue
            
            source_name = csv_file.stem
            
            df_docs, units = preprocess_to_units(df_raw, source_name)
            
            doc_out_path = output_path / "03_X_datasets" / f"{csv_file.stem}_docs.csv"
            df_docs.to_csv(doc_out_path, index=False, encoding="utf-8")
            
            unit_out_path = output_path / "03_X_datasets" / f"{csv_file.stem}_units.jsonl"
            to_llm_jsonl(units, str(unit_out_path))
            
            quick_report(df_docs, units, source_name)
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*80)
    print("All processing complete!")
    print("="*80)

if __name__ == "__main__":
    raw_dir = r"01_Data\01_raw_datasets"
    output_dir = r"01_Data\02_processed_datasets"
    
    process_all_datasets(raw_dir, output_dir)

