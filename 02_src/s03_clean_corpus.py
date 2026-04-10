"""Stage 3 — Clean refined corpus and produce cross-corpus merged_deduped.csv.

Reads ``01_data/03_refined_datasets/*_units.csv``, applies per-source cleaning
(news copyright/byline stripping, X hashtag removal + semantic gate, debate
moderator/filler removal), relinks prev/next, and cross-corpus deduplicates.

Outputs (under ``--output``, default ``01_data/04_cleaned_datasets``):
  - Per-file mirrored CSVs
  - ``merged_deduped.csv``  (conference > X > news priority)
  - ``_cleaning_stats.json``
  - ``_sanity_length_hist.png``  (requires matplotlib)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# utils_wrangling_bert
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "wrangling_bert", Path(__file__).resolve().parent / "utils_wrangling_bert.py"
)
_wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wr)
TextFilterStack = _wr.TextFilterStack
hit_strong_keys = _wr.hit_strong_keys

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None  # type: ignore

MIN_COLS = [
    "doc_id",
    "unit_id",
    "sentence",
    "role",
    "prev",
    "next",
    "speakers",
    "targets",
    "source",
    "date",
    "final_status",
]

# 写出时始终带上的元数据列（若原表存在则保留，否则可缺省）
META_COLS_PREFERRED = ["unit_hash", "final_status", "debate_short_review"]

COPYRIGHT_MARKER = "本網站之文字"
COPYRIGHT_TAIL = "本網站之文字、圖片及影音,非經授權,不得轉載"
COPYRIGHT_END_PAT = re.compile(
    r"本網站之文字、圖片及影音[,，][^。]*非經授權[,，]不得轉載[、。.\s]*$"
)
LEAD_MULTI_TS_PAT = re.compile(
    r"^(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}\s*){2,}"
)

CTS_BYLINE = re.compile(r"（中央社記者[^）]*?電）")
TS_LINE = re.compile(r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}")

# 仅在前缀内做「标题+时间」剥离（避免正文中的日期被当成导语时间）
TS_PREFIX_MAX_CHARS = 200

X_CONTEXT_SIM_THRESHOLD = 0.40

_EN_POLITICAL = re.compile(
    r"\b(taiwan|taipei|china|prc|roc|cross[- ]?strait|sovereignty|"
    r"election|democracy|taiwanese|kmt|dpp|tpp)\b",
    re.I,
)
_ZH_POLITICAL = re.compile(
    r"台灣|臺灣|中國|中華人民共和國|主權|兩岸|選舉|總統|候選人|獨立|統一"
)

DEBATE_DEBATE_FILES = frozenset({"Debate_President_units.csv", "vice_president_units.csv"})

# 短句：立场/政策线索（保留）
DEBATE_SHORT_KEEP_RE = re.compile(
    r"反對|反对|支持|主張|主张|認為|认为|必須|不会|不能|要當|兩岸|主權|九二|"
    r"獨立|统一|統一|台獨|台独|民進黨|國民黨|民眾黨|候選人|候选人|總統|总统|副總統|副总统|選民|选民|民調|民调|"
    r"同意|不同意|否決|否决|堅持|坚持|承諾|承诺|保證|保证|"
    r"貪污|贪污|民意|政府|預算|预算|官員|官员|地方|中央"
)

# 明显无标注价值的过渡语（删行）
DEBATE_FILLER_DROP = re.compile(
    r"^(好[。．]?|謝謝[。．]?|谢谢[。．]?|嗯[。．]?|對[。．]?|对[。．]?|是[。．]?|"
    r"哦[。．]?|喔[。．]?|了解[。．]?|知道[。．]?|好的[。．]?|謝謝|谢谢)$"
)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """补全 MIN_COLS，不删除任何已有列（避免丢失 final_status / unit_hash 等）。"""
    out = df.copy()
    for c in MIN_COLS:
        if c not in out.columns:
            out[c] = ""
    return out


def _count_final_status_review(df: pd.DataFrame) -> int:
    if df.empty or "final_status" not in df.columns:
        return 0
    return int(df["final_status"].astype(str).str.upper().eq("REVIEW").sum())


def corpus_tier_from_path(p: Path) -> int:
    parts = p.parts
    if "01_news_datasets" in parts:
        return 1
    if "03_X_datasets" in parts:
        return 2
    if "02_conference_datasets" in parts:
        return 3
    return 0


def relink_prev_next(df: pd.DataFrame) -> pd.DataFrame:
    """同一 doc_id 内按 unit_id 排序后重写 prev / next。"""
    if df.empty or "doc_id" not in df.columns:
        return df
    # 不用 groupby.apply：部分 pandas 版本子表会丢掉分组列 doc_id
    chunks: List[pd.DataFrame] = []
    for doc_id, g in df.groupby("doc_id", sort=False):
        g = g.sort_values("unit_id").copy()
        if "doc_id" not in g.columns:
            g["doc_id"] = doc_id
        sents = g["sentence"].astype(str).tolist()
        for i, idx in enumerate(g.index):
            g.loc[idx, "prev"] = sents[i - 1] if i > 0 else ""
            g.loc[idx, "next"] = sents[i + 1] if i + 1 < len(sents) else ""
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True) if chunks else df.iloc[0:0].copy()


def debate_short_keep_auto(s: str) -> bool:
    if hit_strong_keys(s):
        return True
    return bool(DEBATE_SHORT_KEEP_RE.search(s))


def _strip_prefix_timestamps_non_cts(s: str) -> str:
    """仅在句首 TS_PREFIX_MAX_CHARS 内寻找最后一个 yyyy/mm/dd HH:MM，其后视为正文。"""
    if not s:
        return s
    prefix = s[:TS_PREFIX_MAX_CHARS]
    matches = list(TS_LINE.finditer(prefix))
    if not matches:
        return s
    last = matches[-1]
    return s[last.end() :].strip()


# --- 1. 新闻 ---
def clean_news_sentence(s: str) -> Tuple[Optional[str], str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None, "empty"
    s = str(s).strip()
    if not s:
        return None, "empty"

    # 1a：先截到「（編輯:」之前
    if "（編輯:" in s or "（编辑:" in s:
        cut = s.find("（編輯:")
        if cut < 0:
            cut = s.find("（编辑:")
        s = s[:cut].strip()
        if not s:
            return None, "1a_trunc_empty"

    # 1a：正文后拼接版权/推荐列表 → 从版权声明起始截断，不整行删
    idx = s.find(COPYRIGHT_MARKER)
    if idx >= 0:
        s = s[:idx].strip()
    if not s:
        return None, "1a_empty_after_copyright_strip"

    if len(s) > 200 and LEAD_MULTI_TS_PAT.match(s.strip()):
        return None, "1a_lead_timestamps_garbage"

    if s.endswith(COPYRIGHT_TAIL) or bool(COPYRIGHT_END_PAT.search(s)):
        return None, "1a_copyright_tail"

    # 1b 中央社导语（全文找首次导语即可）
    m = CTS_BYLINE.search(s)
    if m:
        s = s[m.end() :].strip()
    else:
        s = _strip_prefix_timestamps_non_cts(s)

    if not s or len(s) < 10:
        return None, "1b_prefix_strip_too_short"

    t = s.strip()
    if len(t) < 15:
        if re.fullmatch(r"（編輯:[^）]+）\d*", t) or re.fullmatch(r"（编辑:[^）]+）\d*", t):
            return None, "1c_editor_only"
        if re.fullmatch(r"\d{5,7}", t):
            return None, "1c_roc_date_only"
        if "編輯" in t or "编辑" in t:
            return None, "1c_editor_residual"

    return s, ""


def clean_news_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "in": len(df),
        "final_status_REVIEW_in": _count_final_status_review(df),
    }
    reasons: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        new_s, why = clean_news_sentence(r["sentence"])
        if new_s is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        nr = r.to_dict()
        nr["sentence"] = new_s
        rows.append(nr)
    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(df.columns))
    stats["out"] = len(out)
    stats["drops"] = stats["in"] - stats["out"]
    stats["reasons"] = reasons
    stats["final_status_REVIEW_out"] = _count_final_status_review(out)
    return out, stats


# --- 2. X ---
def clean_x_dataframe(
    df: pd.DataFrame,
    seed_texts: List[str],
    stack: TextFilterStack,
    x_sim_threshold: float,
    skip_semantic: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Returns (kept_df, stats, drop_rows_df). drop_rows_df has column drop_reason."""
    stats: Dict[str, Any] = {
        "in": len(df),
        "final_status_REVIEW_in": _count_final_status_review(df),
        "drop_pure_hashtag": 0,
        "strip_hashtag": 0,
    }
    work = df.copy()
    work["sentence"] = work["sentence"].astype(str)

    pure = work["sentence"].str.match(r"^#\S+$", na=False)
    stats["drop_pure_hashtag"] = int(pure.sum())
    drop_records: List[Dict[str, Any]] = []
    for ix in work.index[pure]:
        rec = work.loc[ix].to_dict()
        rec["drop_reason"] = "drop_pure_hashtag"
        drop_records.append(rec)
    work = work.loc[~pure].copy()

    has_tag = work["sentence"].str.contains(r"#\S+", regex=True, na=False)
    stats["rows_with_hashtag"] = int(has_tag.sum())
    work.loc[has_tag, "sentence"] = (
        work.loc[has_tag, "sentence"].str.replace(r"#\S+", "", regex=True).str.strip()
    )
    stats["strip_hashtag"] = int(has_tag.sum())

    ctx_mask = work["role"].astype(str).str.lower().eq("context")
    idx_ctx = work.index[ctx_mask]
    sims: Dict[int, float] = {}

    if not skip_semantic and len(idx_ctx) > 0 and seed_texts and cosine_similarity is not None:
        stack.load_bert_model()
        model = stack.bert_model
        seed_emb = model.encode(seed_texts, show_progress_bar=False, batch_size=128)
        center = np.mean(seed_emb, axis=0, keepdims=True)
        texts = work.loc[idx_ctx, "sentence"].astype(str).tolist()
        te = model.encode(texts, show_progress_bar=True, batch_size=128)
        sim_arr = cosine_similarity(te, center).flatten()
        for i, ix in enumerate(idx_ctx):
            sims[int(ix)] = float(sim_arr[i])

    drop_idx: List[int] = []
    drop_reasons: Dict[int, str] = {}
    for ix in work.index:
        if str(work.loc[ix, "role"]).lower() != "context":
            continue
        sent = str(work.loc[ix, "sentence"])
        letters = len(re.findall(r"[A-Za-z]", sent))
        non_space = len(re.sub(r"\s", "", sent))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", sent))
        mostly_en = non_space > 0 and letters / non_space > 0.55 and cjk < 3
        if mostly_en and not (_EN_POLITICAL.search(sent) or _ZH_POLITICAL.search(sent)):
            drop_idx.append(ix)
            drop_reasons[ix] = "drop_context_non_political_english"
            continue
        if skip_semantic or not seed_texts:
            continue
        sim = sims.get(int(ix), 1.0)
        if sim < x_sim_threshold:
            drop_idx.append(ix)
            drop_reasons[ix] = "drop_context_low_similarity"

    for ix in drop_idx:
        rec = work.loc[ix].to_dict()
        rec["drop_reason"] = drop_reasons.get(ix, "drop_context_unknown")
        rec["x_similarity_to_seed_center"] = sims.get(int(ix), None)
        drop_records.append(rec)

    stats["drop_context_semantic_or_english"] = len(drop_idx)
    work = work.drop(index=drop_idx, errors="ignore")
    stats["out"] = len(work)
    stats["final_status_REVIEW_out"] = _count_final_status_review(work)
    drops_df = pd.DataFrame(drop_records) if drop_records else pd.DataFrame()
    return work, stats, drops_df


# --- 3. 辩论 ---
HU_PROC_MARKERS = [
    "第一位要发表申论",
    "依照先前抽签决定",
    "依照先前抽签的顺序",
    "时间八分钟",
    "时间一样是八分钟",
    "欢迎回到",
    "共同主办单位",
    "进入第三个阶段",
    "辩论会到此圆满结束",
    "每次提问时间一分钟",
]

HU_KEEP_HINTS = [
    "女士提问",
    "先生提出",
    "请三立",
    "请镜电视",
    "请中国时报",
    "请中央社",
    "媒体代表",
    "提出第",
    "提出问题",
]


def is_hu_yuanhui_procedural(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s.startswith("胡元辉"):
        return False
    if any(h in s for h in HU_KEEP_HINTS):
        return False
    return any(m in s for m in HU_PROC_MARKERS)


def clean_debate_dataframe(
    df: pd.DataFrame, filename: str
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    总统/副总统：删主持人程序性发言 + 短句硬规则 + 输出待审短句表。
    其他 conference 文件：原样通过。
    """
    stats: Dict[str, Any] = {
        "in": len(df),
        "final_status_REVIEW_in": _count_final_status_review(df),
    }
    if filename not in DEBATE_DEBATE_FILES:
        stats["out"] = len(df)
        stats["final_status_REVIEW_out"] = stats["final_status_REVIEW_in"]
        return df.copy(), stats, pd.DataFrame()

    work = df.copy()
    mask = work["sentence"].astype(str).map(is_hu_yuanhui_procedural)
    stats["drop_mod_procedural"] = int(mask.sum())
    work = work.loc[~mask]

    kept: List[Dict[str, Any]] = []
    short_drop_filler = 0
    short_keep_auto = 0
    short_keep_bulk = 0  # 未命中关键词但人工确认可保留的短句（与长句同样进主表）

    for _, row in work.iterrows():
        r = row.to_dict()
        s = str(r.get("sentence", "")).strip()
        if len(s) >= 15:
            r["debate_short_review"] = ""
            kept.append(r)
            continue
        if DEBATE_FILLER_DROP.match(s):
            short_drop_filler += 1
            continue
        if debate_short_keep_auto(s):
            r["debate_short_review"] = "KEEP_AUTO"
            short_keep_auto += 1
            kept.append(r)
            continue
        r["debate_short_review"] = ""
        short_keep_bulk += 1
        kept.append(r)

    cols = list(df.columns)
    if "debate_short_review" not in cols:
        cols = cols + ["debate_short_review"]
    out = pd.DataFrame(kept) if kept else pd.DataFrame(columns=cols)
    review_df = pd.DataFrame()

    stats["out"] = len(out)
    stats["short_drop_filler"] = short_drop_filler
    stats["short_keep_auto"] = short_keep_auto
    stats["short_keep_bulk"] = short_keep_bulk
    stats["final_status_REVIEW_out"] = _count_final_status_review(out)
    return out, stats, review_df


def build_debate_seed_texts(cleaned_debate_dfs: List[pd.DataFrame], stack: TextFilterStack) -> List[str]:
    """种子仅用清洗后且长度>=15的句子，降低导语/碎片污染。"""
    long: List[str] = []
    for cdf in cleaned_debate_dfs:
        for s in cdf["sentence"].astype(str):
            t = s.strip()
            if len(t) >= 15:
                long.append(t)
    if not long:
        return []
    return stack.build_seeds(long, k=48)


def dedupe_by_unit_hash(
    df: pd.DataFrame, tier_col: str = "_corpus_tier"
) -> Tuple[pd.DataFrame, int]:
    if df.empty or "unit_hash" not in df.columns:
        return df, 0
    before = len(df)
    work = df.copy()
    work["_dedup_key"] = work["unit_hash"].astype(str)
    has_h = work["_dedup_key"].str.len() > 0
    w_h = work.loc[has_h].sort_values(["_dedup_key", tier_col], ascending=[True, False])
    w_h = w_h.drop_duplicates(subset=["_dedup_key"], keep="first")
    w_nh = work.loc[~has_h]
    out = pd.concat([w_h, w_nh], ignore_index=True)
    out = out.drop(columns=["_dedup_key"], errors="ignore")
    return out, before - len(out)


def run_sanity_plots(df: pd.DataFrame, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    lens = df["sentence"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lens.clip(0, 800), bins=60, color="steelblue", edgecolor="white")
    ax.set_xlabel("sentence length (chars, capped at 800 for binning)")
    ax.set_ylabel("count")
    ax.set_title("Cleaned corpus: sentence length")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def process_all(
    input_root: Path,
    output_root: Path,
    skip_semantic: bool,
    x_sim_threshold: float,
) -> None:
    csv_files = sorted(p for p in input_root.rglob("*_units.csv") if p.is_file())
    stack = TextFilterStack()

    debate_paths = [
        input_root / "02_conference_datasets" / n
        for n in ("Debate_President_units.csv", "vice_president_units.csv")
    ]
    debate_paths = [p.resolve() for p in debate_paths if p.is_file()]

    debate_cache: Dict[Path, Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]] = {}
    cleaned_for_seed: List[pd.DataFrame] = []
    for p in debate_paths:
        df0 = _ensure_required_columns(_load_csv(p))
        dfc, st, srev = clean_debate_dataframe(df0, p.name)
        debate_cache[p] = (dfc, st, srev)
        cleaned_for_seed.append(dfc)

    seed_texts = build_debate_seed_texts(cleaned_for_seed, stack)

    all_clean_parts: List[pd.DataFrame] = []
    report: Dict[str, Any] = {
        "files": {},
        "x_threshold": x_sim_threshold,
        "debate_seed_pool_ge15": sum(
            1
            for cdf in cleaned_for_seed
            for s in cdf["sentence"].astype(str)
            if len(s.strip()) >= 15
        ),
        "debate_seed_count": len(seed_texts),
    }
    short_review_parts: List[pd.DataFrame] = []

    for path in csv_files:
        rel = path.relative_to(input_root)
        rp = path.resolve()
        tier = corpus_tier_from_path(path)

        if rp in debate_cache:
            df1, st, srev = debate_cache[rp]
            if len(srev):
                srev = srev.copy()
                srev["_source_file"] = str(rel).replace("\\", "/")
                short_review_parts.append(srev)
        else:
            df0 = _ensure_required_columns(_load_csv(path))
            if "01_news_datasets" in path.parts:
                df1, st = clean_news_dataframe(df0)
            elif "03_X_datasets" in path.parts:
                df1, st, _ = clean_x_dataframe(
                    df0, seed_texts, stack, x_sim_threshold, skip_semantic
                )
            elif "02_conference_datasets" in path.parts:
                df1, st, srev = clean_debate_dataframe(df0, path.name)
                if len(srev):
                    srev = srev.copy()
                    srev["_source_file"] = str(rel).replace("\\", "/")
                    short_review_parts.append(srev)
            else:
                df1, st = df0, {"note": "unknown_subfolder", "in": len(df0), "out": len(df0)}

        df1 = relink_prev_next(df1)
        df1["_source_file"] = str(rel).replace("\\", "/")
        df1["_corpus_tier"] = tier

        if "unit_hash" not in df1.columns:
            df1["unit_hash"] = ""

        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df1.to_csv(out_path, index=False, encoding="utf-8-sig")

        report["files"][str(rel).replace("\\", "/")] = st
        all_clean_parts.append(df1)

    short_review_path = output_root / "_debate_short_review.csv"
    if short_review_parts:
        pd.concat(short_review_parts, ignore_index=True).to_csv(
            short_review_path, index=False, encoding="utf-8-sig"
        )
    else:
        short_review_path.unlink(missing_ok=True)
    report["debate_short_review_rows"] = (
        sum(len(x) for x in short_review_parts) if short_review_parts else 0
    )

    base_cols = list({*MIN_COLS, *META_COLS_PREFERRED, "_source_file", "_corpus_tier"})
    merged = (
        pd.concat(all_clean_parts, ignore_index=True)
        if all_clean_parts
        else pd.DataFrame(columns=base_cols)
    )
    if len(merged) and "source" in merged.columns:
        before_by_src = merged.groupby("source", dropna=False).size().to_dict()
    else:
        before_by_src = {}
    merged_dedup, ndrop = dedupe_by_unit_hash(merged)
    if len(merged_dedup) and "source" in merged_dedup.columns:
        after_by_src = merged_dedup.groupby("source", dropna=False).size().to_dict()
    else:
        after_by_src = {}
    all_src = set(before_by_src) | set(after_by_src)
    dedup_per_source: Dict[str, Any] = {}
    for s in sorted(all_src, key=lambda x: str(x)):
        sk = str(s)
        b = int(before_by_src.get(s, 0))
        a = int(after_by_src.get(s, 0))
        dedup_per_source[sk] = {"before": b, "after": a, "dropped": b - a}

    merged_path = output_root / "merged_deduped.csv"
    merged_dedup.to_csv(merged_path, index=False, encoding="utf-8-sig")
    report["merged_rows"] = len(merged)
    report["merged_after_dedup"] = len(merged_dedup)
    report["cross_corpus_dup_dropped"] = ndrop
    report["per_source_counts"] = (
        merged_dedup["source"].value_counts().to_dict() if len(merged_dedup) else {}
    )
    report["merged_final_status_REVIEW"] = _count_final_status_review(merged_dedup)
    report["dedup_per_source"] = dedup_per_source

    stats_path = output_root / "_cleaning_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    run_sanity_plots(merged_dedup, output_root / "_sanity_length_hist.png")

    print(
        f"Done. per-file outputs under {output_root}, merged={merged_path} "
        f"(rows={len(merged_dedup)}, cross_dup_dropped={ndrop})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean 03_refined_datasets corpus.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("01_data/03_refined_datasets"),
        help="Refined datasets root",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("01_data/04_cleaned_datasets"),
        help="Output root (mirrors subfolders + merged)",
    )
    ap.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip X context cosine gate (no sentence-transformers run)",
    )
    ap.add_argument(
        "--x-sim-threshold",
        type=float,
        default=X_CONTEXT_SIM_THRESHOLD,
        help="Cosine similarity cutoff for X role=context vs debate seed center",
    )
    args = ap.parse_args()
    ir = args.input.resolve()
    if not ir.is_dir():
        raise SystemExit(f"Input dir not found: {ir}")
    process_all(ir, args.output.resolve(), args.skip_semantic, args.x_sim_threshold)


if __name__ == "__main__":
    main()
