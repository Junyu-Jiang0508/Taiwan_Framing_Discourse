"""Scrape-residue (boilerplate) detection for the labeled corpus.

Residue enters the corpus in two structurally different ways, and they need
different treatment:

``drop``    The row *is* residue — a concatenated headline list (CNA article
            tails), a promo / hotline / syndication block. There is no
            annotatable proposition, so whatever L2 labels the annotator
            produced are noise on non-text.
``prefix``  A real sentence carrying a scrape prefix (upload/update timestamp,
            Google-News follow line). The proposition survives; only the
            character count is inflated, which matters when sentence length
            is used as a covariate or control.

Blanket-dropping every marker (the ``drop_all`` policy) discards ~1.8k real
sentences for the sake of a few dozen characters of prefix, so the default
policy drops only the ``drop`` tier and keeps the ``prefix`` tier flagged.

Tiers were assigned by reading samples of every pattern; the per-pattern
label-load profile in ``diagnostics/boilerplate_audit.json`` is the check:
``drop``-tier patterns carry near-zero mean L2 load (0.04-0.09 for promo
blocks) or a headline-soup profile (mean length 217 chars), while
``prefix``-tier patterns sit at the corpus mean (0.55-0.68 vs 0.61).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd

#: name -> (tier, regex). Tier is "drop" (row is residue) or "prefix"
#: (real sentence with a scrape prefix).
PATTERNS: Dict[str, Tuple[str, str]] = {
    # --- tier: drop -------------------------------------------------------
    # CNA article tail: "(編輯:潘羿菁)1121030" followed by a run of headlines.
    "cna_editor_tail": ("drop", r"[(（]編輯[:：][^)）]{1,40}[)）]\s*1\d{6}"),
    # TVBS election promo strip, appended to unrelated entertainment items.
    "tvbs_promo": ("drop", r"總統大選倒數[、,]?(?:候選人[、,]?民調)?.{0,8}分析盡在"),
    # "更多新聞:" / "延伸閱讀:" headline lists.
    "more_news": ("drop", r"更多新聞\s*[:：]|延伸閱讀\s*[:：]|相關新聞\s*[:：]|更多鏡週刊報導"),
    # Syndication / disclaimer footers.
    "syndication": (
        "drop",
        r"本文由.{0,20}授權(?:提供|刊登)|獨家授權刊登|版權所有[,，]\s*未經許可"
        r"|不代表.{0,8}(?:新聞網|本網|本台|本刊|本報|TVBS)?\s*立場",
    ),
    # Helpline blocks ("《TVBS》提醒您:◎拒絕暴力 請撥打110…").
    "hotline": ("drop", r"提醒您\s*[:：]?\s*◎|請撥打\s*1\d{2}|反霸凌專線|自殺防治|生命線協談"),
    # App / voucher promos.
    "app_promo": ("drop", r"點我(?:拿|看|下載)|立即下載|下載.{0,6}(?:APP|app)"),
    # --- tier: prefix -----------------------------------------------------
    # "透過 Google News" follow line prepended to a real sentence.
    "gnews_follow": ("prefix", r"透過 Google News|透过 Google News"),
    # CNA revision stamp "(11/19 20:27 更新)".
    "cna_ts_update": ("prefix", r"\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}\s*更新"),
    # Liberty Times "首次上稿 12:41更新時間 13:45".
    "ltn_upload_ts": ("prefix", r"首次上稿|更新時間\s*[:：]?\s*\d"),
}

TIERS = ("drop", "prefix")
POLICIES = ("drop_residue", "drop_all", "flag_only")

_COMPILED = {name: (tier, re.compile(pat)) for name, (tier, pat) in PATTERNS.items()}


def _hits(text: str) -> List[str]:
    return [name for name, (_, rx) in _COMPILED.items() if rx.search(text)]


def classify(text: str) -> Tuple[str, str]:
    """Return ``(kind, tier)`` for one sentence.

    ``kind`` is a ``|``-joined list of matched pattern names (empty if clean).
    ``tier`` is ``"drop"`` if any drop-tier pattern matched, else ``"prefix"``
    if any prefix-tier pattern matched, else ``""``.
    """
    names = _hits(str(text))
    if not names:
        return "", ""
    tier = "drop" if any(_COMPILED[n][0] == "drop" for n in names) else "prefix"
    return "|".join(names), tier


def annotate(text: pd.Series) -> pd.DataFrame:
    """Per-row ``boiler_kind`` / ``boiler_tier`` for a sentence column."""
    pairs = text.astype(str).map(classify)
    return pd.DataFrame(
        {
            "boiler_kind": [p[0] for p in pairs],
            "boiler_tier": [p[1] for p in pairs],
        },
        index=text.index,
    )


def drops(tier: str, policy: str) -> bool:
    """Whether a row of tier ``tier`` is removed under ``policy``."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown boilerplate policy {policy!r}; expected one of {POLICIES}")
    if policy == "flag_only":
        return False
    if policy == "drop_all":
        return tier != ""
    return tier == "drop"


def drop_mask(tier: pd.Series, policy: str) -> pd.Series:
    """Rows to remove under ``policy``."""
    return tier.fillna("").map(lambda t: drops(t, policy))


def audit(text: pd.Series, n_l2: pd.Series | None = None) -> List[dict]:
    """Per-pattern and per-tier counts, mean length and mean L2 load.

    The L2 load column is the evidence for the tier assignment, so it is worth
    regenerating on every corpus rather than trusting the constants above.
    """
    txt = text.astype(str)
    slen = txt.map(len)
    rows: List[dict] = []

    def _row(label: str, kind: str, mask: pd.Series) -> dict:
        n = int(mask.sum())
        rec = {
            "label": label,
            "kind": kind,
            "n": n,
            "pct": 100.0 * float(mask.mean()) if len(mask) else 0.0,
            "mean_len": float(slen[mask].mean()) if n else None,
        }
        if n_l2 is not None:
            rec["mean_n_l2"] = float(n_l2[mask].mean()) if n else None
        return rec

    for name, (tier, _) in PATTERNS.items():
        m = txt.str.contains(PATTERNS[name][1], regex=True)
        rows.append({**_row(name, f"pattern:{tier}", m)})

    ann = annotate(txt)
    for tier in TIERS:
        rows.append(_row(f"tier:{tier}", "tier", ann["boiler_tier"] == tier))
    rows.append(_row("any_boilerplate", "total", ann["boiler_tier"] != ""))
    rows.append(_row("clean_body", "total", ann["boiler_tier"] == ""))
    return rows
