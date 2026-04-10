"""
Taiwan Framing Discourse — L1/L2 manual annotation (Streamlit + Supabase).

Standalone repo: codebooks in ./codebooks/, corpus ./label1_set_v2.csv.
Monorepo fallback: ../01_data/05_labels_guidance/ and ../01_data/06_validation_sets/...
"""
from __future__ import annotations

import html
import os
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

import pandas as pd
import streamlit as st
from supabase import Client, create_client

_BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(_BASE, "label1_set_v2.csv")
_CORPUS_FALLBACK = os.path.join(
    _BASE, "..", "01_data", "06_validation_sets", "04_validation_set_v2", "label1_set_v2.csv"
)
_L1_BUNDLE = os.path.join(_BASE, "codebooks", "01_annotation_guide_label1_v9.csv")
_L1_MONO = os.path.join(_BASE, "..", "01_data", "05_labels_guidance", "01_annotation_guide_label1_v9.csv")
_L2_BUNDLE = os.path.join(_BASE, "codebooks", "02_annotation_guide_label2_v10.csv")
_L2_MONO = os.path.join(_BASE, "..", "01_data", "05_labels_guidance", "02_annotation_guide_label2_v10.csv")


def _first_existing(*paths: str) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(" | ".join(paths))


def _pick_url_key(u: Any, k: Any) -> tuple[str, str] | None:
    if u and k and str(u).strip() and str(k).strip():
        return str(u).strip(), str(k).strip()
    return None


def _resolve_supabase_creds() -> tuple[str, str] | None:
    """Streamlit Cloud: st.secrets. Also: flat keys, env, app-dir secrets.toml."""
    # Nested [supabase] (Streamlit Cloud Advanced settings / Secrets)
    try:
        sec = st.secrets.get("supabase") if hasattr(st.secrets, "get") else None
        if sec is None and "supabase" in st.secrets:
            sec = st.secrets["supabase"]
        if sec is not None:
            u = sec.get("url") if hasattr(sec, "get") else sec["url"]
            k = sec.get("key") if hasattr(sec, "get") else sec["key"]
            got = _pick_url_key(u, k)
            if got:
                return got
    except Exception:
        pass
    # Flat TOML keys (some teams paste this style)
    try:
        u = st.secrets.get("SUPABASE_URL") if hasattr(st.secrets, "get") else None
        k = None
        if hasattr(st.secrets, "get"):
            k = st.secrets.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_ANON_KEY")
        got = _pick_url_key(u, k)
        if got:
            return got
    except Exception:
        pass
    u, k = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    got = _pick_url_key(u, k)
    if got:
        return got
    path = os.path.join(_BASE, ".streamlit", "secrets.toml")
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            sec = data.get("supabase") or {}
            got = _pick_url_key(sec.get("url"), sec.get("key"))
            if got:
                return got
            got = _pick_url_key(data.get("SUPABASE_URL"), data.get("SUPABASE_KEY") or data.get("SUPABASE_ANON_KEY"))
            if got:
                return got
        except Exception:
            pass
    return None


def get_supabase() -> Client:
    creds = _resolve_supabase_creds()
    if not creds:
        raise RuntimeError("Missing Supabase url/key")
    return create_client(creds[0], creds[1])


@st.cache_data
def load_corpus() -> pd.DataFrame:
    path = CORPUS_FILE if os.path.exists(CORPUS_FILE) else _CORPUS_FALLBACK
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"id": "Int64"})
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    return df.reset_index(drop=True)


@st.cache_data
def load_l1_guide() -> pd.DataFrame:
    path = _first_existing(_L1_BUNDLE, _L1_MONO)
    return pd.read_csv(path)


@st.cache_data
def load_l2_guide() -> pd.DataFrame:
    path = _first_existing(_L2_BUNDLE, _L2_MONO)
    return pd.read_csv(path)


def _l1_option_label(row: pd.Series) -> str:
    return f"{row['label_id']} — {row['label_cn']} ({row['label_en']})"


def _l2_option_label(row: pd.Series) -> str:
    return f"{row['label_id']} — {row['label_cn']} ({row['label_en']})"


def parse_pipe_labels(s: Any) -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    t = str(s).strip()
    if not t:
        return []
    return [p.strip() for p in t.split("|") if p.strip()]


def load_my_framing_rows(annotator_id: str) -> dict[str, dict[str, Any]]:
    try:
        sb = get_supabase()
        r = sb.table("framing_annotations").select("*").eq("annotator_id", annotator_id).execute()
        out: dict[str, dict[str, Any]] = {}
        for row in r.data or []:
            uid = str(int(row["utterance_id"]))
            out[uid] = {
                "l1_label": row.get("l1_label", ""),
                "l2_labels": row.get("l2_labels") or "",
                "unsure": bool(row.get("unsure", False)),
            }
        return out
    except Exception as e:
        st.error(f"無法載入標註：{e}")
        return {}


def upsert_framing_annotation(
    annotator_id: str,
    utterance_id: int,
    l1_label: str,
    l2_sorted: list[str],
    unsure: bool,
) -> None:
    sb = get_supabase()
    row = {
        "annotator_id": annotator_id,
        "utterance_id": int(utterance_id),
        "l1_label": l1_label,
        "l2_labels": "|".join(l2_sorted),
        "unsure": unsure,
    }
    sb.table("framing_annotations").upsert(row, on_conflict="annotator_id,utterance_id").execute()


def load_peer_notes(utterance_id: int) -> list[dict]:
    try:
        sb = get_supabase()
        r = (
            sb.table("framing_peer_notes")
            .select("*")
            .eq("utterance_id", int(utterance_id))
            .order("created_at", desc=True)
            .limit(80)
            .execute()
        )
        return list(r.data or [])
    except Exception as e:
        st.warning(f"無法載入互助筆記：{e}")
        return []


def insert_peer_note(utterance_id: int, annotator_id: str, body: str) -> None:
    sb = get_supabase()
    sb.table("framing_peer_notes").insert(
        {"utterance_id": int(utterance_id), "annotator_id": annotator_id, "body": body.strip()}
    ).execute()


def build_export_csv(corpus: pd.DataFrame, my_rows: dict[str, dict[str, Any]], annotator_id: str) -> str:
    rows = []
    for _, r in corpus.iterrows():
        uid = str(int(r["id"]))
        ann = my_rows.get(uid, {})
        rows.append(
            {
                "id": r["id"],
                "candidate": r.get("candidate", ""),
                "party": r.get("party", ""),
                "source_type": r.get("source_type", ""),
                "date": r.get("date", ""),
                "sentence": r.get("sentence", ""),
                "ref_L1_label_v2": r.get("L1_label_v2", ""),
                "ref_L2_labels": r.get("L2_labels", ""),
                "annotator_id": annotator_id,
                "your_L1": ann.get("l1_label", ""),
                "your_L2": ann.get("l2_labels", ""),
                "unsure": ann.get("unsure", False),
            }
        )
    return pd.DataFrame(rows).to_csv(index=False, encoding="utf-8-sig")


def main():
    st.set_page_config(page_title="Taiwan Framing 標註", layout="wide")
    st.title("台灣選舉論述 Framing 標註（L1 / L2）")

    if _resolve_supabase_creds() is None:
        with st.expander("除錯：Streamlit 是否載入 Secrets？（不顯示金鑰內容）"):
            try:
                top = list(st.secrets.keys()) if hasattr(st.secrets, "keys") else []
                st.write("`st.secrets` 頂層鍵：", top)
                if "supabase" in st.secrets:
                    sub = st.secrets["supabase"]
                    sk = list(sub.keys()) if hasattr(sub, "keys") else []
                    st.write("`[supabase]` 子鍵：", sk)
                    st.caption("應至少含 `url` 與 `key`（全小寫）。")
            except Exception as ex:
                st.write("讀取失敗：", ex)
        st.error(
            "找不到 Supabase 的 `url` / `key`。\n\n"
            "**Streamlit Cloud：**\n"
            "1) 已部署後到 **App 右上角 ⋮ → Settings → Secrets**（或部署時 Advanced settings）貼上 TOML，按 **Save**。\n"
            "2) **Manage app → ⋮ → Reboot app**，Secrets 才會進到執行環境。\n"
            "3) **Python 版本**請改選 **3.12** 或 **3.11**（勿用 3.14 預覽版，易與套件不相容）。\n"
            "4) `key` 請從 Supabase **Project Settings → API** **完整複製**（`sb_publishable_...` 或 `eyJ...` anon 皆可）；"
            "避免手打，以免 `0`/`O` 打錯。\n\n"
            "**本機：** `05_annotation_app/.streamlit/secrets.toml` 同上格式，或設 `SUPABASE_URL` / `SUPABASE_KEY`。"
        )
        st.stop()

    try:
        _ = _first_existing(_L1_BUNDLE, _L1_MONO)
        _ = _first_existing(_L2_BUNDLE, _L2_MONO)
    except FileNotFoundError:
        st.error(
            "找不到 L1/L2 codebook。獨立部署請確認 `codebooks/` 內含兩個 CSV；"
            "或在完整專案中執行以使用 `01_data/05_labels_guidance/`。"
        )
        st.stop()

    try:
        corpus = load_corpus()
    except FileNotFoundError as e:
        st.error(f"找不到語料 CSV：{e}。請將 `label1_set_v2.csv` 放於本資料夾，或使用專案內驗證集路徑。")
        st.stop()

    l1_df = load_l1_guide()
    l2_df = load_l2_guide()

    l1_ids = l1_df["label_id"].astype(str).tolist()
    l1_display = [_l1_option_label(l1_df[l1_df["label_id"] == lid].iloc[0]) for lid in l1_ids]
    l1_display_to_id = dict(zip(l1_display, l1_ids))

    l2_ids = l2_df["label_id"].astype(str).tolist()
    l2_display = [_l2_option_label(l2_df[l2_df["label_id"] == lid].iloc[0]) for lid in l2_ids]
    l2_display_to_id = {d: lid for d, lid in zip(l2_display, l2_ids)}
    l2_id_to_display = {lid: d for d, lid in l2_display_to_id.items()}

    st.sidebar.header("標註者")
    annotator_name = st.sidebar.text_input(
        "姓名或代號",
        value=st.session_state.get("framing_annotator", ""),
        key="framing_annotator_input",
        placeholder="用於區分多人標註",
    )
    if not annotator_name or not str(annotator_name).strip():
        st.warning("請在側欄輸入姓名或代號後開始。")
        st.stop()
    annotator_id = str(annotator_name).strip()
    st.session_state["framing_annotator"] = annotator_id

    if "force_nav_idx" in st.session_state:
        st.session_state.framing_nav_idx = st.session_state.force_nav_idx
        st.session_state["framing_nav_idx_input"] = st.session_state.force_nav_idx
        del st.session_state.force_nav_idx

    if not st.session_state.get("framing_loaded_for") or st.session_state.get("framing_loaded_for") != annotator_id:
        st.session_state.framing_my_rows = load_my_framing_rows(annotator_id)
        st.session_state.framing_loaded_for = annotator_id

    my_rows: dict[str, dict[str, Any]] = st.session_state.framing_my_rows

    with st.sidebar:
        st.header("篩選")
        parties = sorted(corpus["party"].dropna().astype(str).unique().tolist())
        candidates = sorted(corpus["candidate"].dropna().astype(str).unique().tolist())
        stypes = sorted(corpus["source_type"].dropna().astype(str).unique().tolist())
        pf = st.multiselect("政黨", parties, default=parties)
        cf = st.multiselect("候選人/發言主體", candidates, default=candidates)
        sf = st.multiselect("來源類型", stypes, default=stypes)
        display_df = corpus[
            corpus["party"].astype(str).isin(pf)
            & corpus["candidate"].astype(str).isin(cf)
            & corpus["source_type"].astype(str).isin(sf)
        ].copy()
        display_df = display_df.reset_index(drop=True)

        jump = st.text_input("依 id 跳轉", placeholder="例如 10", key="jump_uid")
        if jump and jump.strip().isdigit():
            j = int(jump.strip())
            hit = display_df[display_df["id"] == j]
            if not hit.empty:
                pos = int(hit.index[0])
                if st.button("前往該句", key="btn_jump_uid"):
                    st.session_state.force_nav_idx = pos
                    st.rerun()
            else:
                st.caption("目前篩選下找不到此 id。")

        reviewed = set(my_rows.keys())
        total = len(display_df)
        done = sum(1 for _, r in display_df.iterrows() if str(int(r["id"])) in reviewed)
        st.metric("本檢視範圍句數", total, f"已標 {done}")

        st.divider()
        if "framing_nav_idx" not in st.session_state:
            st.session_state.framing_nav_idx = 0
        max_idx = max(0, total - 1)
        safe_nav = min(max(0, st.session_state.framing_nav_idx), max_idx)
        nav = st.number_input(
            "句序（0 起）", min_value=0, max_value=max_idx, value=safe_nav, step=1, key="framing_nav_idx_input"
        )
        st.session_state.framing_nav_idx = int(nav)
        if st.button("下一句 →"):
            st.session_state.force_nav_idx = min(int(nav) + 1, max_idx)
            st.rerun()

    st.sidebar.divider()
    with st.sidebar.expander("Codebook 速查（側欄）", expanded=False):
        q1 = st.selectbox("看 L1 定義", l1_display, key="sidebar_l1_peek")
        g1p = l1_df[l1_df["label_id"] == l1_display_to_id[q1]].iloc[0]
        st.caption(str(g1p.get("core_definition", "")))
        q2 = st.selectbox("看 L2 定義", l2_display, key="sidebar_l2_peek")
        g2p = l2_df[l2_df["label_id"] == l2_display_to_id[q2]].iloc[0]
        st.caption(str(g2p.get("core_definition", "")))

    if display_df.empty:
        st.warning("篩選後無資料。")
        st.stop()

    idx = int(st.session_state.framing_nav_idx)
    idx = min(max(0, idx), len(display_df) - 1)
    row = display_df.iloc[idx]
    uid = int(row["id"])
    uid_s = str(uid)
    key_suffix = f"{uid}_{idx}"

    ref_l1 = row.get("L1_label_v2", "")
    ref_l2 = parse_pipe_labels(row.get("L2_labels", ""))
    ref_l2_s = "、".join(ref_l2) if ref_l2 else "（驗證集未填 L2）"

    st.subheader(f"第 {idx + 1} / {len(display_df)} 句 · id={uid} · {row.get('candidate', '')} / {row.get('party', '')}")

    meta = f"{row.get('source_type', '')} · {row.get('date', '')}"
    st.caption(meta)

    sentence_text = row.get("sentence", "")
    if isinstance(sentence_text, str):
        sentence_text = html.unescape(sentence_text)
    st.info(sentence_text)

    with st.expander("驗證集參考標籤（校對用，非必須一致）", expanded=False):
        st.write(f"**L1：** {ref_l1 if pd.notna(ref_l1) and str(ref_l1).strip() else '（無）'}")
        st.write(f"**L2：** {ref_l2_s}")

    saved = my_rows.get(uid_s, {})
    default_l1 = str(saved.get("l1_label", "")).strip()
    default_l1_idx = l1_ids.index(default_l1) if default_l1 in l1_ids else 0
    sel_l1_display = st.selectbox("L1 主框架（擇一）", l1_display, index=default_l1_idx, key=f"l1_{key_suffix}")
    sel_l1 = l1_display_to_id[sel_l1_display]

    with st.expander("目前 L1 的 codebook 全文", expanded=False):
        g1 = l1_df[l1_df["label_id"] == sel_l1].iloc[0]
        st.markdown(f"### {g1['label_id']} {g1['label_cn']}")
        for col, title in [
            ("core_definition", "核心定義"),
            ("inclusion_criteria", "納入"),
            ("exclusion_criteria", "排除"),
            ("boundary_test", "邊界測試"),
        ]:
            if col in g1.index and pd.notna(g1[col]) and str(g1[col]).strip():
                st.markdown(f"**{title}**")
                st.write(str(g1[col]))

    prev_l2 = parse_pipe_labels(saved.get("l2_labels", ""))
    prev_l2_display = [l2_id_to_display[x] for x in prev_l2 if x in l2_id_to_display]
    sel_l2_display = st.multiselect(
        "L2 子框架（可複選，無則留空）",
        l2_display,
        default=prev_l2_display,
        key=f"l2_{key_suffix}",
    )
    sel_l2_ids = sorted({l2_display_to_id[d] for d in sel_l2_display})

    with st.expander("已選 L2 的 codebook 摘要", expanded=False):
        for lid in sel_l2_ids:
            g2 = l2_df[l2_df["label_id"] == lid].iloc[0]
            st.markdown(f"**{lid}** {g2['label_cn']}")
            st.caption(str(g2.get("core_definition", "")))

    unsure = st.checkbox("標註不確定（仍會儲存）", value=bool(saved.get("unsure", False)), key=f"unsure_{key_suffix}")

    st.divider()
    st.markdown("### 互助筆記")
    st.caption("同一句上其他標註者可見；用於邊界案例討論。")
    notes = load_peer_notes(uid)
    if notes:
        for n in reversed(notes):
            who = n.get("annotator_id", "?")
            when = n.get("created_at", "")
            st.markdown(f"**{who}** · `{when}`")
            st.write(n.get("body", ""))
    else:
        st.info("尚無筆記。")

    note_body = st.text_area("新增筆記（公開給團隊）", key=f"note_draft_{key_suffix}", height=68)
    if st.button("發佈筆記", key=f"note_post_{key_suffix}"):
        if note_body.strip():
            try:
                insert_peer_note(uid, annotator_id, note_body)
                st.success("已發佈")
                st.rerun()
            except Exception as e:
                st.error(f"失敗：{e}")
        else:
            st.warning("請輸入內容。")

    def _save(and_next: bool):
        upsert_framing_annotation(annotator_id, uid, sel_l1, sel_l2_ids, unsure)
        st.session_state.framing_my_rows = load_my_framing_rows(annotator_id)
        if and_next:
            st.session_state.force_nav_idx = min(idx + 1, len(display_df) - 1)
        st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("儲存（停留本句）", key=f"save_stay_{key_suffix}"):
            try:
                _save(and_next=False)
            except Exception as e:
                st.error(f"儲存失敗：{e}")
    with c2:
        if st.button("儲存並下一句", key=f"save_next_{key_suffix}"):
            try:
                _save(and_next=True)
            except Exception as e:
                st.error(f"儲存失敗：{e}")

    st.divider()
    csv_out = build_export_csv(corpus, st.session_state.framing_my_rows, annotator_id)
    st.download_button(
        "下載我的標註（CSV，含參考欄）",
        csv_out,
        file_name=f"framing_annotation_{annotator_id.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    with st.expander("預覽：我的標註（最近 50 筆）"):
        recs = []
        for k, v in st.session_state.framing_my_rows.items():
            recs.append({"utterance_id": k, **v})
        if recs:
            st.dataframe(pd.DataFrame(recs).tail(50), width="stretch")
        else:
            st.info("尚無標註")


if __name__ == "__main__":
    main()
