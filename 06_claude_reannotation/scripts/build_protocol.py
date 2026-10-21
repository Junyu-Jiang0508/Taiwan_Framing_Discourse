"""Render the frozen annotation protocol: system prompts, JSON schemas, codebook snapshot.

L1 = codebook v9 with the v9.1 patch (resolves the L1-02 / L1-07 contradiction, decided
2026-10-18 before any Claude output was seen). L2 = codebook v10, unchanged.
Also renders an order-reversed variant of each prompt for the order-invariance test.

Usage: python build_protocol.py
"""
import json

import pandas as pd

from common import (L1, L1_FEWSHOT, L1_GUIDE, L2, L2_FEWSHOT, L2_GUIDE, PROTOCOL,
                    clean, sha256_text)

# ---- v9.1 patch: (label_id, column, old, new); each `old` must occur exactly once ----
V91_PATCH = [
    ("L1-02", "core_definition",
     "僅當文本主題並非道德品質、而道德詞彙純粹作為情緒宣洩的修飾語時→L1-07。",
     "僅當文本主題並非道德品質、而道德詞彙純粹作為情緒宣洩的修飾語時→L1-07；"
     "不指向具體事件或標準的全稱式、口號式道德貶斥，亦屬情緒宣洩→L1-07。"),
    ("L1-07", "core_definition",
     "道德詞彙若僅作為情緒彈藥而無論證結構→歸本類。",
     "道德詞彙若僅作為情緒彈藥、不指向具體操守事件或標準→歸本類。"),
    ("L1-02", "boundary_test",
     "語氣激烈不排除L1-02——政治人物批評對手操守時本來就會激動。",
     "語氣激烈不排除L1-02——政治人物批評對手操守時本來就會激動；但須指向可辨識的具體操守議題"
     "（具體事件、行為、被違背的承諾或標準、道德高下的對照）。"),
    ("L1-02", "boundary_test",
     "（c）情緒事件報導（恐慌、感動、慶典）。",
     "（c）情緒事件報導（恐慌、感動、慶典）；（d）不指向具體事件或標準的全稱式、口號式道德貶斥，"
     "主要功能為激起憤怒（如「專制無能、貪污腐敗已經讓民進黨從頭爛到尾」「貪腐就是某黨的體質」）。"),
    ("L1-02", "exclusion_criteria",
     "注意：只要文本在評價某個具體行為者的操守/品格，即使語氣激烈、即使夾雜情緒，仍為L1-02。",
     "注意：只要文本針對某個具體行為者的具體操守事件、行為或被違背的標準作評價，即使語氣激烈、"
     "即使夾雜情緒，仍為L1-02；若只是以道德貶義詞對行為者作全稱式攻擊、不指向任何具體事件或標準→L1-07。"),
    ("L1-07", "exclusion_criteria",
     "（1）情緒語言服務於完整道德論證（有原則+論證鏈+結論）→L1-02。",
     "（1）情緒語言指向具體的操守事件、行為或被違背的標準（不要求完整論證鏈）→L1-02。"),
    ("L1-07", "inclusion_criteria",
     "若以激發憤怒/厭惡為主功能且無論證結構→L1-07。",
     "若以激發憤怒/厭惡為主功能、且不指向具體操守事件或標準→L1-07。"),
    ("L1-07", "boundary_test",
     "→建築結構：支撐整個論證→L1-02。→攻擊彈藥：觸發情緒反應→L1-07。",
     "→建築結構：指向具體操守事件、行為或標準→L1-02。→攻擊彈藥：全稱式貶斥、觸發情緒反應→L1-07。"),
]

LEGEND = (
    "符號說明：「+」為納入條件（inclusion_criteria）；「-」為排除條件（exclusion_criteria）；"
    "「⚖」為邊界測試（boundary_test）；[例1][例2] 為正例；[對比] 為與他類的邊界對照（不是額外正例）。\n"
)

L1_HEADER = (
    "角色：台灣政治文本分析專家。\n"
    "任務：依照下列碼本（L1 v9.1），為【本句】標註 L1 具體語境框架（單選，必須選出一個）。\n\n"
    + LEGEND +
    "\n【判定程序】\n"
    "1. 問題定義優先：先問「本句定義的問題是什麼？」，再判斷指涉對象。政策、預算、法案、行政步驟等語言本身"
    "不決定框架；只有當問題定義本身就是政策、制度或行政時才選 L1-06（見 L1-06 准入決策樹）。"
    "L1-06 不應常態性地作為主框架。\n"
    "2. 【L1-03 vs L1-06】掃描安全化語法三要素（存在性威脅、不可逆時點、超常規措施）：符合兩項以上→L1-03，"
    "無論是否出現政策語言；僅符合零至一項→看處方性質：對抗性（回應威脅、防禦主權）→L1-03，"
    "制度性（機制、程序、協商框架）→L1-06。\n"
    "3. 【L1-02 vs L1-07】針對具體行為者的具體操守事件、行為、被違背的承諾或標準所作的指控或評價→L1-02"
    "（不要求完整論證鏈，語氣激烈亦然）；不指向具體事件或標準的全稱式、口號式道德貶斥，以激起憤怒為主要功能"
    "（如「從頭爛到尾」「貪腐就是某黨的體質」）→L1-07；沒有道德指控的純情緒、嘲諷、恐慌、感動→L1-07。\n"
    "4. 【L1-04 vs L1-05】描寫問題對人的影響與嚴重性→L1-04；批判制度性欠缺、結構性失衡或分配不公→L1-05。\n"
    "5. 前文、後文僅供理解語境；標註對象嚴格限定為【本句】。\n"
    "6. 若本句另有明顯的次要框架，填入 L1_secondary；沒有則填 null。\n\n"
    "【L1 定義】\n"
)

L1_FOOTER = (
    "\n【輸出】\n"
    "只輸出一個 JSON 物件，欄位依序為：\n"
    "- reason：≤40字，選定理由＋已排除的最易混淆類別\n"
    "- L1：必須是 L1-01 至 L1-07 之一，只填編號\n"
    "- L1_secondary：L1-01 至 L1-07 之一，或 null\n"
    "- confidence：high、medium 或 low\n"
)

L2_HEADER = (
    "角色：台灣政治文本分析專家。\n"
    "任務：依照下列碼本（L2 v10），逐一判定【本句】是否觸發 8 個 L2 國族建構子框架。"
    "8 個碼各自獨立判定：可以同時有多個 true，也可以全部為 false。\n\n"
    + LEGEND +
    "\n【判定程序】\n"
    "1. 8 個碼都必須給出判定。判 true 時，在該碼的 _ev 欄位填入【本句】中支持該碼的原文片段（≤30字）；"
    "判 false 時 _ev 填空字串。\n"
    "2. 每個 true 都須有本句中獨立的語段證據；同一語段同時符合兩個碼時，依 ⚖ 邊界測試決定歸屬，"
    "碼本允許並標時才並標。\n"
    "3. 依碼本判斷，不預設偏向：證據符合定義與納入條件、且未觸發排除條件→true；"
    "只有模糊聯想、需要大量推測才能成立→false。\n"
    "4. 純事實性的新聞報導（出席活動、行程、民調數字、程序說明）若無國族建構語意，8 碼皆為 false。\n"
    "5. 前文、後文僅供理解語境；標註對象嚴格限定為【本句】。\n\n"
    "【L2 定義】\n"
)

L2_FOOTER = (
    "\n【輸出】\n"
    "只輸出一個 JSON 物件，欄位依序為 L2-01_ev, L2-01, L2-02_ev, L2-02, …, L2-08_ev, L2-08；"
    "_ev 為本句原文片段（判 false 時為空字串），L2-0X 為 true 或 false。\n"
)


# ---- L2 v10.1 (2026-10-18, after the pilot; validated only on holdout) ----
# Rule-level clarifications only: no sentence or phrase from the gold set or the corpus.
L2_V101_HEADER = (
    "角色：台灣政治文本分析專家。\n"
    "任務：依照下列碼本（L2 v10.1），逐一判定【本句】是否觸發 8 個 L2 國族建構子框架。"
    "8 個碼各自獨立判定：可以同時有多個 true，也可以全部為 false。\n\n"
    + LEGEND +
    "\n【判定程序】\n"
    "1. 8 個碼都必須給出判定。判 true 時，在該碼的 _ev 欄位填入【本句】中支持該碼的原文片段（≤30字）；"
    "判 false 時 _ev 填空字串。\n"
    "2. 碼本列出的顯性與隱性觸發情形同樣算數：句子不必出現字面宣示或關鍵詞，只要符合納入條件所描述的功能即判 true。"
    "新聞報導中轉述的言行（某人表示、某機關指出）同樣依其內容判定。\n"
    "3. 一句話常同時觸發多個碼；符合多個碼的納入條件時應並標，只有在 ⚖ 邊界測試或排除條件明確要求二擇一時才只標一個。\n"
    "4. 只有完全不涉及國家、主權、他者、威脅、歷史記憶、成就、動員或民主價值的句子，才 8 碼皆為 false。\n"
    "5. 前文、後文僅供理解語境；標註對象嚴格限定為【本句】。\n\n"
    "【L2 定義】\n"
)

V101_APPEND = [
    ("L2-01", "inclusion_criteria",
     "【v10.1 補充：隱性主體性展演的常見形式】（a）國家機關、政府部門或國家元首與官員以國家名義發聲、表態、"
     "抗議或回應外國與中國；（b）援引中華民國憲法、法律或政府體制來界定兩岸關係或台灣的地位；"
     "（c）以國家身分出訪、會見外國政要、參與國際事務。以上皆預設國家作為主權行為主體，判 L2-01。"),
    ("L2-03", "inclusion_criteria",
     "【v10.1 補充】（3）以台灣在國際社會、國際夥伴關係、民主陣營或區域秩序中的角色來證成其正當性；"
     "（4）援引人權、平等、法治等國際通行規範或標準來證成台灣的作為。"),
    ("L2-06", "inclusion_criteria",
     "【v10.1 補充】自豪不必出現「驕傲」「光榮」等字眼：成就、肯定、品牌形象的陳述，以及批評中隱含的應有標準，即可觸發。"),
]


def apply_append(guide):
    g = guide.copy()
    for lab, col, text in V101_APPEND:
        idx = g.index[g["label_id"] == lab]
        assert len(idx) == 1, lab
        g.at[idx[0], col] = str(g.at[idx[0], col]) + text
    return g


def apply_patch(guide):
    g = guide.copy()
    for lab, col, old, new in V91_PATCH:
        idx = g.index[g["label_id"] == lab]
        assert len(idx) == 1, lab
        cell = str(g.at[idx[0], col])
        assert cell.count(old) == 1, f"patch target not found exactly once: {lab}.{col}: {old[:30]}"
        g.at[idx[0], col] = cell.replace(old, new)
    return g


def crit(val, sym):
    v = clean(val)
    return f" {sym} {v}\n" if v and v not in ("nan", "N/A", "None") else ""


def fewshots(fs, label_id):
    row = fs[fs["label_id"].astype(str) == label_id]
    if row.empty:
        return ""
    r = row.iloc[0]
    out = ""
    for col, tag in [("Fewshot_v1", "[例1]"), ("Fewshot_v2", "[例2]"), ("Fewshot_contrast", "[對比]")]:
        v = clean(r.get(col))
        if v and v not in ("nan", "N/A", "None"):
            out += f" {tag} {v}\n"
    return out


def render(guide, fs, header, footer, reverse=False):
    rows = list(guide.iterrows())
    if reverse:
        rows = rows[::-1]
    body = ""
    for _, r in rows:
        lab = r["label_id"]
        body += f"[{lab}_{r['label_cn']}] 定義：{clean(r['core_definition'])}\n"
        body += crit(r.get("inclusion_criteria"), "+")
        body += crit(r.get("exclusion_criteria"), "-")
        body += crit(r.get("boundary_test"), "⚖")
        body += fewshots(fs, lab)
    return header + body + footer


def schemas():
    l1 = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "L1": {"type": "string", "enum": L1},
            "L1_secondary": {"anyOf": [{"type": "string", "enum": L1}, {"type": "null"}]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["reason", "L1", "L1_secondary", "confidence"],
        "additionalProperties": False,
    }
    props, req = {}, []
    for lab in L2:
        props[f"{lab}_ev"] = {"type": "string"}
        props[lab] = {"type": "boolean"}
        req += [f"{lab}_ev", lab]
    l2 = {"type": "object", "properties": props, "required": req, "additionalProperties": False}
    return l1, l2


def main():
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    g1 = pd.read_csv(L1_GUIDE)
    g2 = pd.read_csv(L2_GUIDE)
    fs1 = pd.read_csv(L1_FEWSHOT)
    fs2 = pd.read_csv(L2_FEWSHOT)
    assert list(g1["label_id"]) == L1 and list(g2["label_id"]) == L2

    g1p = apply_patch(g1)
    g1p.to_csv(PROTOCOL / "codebook_L1_v9.1.csv", index=False)
    g2.to_csv(PROTOCOL / "codebook_L2_v10.csv", index=False)

    diff = ["# L1 v9 → v9.1 修订（2026-10-18，在看到任何 Claude 输出之前定稿）\n",
            "只修 L1-02 与 L1-07 之间的自相矛盾；其余五类与 L2 v10 一字未改。\n"]
    for lab, col, old, new in V91_PATCH:
        diff.append(f"\n## {lab} · {col}\n\n- 原文：{old}\n- 改为：{new}\n")
    (PROTOCOL / "CHANGES_v9.1.md").write_text("".join(diff), encoding="utf-8")

    prompts = {
        "system_L1.txt": render(g1p, fs1, L1_HEADER, L1_FOOTER),
        "system_L1_rev.txt": render(g1p, fs1, L1_HEADER, L1_FOOTER, reverse=True),
        "system_L2.txt": render(g2, fs2, L2_HEADER, L2_FOOTER),
        "system_L2_rev.txt": render(g2, fs2, L2_HEADER, L2_FOOTER, reverse=True),
        "system_L2_v101.txt": render(apply_append(g2), fs2, L2_V101_HEADER, L2_FOOTER),
    }
    apply_append(g2).to_csv(PROTOCOL / "codebook_L2_v10.1.csv", index=False)
    (PROTOCOL / "CHANGES_v10.1.md").write_text(
        "# L2 v10 → v10.1（2026-10-18，试点之后；只在 holdout 上验证）\n\n"
        "判定程序改为承认隐性触发、鼓励并标，删除「纯事实新闻 8 码皆 false」与「不预设偏向」两条（试点显示它们造成系统性漏标，"
        "定义回收测试 0/8）。新增条款只写规则，不含金标准或语料中的任何句子。\n\n"
        + "".join(f"## {lab} · {col}\n\n追加：{text}\n\n" for lab, col, text in V101_APPEND),
        encoding="utf-8")
    for name, text in prompts.items():
        (PROTOCOL / name).write_text(text, encoding="utf-8")

    s1, s2 = schemas()
    (PROTOCOL / "schema_L1.json").write_text(json.dumps(s1, ensure_ascii=False, indent=1), encoding="utf-8")
    (PROTOCOL / "schema_L2.json").write_text(json.dumps(s2, ensure_ascii=False, indent=1), encoding="utf-8")

    manifest = {name: {"sha256": sha256_text(t), "chars": len(t)} for name, t in prompts.items()}
    manifest["schema_L1.json"] = {"sha256": sha256_text(json.dumps(s1, ensure_ascii=False, indent=1))}
    manifest["schema_L2.json"] = {"sha256": sha256_text(json.dumps(s2, ensure_ascii=False, indent=1))}
    (PROTOCOL / "protocol_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    for k, v in manifest.items():
        print(k, v)


if __name__ == "__main__":
    main()
