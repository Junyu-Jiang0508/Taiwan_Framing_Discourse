"""P4 measurement experiments: build inputs and prompts (no gold text, no gold labels).

1. Code-specific prompts (third call architecture) for the two weakest L2 codes:
   protocol/system_code_L2-07.txt, system_code_L2-01.txt + schema_code.json.
   The target code gets its full v10.1 block; contrast codes get definition + criteria only.
2. Party-cue masking test: 300 random corpus sentences that name a candidate / party / camp colour,
   stratified by camp; names replaced by neutral placeholders in prev / sentence / next.
   input/mask_300.jsonl (masked) and input/mask_300_orig.jsonl (same items, unmasked, for a rerun control).

Usage: python build_p4.py
"""
import json
import random
import re

import pandas as pd

from build_protocol import LEGEND, apply_append, crit, fewshots
from common import ANALYSIS_FRAME, INPUT, L2_FEWSHOT, L2_GUIDE, PROTOCOL, read_jsonl, write_jsonl

SEED = 20261019
CONTRAST = {"L2-07": ["L2-01", "L2-06", "L2-08"], "L2-01": ["L2-02", "L2-03", "L2-07", "L2-08"]}

# placeholders keep actors distinct but remove party identity
MASK = {
    "賴清德": "候選人甲", "赖清德": "候選人甲", "侯友宜": "候選人乙", "侯友谊": "候選人乙", "柯文哲": "候選人丙",
    "蕭美琴": "副手甲", "萧美琴": "副手甲", "趙少康": "副手乙", "赵少康": "副手乙", "吳欣盈": "副手丙", "吴欣盈": "副手丙",
    "民進黨": "甲黨", "民进党": "甲黨", "國民黨": "乙黨", "国民党": "乙黨", "民眾黨": "丙黨", "民众党": "丙黨",
    "綠營": "甲營", "绿营": "甲營", "藍營": "乙營", "蓝营": "乙營", "白營": "丙營", "白营": "丙營",
}
MASK_RE = re.compile("|".join(sorted(map(re.escape, MASK), key=len, reverse=True)))


def code_prompt(guide, fs, target):
    g = guide.set_index("label_id")
    r = g.loc[target]
    body = (f"[{target}_{r['label_cn']}] 定義：{r['core_definition']}\n" + crit(r.get("inclusion_criteria"), "+")
            + crit(r.get("exclusion_criteria"), "-") + crit(r.get("boundary_test"), "⚖") + fewshots(fs, target))
    contrast = ""
    for c in CONTRAST[target]:
        rc = g.loc[c]
        contrast += (f"[{c}_{rc['label_cn']}] 定義：{rc['core_definition']}\n" + crit(rc.get("inclusion_criteria"), "+")
                     + crit(rc.get("exclusion_criteria"), "-") + crit(rc.get("boundary_test"), "⚖"))
    return (
        "角色：台灣政治文本分析專家。\n"
        f"任務：依照下列碼本（L2 v10.1），只判定【本句】是否觸發 {target}（{r['label_cn']}）這一個碼。\n\n"
        + LEGEND +
        "\n【判定程序】\n"
        f"1. 逐條檢查 {target} 的納入條件、排除條件與每一項 ⚖ 邊界測試；碼本列出的隱性觸發情形同樣算數。\n"
        "2. 邊界測試提到的對照碼，其定義列在後面，只用於判斷歸屬，不需要標註。"
        "若本句同時符合對照碼，且碼本允許並標，仍判本碼為 true；只有排除條件明確要求歸給對照碼時才判 false。\n"
        "3. 前文、後文僅供理解語境；標註對象嚴格限定為【本句】。\n\n"
        f"【目標碼】\n{body}\n【對照碼（僅供邊界判斷）】\n{contrast}\n"
        "【輸出】\n只輸出一個 JSON 物件：ev 為支持判定的本句原文片段（判 false 時為空字串），value 為 true 或 false。\n"
    )


def mask(text):
    return MASK_RE.sub(lambda m: MASK[m.group(0)], text or "")


def main():
    guide = apply_append(pd.read_csv(L2_GUIDE))
    fs = pd.read_csv(L2_FEWSHOT)
    for t in CONTRAST:
        (PROTOCOL / f"system_code_{t}.txt").write_text(code_prompt(guide, fs, t), encoding="utf-8")
    schema = {"type": "object", "properties": {"ev": {"type": "string"}, "value": {"type": "boolean"}},
              "required": ["ev", "value"], "additionalProperties": False}
    (PROTOCOL / "schema_code.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")

    corpus = {r["item_id"]: r for r in read_jsonl(INPUT / "corpus_20900.jsonl")}
    af = pd.read_csv(ANALYSIS_FRAME, usecols=["unit_id", "camp"]).drop_duplicates("unit_id")
    af = af[af.unit_id.map(lambda u: bool(MASK_RE.search(corpus[u]["sentence"])))]
    rng = random.Random(SEED)
    picked = []
    for camp in ["DPP", "KMT", "TPP"]:
        ids = sorted(af.loc[af.camp == camp, "unit_id"])
        picked += rng.sample(ids, 100)
    orig = [corpus[u] for u in picked]
    masked = [{"item_id": it["item_id"], "prev": mask(it["prev"]), "sentence": mask(it["sentence"]),
               "next": mask(it["next"])} for it in orig]
    write_jsonl(INPUT / "mask_300.jsonl", masked)
    write_jsonl(INPUT / "mask_300_orig.jsonl", orig)
    n_hits = sum(len(MASK_RE.findall(it["sentence"])) for it in orig)
    print(f"code prompts: {list(CONTRAST)}; mask items: {len(masked)} (names replaced in sentences: {n_hits})")
    print("example:", masked[0]["sentence"][:120])


if __name__ == "__main__":
    main()
