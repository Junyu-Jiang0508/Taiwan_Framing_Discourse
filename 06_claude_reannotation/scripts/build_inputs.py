"""Build the blind annotation inputs and the (separate) answer keys.

Outputs in input/:
  corpus_20900.jsonl      analysis-frame sentences (unique unit_id), fixed-seed shuffle
  pilot_100.jsonl         gold items for config selection   + pilot_100_key.csv
  holdout_<n>.jsonl       gold items for final confirmation + holdout_<n>_key.csv
  stability_30.jsonl      random 30 of the pilot, for rerun / reversed-order runs
  definitions_15.jsonl    each label's core definition as a sentence (definition-recovery test)
  manifest.json
Inputs carry only item_id / prev / sentence / next: no camp, outlet, date or existing label.

Usage: python build_inputs.py
"""
import json
import math
import random
from datetime import datetime
from difflib import SequenceMatcher

import pandas as pd

from common import (ANALYSIS_FRAME, DS_L1_RUN, DS_L2_RUN, GOLD, GPT_RUN, HARD_POOL, INPUT, L1, L2,
                    L1_GUIDE, L2_GUIDE, MERGED_DEDUPED, PROTOCOL, ZXH, clean, match_gold_to_units,
                    norm, overlaps, parse_l2, sha256_file, write_jsonl)

SEED = 20261018
N_PILOT = 100
POS_SHARE_IN_PILOT = 0.7


def item(item_id, prev, sentence, nxt):
    return {"item_id": item_id, "prev": clean(prev), "sentence": clean(sentence), "next": clean(nxt)}


def main():
    INPUT.mkdir(parents=True, exist_ok=True)
    md = pd.read_csv(MERGED_DEDUPED, usecols=["unit_id", "sentence", "prev", "next"])
    dups = md[md.unit_id.duplicated(keep=False)]
    for uid, grp in dups.groupby("unit_id"):
        # known case: the same scraped headline list fetched twice, differing only in crawl timestamp
        a, b = grp["sentence"].iloc[0], grp["sentence"].iloc[1]
        assert SequenceMatcher(None, a, b).ratio() > 0.95, f"duplicate unit_id with different text: {uid}"
    md = md.drop_duplicates("unit_id").set_index("unit_id")

    # ---- corpus: analysis frame, unique unit_id ----
    af = pd.read_csv(ANALYSIS_FRAME, usecols=["unit_id"])
    uids = sorted(af["unit_id"].unique())
    random.Random(SEED).shuffle(uids)
    corpus = [item(u, md.at[u, "prev"], md.at[u, "sentence"], md.at[u, "next"]) for u in uids]
    write_jsonl(INPUT / f"corpus_{len(corpus)}.jsonl", corpus)

    # ---- gold: match to corpus units, attach baselines ----
    gold = pd.read_csv(GOLD)
    gpt = pd.read_csv(GPT_RUN, usecols=["unit_id", "sentence", "L1_label", "L2_labels"]).drop_duplicates("unit_id")
    gold["unit_id"], gold["match"] = match_gold_to_units(gold, gpt)
    gpt = gpt.set_index("unit_id")
    ds2 = pd.read_csv(DS_L2_RUN, usecols=["unit_id", "L2_labels"]).drop_duplicates("unit_id").set_index("unit_id")
    ds1 = pd.read_csv(DS_L1_RUN, usecols=["unit_id", "L1_label"]).drop_duplicates("unit_id").set_index("unit_id")
    zxh = pd.read_csv(ZXH).set_index("id")
    hard_keys = [norm(s) for s in pd.read_csv(HARD_POOL)["sentence"].astype(str)]

    prompt_text = "".join((PROTOCOL / f).read_text(encoding="utf-8") for f in ["system_L1.txt", "system_L2.txt"])
    # generic codebook terms that are rules, not leaked examples (gold 33 only shares this hashtag)
    for term in ["TaiwanCanHelp"]:
        prompt_text = prompt_text.replace(term, "")

    rows = []
    for _, g in gold.iterrows():
        u = g["unit_id"]
        matched = isinstance(u, str)
        sent = md.at[u, "sentence"] if matched else g["sentence"]
        k = norm(g["sentence"])
        rows.append({
            "item_id": f"G{g['id']}",
            "gold_id": g["id"],
            "unit_id": u if matched else "",
            "match": g["match"],
            "party": g["party"],
            "source_type": g["source_type"],
            "has_context": bool(matched and (clean(md.at[u, "prev"]) or clean(md.at[u, "next"]))),
            "in_prompt": overlaps(g["sentence"], prompt_text),
            "in_hard_pool": any(k and (k in h or h in k) for h in hard_keys),
            "gold_L1": g["L1_label_v2"],
            "gold_L2": "|".join(sorted(parse_l2(g["L2_labels"]))),
            "gpt_L1": gpt.at[u, "L1_label"] if matched else "",
            "gpt_L2": "|".join(sorted(parse_l2(gpt.at[u, "L2_labels"]))) if matched else "",
            "ds_L1": clean(ds1.at[u, "L1_label"]) if matched and u in ds1.index else "",
            "ds_L2": "|".join(sorted(parse_l2(ds2.at[u, "L2_labels"]))) if matched and u in ds2.index else "",
            "zxh_L1": clean(zxh.at[g["id"], "your_L1"]) if g["id"] in zxh.index else "",
            "zxh_L2": "|".join(sorted(parse_l2(zxh.at[g["id"], "your_L2"]))) if g["id"] in zxh.index else "",
            "ctx_prev": md.at[u, "prev"] if matched else "",
            "text": sent,
            "ctx_next": md.at[u, "next"] if matched else "",
        })
    key = pd.DataFrame(rows)
    # sentences whose prior-annotator error pattern motivated v9.1 (sensitivity analysis)
    key["v91_motivating"] = (key.gold_L1 == "L1-07") & (key.gpt_L1 == "L1-02") & (key.ds_L1 == "L1-02")

    clean_pool = key[~key.in_prompt]
    pool = clean_pool[clean_pool.unit_id != ""].copy()
    print(f"gold {len(key)} | in prompt {int(key.in_prompt.sum())} | clean {len(clean_pool)} | "
          f"clean & matched {len(pool)} | v9.1-motivating {int(key.v91_motivating.sum())}")

    # ---- stratified pilot: rarest labels first, >=70% of each label's positives ----
    def labs(r):
        return {r.gold_L1} | set(filter(None, r.gold_L2.split("|")))
    ids = list(pool.item_id)
    random.Random(SEED).shuffle(ids)
    lab_of = {r.item_id: labs(r) for r in pool.itertuples()}
    counts = {l: sum(l in lab_of[i] for i in ids) for l in L1 + L2}
    selected = []
    for lab in sorted(counts, key=counts.get):
        need = math.ceil(POS_SHARE_IN_PILOT * counts[lab])
        have = sum(lab in lab_of[i] for i in selected)
        for i in ids:
            if have >= need or len(selected) >= N_PILOT:
                break
            if i not in selected and lab in lab_of[i]:
                selected.append(i); have += 1
    for i in ids:
        if len(selected) >= N_PILOT:
            break
        if i not in selected:
            selected.append(i)
    pilot = key[key.item_id.isin(selected)]
    holdout = clean_pool[~clean_pool.item_id.isin(selected)]

    def to_items(df):
        return [item(r.item_id, r.ctx_prev, r.text, r.ctx_next) for r in df.itertuples()]

    key_cols = [c for c in key.columns if c not in ("ctx_prev", "text", "ctx_next")]
    for name, df in [(f"pilot_{len(pilot)}", pilot), (f"holdout_{len(holdout)}", holdout)]:
        write_jsonl(INPUT / f"{name}.jsonl", to_items(df))
        df[key_cols + ["text"]].rename(columns={"text": "sentence"}).to_csv(
            INPUT / f"{name}_key.csv", index=False)
    key[key_cols + ["text"]].rename(columns={"text": "sentence"}).to_csv(
        INPUT / "gold_all_key.csv", index=False)

    stab = random.Random(SEED + 1).sample(sorted(pilot.item_id), 30)
    write_jsonl(INPUT / "stability_30.jsonl", [x for x in to_items(pilot) if x["item_id"] in stab])

    # ---- definition recovery: each label's core definition as the sentence ----
    defs = []
    for path, layer in [(L1_GUIDE, "L1"), (L2_GUIDE, "L2")]:
        for r in pd.read_csv(path).itertuples():
            defs.append(item(f"DEF_{r.label_id}", "", r.core_definition, ""))
    write_jsonl(INPUT / "definitions_15.jsonl", defs)

    def support(df):
        s = {l: int((df.gold_L1 == l).sum()) for l in L1}
        s.update({l: int(df.gold_L2.str.contains(l).sum()) for l in L2})
        return s

    manifest = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "sources": {p.name: sha256_file(p) for p in [ANALYSIS_FRAME, MERGED_DEDUPED, GOLD, GPT_RUN, DS_L2_RUN,
                                                     DS_L1_RUN, ZXH, HARD_POOL]},
        "corpus_items": len(corpus),
        "corpus_rule": "analysis_frame_n20901 unique unit_id (one sentence retrieved under two camps)",
        "gold_items": len(key),
        "excluded_in_prompt": sorted(key.loc[key.in_prompt, "gold_id"].tolist()),
        "pilot_items": len(pilot), "holdout_items": len(holdout),
        "pilot_support": support(pilot), "holdout_support": support(holdout),
        "pilot_in_hard_pool": int(pilot.in_hard_pool.sum()),
        "stability_items": stab,
    }
    (INPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["corpus_items", "excluded_in_prompt", "pilot_support",
                                               "holdout_support", "pilot_in_hard_pool"]},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
