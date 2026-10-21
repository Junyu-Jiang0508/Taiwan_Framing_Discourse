"""Build the two human-coding packages (blind: no gold, no model labels, no camp / outlet / date).

P2  review_gold: the 11 sentences where Claude, ZXH, GPT and DeepSeek all disagree with the gold,
    the v9.1-motivating L1-07/L1-02 sentences, and 12 random clean gold sentences as controls,
    shuffled so coders cannot tell which items are suspect. Re-coded from scratch by the two gold coders.
P1  gold2: new validation set from the corpus (excluding existing gold sentences):
      - SRS component: 100 sentences, simple random sample of the 20,900 (known probability 100/N)
      - stratified component: from the remaining sentences, each assigned to the stratum of its rarest
        Claude-predicted label; 30 per rare stratum, 12 per common stratum (known probability m_h / N_h)
Outputs in 06_claude_reannotation/human_tasks/: app-format CSVs (ids 9101+ review, 10001+ gold2),
answer-free; design files with inclusion probabilities kept separately (never shown to coders).

Usage: python build_human_tasks.py
"""
import json
import random

import pandas as pd

from common import INPUT, L1, L2, SUB, read_jsonl

SEED = 20261020
OUT = SUB / "human_tasks"
RARE_M, COMMON_M, N_SRS = 30, 12, 100
SUSPECT = ["G702", "G270", "G301", "G1363", "G1181", "G45", "G269", "G448", "G1306", "G1367", "G461"]


def cell(it):
    """App shows one text field: put the ±1 context around the target sentence, as the model saw it."""
    parts = []
    if it.get("prev"):
        parts.append(f"【前文（僅供語境）】{it['prev']}")
    parts.append(f"【本句（請標註）】{it['sentence']}")
    if it.get("next"):
        parts.append(f"【後文（僅供語境）】{it['next']}")
    return "\n".join(parts)


def app_rows(items, start_id, tag):
    return pd.DataFrame([{"id": start_id + i, "candidate": "—", "party": tag, "source_type": "—", "date": "—",
                          "sentence": cell(it), "L1_label_v2": "", "L2_labels": ""} for i, it in enumerate(items)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # ---------------- P2: blind re-coding of suspect gold items ----------------
    key = pd.read_csv(INPUT / "gold_all_key.csv", keep_default_na=False)
    clean = key[~key.in_prompt.astype(str).str.lower().eq("true")]
    motiv = list(clean.loc[clean.v91_motivating.astype(str).str.lower().eq("true"), "item_id"])
    targets = list(dict.fromkeys(SUSPECT + motiv))
    controls = rng.sample(sorted(set(clean.item_id) - set(targets)), 12)
    pilot = {r["item_id"]: r for f in ["pilot_100.jsonl", "holdout_39.jsonl"] for r in read_jsonl(INPUT / f)}
    review_ids = targets + controls
    rng.shuffle(review_ids)
    review_items = [pilot[i] for i in review_ids]
    rev = app_rows(review_items, 9101, "REVIEW")
    rev.to_csv(OUT / "P2_review_gold_app.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"app_id": rev.id, "item_id": review_ids,
                  "role": ["suspect" if i in SUSPECT else "v91_motivating" if i in motiv else "control" for i in review_ids]}
                 ).to_csv(OUT / "P2_review_gold_KEY_do_not_show.csv", index=False)

    # ---------------- P1: new stratified + random validation set ----------------
    frame = pd.read_csv(SUB / "eval" / "full" / "analysis_frame_with_claude.csv", keep_default_na=False).drop_duplicates("unit_id")
    frame = frame[~frame.unit_id.isin(set(key.unit_id) - {""})]
    corpus = {r["item_id"]: r for r in read_jsonl(INPUT / "corpus_20900.jsonl")}
    labs = frame.apply(lambda r: [r.L1_claude] + [x for x in r.L2_claude.split("|") if x], axis=1)
    prev = {l: labs.map(lambda s: l in s).mean() for l in L1 + L2}
    order = sorted(prev, key=prev.get)
    frame = frame.assign(stratum=labs.map(lambda s: min(s, key=order.index)))
    rare = set(order[:7])
    N = len(frame)

    ids = sorted(frame.unit_id)
    srs = set(rng.sample(ids, N_SRS))
    rest = frame[~frame.unit_id.isin(srs)]
    design = [{"unit_id": u, "component": "srs", "stratum": frame.set_index("unit_id").at[u, "stratum"],
               "prob": N_SRS / N} for u in sorted(srs)]
    for h, grp in rest.groupby("stratum"):
        m = RARE_M if h in rare else COMMON_M
        pick = rng.sample(sorted(grp.unit_id), min(m, len(grp)))
        design += [{"unit_id": u, "component": "stratified", "stratum": h, "prob": len(pick) / len(grp),
                    "N_h": len(grp)} for u in pick]
    design = pd.DataFrame(design)
    design = design.sample(frac=1, random_state=SEED).reset_index(drop=True)
    gold2 = app_rows([corpus[u] for u in design.unit_id], 10001, "GOLD2")
    design.insert(0, "app_id", gold2.id)
    gold2.to_csv(OUT / "P1_gold2_app.csv", index=False, encoding="utf-8-sig")
    design.to_csv(OUT / "P1_gold2_DESIGN_do_not_show.csv", index=False)

    exp = {l: int(round(sum(labs[frame.unit_id.isin(design.unit_id)].map(lambda s: l in s)))) for l in order}
    summary = {"P2_items": len(review_ids), "P2_targets": len(targets), "P2_controls": len(controls),
               "P1_items": len(design), "P1_srs": N_SRS, "P1_strata": design.stratum.value_counts().to_dict(),
               "P1_predicted_positives_by_code": exp, "rare_strata": sorted(rare), "population_N": N}
    (OUT / "packages_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
