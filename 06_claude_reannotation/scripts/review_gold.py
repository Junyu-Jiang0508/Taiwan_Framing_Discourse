"""P2: compare the blind re-coding of the REVIEW batch with the original gold (label1_set_v2).

For each re-coder: on suspect items, how often the re-code departs from the original gold;
on control items, the same rate (= the coder's test-retest baseline). A suspect cell is proposed for
correction only if BOTH re-coders independently depart from the original gold in the same direction.

Usage: python review_gold.py --coders junyu.csv collaborator.csv --out ../eval/review
"""
import argparse
from pathlib import Path

import pandas as pd

from common import INPUT, L2, SUB


def sset(s):
    return frozenset(x for x in str(s).replace(",", "|").split("|") if x in L2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coders", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = pd.read_csv(SUB / "human_tasks" / "P2_review_gold_KEY_do_not_show.csv")
    gold = pd.read_csv(INPUT / "gold_all_key.csv", keep_default_na=False).set_index("item_id")
    coders = {Path(p).stem: pd.read_csv(p, keep_default_na=False).set_index("id") for p in a.coders}
    rows = []
    for r in key.itertuples():
        g = gold.loc[r.item_id]
        rec = {"item_id": r.item_id, "role": r.role, "gold_L1": g.gold_L1, "gold_L2": g.gold_L2}
        for n, c in coders.items():
            rec[f"{n}_L1"] = c.at[r.app_id, "your_L1"]
            rec[f"{n}_L2"] = "|".join(sorted(sset(c.at[r.app_id, "your_L2"])))
        rows.append(rec)
    t = pd.DataFrame(rows)
    names = list(coders)
    rep = ["# P2 金标准复核结果\n\n| 角色 | 句数 | " + " | ".join(f"{n} 的 L1 与原金标准不同" for n in names) + " |\n|---|---|"
           + "---|" * len(names) + "\n"]
    for role, grp in t.groupby("role"):
        rep.append(f"| {role} | {len(grp)} | " + " | ".join(f"{(grp[f'{n}_L1'] != grp.gold_L1).mean():.0%}" for n in names) + " |\n")
    rep.append("\n## 建议改正的格（两位复核者独立一致、且与原金标准不同）\n\n")
    prop = []
    for r in t.itertuples():
        l1s = {getattr(r, f"{n}_L1") for n in names}
        if len(l1s) == 1 and r.gold_L1 not in l1s:
            prop.append(f"- {r.item_id}（{r.role}）L1：{r.gold_L1} → {l1s.pop()}\n")
        for code in L2:
            vals = {code in sset(getattr(r, f"{n}_L2")) for n in names}
            if len(vals) == 1 and (code in sset(r.gold_L2)) not in vals:
                prop.append(f"- {r.item_id}（{r.role}）{code}：{'1→0' if code in sset(r.gold_L2) else '0→1'}\n")
    rep += prop or ["（无）\n"]
    rep.append("\n对照句的偏离率 = 复核者自身的重测基线。疑似句的偏离率若不明显高于它，说明那些句子只是难句，不是错标。\n")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t.to_csv(Path(a.out) / "review_merged.csv", index=False)
    (Path(a.out) / "report.md").write_text("".join(rep), encoding="utf-8")
    print("".join(rep))


if __name__ == "__main__":
    main()
