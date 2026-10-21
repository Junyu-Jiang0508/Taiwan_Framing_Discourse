"""Majority vote over three identical full runs (2 of 3) and export a Phase-2-compatible corpus file.

Runs: full_low (run 1), full_low_r2, full_low_r3 — same model, prompt, effort, input.
  L1  : the label chosen by >= 2 runs; if all three differ, the label of the most confident run
        (high > medium > low; ties -> run 1). L1_votes records 3 / 2 / 1.
  L2  : each code true if >= 2 of 3 runs say true. L2-0X_votes records 0-3.
  secondary (pre-registered): L2-07 from the code-specific prompt, majority of its three runs.
Writes 03_outputs/01_results_labelings/01_results_datasets/Run_claude_maj3_<tag>/final_results.csv
(same rows and metadata as the GPT run), run_manifest.json, the pointer claude_maj3_run.txt,
and eval/maj3/report.md (how much the vote changes vs run 1, and gold evaluation run 1 vs majority).

Usage: python merge_majority.py --tag 20261019 [--allow-partial]
"""
import argparse
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from common import EVAL, GPT_RUN, INPUT, L1, L2, ROOT, RUNS, read_jsonl, sha256_file

DATASETS = ROOT / "03_outputs" / "01_results_labelings" / "01_results_datasets"
RUNS_L = ["full_low", "full_low_r2", "full_low_r3"]
RUNS_CODE = ["full_code_L2-07", "full_code_L2-07_r2", "full_code_L2-07_r3"]
CONF = {"high": 3, "medium": 2, "low": 1}


def load(run, tag):
    return {r["item_id"]: r["parsed"] for r in read_jsonl(RUNS / run / f"{tag}.jsonl") if r.get("ok")}


def vote_l1(ps):
    c = Counter(p["L1"] for p in ps)
    lab, n = c.most_common(1)[0]
    if n >= 2:
        return lab, n
    best = max(range(3), key=lambda k: (CONF.get(ps[k]["confidence"], 0), -k))
    return ps[best]["L1"], 1


def f1(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp = (y & p).sum(); d = 2 * tp + (y & ~p).sum() + (~y & p).sum()
    return 2 * tp / d if d else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--allow-partial", action="store_true", help="use only units present in all three runs")
    a = ap.parse_args()

    r1 = [load(r, "L1") for r in RUNS_L]
    r2 = [load(r, "L2") for r in RUNS_L]
    rc = [load(r, "L2-07") for r in RUNS_CODE]
    units = set.intersection(*(set(x) for x in r1 + r2))
    if not a.allow_partial:
        assert len(units) == 20900, f"only {len(units)} units complete in all three runs (use --allow-partial to test)"
    have_code = all(len(x) == 20900 for x in rc) or (a.allow_partial and all(x for x in rc))
    code_units = set.intersection(*(set(x) for x in rc)) if have_code else set()

    lab = {}
    for u in units:
        l1, v1 = vote_l1([r[u] for r in r1])
        votes = {l: sum(bool(r[u][l]) for r in r2) for l in L2}
        rec = {"L1": l1, "L1_votes": v1, **{f"{l}_votes": votes[l] for l in L2},
               "L2": {l for l in L2 if votes[l] >= 2}, "L2_run1": {l for l in L2 if r2[0][u][l]}, "L1_run1": r1[0][u]["L1"]}
        if u in code_units:
            cv = sum(bool(r[u]["value"]) for r in rc)
            rec["L2-07_code_votes"] = cv
            rec["L2_secondary"] = (rec["L2"] - {"L2-07"}) | ({"L2-07"} if cv >= 2 else set())
        lab[u] = rec

    # ---- export (same layout as merge.py) ----
    df = pd.read_csv(GPT_RUN, encoding="utf-8-sig")
    df = df.drop(columns=[c for c in df.columns if c.startswith(("L1_", "L2_")) or c == "custom_id"])
    df["claude_annotated"] = df.unit_id.isin(lab)
    g = df.unit_id.map(lab)
    df["L1_label"] = g.map(lambda r: r["L1"] if isinstance(r, dict) else "")
    df["L1_votes"] = g.map(lambda r: r["L1_votes"] if isinstance(r, dict) else pd.NA)
    for l in L2:
        df[l] = g.map(lambda r, l=l: l in r["L2"] if isinstance(r, dict) else pd.NA)
        df[f"{l}_votes"] = g.map(lambda r, l=l: r[f"{l}_votes"] if isinstance(r, dict) else pd.NA)
    df["L2_labels"] = g.map(lambda r: "|".join(sorted(r["L2"])) if isinstance(r, dict) else "")
    if code_units:
        df["L2-07_code_votes"] = g.map(lambda r: r.get("L2-07_code_votes", pd.NA) if isinstance(r, dict) else pd.NA)
        df["L2_labels_secondary"] = g.map(lambda r: "|".join(sorted(r["L2_secondary"])) if isinstance(r, dict) and "L2_secondary" in r else "")
    out_dir = DATASETS / f"Run_claude_maj3_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "final_results.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "rule": "2 of 3 identical runs",
                "runs": RUNS_L, "secondary_L2-07_runs": RUNS_CODE if code_units else None,
                "units": len(lab), "partial": len(lab) < 20900, "final_results_sha256": sha256_file(out)}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    if not manifest["partial"]:
        (DATASETS / "claude_maj3_run.txt").write_text(str(out_dir.relative_to(ROOT)), encoding="utf-8")  # run-dir pointer

    # ---- report ----
    U = sorted(lab)
    rep = [f"# 三次标注取多数（{len(U):,} 句{'，部分数据测试' if manifest['partial'] else ''}）\n\n## 多数票与第 1 次运行的差别\n\n",
           f"- L1：三次全同 {np.mean([lab[u]['L1_votes'] == 3 for u in U]):.1%}，两同一异 {np.mean([lab[u]['L1_votes'] == 2 for u in U]):.1%}，"
           f"三次各异 {np.mean([lab[u]['L1_votes'] == 1 for u in U]):.1%}；多数票改变了第 1 次标签的比例 {np.mean([lab[u]['L1'] != lab[u]['L1_run1'] for u in U]):.1%}\n\n",
           "| 码 | 三票一致（0 或 3）| 多数票改变第 1 次的比例 | 出现率：第 1 次 → 多数票 |\n|---|---|---|---|\n"]
    for l in L2:
        v = np.array([lab[u][f"{l}_votes"] for u in U])
        ch = np.mean([(l in lab[u]["L2"]) != (l in lab[u]["L2_run1"]) for u in U])
        p1 = np.mean([l in lab[u]["L2_run1"] for u in U]); pm = np.mean([l in lab[u]["L2"] for u in U])
        rep.append(f"| {l} | {np.mean((v == 0) | (v == 3)):.1%} | {ch:.2%} | {p1:.1%} → {pm:.1%} |\n")
    # gold evaluation on gold sentences that are in the corpus
    key = pd.read_csv(INPUT / "gold_all_key.csv", keep_default_na=False)
    key = key[(~key.in_prompt.astype(str).str.lower().eq("true")) & key.unit_id.isin(lab)]
    if len(key):
        S = lambda s: frozenset(x for x in str(s).split("|") if x)
        rep.append(f"\n## 金标准评估（{len(key)} 句干净金标准中、位于语料内者；第 1 次 vs 多数票）\n\n| 码 | 第 1 次 F1 | 多数票 F1 |\n|---|---|---|\n")
        m1 = []; mm = []
        for l in L1:
            y = (key.gold_L1 == l).values
            a1 = f1(y, [lab[u]["L1_run1"] == l for u in key.unit_id]); am = f1(y, [lab[u]["L1"] == l for u in key.unit_id])
            m1.append(a1); mm.append(am); rep.append(f"| {l} | {a1:.2f} | {am:.2f} |\n")
        for l in L2:
            y = key.gold_L2.map(lambda s: l in S(s)).values
            a1 = f1(y, [l in lab[u]["L2_run1"] for u in key.unit_id]); am = f1(y, [l in lab[u]["L2"] for u in key.unit_id])
            m1.append(a1); mm.append(am); rep.append(f"| {l} | {a1:.2f} | {am:.2f} |\n")
        rep.append(f"| **macro（15 码）** | **{np.nanmean(m1):.3f}** | **{np.nanmean(mm):.3f}** |\n")
        rep.append("\n注：其中多数句子是 L2 v10.1 修订时看过的开发集，L2 数字偏乐观；第 1 次与多数票之间的**差**不受此影响。\n")
    (EVAL / "maj3").mkdir(parents=True, exist_ok=True)
    (EVAL / "maj3" / "report.md").write_text("".join(rep), encoding="utf-8")
    print("".join(rep)); print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
