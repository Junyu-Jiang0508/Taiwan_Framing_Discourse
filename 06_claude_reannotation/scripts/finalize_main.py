"""Build the main-configuration label file (decided 2026-10-20, pre-registered before any gold2 coding).

Main configuration = L1 and L2-01..06, L2-08 from the 3-run majority of the joint call;
                     L2-07 from the 3-run majority of the code-specific prompt.
Outputs
  03_outputs/.../Run_claude_final_<tag>/final_results.csv   Phase-2-compatible (L1_label, L2_labels '|'),
      with L2-07_joint / L2-07_joint_votes kept for the sensitivity analysis; pointer claude_final_run.txt
  eval/full/analysis_frame_with_claude_final.csv           analysis frame with L1_claude / L2_claude = main
      configuration; the single-run labels are kept as L1_claude_single / L2_claude_single, because the
      gold2 strata were drawn from them and must stay fixed (design-based inference).

Usage: python finalize_main.py --maj3 Run_claude_maj3_20261020 --tag 20261020
"""
import argparse
import json
from datetime import datetime

import pandas as pd

from common import EVAL, ROOT, sha256_file

DATASETS = ROOT / "03_outputs" / "01_results_labelings" / "01_results_datasets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maj3", default="Run_claude_maj3_20261020")
    ap.add_argument("--tag", default="20261020")
    a = ap.parse_args()
    src = DATASETS / a.maj3 / "final_results.csv"
    df = pd.read_csv(src, keep_default_na=False, encoding="utf-8-sig")
    ann = df.claude_annotated.astype(str).str.lower().eq("true")
    for c in ["L2-07", "L2-07_votes"]:
        df[c] = df[c].astype(object)
    df["L2-07_joint"] = df["L2-07"]
    df["L2-07_joint_votes"] = df["L2-07_votes"]
    code_votes = pd.to_numeric(df["L2-07_code_votes"], errors="coerce")
    df.loc[ann, "L2-07"] = (code_votes[ann] >= 2).map(bool).astype(object)
    df.loc[ann, "L2-07_votes"] = code_votes[ann].astype(int).astype(object)
    df["L2_labels"] = df["L2_labels_secondary"]
    df = df.drop(columns=["L2_labels_secondary", "L2-07_code_votes"])
    # consistency: the L2-07 flag and the label string must agree on every annotated row
    flag = df.loc[ann, "L2-07"].map(lambda v: v is True or str(v) == "True")
    in_str = df.loc[ann, "L2_labels"].str.split("|").map(lambda s: "L2-07" in s)
    assert (flag.values == in_str.values).all(), "L2-07 flag and L2_labels disagree"

    out_dir = DATASETS / f"Run_claude_final_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "final_results.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    manifest = {"built": datetime.now().isoformat(timespec="seconds"),
                "configuration": "3-run majority (joint call) for L1 and L2 except L2-07; L2-07 = 3-run majority of the code-specific prompt",
                "source": str(src.relative_to(ROOT)), "rows": len(df), "annotated_rows": int(ann.sum()),
                "preregistration": "06_claude_reannotation/human_tasks/PREREG_gold2.md (deviation log 2026-10-20)",
                "final_results_sha256": sha256_file(out)}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATASETS / "claude_final_run.txt").write_text(str(out_dir.relative_to(ROOT)), encoding="utf-8")  # run-dir pointer

    fr = pd.read_csv(EVAL / "full" / "analysis_frame_with_claude.csv", keep_default_na=False)
    fr = fr.rename(columns={"L1_claude": "L1_claude_single", "L2_claude": "L2_claude_single"})
    lab = df[ann].drop_duplicates("unit_id").set_index("unit_id")
    fr["L1_claude"] = fr.unit_id.map(lab.L1_label)
    fr["L2_claude"] = fr.unit_id.map(lab.L2_labels)
    assert fr.L1_claude.notna().all() and fr.L2_claude.notna().all()
    fr.to_csv(EVAL / "full" / "analysis_frame_with_claude_final.csv", index=False)
    changed = (fr.L2_claude != fr.L2_claude_single).mean()
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"analysis frame: {len(fr):,} rows; L2 set differs from the single run in {changed:.1%} of rows; "
          f"L1 differs in {(fr.L1_claude != fr.L1_claude_single).mean():.1%}")


if __name__ == "__main__":
    main()
