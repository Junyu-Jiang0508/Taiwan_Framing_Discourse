"""Merge the full Claude run into a Phase-2-compatible corpus file.

Output: 03_outputs/01_results_labelings/01_results_datasets/Run_claude_<date>/final_results.csv
  - same rows and metadata columns as the GPT run (22,560 rows, joined on unit_id)
  - L1_label / L2_labels ('|'-separated) from Claude; rows Claude did not annotate (outside the
    analysis frame: scrape residue, Unknown camp, X posts; all dropped by p2_00) are left empty
    and flagged claude_annotated=False
  - extra columns: L1_secondary, L1_confidence, L1_reasoning, L2-0X (bool), L2-0X_ev
Also writes run_manifest.json and the pointer file claude_run.txt next to the other run pointers.

Usage: python merge.py --run full_low --tag 20261018
"""
import argparse
import json
from datetime import datetime

import pandas as pd

from common import GPT_RUN, L2, PROTOCOL, ROOT, RUNS, read_jsonl, sha256_file

DATASETS = ROOT / "03_outputs" / "01_results_labelings" / "01_results_datasets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_low")
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d"))
    a = ap.parse_args()

    l1 = {r["item_id"]: r["parsed"] for r in read_jsonl(RUNS / a.run / "L1.jsonl") if r.get("ok")}
    l2 = {r["item_id"]: r["parsed"] for r in read_jsonl(RUNS / a.run / "L2.jsonl") if r.get("ok")}
    assert len(l1) == len(l2) == 20900, (len(l1), len(l2))

    df = pd.read_csv(GPT_RUN, encoding="utf-8-sig")
    drop = [c for c in df.columns if c.startswith(("L1_", "L2_"))] + ["custom_id"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df["claude_annotated"] = df["unit_id"].isin(l1)

    p1 = df["unit_id"].map(l1)
    p2 = df["unit_id"].map(l2)
    df["L1_label"] = p1.map(lambda p: p["L1"] if isinstance(p, dict) else "")
    df["L1_secondary"] = p1.map(lambda p: (p["L1_secondary"] or "") if isinstance(p, dict) else "")
    df["L1_confidence"] = p1.map(lambda p: p["confidence"] if isinstance(p, dict) else "")
    df["L1_reasoning"] = p1.map(lambda p: p["reason"] if isinstance(p, dict) else "")
    for lab in L2:
        df[lab] = p2.map(lambda p, lab=lab: bool(p[lab]) if isinstance(p, dict) else pd.NA)
        df[f"{lab}_ev"] = p2.map(lambda p, lab=lab: p.get(f"{lab}_ev", "") if isinstance(p, dict) else "")
    df["L2_labels"] = p2.map(lambda p: "|".join(l for l in L2 if p[l]) if isinstance(p, dict) else "")

    out_dir = DATASETS / f"Run_claude_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "final_results.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    cfg1 = json.loads((RUNS / a.run / "config_L1.json").read_text(encoding="utf-8"))
    cfg2 = json.loads((RUNS / a.run / "config_L2.json").read_text(encoding="utf-8"))
    manifest = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "annotator": "Claude Code headless (claude -p), subscription",
        "model": cfg1["model"], "effort": {"L1": cfg1["effort"], "L2": cfg2["effort"]},
        "codebook": {"L1": "v9.1", "L2": "v10.1"},
        "system_prompt": {"L1": cfg1["system_prompt"], "L2": cfg2["system_prompt"]},
        "system_prompt_sha256": {"L1": cfg1["system_prompt_sha256"], "L2": cfg2["system_prompt_sha256"]},
        "cli_version": cfg1["cli_version"],
        "source_run": str(RUNS / a.run),
        "rows": len(df), "annotated_rows": int(df["claude_annotated"].sum()),
        "unique_annotated_units": len(l1),
        "metadata_from": str(GPT_RUN.relative_to(ROOT)),
        "final_results_sha256": sha256_file(out),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATASETS / "claude_run.txt").write_text(str(out_dir.relative_to(ROOT)), encoding="utf-8")  # run-dir pointer, as Phase 2 expects
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
