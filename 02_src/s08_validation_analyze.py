"""Stage 8 — Validation analysis: confusion matrices, P/R/F1, error cases.

Reads the merged result CSV produced by ``s05_batch_validate.py retrieve``
(follows ``latest_run.txt → Run_*/final_results.csv``).  Optionally merges a
reference (ground-truth) CSV to compute L1/L2 classification metrics.
"""
import os
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Must match batch pipeline output base + latest_run.txt (see framing_batch_core / 05_validation_l*_async).
_VALIDATION_OUTPUT_BASE = Path("03_outputs/01_results_labelings/03_validation_output")
_LATEST_RUN_FILE = "latest_run.txt"


def _resolve_default_data_path(base_dir: Path) -> Optional[Path]:
    """
    Prefer async pipeline output: latest_run.txt -> Run_*/final_results.csv.
    Fall back to legacy Validation_v5/final_validation_results.csv.
    """
    output_base = base_dir / _VALIDATION_OUTPUT_BASE
    pointer = output_base / _LATEST_RUN_FILE
    legacy = output_base / "Validation_v5" / "final_validation_results.csv"

    if pointer.exists():
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        candidate = run_dir / "final_results.csv"
        if candidate.is_file():
            return candidate
        logger.warning(
            "latest_run.txt points to %s but final_results.csv is missing "
            "(finish retrieve_l1?).",
            run_dir,
        )

    if legacy.is_file():
        if pointer.exists():
            logger.info("Falling back to legacy data: %s", legacy)
        return legacy
    return None


try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def merge_with_reference(
    df: pd.DataFrame,
    ref_path: str,
    sentence_col: str = "sentence",
    ref_l1_col: str = "L1_label_v2",
    ref_l2_col: str = "L2_labels",
) -> pd.DataFrame:
    """Merge reference CSV; prefer 'id' key, fall back to sentence_col."""
    ref = pd.read_csv(ref_path, encoding="utf-8-sig")
    if ref_l1_col not in ref.columns:
        logger.warning(f"Reference missing '{ref_l1_col}', skip merge.")
        return df

    merge_key = "id" if ("id" in df.columns and "id" in ref.columns) else sentence_col
    if merge_key not in ref.columns or merge_key not in df.columns:
        logger.warning(f"Neither 'id' nor '{sentence_col}' available in both files, skip merge.")
        return df

    ref_cols = [merge_key, ref_l1_col]
    if ref_l2_col in ref.columns:
        ref_cols.append(ref_l2_col)
    ref_dedup = ref[ref_cols].drop_duplicates(subset=[merge_key], keep="first")
    rename = {ref_l1_col: "L1_label_ref"}
    if ref_l2_col in ref_cols:
        rename[ref_l2_col] = "L2_labels_ref"
    ref_dedup = ref_dedup.rename(columns=rename)
    merged = df.merge(ref_dedup, on=merge_key, how="left")
    n_matched = merged["L1_label_ref"].notna().sum()
    logger.info(f"Merged on '{merge_key}': {n_matched}/{len(df)} rows matched.")
    return merged


def _parse_l2(s):
    if pd.isna(s) or str(s).strip() == "":
        return set()
    return set(x.strip() for x in str(s).split("|") if x.strip())


# Output filenames
OUT_L1_CM = "L1_confusion_matrix.csv"
OUT_L1_METRICS = "L1_metrics.csv"
OUT_L2_CM = "L2_confusion_matrix.csv"
OUT_L2_METRICS = "L2_metrics.csv"
OUT_L1_DIST = "L1_prediction_distribution.csv"
OUT_L2_DIST = "L2_prediction_distribution.csv"
OUT_ERROR = "error_cases.csv"
OUT_L1_CONDITIONAL = "L1_conditional_diagnostic_metrics.csv"


def _safe_to_csv(df: pd.DataFrame, path: Path, max_retries: int = 3, **kwargs):
    """Write DataFrame to CSV with retry on PermissionError (e.g. file open in Excel)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = {"encoding": "utf-8-sig", **kwargs}
    for attempt in range(max_retries):
        try:
            df.to_csv(path, **opts)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.error(
                    f"Permission denied: {path}\n"
                    "Close the file in Excel or another program and retry."
                )
                raise


def validate_and_report(
    data_path: str,
    report_path: str,
    ref_csv: str = None,
    ref_l1_col: str = "L1_label_v2",
    ref_l2_col: str = "L2_labels",
):
    """L1/L2 confusion matrices, per-label prediction distribution, error cases."""
    logger.info("Starting validation analysis...")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    ground_truth_col = None
    if ref_csv and os.path.exists(ref_csv):
        df = merge_with_reference(df, ref_csv, ref_l1_col=ref_l1_col, ref_l2_col=ref_l2_col)
        ground_truth_col = "L1_label_ref"

    report_dir = Path(report_path).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    report_lines.append("# Taiwan Political Discourse Validation Report")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if ref_csv:
        report_lines.append(f"- **Reference**: `{ref_csv}`")
    report_lines.append(
        "- **Note**: If the validation set was constructed via keyword/stratified sampling, "
        "the label distribution may differ from the full corpus. P/R/F1 numbers should be "
        "interpreted with this sampling bias in mind and may not generalize to the overall dataset.\n"
    )

    l1_filled = df["L1_label"].notna() & (df["L1_label"].astype(str).str.strip() != "")

    # 1. L1 confusion matrix + metrics
    report_lines.append("## 1. L1 confusion matrix")
    if ground_truth_col and ground_truth_col in df.columns:
        valid_df = df.dropna(subset=["L1_label", ground_truth_col])
        valid_df = valid_df[valid_df["L1_label"].astype(str).str.strip() != ""]
        valid_df = valid_df[valid_df[ground_truth_col].astype(str).str.strip() != ""]
        if len(valid_df) > 0:
            y_true = valid_df[ground_truth_col].astype(str).str.strip()
            y_pred = valid_df["L1_label"].astype(str).str.strip()
            labels = sorted(set(y_true) | set(y_pred))
            if HAS_SKLEARN:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            else:
                cm_df = pd.crosstab(y_true, y_pred)
                cm_df = cm_df.reindex(index=labels, columns=labels, fill_value=0).astype(int)
            acc = (y_true == y_pred).mean()
            report_lines.append(f"- **L1 accuracy**: {acc:.2%}")
            report_lines.append("- Rows = reference (manual), columns = model prediction")
            if HAS_TABULATE:
                report_lines.append(tabulate(cm_df, headers="keys", tablefmt="github"))
            else:
                report_lines.append(cm_df.to_string())
            path = report_dir / OUT_L1_CM
            _safe_to_csv(cm_df, path)
            report_lines.append(f"\n- Wrote `{path.name}`")

            metrics_rows = []
            for lbl in labels:
                tp = int(cm_df.loc[lbl, lbl]) if lbl in cm_df.index and lbl in cm_df.columns else 0
                fn = int(cm_df.loc[lbl].sum() - tp) if lbl in cm_df.index else 0
                fp = int(cm_df[lbl].sum() - tp) if lbl in cm_df.columns else 0
                support = int(cm_df.loc[lbl].sum()) if lbl in cm_df.index else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                metrics_rows.append({
                    "label": lbl,
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1": round(f1, 4),
                    "Support": support,
                })
            metrics_df = pd.DataFrame(metrics_rows)
            path_m = report_dir / OUT_L1_METRICS
            _safe_to_csv(metrics_df, path_m, index=False)
            report_lines.append("\n### L1 metrics (Precision, Recall, F1, Support)")
            if HAS_TABULATE:
                report_lines.append(tabulate(metrics_df, headers="keys", tablefmt="github", showindex=False))
            else:
                report_lines.append(metrics_df.to_string(index=False))
            report_lines.append(f"\n- Wrote `{path_m.name}`")

            if HAS_SKLEARN:
                cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                macro_f1 = cr["macro avg"]["f1-score"]
                weighted_f1 = cr["weighted avg"]["f1-score"]
                report_lines.append(
                    f"\n- **Macro F1**: {macro_f1:.4f}  |  **Weighted F1**: {weighted_f1:.4f}")
        else:
            report_lines.append("(No rows with both L1 and reference nonempty)")
    else:
        report_lines.append("(No reference; pass --ref_csv)")

    # 2. L1 prediction distribution per true label
    report_lines.append("\n## 2. L1 prediction distribution per true label")
    if ground_truth_col and ground_truth_col in df.columns:
        valid_df = df.dropna(subset=["L1_label", ground_truth_col])
        valid_df = valid_df[valid_df["L1_label"].astype(str).str.strip() != ""]
        valid_df = valid_df[valid_df[ground_truth_col].astype(str).str.strip() != ""]
        if len(valid_df) > 0:
            y_true = valid_df[ground_truth_col].astype(str).str.strip()
            y_pred = valid_df["L1_label"].astype(str).str.strip()
            labels = sorted(set(y_true) | set(y_pred))
            per_frame_rows = []
            for true_label in labels:
                mask = y_true == true_label
                n_true = mask.sum()
                correct = (y_true == y_pred)[mask].sum()
                recall = (correct / n_true) if n_true > 0 else 0.0
                row_counts = y_pred[mask].value_counts().reindex(labels, fill_value=0).astype(int)
                row_str = ", ".join(f"{c}:{int(v)}" for c, v in row_counts.items() if v > 0)
                per_frame_rows.append({
                    "true_label": true_label,
                    "n_samples": int(n_true),
                    "n_correct": int(correct),
                    "recall": f"{recall:.2%}",
                    "pred_distribution": row_str,
                })
            tbl = pd.DataFrame(per_frame_rows)
            if HAS_TABULATE:
                report_lines.append(tabulate(tbl, headers="keys", tablefmt="github", showindex=False))
            else:
                report_lines.append(tbl.to_string(index=False))
            path = report_dir / OUT_L1_DIST
            _safe_to_csv(tbl, path, index=False)
            report_lines.append(f"\n- Wrote `{path.name}`")
        else:
            report_lines.append("(No valid rows)")
    else:
        report_lines.append("(No reference)")

    # 2.5 Conditional L1 diagnostic
    report_lines.append("\n## 2.5 Conditional L1 diagnostic")
    report_lines.append(
        "Subset where predicted L2 and reference L2 are both nonempty; "
        "L1 metrics isolate L1 prompt understanding from empty-L2 cases."
    )
    if (
        ground_truth_col
        and ground_truth_col in df.columns
        and "L2_labels" in df.columns
        and "L2_labels_ref" in df.columns
    ):
        model_l2_nonempty = df["L2_labels"].notna() & (df["L2_labels"].astype(str).str.strip() != "")
        ref_l2_nonempty = df["L2_labels_ref"].notna() & (df["L2_labels_ref"].astype(str).str.strip() != "")
        cond_mask = model_l2_nonempty & ref_l2_nonempty
        cond_df = df.loc[cond_mask].copy()
        cond_df = cond_df.dropna(subset=["L1_label", ground_truth_col])
        cond_df = cond_df[cond_df["L1_label"].astype(str).str.strip() != ""]
        cond_df = cond_df[cond_df[ground_truth_col].astype(str).str.strip() != ""]
        n_subset = len(cond_df)
        full_valid = df.dropna(subset=["L1_label", ground_truth_col])
        full_valid = full_valid[full_valid["L1_label"].astype(str).str.strip() != ""]
        full_valid = full_valid[full_valid[ground_truth_col].astype(str).str.strip() != ""]
        n_excluded = len(full_valid) - n_subset
        if n_subset > 0:
            y_true = cond_df[ground_truth_col].astype(str).str.strip()
            y_pred = cond_df["L1_label"].astype(str).str.strip()
            labels = sorted(set(y_true) | set(y_pred))
            if HAS_SKLEARN:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            else:
                cm_df = pd.crosstab(y_true, y_pred)
                cm_df = cm_df.reindex(index=labels, columns=labels, fill_value=0).astype(int)
            acc = (y_true == y_pred).mean()
            report_lines.append("- **Subset**: predicted L2 nonempty AND reference L2 nonempty")
            report_lines.append(f"- **Subset size**: {n_subset} (excluded {n_excluded} from full L1-valid set)")
            report_lines.append(f"- **L1 accuracy (subset)**: {acc:.2%}")
            metrics_rows = []
            for lbl in labels:
                tp = int(cm_df.loc[lbl, lbl]) if lbl in cm_df.index and lbl in cm_df.columns else 0
                fn = int(cm_df.loc[lbl].sum() - tp) if lbl in cm_df.index else 0
                fp = int(cm_df[lbl].sum() - tp) if lbl in cm_df.columns else 0
                support = int(cm_df.loc[lbl].sum()) if lbl in cm_df.index else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                metrics_rows.append({
                    "label": lbl,
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1": round(f1, 4),
                    "Support": support,
                })
            cond_metrics_df = pd.DataFrame(metrics_rows)
            path_cond = report_dir / OUT_L1_CONDITIONAL
            _safe_to_csv(cond_metrics_df, path_cond, index=False)
            report_lines.append("\n### L1 metrics (conditional subset)")
            if HAS_TABULATE:
                report_lines.append(tabulate(cond_metrics_df, headers="keys", tablefmt="github", showindex=False))
            else:
                report_lines.append(cond_metrics_df.to_string(index=False))
            report_lines.append(f"\n- Wrote `{path_cond.name}`")
            logger.info(f"Conditional L1 diagnostic: {n_subset} samples, accuracy {acc:.2%}")
        else:
            report_lines.append("- No rows in subset (both L2 nonempty)")
    else:
        report_lines.append("(Missing L2 reference or prediction columns; skipped)")

    # 3. L2 confusion matrix + metrics + distribution
    report_lines.append("\n## 3. L2 confusion matrix")
    if (
        ground_truth_col
        and ground_truth_col in df.columns
        and "L2_labels" in df.columns
        and "L2_labels_ref" in df.columns
    ):
        l2_df = df.dropna(subset=["L2_labels_ref"]).copy()
        l2_df = l2_df[l2_df["L2_labels_ref"].astype(str).str.strip() != ""]
        l2_df["_ref_set"] = l2_df["L2_labels_ref"].map(_parse_l2)
        l2_df["_pred_set"] = l2_df["L2_labels"].map(_parse_l2)
        all_l2 = sorted(set().union(*l2_df["_ref_set"], *l2_df["_pred_set"]))
        if all_l2:
            l2_pair_count = defaultdict(int)
            for _, row in l2_df.iterrows():
                for r in row["_ref_set"]:
                    for p in row["_pred_set"]:
                        l2_pair_count[(r, p)] += 1
            l2_cm = pd.DataFrame(0, index=all_l2, columns=all_l2)
            for (r, p), c in l2_pair_count.items():
                if r in l2_cm.index and p in l2_cm.columns:
                    l2_cm.loc[r, p] = c
            l2_cm = l2_cm.astype(int)
            report_lines.append(
                "- M[r,p] = count where reference contains r and prediction contains p; "
                "rows = ref L2, cols = pred L2."
            )
            if HAS_TABULATE:
                report_lines.append(tabulate(l2_cm, headers="keys", tablefmt="github"))
            else:
                report_lines.append(l2_cm.to_string())
            path = report_dir / OUT_L2_CM
            _safe_to_csv(l2_cm, path)
            report_lines.append(f"\n- Wrote `{path.name}`")

            l2_metrics_rows = []
            for l2 in all_l2:
                ref_has = l2_df["_ref_set"].map(lambda s: l2 in s)
                pred_has = l2_df["_pred_set"].map(lambda s: l2 in s)
                tp = (ref_has & pred_has).sum()
                fp = (~ref_has & pred_has).sum()
                fn = (ref_has & ~pred_has).sum()
                support = ref_has.sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                l2_metrics_rows.append({
                    "label": l2,
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1": round(f1, 4),
                    "Support": int(support),
                })
            l2_metrics_df = pd.DataFrame(l2_metrics_rows)
            path_m = report_dir / OUT_L2_METRICS
            _safe_to_csv(l2_metrics_df, path_m, index=False)
            report_lines.append("\n### L2 metrics (per-label binary)")
            if HAS_TABULATE:
                report_lines.append(tabulate(l2_metrics_df, headers="keys", tablefmt="github", showindex=False))
            else:
                report_lines.append(l2_metrics_df.to_string(index=False))
            report_lines.append(f"\n- Wrote `{path_m.name}`")

            total_ref = sum(r["Support"] for r in l2_metrics_rows)
            total_pred = sum(
                l2_df["_pred_set"].map(lambda s: l2 in s).sum()
                for l2 in all_l2
            )
            total_tp_int = sum(
                (l2_df["_ref_set"].map(lambda s: l2 in s) & l2_df["_pred_set"].map(lambda s: l2 in s)).sum()
                for l2 in all_l2
            )
            micro_p = total_tp_int / total_pred if total_pred else 0.0
            micro_r = total_tp_int / total_ref if total_ref else 0.0
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
            macro_f1_l2 = sum(r["F1"] for r in l2_metrics_rows) / len(l2_metrics_rows) if l2_metrics_rows else 0.0
            report_lines.append(
                f"\n- **L2 Micro F1**: {micro_f1:.4f}  |  **L2 Macro F1**: {macro_f1_l2:.4f}")

            l2_dist_rows = []
            for l2 in all_l2:
                sub = l2_df[l2_df["_ref_set"].map(lambda s: l2 in s)]
                n_ref = len(sub)
                if n_ref == 0:
                    continue
                pred_counts = Counter()
                for s in sub["_pred_set"]:
                    pred_counts.update(s)
                row_str = ", ".join(f"{c}:{v}" for c, v in pred_counts.most_common())
                n_correct = int(sub["_pred_set"].map(lambda s: l2 in s).sum())
                rec = n_correct / n_ref if n_ref > 0 else 0.0
                l2_dist_rows.append({
                    "true_label": l2,
                    "n_samples": n_ref,
                    "n_pred_correct": n_correct,
                    "recall": f"{rec:.2%}",
                    "pred_distribution": row_str,
                })
            l2_dist_df = pd.DataFrame(l2_dist_rows)
            path_d = report_dir / OUT_L2_DIST
            _safe_to_csv(l2_dist_df, path_d, index=False)
            report_lines.append("\n### L2 prediction distribution per true label")
            if HAS_TABULATE:
                report_lines.append(tabulate(l2_dist_df, headers="keys", tablefmt="github", showindex=False))
            else:
                report_lines.append(l2_dist_df.to_string(index=False))
            report_lines.append(f"\n- Wrote `{path_d.name}`")
        else:
            report_lines.append("(No L2 labels)")
    else:
        report_lines.append("(Reference has no L2 or merge skipped)")

    # 4. Error cases
    report_lines.append("\n## 4. Error cases")
    unc_col = "uncertain_boundary"
    conditions = []
    if "confidence" in df.columns:
        conditions.append(df["confidence"].astype(str).str.strip().str.lower() == "low")
    if unc_col in df.columns:
        conditions.append(df[unc_col].notna() & (df[unc_col].astype(str).str.strip() != ""))
    if ground_truth_col and ground_truth_col in df.columns:
        same = (
            df["L1_label"].astype(str).str.strip()
            == df[ground_truth_col].astype(str).str.strip()
        )
        conditions.append(
            ~same & l1_filled & df[ground_truth_col].notna()
            & (df[ground_truth_col].astype(str).str.strip() != "")
        )
    if "L2_labels_ref" in df.columns and "L2_labels" in df.columns:
        ref_l2_present = df["L2_labels_ref"].map(lambda s: len(_parse_l2(s)) > 0)
        l2_overlap = df.apply(
            lambda r: len(_parse_l2(r.get("L2_labels")) & _parse_l2(r.get("L2_labels_ref"))) == 0,
            axis=1,
        )
        conditions.append(ref_l2_present & l2_overlap)

    if conditions:
        error_mask = pd.concat(conditions, axis=1).any(axis=1)
        error_df = df.loc[error_mask].copy()
    else:
        error_df = pd.DataFrame()

    def _reason(row):
        reasons = []
        if "confidence" in row and str(row.get("confidence", "")).strip().lower() == "low":
            reasons.append("Low_Conf")
        if unc_col in row and pd.notna(row.get(unc_col)) and str(row.get(unc_col, "")).strip():
            reasons.append("Has_Note")
        if ground_truth_col and ground_truth_col in row:
            gt = str(row.get(ground_truth_col, "")).strip()
            l1 = str(row.get("L1_label", "")).strip()
            if gt and l1 != gt:
                reasons.append("Wrong_L1")
        ref_l2_set = _parse_l2(row.get("L2_labels_ref"))
        if ref_l2_set and len(_parse_l2(row.get("L2_labels")) & ref_l2_set) == 0:
            reasons.append("L2_No_Overlap")
        return "|".join(reasons) if reasons else "Other"

    if not error_df.empty:
        error_df["review_reason"] = error_df.apply(_reason, axis=1)
        cols = [
            "review_reason", "custom_id", "id", "sentence",
            "L1_label", "L2_labels", "confidence", "L1_reasoning", unc_col,
        ]
        if "L1_label_ref" in error_df.columns:
            cols.append("L1_label_ref")
        if "L2_labels_ref" in error_df.columns:
            cols.append("L2_labels_ref")
        cols = [c for c in cols if c in error_df.columns]
        rest = [c for c in error_df.columns if c not in cols]
        error_df = error_df[cols + rest]
        path = report_dir / OUT_ERROR
        _safe_to_csv(error_df, path, index=False)
        report_lines.append(f"- **Error rows**: {len(error_df)}")
        report_lines.append(f"- Wrote `{path.name}`")
        logger.info(f"Error cases saved to: {path} ({len(error_df)} rows)")
    else:
        report_lines.append("- No error rows")
        logger.info("No error cases to export.")

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(1)
            else:
                logger.error(
                    f"Permission denied: {report_path}\n"
                    "Close the report file and retry."
                )
                raise
    logger.info(f"Report saved to: {report_path}")
    logger.info(
        f"Outputs: {OUT_L1_CM}, {OUT_L1_METRICS}, {OUT_L2_CM}, {OUT_L2_METRICS}, "
        f"{OUT_L1_DIST}, {OUT_L2_DIST}, {OUT_L1_CONDITIONAL}, {OUT_ERROR}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validation analysis: L1/L2 confusion matrices, prediction distribution, error cases.",
    )
    base_dir = Path(os.getcwd())
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Merged result CSV (retrieve_l1 output). "
            "Default: latest_run.txt -> Run_*/final_results.csv, else Validation_v5/final_validation_results.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output Markdown report path (default: same folder as --data)",
    )
    parser.add_argument(
        "--ref_csv",
        default=str(base_dir / "01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv"),
        help="Reference CSV (ground truth). Merged on 'sentence'.",
    )
    parser.add_argument(
        "--ref_l1_col",
        default="L1_label_v2",
        help="Reference L1 column name in ref_csv",
    )
    parser.add_argument(
        "--ref_l2_col",
        default="L2_labels",
        help="Reference L2 column name in ref_csv",
    )
    args = parser.parse_args()

    data_path = args.data
    if not data_path:
        resolved = _resolve_default_data_path(base_dir)
        if not resolved:
            logger.error(
                "No validation result CSV found. Run retrieve_l1 after submit_l2/submit_l1, "
                "or pass --data explicitly."
            )
            return
        data_path = str(resolved)
    report_path = args.report
    if not report_path:
        report_path = str(Path(data_path).parent / "validation_report.md")

    logger.info("Data file: %s", data_path)
    logger.info("Report file: %s", report_path)

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    validate_and_report(
        data_path=data_path,
        report_path=report_path,
        ref_csv=args.ref_csv,
        ref_l1_col=args.ref_l1_col,
        ref_l2_col=args.ref_l2_col,
    )


if __name__ == "__main__":
    main()
