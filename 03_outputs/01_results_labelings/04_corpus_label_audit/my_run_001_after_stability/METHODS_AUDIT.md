# Label audit — methods (Stage 10)

## What is being compared

Unless you pass alternate CSVs, **both** inputs are `final_results.csv` files produced by the batch annotation pipeline (model predictions). The validation file is **not** automatically merged with human reference labels. Any statement about “validation set distribution” in this audit refers to **model output on the validation sample**, not gold-standard prevalence. To compare corpus to human gold, supply a reference CSV and extend the pipeline or merge externally, then re-run audit.

## Row inclusion

- **L1-distribution rows**: units with non-empty `L1_label` after stripping whitespace.
- **Empty L2**: `L2_labels` parses to zero tokens (empty string, NaN, or no pipe-separated codes).

## Step 1 — Distribution consistency

- **L1**: Multinomial counts by label; **Pearson χ² test of homogeneity** on a 2×K table (row 1 = validation counts, row 2 = corpus counts). Columns with zero total counts are dropped. If fewer than two nonempty columns remain, the test is skipped (`note` in machine-readable outputs).
- **L1 practical flag**: any label whose corpus percentage minus validation percentage exceeds **±10.0 percentage points** triggers a supplemental random sample CSV.
- **L2**: For each L2 code, **row prevalence** = fraction of rows (with valid L1) where that code appears in the pipe-separated `L2_labels` list. χ² homogeneity uses implied counts (prevalence × N) per dataset.

## Step 2 — Empty L2 rate

- Rates are **row fractions** among L1-valid rows.
- Heuristic warnings compare corpus vs validation and vs an informal 18–20% band on validation only.

## Step 3 — Slice diagnostics

- Slices: `candidate` (inferred from `source` / `_source_file` if missing), raw `source`, `source_type`, calendar **month** from `date` (`_nodate` if unparsable).
- **dominant_L1**: top L1 share ≥ **0.75** within the slice.
- **skewed_L2_vs_slice_mean**: let *m* = max label row-prevalence among L2 codes in the slice, and *μ* = mean of the remaining labels’ prevalences (empty slice of L2 codes skips). Flag if *m* / (*μ* + ε) > **4.0** (ε = 1e-9). This is a **screening heuristic**, not a formal test.
- `key_normalized` on `source_raw` strips a trailing `.csv` for readability.

## Step 4 — L1×L2 joint table

- Each **(L1, L2) pair** from the same row expands to one count in the crosstab (multi-L2 rows contribute multiple cells).
- **Theoretical low-prior list**: configurable; default includes (L1-01, L2-01). **Empirical flag**: validation joint count = 0 and corpus joint count ≥ **20**.

## Step 5 — Stability (subcommand `stability`)

- Triplicate **Chat Completions** (not Batch API), same prompts as parallel corpus labeling: one L1 call + one L2 call per replicate, user-specified `temperature` (default 0.2).

## Step 6 — Descriptive tables

- Written under `06_descriptive/` when gates allow or when forced; see `run_manifest.json` for flags.

## Output layout (this run)

| Path | Role |
|------|------|
| `run_manifest.json` | Reproducibility: argv, hashes, versions |
| `METHODS_AUDIT.md` | This document |
| `tables_for_manuscript/` | Rounded CSVs for supplementary tables |
| `corpus_*`, `validation_*` | Full-precision internal frequency tables |
| `06_descriptive/` | Full-corpus descriptive exports (if emitted) |
| `compare_*.csv` | Machine-precision comparison |
| `joint_L1_x_L2_*_crosstab_counts.csv` | L1×L2 count matrices |

## Software

- **scipy** for χ²: available
