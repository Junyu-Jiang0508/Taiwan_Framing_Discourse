# Corpus label audit (Stage 10)

- Corpus: `03_outputs/01_results_labelings/01_results_datasets/Run_20260407_221407/final_results.csv`
- Validation: `03_outputs/01_results_labelings/03_validation_output/Run_20260407_1137/final_results.csv`

## Gate summary

- **Suggested pass** (automated heuristics only): `False`
- L1 labels with |Δpp| > threshold vs validation: L1-06
- L1 χ² p-value (display): <1e-8 (exact 2.73e-29)
- L2 row-prevalence χ² p-value (display): 0.0004
- Empty L2: corpus 40.1%, validation 10.8%
- Slice anomalies flagged: 1
- Joint low-prior cells: 4

Interpretation is left to the researcher; any failed gate should be investigated before descriptive stats.

## Files

- `run_manifest.json` — argv, input SHA-256, environment
- `METHODS_AUDIT.md` — definitions and comparison basis
- `tables_for_manuscript/` — rounded CSVs for supplementary tables
- `compare_L1_validation_vs_corpus.csv`, `compare_L2_row_prev_validation_vs_corpus.csv` (full precision)
- `slice_prevalence_summary.csv`, `joint_L1_x_L2_*_crosstab_counts.csv`
- `audit_summary.json`
