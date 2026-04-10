# Annotator vs reference (label1_set_v2, n=158)
Date: 2026-04-07 12:45:45

- **Annotator file**: `01_data/06_validation_sets/05_validation_set/01_ZXH.csv`
- **Reference**: `01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv` (`L1_label_v2`, `L2_labels`)
- **Predictions**: `your_L1`, `your_L2`

## L1 confusion matrix
- **L1 accuracy** (both ref & pred nonempty): 86.71% (n=158)
- Rows = reference (label1_set_v2), columns = annotator



### L1 matrix (rows=ref, cols=annotator)

||L1-01|L1-02|L1-03|L1-04|L1-05|L1-06|L1-07|
|---|---|---|---|---|---|---|---|
|L1-01|14|0|0|0|0|0|0|
|L1-02|0|16|0|1|0|1|0|
|L1-03|0|1|36|0|0|3|0|
|L1-04|0|0|2|22|3|2|1|
|L1-05|0|0|0|0|16|0|1|
|L1-06|0|0|1|0|1|14|1|
|L1-07|1|1|0|1|0|0|19|
- **Macro F1**: 0.8666  |  **Weighted F1**: 0.8670

## L2 confusion matrix
- **Subset**: rows where reference `L2_labels` is nonempty (NaN/blank excluded), same rule as `15_validation_analyze.py` model validation.
- **Subset size**: 131 (excluded 27 rows with empty ref L2)
- M[r,p] = count where reference contains r and prediction contains p; rows = ref L2, cols = pred L2.



### L2 matrix M[r,p]

||L2-01|L2-02|L2-03|L2-04|L2-05|L2-06|L2-07|L2-08|
|---|---|---|---|---|---|---|---|---|
|L2-01|36|6|12|3|9|6|14|18|
|L2-02|6|11|1|1|3|2|1|5|
|L2-03|10|0|13|0|1|3|4|6|
|L2-04|2|3|0|17|5|3|3|5|
|L2-05|12|4|3|6|27|1|7|9|
|L2-06|9|4|5|7|4|20|16|10|
|L2-07|7|1|4|4|3|12|42|9|
|L2-08|16|5|8|7|8|6|10|34|
- **L2 Micro F1**: 0.8511  |  **L2 Macro F1**: 0.8473

## Notes
- The 27 sentences with empty reference L2 are excluded from the L2 matrix; annotator `your_L2` on those rows is not counted here.