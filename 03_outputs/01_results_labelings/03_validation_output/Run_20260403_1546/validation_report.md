# Taiwan Political Discourse Validation Report
Date: 2026-04-03 15:53:57

- **Reference**: `/mnt/d/Projects/Taiwan_Framing_Discourse/01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv`
- **Note**: If the validation set was constructed via keyword/stratified sampling, the label distribution may differ from the full corpus. P/R/F1 numbers should be interpreted with this sampling bias in mind and may not generalize to the overall dataset.

## 1. L1 confusion matrix
- **L1 accuracy**: 62.71%
- Rows = reference (manual), columns = model prediction
       L1-01  L1-02  L1-03  L1-04  L1-05  L1-06  L1-07
L1-01     10      1      4      1      3      1      0
L1-02      0      4      0      0      2      0      0
L1-03      1      1     23      0      0      3      1
L1-04      0      0      1      6      1      5      3
L1-05      0      0      1      0     16      0      0
L1-06      0      1     11      0      0      7      1
L1-07      0      0      1      1      0      0      8

- Wrote `L1_confusion_matrix.csv`

### L1 metrics (Precision, Recall, F1, Support)
label  Precision  Recall     F1  Support
L1-01     0.9091  0.5000 0.6452       20
L1-02     0.5714  0.6667 0.6154        6
L1-03     0.5610  0.7931 0.6571       29
L1-04     0.7500  0.3750 0.5000       16
L1-05     0.7273  0.9412 0.8205       17
L1-06     0.4375  0.3500 0.3889       20
L1-07     0.6154  0.8000 0.6957       10

- Wrote `L1_metrics.csv`

- **Macro F1**: 0.6175  |  **Weighted F1**: 0.6130

## 2. L1 prediction distribution per true label
true_label  n_samples  n_correct recall                                     pred_distribution
     L1-01         20         10 50.00% L1-01:10, L1-02:1, L1-03:4, L1-04:1, L1-05:3, L1-06:1
     L1-02          6          4 66.67%                                      L1-02:4, L1-05:2
     L1-03         29         23 79.31%          L1-01:1, L1-02:1, L1-03:23, L1-06:3, L1-07:1
     L1-04         16          6 37.50%           L1-03:1, L1-04:6, L1-05:1, L1-06:5, L1-07:3
     L1-05         17         16 94.12%                                     L1-03:1, L1-05:16
     L1-06         20          7 35.00%                   L1-02:1, L1-03:11, L1-06:7, L1-07:1
     L1-07         10          8 80.00%                             L1-03:1, L1-04:1, L1-07:8

- Wrote `L1_prediction_distribution.csv`

## 2.5 Conditional L1 diagnostic
Subset where predicted L2 and reference L2 are both nonempty; L1 metrics isolate L1 prompt understanding from empty-L2 cases.
- **Subset**: predicted L2 nonempty AND reference L2 nonempty
- **Subset size**: 118 (excluded 0 from full L1-valid set)
- **L1 accuracy (subset)**: 62.71%

### L1 metrics (conditional subset)
label  Precision  Recall     F1  Support
L1-01     0.9091  0.5000 0.6452       20
L1-02     0.5714  0.6667 0.6154        6
L1-03     0.5610  0.7931 0.6571       29
L1-04     0.7500  0.3750 0.5000       16
L1-05     0.7273  0.9412 0.8205       17
L1-06     0.4375  0.3500 0.3889       20
L1-07     0.6154  0.8000 0.6957       10

- Wrote `L1_conditional_diagnostic_metrics.csv`

## 3. L2 confusion matrix
- M[r,p] = count where reference contains r and prediction contains p; rows = ref L2, cols = pred L2.
       L2-01  L2-02  L2-03  L2-04  L2-05  L2-06  L2-07  L2-08
L2-01     23      5      6      1     14      6     16     18
L2-02      6      7      1      0      3      1      2      7
L2-03     13      1      6      0      6      6      6      9
L2-04      2      4      1      5      2      1      8      5
L2-05      8      5      7      5     28      2     10      7
L2-06      4      1      2      0      5      9     10      7
L2-07      9      2      2      0     10      4     37      6
L2-08     18      8      7      4     16      7     15     30

- Wrote `L2_confusion_matrix.csv`

### L2 metrics (per-label binary)
label  Precision  Recall     F1  Support
L2-01     0.6216  0.5750 0.5974       40
L2-02     0.4667  0.3500 0.4000       20
L2-03     0.4286  0.2857 0.3429       21
L2-04     0.7143  0.2500 0.3704       20
L2-05     0.6222  0.5091 0.5600       55
L2-06     0.6000  0.2903 0.3913       31
L2-07     0.6852  0.6852 0.6852       54
L2-08     0.8108  0.6818 0.7407       44

- Wrote `L2_metrics.csv`

- **L2 Micro F1**: 0.5697  |  **L2 Macro F1**: 0.5110

### L2 prediction distribution per true label
true_label  n_samples  n_pred_correct recall                                                          pred_distribution
     L2-01         40              23 57.50% L2-01:23, L2-08:18, L2-07:16, L2-05:14, L2-03:6, L2-06:6, L2-02:5, L2-04:1
     L2-02         20               7 35.00%              L2-08:7, L2-02:7, L2-01:6, L2-05:3, L2-07:2, L2-06:1, L2-03:1
     L2-03         21               6 28.57%             L2-01:13, L2-08:9, L2-05:6, L2-07:6, L2-03:6, L2-06:6, L2-02:1
     L2-04         20               5 25.00%     L2-07:8, L2-08:5, L2-04:5, L2-02:4, L2-05:2, L2-01:2, L2-03:1, L2-06:1
     L2-05         55              28 50.91%   L2-05:28, L2-07:10, L2-01:8, L2-03:7, L2-08:7, L2-02:5, L2-04:5, L2-06:2
     L2-06         31               9 29.03%             L2-07:10, L2-06:9, L2-08:7, L2-05:5, L2-01:4, L2-03:2, L2-02:1
     L2-07         54              37 68.52%            L2-07:37, L2-05:10, L2-01:9, L2-08:6, L2-06:4, L2-03:2, L2-02:2
     L2-08         44              30 68.18% L2-08:30, L2-01:18, L2-05:16, L2-07:15, L2-02:8, L2-03:7, L2-06:7, L2-04:4

- Wrote `L2_prediction_distribution.csv`

## 4. Error cases
- **Error rows**: 94
- Wrote `error_cases.csv`