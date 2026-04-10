# Taiwan Political Discourse Validation Report
Date: 2026-04-03 15:38:02

- **Reference**: `/mnt/d/Projects/Taiwan_Framing_Discourse/01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv`
- **Note**: If the validation set was constructed via keyword/stratified sampling, the label distribution may differ from the full corpus. P/R/F1 numbers should be interpreted with this sampling bias in mind and may not generalize to the overall dataset.

## 1. L1 confusion matrix
- **L1 accuracy**: 67.27%
- Rows = reference (manual), columns = model prediction
       L1-01  L1-02  L1-03  L1-04  L1-05  L1-06  L1-07
L1-01     12      1      3      1      2      1      0
L1-02      0      3      0      0      2      0      0
L1-03      2      1     22      0      0      1      0
L1-04      0      0      0      6      1      4      2
L1-05      1      0      1      0     15      0      0
L1-06      0      1      8      0      0      6      1
L1-07      0      1      1      1      0      0     10

- Wrote `L1_confusion_matrix.csv`

### L1 metrics (Precision, Recall, F1, Support)
label  Precision  Recall     F1  Support
L1-01     0.8000  0.6000 0.6857       20
L1-02     0.4286  0.6000 0.5000        5
L1-03     0.6286  0.8462 0.7213       26
L1-04     0.7500  0.4615 0.5714       13
L1-05     0.7500  0.8824 0.8108       17
L1-06     0.5000  0.3750 0.4286       16
L1-07     0.7692  0.7692 0.7692       13

- Wrote `L1_metrics.csv`

- **Macro F1**: 0.6410  |  **Weighted F1**: 0.6640

## 2. L1 prediction distribution per true label
true_label  n_samples  n_correct recall                                     pred_distribution
     L1-01         20         12 60.00% L1-01:12, L1-02:1, L1-03:3, L1-04:1, L1-05:2, L1-06:1
     L1-02          5          3 60.00%                                      L1-02:3, L1-05:2
     L1-03         26         22 84.62%                   L1-01:2, L1-02:1, L1-03:22, L1-06:1
     L1-04         13          6 46.15%                    L1-04:6, L1-05:1, L1-06:4, L1-07:2
     L1-05         17         15 88.24%                            L1-01:1, L1-03:1, L1-05:15
     L1-06         16          6 37.50%                    L1-02:1, L1-03:8, L1-06:6, L1-07:1
     L1-07         13         10 76.92%                   L1-02:1, L1-03:1, L1-04:1, L1-07:10

- Wrote `L1_prediction_distribution.csv`

## 2.5 Conditional L1 diagnostic
Subset where predicted L2 and reference L2 are both nonempty; L1 metrics isolate L1 prompt understanding from empty-L2 cases.
- **Subset**: predicted L2 nonempty AND reference L2 nonempty
- **Subset size**: 110 (excluded 0 from full L1-valid set)
- **L1 accuracy (subset)**: 67.27%

### L1 metrics (conditional subset)
label  Precision  Recall     F1  Support
L1-01     0.8000  0.6000 0.6857       20
L1-02     0.4286  0.6000 0.5000        5
L1-03     0.6286  0.8462 0.7213       26
L1-04     0.7500  0.4615 0.5714       13
L1-05     0.7500  0.8824 0.8108       17
L1-06     0.5000  0.3750 0.4286       16
L1-07     0.7692  0.7692 0.7692       13

- Wrote `L1_conditional_diagnostic_metrics.csv`

## 3. L2 confusion matrix
- M[r,p] = count where reference contains r and prediction contains p; rows = ref L2, cols = pred L2.
       L2-01  L2-02  L2-03  L2-04  L2-05  L2-06  L2-07  L2-08
L2-01     19      9      4      2     11      3     15     11
L2-02      4      7      1      0      4      1      4      5
L2-03      8      1      6      0      3      3      4      1
L2-04      2      4      1      7      3      2     10      5
L2-05     11      5      7      6     28      1     15      6
L2-06      1      2      2      1      4      7     12      4
L2-07      8      4      2      0      8      3     38      5
L2-08     14     11      4      6     12      5     16     23

- Wrote `L2_confusion_matrix.csv`

### L2 metrics (per-label binary)
label  Precision  Recall     F1  Support
L2-01     0.6333  0.4750 0.5429       40
L2-02     0.3333  0.3500 0.3415       20
L2-03     0.4615  0.2857 0.3529       21
L2-04     0.7000  0.3500 0.4667       20
L2-05     0.7179  0.5091 0.5957       55
L2-06     0.6364  0.2258 0.3333       31
L2-07     0.6129  0.7037 0.6552       54
L2-08     0.8519  0.5227 0.6479       44

- Wrote `L2_metrics.csv`

- **L2 Micro F1**: 0.5422  |  **L2 Macro F1**: 0.4920

### L2 prediction distribution per true label
true_label  n_samples  n_pred_correct recall                                                           pred_distribution
     L2-01         40              19 47.50%  L2-01:19, L2-07:15, L2-05:11, L2-08:11, L2-02:9, L2-03:4, L2-06:3, L2-04:2
     L2-02         20               7 35.00%               L2-02:7, L2-08:5, L2-05:4, L2-01:4, L2-07:4, L2-03:1, L2-06:1
     L2-03         21               6 28.57%               L2-01:8, L2-03:6, L2-07:4, L2-05:3, L2-06:3, L2-02:1, L2-08:1
     L2-04         20               7 35.00%     L2-07:10, L2-04:7, L2-08:5, L2-02:4, L2-05:3, L2-01:2, L2-06:2, L2-03:1
     L2-05         55              28 50.91%   L2-05:28, L2-07:15, L2-01:11, L2-03:7, L2-08:6, L2-04:6, L2-02:5, L2-06:1
     L2-06         31               7 22.58%     L2-07:12, L2-06:7, L2-08:4, L2-05:4, L2-02:2, L2-03:2, L2-01:1, L2-04:1
     L2-07         54              38 70.37%              L2-07:38, L2-05:8, L2-01:8, L2-08:5, L2-02:4, L2-06:3, L2-03:2
     L2-08         44              23 52.27% L2-08:23, L2-07:16, L2-01:14, L2-05:12, L2-02:11, L2-04:6, L2-06:5, L2-03:4

- Wrote `L2_prediction_distribution.csv`

## 4. Error cases
- **Error rows**: 91
- Wrote `error_cases.csv`