# Taiwan Political Discourse Validation Report
Date: 2026-04-06 22:03:13

- **Reference**: `/mnt/d/Projects/Taiwan_Framing_Discourse/01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv`
- **Note**: If the validation set was constructed via keyword/stratified sampling, the label distribution may differ from the full corpus. P/R/F1 numbers should be interpreted with this sampling bias in mind and may not generalize to the overall dataset.

## 1. L1 confusion matrix
- **L1 accuracy**: 68.35%
- Rows = reference (manual), columns = model prediction
       L1-01  L1-02  L1-03  L1-04  L1-05  L1-06  L1-07
L1-01     12      0      1      1      0      0      0
L1-02      0      7      0      0      1      0     10
L1-03      3      0     33      2      1      0      1
L1-04      1      0      2     19      3      4      1
L1-05      1      1      1      1     12      0      1
L1-06      0      1      8      0      0      6      2
L1-07      1      1      0      1      0      0     19

- Wrote `L1_confusion_matrix.csv`

### L1 metrics (Precision, Recall, F1, Support)
label  Precision  Recall     F1  Support
L1-01     0.6667  0.8571 0.7500       14
L1-02     0.7000  0.3889 0.5000       18
L1-03     0.7333  0.8250 0.7765       40
L1-04     0.7917  0.6333 0.7037       30
L1-05     0.7059  0.7059 0.7059       17
L1-06     0.6000  0.3529 0.4444       17
L1-07     0.5588  0.8636 0.6786       22

- Wrote `L1_metrics.csv`

- **Macro F1**: 0.6513  |  **Weighted F1**: 0.6719

## 2. L1 prediction distribution per true label
true_label  n_samples  n_correct recall                                     pred_distribution
     L1-01         14         12 85.71%                            L1-01:12, L1-03:1, L1-04:1
     L1-02         18          7 38.89%                            L1-02:7, L1-05:1, L1-07:10
     L1-03         40         33 82.50%          L1-01:3, L1-03:33, L1-04:2, L1-05:1, L1-07:1
     L1-04         30         19 63.33% L1-01:1, L1-03:2, L1-04:19, L1-05:3, L1-06:4, L1-07:1
     L1-05         17         12 70.59% L1-01:1, L1-02:1, L1-03:1, L1-04:1, L1-05:12, L1-07:1
     L1-06         17          6 35.29%                    L1-02:1, L1-03:8, L1-06:6, L1-07:2
     L1-07         22         19 86.36%                   L1-01:1, L1-02:1, L1-04:1, L1-07:19

- Wrote `L1_prediction_distribution.csv`

## 2.5 Conditional L1 diagnostic
Subset where predicted L2 and reference L2 are both nonempty; L1 metrics isolate L1 prompt understanding from empty-L2 cases.
- **Subset**: predicted L2 nonempty AND reference L2 nonempty
- **Subset size**: 127 (excluded 31 from full L1-valid set)
- **L1 accuracy (subset)**: 68.50%

### L1 metrics (conditional subset)
label  Precision  Recall     F1  Support
L1-01     0.6471  0.9167 0.7586       12
L1-02     0.4000  0.2500 0.3077        8
L1-03     0.7209  0.8378 0.7750       37
L1-04     0.8333  0.6250 0.7143       24
L1-05     0.7500  0.7500 0.7500       16
L1-06     0.6250  0.3125 0.4167       16
L1-07     0.5500  0.7857 0.6471       14

- Wrote `L1_conditional_diagnostic_metrics.csv`

## 3. L2 confusion matrix
- M[r,p] = count where reference contains r and prediction contains p; rows = ref L2, cols = pred L2.
       L2-01  L2-02  L2-03  L2-04  L2-05  L2-06  L2-07  L2-08
L2-01     27     10     10      2     13     13     25     17
L2-02      6      8      0      1      6      3      3      4
L2-03      7      1     10      0      1      7      5      9
L2-04      4      4      1     10      8      7      5      4
L2-05     10      7      3      4     28      6     14      9
L2-06      4      4      7      2     11     29     24     10
L2-07      7      2      6      2      9     19     42      7
L2-08     14      9      8      6     11      7     14     30

- Wrote `L2_confusion_matrix.csv`

### L2 metrics (per-label binary)
label  Precision  Recall     F1  Support
L2-01     0.8438  0.6279 0.7200       43
L2-02     0.3810  0.6667 0.4848       12
L2-03     0.5000  0.7692 0.6061       13
L2-04     0.7692  0.5263 0.6250       19
L2-05     0.6667  0.8485 0.7467       33
L2-06     0.6170  0.7073 0.6591       41
L2-07     0.6000  0.9130 0.7241       46
L2-08     0.8108  0.8333 0.8219       36

- Wrote `L2_metrics.csv`

- **L2 Micro F1**: 0.7010  |  **L2 Macro F1**: 0.6735

### L2 prediction distribution per true label
true_label  n_samples  n_pred_correct recall                                                             pred_distribution
     L2-01         43              27 62.79% L2-01:27, L2-07:25, L2-08:17, L2-05:13, L2-06:13, L2-03:10, L2-02:10, L2-04:2
     L2-02         12               8 66.67%                 L2-02:8, L2-01:6, L2-05:6, L2-08:4, L2-06:3, L2-07:3, L2-04:1
     L2-03         13              10 76.92%                L2-03:10, L2-08:9, L2-06:7, L2-01:7, L2-07:5, L2-02:1, L2-05:1
     L2-04         19              10 52.63%       L2-04:10, L2-05:8, L2-06:7, L2-07:5, L2-01:4, L2-08:4, L2-02:4, L2-03:1
     L2-05         33              28 84.85%     L2-05:28, L2-07:14, L2-01:10, L2-08:9, L2-02:7, L2-06:6, L2-04:4, L2-03:3
     L2-06         41              29 70.73%    L2-06:29, L2-07:24, L2-05:11, L2-08:10, L2-03:7, L2-02:4, L2-01:4, L2-04:2
     L2-07         46              42 91.30%      L2-07:42, L2-06:19, L2-05:9, L2-01:7, L2-08:7, L2-03:6, L2-04:2, L2-02:2
     L2-08         36              30 83.33%    L2-08:30, L2-07:14, L2-01:14, L2-05:11, L2-02:9, L2-03:8, L2-06:7, L2-04:6

- Wrote `L2_prediction_distribution.csv`

## 4. Error cases
- **Error rows**: 59
- Wrote `error_cases.csv`