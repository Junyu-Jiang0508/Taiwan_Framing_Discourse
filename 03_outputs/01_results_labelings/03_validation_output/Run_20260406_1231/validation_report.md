# Taiwan Political Discourse Validation Report
Date: 2026-04-06 12:49:04

- **Reference**: `/mnt/d/Projects/Taiwan_Framing_Discourse/01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv`
- **Note**: If the validation set was constructed via keyword/stratified sampling, the label distribution may differ from the full corpus. P/R/F1 numbers should be interpreted with this sampling bias in mind and may not generalize to the overall dataset.

## 1. L1 confusion matrix
- **L1 accuracy**: 63.75%
- Rows = reference (manual), columns = model prediction
       L1-01  L1-02  L1-03  L1-04  L1-05  L1-06  L1-07
L1-01     10      1      2      0      2      4      1
L1-02      0     19      0      0      0      0      1
L1-03      1      2     22      0      0      7      0
L1-04      0      2      1      9      1      7      2
L1-05      1      2      0      0     15      1      1
L1-06      0      3      5      0      0     15      2
L1-07      1      6      2      0      0      0     12

- Wrote `L1_confusion_matrix.csv`

### L1 metrics (Precision, Recall, F1, Support)
label  Precision  Recall     F1  Support
L1-01     0.7692  0.5000 0.6061       20
L1-02     0.5429  0.9500 0.6909       20
L1-03     0.6875  0.6875 0.6875       32
L1-04     1.0000  0.4091 0.5806       22
L1-05     0.8333  0.7500 0.7895       20
L1-06     0.4412  0.6000 0.5085       25
L1-07     0.6316  0.5714 0.6000       21

- Wrote `L1_metrics.csv`

- **Macro F1**: 0.6376  |  **Weighted F1**: 0.6363

## 2. L1 prediction distribution per true label
true_label  n_samples  n_correct recall                                     pred_distribution
     L1-01         20         10 50.00% L1-01:10, L1-02:1, L1-03:2, L1-05:2, L1-06:4, L1-07:1
     L1-02         20         19 95.00%                                     L1-02:19, L1-07:1
     L1-03         32         22 68.75%                   L1-01:1, L1-02:2, L1-03:22, L1-06:7
     L1-04         22          9 40.91%  L1-02:2, L1-03:1, L1-04:9, L1-05:1, L1-06:7, L1-07:2
     L1-05         20         15 75.00%          L1-01:1, L1-02:2, L1-05:15, L1-06:1, L1-07:1
     L1-06         25         15 60.00%                   L1-02:3, L1-03:5, L1-06:15, L1-07:2
     L1-07         21         12 57.14%                   L1-01:1, L1-02:6, L1-03:2, L1-07:12

- Wrote `L1_prediction_distribution.csv`

## 2.5 Conditional L1 diagnostic
Subset where predicted L2 and reference L2 are both nonempty; L1 metrics isolate L1 prompt understanding from empty-L2 cases.
- **Subset**: predicted L2 nonempty AND reference L2 nonempty
- **Subset size**: 130 (excluded 30 from full L1-valid set)
- **L1 accuracy (subset)**: 66.15%

### L1 metrics (conditional subset)
label  Precision  Recall     F1  Support
L1-01     0.7692  0.5263 0.6250       19
L1-02     0.6316  0.9231 0.7500       13
L1-03     0.6875  0.7333 0.7097       30
L1-04     1.0000  0.4706 0.6400       17
L1-05     0.8125  0.7222 0.7647       18
L1-06     0.4444  0.6000 0.5106       20
L1-07     0.6000  0.6923 0.6429       13

- Wrote `L1_conditional_diagnostic_metrics.csv`

## 3. L2 confusion matrix
- M[r,p] = count where reference contains r and prediction contains p; rows = ref L2, cols = pred L2.
       L2-01  L2-02  L2-03  L2-04  L2-05  L2-06  L2-07  L2-08
L2-01     26      9      9      2     14      8     20     17
L2-02      5     11      2      0      8      5      3      6
L2-03     14      3     11      0      7      8      7      9
L2-04      2      4      0      6      4      5     10      6
L2-05      8      6      6      6     36      2     15      6
L2-06      4      1      2      0      8     15     12      7
L2-07      7      2      2      0      9     10     41      6
L2-08     17     11     10      6     14     10     18     29

- Wrote `L2_confusion_matrix.csv`

### L2 metrics (per-label binary)
label  Precision  Recall     F1  Support
L2-01     0.7429  0.6500 0.6933       40
L2-02     0.5238  0.5500 0.5366       20
L2-03     0.6111  0.5238 0.5641       21
L2-04     0.6667  0.3000 0.4138       20
L2-05     0.6667  0.6545 0.6606       55
L2-06     0.5357  0.4839 0.5085       31
L2-07     0.6119  0.7593 0.6777       54
L2-08     0.7838  0.6591 0.7160       44

- Wrote `L2_metrics.csv`

- **L2 Micro F1**: 0.6318  |  **L2 Macro F1**: 0.5963

### L2 prediction distribution per true label
true_label  n_samples  n_pred_correct recall                                                             pred_distribution
     L2-01         40              26 65.00%    L2-01:26, L2-07:20, L2-08:17, L2-05:14, L2-03:9, L2-02:9, L2-06:8, L2-04:2
     L2-02         20              11 55.00%                L2-02:11, L2-05:8, L2-08:6, L2-06:5, L2-01:5, L2-07:3, L2-03:2
     L2-03         21              11 52.38%               L2-01:14, L2-03:11, L2-08:9, L2-06:8, L2-05:7, L2-07:7, L2-02:3
     L2-04         20               6 30.00%                L2-07:10, L2-04:6, L2-08:6, L2-06:5, L2-02:4, L2-05:4, L2-01:2
     L2-05         55              36 65.45%      L2-05:36, L2-07:15, L2-01:8, L2-03:6, L2-08:6, L2-02:6, L2-04:6, L2-06:2
     L2-06         31              15 48.39%               L2-06:15, L2-07:12, L2-05:8, L2-08:7, L2-01:4, L2-03:2, L2-02:1
     L2-07         54              41 75.93%               L2-07:41, L2-06:10, L2-05:9, L2-01:7, L2-08:6, L2-03:2, L2-02:2
     L2-08         44              29 65.91% L2-08:29, L2-07:18, L2-01:17, L2-05:14, L2-02:11, L2-03:10, L2-06:10, L2-04:6

- Wrote `L2_prediction_distribution.csv`

## 4. Error cases
- **Error rows**: 83
- Wrote `error_cases.csv`