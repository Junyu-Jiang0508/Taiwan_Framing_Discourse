# Taiwan Political Discourse Validation Report
Date: 2026-04-07 11:44:22

- **Reference**: `/mnt/d/Projects/Taiwan_Framing_Discourse/01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv`
- **Note**: If the validation set was constructed via keyword/stratified sampling, the label distribution may differ from the full corpus. P/R/F1 numbers should be interpreted with this sampling bias in mind and may not generalize to the overall dataset.

## 1. L1 confusion matrix
- **L1 accuracy**: 74.05%
- Rows = reference (manual), columns = model prediction
       L1-01  L1-02  L1-03  L1-04  L1-05  L1-06  L1-07
L1-01     13      0      1      0      0      0      0
L1-02      0     16      0      0      2      0      0
L1-03      2      1     33      1      0      2      1
L1-04      1      1      2     17      4      3      2
L1-05      1      2      0      1     11      1      1
L1-06      0      0      2      1      0     13      1
L1-07      0      6      1      1      0      0     14

- Wrote `L1_confusion_matrix.csv`

### L1 metrics (Precision, Recall, F1, Support)
label  Precision  Recall     F1  Support
L1-01     0.7647  0.9286 0.8387       14
L1-02     0.6154  0.8889 0.7273       18
L1-03     0.8462  0.8250 0.8354       40
L1-04     0.8095  0.5667 0.6667       30
L1-05     0.6471  0.6471 0.6471       17
L1-06     0.6842  0.7647 0.7222       17
L1-07     0.7368  0.6364 0.6829       22

- Wrote `L1_metrics.csv`

- **Macro F1**: 0.7315  |  **Weighted F1**: 0.7377

## 2. L1 prediction distribution per true label
true_label  n_samples  n_correct recall                                              pred_distribution
     L1-01         14         13 92.86%                                              L1-01:13, L1-03:1
     L1-02         18         16 88.89%                                              L1-02:16, L1-05:2
     L1-03         40         33 82.50%          L1-01:2, L1-02:1, L1-03:33, L1-04:1, L1-06:2, L1-07:1
     L1-04         30         17 56.67% L1-01:1, L1-02:1, L1-03:2, L1-04:17, L1-05:4, L1-06:3, L1-07:2
     L1-05         17         11 64.71%          L1-01:1, L1-02:2, L1-04:1, L1-05:11, L1-06:1, L1-07:1
     L1-06         17         13 76.47%                            L1-03:2, L1-04:1, L1-06:13, L1-07:1
     L1-07         22         14 63.64%                            L1-02:6, L1-03:1, L1-04:1, L1-07:14

- Wrote `L1_prediction_distribution.csv`

## 2.5 Conditional L1 diagnostic
Subset where predicted L2 and reference L2 are both nonempty; L1 metrics isolate L1 prompt understanding from empty-L2 cases.
- **Subset**: predicted L2 nonempty AND reference L2 nonempty
- **Subset size**: 125 (excluded 33 from full L1-valid set)
- **L1 accuracy (subset)**: 73.60%

### L1 metrics (conditional subset)
label  Precision  Recall     F1  Support
L1-01     0.7333  0.9167 0.8148       12
L1-02     0.4667  1.0000 0.6364        7
L1-03     0.8421  0.8421 0.8421       38
L1-04     0.7500  0.5000 0.6000       24
L1-05     0.7333  0.6875 0.7097       16
L1-06     0.7333  0.7857 0.7586       14
L1-07     0.7273  0.5714 0.6400       14

- Wrote `L1_conditional_diagnostic_metrics.csv`

## 3. L2 confusion matrix
- M[r,p] = count where reference contains r and prediction contains p; rows = ref L2, cols = pred L2.
       L2-01  L2-02  L2-03  L2-04  L2-05  L2-06  L2-07  L2-08
L2-01     27      8     10      4     15     12     19     17
L2-02      6     10      0      1      6      2      4      4
L2-03      8      2      9      0      3      7      5      8
L2-04      5      3      1     10      8      7      4      4
L2-05     10      6      2      5     30      8     11      8
L2-06      6      4      7      3      9     30     26      9
L2-07      7      2      6      3      8     21     42      6
L2-08     15      9      7      7     14      7     11     29

- Wrote `L2_confusion_matrix.csv`

### L2 metrics (per-label binary)
label  Precision  Recall     F1  Support
L2-01     0.7500  0.6279 0.6835       43
L2-02     0.5263  0.8333 0.6452       12
L2-03     0.5000  0.6923 0.5806       13
L2-04     0.7143  0.5263 0.6061       19
L2-05     0.6818  0.9091 0.7792       33
L2-06     0.6000  0.7317 0.6593       41
L2-07     0.6562  0.9130 0.7636       46
L2-08     0.8529  0.8056 0.8286       36

- Wrote `L2_metrics.csv`

- **L2 Micro F1**: 0.7165  |  **L2 Macro F1**: 0.6933

### L2 prediction distribution per true label
true_label  n_samples  n_pred_correct recall                                                            pred_distribution
     L2-01         43              27 62.79% L2-01:27, L2-07:19, L2-08:17, L2-05:15, L2-06:12, L2-03:10, L2-02:8, L2-04:4
     L2-02         12              10 83.33%               L2-02:10, L2-01:6, L2-05:6, L2-08:4, L2-07:4, L2-06:2, L2-04:1
     L2-03         13               9 69.23%                L2-03:9, L2-01:8, L2-08:8, L2-06:7, L2-07:5, L2-05:3, L2-02:2
     L2-04         19              10 52.63%      L2-04:10, L2-05:8, L2-06:7, L2-01:5, L2-08:4, L2-07:4, L2-02:3, L2-03:1
     L2-05         33              30 90.91%    L2-05:30, L2-07:11, L2-01:10, L2-06:8, L2-08:8, L2-02:6, L2-04:5, L2-03:2
     L2-06         41              30 73.17%     L2-06:30, L2-07:26, L2-08:9, L2-05:9, L2-03:7, L2-01:6, L2-02:4, L2-04:3
     L2-07         46              42 91.30%     L2-07:42, L2-06:21, L2-05:8, L2-01:7, L2-03:6, L2-08:6, L2-04:3, L2-02:2
     L2-08         36              29 80.56%   L2-08:29, L2-01:15, L2-05:14, L2-07:11, L2-02:9, L2-06:7, L2-04:7, L2-03:7

- Wrote `L2_prediction_distribution.csv`

## 4. Error cases
- **Error rows**: 50
- Wrote `error_cases.csv`