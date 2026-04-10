# Taiwan Political Discourse Validation Report
Date: 2026-04-03 15:20:27

- **Reference**: `/mnt/d/Projects/Taiwan_Framing_Discourse/01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv`
- **Note**: If the validation set was constructed via keyword/stratified sampling, the label distribution may differ from the full corpus. P/R/F1 numbers should be interpreted with this sampling bias in mind and may not generalize to the overall dataset.

## 1. L1 confusion matrix
- **L1 accuracy**: 57.39%
- Rows = reference (manual), columns = model prediction
       L1-01  L1-02  L1-03  L1-04  L1-05  L1-06  L1-07
L1-01     11      1      4      1      1      1      1
L1-02      0      1      0      0      2      0      1
L1-03      2      0     27      0      0      2      0
L1-04      0      0      2      4      2      5      3
L1-05      0      1      1      0     13      1      0
L1-06      0      1     13      0      0      4      1
L1-07      1      0      1      1      0      0      6

- Wrote `L1_confusion_matrix.csv`

### L1 metrics (Precision, Recall, F1, Support)
label  Precision  Recall     F1  Support
L1-01     0.7857  0.5500 0.6471       20
L1-02     0.2500  0.2500 0.2500        4
L1-03     0.5625  0.8710 0.6835       31
L1-04     0.6667  0.2500 0.3636       16
L1-05     0.7222  0.8125 0.7647       16
L1-06     0.3077  0.2105 0.2500       19
L1-07     0.5000  0.6667 0.5714        9

- Wrote `L1_metrics.csv`

- **Macro F1**: 0.5043  |  **Weighted F1**: 0.5485

## 2. L1 prediction distribution per true label
true_label  n_samples  n_correct recall                                              pred_distribution
     L1-01         20         11 55.00% L1-01:11, L1-02:1, L1-03:4, L1-04:1, L1-05:1, L1-06:1, L1-07:1
     L1-02          4          1 25.00%                                      L1-02:1, L1-05:2, L1-07:1
     L1-03         31         27 87.10%                                     L1-01:2, L1-03:27, L1-06:2
     L1-04         16          4 25.00%                    L1-03:2, L1-04:4, L1-05:2, L1-06:5, L1-07:3
     L1-05         16         13 81.25%                            L1-02:1, L1-03:1, L1-05:13, L1-06:1
     L1-06         19          4 21.05%                            L1-02:1, L1-03:13, L1-06:4, L1-07:1
     L1-07          9          6 66.67%                             L1-01:1, L1-03:1, L1-04:1, L1-07:6

- Wrote `L1_prediction_distribution.csv`

## 2.5 Conditional L1 diagnostic
Subset where predicted L2 and reference L2 are both nonempty; L1 metrics isolate L1 prompt understanding from empty-L2 cases.
- **Subset**: predicted L2 nonempty AND reference L2 nonempty
- **Subset size**: 115 (excluded 0 from full L1-valid set)
- **L1 accuracy (subset)**: 57.39%

### L1 metrics (conditional subset)
label  Precision  Recall     F1  Support
L1-01     0.7857  0.5500 0.6471       20
L1-02     0.2500  0.2500 0.2500        4
L1-03     0.5625  0.8710 0.6835       31
L1-04     0.6667  0.2500 0.3636       16
L1-05     0.7222  0.8125 0.7647       16
L1-06     0.3077  0.2105 0.2500       19
L1-07     0.5000  0.6667 0.5714        9

- Wrote `L1_conditional_diagnostic_metrics.csv`

## 3. L2 confusion matrix
- M[r,p] = count where reference contains r and prediction contains p; rows = ref L2, cols = pred L2.
       L2-01  L2-02  L2-03  L2-04  L2-05  L2-06  L2-07  L2-08
L2-01     21      8      7      2     13      7     21     17
L2-02      5      7      1      0      4      1      3      5
L2-03     13      1      8      0      5      6      9      9
L2-04      2      4      2      7      3      2      9      5
L2-05      9      6      9      6     26      1     15      6
L2-06      2      1      2      1      4     11     13      8
L2-07      7      3      3      0      8      4     35      6
L2-08     17     11      6      7     11      7     21     29

- Wrote `L2_confusion_matrix.csv`

### L2 metrics (per-label binary)
label  Precision  Recall     F1  Support
L2-01     0.6562  0.5250 0.5833       40
L2-02     0.3684  0.3500 0.3590       20
L2-03     0.4211  0.3810 0.4000       21
L2-04     0.6364  0.3500 0.4516       20
L2-05     0.6667  0.4727 0.5532       55
L2-06     0.6875  0.3548 0.4681       31
L2-07     0.5556  0.6481 0.5983       54
L2-08     0.8056  0.6591 0.7250       44

- Wrote `L2_metrics.csv`

- **L2 Micro F1**: 0.5538  |  **L2 Macro F1**: 0.5173

### L2 prediction distribution per true label
true_label  n_samples  n_pred_correct recall                                                           pred_distribution
     L2-01         40              21 52.50%  L2-07:21, L2-01:21, L2-08:17, L2-05:13, L2-02:8, L2-06:7, L2-03:7, L2-04:2
     L2-02         20               7 35.00%               L2-02:7, L2-08:5, L2-01:5, L2-05:4, L2-07:3, L2-03:1, L2-06:1
     L2-03         21               8 38.10%              L2-01:13, L2-07:9, L2-08:9, L2-03:8, L2-06:6, L2-05:5, L2-02:1
     L2-04         20               7 35.00%      L2-07:9, L2-04:7, L2-08:5, L2-02:4, L2-05:3, L2-01:2, L2-03:2, L2-06:2
     L2-05         55              26 47.27%    L2-05:26, L2-07:15, L2-03:9, L2-01:9, L2-02:6, L2-04:6, L2-08:6, L2-06:1
     L2-06         31              11 35.48%    L2-07:13, L2-06:11, L2-08:8, L2-05:4, L2-03:2, L2-01:2, L2-02:1, L2-04:1
     L2-07         54              35 64.81%              L2-07:35, L2-05:8, L2-01:7, L2-08:6, L2-06:4, L2-02:3, L2-03:3
     L2-08         44              29 65.91% L2-08:29, L2-07:21, L2-01:17, L2-05:11, L2-02:11, L2-06:7, L2-04:7, L2-03:6

- Wrote `L2_prediction_distribution.csv`

## 4. Error cases
- **Error rows**: 104
- Wrote `error_cases.csv`