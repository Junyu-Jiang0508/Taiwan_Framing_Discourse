# Taiwan Political Discourse Validation Report
Date: 2026-04-03 15:16:11

- **Reference**: `/mnt/d/Projects/Taiwan_Framing_Discourse/01_data/06_validation_sets/04_validation_set_v2/label1_set_v2.csv`
- **Note**: If the validation set was constructed via keyword/stratified sampling, the label distribution may differ from the full corpus. P/R/F1 numbers should be interpreted with this sampling bias in mind and may not generalize to the overall dataset.

## 1. L1 confusion matrix
- **L1 accuracy**: 63.93%
- Rows = reference (manual), columns = model prediction
       L1-01  L1-02  L1-03  L1-04  L1-05  L1-06  L1-07
L1-01     11      0      3      2      3      1      0
L1-02      0      4      0      0      2      0      1
L1-03      1      0     29      0      0      0      0
L1-04      0      0      1      6      1      6      2
L1-05      0      1      1      0     13      0      1
L1-06      0      1     12      0      0      5      2
L1-07      0      0      2      1      0      0     10

- Wrote `L1_confusion_matrix.csv`

### L1 metrics (Precision, Recall, F1, Support)
label  Precision  Recall     F1  Support
L1-01     0.9167  0.5500 0.6875       20
L1-02     0.6667  0.5714 0.6154        7
L1-03     0.6042  0.9667 0.7436       30
L1-04     0.6667  0.3750 0.4800       16
L1-05     0.6842  0.8125 0.7429       16
L1-06     0.4167  0.2500 0.3125       20
L1-07     0.6250  0.7692 0.6897       13

- Wrote `L1_metrics.csv`

- **Macro F1**: 0.6102  |  **Weighted F1**: 0.6160

## 2. L1 prediction distribution per true label
true_label  n_samples  n_correct recall                            pred_distribution
     L1-01         20         11 55.00% L1-01:11, L1-03:3, L1-04:2, L1-05:3, L1-06:1
     L1-02          7          4 57.14%                    L1-02:4, L1-05:2, L1-07:1
     L1-03         30         29 96.67%                            L1-01:1, L1-03:29
     L1-04         16          6 37.50%  L1-03:1, L1-04:6, L1-05:1, L1-06:6, L1-07:2
     L1-05         16         13 81.25%          L1-02:1, L1-03:1, L1-05:13, L1-07:1
     L1-06         20          5 25.00%          L1-02:1, L1-03:12, L1-06:5, L1-07:2
     L1-07         13         10 76.92%                   L1-03:2, L1-04:1, L1-07:10

- Wrote `L1_prediction_distribution.csv`

## 2.5 Conditional L1 diagnostic
Subset where predicted L2 and reference L2 are both nonempty; L1 metrics isolate L1 prompt understanding from empty-L2 cases.
- **Subset**: predicted L2 nonempty AND reference L2 nonempty
- **Subset size**: 122 (excluded 0 from full L1-valid set)
- **L1 accuracy (subset)**: 63.93%

### L1 metrics (conditional subset)
label  Precision  Recall     F1  Support
L1-01     0.9167  0.5500 0.6875       20
L1-02     0.6667  0.5714 0.6154        7
L1-03     0.6042  0.9667 0.7436       30
L1-04     0.6667  0.3750 0.4800       16
L1-05     0.6842  0.8125 0.7429       16
L1-06     0.4167  0.2500 0.3125       20
L1-07     0.6250  0.7692 0.6897       13

- Wrote `L1_conditional_diagnostic_metrics.csv`

## 3. L2 confusion matrix
- M[r,p] = count where reference contains r and prediction contains p; rows = ref L2, cols = pred L2.
       L2-01  L2-02  L2-03  L2-04  L2-05  L2-06  L2-07  L2-08
L2-01     18      8      6      0      8      8     17     17
L2-02      4      9      0      0      5      3      3      5
L2-03      8      2     11      0      3      6      8      8
L2-04      1      6      2      5      7      2      5      5
L2-05      3      8      7      4     32      3     13      7
L2-06      2      2      3      2      7     11     10      9
L2-07      5      1      7      1      8      4     33      7
L2-08     14     11      7      2     14      9     14     29

- Wrote `L2_confusion_matrix.csv`

### L2 metrics (per-label binary)
label  Precision  Recall     F1  Support
L2-01     0.8182  0.4500 0.5806       40
L2-02     0.3913  0.4500 0.4186       20
L2-03     0.5000  0.5238 0.5116       21
L2-04     0.7143  0.2500 0.3704       20
L2-05     0.7111  0.5818 0.6400       55
L2-06     0.5500  0.3548 0.4314       31
L2-07     0.6226  0.6111 0.6168       54
L2-08     0.7838  0.6591 0.7160       44

- Wrote `L2_metrics.csv`

- **L2 Micro F1**: 0.5759  |  **L2 Macro F1**: 0.5357

### L2 prediction distribution per true label
true_label  n_samples  n_pred_correct recall                                                           pred_distribution
     L2-01         40              18 45.00%            L2-01:18, L2-07:17, L2-08:17, L2-05:8, L2-06:8, L2-02:8, L2-03:6
     L2-02         20               9 45.00%                        L2-02:9, L2-05:5, L2-08:5, L2-01:4, L2-06:3, L2-07:3
     L2-03         21              11 52.38%              L2-03:11, L2-07:8, L2-01:8, L2-08:8, L2-06:6, L2-05:3, L2-02:2
     L2-04         20               5 25.00%      L2-05:7, L2-02:6, L2-08:5, L2-04:5, L2-07:5, L2-03:2, L2-06:2, L2-01:1
     L2-05         55              32 58.18%    L2-05:32, L2-07:13, L2-02:8, L2-03:7, L2-08:7, L2-04:4, L2-06:3, L2-01:3
     L2-06         31              11 35.48%    L2-06:11, L2-07:10, L2-08:9, L2-05:7, L2-03:3, L2-02:2, L2-04:2, L2-01:2
     L2-07         54              33 61.11%     L2-07:33, L2-05:8, L2-03:7, L2-08:7, L2-01:5, L2-06:4, L2-02:1, L2-04:1
     L2-08         44              29 65.91% L2-08:29, L2-05:14, L2-07:14, L2-01:14, L2-02:11, L2-06:9, L2-03:7, L2-04:2

- Wrote `L2_prediction_distribution.csv`

## 4. Error cases
- **Error rows**: 90
- Wrote `error_cases.csv`