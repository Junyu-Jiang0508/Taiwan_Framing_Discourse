# L2-07 最终金标准 vs GPT/DeepSeek 分叉分析

- 金标准构建规则（2026-07-07 人工裁决）：
  1. Junyu 与 z 一致的 39 条 → 取一致值。
  2. `gold_label_adjudication_worksheet.md` 中的 5 条（A011/A023/A043/A045/A048）→ 定义为 **True**（采纳 Junyu 判读）。
  3. 其余 6 条分歧（A002/A026/A027/A033/A038/A046）→ 采纳 **z** 的判读。
- 来源构成核验：consensus=39, worksheet=True=5, z=6（合计50）。
- 金标准中 L2-07=True 共 15/50 条。

## GPT / DeepSeek vs 金标准

| 模型 | 一致率 | κ (95% CI) | Precision | Recall | F1 | TP/FP/FN/TN |
|---|---|---|---|---|---|---|
| GPT | 40% | -0.20 [-0.44, 0.05] | 0.20 | 0.33 | 0.25 | 5/20/10/15 |
| DeepSeek | 72% | 0.44 [0.20, 0.67] | 0.52 | 0.87 | 0.65 | 13/12/2/23 |

GPT vs DeepSeek 相互一致率 20%，κ = -0.60 [-0.79, -0.35]（两模型间，不涉及金标准）。

## 分叉点清单（模型 ≠ 金标准）

**GPT 判错 30/50 条：**

| item_id   | camp   | genre   | gold_source      | gold_l207   | gpt_l207   | ds_l207   |
|:----------|:-------|:--------|:-----------------|:------------|:-----------|:----------|
| A001      | KMT    | news    | consensus        | False       | True       | False     |
| A002      | TPP    | news    | z                | False       | True       | False     |
| A003      | DPP    | news    | consensus        | True        | False      | True      |
| A005      | KMT    | news    | consensus        | True        | False      | True      |
| A006      | KMT    | news    | consensus        | True        | False      | True      |
| A008      | DPP    | news    | consensus        | True        | False      | True      |
| A009      | KMT    | news    | consensus        | False       | True       | False     |
| A011      | KMT    | news    | worksheet(=True) | True        | False      | True      |
| A013      | TPP    | debate  | consensus        | True        | False      | True      |
| A014      | TPP    | news    | consensus        | False       | True       | False     |
| A015      | TPP    | news    | consensus        | False       | True       | False     |
| A018      | TPP    | news    | consensus        | False       | True       | True      |
| A019      | KMT    | news    | consensus        | False       | True       | False     |
| A020      | DPP    | news    | consensus        | False       | True       | False     |
| A023      | DPP    | news    | worksheet(=True) | True        | False      | True      |
| A027      | DPP    | news    | z                | False       | True       | False     |
| A028      | KMT    | debate  | consensus        | False       | True       | False     |
| A030      | DPP    | news    | consensus        | False       | True       | False     |
| A032      | DPP    | news    | consensus        | False       | True       | True      |
| A033      | KMT    | news    | z                | False       | True       | False     |
| A034      | DPP    | news    | consensus        | False       | True       | False     |
| A035      | KMT    | news    | consensus        | False       | True       | False     |
| A040      | KMT    | news    | consensus        | False       | True       | False     |
| A042      | TPP    | news    | consensus        | False       | True       | False     |
| A043      | KMT    | news    | worksheet(=True) | True        | False      | True      |
| A045      | DPP    | news    | worksheet(=True) | True        | False      | True      |
| A046      | TPP    | news    | z                | False       | True       | False     |
| A047      | DPP    | news    | consensus        | False       | True       | False     |
| A048      | DPP    | news    | worksheet(=True) | True        | False      | True      |
| A049      | DPP    | news    | consensus        | False       | True       | False     |

**DeepSeek 判错 14/50 条：**

| item_id   | camp   | genre   | gold_source   | gold_l207   | gpt_l207   | ds_l207   |
|:----------|:-------|:--------|:--------------|:------------|:-----------|:----------|
| A007      | DPP    | news    | consensus     | True        | True       | False     |
| A010      | TPP    | news    | consensus     | False       | False      | True      |
| A016      | TPP    | news    | consensus     | False       | False      | True      |
| A017      | KMT    | news    | consensus     | False       | False      | True      |
| A018      | TPP    | news    | consensus     | False       | True       | True      |
| A021      | KMT    | news    | consensus     | False       | False      | True      |
| A022      | TPP    | news    | consensus     | True        | True       | False     |
| A024      | KMT    | news    | consensus     | False       | False      | True      |
| A025      | TPP    | news    | consensus     | False       | False      | True      |
| A026      | DPP    | news    | z             | False       | False      | True      |
| A031      | TPP    | news    | consensus     | False       | False      | True      |
| A032      | DPP    | news    | consensus     | False       | True       | True      |
| A038      | DPP    | news    | z             | False       | False      | True      |
| A044      | TPP    | news    | consensus     | False       | False      | True      |

## 两模型互相分歧的 40 条上，金标准站边情况

（GPT 与 DeepSeek 在全部 50 条中仅 10 条锚点句一致，其余为模型间分歧句；此前分析受限于 11 条人工分歧句未定案，只能在 29 条『两人工一致』的可靠子集上做站边统计——现在金标准已覆盖全部 50 条，可直接在完整分歧集上统计。）

金标准支持 DeepSeek 28 条，支持 GPT 12 条；符号检验双侧 p = 0.0166。

按分歧方向：

| disagreement_type   |   DeepSeek |   GPT |
|:--------------------|-----------:|------:|
| gpt_false_ds_true   |         10 |    10 |
| gpt_true_ds_false   |         18 |     2 |

按阵营：

| camp   |   DeepSeek |   GPT |
|:-------|-----------:|------:|
| DPP    |         11 |     3 |
| KMT    |         11 |     3 |
| TPP    |          6 |     6 |

## 结论

1. 以此次裁决金标准衡量，**DeepSeek** 与人工标准更接近（GPT 一致率 40%/κ=-0.20 vs DeepSeek 一致率 72%/κ=0.44）。
2. 在两模型互相分歧的 40 条判定题上，金标准站边 DeepSeek 28/40（符号检验 p=0.0166），比此前仅在 29 条可靠子集上估计的比例（DS 19/GPT 10）覆盖面更全、也更极端——本轮裁决把此前5条最具争议的『隐性动员』句判给了 DeepSeek，进一步拉开了两模型的差距。
3. GPT 的错误集中在 `gpt_true_ds_false` 方向的假阳性（把非动员句误判为动员）和 `gpt_false_ds_true` 方向的假阴性（漏判隐性动员句），两种错误在分叉点清单中均有体现，可在论文中直接引用 `gold_standard_l207.csv` 逐条核对。
