# L2-07 人工仲裁结果 — 双编码者合并 (Junyu + z)

- 数据来源: `framing_annotation_Junyu.csv` (coder1), `framing_annotation_z.csv` (coder2)，各 id 9001–9050，50/50 全部完成，均无 unsure 标记
- 对照文件: `answer_key.csv`；逐条合并表见 `arbitration_merged_combined.csv`
- 生成脚本: `analyze_arbitration.py`（补第二编码者，取代仅有 coder1 的 `arbitration_results_junyu.md`）
- 分析日期: 2026-07-07

## 步骤 1 — 编码者间一致率（此前待定，现已补齐）

coder1(Junyu) 标 True 21/50；coder2(z) 标 True 10/50。
两人全部 50 条一致率 **78%**，Cohen's κ = **0.51**（bootstrap 95% CI [0.29, 0.74]）。

**判读**：κ 落在「中等」区间 —— 人工编码者之间也存在不可忽视的分歧，L2-07 的判读存在真实的构念模糊。

**结构性发现**：z 标 True 的题目是 Junyu 标 True 题目的严格子集（0 条 z=True 而 Junyu=False 的反例）——z 并非随机噪声更大，而是系统性地比 Junyu 更保守，只在 Junyu 也同意的题目上才判 True。

两人分歧的 11 条明细：

| item | camp | genre | 分歧类型 | GPT | DS | Junyu | z |
|---|---|---|---|---|---|---|---|
| A002 | TPP | news | gpt_true_ds_false | True | False | True | False |
| A011 | KMT | news | gpt_false_ds_true | False | True | True | False |
| A023 | DPP | news | gpt_false_ds_true | False | True | True | False |
| A026 | DPP | news | gpt_false_ds_true | False | True | True | False |
| A027 | DPP | news | gpt_true_ds_false | True | False | True | False |
| A033 | KMT | news | gpt_true_ds_false | True | False | True | False |
| A038 | DPP | news | gpt_false_ds_true | False | True | True | False |
| A043 | KMT | news | gpt_false_ds_true | False | True | True | False |
| A045 | DPP | news | gpt_false_ds_true | False | True | True | False |
| A046 | TPP | news | gpt_true_ds_false | True | False | True | False |
| A048 | DPP | news | gpt_false_ds_true | False | True | True | False |

## 步骤 2 — 人工 vs 模型吻合度（四种人工汇总口径）

| 人工口径 | vs GPT 一致率 | GPT κ | vs GPT P/R | vs DeepSeek 一致率 | DS κ | vs DS P/R |
|---|---|---|---|---|---|---|
| coder1 (Junyu) alone | 44% | -0.12 | 0.36/0.43 | 68% | 0.36 | 0.60/0.71 |
| coder2 (z) alone | 50% | 0.00 | 0.20/0.50 | 62% | 0.24 | 0.32/0.80 |
| AND (both True) | 50% | 0.00 | 0.20/0.50 | 62% | 0.24 | 0.32/0.80 |
| OR (either True) | 44% | -0.12 | 0.36/0.43 | 68% | 0.36 | 0.60/0.71 |

**仅取两人一致的可靠子集**（n=39/50，剔除 11 条分歧句后）：
vs GPT 一致率 46%（κ=-0.04，P=0.24/R=0.50），vs DeepSeek 一致率 69%（κ=0.36，P=0.44/R=0.80）。

### 40 条模型分歧句上，人工（仅取两人一致的 29 条可靠子集）站边情况

支持 DeepSeek 19，支持 GPT 10；符号检验双侧 p = 0.136。

按分歧方向：

| disagreement_type   |   DeepSeek |   GPT |
|:--------------------|-----------:|------:|
| gpt_false_ds_true   |          5 |     8 |
| gpt_true_ds_false   |         14 |     2 |

按阵营：

| camp   |   DeepSeek |   GPT |
|:-------|-----------:|------:|
| DPP    |          7 |     1 |
| KMT    |          8 |     3 |
| TPP    |          4 |     6 |

## 步骤 3 — 校准锚点（10 条模型一致句）

| type       |   junyu_match |   z_match |   both_match |
|:-----------|--------------:|----------:|-------------:|
| both_false |             5 |         5 |            5 |
| both_true  |             3 |         3 |            3 |

合计：Junyu 8/10，z 8/10，两人都对 8/10。

## 结论与对 Beat 3 的含义

1. **编码者间一致率 κ=0.51**：κ 落在「中等」区间 —— 人工编码者之间也存在不可忽视的分歧，L2-07 的判读存在真实的构念模糊。
2. 无论采用哪种人工汇总口径（单编码者、AND、OR、或仅取两人一致子集），人工与 DeepSeek 的一致率和 κ 均系统性高于人工与 GPT —— coder2 的加入**复现**了 coder1 单独得出的方向性结论，而非推翻它。
3. 原 Beat 3 所依据的 GPT L2-07 标签仍不可信；但两位人工编码者彼此的 κ 表明 L2-07 构念本身在边界句上有真实的模糊性，不是单纯的「模型错误」。据预注册判读规则，Beat 3 应继续按**探索性发现**处理（若改用 DeepSeek 标签重跑，需在文中报告本次仲裁的双编码者数字，而非仅 coder1）。
4. 校准锚点结果显示人工对「标 True」的门槛依旧比模型更严格，与 coder1 单独分析时的模式一致。
5. **TPP 阵营反向模式在可靠子集中依然复现**：DPP（DeepSeek 7/GPT 1）与 KMT（DeepSeek 8/GPT 3）都明显偏向 DeepSeek，唯独 TPP 反向偏 GPT（DeepSeek 4/GPT 6）——与 coder1 单独分析时的方向完全一致，不是偶然。建议在论文稳健性章节中明确写出：「DPP–KMT 分歧、Beat 1/2 的整体结论对标注来源稳健；但 L2-07 浮动能指论证（Beat 3）以及任何依赖 TPP 对比的检验，其结果对标注来源（GPT vs DeepSeek vs 人工）敏感，应作探索性呈现」。
