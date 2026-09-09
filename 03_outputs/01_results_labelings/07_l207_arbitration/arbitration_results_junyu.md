# L2-07 人工仲裁结果 — 编码者 Junyu(coder 1)

- 数据来源:`framing_annotation_Junyu.csv`(annotation app 导出,id 9001–9050,50/50 全部完成,无 unsure 标记)
- 对照文件:`answer_key.csv`;逐条合并表见 `arbitration_merged_junyu.csv`
- 分析日期:2026-07-07

## 总体结果

人工标注 L2-07=True 共 21/50。

| 对照 | 一致率(全部 50 条) | Cohen's κ |
|---|---|---|
| 人工 vs GPT | 44% | **−0.12**(低于随机) |
| 人工 vs DeepSeek | 68% | **0.36**(中等偏弱) |

以人工为参照的混淆矩阵(n=50):

| 模型 | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| GPT | 9 | 16 | 12 | 13 | 0.36 | 0.43 |
| DeepSeek | 15 | 10 | 6 | 19 | 0.60 | 0.71 |

## 按 INSTRUCTIONS.md 三步判读逻辑

### 1. 编码者间一致率 — 待定
目前只有 coder 1(Junyu)完成;coder 2 尚未编码,κ_inter 无法计算。

### 2. 人工与模型吻合度 — 明显偏向 DeepSeek
40 条模型分歧句中,人工判定支持 **DeepSeek 26/40(65%)**,支持 GPT 14/40(35%);
符号检验双侧 p = 0.081(方向明确但未达 .05,受 n=40 限制)。

分方向:
- `gpt_true_ds_false`(GPT 疑似过标,n=20):人工支持 DeepSeek 14、GPT 6 → GPT 的 True 判定约七成是人工不认可的假阳性。
- `gpt_false_ds_true`(GPT 疑似漏标,n=20):人工支持 DeepSeek 12、GPT 8 → DeepSeek 补出的 True 六成获人工确认。

分阵营(分歧句):
- DPP:DeepSeek 12/14,GPT 2/14
- KMT:DeepSeek 10/14,GPT 4/14
- **TPP:GPT 8/12,DeepSeek 4/12**(唯一反向;两模型的 L2-07 分歧在 TPP 语料上性质不同,值得质性复查)

### 3. 校准锚点 — 8/10 通过
- `both_false` 5/5 全部一致。
- `both_true` 3/5:A018(TPP/news)、A032(DPP/news)人工判 False —— 人工对"标 True"的门槛比两个模型都更严格,与混淆矩阵中两模型 True 标注量(各 25/50)高于人工(21/50)一致。

## 对 Beat 3 的含义

1. **原 Beat 3 所依据的 GPT L2-07 标签不可信**:在争议句上 GPT 与人工的 κ 为负,以人工为准的 precision 只有 0.36。DeepSeek 复现失败更可能反映 GPT 标签本身的噪声,而非 DeepSeek 的错误。
2. **但 DeepSeek 也只达到中等一致(κ=0.36)**,说明 L2-07 构念在边界句上确有模糊性;即便改用 DeepSeek 标签重跑 Beat 3,也应按预注册的判读规则将其定位为**探索性发现**,并在文中报告本次仲裁数字。
3. TPP 阵营的反向模式提示:两模型在 TPP 语料上的系统性偏差方向不同,若 Beat 3 结论依赖 TPP–其他阵营的对比,需单独敏感性分析。
4. 下一步:补第二名编码者以计算 κ_inter(判读逻辑第 1 步);若 κ_inter 也偏低,则支持"构念模糊"解释而非"模型错误"解释。
