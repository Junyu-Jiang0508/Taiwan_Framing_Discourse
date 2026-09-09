# 抓取残留过滤：前后对比

对应 `04_documents/cohesion_next_steps.md` §0.3 与 §5 表第 1 项。

## 这里有什么

| 文件 | 内容 |
| --- | --- |
| `delta_gpt.md` | GPT L2（对照规格）过滤前后的全部 Phase 2 估计对比 |
| `delta_ds.md` | DeepSeek L2（主规格）同上 |

两份都按同一结构：语料规模 → 边 NPMI 与 FDR 边集 → 凝聚簇六条边逐条 →
Leiden 共识划分 ARI → QAP γ 与阵营置换 → 差异化子网 → 窗口/时间稳健性。

## 怎么重现

```bash
# 过滤策略在 config 里：corpus_filters.boilerplate_policy
python3 02_src/phase2/run_phase2.py --mode full                                   # GPT
python3 02_src/phase2/run_phase2.py --mode full --config 02_src/phase2/phase2_config_deepseek.yaml

# 对比（BASELINE 需是过滤前的 artifacts 快照）
python3 02_src/phase2/probes/boilerplate_filter_delta.py BASELINE CURRENT --label GPT --out delta_gpt.md

# 策略代价：drop_residue vs drop_all（后者需先用改了 policy 的 config 跑 p2_00-p2_02）
python3 02_src/phase2/probes/policy_sensitivity.py NOFILTER DROP_RESIDUE DROP_ALL
```

模式与分层定义在 `02_src/phase2/utils/boilerplate.py`；逐模式的标签负荷每次跑都重算到
各 artifacts 目录的 `diagnostics/boilerplate_audit.json`——那张表是分层判断的依据，
不要依赖代码注释里的常数。

## 一句话结论

删掉 1,314 行（5.86%，全部新闻）后：边估计、FDR 边集、凝聚簇六条边、QAP、阵营对比
全部存活（NPMI 相关 0.997–0.998）；**唯一变动的是 Leiden 社群成员名单**，说明社群检测
在 8 节点网络上本身不稳，论文应只报 QAP 层面的结构相似度，不报具体成员划分。
