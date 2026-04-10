# Batch 标注运行手册

在项目根目录执行（`/mnt/d/Projects/Taiwan_Framing_Discourse` 或你的本地等价路径）。需配置 `OPENAI_API_KEY`（根目录 `.env` 或环境变量）。

**所有脚本位于 `02_src/`，从项目根目录运行：**

```bash
cd /mnt/d/Projects/Taiwan_Framing_Discourse
python 02_src/<脚本名>.py [选项]
```

**模型（可选）：** `--model` 必须写在子命令**之前**。

```bash
python 02_src/05_validation_parallel_async.py --model gpt-5.1 submit --input …
```

**默认指引文件（脚本内已写死路径，可改参数覆盖）：**

| 类型 | 路径 |
|------|------|
| L1 codebook | `01_data/05_labels_guidance/01_annotation_guide_label1_v9.csv` |
| L2 codebook | `01_data/05_labels_guidance/02_annotation_guide_label2_v10.csv` |
| L1 few-shot | `01_data/05_labels_guidance/03_fewshot_L1_v9.csv` |
| L2 few-shot | `01_data/05_labels_guidance/03_fewshot_L2_v10.csv` |
| Hard-case 池 | `01_data/05_labels_guidance/04_hard_case_pool.csv` |

---

## 1. 验证集 — 并行 L1 + L2（L2 不看预测 L1，同时提交两批）

输出：`03_outputs/01_results_labelings/03_validation_output/Run_*/`，并更新 `latest_run.txt`。

```bash
cd /mnt/d/Projects/Taiwan_Framing_Discourse

python 02_src/05_validation_parallel_async.py submit \
  --input 01_data/06_validation_sets/04_validation_set_v2/label1_set_v2_validation.csv

# 等 OpenAI Batch 全部 completed 后：
python 02_src/05_validation_parallel_async.py retrieve

python 02_src/15_validation_analyze.py
```

**常用可选参数：**

```bash
python 02_src/05_validation_parallel_async.py submit \
  --input 01_data/06_validation_sets/04_validation_set_v2/label1_set_v2_validation.csv \
  --no-hard-pool \
  --hard-pool-seed 42
```

---

## 2. 验证集 — 顺序 L1 → L2（L2 条件于预测 L1）

同样写入 `03_validation_output/`，`latest_run.txt` 指向该次 L1 submit 创建的 run。

```bash
cd /mnt/d/Projects/Taiwan_Framing_Discourse

python 02_src/05_validation_l1_async.py submit \
  --input 01_data/06_validation_sets/04_validation_set_v2/label1_set_v2_validation.csv

python 02_src/05_validation_l1_async.py retrieve

python 02_src/05_validation_l2_async.py submit
python 02_src/05_validation_l2_async.py retrieve

python 02_src/15_validation_analyze.py
```

---

## 3. 全语料 — 并行 L1 + L2

递归扫描 `**/*.csv`，需含 `sentence` 列。输出：`03_outputs/01_results_labelings/01_results_datasets/Run_YYYYMMDD_HHMMSS/`，并更新 **`latest_corpus_run.txt`**（与验证的 `latest_run.txt` 独立）。

```bash
cd /mnt/d/Projects/Taiwan_Framing_Discourse

# 默认输入：01_data/04_refined_datasets
python 02_src/05_corpus_parallel_async.py submit

# 或指定根目录：
python 02_src/05_corpus_parallel_async.py submit --input 01_data/04_refined_datasets

python 02_src/05_corpus_parallel_async.py retrieve
```

`retrieve` 只认「语料 run」：run 目录内需有 `ref_parallel.csv` 与 `l1_parallel_batch_ids.json` / `l2_parallel_batch_ids.json`。

---

## 4. 分析脚本（验证指标）

默认读取 `03_validation_output/latest_run.txt` → 对应 `Run_*/final_results.csv`。

```bash
cd /mnt/d/Projects/Taiwan_Framing_Discourse
python 02_src/15_validation_analyze.py
```

若需指定数据或输出目录，见脚本内 `argparse` 说明（`python 02_src/15_validation_analyze.py --help`）。

---

## 5. 各 run 目录内典型文件

| 文件 | 说明 |
|------|------|
| `ref_parallel.csv` / `ref_l1.csv` | 提交时的参考行与 `custom_id` |
| `batch_*_parallel.jsonl` 或 `batch_l1.jsonl` | 已上传的 Batch 输入 |
| `*_batch_ids.json` | OpenAI batch id 列表 |
| `results_l1.csv` | 合并 L1 模型输出后 |
| `final_results.csv` | L2 合并后终表（并行流程在 retrieve 后生成） |
| `run_manifest.json` | 仅语料脚本：记录 `input_resolved`、`n_requests`、`model` |
