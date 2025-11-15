# Taiwan Framing Discourse

Code for the project **“Taiwan Framing Discourse”**, which analyzes how political actors frame “Taiwan” across news coverage, speeches, and other campaign texts during Taiwanese elections.  
The project combines **LLM-assisted annotation** with **network, lexical, and sequence analysis** to study how contested political meanings are articulated and stabilized.

> **Important:**  
> This repository is a **code-only repository**.  
> It contains the full processing and analysis **pipeline**, but **does not include** the original corpus or the full annotated framing database (for legal and ethical reasons).

---

## 1. About this repository

This repo is intended to:

- Store the **Python scripts** that implement the end-to-end pipeline:
  - from raw text → analysis units → LLM-assisted frame labels → merged framing database;
  - plus downstream analyses (equivalence chains, nodal signifiers, antagonistic boundaries).
- Document the **methodology** used in the accompanying paper / thesis.
- Provide a template for others to **adapt the pipeline** to their own corpora.

It is **not** a data release.  
To respect copyright and privacy constraints, the original news and social media texts, as well as the full annotated dataset, are stored and distributed separately.

For a step-by-step description of the workflow, see:  
`Manual_Scripts.pdf`.

---

## 2. Project overview

The project asks:

- How do different parties and candidates **define and re-frame “Taiwan”** in electoral communication?
- Which **framing patterns** become dominant, and how do they vary across outlets, time, and actors?
- How can we use **LLM-assisted annotation** to build a large-scale framing dataset that supports:
  - rich **descriptive analysis**, and  
  - downstream **causal and structural modeling** with standard ML tools?

Conceptually, the project draws on:

- **Framing theory** and **discourse theory** (e.g., Laclau’s *equivalence chains* and *floating signifiers*).
- Work on **antagonistic boundaries** and **“us vs. them” constructions**, including spatial metaphors in populist discourse (e.g., De Cleen).

Methodologically, we:

1. Build a **multi-level framing label system** (L1 / L2 / rare L3).
2. Use LLMs (e.g., DeepSeek) plus exemplar-based prompts to annotate frames at scale.
3. Construct a **framing database** that serves as a **descriptive backbone**, and then:
   - analyze **equivalence chains** of L2 frames;
   - identify **nodal / floating signifiers** and their re-articulation across parties;
   - examine **antagonistic “us vs them” boundaries** using focal L2 tags and NER.

---

## 3. Repository structure

```text
Taiwan_Framing_Discourse/
├── 00_preprocess_to_units.py
├── 01_convert_units_to_csv.py
├── 02_wrangling_units.py
├── 03_post_filter_review.py
├── 04_create_validation_set.py
├── 05_validation_deepseek_annotation_async.py
├── 06_analyze_annotations.py
├── 07_smart_sampling_for_completion.py
├── 08_prepare_candidates_for_annotation.py
├── 09_compare_target_vs_annotated.py
├── 10_merge_annotations.py
├── 11_targeted_sampling_for_gaps.py
├── 12_compare_round2_and_merge_all.py
├── 13_analyze_user_merged_and_sample.py
├── 14_current_status_summary.py
├── Manual_Scripts.pdf
└── utils_wrangling_bert.py
