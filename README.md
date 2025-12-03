# Framing the "Nation": Discourse of National Identity Construction in the Taiwan Region

**A Computational Analysis of the 2024 Electoral Campaign**

> **📄 Manuscript & Documentation**
>
> The full theoretical background, methodological rationale, and detailed analysis logic are documented in the manuscript file:  
> **`000000_Manual_Scripts.pdf`** > Please refer to this document for the complete academic context of the code provided in this repository.

---

## 1. Project Overview

This repository contains the computational pipeline for the study **"Framing the 'Nation': Discourse of National Identity Construction in the Taiwan Region."**

Drawing on constructivist theories of nationalism and Laclau’s post-foundational discourse theory, this project investigates how the concept of "the nation" (articulated via *minzu*, *guojia*, *Taiwan*, and *ROC*) functions as a floating signifier in Taiwan's 2024 presidential election.

By integrating **Large Language Model (LLM) annotation** with **network science** and **vector space analysis**, we operationalize theoretical concepts—such as equivalential chains, nodal points, and antagonistic frontiers—at a corpus scale.

**Key Research Objectives:**
1.  **Framing the "Nation":** How do candidates and media outlets organize national identity into recurrent equivalential chains?
2.  **Nodal Signifiers:** Which high-frequency terms occupy central structural positions within these chains?
3.  **Antagonistic Frontiers:** How are "We" and "They" configured along cross-Strait, partisan, and generational axes?

---

## 2. Methodology & Pipeline

The repository implements an end-to-end pipeline ranging from raw text preprocessing to complex inferential analysis.

### A. Two-Layer Annotation Scheme (DPM)
We developed a novel **Defining-Positioning-Mobilizing (DPM)** framing scheme:
* **L1 Generic Frames:** Adapted from the Comparative Agendas Project (e.g., Economic Factors, Security).
* **L2 National Identity Frames:** 15 distinct frames capturing identity construction (e.g., *L2-02 Differentiated Identity*, *L2-07 Shared Crisis*, *L2-15 Democratic Values*).

### B. LLM-Assisted Annotation (GPT-5.1)
To handle the corpus of 15,847 sentence-level units, we utilized **GPT-5.1** via the OpenAI Batch API for cost-effective, high-quality annotation.
* **Validation:** A dual-verified "Gold Set" of 300 units was established.
* **Reliability:** The automated annotation achieved a Krippendorff’s $\alpha > 0.7$ against human coders on key identity frames.
* **Efficiency:** The batch processing pipeline allowed for full-corpus annotation at a highly optimized cost (<$25 USD), balancing scale with precision.

### C. Structural & Causal Analysis
The analytical scripts in this repository perform the following:

#### 1. Co-occurrence Network Analysis (Equivalential Chains)
* **Equivalence Chains:** Identifies which L2 frames frequently co-occur to form stable discursive structures.
* **Conditional Probability:** Calculates $P(L2_B | L2_A)$ to determine the directional strength of frame associations.
* **Community Detection:** Uses weighted network graphs (Nodes = Frames, Edges = Co-occurrence strength) to map distinct discourse clusters, analyzing how different candidates articulate unique identity chains.

#### 2. Nodal Signifier Identification
* **Keyword Extraction:** Extracts high-frequency keywords associated with specific L2 frames.
* **Topic Modeling:** Contextualizes how key terms (e.g., "Democracy," "Peace") function differently depending on the active framing cluster.

#### 3. Semantic Drift & Vector Space Analysis
* **Semantic Drift:** Measures the instability of the signifier "Nation" over time. For each time period $t$, we calculate the mean vector of "Nation" embeddings; higher variance indicates significant semantic contestation.
* **Regression Analysis:** Models semantic drift as a dependent variable against Time, Camp (Partisan affiliation), and Platform to test if semantic divergence increases as the election approaches.
* **UMAP Visualization:** Projects sentence-level embeddings into 2D space (colored by L2 Frame) to visualize how the "Nation" signifier fragments into distinct semantic islands.

#### 4. Antagonistic Frontiers (Us vs. Them)
* **Frontier Construction:** Focuses on exclusionary frames: *L2-02 Differentiated Identity*, *L2-08 Common Enemy*, and *L2-07 Shared Crisis*.
* **Named Entity Recognition (NER):** Automates the extraction of entities constructed as the "Other" (e.g., specific political parties, cross-Strait actors) to map the boundaries of political antagonism.

---

## 3. Repository Structure

This is a **code-only repository**. Due to copyright and privacy constraints, the raw news/social media corpus and the full annotated datasets are not included.

```text
Taiwan_Framing_Discourse/
├── 000000_Manual_Scripts.pdf               # MANUSCRIPT: Theory, method, and full analysis details
├── 05_labels_guidance/                     # Codebook, prompt schemas, and definitions
├── 06_manual_sets/                         # Gold-standard validation sets (300 units)
├── 00_preprocess_to_units.py               # Text cleaning and segmentation
├── 01_convert_units_to_csv.py              # Data formatting
├── 02_wrangling_units.py                   # Corpus preparation
├── 05_validation_gpt5_batch.py             # GPT-5.1 Batch API annotation script
├── 06_analyze_reliability.py               # Krippendorff's Alpha calculation & validation
├── 07_network_analysis.py                  # Co-occurrence networks & community detection
├── 08_nodal_signifiers.py                  # Keyword extraction & topic modeling
├── 09_semantic_drift_umap.py               # Vector analysis, semantic drift, UMAP viz
├── 10_antagonistic_ner.py                  # NER for "Us/Them" boundary mapping
├── 14_current_status_summary.py            # Project status tracker
└── utils_wrangling_bert.py                 # Utility functions
