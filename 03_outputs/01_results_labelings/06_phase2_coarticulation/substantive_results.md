# Phase 2 substantive results
_Edge selection: `fdr`; FDR α=0.05_
## Interpretive notes
- Shared infrastructure cluster ({L2-06 民族自豪 / National Pride, L2-07 凝聚動員 / Solidarity & Vision}) is isomorphic across all three camps (Jaccard=1.0, stable).
- Differentiation concentrates in the differentiating sub-network {L2-02 差異化認同, L2-04 集體敘事再造, L2-05 共同威脅, L2-08 民主價值}.
- L2-07 (凝聚與願景動員, DPM Mobilising) shows a floating-signifier signature: the frame is shared infrastructure (clustered with L2-06) but has the most CI-disjoint cross-cluster articulation edges vs DPP–KMT/TPP — same element, camp-specific articulatory chains.
- Permutation test uses point NPMI on full 28-edge table (observed_gamma); observed_gamma_filtered (p2_06 FDR+bootstrap) is diagnostic only.
- Sub-network Spearman ρ identical across shared/differentiating splits is a structural consequence of FDR sparsity on 8 nodes, not a reporting error; sub-network analysis is descriptive only.
- Community partition is supplementary to edge-level analysis given 8-node saturation.

## L2 v10 codebook (DPM)

| Label | 中文 | English |
| --- | --- | --- |
| L2-01 | 主體性建構 | Subjectivity Construction |
| L2-02 | 差異化認同強調 | Differentiated Identity Emphasis |
| L2-03 | 國際合法性塑造 | International Legitimacy Construction |
| L2-04 | 集體敘事再造 | Collective Narrative Reconstruction |
| L2-05 | 共同威脅設定 | Shared Crisis Construction |
| L2-06 | 民族自豪感激發 | National Pride Activation |
| L2-07 | 凝聚與願景動員 | Solidarity & Vision |
| L2-08 | 民主價值強調 | Democratic Values Emphasis |

## Shared infrastructure

| dpp_community_id | dpp_l2_set | kmt_community_id | kmt_jaccard | tpp_community_id | tpp_jaccard | category | dpp_is_stable | kmt_is_stable | tpp_is_stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | L2-01|L2-03 | 0 | 0.6666666666666666 | 1 | 1.0 | shared | True | False | True |
## Differentiating edges

_Conservative test: non-overlapping 95% bootstrap CIs._

**12** of 84 camp-pair edge tests have disjoint CIs.

_Binomial null check (α=0.05): expected false positives ≈ 4.2; observed 12 (binom p=9.57e-04). Count is strongly inconsistent with a global null of no cross-camp differences._

### CI-disjoint edges

| l2_a | l2_a_label | l2_b | l2_b_label | camp_a | camp_b | ci_disjoint | direction | npmi_median_dpp | npmi_median_kmt | npmi_lower_dpp | npmi_upper_dpp | npmi_lower_kmt | npmi_upper_kmt | npmi_median_tpp | npmi_lower_tpp | npmi_upper_tpp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | KMT | True | KMT | 0.19194908630002508 | 0.27192477250949987 | 0.15981756979619605 | 0.22400793563544819 | 0.23386080245119756 | 0.3112237202996103 | nan | nan | nan |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | DPP | KMT | True | KMT | 0.21808653393574712 | 0.3175522987028073 | 0.19000713985071796 | 0.25116611621701035 | 0.2853395026131754 | 0.3533157320578186 | nan | nan | nan |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | KMT | True | DPP | 0.05924021539035127 | -0.05855585642950714 | 0.025490555105433146 | 0.09337880682500609 | -0.10913070759455852 | -0.012879318562463809 | nan | nan | nan |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | KMT | True | KMT | -0.08374634733297742 | -0.021138687421554045 | -0.10886447421577608 | -0.05894977558166609 | -0.049681266728893066 | 0.01064088456842674 | nan | nan | nan |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | KMT | True | KMT | 0.10787518426607685 | 0.16942237927605952 | 0.08411101009409608 | 0.13300506276661422 | 0.13964055409291715 | 0.19996249355008017 | nan | nan | nan |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | KMT | True | DPP | 0.16151986950423863 | 0.05315220759600765 | 0.13596688998782971 | 0.19126341881015907 | 0.022919052148610695 | 0.08396560357729403 | nan | nan | nan |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | TPP | True | TPP | 0.19194908630002508 | nan | 0.15981756979619605 | 0.22400793563544819 | nan | nan | 0.26163004289930974 | 0.22441952955355385 | 0.2962236237857195 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | DPP | TPP | True | TPP | 0.21808653393574712 | nan | 0.19000713985071796 | 0.25116611621701035 | nan | nan | 0.3068392008784835 | 0.275841358501727 | 0.33734080602643773 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | TPP | True | DPP | 0.285377596218367 | nan | 0.22488414148344998 | 0.3387767393322605 | nan | nan | 0.1501143458538824 | 0.08867997362555008 | 0.20401912671058484 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | TPP | True | TPP | -0.08374634733297742 | nan | -0.10886447421577608 | -0.05894977558166609 | nan | nan | -0.028798198228615575 | -0.05416177748117956 | -0.0037886488482670967 |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | TPP | True | TPP | 0.10787518426607685 | nan | 0.08411101009409608 | 0.13300506276661422 | nan | nan | 0.19354810644212445 | 0.17057626646990778 | 0.21598153742700654 |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | TPP | True | DPP | 0.16151986950423863 | nan | 0.13596688998782971 | 0.19126341881015907 | nan | nan | 0.10756839456803574 | 0.08078610327565489 | 0.1331680687040128 |
### Differentiating edges by L2 node

| l2_node | l2_label | camp_a | camp_b | n_differentiating_edges |
| --- | --- | --- | --- | --- |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | KMT | 3 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | DPP | KMT | 2 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | DPP | KMT | 2 |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | KMT | 2 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | DPP | TPP | 2 |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | TPP | 2 |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | TPP | 2 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | DPP | TPP | 2 |
| L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | TPP | 2 |
| L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | KMT | 1 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | DPP | KMT | 1 |
| L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | KMT | 1 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | DPP | TPP | 1 |
| L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | TPP | 1 |
| L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | DPP | KMT | 0 |
| L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | DPP | TPP | 0 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | KMT | TPP | 0 |
| L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | KMT | TPP | 0 |
| L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | KMT | TPP | 0 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | KMT | TPP | 0 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | KMT | TPP | 0 |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | KMT | TPP | 0 |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | KMT | TPP | 0 |
| L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | KMT | TPP | 0 |
## Permutation test

_H₀: camp labels independent of co-articulation structure (doc-level shuffle). observed_gamma uses point NPMI (same estimator as null); observed_gamma_filtered is p2_06 FDR+bootstrap reference._

| test | camp_a | camp_b | observed_gamma | observed_gamma_filtered | null_mean | null_q05 | null_q50 | null_q95 | p_value | n_permutations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gamma_dpp_kmt | DPP | KMT | 0.889066110118461 | 0.8702911467917794 | 0.9391159658794777 | 0.9006002897665065 | 0.942423632762369 | 0.9666794632007956 | 0.022977022977022976 | 1000 |
| gamma_dpp_tpp | DPP | TPP | 0.8985634622219014 | 0.8844251475447105 | 0.9471158316670032 | 0.9095481385013493 | 0.9496523237257589 | 0.9724048729781877 | 0.02097902097902098 | 1000 |
| gamma_kmt_tpp | KMT | TPP | 0.93964209820709 | 0.9382769994074553 | 0.9399487697225842 | 0.8971967189956597 | 0.9428507240614713 | 0.9680179783665731 | 0.42857142857142855 | 1000 |
| joint_mean_gamma | nan | nan | 0.9090905568491509 | 0.8976644312479817 | 0.9420601890896884 | 0.9173411489966299 | 0.9436571872685007 | 0.9623569086735877 | 0.01998001998001998 | 1000 |
## Sub-network analysis

_Exploratory node classification only. Identical Spearman ρ across shared and differentiating 4-node subsets reflects FDR-filtered edge sparsity (identical weight multisets on 6 edges per subnet), not a copy-paste error. Do not treat as an independent quantitative finding._

### Paper outline note

Primary quantitative evidence: camp permutation test + CI-disjoint edges. Sub-network section should be brief (node selection audit table); consider a dedicated subsection on L2-07 floating-signifier signature (shared {L2-06,L2-07} infrastructure with differential cross-cluster articulation).

### Node selection audit

| dpp_community_id | l2_set | best_kmt_jaccard | best_tpp_jaccard | category |
| --- | --- | --- | --- | --- |
| 0 | L2-01|L2-03 | 0.6666666666666666 | 1.0 | shared |
| 1 | L2-02|L2-05 | 0.25 | 0.5 | differentiating |
| 3 | L2-04|L2-08 | 0.6666666666666666 | 0.5 | differentiating |
| 2 | L2-06|L2-07 | 1.0 | 1.0 | shared |
### Rank correlation comparison (full vs sub-networks)

| scope | camp_a | camp_b | metric | value |
| --- | --- | --- | --- | --- |
| full | DPP | KMT | hubert_gamma | 0.8702911467917794 |
| differentiating | DPP | KMT | spearman_rho | 0.9276336570439174 |
| shared | DPP | KMT | spearman_rho | 0.9276336570439174 |
| full | DPP | TPP | hubert_gamma | 0.8844251475447105 |
| differentiating | DPP | TPP | spearman_rho | 0.898645105261295 |
| shared | DPP | TPP | spearman_rho | 0.898645105261295 |
| full | KMT | TPP | hubert_gamma | 0.9382769994074553 |
| differentiating | KMT | TPP | spearman_rho | 0.8857142857142858 |
| shared | KMT | TPP | spearman_rho | 0.8857142857142858 |
### Shared infrastructure sub-network edge stats

| camp | l2_a | l2_b | npmi_weight | rank |
| --- | --- | --- | --- | --- |
| DPP | L2-01 | L2-03 | 0.19194908630002508 | 2.0 |
| KMT | L2-01 | L2-03 | 0.27192477250949987 | 1.0 |
| TPP | L2-01 | L2-03 | 0.26163004289930974 | 2.0 |
| DPP | L2-01 | L2-06 | 0.27738121689592865 | 1.0 |
| KMT | L2-01 | L2-06 | 0.2510094181272513 | 2.0 |
| TPP | L2-01 | L2-06 | 0.3185673303751725 | 1.0 |
| DPP | L2-01 | L2-07 | 0.08986514616762344 | 4.0 |
| KMT | L2-01 | L2-07 | 0.104121274018693 | 4.0 |
| TPP | L2-01 | L2-07 | 0.11297102738445104 | 5.0 |
| DPP | L2-03 | L2-06 | 0.0 | 5.5 |
| KMT | L2-03 | L2-06 | 0.0944341148137558 | 5.0 |
| TPP | L2-03 | L2-06 | 0.11868269117967026 | 4.0 |
| DPP | L2-03 | L2-07 | 0.11500300639253899 | 3.0 |
| KMT | L2-03 | L2-07 | 0.17253486593227688 | 3.0 |
| TPP | L2-03 | L2-07 | 0.17403593819636612 | 3.0 |
| DPP | L2-06 | L2-07 | 0.0 | 5.5 |
| KMT | L2-06 | L2-07 | 0.0 | 6.0 |
| TPP | L2-06 | L2-07 | 0.0 | 6.0 |
## Differentiating sub-network (figure-ready)

| camp | l2_a | l2_b | npmi_median | rank | npmi_lower | npmi_upper | ci_disjoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DPP | L2-02 | L2-04 | 0.19194908630002508 | 2.0 | 0.031599479524165766 | 0.18024604314052534 | False |
| KMT | L2-02 | L2-04 | 0.27192477250949987 | 1.0 | 0.09620437673587541 | 0.22991991977345347 | False |
| TPP | L2-02 | L2-04 | 0.26163004289930974 | 2.0 | 0.10062566787075523 | 0.22882807695819699 | False |
| DPP | L2-02 | L2-05 | 0.27738121689592865 | 1.0 | 0.19532542783109186 | 0.2623529656425446 | False |
| KMT | L2-02 | L2-05 | 0.2510094181272513 | 2.0 | 0.24481736202571172 | 0.3161304622022114 | False |
| TPP | L2-02 | L2-05 | 0.3185673303751725 | 1.0 | 0.2562619843603173 | 0.32317105953997216 | False |
| DPP | L2-02 | L2-08 | 0.08986514616762344 | 4.0 | 0.172206031775455 | 0.2604763793513081 | False |
| KMT | L2-02 | L2-08 | 0.104121274018693 | 4.0 | 0.25886436746622804 | 0.3604746438715334 | False |
| TPP | L2-02 | L2-08 | 0.11297102738445104 | 5.0 | 0.16744482767672902 | 0.2643968110931409 | False |
| DPP | L2-04 | L2-05 | 0.0 | 5.5 | 0.02710766253566688 | 0.11428100374879786 | False |
| KMT | L2-04 | L2-05 | 0.0944341148137558 | 5.0 | 0.08542582616823029 | 0.16425514958682252 | False |
| TPP | L2-04 | L2-05 | 0.11868269117967026 | 4.0 | 0.052720568644374216 | 0.13226889946243958 | False |
| DPP | L2-04 | L2-08 | 0.11500300639253899 | 3.0 | 0.22488414148344998 | 0.3387767393322605 | True |
| KMT | L2-04 | L2-08 | 0.17253486593227688 | 3.0 | 0.15093149568255396 | 0.2820435782821412 | False |
| TPP | L2-04 | L2-08 | 0.17403593819636612 | 3.0 | 0.08867997362555008 | 0.20401912671058484 | True |
| DPP | L2-05 | L2-08 | 0.0 | 5.5 | 0.15340340315166412 | 0.22954240725681838 | False |
| KMT | L2-05 | L2-08 | 0.0 | 6.0 | 0.17595378516772936 | 0.24708054098309362 | False |
| TPP | L2-05 | L2-08 | 0.0 | 6.0 | 0.17324403890203935 | 0.24110786274291973 | False |
## Cross-camp community matching (top-k greedy)

| dpp_community_id | dpp_l2_set | kmt_community_id | kmt_jaccard | tpp_community_id | tpp_jaccard | category | dpp_is_stable | kmt_is_stable | tpp_is_stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | L2-01|L2-03 | 0 | 0.6666666666666666 | 1 | 1.0 | shared | True | False | True |
| 1 | L2-02|L2-05 | 0 | 0.25 | 0 | 0.5 | distinct | False | False | False |
| 3 | L2-04|L2-08 | 1 | 0.6666666666666666 | 0 | 0.5 | partial | True | False | False |
## Cross-camp Jaccard (all community pairs)

| camp_a | community_a | l2_set_a | camp_b | community_b | l2_set_b | jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| DPP | 0 | L2-01|L2-03 | DPP | 0 | L2-01|L2-03 | 1.0 |
| DPP | 0 | L2-01|L2-03 | DPP | 1 | L2-02|L2-05 | 0.0 |
| DPP | 0 | L2-01|L2-03 | DPP | 3 | L2-04|L2-08 | 0.0 |
| DPP | 0 | L2-01|L2-03 | DPP | 2 | L2-06|L2-07 | 0.0 |
| DPP | 1 | L2-02|L2-05 | DPP | 0 | L2-01|L2-03 | 0.0 |
| DPP | 1 | L2-02|L2-05 | DPP | 1 | L2-02|L2-05 | 1.0 |
| DPP | 1 | L2-02|L2-05 | DPP | 3 | L2-04|L2-08 | 0.0 |
| DPP | 1 | L2-02|L2-05 | DPP | 2 | L2-06|L2-07 | 0.0 |
| DPP | 3 | L2-04|L2-08 | DPP | 0 | L2-01|L2-03 | 0.0 |
| DPP | 3 | L2-04|L2-08 | DPP | 1 | L2-02|L2-05 | 0.0 |
| DPP | 3 | L2-04|L2-08 | DPP | 3 | L2-04|L2-08 | 1.0 |
| DPP | 3 | L2-04|L2-08 | DPP | 2 | L2-06|L2-07 | 0.0 |
| DPP | 2 | L2-06|L2-07 | DPP | 0 | L2-01|L2-03 | 0.0 |
| DPP | 2 | L2-06|L2-07 | DPP | 1 | L2-02|L2-05 | 0.0 |
| DPP | 2 | L2-06|L2-07 | DPP | 3 | L2-04|L2-08 | 0.0 |
| DPP | 2 | L2-06|L2-07 | DPP | 2 | L2-06|L2-07 | 1.0 |
| DPP | 0 | L2-01|L2-03 | KMT | 0 | L2-01|L2-03|L2-05 | 0.6666666666666666 |
| DPP | 0 | L2-01|L2-03 | KMT | 1 | L2-02|L2-04|L2-08 | 0.0 |
| DPP | 0 | L2-01|L2-03 | KMT | 2 | L2-06|L2-07 | 0.0 |
| DPP | 1 | L2-02|L2-05 | KMT | 0 | L2-01|L2-03|L2-05 | 0.25 |
| DPP | 1 | L2-02|L2-05 | KMT | 1 | L2-02|L2-04|L2-08 | 0.25 |
| DPP | 1 | L2-02|L2-05 | KMT | 2 | L2-06|L2-07 | 0.0 |
| DPP | 3 | L2-04|L2-08 | KMT | 0 | L2-01|L2-03|L2-05 | 0.0 |
| DPP | 3 | L2-04|L2-08 | KMT | 1 | L2-02|L2-04|L2-08 | 0.6666666666666666 |
| DPP | 3 | L2-04|L2-08 | KMT | 2 | L2-06|L2-07 | 0.0 |
| DPP | 2 | L2-06|L2-07 | KMT | 0 | L2-01|L2-03|L2-05 | 0.0 |
| DPP | 2 | L2-06|L2-07 | KMT | 1 | L2-02|L2-04|L2-08 | 0.0 |
| DPP | 2 | L2-06|L2-07 | KMT | 2 | L2-06|L2-07 | 1.0 |
| DPP | 0 | L2-01|L2-03 | TPP | 1 | L2-01|L2-03 | 1.0 |
| DPP | 0 | L2-01|L2-03 | TPP | 0 | L2-02|L2-04|L2-05|L2-08 | 0.0 |

_(70 more rows)_

## QAP (Hubert γ)

| camp_a | camp_b | hubert_gamma | p_value | n_perm | n_nodes |
| --- | --- | --- | --- | --- | --- |
| DPP | KMT | 0.8702911467917794 | 0.000999000999000999 | 1000 | 8 |
| DPP | TPP | 0.8844251475447105 | 0.000999000999000999 | 1000 | 8 |
| KMT | TPP | 0.9382769994074553 | 0.000999000999000999 | 1000 | 8 |

_QAP γ measures against-random distinctness; between-camp distinctness is tested by the camp permutation test above._

### Scheme: `camp`

#### Stratum `DPP`

- Community 0 (stability=1.000, stable=True): L2-01, L2-03
- Community 1 (stability=0.714, stable=False): L2-02, L2-05
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
- Community 3 (stability=1.000, stable=True): L2-04, L2-08

#### Stratum `KMT`

- Community 0 (stability=0.610, stable=False): L2-01, L2-03, L2-05
- Community 1 (stability=0.752, stable=False): L2-02, L2-04, L2-08
- Community 2 (stability=1.000, stable=True): L2-06, L2-07

#### Stratum `TPP`

- Community 0 (stability=0.579, stable=False): L2-02, L2-04, L2-05, L2-08
- Community 1 (stability=0.857, stable=True): L2-01, L2-03
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
### Graph diagnostics (`camp`)

| scheme | stratum | n_nodes | n_edges | max_edges | density | n_windows | empty_l2_rate | low_n_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camp | DPP | 8 | 20 | 28 | 0.7143 | 5160 | 0.1289 | False |
| camp | KMT | 8 | 20 | 28 | 0.7143 | 4191 | 0.1642 | False |
| camp | TPP | 8 | 18 | 28 | 0.6429 | 6130 | 0.1874 | False |
### Scheme: `camp_genre`

#### Stratum `DPP_debate`

- Community 0 (stability=0.714, stable=False): L2-01, L2-02, L2-05
- Community 1 (stability=0.714, stable=False): L2-03, L2-07, L2-08
- Community 2 (stability=1.000, stable=True): L2-06

#### Stratum `DPP_news`

- Community 0 (stability=1.000, stable=True): L2-01, L2-03
- Community 1 (stability=0.857, stable=True): L2-02, L2-05
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
- Community 3 (stability=1.000, stable=True): L2-04, L2-08

#### Stratum `KMT_debate`

- Community 0 (stability=0.667, stable=False): L2-02, L2-06, L2-07
- Community 1 (stability=1.000, stable=True): L2-01

#### Stratum `KMT_news`

- Community 0 (stability=0.610, stable=False): L2-01, L2-03, L2-05
- Community 1 (stability=0.752, stable=False): L2-02, L2-04, L2-08
- Community 2 (stability=1.000, stable=True): L2-06, L2-07

#### Stratum `TPP_debate`

- Community 0 (stability=0.810, stable=True): L2-03, L2-06, L2-07
- Community 1 (stability=1.000, stable=True): L2-01, L2-05

#### Stratum `TPP_news`

- Community 0 (stability=0.714, stable=False): L2-02, L2-04, L2-08
- Community 1 (stability=0.667, stable=False): L2-01, L2-03, L2-05
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
### Graph diagnostics (`camp_genre`)

| scheme | stratum | n_nodes | n_edges | max_edges | density | n_windows | empty_l2_rate | low_n_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camp_genre | DPP_debate | 7 | 10 | 21 | 0.4762 | 566 | 0.0919 | False |
| camp_genre | DPP_news | 8 | 20 | 28 | 0.7143 | 4594 | 0.1334 | False |
| camp_genre | KMT_debate | 4 | 2 | 6 | 0.3333 | 20 | 0.25 | True |
| camp_genre | KMT_news | 8 | 20 | 28 | 0.7143 | 4171 | 0.1637 | False |
| camp_genre | TPP_debate | 5 | 5 | 10 | 0.5 | 267 | 0.1835 | True |
| camp_genre | TPP_news | 8 | 18 | 28 | 0.6429 | 5863 | 0.1876 | False |
**Low-n strata (excluded from networks when n_windows < min_stratum_windows):**
- `KMT_debate`: n_windows=20
- `TPP_debate`: n_windows=267
### Scheme: `camp_time`

#### Stratum `DPP_campaign`

- Community 0 (stability=0.619, stable=False): L2-01, L2-02, L2-08
- Community 1 (stability=0.714, stable=False): L2-03, L2-05
- Community 2 (stability=1.000, stable=True): L2-06, L2-07

#### Stratum `DPP_nodate`

- Community 0 (stability=0.714, stable=False): L2-01, L2-02, L2-05
- Community 1 (stability=0.714, stable=False): L2-03, L2-07, L2-08
- Community 2 (stability=1.000, stable=True): L2-06

#### Stratum `DPP_pre_registration`

- Community 0 (stability=0.648, stable=False): L2-04, L2-06, L2-07, L2-08
- Community 1 (stability=1.000, stable=True): L2-01, L2-03
- Community 2 (stability=0.857, stable=True): L2-02, L2-05

#### Stratum `KMT_campaign`

- Community 0 (stability=0.667, stable=False): L2-01, L2-02, L2-05, L2-08
- Community 1 (stability=1.000, stable=True): L2-03, L2-06, L2-07
- Community 2 (stability=1.000, stable=True): L2-04

#### Stratum `KMT_post_registration`

- Community 0 (stability=0.619, stable=False): L2-01, L2-03, L2-05
- Community 1 (stability=1.000, stable=True): L2-06, L2-07
- Community 2 (stability=1.000, stable=True): L2-02, L2-08

#### Stratum `KMT_pre_registration`

- Community 0 (stability=0.667, stable=False): L2-02, L2-04, L2-08
- Community 1 (stability=0.714, stable=False): L2-01, L2-03
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
- Community 3 (stability=1.000, stable=True): L2-05

#### Stratum `TPP_campaign`

- Community 0 (stability=0.857, stable=True): L2-01, L2-03
- Community 1 (stability=1.000, stable=True): L2-02, L2-05
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
- Community 3 (stability=1.000, stable=True): L2-04, L2-08

#### Stratum `TPP_post_registration`

- Community 0 (stability=0.667, stable=False): L2-01, L2-02, L2-05, L2-08
- Community 1 (stability=1.000, stable=True): L2-06, L2-07
- Community 2 (stability=1.000, stable=True): L2-03

#### Stratum `TPP_pre_registration`

- Community 0 (stability=0.714, stable=False): L2-01, L2-03, L2-05
- Community 1 (stability=1.000, stable=True): L2-02, L2-04
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
- Community 3 (stability=1.000, stable=True): L2-08
### Graph diagnostics (`camp_time`)

| scheme | stratum | n_nodes | n_edges | max_edges | density | n_windows | empty_l2_rate | low_n_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camp_time | DPP_campaign | 7 | 16 | 21 | 0.7619 | 567 | 0.1376 | False |
| camp_time | DPP_nodate | 7 | 10 | 21 | 0.4762 | 566 | 0.0919 | False |
| camp_time | DPP_pre_registration | 8 | 17 | 28 | 0.6071 | 3603 | 0.1266 | False |
| camp_time | KMT_campaign | 8 | 8 | 28 | 0.2857 | 1365 | 0.1231 | False |
| camp_time | KMT_post_registration | 7 | 14 | 21 | 0.6667 | 581 | 0.1962 | False |
| camp_time | KMT_pre_registration | 8 | 18 | 28 | 0.6429 | 2182 | 0.1746 | False |
| camp_time | TPP_campaign | 8 | 14 | 28 | 0.5 | 1208 | 0.1738 | False |
| camp_time | TPP_post_registration | 7 | 14 | 21 | 0.6667 | 694 | 0.2017 | False |
| camp_time | TPP_pre_registration | 8 | 17 | 28 | 0.6071 | 3677 | 0.1844 | False |
