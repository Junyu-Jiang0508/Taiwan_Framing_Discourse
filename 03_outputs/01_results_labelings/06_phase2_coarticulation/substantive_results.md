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

**15** of 84 camp-pair edge tests have disjoint CIs.

_Binomial null check (α=0.05): expected false positives ≈ 4.2; observed 15 (binom p=1.69e-05). Count is strongly inconsistent with a global null of no cross-camp differences._

### CI-disjoint edges

| l2_a | l2_a_label | l2_b | l2_b_label | camp_a | camp_b | ci_disjoint | direction | npmi_median_dpp | npmi_median_kmt | npmi_lower_dpp | npmi_upper_dpp | npmi_lower_kmt | npmi_upper_kmt | npmi_median_tpp | npmi_lower_tpp | npmi_upper_tpp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | KMT | True | KMT | 0.1874041756443085 | 0.27323844539997943 | 0.15703345883804143 | 0.21579610096882293 | 0.23834378526149277 | 0.3087844142374538 | nan | nan | nan |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | DPP | KMT | True | KMT | 0.21884672957057033 | 0.3156271137845629 | 0.19174336386994595 | 0.25184050786594486 | 0.2799417568972123 | 0.34773394032280613 | nan | nan | nan |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | KMT | True | DPP | 0.05931911983731307 | -0.060166489963657796 | 0.02658105602821392 | 0.09334098467597209 | -0.10328989420740907 | -0.014954241285816974 | nan | nan | nan |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | KMT | True | KMT | -0.07632472546409261 | -0.02271057866036511 | -0.09876258407492851 | -0.053792463231776166 | -0.053245893213594075 | 0.010341441791580553 | nan | nan | nan |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | KMT | True | KMT | 0.11530260594454125 | 0.1731544236593423 | 0.08922142680106034 | 0.14069897955952973 | 0.14407692995298443 | 0.20306006348844405 | nan | nan | nan |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | KMT | True | DPP | 0.16223799585085974 | 0.0617537759556533 | 0.1363997119552298 | 0.1888925259141272 | 0.030992511793288647 | 0.09092287372975905 | nan | nan | nan |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | TPP | True | TPP | 0.1874041756443085 | nan | 0.15703345883804143 | 0.21579610096882293 | nan | nan | 0.26645176151057304 | 0.23178536502293726 | 0.2973057873619096 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | DPP | TPP | True | TPP | 0.21884672957057033 | nan | 0.19174336386994595 | 0.25184050786594486 | nan | nan | 0.3105121511943889 | 0.28087671304514095 | 0.3394191036609859 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | TPP | True | DPP | 0.28756678879403974 | nan | 0.2321659444022987 | 0.3362680311917086 | nan | nan | 0.15040126522725328 | 0.09411206947330793 | 0.1993368042916485 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | TPP | True | TPP | -0.07632472546409261 | nan | -0.09876258407492851 | -0.053792463231776166 | nan | nan | -0.019500499787284582 | -0.04535014186782592 | 0.00830914901265457 |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | TPP | True | TPP | 0.11530260594454125 | nan | 0.08922142680106034 | 0.14069897955952973 | nan | nan | 0.19810502307359487 | 0.17597194538974031 | 0.2204660213380601 |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | TPP | True | DPP | 0.16223799585085974 | nan | 0.1363997119552298 | 0.1888925259141272 | nan | nan | 0.1078920736817679 | 0.08001168621369983 | 0.1341797249651552 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | KMT | TPP | True | TPP | nan | 0.24345645414878586 | nan | nan | 0.20467640079054025 | 0.27969931910699486 | 0.31992283968768387 | 0.29154494147948434 | 0.35134111515657024 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | KMT | TPP | True | TPP | nan | -0.047082622944478875 | nan | nan | -0.07735588089966818 | -0.017368111711706765 | 0.013045454192919236 | -0.010617264922024779 | 0.0404352966536982 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | KMT | TPP | True | TPP | nan | -0.060166489963657796 | nan | nan | -0.10328989420740907 | -0.014954241285816974 | 0.028424667627277925 | -0.009336161017355649 | 0.06052736105144056 |
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
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | KMT | TPP | 2 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | DPP | KMT | 1 |
| L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | KMT | 1 |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | KMT | TPP | 1 |
| L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | KMT | 1 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | DPP | TPP | 1 |
| L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | TPP | 1 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | KMT | TPP | 1 |
| L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | KMT | TPP | 1 |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | KMT | TPP | 1 |
| L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | DPP | KMT | 0 |
| L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | DPP | TPP | 0 |
| L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | KMT | TPP | 0 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | KMT | TPP | 0 |
| L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | KMT | TPP | 0 |
## Permutation test

_H₀: camp labels independent of co-articulation structure (doc-level shuffle). observed_gamma uses point NPMI (same estimator as null); observed_gamma_filtered is p2_06 FDR+bootstrap reference._

| test | camp_a | camp_b | observed_gamma | observed_gamma_filtered | null_mean | null_q05 | null_q50 | null_q95 | p_value | n_permutations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gamma_dpp_kmt | DPP | KMT | 0.8816303660630427 | 0.8771324746118299 | 0.938383937048258 | 0.8972625997813548 | 0.9429830914933753 | 0.9684277768438919 | 0.023976023976023976 | 1000 |
| gamma_dpp_tpp | DPP | TPP | 0.8956610632456885 | 0.8877425961143403 | 0.9490998215198752 | 0.9171677318653351 | 0.9515054359530031 | 0.9733696194220584 | 0.015984015984015984 | 1000 |
| gamma_kmt_tpp | KMT | TPP | 0.9449991085767732 | 0.949434135985186 | 0.9405742387547191 | 0.9032135232238347 | 0.9440255381866092 | 0.9696569818758948 | 0.5204795204795205 | 1000 |
| joint_mean_gamma | nan | nan | 0.9074301792951681 | 0.904769735570452 | 0.9426859991076175 | 0.9156266459823477 | 0.9444672427081813 | 0.9628070367428786 | 0.02197802197802198 | 1000 |
## Sub-network analysis

_Exploratory node classification only. Identical Spearman ρ across shared and differentiating 4-node subsets reflects FDR-filtered edge sparsity (identical weight multisets on 6 edges per subnet), not a copy-paste error. Do not treat as an independent quantitative finding._

### Paper outline note

Primary quantitative evidence: camp permutation test + CI-disjoint edges. Sub-network section should be brief (node selection audit table); consider a dedicated subsection on L2-07 floating-signifier signature (shared {L2-06,L2-07} infrastructure with differential cross-cluster articulation).

### Node selection audit

| dpp_community_id | l2_set | best_kmt_jaccard | best_tpp_jaccard | category |
| --- | --- | --- | --- | --- |
| 0 | L2-01|L2-03 | 0.6666666666666666 | 1.0 | shared |
| 1 | L2-02|L2-05 | 0.25 | 0.6666666666666666 | differentiating |
| 3 | L2-04|L2-08 | 0.6666666666666666 | 0.5 | differentiating |
| 2 | L2-06|L2-07 | 1.0 | 1.0 | shared |
### Rank correlation comparison (full vs sub-networks)

| scope | camp_a | camp_b | metric | value |
| --- | --- | --- | --- | --- |
| full | DPP | KMT | hubert_gamma | 0.8771324746118299 |
| differentiating | DPP | KMT | spearman_rho | 0.8857142857142858 |
| shared | DPP | KMT | spearman_rho | 0.8857142857142858 |
| full | DPP | TPP | hubert_gamma | 0.8877425961143403 |
| differentiating | DPP | TPP | spearman_rho | 0.942857142857143 |
| shared | DPP | TPP | spearman_rho | 0.942857142857143 |
| full | KMT | TPP | hubert_gamma | 0.949434135985186 |
| differentiating | KMT | TPP | spearman_rho | 0.942857142857143 |
| shared | KMT | TPP | spearman_rho | 0.942857142857143 |
### Shared infrastructure sub-network edge stats

| camp | l2_a | l2_b | npmi_weight | rank |
| --- | --- | --- | --- | --- |
| DPP | L2-01 | L2-03 | 0.1874041756443085 | 2.0 |
| KMT | L2-01 | L2-03 | 0.27323844539997943 | 1.0 |
| TPP | L2-01 | L2-03 | 0.26645176151057304 | 2.0 |
| DPP | L2-01 | L2-06 | 0.2781851000838341 | 1.0 |
| KMT | L2-01 | L2-06 | 0.24345645414878586 | 2.0 |
| TPP | L2-01 | L2-06 | 0.31992283968768387 | 1.0 |
| DPP | L2-01 | L2-07 | 0.0891540865144757 | 4.0 |
| KMT | L2-01 | L2-07 | 0.10078924253621294 | 5.0 |
| TPP | L2-01 | L2-07 | 0.10769293169155078 | 5.0 |
| DPP | L2-03 | L2-06 | 0.06610267871991624 | 5.0 |
| KMT | L2-03 | L2-06 | 0.1037311642341012 | 4.0 |
| TPP | L2-03 | L2-06 | 0.13685750815356074 | 4.0 |
| DPP | L2-03 | L2-07 | 0.1098279809668919 | 3.0 |
| KMT | L2-03 | L2-07 | 0.16737892880790517 | 3.0 |
| TPP | L2-03 | L2-07 | 0.16234347483231606 | 3.0 |
| DPP | L2-06 | L2-07 | 0.0 | 6.0 |
| KMT | L2-06 | L2-07 | 0.0 | 6.0 |
| TPP | L2-06 | L2-07 | 0.0 | 6.0 |
## Differentiating sub-network (figure-ready)

| camp | l2_a | l2_b | npmi_median | rank | npmi_lower | npmi_upper | ci_disjoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DPP | L2-02 | L2-04 | 0.1874041756443085 | 2.0 | 0.0211807283469406 | 0.16993569112785326 | False |
| KMT | L2-02 | L2-04 | 0.27323844539997943 | 1.0 | 0.09346750990939871 | 0.2295734861217685 | False |
| TPP | L2-02 | L2-04 | 0.26645176151057304 | 2.0 | 0.09561077927250247 | 0.22358839577870032 | False |
| DPP | L2-02 | L2-05 | 0.2781851000838341 | 1.0 | 0.20042518747952828 | 0.26515396044955725 | False |
| KMT | L2-02 | L2-05 | 0.24345645414878586 | 2.0 | 0.2414688702895535 | 0.31433249210405556 | False |
| TPP | L2-02 | L2-05 | 0.31992283968768387 | 1.0 | 0.2536838216439681 | 0.31689461364088645 | False |
| DPP | L2-02 | L2-08 | 0.0891540865144757 | 4.0 | 0.16999034753640102 | 0.2505986137720306 | False |
| KMT | L2-02 | L2-08 | 0.10078924253621294 | 5.0 | 0.2353562175409167 | 0.33189022508474286 | False |
| TPP | L2-02 | L2-08 | 0.10769293169155078 | 5.0 | 0.1670786966614453 | 0.2618373145963662 | False |
| DPP | L2-04 | L2-05 | 0.06610267871991624 | 5.0 | 0.040221778768294296 | 0.11663207205907042 | False |
| KMT | L2-04 | L2-05 | 0.1037311642341012 | 4.0 | 0.08583289528380839 | 0.16705303051723921 | False |
| TPP | L2-04 | L2-05 | 0.13685750815356074 | 4.0 | 0.05815576358545669 | 0.13552561045662767 | False |
| DPP | L2-04 | L2-08 | 0.1098279809668919 | 3.0 | 0.2321659444022987 | 0.3362680311917086 | True |
| KMT | L2-04 | L2-08 | 0.16737892880790517 | 3.0 | 0.14070038435556792 | 0.25821817614865666 | False |
| TPP | L2-04 | L2-08 | 0.16234347483231606 | 3.0 | 0.09411206947330793 | 0.1993368042916485 | True |
| DPP | L2-05 | L2-08 | 0.0 | 6.0 | 0.15516377690117045 | 0.22660904151390965 | False |
| KMT | L2-05 | L2-08 | 0.0 | 6.0 | 0.16845309528397603 | 0.23626349823708878 | False |
| TPP | L2-05 | L2-08 | 0.0 | 6.0 | 0.17070999444062854 | 0.23775457674055575 | False |
## Cross-camp community matching (top-k greedy)

| dpp_community_id | dpp_l2_set | kmt_community_id | kmt_jaccard | tpp_community_id | tpp_jaccard | category | dpp_is_stable | kmt_is_stable | tpp_is_stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | L2-01|L2-03 | 0 | 0.6666666666666666 | 1 | 1.0 | shared | True | False | True |
| 1 | L2-02|L2-05 | 0 | 0.25 | 0 | 0.6666666666666666 | partial | True | False | False |
| 3 | L2-04|L2-08 | 1 | 0.6666666666666666 | 0 | 0.25 | partial | True | False | False |
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
| DPP | 0 | L2-01|L2-03 | TPP | 0 | L2-02|L2-04|L2-05 | 0.0 |

_(91 more rows)_

## QAP (Hubert γ)

| camp_a | camp_b | hubert_gamma | p_value | n_perm | n_nodes |
| --- | --- | --- | --- | --- | --- |
| DPP | KMT | 0.8771324746118299 | 0.000999000999000999 | 1000 | 8 |
| DPP | TPP | 0.8877425961143403 | 0.000999000999000999 | 1000 | 8 |
| KMT | TPP | 0.949434135985186 | 0.000999000999000999 | 1000 | 8 |

_QAP γ measures against-random distinctness; between-camp distinctness is tested by the camp permutation test above._

### Scheme: `camp`

#### Stratum `DPP`

- Community 0 (stability=1.000, stable=True): L2-01, L2-03
- Community 1 (stability=0.857, stable=True): L2-02, L2-05
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
- Community 3 (stability=1.000, stable=True): L2-04, L2-08

#### Stratum `KMT`

- Community 0 (stability=0.610, stable=False): L2-01, L2-03, L2-05
- Community 1 (stability=0.752, stable=False): L2-02, L2-04, L2-08
- Community 2 (stability=1.000, stable=True): L2-06, L2-07

#### Stratum `TPP`

- Community 0 (stability=0.667, stable=False): L2-02, L2-04, L2-05
- Community 1 (stability=0.857, stable=True): L2-01, L2-03
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
- Community 3 (stability=1.000, stable=True): L2-08
### Graph diagnostics (`camp`)

| scheme | stratum | n_nodes | n_edges | max_edges | density | n_windows | empty_l2_rate | low_n_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camp | DPP | 8 | 22 | 28 | 0.7857 | 5464 | 0.1336 | False |
| camp | KMT | 8 | 20 | 28 | 0.7143 | 4476 | 0.1642 | False |
| camp | TPP | 8 | 19 | 28 | 0.6786 | 6611 | 0.1942 | False |
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
| camp_genre | DPP_news | 8 | 23 | 28 | 0.8214 | 4898 | 0.1384 | False |
| camp_genre | KMT_debate | 4 | 2 | 6 | 0.3333 | 20 | 0.25 | True |
| camp_genre | KMT_news | 8 | 20 | 28 | 0.7143 | 4456 | 0.1638 | False |
| camp_genre | TPP_debate | 5 | 5 | 10 | 0.5 | 267 | 0.1835 | True |
| camp_genre | TPP_news | 8 | 19 | 28 | 0.6786 | 6344 | 0.1947 | False |
**Low-n strata (excluded from networks when n_windows < min_stratum_windows):**
- `KMT_debate`: n_windows=20
- `TPP_debate`: n_windows=267
### Scheme: `camp_time`

#### Stratum `DPP_campaign`

- Community 0 (stability=0.619, stable=False): L2-03, L2-05, L2-08
- Community 1 (stability=0.857, stable=True): L2-01, L2-02
- Community 2 (stability=1.000, stable=True): L2-06, L2-07

#### Stratum `DPP_nodate`

- Community 0 (stability=0.714, stable=False): L2-01, L2-02, L2-05
- Community 1 (stability=0.714, stable=False): L2-03, L2-07, L2-08
- Community 2 (stability=1.000, stable=True): L2-06

#### Stratum `DPP_pre_registration`

- Community 0 (stability=0.690, stable=False): L2-04, L2-06, L2-07, L2-08
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

- Community 0 (stability=0.571, stable=False): L2-01, L2-02, L2-03, L2-05
- Community 1 (stability=1.000, stable=True): L2-06, L2-07
- Community 2 (stability=0.857, stable=True): L2-04, L2-08

#### Stratum `TPP_post_registration`

- Community 0 (stability=0.600, stable=False): L2-01, L2-02, L2-03, L2-05, L2-08
- Community 1 (stability=1.000, stable=True): L2-06, L2-07

#### Stratum `TPP_pre_registration`

- Community 0 (stability=0.714, stable=False): L2-02, L2-04, L2-08
- Community 1 (stability=0.667, stable=False): L2-01, L2-03, L2-05
- Community 2 (stability=1.000, stable=True): L2-06, L2-07
### Graph diagnostics (`camp_time`)

| scheme | stratum | n_nodes | n_edges | max_edges | density | n_windows | empty_l2_rate | low_n_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camp_time | DPP_campaign | 7 | 16 | 21 | 0.7619 | 610 | 0.1344 | False |
| camp_time | DPP_nodate | 7 | 10 | 21 | 0.4762 | 566 | 0.0919 | False |
| camp_time | DPP_pre_registration | 8 | 22 | 28 | 0.7857 | 3830 | 0.1316 | False |
| camp_time | KMT_campaign | 8 | 10 | 28 | 0.3571 | 1464 | 0.1154 | False |
| camp_time | KMT_post_registration | 7 | 14 | 21 | 0.6667 | 622 | 0.1961 | False |
| camp_time | KMT_pre_registration | 8 | 19 | 28 | 0.6786 | 2323 | 0.1804 | False |
| camp_time | TPP_campaign | 8 | 14 | 28 | 0.5 | 1322 | 0.1808 | False |
| camp_time | TPP_post_registration | 7 | 15 | 21 | 0.7143 | 747 | 0.2182 | False |
| camp_time | TPP_pre_registration | 8 | 17 | 28 | 0.6071 | 3971 | 0.1909 | False |
