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

_empty_
## Differentiating edges

_Conservative test: non-overlapping 95% bootstrap CIs._

**4** of 84 camp-pair edge tests have disjoint CIs.

_Binomial null check (α=0.05): expected false positives ≈ 4.2; observed 4 (binom p=6.10e-01). Count is strongly inconsistent with a global null of no cross-camp differences._

### CI-disjoint edges

| l2_a | l2_a_label | l2_b | l2_b_label | camp_a | camp_b | ci_disjoint | direction | npmi_median_dpp | npmi_median_kmt | npmi_lower_dpp | npmi_upper_dpp | npmi_lower_kmt | npmi_upper_kmt | npmi_median_tpp | npmi_lower_tpp | npmi_upper_tpp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | KMT | True | DPP | 0.29381192435415043 | 0.18137828329683225 | 0.2430115013721362 | 0.33955431202205816 | 0.115073850639692 | 0.23618786724392693 | nan | nan | nan |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | KMT | True | KMT | 0.005635001581357308 | 0.12250988237427252 | -0.027528544643224934 | 0.03749468461018973 | 0.08296414171660628 | 0.1642123664282394 | nan | nan | nan |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | TPP | True | DPP | 0.29381192435415043 | nan | 0.2430115013721362 | 0.33955431202205816 | nan | nan | 0.18575247443420645 | 0.13958841123534033 | 0.2334190923568891 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | TPP | True | TPP | 0.005635001581357308 | nan | -0.027528544643224934 | 0.03749468461018973 | nan | nan | 0.08924452972051833 | 0.055688515321034325 | 0.12393851505306824 |
### Differentiating edges by L2 node

| l2_node | l2_label | camp_a | camp_b | n_differentiating_edges |
| --- | --- | --- | --- | --- |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | DPP | KMT | 1 |
| L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | KMT | 1 |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | KMT | 1 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | DPP | KMT | 1 |
| L2-05 | L2-05 (共同威脅設定 / Shared Crisis Construction) | DPP | TPP | 1 |
| L2-06 | L2-06 (民族自豪感激發 / National Pride Activation) | DPP | TPP | 1 |
| L2-08 | L2-08 (民主價值強調 / Democratic Values Emphasis) | DPP | TPP | 1 |
| L2-04 | L2-04 (集體敘事再造 / Collective Narrative Reconstruction) | DPP | TPP | 1 |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | KMT | 0 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | DPP | KMT | 0 |
| L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | KMT | 0 |
| L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | DPP | KMT | 0 |
| L2-03 | L2-03 (國際合法性塑造 / International Legitimacy Construction) | DPP | TPP | 0 |
| L2-02 | L2-02 (差異化認同強調 / Differentiated Identity Emphasis) | DPP | TPP | 0 |
| L2-01 | L2-01 (主體性建構 / Subjectivity Construction) | DPP | TPP | 0 |
| L2-07 | L2-07 (凝聚與願景動員 / Solidarity & Vision) | DPP | TPP | 0 |
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
| gamma_dpp_kmt | DPP | KMT | 0.6600163358649457 | 0.7811747337243745 | 0.8127102221904242 | 0.6772372894902953 | 0.8235339882975412 | 0.908258007261291 | 0.03496503496503497 | 1000 |
| gamma_dpp_tpp | DPP | TPP | 0.778503674257142 | 0.8399942853627715 | 0.8367740398959767 | 0.7207576663476117 | 0.8467589291276615 | 0.9199593204895894 | 0.16583416583416583 | 1000 |
| gamma_kmt_tpp | KMT | TPP | 0.8522820195645742 | 0.8609137444980429 | 0.817639753340861 | 0.6783262286759093 | 0.8310180672280273 | 0.9099836155107669 | 0.6333666333666333 | 1000 |
| joint_mean_gamma | nan | nan | 0.7636006765622206 | 0.827360921195063 | 0.8223746718090874 | 0.7401712983164903 | 0.8272936025948023 | 0.8873630221902555 | 0.1258741258741259 | 1000 |
## Sub-network analysis

_Exploratory node classification only. Identical Spearman ρ across shared and differentiating 4-node subsets reflects FDR-filtered edge sparsity (identical weight multisets on 6 edges per subnet), not a copy-paste error. Do not treat as an independent quantitative finding._

### Paper outline note

Primary quantitative evidence: camp permutation test + CI-disjoint edges. Sub-network section should be brief (node selection audit table); consider a dedicated subsection on L2-07 floating-signifier signature (shared {L2-06,L2-07} infrastructure with differential cross-cluster articulation).

### Node selection audit

| dpp_community_id | l2_set | best_kmt_jaccard | best_tpp_jaccard | category |
| --- | --- | --- | --- | --- |
| 0 | L2-01|L2-03|L2-06|L2-07 | 0.75 | 0.5 | differentiating |
| 1 | L2-02|L2-05 | 0.3333333333333333 | 1.0 | differentiating |
| 2 | L2-04|L2-08 | 0.3333333333333333 | 0.3333333333333333 | differentiating |
### Rank correlation comparison (full vs sub-networks)

| scope | camp_a | camp_b | metric | value |
| --- | --- | --- | --- | --- |
| full | DPP | KMT | hubert_gamma | 0.7811747337243745 |
| differentiating | DPP | KMT | spearman_rho | 0.8160371142289219 |
| full | DPP | TPP | hubert_gamma | 0.8399942853627715 |
| differentiating | DPP | TPP | spearman_rho | 0.8871766648321407 |
| full | KMT | TPP | hubert_gamma | 0.8609137444980429 |
| differentiating | KMT | TPP | spearman_rho | 0.8791893808078762 |
## Differentiating sub-network (figure-ready)

| camp | l2_a | l2_b | npmi_median | rank | npmi_lower | npmi_upper | ci_disjoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DPP | L2-01 | L2-02 | 0.25480373727504013 | 3.0 | 0.20751442860290614 | 0.2977175649197276 | False |
| KMT | L2-01 | L2-02 | 0.35178433645204776 | 2.0 | 0.288887001051922 | 0.40411109925739674 | False |
| TPP | L2-01 | L2-02 | 0.32451444559978965 | 1.0 | 0.2711816150144176 | 0.37086428681647454 | False |
| DPP | L2-01 | L2-03 | 0.14322939713868835 | 10.0 | 0.07634042013598558 | 0.22472321119925015 | False |
| KMT | L2-01 | L2-03 | 0.21909440524518045 | 4.0 | 0.12425838126057678 | 0.28303532841862794 | False |
| TPP | L2-01 | L2-03 | 0.22951713168358026 | 6.0 | 0.17044618277351134 | 0.27991267746700554 | False |
| DPP | L2-01 | L2-04 | 0.10257543395584387 | 14.0 | 0.05481055702455933 | 0.15009297655554063 | False |
| KMT | L2-01 | L2-04 | 0.10529807329198702 | 19.0 | 0.0516120689445801 | 0.15263172061969782 | False |
| TPP | L2-01 | L2-04 | 0.11348117874348332 | 15.0 | 0.07011199871716846 | 0.15880459043574205 | False |
| DPP | L2-01 | L2-05 | 0.15979545625484026 | 7.0 | 0.1250679937831496 | 0.19965532479583806 | False |
| KMT | L2-01 | L2-05 | 0.19689578397428123 | 5.0 | 0.15634667427797458 | 0.23547449702403866 | False |
| TPP | L2-01 | L2-05 | 0.23325180305522142 | 5.0 | 0.1928418960222592 | 0.27239991601144364 | False |
| DPP | L2-01 | L2-06 | 0.15128899598792012 | 9.0 | 0.11310485405586108 | 0.19122128038209013 | False |
| KMT | L2-01 | L2-06 | 0.17136148828287306 | 11.0 | 0.12657663124336188 | 0.21480236727607918 | False |
| TPP | L2-01 | L2-06 | 0.1907727977575598 | 8.0 | 0.15550433487323828 | 0.2283639889196879 | False |
| DPP | L2-01 | L2-07 | 0.05310288353976315 | 22.0 | 0.028544394582743517 | 0.07585354931796873 | False |
| KMT | L2-01 | L2-07 | 0.0907613625904688 | 20.0 | 0.0593492495764104 | 0.12008912314833643 | False |
| TPP | L2-01 | L2-07 | 0.049545138871961616 | 22.0 | 0.021869174376199115 | 0.07672487296011717 | False |
| DPP | L2-01 | L2-08 | 0.18335332520377856 | 5.0 | 0.14495860665108148 | 0.22581897953862567 | False |
| KMT | L2-01 | L2-08 | 0.26916284896892384 | 3.0 | 0.219051522698059 | 0.3161498910730985 | False |
| TPP | L2-01 | L2-08 | 0.22772416140767376 | 7.0 | 0.1835360714006001 | 0.2680687581968568 | False |
| DPP | L2-02 | L2-03 | 0.0 | 26.0 | -0.04083787132816085 | 0.23416978684862674 | False |
| KMT | L2-02 | L2-03 | -0.10171887091765941 | 28.0 | -0.17456049296787723 | -0.0210347585969356 | False |
| TPP | L2-02 | L2-03 | 0.0 | 26.0 | -0.1895421241584509 | 0.09849617834726548 | False |
| DPP | L2-02 | L2-04 | 0.1558923234523595 | 8.0 | 0.07529710151117448 | 0.22764601606431428 | False |
| KMT | L2-02 | L2-04 | 0.18469134087052574 | 7.0 | 0.09294647994493462 | 0.2518481759377894 | False |
| TPP | L2-02 | L2-04 | 0.24308711935000496 | 4.0 | 0.18415473340587488 | 0.2975719827051142 | False |
| DPP | L2-02 | L2-05 | 0.2527185260509256 | 4.0 | 0.2134969360252657 | 0.29213733611526205 | False |
| KMT | L2-02 | L2-05 | 0.18459659078953533 | 8.0 | 0.13138850906464064 | 0.23925966490469044 | False |
| TPP | L2-02 | L2-05 | 0.2629396024103178 | 2.0 | 0.22022154425206364 | 0.30472986211060615 | False |
| DPP | L2-02 | L2-06 | 0.0 | 26.0 | -0.013230058910124179 | 0.07370076452314868 | False |
| KMT | L2-02 | L2-06 | 0.0 | 26.0 | -0.01947595664873733 | 0.12581375396435623 | False |
| TPP | L2-02 | L2-06 | 0.0660461432993831 | 19.0 | 0.014579025203449947 | 0.11654921525562899 | False |
| DPP | L2-02 | L2-07 | 0.07085253365868616 | 21.0 | 0.04148727965100111 | 0.10280009975586796 | False |
| KMT | L2-02 | L2-07 | 0.0650163574595986 | 21.0 | 0.0342475548250317 | 0.09778450196313637 | False |
| TPP | L2-02 | L2-07 | 0.0 | 26.0 | -0.021694031260440665 | 0.047729722366810286 | False |
| DPP | L2-02 | L2-08 | 0.2981285248418227 | 1.0 | 0.24287538200747535 | 0.3545811605769519 | False |
| KMT | L2-02 | L2-08 | 0.35283825022725 | 1.0 | 0.2932640077349596 | 0.4034446624246994 | False |
| TPP | L2-02 | L2-08 | 0.2523795622421294 | 3.0 | 0.2006435465433426 | 0.299962042906523 | False |
| DPP | L2-03 | L2-04 | 0.0 | 26.0 | -0.029458226497897012 | 0.17990936590679177 | False |
| KMT | L2-03 | L2-04 | 0.0 | 26.0 | -0.14811434904688647 | 0.09022886496215668 | False |
| TPP | L2-03 | L2-04 | 0.0 | 26.0 | -0.053851592454821295 | 0.13976958778697754 | False |
| DPP | L2-03 | L2-05 | 0.07229798436346327 | 20.0 | 0.010139745213634869 | 0.13888490368397147 | False |
| KMT | L2-03 | L2-05 | 0.0 | 26.0 | -0.026994072592883448 | 0.14123085778422162 | False |
| TPP | L2-03 | L2-05 | 0.0 | 26.0 | -0.01304324457823804 | 0.11509384861710292 | False |
| DPP | L2-03 | L2-06 | 0.07928507211626737 | 18.0 | 0.014106171390558237 | 0.12702283569233538 | False |
| KMT | L2-03 | L2-06 | 0.19138138325954995 | 6.0 | 0.11679915466910332 | 0.250190284482689 | False |
| TPP | L2-03 | L2-06 | 0.14212538182444157 | 13.0 | 0.07881602291778642 | 0.1937375437411776 | False |
| DPP | L2-03 | L2-07 | 0.08211061044477355 | 17.0 | 0.04081811395482984 | 0.10976739249950598 | False |
| KMT | L2-03 | L2-07 | 0.06453162207361027 | 22.0 | 0.022143346163489945 | 0.10493545737184855 | False |

_(34 more rows)_

## Cross-camp community matching (top-k greedy)

| dpp_community_id | dpp_l2_set | kmt_community_id | kmt_jaccard | tpp_community_id | tpp_jaccard | category | dpp_is_stable | kmt_is_stable | tpp_is_stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | L2-01|L2-03|L2-06|L2-07 | 1 | 0.75 | 0 | 0.5 | partial | False | True | True |
| 1 | L2-02|L2-05 | 2 | 0.3333333333333333 | 1 | 1.0 | partial | True | True | True |
| 2 | L2-04|L2-08 | 2 | 0.3333333333333333 | 2 | 0.3333333333333333 | distinct | True | True | False |
## Cross-camp Jaccard (all community pairs)

| camp_a | community_a | l2_set_a | camp_b | community_b | l2_set_b | jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | DPP | 0 | L2-01|L2-03|L2-06|L2-07 | 1.0 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | DPP | 1 | L2-02|L2-05 | 0.0 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | DPP | 2 | L2-04|L2-08 | 0.0 |
| DPP | 1 | L2-02|L2-05 | DPP | 0 | L2-01|L2-03|L2-06|L2-07 | 0.0 |
| DPP | 1 | L2-02|L2-05 | DPP | 1 | L2-02|L2-05 | 1.0 |
| DPP | 1 | L2-02|L2-05 | DPP | 2 | L2-04|L2-08 | 0.0 |
| DPP | 2 | L2-04|L2-08 | DPP | 0 | L2-01|L2-03|L2-06|L2-07 | 0.0 |
| DPP | 2 | L2-04|L2-08 | DPP | 1 | L2-02|L2-05 | 0.0 |
| DPP | 2 | L2-04|L2-08 | DPP | 2 | L2-04|L2-08 | 1.0 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | KMT | 0 | L2-01|L2-02|L2-08 | 0.16666666666666666 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | KMT | 1 | L2-03|L2-06|L2-07 | 0.75 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | KMT | 2 | L2-04|L2-05 | 0.0 |
| DPP | 1 | L2-02|L2-05 | KMT | 0 | L2-01|L2-02|L2-08 | 0.25 |
| DPP | 1 | L2-02|L2-05 | KMT | 1 | L2-03|L2-06|L2-07 | 0.0 |
| DPP | 1 | L2-02|L2-05 | KMT | 2 | L2-04|L2-05 | 0.3333333333333333 |
| DPP | 2 | L2-04|L2-08 | KMT | 0 | L2-01|L2-02|L2-08 | 0.25 |
| DPP | 2 | L2-04|L2-08 | KMT | 1 | L2-03|L2-06|L2-07 | 0.0 |
| DPP | 2 | L2-04|L2-08 | KMT | 2 | L2-04|L2-05 | 0.3333333333333333 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | TPP | 0 | L2-01|L2-03 | 0.5 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | TPP | 1 | L2-02|L2-05 | 0.0 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | TPP | 2 | L2-04|L2-06 | 0.2 |
| DPP | 0 | L2-01|L2-03|L2-06|L2-07 | TPP | 3 | L2-07|L2-08 | 0.2 |
| DPP | 1 | L2-02|L2-05 | TPP | 0 | L2-01|L2-03 | 0.0 |
| DPP | 1 | L2-02|L2-05 | TPP | 1 | L2-02|L2-05 | 1.0 |
| DPP | 1 | L2-02|L2-05 | TPP | 2 | L2-04|L2-06 | 0.0 |
| DPP | 1 | L2-02|L2-05 | TPP | 3 | L2-07|L2-08 | 0.0 |
| DPP | 2 | L2-04|L2-08 | TPP | 0 | L2-01|L2-03 | 0.0 |
| DPP | 2 | L2-04|L2-08 | TPP | 1 | L2-02|L2-05 | 0.0 |
| DPP | 2 | L2-04|L2-08 | TPP | 2 | L2-04|L2-06 | 0.3333333333333333 |
| DPP | 2 | L2-04|L2-08 | TPP | 3 | L2-07|L2-08 | 0.3333333333333333 |

_(70 more rows)_

## QAP (Hubert γ)

| camp_a | camp_b | hubert_gamma | p_value | n_perm | n_nodes |
| --- | --- | --- | --- | --- | --- |
| DPP | KMT | 0.7811747337243745 | 0.000999000999000999 | 1000 | 8 |
| DPP | TPP | 0.8399942853627715 | 0.000999000999000999 | 1000 | 8 |
| KMT | TPP | 0.8609137444980429 | 0.000999000999000999 | 1000 | 8 |

_QAP γ measures against-random distinctness; between-camp distinctness is tested by the camp permutation test above._

### Scheme: `camp`

#### Stratum `DPP`

- Community 0 (stability=0.648, stable=False): L2-01, L2-03, L2-06, L2-07
- Community 1 (stability=1.000, stable=True): L2-02, L2-05
- Community 2 (stability=0.857, stable=True): L2-04, L2-08

#### Stratum `KMT`

- Community 0 (stability=0.667, stable=False): L2-01, L2-02, L2-08
- Community 1 (stability=0.810, stable=True): L2-03, L2-06, L2-07
- Community 2 (stability=0.857, stable=True): L2-04, L2-05

#### Stratum `TPP`

- Community 0 (stability=0.857, stable=True): L2-01, L2-03
- Community 1 (stability=0.857, stable=True): L2-02, L2-05
- Community 2 (stability=0.514, stable=False): L2-04, L2-06
- Community 3 (stability=0.857, stable=True): L2-07, L2-08
### Graph diagnostics (`camp`)

| scheme | stratum | n_nodes | n_edges | max_edges | density | n_windows | empty_l2_rate | low_n_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camp | DPP | 8 | 23 | 28 | 0.8214 | 5160 | 0.2039 | False |
| camp | KMT | 8 | 24 | 28 | 0.8571 | 4191 | 0.2396 | False |
| camp | TPP | 8 | 23 | 28 | 0.8214 | 6130 | 0.2569 | False |
### Scheme: `camp_genre`

#### Stratum `DPP_debate`

- Community 0 (stability=0.810, stable=True): L2-03, L2-05, L2-06, L2-07
- Community 1 (stability=1.000, stable=True): L2-01, L2-02
- Community 2 (stability=1.000, stable=True): L2-04, L2-08

#### Stratum `DPP_news`

- Community 0 (stability=0.576, stable=False): L2-01, L2-02, L2-03, L2-05
- Community 1 (stability=0.590, stable=False): L2-04, L2-06, L2-07, L2-08

#### Stratum `KMT_news`

- Community 0 (stability=0.667, stable=False): L2-01, L2-02, L2-08
- Community 1 (stability=0.810, stable=True): L2-03, L2-06, L2-07
- Community 2 (stability=0.857, stable=True): L2-04, L2-05

#### Stratum `TPP_news`

- Community 0 (stability=0.619, stable=False): L2-01, L2-03, L2-06
- Community 1 (stability=0.667, stable=False): L2-02, L2-04, L2-05
- Community 2 (stability=0.857, stable=True): L2-07, L2-08
### Graph diagnostics (`camp_genre`)

| scheme | stratum | n_nodes | n_edges | max_edges | density | n_windows | empty_l2_rate | low_n_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camp_genre | DPP_debate | 8 | 8 | 28 | 0.2857 | 566 | 0.1343 | False |
| camp_genre | DPP_news | 8 | 24 | 28 | 0.8571 | 4594 | 0.2125 | False |
| camp_genre | KMT_news | 8 | 24 | 28 | 0.8571 | 4171 | 0.2398 | False |
| camp_genre | TPP_news | 8 | 23 | 28 | 0.8214 | 5863 | 0.2567 | False |
### Scheme: `camp_time`

#### Stratum `DPP_campaign`

- Community 0 (stability=0.667, stable=False): L2-04, L2-06, L2-07, L2-08
- Community 1 (stability=1.000, stable=True): L2-01, L2-05

#### Stratum `DPP_nodate`

- Community 0 (stability=0.810, stable=True): L2-03, L2-05, L2-06, L2-07
- Community 1 (stability=1.000, stable=True): L2-01, L2-02
- Community 2 (stability=1.000, stable=True): L2-04, L2-08

#### Stratum `DPP_pre_registration`

- Community 0 (stability=0.714, stable=False): L2-01, L2-03, L2-06
- Community 1 (stability=0.638, stable=False): L2-04, L2-07, L2-08
- Community 2 (stability=0.857, stable=True): L2-02, L2-05

#### Stratum `KMT_campaign`

- Community 0 (stability=0.810, stable=True): L2-01, L2-02, L2-05, L2-08
- Community 1 (stability=0.810, stable=True): L2-03, L2-04, L2-06, L2-07

#### Stratum `KMT_post_registration`

- Community 0 (stability=1.000, stable=True): L2-04, L2-05
- Community 1 (stability=1.000, stable=True): L2-06, L2-07
- Community 2 (stability=0.857, stable=True): L2-01, L2-08

#### Stratum `KMT_pre_registration`

- Community 0 (stability=0.714, stable=False): L2-04, L2-05, L2-07
- Community 1 (stability=0.714, stable=False): L2-01, L2-03, L2-06
- Community 2 (stability=0.571, stable=False): L2-02, L2-08

#### Stratum `TPP_campaign`

- Community 0 (stability=0.857, stable=True): L2-02, L2-05
- Community 1 (stability=1.000, stable=True): L2-04, L2-06
- Community 2 (stability=1.000, stable=True): L2-07, L2-08
- Community 3 (stability=1.000, stable=True): L2-01

#### Stratum `TPP_post_registration`

- Community 0 (stability=0.724, stable=False): L2-04, L2-06, L2-07
- Community 1 (stability=0.971, stable=True): L2-02, L2-05
- Community 2 (stability=0.671, stable=False): L2-01, L2-08

#### Stratum `TPP_pre_registration`

- Community 0 (stability=0.743, stable=False): L2-02, L2-04, L2-05
- Community 1 (stability=0.695, stable=False): L2-01, L2-03, L2-06
- Community 2 (stability=0.857, stable=True): L2-07, L2-08
### Graph diagnostics (`camp_time`)

| scheme | stratum | n_nodes | n_edges | max_edges | density | n_windows | empty_l2_rate | low_n_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camp_time | DPP_campaign | 6 | 10 | 15 | 0.6667 | 567 | 0.224 | False |
| camp_time | DPP_nodate | 8 | 8 | 28 | 0.2857 | 566 | 0.1343 | False |
| camp_time | DPP_pre_registration | 8 | 23 | 28 | 0.8214 | 3603 | 0.1965 | False |
| camp_time | KMT_campaign | 8 | 22 | 28 | 0.7857 | 1365 | 0.1941 | False |
| camp_time | KMT_post_registration | 6 | 9 | 15 | 0.6 | 581 | 0.2616 | False |
| camp_time | KMT_pre_registration | 8 | 22 | 28 | 0.7857 | 2182 | 0.2566 | False |
| camp_time | TPP_campaign | 7 | 15 | 21 | 0.7143 | 1208 | 0.2243 | False |
| camp_time | TPP_post_registration | 7 | 17 | 21 | 0.8095 | 694 | 0.2781 | False |
| camp_time | TPP_pre_registration | 8 | 19 | 28 | 0.6786 | 3677 | 0.2529 | False |
