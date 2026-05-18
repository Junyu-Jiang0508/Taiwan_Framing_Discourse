# Phase 2 pipeline log

## p2_01

_Updated: 2026-05-18T21:49:39.824675+00:00_

**Parameters**
- default_n: `3`
- step: `1`
- robustness_ns: `[1, 5, 'document']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_n3.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_n1.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_n5.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_ndocument.parquet`

**Stats**
- n_windows: 16551
- truncated_rate: 5.6130%
- empty_l2_rate: 16.6093%
- elapsed_sec: 28.97

**Notes**
- Sliding step=1; no cross-document windows.

## p2_02

_Updated: 2026-05-18T21:49:40.649714+00:00_

**Parameters**
- min_marginal_count: `30`
- schemes: `['camp_genre', 'camp', 'camp_time']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp_genre`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp_time`

**Stats**
- camp_genre: {'n_edges': 115, 'n_strata': 6, 'low_n_strata': [{'camp': 'KMT', 'genre': 'debate'}, {'camp': 'TPP', 'genre': 'debate'}], 'per_stratum_edges': {"('DPP', 'debate')": 21, "('DPP', 'news')": 28, "('KMT', 'news')": 28, "('TPP', 'debate')": 10, "('TPP', 'news')": 28}, 'empty_l2_rates': {"{'camp': 'DPP', 'genre': 'debate'}": '0.0919', "{'camp': 'DPP', 'genre': 'news'}": '0.1384', "{'camp': 'KMT', 'genre': 'debate'}": '0.2500', "{'camp': 'KMT', 'genre': 'news'}": '0.1638', "{'camp': 'TPP', 'genre': 'debate'}": '0.1835', "{'camp': 'TPP', 'genre': 'news'}": '0.1947'}}
- camp: {'n_edges': 84, 'n_strata': 3, 'low_n_strata': [], 'per_stratum_edges': {'DPP': 28, 'KMT': 28, 'TPP': 28}, 'empty_l2_rates': {"{'camp': 'DPP'}": '0.1336', "{'camp': 'KMT'}": '0.1642', "{'camp': 'TPP'}": '0.1942'}}
- camp_time: {'n_edges': 262, 'n_strata': 15, 'low_n_strata': [{'camp': 'DPP', 'time_bucket': 'election_plus'}, {'camp': 'DPP', 'time_bucket': 'post_registration'}, {'camp': 'KMT', 'time_bucket': 'election_plus'}, {'camp': 'KMT', 'time_bucket': 'nodate'}, {'camp': 'TPP', 'time_bucket': 'election_plus'}, {'camp': 'TPP', 'time_bucket': 'nodate'}], 'per_stratum_edges': {"('DPP', 'campaign')": 21, "('DPP', 'election_plus')": 3, "('DPP', 'nodate')": 21, "('DPP', 'post_registration')": 10, "('DPP', 'pre_registration')": 28, "('KMT', 'campaign')": 28, "('KMT', 'post_registration')": 21, "('KMT', 'pre_registration')": 28, "('TPP', 'campaign')": 28, "('TPP', 'election_plus')": 15, "('TPP', 'nodate')": 10, "('TPP', 'post_registration')": 21, "('TPP', 'pre_registration')": 28}, 'empty_l2_rates': {"{'camp': 'DPP', 'time_bucket': 'campaign'}": '0.1344', "{'camp': 'DPP', 'time_bucket': 'election_plus'}": '0.1161', "{'camp': 'DPP', 'time_bucket': 'nodate'}": '0.0919', "{'camp': 'DPP', 'time_bucket': 'post_registration'}": '0.2283', "{'camp': 'DPP', 'time_bucket': 'pre_registration'}": '0.1316', "{'camp': 'KMT', 'time_bucket': 'campaign'}": '0.1154', "{'camp': 'KMT', 'time_bucket': 'election_plus'}": '0.4255', "{'camp': 'KMT', 'time_bucket': 'nodate'}": '0.2500', "{'camp': 'KMT', 'time_bucket': 'post_registration'}": '0.1961', "{'camp': 'KMT', 'time_bucket': 'pre_registration'}": '0.1804', "{'camp': 'TPP', 'time_bucket': 'campaign'}": '0.1808', "{'camp': 'TPP', 'time_bucket': 'election_plus'}": '0.2467', "{'camp': 'TPP', 'time_bucket': 'nodate'}": '0.1835', "{'camp': 'TPP', 'time_bucket': 'post_registration'}": '0.2182', "{'camp': 'TPP', 'time_bucket': 'pre_registration'}": '0.1909'}}
- elapsed_sec: 0.82

**Notes**
- Empty-L2 windows are retained in the denominator to estimate P(A) as the unconditional probability of label presence across all articulatory sites in the stratum, consistent with the theoretical claim that absence of articulation is itself meaningful.
- Pair counts exclude is_empty_l2 windows; marginals include all windows.
- Step=1 window overlap inflates counts equally in P(A), P(B), P(AB) — ratio cancels in NPMI.

## p2_00

_Updated: 2026-05-18T21:52:25.258728+00:00_

**Parameters**
- corpus_csv: `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/01_results_datasets/Run_20260407_221407/final_results.csv`
- corpus_hash: `1b2ae99e572c3bb4`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/labeled_units.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/diagnostics/unknown_camp_audit.json`

**Stats**
- n_sentences_main: 22215
- stratum_counts: {('DPP', 'debate'): 388, ('DPP', 'news'): 6818, ('KMT', 'debate'): 196, ('KMT', 'news'): 6006, ('TPP', 'debate'): 382, ('TPP', 'news'): 8425}
- news_unknown_rate: 0.0000%
- n_multi_party_sentences: 3283
- elapsed_sec: 9.88

## p2_03

_Updated: 2026-05-18T21:56:17.886505+00:00_

**Parameters**
- n_resamples: `1000`
- ci_alpha: `0.05`
- n_jobs: `-1`
- fdr_alpha: `0.05`
- edge_selection: `fdr`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp_genre/npmi_bootstrap.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp/npmi_bootstrap.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp_time/npmi_bootstrap.parquet`

**Stats**
- camp_genre: 115
- camp: 84
- camp_time: 262
- elapsed_sec: 201.91

**Notes**
- Resample unit = document (cluster bootstrap). BH-FDR applied per stratum.

## p2_04

_Updated: 2026-05-18T21:56:21.722619+00:00_

**Parameters**
- schemes: `['camp_genre', 'camp', 'camp_time']`
- edge_selection: `fdr`

**Outputs**
- `camp_genre/DPP_debate`
- `camp_genre/DPP_news`
- `camp_genre/KMT_news`
- `camp_genre/TPP_news`
- `camp/DPP`
- `camp/KMT`
- `camp/TPP`
- `camp_time/DPP_campaign`
- `camp_time/DPP_nodate`
- `camp_time/DPP_pre_registration`
- `camp_time/KMT_campaign`
- `camp_time/KMT_post_registration`
- `camp_time/KMT_pre_registration`
- `camp_time/TPP_campaign`
- `camp_time/TPP_post_registration`
- `camp_time/TPP_pre_registration`

**Stats**
- n_graphs: 32
- skipped_low_n: ['TPP_debate', 'DPP_election_plus', 'DPP_post_registration', 'TPP_election_plus', 'TPP_nodate']
- graph_stats: [{'scheme': 'camp_genre', 'stratum': 'DPP_debate', 'n_sig_ci': 13, 'n_sig_fdr': 12, 'n_sig_selected': 12, 'n_nodes': 7, 'n_edges': 10, 'density': 0.4762}, {'scheme': 'camp_genre', 'stratum': 'DPP_news', 'n_sig_ci': 24, 'n_sig_fdr': 24, 'n_sig_selected': 24, 'n_nodes': 8, 'n_edges': 23, 'density': 0.8214}, {'scheme': 'camp_genre', 'stratum': 'KMT_news', 'n_sig_ci': 22, 'n_sig_fdr': 22, 'n_sig_selected': 22, 'n_nodes': 8, 'n_edges': 20, 'density': 0.7143}, {'scheme': 'camp_genre', 'stratum': 'TPP_news', 'n_sig_ci': 20, 'n_sig_fdr': 20, 'n_sig_selected': 20, 'n_nodes': 8, 'n_edges': 19, 'density': 0.6786}, {'scheme': 'camp', 'stratum': 'DPP', 'n_sig_ci': 23, 'n_sig_fdr': 23, 'n_sig_selected': 23, 'n_nodes': 8, 'n_edges': 22, 'density': 0.7857}, {'scheme': 'camp', 'stratum': 'KMT', 'n_sig_ci': 22, 'n_sig_fdr': 22, 'n_sig_selected': 22, 'n_nodes': 8, 'n_edges': 20, 'density': 0.7143}, {'scheme': 'camp', 'stratum': 'TPP', 'n_sig_ci': 20, 'n_sig_fdr': 20, 'n_sig_selected': 20, 'n_nodes': 8, 'n_edges': 19, 'density': 0.6786}, {'scheme': 'camp_time', 'stratum': 'DPP_campaign', 'n_sig_ci': 16, 'n_sig_fdr': 16, 'n_sig_selected': 16, 'n_nodes': 7, 'n_edges': 16, 'density': 0.7619}, {'scheme': 'camp_time', 'stratum': 'DPP_nodate', 'n_sig_ci': 13, 'n_sig_fdr': 12, 'n_sig_selected': 12, 'n_nodes': 7, 'n_edges': 10, 'density': 0.4762}, {'scheme': 'camp_time', 'stratum': 'DPP_pre_registration', 'n_sig_ci': 23, 'n_sig_fdr': 23, 'n_sig_selected': 23, 'n_nodes': 8, 'n_edges': 22, 'density': 0.7857}, {'scheme': 'camp_time', 'stratum': 'KMT_campaign', 'n_sig_ci': 14, 'n_sig_fdr': 12, 'n_sig_selected': 12, 'n_nodes': 8, 'n_edges': 10, 'density': 0.3571}, {'scheme': 'camp_time', 'stratum': 'KMT_post_registration', 'n_sig_ci': 16, 'n_sig_fdr': 14, 'n_sig_selected': 14, 'n_nodes': 7, 'n_edges': 14, 'density': 0.6667}, {'scheme': 'camp_time', 'stratum': 'KMT_pre_registration', 'n_sig_ci': 19, 'n_sig_fdr': 19, 'n_sig_selected': 19, 'n_nodes': 8, 'n_edges': 19, 'density': 0.6786}, {'scheme': 'camp_time', 'stratum': 'TPP_campaign', 'n_sig_ci': 14, 'n_sig_fdr': 14, 'n_sig_selected': 14, 'n_nodes': 8, 'n_edges': 14, 'density': 0.5}, {'scheme': 'camp_time', 'stratum': 'TPP_post_registration', 'n_sig_ci': 16, 'n_sig_fdr': 15, 'n_sig_selected': 15, 'n_nodes': 7, 'n_edges': 15, 'density': 0.7143}, {'scheme': 'camp_time', 'stratum': 'TPP_pre_registration', 'n_sig_ci': 18, 'n_sig_fdr': 18, 'n_sig_selected': 18, 'n_nodes': 8, 'n_edges': 17, 'density': 0.6071}]
- elapsed_sec: 0.34

**Notes**
- No post-hoc genre aggregation; camp scheme uses independent NPMI estimates.
- Strata with n_windows < 500 are skipped.

## p2_05

_Updated: 2026-05-18T21:56:23.016129+00:00_

**Parameters**
- gammas: `[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]`
- seeds_per_gamma: `10`
- consensus_gamma: `1.0`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/partitions_by_camp_genre`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/partitions_by_camp`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/partitions_by_camp_time`

**Stats**
- n_raw_partitions: 9240
- n_consensus_nodes: 132
- stratum_partitions: [{'scheme': 'camp_genre', 'stratum_key': 'DPP_debate', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-02', 'L2-05'], '1': ['L2-03', 'L2-07', 'L2-08'], '2': ['L2-06']}}, {'scheme': 'camp_genre', 'stratum_key': 'DPP_news', 'n_communities': 4, 'n_stable_communities': 4, 'communities': {'0': ['L2-01', 'L2-03'], '1': ['L2-02', 'L2-05'], '3': ['L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_genre', 'stratum_key': 'KMT_debate', 'n_communities': 2, 'n_stable_communities': 1, 'communities': {'1': ['L2-01'], '0': ['L2-02', 'L2-06', 'L2-07']}}, {'scheme': 'camp_genre', 'stratum_key': 'KMT_news', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-03', 'L2-05'], '1': ['L2-02', 'L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_genre', 'stratum_key': 'TPP_debate', 'n_communities': 2, 'n_stable_communities': 2, 'communities': {'1': ['L2-01', 'L2-05'], '0': ['L2-03', 'L2-06', 'L2-07']}}, {'scheme': 'camp_genre', 'stratum_key': 'TPP_news', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'1': ['L2-01', 'L2-03', 'L2-05'], '0': ['L2-02', 'L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp', 'stratum_key': 'DPP', 'n_communities': 4, 'n_stable_communities': 4, 'communities': {'0': ['L2-01', 'L2-03'], '1': ['L2-02', 'L2-05'], '3': ['L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp', 'stratum_key': 'KMT', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-03', 'L2-05'], '1': ['L2-02', 'L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp', 'stratum_key': 'TPP', 'n_communities': 4, 'n_stable_communities': 3, 'communities': {'1': ['L2-01', 'L2-03'], '0': ['L2-02', 'L2-04', 'L2-05'], '2': ['L2-06', 'L2-07'], '3': ['L2-08']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_campaign', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'1': ['L2-01', 'L2-02'], '0': ['L2-03', 'L2-05', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_nodate', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-02', 'L2-05'], '1': ['L2-03', 'L2-07', 'L2-08'], '2': ['L2-06']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_pre_registration', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'1': ['L2-01', 'L2-03'], '2': ['L2-02', 'L2-05'], '0': ['L2-04', 'L2-06', 'L2-07', 'L2-08']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_campaign', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-02', 'L2-05', 'L2-08'], '1': ['L2-03', 'L2-06', 'L2-07'], '2': ['L2-04']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_post_registration', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-03', 'L2-05'], '2': ['L2-02', 'L2-08'], '1': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_pre_registration', 'n_communities': 4, 'n_stable_communities': 2, 'communities': {'1': ['L2-01', 'L2-03'], '0': ['L2-02', 'L2-04', 'L2-08'], '3': ['L2-05'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_campaign', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-02', 'L2-03', 'L2-05'], '2': ['L2-04', 'L2-08'], '1': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_post_registration', 'n_communities': 2, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-02', 'L2-03', 'L2-05', 'L2-08'], '1': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_pre_registration', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'1': ['L2-01', 'L2-03', 'L2-05'], '0': ['L2-02', 'L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}]
- elapsed_sec: 0.28

**Notes**
- RBConfigurationVertexPartition; edge weight=npmi_median.
- stability(C)=mean co_assignment_freq(i,j) for i!=j in C across 70 partitions.
- consensus_gamma=1.0 (median of sweep; post-threshold graph).

## p2_06

_Updated: 2026-05-18T21:56:24.233981+00:00_

**Parameters**
- scheme: `camp`
- top_k: `3`
- qap_permutations: `1000`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/cross_camp/community_table.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/cross_camp/jaccard_all_pairs.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/cross_camp/qap_results.parquet`

**Stats**
- n_matched_triples: 3
- community_table: [{'dpp_community_id': 0, 'dpp_l2_set': 'L2-01|L2-03', 'kmt_community_id': 0, 'kmt_jaccard': 0.6666666666666666, 'tpp_community_id': 1, 'tpp_jaccard': 1.0}, {'dpp_community_id': 1, 'dpp_l2_set': 'L2-02|L2-05', 'kmt_community_id': 0, 'kmt_jaccard': 0.25, 'tpp_community_id': 0, 'tpp_jaccard': 0.6666666666666666}, {'dpp_community_id': 3, 'dpp_l2_set': 'L2-04|L2-08', 'kmt_community_id': 1, 'kmt_jaccard': 0.6666666666666666, 'tpp_community_id': 0, 'tpp_jaccard': 0.25}]
- qap_results: [{'camp_a': 'DPP', 'camp_b': 'KMT', 'hubert_gamma': 0.8771324746118299, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}, {'camp_a': 'DPP', 'camp_b': 'TPP', 'hubert_gamma': 0.8877425961143403, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}, {'camp_a': 'KMT', 'camp_b': 'TPP', 'hubert_gamma': 0.949434135985186, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}]
- jaccard_all_pairs_n: 121
- elapsed_sec: 0.19

**Notes**
- Report Hubert gamma as effect size; p-values on 28 edge pairs have limited power.

## p2_09

_Updated: 2026-05-18T21:56:25.377705+00:00_

**Parameters**
- edge_selection: `fdr`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/substantive_results.md`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/substantive_results.parquet`

**Stats**
- n_long_rows: 24
- elapsed_sec: 0.12

**Notes**
- 8-node L2 co-occurrence graphs are often near-saturated; community structure reflects edge-weight ranking.
- These are co-articulation patterns among framing functions, not Laclau equivalential chains (signifier-level).
- Hubert QAP gamma is reported as effect size; significance on ~28 edges has limited interpretive value.

## p2_11

_Updated: 2026-05-18T21:56:26.410291+00:00_

**Parameters**
- scheme: `camp_time`
- time_buckets: `['pre_registration', 'post_registration', 'campaign', 'election_plus']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/temporal_edge_stability.parquet`

**Stats**
- n_pairwise_comparisons: 235
- n_ci_disjoint: 30
- disjoint_rate: 0.1277
- elapsed_sec: 0.03

**Notes**
- ci_disjoint=True means 95% bootstrap CIs do not overlap between time buckets.

## p2_10

_Updated: 2026-05-18T21:56:29.356934+00:00_

**Parameters**
- robustness_ns: `[1, 5, 'document']`
- ref_n: `3`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/robustness_summary.parquet`

**Stats**
- n_comparisons: 23
- summary: [{'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': 'debate', 'spearman_rho': 0.7696969696969697, 'n_pairs': 10, 'sig_edge_jaccard': 0.4444, 'n_sig_ref': 6, 'n_sig_alt': 7}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.9195402298850575, 'n_pairs': 28, 'sig_edge_jaccard': 0.8333, 'n_sig_ref': 24, 'n_sig_alt': 20}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9414340448823207, 'n_pairs': 28, 'sig_edge_jaccard': 0.8333, 'n_sig_ref': 22, 'n_sig_alt': 22}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': 0.6, 'n_pairs': 6, 'sig_edge_jaccard': 0.5, 'n_sig_ref': 1, 'n_sig_alt': 2}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.9781061850027366, 'n_pairs': 28, 'sig_edge_jaccard': 0.8636, 'n_sig_ref': 20, 'n_sig_alt': 21}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': 'debate', 'spearman_rho': 0.9415584415584415, 'n_pairs': 21, 'sig_edge_jaccard': 0.9167, 'n_sig_ref': 12, 'n_sig_alt': 11}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.9682539682539683, 'n_pairs': 28, 'sig_edge_jaccard': 0.92, 'n_sig_ref': 24, 'n_sig_alt': 24}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9792008757525997, 'n_pairs': 28, 'sig_edge_jaccard': 0.9545, 'n_sig_ref': 22, 'n_sig_alt': 21}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': 0.7818181818181817, 'n_pairs': 10, 'sig_edge_jaccard': 0.6, 'n_sig_ref': 4, 'n_sig_alt': 4}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.9709906951286261, 'n_pairs': 28, 'sig_edge_jaccard': 0.7917, 'n_sig_ref': 20, 'n_sig_alt': 23}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.8237547892720307, 'n_pairs': 28, 'sig_edge_jaccard': 0.8571, 'n_sig_ref': 24, 'n_sig_alt': 28}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.8954570333880678, 'n_pairs': 28, 'sig_edge_jaccard': 0.8148, 'n_sig_ref': 22, 'n_sig_alt': 27}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': nan, 'n_pairs': 1, 'sig_edge_jaccard': 0.0, 'n_sig_ref': 1, 'n_sig_alt': 0}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.8839627805145046, 'n_pairs': 28, 'sig_edge_jaccard': 0.7143, 'n_sig_ref': 20, 'n_sig_alt': 28}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.9272030651340997, 'n_pairs': 28, 'sig_edge_jaccard': 0.8261, 'n_sig_ref': 23, 'n_sig_alt': 19}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9507389162561576, 'n_pairs': 28, 'sig_edge_jaccard': 0.7917, 'n_sig_ref': 22, 'n_sig_alt': 21}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9824849480021893, 'n_pairs': 28, 'sig_edge_jaccard': 0.8696, 'n_sig_ref': 20, 'n_sig_alt': 23}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.9885057471264367, 'n_pairs': 28, 'sig_edge_jaccard': 0.9167, 'n_sig_ref': 23, 'n_sig_alt': 23}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9819376026272578, 'n_pairs': 28, 'sig_edge_jaccard': 0.9545, 'n_sig_ref': 22, 'n_sig_alt': 21}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9698960043787629, 'n_pairs': 28, 'sig_edge_jaccard': 0.8261, 'n_sig_ref': 20, 'n_sig_alt': 22}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.856048166392994, 'n_pairs': 28, 'sig_edge_jaccard': 0.8214, 'n_sig_ref': 23, 'n_sig_alt': 28}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9113300492610837, 'n_pairs': 28, 'sig_edge_jaccard': 0.8148, 'n_sig_ref': 22, 'n_sig_alt': 27}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.8784893267651889, 'n_pairs': 28, 'sig_edge_jaccard': 0.7143, 'n_sig_ref': 20, 'n_sig_alt': 28}]
- elapsed_sec: 0.24
