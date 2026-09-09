# Phase 2 pipeline log

## p2_00

_Updated: 2026-08-05T05:07:00.907934+00:00_

**Parameters**
- corpus_csv: `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/01_results_datasets/Run_20260407_221407/final_results.csv`
- corpus_hash: `15a2af644ea51de9`
- boilerplate_policy: `drop_residue`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/labeled_units.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/diagnostics/unknown_camp_audit.json`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/diagnostics/boilerplate_audit.json`

**Stats**
- n_sentences_main: 20901
- stratum_counts: {('DPP', 'debate'): 388, ('DPP', 'news'): 6445, ('KMT', 'debate'): 196, ('KMT', 'news'): 5661, ('TPP', 'debate'): 382, ('TPP', 'news'): 7829}
- news_unknown_rate: 0.0000%
- n_multi_party_sentences: 2654
- n_boilerplate_dropped: 1314
- boilerplate_dropped_rate: 5.8629%
- elapsed_sec: 9.99

**Notes**
- Scrape-residue filter 'drop_residue' applied before the camp gate; prefix-tier rows (real sentence + scrape prefix) are kept and flagged in boiler_kind / boiler_tier.
- Dropping rows makes previously non-adjacent sentences adjacent in p2_01 windows; window spans are contiguous in position, not in original sent_idx.

## p2_02

_Updated: 2026-08-05T05:07:57.553026+00:00_

**Parameters**
- min_marginal_count: `30`
- schemes: `['camp_genre', 'camp', 'camp_time']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp_genre`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp_time`

**Stats**
- camp_genre: {'n_edges': 115, 'n_strata': 6, 'low_n_strata': [{'camp': 'KMT', 'genre': 'debate'}, {'camp': 'TPP', 'genre': 'debate'}], 'per_stratum_edges': {"('DPP', 'debate')": 21, "('DPP', 'news')": 28, "('KMT', 'news')": 28, "('TPP', 'debate')": 10, "('TPP', 'news')": 28}, 'empty_l2_rates': {"{'camp': 'DPP', 'genre': 'debate'}": '0.0919', "{'camp': 'DPP', 'genre': 'news'}": '0.1334', "{'camp': 'KMT', 'genre': 'debate'}": '0.2500', "{'camp': 'KMT', 'genre': 'news'}": '0.1637', "{'camp': 'TPP', 'genre': 'debate'}": '0.1835', "{'camp': 'TPP', 'genre': 'news'}": '0.1876'}}
- camp: {'n_edges': 84, 'n_strata': 3, 'low_n_strata': [], 'per_stratum_edges': {'DPP': 28, 'KMT': 28, 'TPP': 28}, 'empty_l2_rates': {"{'camp': 'DPP'}": '0.1289', "{'camp': 'KMT'}": '0.1642', "{'camp': 'TPP'}": '0.1874'}}
- camp_time: {'n_edges': 260, 'n_strata': 15, 'low_n_strata': [{'camp': 'DPP', 'time_bucket': 'election_plus'}, {'camp': 'DPP', 'time_bucket': 'post_registration'}, {'camp': 'KMT', 'time_bucket': 'election_plus'}, {'camp': 'KMT', 'time_bucket': 'nodate'}, {'camp': 'TPP', 'time_bucket': 'election_plus'}, {'camp': 'TPP', 'time_bucket': 'nodate'}], 'per_stratum_edges': {"('DPP', 'campaign')": 21, "('DPP', 'election_plus')": 1, "('DPP', 'nodate')": 21, "('DPP', 'post_registration')": 10, "('DPP', 'pre_registration')": 28, "('KMT', 'campaign')": 28, "('KMT', 'post_registration')": 21, "('KMT', 'pre_registration')": 28, "('TPP', 'campaign')": 28, "('TPP', 'election_plus')": 15, "('TPP', 'nodate')": 10, "('TPP', 'post_registration')": 21, "('TPP', 'pre_registration')": 28}, 'empty_l2_rates': {"{'camp': 'DPP', 'time_bucket': 'campaign'}": '0.1376', "{'camp': 'DPP', 'time_bucket': 'election_plus'}": '0.1058', "{'camp': 'DPP', 'time_bucket': 'nodate'}": '0.0919', "{'camp': 'DPP', 'time_bucket': 'post_registration'}": '0.2125', "{'camp': 'DPP', 'time_bucket': 'pre_registration'}": '0.1266', "{'camp': 'KMT', 'time_bucket': 'campaign'}": '0.1231', "{'camp': 'KMT', 'time_bucket': 'election_plus'}": '0.4651', "{'camp': 'KMT', 'time_bucket': 'nodate'}": '0.2500', "{'camp': 'KMT', 'time_bucket': 'post_registration'}": '0.1962', "{'camp': 'KMT', 'time_bucket': 'pre_registration'}": '0.1746', "{'camp': 'TPP', 'time_bucket': 'campaign'}": '0.1738', "{'camp': 'TPP', 'time_bucket': 'election_plus'}": '0.2535', "{'camp': 'TPP', 'time_bucket': 'nodate'}": '0.1835', "{'camp': 'TPP', 'time_bucket': 'post_registration'}": '0.2017', "{'camp': 'TPP', 'time_bucket': 'pre_registration'}": '0.1844'}}
- elapsed_sec: 0.75

**Notes**
- Empty-L2 windows are retained in the denominator to estimate P(A) as the unconditional probability of label presence across all articulatory sites in the stratum, consistent with the theoretical claim that absence of articulation is itself meaningful.
- Pair counts exclude is_empty_l2 windows; marginals include all windows.
- Step=1 window overlap inflates counts equally in P(A), P(B), P(AB) — ratio cancels in NPMI.

## p2_03

_Updated: 2026-08-05T05:11:24.464523+00:00_

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
- camp_time: 260
- elapsed_sec: 191.48

**Notes**
- Resample unit = document (cluster bootstrap). BH-FDR applied per stratum.

## p2_04

_Updated: 2026-08-05T13:25:27.720439+00:00_

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
- graph_stats: [{'scheme': 'camp_genre', 'stratum': 'DPP_debate', 'n_sig_ci': 13, 'n_sig_fdr': 12, 'n_sig_selected': 12, 'n_nodes': 7, 'n_edges': 10, 'density': 0.4762}, {'scheme': 'camp_genre', 'stratum': 'DPP_news', 'n_sig_ci': 21, 'n_sig_fdr': 21, 'n_sig_selected': 21, 'n_nodes': 8, 'n_edges': 20, 'density': 0.7143}, {'scheme': 'camp_genre', 'stratum': 'KMT_news', 'n_sig_ci': 22, 'n_sig_fdr': 22, 'n_sig_selected': 22, 'n_nodes': 8, 'n_edges': 20, 'density': 0.7143}, {'scheme': 'camp_genre', 'stratum': 'TPP_news', 'n_sig_ci': 20, 'n_sig_fdr': 20, 'n_sig_selected': 20, 'n_nodes': 8, 'n_edges': 18, 'density': 0.6429}, {'scheme': 'camp', 'stratum': 'DPP', 'n_sig_ci': 21, 'n_sig_fdr': 21, 'n_sig_selected': 21, 'n_nodes': 8, 'n_edges': 20, 'density': 0.7143}, {'scheme': 'camp', 'stratum': 'KMT', 'n_sig_ci': 22, 'n_sig_fdr': 22, 'n_sig_selected': 22, 'n_nodes': 8, 'n_edges': 20, 'density': 0.7143}, {'scheme': 'camp', 'stratum': 'TPP', 'n_sig_ci': 20, 'n_sig_fdr': 20, 'n_sig_selected': 20, 'n_nodes': 8, 'n_edges': 18, 'density': 0.6429}, {'scheme': 'camp_time', 'stratum': 'DPP_campaign', 'n_sig_ci': 16, 'n_sig_fdr': 16, 'n_sig_selected': 16, 'n_nodes': 7, 'n_edges': 16, 'density': 0.7619}, {'scheme': 'camp_time', 'stratum': 'DPP_nodate', 'n_sig_ci': 13, 'n_sig_fdr': 12, 'n_sig_selected': 12, 'n_nodes': 7, 'n_edges': 10, 'density': 0.4762}, {'scheme': 'camp_time', 'stratum': 'DPP_pre_registration', 'n_sig_ci': 18, 'n_sig_fdr': 18, 'n_sig_selected': 18, 'n_nodes': 8, 'n_edges': 17, 'density': 0.6071}, {'scheme': 'camp_time', 'stratum': 'KMT_campaign', 'n_sig_ci': 13, 'n_sig_fdr': 10, 'n_sig_selected': 10, 'n_nodes': 8, 'n_edges': 8, 'density': 0.2857}, {'scheme': 'camp_time', 'stratum': 'KMT_post_registration', 'n_sig_ci': 14, 'n_sig_fdr': 14, 'n_sig_selected': 14, 'n_nodes': 7, 'n_edges': 14, 'density': 0.6667}, {'scheme': 'camp_time', 'stratum': 'KMT_pre_registration', 'n_sig_ci': 18, 'n_sig_fdr': 18, 'n_sig_selected': 18, 'n_nodes': 8, 'n_edges': 18, 'density': 0.6429}, {'scheme': 'camp_time', 'stratum': 'TPP_campaign', 'n_sig_ci': 15, 'n_sig_fdr': 14, 'n_sig_selected': 14, 'n_nodes': 8, 'n_edges': 14, 'density': 0.5}, {'scheme': 'camp_time', 'stratum': 'TPP_post_registration', 'n_sig_ci': 14, 'n_sig_fdr': 14, 'n_sig_selected': 14, 'n_nodes': 7, 'n_edges': 14, 'density': 0.6667}, {'scheme': 'camp_time', 'stratum': 'TPP_pre_registration', 'n_sig_ci': 19, 'n_sig_fdr': 19, 'n_sig_selected': 19, 'n_nodes': 8, 'n_edges': 17, 'density': 0.6071}]
- elapsed_sec: 0.36

**Notes**
- No post-hoc genre aggregation; camp scheme uses independent NPMI estimates.
- Strata with n_windows < 500 are skipped.

## p2_05

_Updated: 2026-08-05T13:25:28.835818+00:00_

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
- stratum_partitions: [{'scheme': 'camp_genre', 'stratum_key': 'DPP_debate', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-02', 'L2-05'], '1': ['L2-03', 'L2-07', 'L2-08'], '2': ['L2-06']}}, {'scheme': 'camp_genre', 'stratum_key': 'DPP_news', 'n_communities': 4, 'n_stable_communities': 4, 'communities': {'0': ['L2-01', 'L2-03'], '1': ['L2-02', 'L2-05'], '3': ['L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_genre', 'stratum_key': 'KMT_debate', 'n_communities': 2, 'n_stable_communities': 1, 'communities': {'1': ['L2-01'], '0': ['L2-02', 'L2-06', 'L2-07']}}, {'scheme': 'camp_genre', 'stratum_key': 'KMT_news', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-03', 'L2-05'], '1': ['L2-02', 'L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_genre', 'stratum_key': 'TPP_debate', 'n_communities': 2, 'n_stable_communities': 2, 'communities': {'1': ['L2-01', 'L2-05'], '0': ['L2-03', 'L2-06', 'L2-07']}}, {'scheme': 'camp_genre', 'stratum_key': 'TPP_news', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'1': ['L2-01', 'L2-03', 'L2-05'], '0': ['L2-02', 'L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp', 'stratum_key': 'DPP', 'n_communities': 4, 'n_stable_communities': 3, 'communities': {'0': ['L2-01', 'L2-03'], '1': ['L2-02', 'L2-05'], '3': ['L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp', 'stratum_key': 'KMT', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-03', 'L2-05'], '1': ['L2-02', 'L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp', 'stratum_key': 'TPP', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'1': ['L2-01', 'L2-03'], '0': ['L2-02', 'L2-04', 'L2-05', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_campaign', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-02', 'L2-08'], '1': ['L2-03', 'L2-05'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_nodate', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-02', 'L2-05'], '1': ['L2-03', 'L2-07', 'L2-08'], '2': ['L2-06']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_pre_registration', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'1': ['L2-01', 'L2-03'], '2': ['L2-02', 'L2-05'], '0': ['L2-04', 'L2-06', 'L2-07', 'L2-08']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_campaign', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-02', 'L2-05', 'L2-08'], '1': ['L2-03', 'L2-06', 'L2-07'], '2': ['L2-04']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_post_registration', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-03', 'L2-05'], '2': ['L2-02', 'L2-08'], '1': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_pre_registration', 'n_communities': 4, 'n_stable_communities': 2, 'communities': {'1': ['L2-01', 'L2-03'], '0': ['L2-02', 'L2-04', 'L2-08'], '3': ['L2-05'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_campaign', 'n_communities': 4, 'n_stable_communities': 4, 'communities': {'0': ['L2-01', 'L2-03'], '1': ['L2-02', 'L2-05'], '3': ['L2-04', 'L2-08'], '2': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_post_registration', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-02', 'L2-05', 'L2-08'], '2': ['L2-03'], '1': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_pre_registration', 'n_communities': 4, 'n_stable_communities': 3, 'communities': {'0': ['L2-01', 'L2-03', 'L2-05'], '1': ['L2-02', 'L2-04'], '2': ['L2-06', 'L2-07'], '3': ['L2-08']}}]
- elapsed_sec: 0.29

**Notes**
- RBConfigurationVertexPartition; edge weight=npmi_median.
- stability(C)=mean co_assignment_freq(i,j) for i!=j in C across 70 partitions.
- consensus_gamma=1.0 (median of sweep; post-threshold graph).

## p2_06

_Updated: 2026-08-05T13:25:30.869920+00:00_

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
- community_table: [{'dpp_community_id': 0, 'dpp_l2_set': 'L2-01|L2-03', 'kmt_community_id': 0, 'kmt_jaccard': 0.6666666666666666, 'tpp_community_id': 1, 'tpp_jaccard': 1.0, 'category': 'shared', 'dpp_is_stable': True, 'kmt_is_stable': False, 'tpp_is_stable': True}, {'dpp_community_id': 1, 'dpp_l2_set': 'L2-02|L2-05', 'kmt_community_id': 0, 'kmt_jaccard': 0.25, 'tpp_community_id': 0, 'tpp_jaccard': 0.5, 'category': 'distinct', 'dpp_is_stable': False, 'kmt_is_stable': False, 'tpp_is_stable': False}, {'dpp_community_id': 3, 'dpp_l2_set': 'L2-04|L2-08', 'kmt_community_id': 1, 'kmt_jaccard': 0.6666666666666666, 'tpp_community_id': 0, 'tpp_jaccard': 0.5, 'category': 'partial', 'dpp_is_stable': True, 'kmt_is_stable': False, 'tpp_is_stable': False}]
- qap_results: [{'camp_a': 'DPP', 'camp_b': 'KMT', 'hubert_gamma': 0.8702911467917794, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}, {'camp_a': 'DPP', 'camp_b': 'TPP', 'hubert_gamma': 0.8844251475447105, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}, {'camp_a': 'KMT', 'camp_b': 'TPP', 'hubert_gamma': 0.9382769994074553, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}]
- jaccard_all_pairs_n: 100
- elapsed_sec: 0.19

**Notes**
- Report Hubert gamma as effect size; p-values on 28 edge pairs have limited power.
- QAP γ measures against-random distinctness, not between-camp distinctness; the latter is tested by p2_12_camp_permutation.py.

## p2_12

_Updated: 2026-08-05T13:29:57.161229+00:00_

**Parameters**
- n_permutations: `1000`
- seed_base: `42`
- min_marginal_count: `30`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/cross_camp/permutation_null_distribution.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/cross_camp/permutation_summary.parquet`

**Stats**
- permutation_summary: [{'test': 'gamma_dpp_kmt', 'camp_a': 'DPP', 'camp_b': 'KMT', 'observed_gamma': 0.889066110118461, 'observed_gamma_filtered': 0.8702911467917794, 'null_mean': 0.9391159658794777, 'null_q05': 0.9006002897665065, 'null_q50': 0.942423632762369, 'null_q95': 0.9666794632007956, 'p_value': 0.022977022977022976, 'n_permutations': 1000}, {'test': 'gamma_dpp_tpp', 'camp_a': 'DPP', 'camp_b': 'TPP', 'observed_gamma': 0.8985634622219014, 'observed_gamma_filtered': 0.8844251475447105, 'null_mean': 0.9471158316670032, 'null_q05': 0.9095481385013493, 'null_q50': 0.9496523237257589, 'null_q95': 0.9724048729781877, 'p_value': 0.02097902097902098, 'n_permutations': 1000}, {'test': 'gamma_kmt_tpp', 'camp_a': 'KMT', 'camp_b': 'TPP', 'observed_gamma': 0.93964209820709, 'observed_gamma_filtered': 0.9382769994074553, 'null_mean': 0.9399487697225842, 'null_q05': 0.8971967189956597, 'null_q50': 0.9428507240614713, 'null_q95': 0.9680179783665731, 'p_value': 0.42857142857142855, 'n_permutations': 1000}, {'test': 'joint_mean_gamma', 'camp_a': None, 'camp_b': None, 'observed_gamma': 0.9090905568491509, 'observed_gamma_filtered': 0.8976644312479817, 'null_mean': 0.9420601890896884, 'null_q05': 0.9173411489966299, 'null_q50': 0.9436571872685007, 'null_q95': 0.9623569086735877, 'p_value': 0.01998001998001998, 'n_permutations': 1000}]
- elapsed_sec: 265.76

**Notes**
- Permutation shuffles doc_id→camp assignments (preserving counts); windows inherit via doc_id.
- Full 28-edge point NPMI used for observed_gamma and null (no FDR inside permutations).
- observed_gamma_filtered is p2_06 FDR+bootstrap-median γ for diagnostic comparison only.

## p2_13

_Updated: 2026-08-05T13:29:57.189541+00:00_

**Parameters**
- ci_alpha: `0.05`
- bootstrap_n_resamples: `1000`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/cross_camp/edge_diff_pairwise.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/cross_camp/diff_by_l2_node.parquet`

**Stats**
- n_edges_tested: 84
- n_ci_disjoint: 12
- elapsed_sec: 0.03

**Notes**
- Non-overlapping 95% bootstrap CIs used as conservative cross-camp edge difference test.

## p2_14

_Updated: 2026-08-05T13:29:57.211635+00:00_

**Parameters**
- shared_jaccard_threshold: `0.66`
- scheme: `camp`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/subnetwork/node_selection_audit.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/subnetwork/differentiating_subnet_stats.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/subnetwork/shared_subnet_stats.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/subnetwork/rank_correlation_comparison.parquet`

**Stats**
- shared_nodes: ['L2-01', 'L2-03', 'L2-06', 'L2-07']
- differentiating_nodes: ['L2-02', 'L2-04', 'L2-05', 'L2-08']
- rank_correlation_comparison: [{'scope': 'full', 'camp_a': 'DPP', 'camp_b': 'KMT', 'metric': 'hubert_gamma', 'value': 0.8702911467917794}, {'scope': 'differentiating', 'camp_a': 'DPP', 'camp_b': 'KMT', 'metric': 'spearman_rho', 'value': 0.9276336570439174}, {'scope': 'shared', 'camp_a': 'DPP', 'camp_b': 'KMT', 'metric': 'spearman_rho', 'value': 0.9276336570439174}, {'scope': 'full', 'camp_a': 'DPP', 'camp_b': 'TPP', 'metric': 'hubert_gamma', 'value': 0.8844251475447105}, {'scope': 'differentiating', 'camp_a': 'DPP', 'camp_b': 'TPP', 'metric': 'spearman_rho', 'value': 0.898645105261295}, {'scope': 'shared', 'camp_a': 'DPP', 'camp_b': 'TPP', 'metric': 'spearman_rho', 'value': 0.898645105261295}, {'scope': 'full', 'camp_a': 'KMT', 'camp_b': 'TPP', 'metric': 'hubert_gamma', 'value': 0.9382769994074553}, {'scope': 'differentiating', 'camp_a': 'KMT', 'camp_b': 'TPP', 'metric': 'spearman_rho', 'value': 0.8857142857142858}, {'scope': 'shared', 'camp_a': 'KMT', 'camp_b': 'TPP', 'metric': 'spearman_rho', 'value': 0.8857142857142858}]
- elapsed_sec: 0.02

**Notes**
- Node selection is exploratory (Jaccard ≥ threshold vs both KMT and TPP); informs structured analysis, not a confirmatory test on the same data.

## p2_07

_Updated: 2026-08-05T13:30:06.752775+00:00_

**Parameters**
- scheme: `camp`
- top_n: `30`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/exemplars`

**Stats**
- n_exemplar_files: 10
- elapsed_sec: 9.54

## p2_08

_Updated: 2026-08-05T13:30:07.615839+00:00_

**Parameters**
- layout: `spring`
- unified_layout: `True`
- ci_disjoint_highlight: `True`
- node_color_scheme: `substantive_role`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/figures/main_3camp_pooled.pdf`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/figures/differentiating_subnet_3camp.pdf`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/figures/ego_l2_07_3camp.pdf`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/figures/permutation_null_distribution.pdf`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/figures/appendix_6grid_camp_genre.pdf`

**Stats**
- n_disjoint_edges_total: 12
- disjoint_by_camp: {'DPP': 7, 'KMT': 6, 'TPP': 6}
- elapsed_sec: 0.86

**Notes**
- Unified layout via union graph (FA2 if available, else spring).
- Node color: blue=shared {01,03,06,07}, orange=differentiating {02,04,05,08}.
- Red edges = CI-disjoint vs at least one other camp (p2_13).
- Differentiating sub-network figure is the primary money figure for sub-claim 1.

## p2_09

_Updated: 2026-08-05T13:30:07.708861+00:00_

**Parameters**
- edge_selection: `fdr`
- export_version: `subclaim1_v3_dpm`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/substantive_results.md`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/substantive_results.parquet`

**Stats**
- n_long_rows: 167
- elapsed_sec: 0.09

**Notes**
- Shared infrastructure cluster ({L2-06 民族自豪 / National Pride, L2-07 凝聚動員 / Solidarity & Vision}) is isomorphic across all three camps (Jaccard=1.0, stable).
- Differentiation concentrates in the differentiating sub-network {L2-02 差異化認同, L2-04 集體敘事再造, L2-05 共同威脅, L2-08 民主價值}.
- L2-07 (凝聚與願景動員, DPM Mobilising) shows a floating-signifier signature: the frame is shared infrastructure (clustered with L2-06) but has the most CI-disjoint cross-cluster articulation edges vs DPP–KMT/TPP — same element, camp-specific articulatory chains.
- Permutation test uses point NPMI on full 28-edge table (observed_gamma); observed_gamma_filtered (p2_06 FDR+bootstrap) is diagnostic only.
- Sub-network Spearman ρ identical across shared/differentiating splits is a structural consequence of FDR sparsity on 8 nodes, not a reporting error; sub-network analysis is descriptive only.
- Community partition is supplementary to edge-level analysis given 8-node saturation.

## p2_11

_Updated: 2026-08-05T13:30:07.951564+00:00_

**Parameters**
- scheme: `camp_time`
- time_buckets: `['pre_registration', 'post_registration', 'campaign', 'election_plus']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/temporal_edge_stability.parquet`

**Stats**
- n_pairwise_comparisons: 229
- n_ci_disjoint: 38
- disjoint_rate: 0.1659
- elapsed_sec: 0.02

**Notes**
- ci_disjoint=True means 95% bootstrap CIs do not overlap between time buckets.

## p2_01

_Updated: 2026-08-05T13:35:09.533242+00:00_

**Parameters**
- default_n: `3`
- step: `1`
- robustness_ns: `[1, 2, 5, 'document']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_n3.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_n1.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_n2.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_n5.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/windows/windows_ndocument.parquet`

**Stats**
- n_windows: 15481
- truncated_rate: 5.5100%
- empty_l2_rate: 16.1617%
- elapsed_sec: 36.02

**Notes**
- Sliding step=1; no cross-document windows.

## p2_10

_Updated: 2026-08-05T13:39:31.488767+00:00_

**Parameters**
- robustness_ns: `[1, 2, 5, 'document']`
- ref_n: `3`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/robustness_summary.parquet`

**Stats**
- n_comparisons: 31
- summary: [{'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': 'debate', 'spearman_rho': 0.7333333333333332, 'n_pairs': 10, 'sig_edge_jaccard': 0.4444, 'n_sig_ref': 6, 'n_sig_alt': 7}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.9282977558839628, 'n_pairs': 28, 'sig_edge_jaccard': 0.8182, 'n_sig_ref': 21, 'n_sig_alt': 19}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9299397920087575, 'n_pairs': 28, 'sig_edge_jaccard': 0.7308, 'n_sig_ref': 22, 'n_sig_alt': 23}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': 0.6, 'n_pairs': 6, 'sig_edge_jaccard': 0.5, 'n_sig_ref': 1, 'n_sig_alt': 2}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.9726327312534209, 'n_pairs': 28, 'sig_edge_jaccard': 0.7391, 'n_sig_ref': 20, 'n_sig_alt': 20}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'DPP', 'genre': 'debate', 'spearman_rho': 0.9688311688311689, 'n_pairs': 21, 'sig_edge_jaccard': 0.5294, 'n_sig_ref': 12, 'n_sig_alt': 14}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.9863163656267103, 'n_pairs': 28, 'sig_edge_jaccard': 0.9048, 'n_sig_ref': 21, 'n_sig_alt': 19}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9857690202517788, 'n_pairs': 28, 'sig_edge_jaccard': 0.9167, 'n_sig_ref': 22, 'n_sig_alt': 24}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': 1.0, 'n_pairs': 6, 'sig_edge_jaccard': 1.0, 'n_sig_ref': 1, 'n_sig_alt': 1}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.9896004378762999, 'n_pairs': 28, 'sig_edge_jaccard': 0.95, 'n_sig_ref': 20, 'n_sig_alt': 19}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': 'debate', 'spearman_rho': 0.9350649350649349, 'n_pairs': 21, 'sig_edge_jaccard': 0.9167, 'n_sig_ref': 12, 'n_sig_alt': 11}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.9633278598795839, 'n_pairs': 28, 'sig_edge_jaccard': 0.8333, 'n_sig_ref': 21, 'n_sig_alt': 23}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9742747673782157, 'n_pairs': 28, 'sig_edge_jaccard': 0.9545, 'n_sig_ref': 22, 'n_sig_alt': 21}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': 0.7818181818181817, 'n_pairs': 10, 'sig_edge_jaccard': 0.6, 'n_sig_ref': 4, 'n_sig_alt': 4}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.9726327312534209, 'n_pairs': 28, 'sig_edge_jaccard': 0.8182, 'n_sig_ref': 20, 'n_sig_alt': 20}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.8626163108921729, 'n_pairs': 28, 'sig_edge_jaccard': 0.7778, 'n_sig_ref': 21, 'n_sig_alt': 27}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9146141215106732, 'n_pairs': 28, 'sig_edge_jaccard': 0.7778, 'n_sig_ref': 22, 'n_sig_alt': 26}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': nan, 'n_pairs': 1, 'sig_edge_jaccard': 0.0, 'n_sig_ref': 1, 'n_sig_alt': 0}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.9217296113847838, 'n_pairs': 28, 'sig_edge_jaccard': 0.7143, 'n_sig_ref': 20, 'n_sig_alt': 28}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.9496442255062943, 'n_pairs': 28, 'sig_edge_jaccard': 0.8261, 'n_sig_ref': 21, 'n_sig_alt': 21}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9293924466338259, 'n_pairs': 28, 'sig_edge_jaccard': 0.6923, 'n_sig_ref': 22, 'n_sig_alt': 22}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9726327312534209, 'n_pairs': 28, 'sig_edge_jaccard': 0.7083, 'n_sig_ref': 20, 'n_sig_alt': 21}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '2', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.986863711001642, 'n_pairs': 28, 'sig_edge_jaccard': 0.9048, 'n_sig_ref': 21, 'n_sig_alt': 19}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '2', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9857690202517788, 'n_pairs': 28, 'sig_edge_jaccard': 0.9167, 'n_sig_ref': 22, 'n_sig_alt': 24}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '2', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9939792008757526, 'n_pairs': 28, 'sig_edge_jaccard': 0.95, 'n_sig_ref': 20, 'n_sig_alt': 19}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.9824849480021893, 'n_pairs': 28, 'sig_edge_jaccard': 0.913, 'n_sig_ref': 21, 'n_sig_alt': 23}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.975369458128079, 'n_pairs': 28, 'sig_edge_jaccard': 0.9545, 'n_sig_ref': 22, 'n_sig_alt': 21}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9737274220032841, 'n_pairs': 28, 'sig_edge_jaccard': 0.7826, 'n_sig_ref': 20, 'n_sig_alt': 21}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.8839627805145046, 'n_pairs': 28, 'sig_edge_jaccard': 0.75, 'n_sig_ref': 21, 'n_sig_alt': 28}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9146141215106732, 'n_pairs': 28, 'sig_edge_jaccard': 0.8148, 'n_sig_ref': 22, 'n_sig_alt': 27}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9184455391351943, 'n_pairs': 28, 'sig_edge_jaccard': 0.7143, 'n_sig_ref': 20, 'n_sig_alt': 28}]
- elapsed_sec: 0.28
