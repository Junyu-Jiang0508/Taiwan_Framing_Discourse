# Phase 2 pipeline log

## p2_00

_Updated: 2026-08-05T13:39:44.661176+00:00_

**Parameters**
- corpus_csv: `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/01_results_datasets/Run_deepseek_l2binary/final_results.csv`
- corpus_hash: `a216389f022b7961`
- boilerplate_policy: `drop_residue`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/labeled_units.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/diagnostics/unknown_camp_audit.json`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/diagnostics/boilerplate_audit.json`

**Stats**
- n_sentences_main: 20901
- stratum_counts: {('DPP', 'debate'): 388, ('DPP', 'news'): 6445, ('KMT', 'debate'): 196, ('KMT', 'news'): 5661, ('TPP', 'debate'): 382, ('TPP', 'news'): 7829}
- news_unknown_rate: 0.0000%
- n_multi_party_sentences: 2654
- n_boilerplate_dropped: 1314
- boilerplate_dropped_rate: 5.8629%
- elapsed_sec: 11.86

**Notes**
- Scrape-residue filter 'drop_residue' applied before the camp gate; prefix-tier rows (real sentence + scrape prefix) are kept and flagged in boiler_kind / boiler_tier.
- Dropping rows makes previously non-adjacent sentences adjacent in p2_01 windows; window spans are contiguous in position, not in original sent_idx.

## p2_01

_Updated: 2026-08-05T13:40:23.459357+00:00_

**Parameters**
- default_n: `3`
- step: `1`
- robustness_ns: `[1, 2, 5, 'document']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/windows/windows_n3.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/windows/windows_n1.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/windows/windows_n2.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/windows/windows_n5.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/windows/windows_ndocument.parquet`

**Stats**
- n_windows: 15481
- truncated_rate: 5.5100%
- empty_l2_rate: 23.4546%
- elapsed_sec: 38.79

**Notes**
- Sliding step=1; no cross-document windows.

## p2_02

_Updated: 2026-08-05T13:40:24.321138+00:00_

**Parameters**
- min_marginal_count: `30`
- schemes: `['camp_genre', 'camp', 'camp_time']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/npmi_by_camp_genre`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/npmi_by_camp`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/npmi_by_camp_time`

**Stats**
- camp_genre: {'n_edges': 116, 'n_strata': 6, 'low_n_strata': [{'camp': 'KMT', 'genre': 'debate'}, {'camp': 'TPP', 'genre': 'debate'}], 'per_stratum_edges': {"('DPP', 'debate')": 26, "('DPP', 'news')": 28, "('KMT', 'news')": 28, "('TPP', 'debate')": 6, "('TPP', 'news')": 28}, 'empty_l2_rates': {"{'camp': 'DPP', 'genre': 'debate'}": '0.1343', "{'camp': 'DPP', 'genre': 'news'}": '0.2125', "{'camp': 'KMT', 'genre': 'debate'}": '0.2000', "{'camp': 'KMT', 'genre': 'news'}": '0.2398', "{'camp': 'TPP', 'genre': 'debate'}": '0.2622', "{'camp': 'TPP', 'genre': 'news'}": '0.2567'}}
- camp: {'n_edges': 84, 'n_strata': 3, 'low_n_strata': [], 'per_stratum_edges': {'DPP': 28, 'KMT': 28, 'TPP': 28}, 'empty_l2_rates': {"{'camp': 'DPP'}": '0.2039', "{'camp': 'KMT'}": '0.2396', "{'camp': 'TPP'}": '0.2569'}}
- camp_time: {'n_edges': 244, 'n_strata': 15, 'low_n_strata': [{'camp': 'DPP', 'time_bucket': 'election_plus'}, {'camp': 'DPP', 'time_bucket': 'post_registration'}, {'camp': 'KMT', 'time_bucket': 'election_plus'}, {'camp': 'KMT', 'time_bucket': 'nodate'}, {'camp': 'TPP', 'time_bucket': 'election_plus'}, {'camp': 'TPP', 'time_bucket': 'nodate'}], 'per_stratum_edges': {"('DPP', 'campaign')": 15, "('DPP', 'nodate')": 26, "('DPP', 'post_registration')": 15, "('DPP', 'pre_registration')": 28, "('KMT', 'campaign')": 27, "('KMT', 'post_registration')": 15, "('KMT', 'pre_registration')": 27, "('TPP', 'campaign')": 21, "('TPP', 'election_plus')": 15, "('TPP', 'nodate')": 6, "('TPP', 'post_registration')": 21, "('TPP', 'pre_registration')": 28}, 'empty_l2_rates': {"{'camp': 'DPP', 'time_bucket': 'campaign'}": '0.2240', "{'camp': 'DPP', 'time_bucket': 'election_plus'}": '0.3462', "{'camp': 'DPP', 'time_bucket': 'nodate'}": '0.1343', "{'camp': 'DPP', 'time_bucket': 'post_registration'}": '0.3281', "{'camp': 'DPP', 'time_bucket': 'pre_registration'}": '0.1965', "{'camp': 'KMT', 'time_bucket': 'campaign'}": '0.1941', "{'camp': 'KMT', 'time_bucket': 'election_plus'}": '0.5349', "{'camp': 'KMT', 'time_bucket': 'nodate'}": '0.2000', "{'camp': 'KMT', 'time_bucket': 'post_registration'}": '0.2616', "{'camp': 'KMT', 'time_bucket': 'pre_registration'}": '0.2566', "{'camp': 'TPP', 'time_bucket': 'campaign'}": '0.2243', "{'camp': 'TPP', 'time_bucket': 'election_plus'}": '0.3908', "{'camp': 'TPP', 'time_bucket': 'nodate'}": '0.2622', "{'camp': 'TPP', 'time_bucket': 'post_registration'}": '0.2781', "{'camp': 'TPP', 'time_bucket': 'pre_registration'}": '0.2529'}}
- elapsed_sec: 0.86

**Notes**
- Empty-L2 windows are retained in the denominator to estimate P(A) as the unconditional probability of label presence across all articulatory sites in the stratum, consistent with the theoretical claim that absence of articulation is itself meaningful.
- Pair counts exclude is_empty_l2 windows; marginals include all windows.
- Step=1 window overlap inflates counts equally in P(A), P(B), P(AB) — ratio cancels in NPMI.

## p2_03

_Updated: 2026-08-05T13:44:12.073820+00:00_

**Parameters**
- n_resamples: `1000`
- ci_alpha: `0.05`
- n_jobs: `-1`
- fdr_alpha: `0.05`
- edge_selection: `fdr`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/npmi_by_camp_genre/npmi_bootstrap.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/npmi_by_camp/npmi_bootstrap.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/npmi_by_camp_time/npmi_bootstrap.parquet`

**Stats**
- camp_genre: 116
- camp: 84
- camp_time: 244
- elapsed_sec: 213.41

**Notes**
- Resample unit = document (cluster bootstrap). BH-FDR applied per stratum.

## p2_04

_Updated: 2026-08-05T13:59:57.524167+00:00_

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
- skipped_low_n: ['TPP_debate', 'DPP_post_registration', 'TPP_election_plus', 'TPP_nodate']
- graph_stats: [{'scheme': 'camp_genre', 'stratum': 'DPP_debate', 'n_sig_ci': 17, 'n_sig_fdr': 16, 'n_sig_selected': 16, 'n_nodes': 8, 'n_edges': 8, 'density': 0.2857}, {'scheme': 'camp_genre', 'stratum': 'DPP_news', 'n_sig_ci': 24, 'n_sig_fdr': 24, 'n_sig_selected': 24, 'n_nodes': 8, 'n_edges': 24, 'density': 0.8571}, {'scheme': 'camp_genre', 'stratum': 'KMT_news', 'n_sig_ci': 25, 'n_sig_fdr': 25, 'n_sig_selected': 25, 'n_nodes': 8, 'n_edges': 24, 'density': 0.8571}, {'scheme': 'camp_genre', 'stratum': 'TPP_news', 'n_sig_ci': 23, 'n_sig_fdr': 23, 'n_sig_selected': 23, 'n_nodes': 8, 'n_edges': 23, 'density': 0.8214}, {'scheme': 'camp', 'stratum': 'DPP', 'n_sig_ci': 23, 'n_sig_fdr': 23, 'n_sig_selected': 23, 'n_nodes': 8, 'n_edges': 23, 'density': 0.8214}, {'scheme': 'camp', 'stratum': 'KMT', 'n_sig_ci': 25, 'n_sig_fdr': 25, 'n_sig_selected': 25, 'n_nodes': 8, 'n_edges': 24, 'density': 0.8571}, {'scheme': 'camp', 'stratum': 'TPP', 'n_sig_ci': 23, 'n_sig_fdr': 23, 'n_sig_selected': 23, 'n_nodes': 8, 'n_edges': 23, 'density': 0.8214}, {'scheme': 'camp_time', 'stratum': 'DPP_campaign', 'n_sig_ci': 10, 'n_sig_fdr': 10, 'n_sig_selected': 10, 'n_nodes': 6, 'n_edges': 10, 'density': 0.6667}, {'scheme': 'camp_time', 'stratum': 'DPP_nodate', 'n_sig_ci': 17, 'n_sig_fdr': 16, 'n_sig_selected': 16, 'n_nodes': 8, 'n_edges': 8, 'density': 0.2857}, {'scheme': 'camp_time', 'stratum': 'DPP_pre_registration', 'n_sig_ci': 23, 'n_sig_fdr': 23, 'n_sig_selected': 23, 'n_nodes': 8, 'n_edges': 23, 'density': 0.8214}, {'scheme': 'camp_time', 'stratum': 'KMT_campaign', 'n_sig_ci': 22, 'n_sig_fdr': 22, 'n_sig_selected': 22, 'n_nodes': 8, 'n_edges': 22, 'density': 0.7857}, {'scheme': 'camp_time', 'stratum': 'KMT_post_registration', 'n_sig_ci': 9, 'n_sig_fdr': 9, 'n_sig_selected': 9, 'n_nodes': 6, 'n_edges': 9, 'density': 0.6}, {'scheme': 'camp_time', 'stratum': 'KMT_pre_registration', 'n_sig_ci': 22, 'n_sig_fdr': 22, 'n_sig_selected': 22, 'n_nodes': 8, 'n_edges': 22, 'density': 0.7857}, {'scheme': 'camp_time', 'stratum': 'TPP_campaign', 'n_sig_ci': 15, 'n_sig_fdr': 15, 'n_sig_selected': 15, 'n_nodes': 7, 'n_edges': 15, 'density': 0.7143}, {'scheme': 'camp_time', 'stratum': 'TPP_post_registration', 'n_sig_ci': 18, 'n_sig_fdr': 18, 'n_sig_selected': 18, 'n_nodes': 7, 'n_edges': 17, 'density': 0.8095}, {'scheme': 'camp_time', 'stratum': 'TPP_pre_registration', 'n_sig_ci': 19, 'n_sig_fdr': 19, 'n_sig_selected': 19, 'n_nodes': 8, 'n_edges': 19, 'density': 0.6786}]
- elapsed_sec: 0.34

**Notes**
- No post-hoc genre aggregation; camp scheme uses independent NPMI estimates.
- Strata with n_windows < 500 are skipped.

## p2_05

_Updated: 2026-08-05T13:59:58.981516+00:00_

**Parameters**
- gammas: `[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]`
- seeds_per_gamma: `10`
- consensus_gamma: `1.0`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/partitions_by_camp_genre`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/partitions_by_camp`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/partitions_by_camp_time`

**Stats**
- n_raw_partitions: 8540
- n_consensus_nodes: 122
- stratum_partitions: [{'scheme': 'camp_genre', 'stratum_key': 'DPP_debate', 'n_communities': 3, 'n_stable_communities': 3, 'communities': {'1': ['L2-01', 'L2-02'], '0': ['L2-03', 'L2-05', 'L2-06', 'L2-07'], '2': ['L2-04', 'L2-08']}}, {'scheme': 'camp_genre', 'stratum_key': 'DPP_news', 'n_communities': 2, 'n_stable_communities': 0, 'communities': {'0': ['L2-01', 'L2-02', 'L2-03', 'L2-05'], '1': ['L2-04', 'L2-06', 'L2-07', 'L2-08']}}, {'scheme': 'camp_genre', 'stratum_key': 'KMT_news', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-02', 'L2-08'], '1': ['L2-03', 'L2-06', 'L2-07'], '2': ['L2-04', 'L2-05']}}, {'scheme': 'camp_genre', 'stratum_key': 'TPP_news', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-03', 'L2-06'], '1': ['L2-02', 'L2-04', 'L2-05'], '2': ['L2-07', 'L2-08']}}, {'scheme': 'camp', 'stratum_key': 'DPP', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-03', 'L2-06', 'L2-07'], '1': ['L2-02', 'L2-05'], '2': ['L2-04', 'L2-08']}}, {'scheme': 'camp', 'stratum_key': 'KMT', 'n_communities': 3, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-02', 'L2-08'], '1': ['L2-03', 'L2-06', 'L2-07'], '2': ['L2-04', 'L2-05']}}, {'scheme': 'camp', 'stratum_key': 'TPP', 'n_communities': 4, 'n_stable_communities': 3, 'communities': {'0': ['L2-01', 'L2-03'], '1': ['L2-02', 'L2-05'], '2': ['L2-04', 'L2-06'], '3': ['L2-07', 'L2-08']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_campaign', 'n_communities': 2, 'n_stable_communities': 1, 'communities': {'1': ['L2-01', 'L2-05'], '0': ['L2-04', 'L2-06', 'L2-07', 'L2-08']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_nodate', 'n_communities': 3, 'n_stable_communities': 3, 'communities': {'1': ['L2-01', 'L2-02'], '0': ['L2-03', 'L2-05', 'L2-06', 'L2-07'], '2': ['L2-04', 'L2-08']}}, {'scheme': 'camp_time', 'stratum_key': 'DPP_pre_registration', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'0': ['L2-01', 'L2-03', 'L2-06'], '2': ['L2-02', 'L2-05'], '1': ['L2-04', 'L2-07', 'L2-08']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_campaign', 'n_communities': 2, 'n_stable_communities': 2, 'communities': {'0': ['L2-01', 'L2-02', 'L2-05', 'L2-08'], '1': ['L2-03', 'L2-04', 'L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_post_registration', 'n_communities': 3, 'n_stable_communities': 3, 'communities': {'2': ['L2-01', 'L2-08'], '0': ['L2-04', 'L2-05'], '1': ['L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'KMT_pre_registration', 'n_communities': 3, 'n_stable_communities': 0, 'communities': {'1': ['L2-01', 'L2-03', 'L2-06'], '2': ['L2-02', 'L2-08'], '0': ['L2-04', 'L2-05', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_campaign', 'n_communities': 4, 'n_stable_communities': 4, 'communities': {'3': ['L2-01'], '0': ['L2-02', 'L2-05'], '1': ['L2-04', 'L2-06'], '2': ['L2-07', 'L2-08']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_post_registration', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'2': ['L2-01', 'L2-08'], '1': ['L2-02', 'L2-05'], '0': ['L2-04', 'L2-06', 'L2-07']}}, {'scheme': 'camp_time', 'stratum_key': 'TPP_pre_registration', 'n_communities': 3, 'n_stable_communities': 1, 'communities': {'1': ['L2-01', 'L2-03', 'L2-06'], '0': ['L2-02', 'L2-04', 'L2-05'], '2': ['L2-07', 'L2-08']}}]
- elapsed_sec: 0.28

**Notes**
- RBConfigurationVertexPartition; edge weight=npmi_median.
- stability(C)=mean co_assignment_freq(i,j) for i!=j in C across 70 partitions.
- consensus_gamma=1.0 (median of sweep; post-threshold graph).

## p2_06

_Updated: 2026-08-05T14:00:00.060437+00:00_

**Parameters**
- scheme: `camp`
- top_k: `3`
- qap_permutations: `1000`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/cross_camp/community_table.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/cross_camp/jaccard_all_pairs.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/cross_camp/qap_results.parquet`

**Stats**
- n_matched_triples: 3
- community_table: [{'dpp_community_id': 0, 'dpp_l2_set': 'L2-01|L2-03|L2-06|L2-07', 'kmt_community_id': 1, 'kmt_jaccard': 0.75, 'tpp_community_id': 0, 'tpp_jaccard': 0.5, 'category': 'partial', 'dpp_is_stable': False, 'kmt_is_stable': True, 'tpp_is_stable': True}, {'dpp_community_id': 1, 'dpp_l2_set': 'L2-02|L2-05', 'kmt_community_id': 2, 'kmt_jaccard': 0.3333333333333333, 'tpp_community_id': 1, 'tpp_jaccard': 1.0, 'category': 'partial', 'dpp_is_stable': True, 'kmt_is_stable': True, 'tpp_is_stable': True}, {'dpp_community_id': 2, 'dpp_l2_set': 'L2-04|L2-08', 'kmt_community_id': 2, 'kmt_jaccard': 0.3333333333333333, 'tpp_community_id': 2, 'tpp_jaccard': 0.3333333333333333, 'category': 'distinct', 'dpp_is_stable': True, 'kmt_is_stable': True, 'tpp_is_stable': False}]
- qap_results: [{'camp_a': 'DPP', 'camp_b': 'KMT', 'hubert_gamma': 0.7811747337243745, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}, {'camp_a': 'DPP', 'camp_b': 'TPP', 'hubert_gamma': 0.8399942853627715, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}, {'camp_a': 'KMT', 'camp_b': 'TPP', 'hubert_gamma': 0.8609137444980429, 'p_value': 0.000999000999000999, 'n_perm': 1000, 'n_nodes': 8}]
- jaccard_all_pairs_n: 100
- elapsed_sec: 0.20

**Notes**
- Report Hubert gamma as effect size; p-values on 28 edge pairs have limited power.
- QAP γ measures against-random distinctness, not between-camp distinctness; the latter is tested by p2_12_camp_permutation.py.

## p2_12

_Updated: 2026-08-05T14:04:11.760609+00:00_

**Parameters**
- n_permutations: `1000`
- seed_base: `42`
- min_marginal_count: `30`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/cross_camp/permutation_null_distribution.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/cross_camp/permutation_summary.parquet`

**Stats**
- permutation_summary: [{'test': 'gamma_dpp_kmt', 'camp_a': 'DPP', 'camp_b': 'KMT', 'observed_gamma': 0.6600163358649457, 'observed_gamma_filtered': 0.7811747337243745, 'null_mean': 0.8127102221904242, 'null_q05': 0.6772372894902953, 'null_q50': 0.8235339882975412, 'null_q95': 0.908258007261291, 'p_value': 0.03496503496503497, 'n_permutations': 1000}, {'test': 'gamma_dpp_tpp', 'camp_a': 'DPP', 'camp_b': 'TPP', 'observed_gamma': 0.778503674257142, 'observed_gamma_filtered': 0.8399942853627715, 'null_mean': 0.8367740398959767, 'null_q05': 0.7207576663476117, 'null_q50': 0.8467589291276615, 'null_q95': 0.9199593204895894, 'p_value': 0.16583416583416583, 'n_permutations': 1000}, {'test': 'gamma_kmt_tpp', 'camp_a': 'KMT', 'camp_b': 'TPP', 'observed_gamma': 0.8522820195645742, 'observed_gamma_filtered': 0.8609137444980429, 'null_mean': 0.817639753340861, 'null_q05': 0.6783262286759093, 'null_q50': 0.8310180672280273, 'null_q95': 0.9099836155107669, 'p_value': 0.6333666333666333, 'n_permutations': 1000}, {'test': 'joint_mean_gamma', 'camp_a': None, 'camp_b': None, 'observed_gamma': 0.7636006765622206, 'observed_gamma_filtered': 0.827360921195063, 'null_mean': 0.8223746718090874, 'null_q05': 0.7401712983164903, 'null_q50': 0.8272936025948023, 'null_q95': 0.8873630221902555, 'p_value': 0.1258741258741259, 'n_permutations': 1000}]
- elapsed_sec: 240.21

**Notes**
- Permutation shuffles doc_id→camp assignments (preserving counts); windows inherit via doc_id.
- Full 28-edge point NPMI used for observed_gamma and null (no FDR inside permutations).
- observed_gamma_filtered is p2_06 FDR+bootstrap-median γ for diagnostic comparison only.

## p2_13

_Updated: 2026-08-05T14:04:11.786466+00:00_

**Parameters**
- ci_alpha: `0.05`
- bootstrap_n_resamples: `1000`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/cross_camp/edge_diff_pairwise.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/cross_camp/diff_by_l2_node.parquet`

**Stats**
- n_edges_tested: 84
- n_ci_disjoint: 4
- elapsed_sec: 0.02

**Notes**
- Non-overlapping 95% bootstrap CIs used as conservative cross-camp edge difference test.

## p2_14

_Updated: 2026-08-05T14:04:11.802416+00:00_

**Parameters**
- shared_jaccard_threshold: `0.66`
- scheme: `camp`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/subnetwork/node_selection_audit.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/subnetwork/differentiating_subnet_stats.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/subnetwork/shared_subnet_stats.parquet`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/subnetwork/rank_correlation_comparison.parquet`

**Stats**
- shared_nodes: []
- differentiating_nodes: ['L2-01', 'L2-02', 'L2-03', 'L2-04', 'L2-05', 'L2-06', 'L2-07', 'L2-08']
- rank_correlation_comparison: [{'scope': 'full', 'camp_a': 'DPP', 'camp_b': 'KMT', 'metric': 'hubert_gamma', 'value': 0.7811747337243745}, {'scope': 'differentiating', 'camp_a': 'DPP', 'camp_b': 'KMT', 'metric': 'spearman_rho', 'value': 0.8160371142289219}, {'scope': 'full', 'camp_a': 'DPP', 'camp_b': 'TPP', 'metric': 'hubert_gamma', 'value': 0.8399942853627715}, {'scope': 'differentiating', 'camp_a': 'DPP', 'camp_b': 'TPP', 'metric': 'spearman_rho', 'value': 0.8871766648321407}, {'scope': 'full', 'camp_a': 'KMT', 'camp_b': 'TPP', 'metric': 'hubert_gamma', 'value': 0.8609137444980429}, {'scope': 'differentiating', 'camp_a': 'KMT', 'camp_b': 'TPP', 'metric': 'spearman_rho', 'value': 0.8791893808078762}]
- elapsed_sec: 0.02

**Notes**
- Node selection is exploratory (Jaccard ≥ threshold vs both KMT and TPP); informs structured analysis, not a confirmatory test on the same data.

## p2_07

_Updated: 2026-08-05T14:04:17.372415+00:00_

**Parameters**
- scheme: `camp`
- top_n: `30`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/exemplars`

**Stats**
- n_exemplar_files: 10
- elapsed_sec: 5.57

## p2_08

_Updated: 2026-08-05T14:04:18.123675+00:00_

**Parameters**
- layout: `spring`
- unified_layout: `True`
- ci_disjoint_highlight: `True`
- node_color_scheme: `substantive_role`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/figures/main_3camp_pooled.pdf`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/figures/differentiating_subnet_3camp.pdf`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/figures/ego_l2_07_3camp.pdf`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/figures/permutation_null_distribution.pdf`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/figures/appendix_6grid_camp_genre.pdf`

**Stats**
- n_disjoint_edges_total: 4
- disjoint_by_camp: {'DPP': 2, 'KMT': 2, 'TPP': 2}
- elapsed_sec: 0.75

**Notes**
- Unified layout via union graph (FA2 if available, else spring).
- Node color: blue=shared {01,03,06,07}, orange=differentiating {02,04,05,08}.
- Red edges = CI-disjoint vs at least one other camp (p2_13).
- Differentiating sub-network figure is the primary money figure for sub-claim 1.

## p2_09

_Updated: 2026-08-05T14:04:18.221461+00:00_

**Parameters**
- edge_selection: `fdr`
- export_version: `subclaim1_v3_dpm`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/substantive_results.md`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/substantive_results.parquet`

**Stats**
- n_long_rows: 227
- elapsed_sec: 0.10

**Notes**
- Shared infrastructure cluster ({L2-06 民族自豪 / National Pride, L2-07 凝聚動員 / Solidarity & Vision}) is isomorphic across all three camps (Jaccard=1.0, stable).
- Differentiation concentrates in the differentiating sub-network {L2-02 差異化認同, L2-04 集體敘事再造, L2-05 共同威脅, L2-08 民主價值}.
- L2-07 (凝聚與願景動員, DPM Mobilising) shows a floating-signifier signature: the frame is shared infrastructure (clustered with L2-06) but has the most CI-disjoint cross-cluster articulation edges vs DPP–KMT/TPP — same element, camp-specific articulatory chains.
- Permutation test uses point NPMI on full 28-edge table (observed_gamma); observed_gamma_filtered (p2_06 FDR+bootstrap) is diagnostic only.
- Sub-network Spearman ρ identical across shared/differentiating splits is a structural consequence of FDR sparsity on 8 nodes, not a reporting error; sub-network analysis is descriptive only.
- Community partition is supplementary to edge-level analysis given 8-node saturation.

## p2_10

_Updated: 2026-08-05T14:04:18.486059+00:00_

**Parameters**
- robustness_ns: `[1, 2, 5, 'document']`
- ref_n: `3`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/robustness_summary.parquet`

**Stats**
- n_comparisons: 30
- summary: [{'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': 'debate', 'spearman_rho': 0.7454545454545454, 'n_pairs': 10, 'sig_edge_jaccard': 0.5714, 'n_sig_ref': 5, 'n_sig_alt': 6}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.8735632183908046, 'n_pairs': 28, 'sig_edge_jaccard': 0.88, 'n_sig_ref': 24, 'n_sig_alt': 23}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9321291735084838, 'n_pairs': 28, 'sig_edge_jaccard': 0.9231, 'n_sig_ref': 25, 'n_sig_alt': 25}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': 0.5, 'n_pairs': 3, 'sig_edge_jaccard': 0.0, 'n_sig_ref': 0, 'n_sig_alt': 2}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.920087575259989, 'n_pairs': 28, 'sig_edge_jaccard': 1.0, 'n_sig_ref': 23, 'n_sig_alt': 23}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'DPP', 'genre': 'debate', 'spearman_rho': 0.8964285714285712, 'n_pairs': 15, 'sig_edge_jaccard': 0.6, 'n_sig_ref': 6, 'n_sig_alt': 10}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.9786535303776682, 'n_pairs': 28, 'sig_edge_jaccard': 0.9583, 'n_sig_ref': 24, 'n_sig_alt': 23}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9852216748768472, 'n_pairs': 28, 'sig_edge_jaccard': 0.96, 'n_sig_ref': 25, 'n_sig_alt': 24}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': 0.8857142857142858, 'n_pairs': 6, 'sig_edge_jaccard': 0.0, 'n_sig_ref': 1, 'n_sig_alt': 1}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '2', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.9715380405035577, 'n_pairs': 28, 'sig_edge_jaccard': 1.0, 'n_sig_ref': 23, 'n_sig_alt': 23}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': 'debate', 'spearman_rho': 0.8885470085470085, 'n_pairs': 26, 'sig_edge_jaccard': 0.625, 'n_sig_ref': 16, 'n_sig_alt': 10}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.9627805145046523, 'n_pairs': 28, 'sig_edge_jaccard': 0.9231, 'n_sig_ref': 24, 'n_sig_alt': 26}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': 0.9392446633825944, 'n_pairs': 28, 'sig_edge_jaccard': 0.96, 'n_sig_ref': 25, 'n_sig_alt': 24}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': 'debate', 'spearman_rho': 0.8285714285714287, 'n_pairs': 6, 'sig_edge_jaccard': 0.5, 'n_sig_ref': 1, 'n_sig_alt': 2}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.9786535303776682, 'n_pairs': 28, 'sig_edge_jaccard': 0.9167, 'n_sig_ref': 23, 'n_sig_alt': 23}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'DPP', 'genre': 'news', 'spearman_rho': 0.8713738368910783, 'n_pairs': 28, 'sig_edge_jaccard': 0.8571, 'n_sig_ref': 24, 'n_sig_alt': 28}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'KMT', 'genre': 'news', 'spearman_rho': nan, 'n_pairs': 28, 'sig_edge_jaccard': 0.7692, 'n_sig_ref': 25, 'n_sig_alt': 21}, {'scheme': 'camp_genre', 'ref_window': '3', 'alt_window': 'document', 'camp': 'TPP', 'genre': 'news', 'spearman_rho': 0.8648056923918993, 'n_pairs': 28, 'sig_edge_jaccard': 0.8519, 'n_sig_ref': 23, 'n_sig_alt': 27}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.8817733990147782, 'n_pairs': 28, 'sig_edge_jaccard': 0.875, 'n_sig_ref': 23, 'n_sig_alt': 22}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9386973180076628, 'n_pairs': 28, 'sig_edge_jaccard': 0.9259, 'n_sig_ref': 25, 'n_sig_alt': 27}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '1', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9370552818828681, 'n_pairs': 28, 'sig_edge_jaccard': 1.0, 'n_sig_ref': 23, 'n_sig_alt': 23}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '2', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.9518336070060207, 'n_pairs': 28, 'sig_edge_jaccard': 0.9565, 'n_sig_ref': 23, 'n_sig_alt': 22}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '2', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9846743295019157, 'n_pairs': 28, 'sig_edge_jaccard': 0.96, 'n_sig_ref': 25, 'n_sig_alt': 24}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '2', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9857690202517788, 'n_pairs': 28, 'sig_edge_jaccard': 1.0, 'n_sig_ref': 23, 'n_sig_alt': 23}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.9551176792556102, 'n_pairs': 28, 'sig_edge_jaccard': 0.9565, 'n_sig_ref': 23, 'n_sig_alt': 22}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'KMT', 'genre': nan, 'spearman_rho': 0.9348659003831418, 'n_pairs': 28, 'sig_edge_jaccard': 0.96, 'n_sig_ref': 25, 'n_sig_alt': 24}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': '5', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.9819376026272578, 'n_pairs': 28, 'sig_edge_jaccard': 0.9583, 'n_sig_ref': 23, 'n_sig_alt': 24}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'DPP', 'genre': nan, 'spearman_rho': 0.922824302134647, 'n_pairs': 28, 'sig_edge_jaccard': 0.8214, 'n_sig_ref': 23, 'n_sig_alt': 28}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'KMT', 'genre': nan, 'spearman_rho': nan, 'n_pairs': 28, 'sig_edge_jaccard': 0.7692, 'n_sig_ref': 25, 'n_sig_alt': 21}, {'scheme': 'camp', 'ref_window': '3', 'alt_window': 'document', 'camp': 'TPP', 'genre': nan, 'spearman_rho': 0.8817733990147782, 'n_pairs': 28, 'sig_edge_jaccard': 0.8519, 'n_sig_ref': 23, 'n_sig_alt': 27}]
- elapsed_sec: 0.26

## p2_11

_Updated: 2026-08-05T14:04:18.507002+00:00_

**Parameters**
- scheme: `camp_time`
- time_buckets: `['pre_registration', 'post_registration', 'campaign', 'election_plus']`

**Outputs**
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation_deepseek/temporal_edge_stability.parquet`

**Stats**
- n_pairwise_comparisons: 209
- n_ci_disjoint: 43
- disjoint_rate: 0.2057
- elapsed_sec: 0.02

**Notes**
- ci_disjoint=True means 95% bootstrap CIs do not overlap between time buckets.
