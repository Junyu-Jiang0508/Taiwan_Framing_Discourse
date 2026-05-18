# Phase 2 substantive results
_Edge selection: `fdr`; FDR α=0.05_
## Interpretive notes
- 8-node L2 co-occurrence graphs are often near-saturated; community structure reflects edge-weight ranking.
- These are co-articulation patterns among framing functions, not Laclau equivalential chains (signifier-level).
- Hubert QAP gamma is reported as effect size; significance on ~28 edges has limited interpretive value.

## Cross-camp community matching (top-k greedy)

| dpp_community_id | dpp_l2_set | kmt_community_id | kmt_jaccard | tpp_community_id | tpp_jaccard |
| --- | --- | --- | --- | --- | --- |
| 0 | L2-01|L2-03 | 0 | 0.6666666666666666 | 1 | 1.0 |
| 1 | L2-02|L2-05 | 0 | 0.25 | 0 | 0.6666666666666666 |
| 3 | L2-04|L2-08 | 1 | 0.6666666666666666 | 0 | 0.25 |
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

_p-values: interpret cautiously with n≈28 edge pairs._

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
