# p2_02 summary

Generated: 2026-05-18T21:22:12.356503+00:00

## Parameters

- **min_marginal_count**: `30`
- **schemes**: `['camp_genre', 'camp']`

## Outputs

- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp_genre`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/03_outputs/01_results_labelings/06_phase2_coarticulation/npmi_by_camp`

## Key statistics

- **camp_genre**: {'n_edges': 115, 'n_strata': 6}
- **camp**: {'n_edges': 84, 'n_strata': 3}
- **elapsed_sec**: 0.56

## Notes

Empty-L2 windows are retained in the denominator to estimate P(A) as the unconditional probability of label presence across all articulatory sites in the stratum, consistent with the theoretical claim that absence of articulation is itself meaningful.
Pair counts exclude is_empty_l2 windows; marginals include all windows.
Step=1 window overlap inflates counts equally in P(A), P(B), P(AB) — ratio cancels in NPMI.
