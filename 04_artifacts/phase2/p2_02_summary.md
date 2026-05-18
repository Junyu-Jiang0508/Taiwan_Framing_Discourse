# p2_02 summary

Generated: 2026-05-18T21:11:50.951909+00:00

## Parameters

- **min_marginal_count**: `30`
- **schemes**: `['camp_genre', 'camp']`

## Outputs

- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/04_artifacts/phase2/npmi_by_camp_genre`
- `/home/jain_farstrider/projects/Taiwan_Framing_Discourse/04_artifacts/phase2/npmi_by_camp`

## Key statistics

- **camp_genre**: {'n_edges': 115, 'n_strata': 6}
- **camp**: {'n_edges': 84, 'n_strata': 3}
- **elapsed_sec**: 0.61

## Notes

Empty-L2 windows are retained in the denominator to estimate P(A) as the unconditional probability of label presence across all articulatory sites in the stratum, consistent with the theoretical claim that absence of articulation is itself meaningful.
Pair counts exclude is_empty_l2 windows; marginals include all windows.
Step=1 window overlap inflates counts equally in P(A), P(B), P(AB) — ratio cancels in NPMI.
