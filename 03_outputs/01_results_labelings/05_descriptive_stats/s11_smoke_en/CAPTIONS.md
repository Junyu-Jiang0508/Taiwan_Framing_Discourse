# Figure and table notes (draft captions)

**Unit of analysis:** sentence-level unit (same as the corpus segmentation and labeling pipeline).

- **N (all rows in corpus CSV):** 22560
- **N with non-empty L1:** 22560 (subset used for L1 distributions, time trends, and L1×L2 joint prevalence)
- **N with non-empty L2 (among L1-valid rows):** 13524 (Table 3 denominators are per-candidate non-empty-L2 counts, see table)
- **Candidate subset:** Lai (DPP), Hou (KMT), Ko (TPP) (rows with other inferred candidates, e.g. Unknown, excluded from Tables 2–3 and figures)

## Table 1 — Corpus profile

By mapped outlet family, candidate (English label), and election phase (**Primary season** / **Nomination period** / **General campaign** / **Final sprint** / **Post-election / other** / **No valid date**): number of units, mean sentence length (characters), and L2 non-empty rate (%). Phase boundaries: 2023-07-01, 2023-11-20 (CEC presidential registration window start), 2024-01-01, 2024-01-13 (polling day). Post-election includes news after polling day and rows without a parseable date.

## Table 2 & Figure 1 — L1 frequency by candidate

Raw counts for seven L1 categories plus **column percentages** (within-candidate L1 composition, not row percentages). Figure 1: stacked bars by candidate, fill = L1 code.

## Figure 2 — L1 over time

Panels for L1-01, L1-03, L1-06, L1-07. Daily shares → **7-day moving average** → mean of smoothed values within each **weekly** bin; line style distinguishes candidates. Companion CSV lists period labels.

## Table 3 — L2 prevalence (multi-label)

Among units with **non-empty L2** for that candidate, the percentage of units carrying each L2 code. With multi-label rows, prevalences can sum to more than 100%.

## Table 4 & Figure 3 — L2 co-occurrence

8×8 matrix of **absolute** co-occurrence counts: diagonal = number of units containing that label; off-diagonal = units containing both labels. No φ or correlation coefficients in this descriptive stage.

## Table 5 — L1×L2 joint prevalence

Rows = L1, columns = L2; cell = among units with L1 = *i*, the percentage that also carry L2 = *j*.

## Figure 4 (optional) — Core L2 time trends

L2-05 and L2-07 (by default) over time by candidate; same smoothing and binning as Figure 2.

---

**Reporting:** percentages rounded to one decimal. **No** χ² or other significance tests at this descriptive stage.
