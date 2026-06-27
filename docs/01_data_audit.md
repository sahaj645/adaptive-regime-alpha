# Phase 1 — Data Audit: Point-in-Time S&P Constituent Files

Scope: `sp500_pit`, `sp400_pit`, `sp600_pit` (jointly, the point-in-time S&P 1500).
Every statement below is inferred directly from the files, not assumed.

## 1. Schema

All three files share an identical 15-column schema:

```
ID, DATE, CURRENCY, ID_DATE, Price, Volume, Market_Cap, Shares_Out,
GICS_Sector, GICS_Ind_Group, GICS_Industry, GICS_Sub_Industry,
Index, Index_Ticker, PIT_Member_Date
```

| File | Rows | Unique securities (ever) | Members/day (median) |
|------|------|--------------------------|----------------------|
| sp500_pit | 1,811,936 | 742 | 505 |
| sp400_pit | 1,438,662 | 887 | 400 |
| sp600_pit | 2,160,277 | 1,305 | 600 |

Member counts match the index definitions (500 / 400 / 600), and combine to a ~1,500-name universe.

## 2. Identifiers

`ID` is a stable Bloomberg-style security identifier (e.g. `1436513D UN Equity`), with an exchange suffix (`UN`=NYSE, `UW`=Nasdaq, `US`=composite). It is stable across ticker changes and is the correct key for tracking a security through time. There are **zero duplicate `(ID, DATE)` rows** in any file.

## 3. Calendar — a trap

`DATE` spans 2016-03-01 to 2025-12-31 but contains **3,593 distinct dates**, i.e. approximately the number of *calendar* days, not the ~2,475 NYSE *trading* days (`rows ≈ members × 3,593`). The panel is therefore a calendar-day grid with non-trading days carried forward, and some dates are degenerate (sp600 has dates with as few as 1 active member).

Implication: **returns must be computed on the NYSE trading calendar only** (align/intersect with the ETF trading dates), otherwise weekends inject spurious zero-return days and degenerate dates inject noise. This is enforced in Phase 3.

## 4. Membership mechanism — clean and point-in-time

`PIT_Member_Date` takes **118 distinct values**, and in 100.0% of rows it equals the month-end of `DATE`. Membership is therefore defined on **monthly point-in-time snapshots** (118 months = 2016-03 … 2025-12). The member set for any day `d` is the set of securities tagged with `PIT_Member_Date = month_end(d)`.

This is exactly what is needed for survivorship-free construction: membership is known as-of each month with no forward information.

## 5. Universe turnover (why survivorship bias would be material)

| File | Exited before end | Entered after start | of (ever) |
|------|-------------------|---------------------|-----------|
| sp500_pit | 238 | 237 | 742 |
| sp400_pit | 482 | 484 | 887 |
| sp600_pit | 694 | 701 | 1,305 |

Roughly 24 / 50 / 70 names churn per year. A static (today's-members) universe would discard ~30–50% of the historical cross-section — a large survivorship bias. Point-in-time membership must be used, and the data supports it.

## 6. GICS classification hierarchy

The full GICS tree is present, with <1% missing on active rows:

| Level | sp500 | sp400 | sp600 | Combined (≈) |
|-------|-------|-------|-------|--------------|
| Sector | 11 | 11 | 11 | 11 |
| Industry Group | 25 | 25 | 25 | 25 |
| Industry | 69 | 71 | 73 | ~74 |
| Sub-Industry | 139 | 139 | 145 | ~155 |
| Missing % (active) | 0.08% | 0.76% | 0.87% | — |

This is the menu for the Phase 2 universe decision: 11 sectors, 25 industry groups, ~74 industries, or ~155 sub-industries.

## 7. Corporate actions and price adjustment — a trap

`Market_Cap = Price × Shares_Out` holds **exactly** (median ratio 1.000, IQR 1.00–1.00), and both fields are ~100% populated. Two consequences:

- **Cap-weighting is feasible and clean** (`Market_Cap` directly available).
- `Price` is **raw / unadjusted**. Because `Shares_Out` absorbs split ratios (keeping `Market_Cap` continuous), naive `Price.pct_change()` produces large split artifacts. Counts of |daily return| > 50%: **sp500 = 6, sp400 = 38, sp600 = 165**, with maxima of +366% / +1094% / +203%. These are overwhelmingly split/data artifacts, concentrated in the small/mid-cap files.

Implication: constituent returns must be **cleaned (winsorized) before use**, and aggregation to portfolios further dilutes single-name glitches. Handled in Phase 3.

## 8. Missing values and quality

On active (priced, in-membership) rows: `Market_Cap` and `Shares_Out` are essentially 0% missing; GICS <1% missing; no duplicate keys. Off-membership rows carry blank `Price` and are correctly excluded by the membership filter. Overall the data is high quality once the calendar and price-adjustment traps are handled.

## 9. Decisions this audit forces (carried into later phases)

1. **Trading-calendar alignment** — restrict to NYSE trading days via the ETF calendar (Phase 3).
2. **Return cleaning** — winsorize daily constituent returns before aggregation; prefer portfolio aggregation to mute single-name artifacts (Phase 3).
3. **Point-in-time membership** — rebuild the member set monthly from `PIT_Member_Date`; never use end-of-sample membership (Phase 3, Phase 5).
4. **Weighting is open but feasible either way** — equal-weight or `Market_Cap`-weight (Phase 3 decision).
5. **Universe granularity is open** — 11 / 25 / 74 / 155 (Phase 2 decision; finer is not automatically better given covariance conditioning).

## 10. Data handling / repository note

The raw PIT files (~1 GB) are **not** committed to the repository (`.gitignore`d). The pipeline will read them from a configurable raw-data path and cache only the small derived portfolio-return matrix to `data/processed/`. The copyrighted source-paper PDFs are likewise excluded from version control.
