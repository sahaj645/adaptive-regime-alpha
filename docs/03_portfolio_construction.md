# Phase 3 — Daily Point-in-Time Industry-Group Portfolios

Module: `src/pit_portfolios.py`. Outputs (cached to `data/processed/`, git-ignored):
`ig_returns_capwt.csv` (primary) and `ig_returns_eqwt.csv` (robustness) — daily
returns for 25 GICS industry groups, 2016-04-01 to 2025-12-31 (2,453 trading
days), 100% populated (no sparse cells).

## No-look-ahead construction

1. **Trading-calendar alignment.** The raw files are on a calendar-day grid (Phase 1). All series are restricted to NYSE trading days (from the ETF file) before returns are computed, so every return is trading-day to trading-day; weekend/holiday carry-forward cannot inject zero-return days.
2. **Membership lagged one month.** Membership snapshots are month-end (`PIT_Member_Date`). A name contributes to month M only if it was a point-in-time member in month **M-1**. This guarantees no within-month future membership is used, at the cost of the first month (hence the series begins 2016-04, not 2016-03).
3. **Trailing returns only.** Each return uses the prior trading day's price for the same security; a >7-day gap voids the return (handles temporary absences).
4. **Classification caveat (disclosed).** Index membership is point-in-time, but GICS classification is time-invariant in the source data (0% reclassification, Phase 2). The grouping is therefore mildly anachronistic; the effect on group-level co-movement is second-order and is carried into the Phase 5 leakage audit and the assumptions list rather than hidden.

## Corporate-action handling

`Price` is raw/split-unadjusted (`Market_Cap = Price x Shares_Out` exactly). Naive price returns would carry split artifacts up to +1000%/day (Phase 1). Handling:

- On any day with a **>10% change in `Shares_Out`** (a split or major corporate action, not organic drift), the split-neutral **Market-Cap return** is used in place of the price return — because `Market_Cap` is continuous through splits.
- All returns are then **winsorized to [-50%, +75%]** as a backstop against residual data errors.
- Aggregation to ~58-name portfolios further dilutes any single-name artifact.

No dividend data is available for constituents, so these are price returns; this is consistent across the universe and immaterial for a covariance/co-movement signal (it is not a total-return backtest of the constituents themselves).

## Weighting (both built)

- **Market-cap weight (primary):** each group return is its members weighted by prior-day `Market_Cap`. This is standard industry-index construction, more investable, dominated by large stable names (lower single-name noise), and naturally split-robust.
- **Equal weight (robustness):** each member counts equally — more sensitive to small-cap breadth, retained for the Phase 8 comparison.

## Validation

The cap-weighted aggregate of all 25 groups (cap-weighted across groups) correlates **0.997** with SPY's daily total return. A faithfully reconstructed point-in-time, cap-weighted S&P 1500 should track the large-cap market almost exactly — it does, which validates the membership reconstruction, split handling, and weighting end to end.

## Reproduction

```bash
PIT_RAW_DIR=/path/to/raw/pit/files python src/pit_portfolios.py
```

Deterministic; raw inputs are read from a configurable directory and only the small derived matrices are cached (the ~1 GB raw files are never committed).
