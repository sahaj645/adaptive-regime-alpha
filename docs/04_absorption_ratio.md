# Phase 4 — Institutional Absorption Ratio

Module: `src/absorption_ratio_pit.py`. Computes the Absorption Ratio on the
25-group cap-weighted matrix (primary) and equal-weighted matrix (robustness),
and benchmarks it against the 11-ETF implementation. The AR *method* is held
identical to the ETF version, so the only variable is the cross-sectional universe.

## Method and alignment with Kritzman et al.

| Choice | This implementation | Kritzman (2010) | Rationale / deviation |
|--------|--------------------|-----------------|-----------------------|
| Universe | 25 GICS industry groups (cap-wt) | ~51 industry portfolios | Conditioning at our sample length; 73 industries is the Phase-8 variant |
| Window | 252-day, equal-weighted | ~500-day, exp-weighted | T/N = 10 matches the paper's 500/51; preserves the short sample. EW-cov & 500d are Phase-8 variants |
| Matrix | Covariance | Covariance | Same as paper and the ETF AR; correlation-matrix is a Phase-8 variant |
| Components | n = round(N/5) = **5** | n ≈ N/5 | Paper convention — and the richness gain (5 vs the ETF's 2) |
| AR | Σ top-5 eigenvalues / Σ all | Same | — |
| Signal | dAR = (MA15 − MA252)/STD252 | Standardized shift | Same as paper |
| Missing data | None (100% populated groups) | — | Group aggregation removes the single-name gaps; no imputation, no shrinkage needed at T/N = 10 |

Numerically the 25×25 covariance over 252 days is well-conditioned; `eigvalsh` on the symmetric matrix is stable.

## Result vs the ETF implementation

| | PIT-cap (N=25, n=5) | ETF (N=10, n=2) |
|---|---|---|
| AR level (mean) | 0.827 | 0.778 |
| corr(level) with ETF | **0.93** | — |
| corr(shift) with ETF | **0.86** | — |

cap-weight vs equal-weight AR levels correlate 0.94, so the weighting choice barely moves the signal.

Forward-20-day SPY return after the shift exceeds +1σ (analysis-only diagnostic of timing quality):

| Signal | after dAR>+1σ | otherwise | gap |
|--------|---------------|-----------|-----|
| PIT-cap | +2.07% | +0.77% | +1.30 |
| PIT-eq | +2.28% | +0.66% | +1.62 |
| ETF | +2.85% | +0.41% | +2.44 |

## Honest early read

The constituent-based Absorption Ratio is a cleaner, more faithful implementation — point-in-time, survivorship-free, 5 principal components instead of 2, validated to track the market at 0.997. **But it is 0.93 correlated with the coarse ETF AR: the richer cross-section measures the same systemic co-movement, just at higher resolution.** Sector ETFs already span most of the cross-sectional fragility, so refining to 25 groups sharpens the signal without transforming it.

Critically, the new AR exhibits the **same coincident-with-bottoms behaviour** — a positive shift is still followed by *higher* forward returns (the wrong sign for a risk-off trigger), only mildly attenuated versus the ETF version. This says the binding limitation was never universe coarseness; it is that cross-sectional fragility, however well measured, does not translate into a directional timing edge for SPY in this sample.

Whether the sharper signal nonetheless helps the jump model is tested honestly in Phase 7. Given a 0.93 correlation with a signal we already showed added no value, expectations are tempered — but the test is run regardless, and reported either way.

## Outputs

`data/processed/ar_pit_capwt.csv`, `ar_pit_eqwt.csv` (git-ignored, regenerable); comparison figure `outputs/ar_pit_vs_etf.png`.
