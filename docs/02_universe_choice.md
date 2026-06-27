# Phase 2 — Cross-Sectional Universe Choice

Decision: build the Absorption Ratio cross-section from **GICS Industry-Group portfolios (N = 25)** constructed from the point-in-time S&P 1500.

## Evidence

Combined S&P 1500, 118 monthly snapshots, ~1,504 names/month:

| GICS level | N | Median stocks/bucket | Min bucket | Sparse bucket-months (<5) | T/N @252d | T/N @500d |
|------------|---|----------------------|-----------|---------------------------|-----------|-----------|
| Sector | 11 | 107 | 48 | 0.0% | 22.9 | 45.5 |
| **Industry Group** | **25** | **58** | **4** | **0.4%** | **10.1** | **20.0** |
| Industry | 73 | 14 | 1 | 10.9% | 3.5 | 6.8 |
| Sub-Industry | 159 | 7 | 1 | 32.9% | 1.6 | 3.1 |
| Individual stocks | ~1,504 | 1 | 1 | n/a | 0.17 | 0.33 |

## Criterion-by-criterion

**Covariance conditioning (the binding constraint).** A rolling N×N covariance estimated over T days requires T ≫ N to be well-conditioned; otherwise the eigenvalues — and therefore the Absorption Ratio — are unstable and biased. Kritzman et al. operated at roughly 500/51 ≈ 10. Industry groups reproduce that exactly at a 252-day window (T/N = 10.1). Individual stocks (T/N = 0.17) and sub-industries (1.6) give singular or near-singular matrices and are disqualified on this criterion alone.

**Cross-sectional richness.** The number of principal components used is ~N/5 (Kritzman's convention): 2 for sectors, **5 for industry groups**, ~15 for industries. Industry groups are a real resolution gain over the 11-sector ETF version while staying well-conditioned.

**Sparsity / portfolio stability.** Each bucket must hold enough names to form a stable portfolio return. Industry groups average 58 names with only 0.4% of bucket-months below five constituents. Industries (10.9%) and sub-industries (32.9%) have many thin buckets whose returns are dominated by idiosyncratic single-name noise, which propagates straight into the covariance.

**Turnover / classification stability.** In this dataset GICS labels are time-invariant per security (0.000% reclassification), so portfolio composition changes only through index membership, which is slow at the group level. This stability is a point in favour of groups; it also flags a data caveat (below).

**Fidelity to Kritzman.** The paper uses "industry portfolios." GICS Industry Groups are the standard, conditioning-matched analog. GICS Industries (73) are closest to his literal count (51) but were estimated in 1998–2010 on a 500-day window; at our sample length and a 252-day window they are poorly conditioned, so they are retained only as a longer-window robustness check (Phase 8), not the primary universe.

## Recommendation

**Primary universe: 25 GICS Industry-Group portfolios, 252-day covariance window, top-5 eigenvectors (N/5).** This is the only choice that is simultaneously (i) a genuine richness upgrade over sectors, (ii) well-conditioned at our data length, and (iii) low-sparsity/stable. Weighting (equal vs market-cap) is decided in Phase 3.

**Robustness alternative (Phase 8): 73 GICS Industries, 500-day window, with Ledoit-Wolf covariance shrinkage** to handle the weaker conditioning — the variant closest to Kritzman's literal granularity.

## Documented caveat (carried to Phase 5)

GICS classification is time-invariant in the source data, indicating the current (extraction-date) classification was backfilled across history rather than supplied point-in-time. Index *membership* is correctly point-in-time; *classification* is not. For an industry-group Absorption Ratio the effect is second-order (grouping is stable and genuine reclassifications are rare at the group level), but it is a real point-in-time impurity and will be disclosed in the leakage audit and assumptions, not hidden.
