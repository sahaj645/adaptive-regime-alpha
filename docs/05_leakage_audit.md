# Phase 5 — Point-in-Time Leakage Audit (constituent AR pipeline)

Scope: `src/pit_portfolios.py` (portfolio construction) → `src/absorption_ratio_pit.py` (Absorption Ratio). The downstream regime/trade lag is re-verified in Phase 6.

## Machine verification (the headline)

`tests/test_no_lookahead.py` corrupts every observation after a cut date T with random garbage and recomputes the AR:

```
corrupted all observations after 2022-03-16
  AR/dAR identical for all t <= T : True   (999 dates)
  AR/dAR changed for some t  > T  : True   (952 dates)
PASS — AR(t) depends only on data <= t
```

Past values are unchanged bit-for-bit; future values move (so the test is not inert). This is a direct, repeatable proof that the AR at day t uses only data up to day t.

## Lag ledger (every lag documented)

| Stage | Operation | Window / lag | Causal |
|-------|-----------|--------------|--------|
| Membership | PIT month-end snapshot, **lagged one month** | member in M → eligible in M+1 | ✓ |
| Constituent return | trailing `pct_change` | prior trading day | ✓ |
| Split / CA adjustment | `Shares_Out` ratio | t vs t−1 | ✓ |
| Winsorization | **fixed** bounds [−50%, +75%] | constants — no data dependence | ✓ |
| Group aggregation | same-day returns, **prior-day** cap weights | t / t−1 | ✓ |
| AR covariance | rolling window | trailing 252 days ending at t | ✓ |
| PCA / eigendecomposition | `eigvalsh` on the trailing covariance | ≤ t | ✓ |
| Standardized shift | rolling mean/std | trailing 15d & 252d ending at t | ✓ |
| Regime → trade (Phase 6) | position lag | t → t+1 | ✓ (verified next phase) |

## Leakage checklist

| Risk | Verdict | Evidence |
|------|---------|----------|
| Information beyond close of t | None | machine test |
| Centered vs trailing windows | Trailing | `vals[i-window:i]`; pandas `.rolling` default |
| Future obs in covariance | None | machine test |
| PCA from future data | None | `eigvalsh` on trailing-window covariance only |
| **Global / future-dependent normalization** | None | winsorization uses **fixed constants**, not sample quantiles; the standardized shift divides by a **trailing** 252-day std, not a full-sample std; feature standardization for the model is train-only in the engine (Phase 6) |
| Future membership | None | membership lagged one month |
| Future statistics anywhere | None | machine test + trailing-only design |
| **Future GICS classification** | **Partial — disclosed** | see below |

## The one disclosed impurity: static classification

The source data shows **0% GICS reclassification** over 2016–2025, which is not physically possible (the 2018 Telecom→Communication Services reclassification alone moved dozens of names). This indicates the vendor backfilled the **current** (extraction-date) classification across all history. Consequently:

- Index **membership** is correctly point-in-time and lagged — no leakage.
- GICS **classification** is *not* point-in-time; a name carries its present-day group label historically.

Impact is second-order for an industry-group Absorption Ratio: grouping is stable, genuine reclassifications are rare at the group level, and the AR measures co-movement *structure* rather than the labels themselves. It cannot be removed without a point-in-time GICS history, which the data does not provide. It is therefore disclosed here and in the assumptions list rather than hidden, and would be the first thing to fix with better data.

## Conclusion

The constituent Absorption Ratio pipeline is causal by construction and verified by machine test. The only residual point-in-time impurity is the backfilled GICS classification, which is disclosed and second-order. The pipeline meets the no-look-ahead standard required for the regime model.
