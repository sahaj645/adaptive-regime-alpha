# Phase 7 — Backtest: Does the Constituent AR Change the Result?

All horses run through the identical no-look-ahead engine (common 2018-start
window, CV-tuned λ, t+1 execution, 2 bps cost). OOS 2020-01 to 2025-12.

| Strategy | CAGR | Ann.Ret | Vol | Sharpe | Sortino | MaxDD | Calmar | Expo | Switches | Avg dur (d) |
|----------|------|---------|-----|--------|---------|-------|--------|------|----------|-------------|
| Buy & hold SPY | 15.0% | 16.2% | 20.7% | 0.78 | 0.96 | −33.7% | 0.45 | 100% | 1 | — |
| JM-only | 6.6% | 6.9% | 10.6% | 0.65 | 0.64 | −16.2% | 0.41 | 63% | 21 | 72 |
| AR-ETF only | 11.6% | 11.9% | 13.7% | 0.87 | 0.54 | −22.2% | 0.52 | 30% | 4 | 377 |
| AR-PIT only | 13.0% | 13.2% | 14.0% | 0.94 | 0.67 | −18.3% | 0.71 | 36% | 7 | 215 |
| Fused-ETF | 6.9% | 7.2% | 10.3% | 0.70 | 0.72 | −14.6% | 0.47 | 62% | 15 | 101 |
| **Fused-PIT** | **10.0%** | 10.0% | 10.1% | **0.99** | **0.99** | −15.1% | **0.66** | 61% | 11 | 137 |

## The within-run finding

In this run the **constituent AR helped, contrary to the expectation set in Phase 4.** The cleanest comparison is apples-to-apples — Fused-ETF vs Fused-PIT differ *only* in the AR source:

- **Fused-PIT 0.99 Sharpe vs Fused-ETF 0.70** — ΔSharpe +0.30, paired bootstrap 95% CI [−0.08, +0.71], **P(Fused-PIT > Fused-ETF) = 94%.** Nearly significant, and it is a controlled comparison. The richer 25-group / 5-PC fragility signal placed the regime boundaries better than the 11-ETF / 2-PC one.
- Fused-PIT also has the best Calmar (0.66) and the joint-best drawdown (−15%) at 61% exposure.

I predicted this would not happen (0.93 AR correlation). It did. Reported as found.

## Significance and the honest brakes

- **Fused-PIT vs Buy & hold: ΔSharpe +0.21, 95% CI [−0.47, +0.88], P(win) = 71%.** Better point estimate, but the interval straddles zero — **not** a statistically established edge over buy-and-hold.
- **Estimate instability is severe.** The *identical* AR-ETF-only signal scored Sharpe **0.60 (worst horse) in the original Stage-7 run and 0.87 here**, purely from a ~6-week change in the training-window start. JM-only has printed 0.65–0.97 across runs. These swings are larger than most of the differences in the table, so single-run rankings are not trustworthy on their own.
- This is one OOS sample (~2 bear episodes) examined across many configurations; deflated-Sharpe / multiple-testing discipline applies, and the Phase-8 robustness sweep is what decides whether the Fused-PIT advantage survives.

## Verdict (provisional)

The constituent-based Absorption Ratio is both better-built *and*, in this run, materially better-performing inside the fusion than the ETF version (Sharpe 0.70 → 0.99, P = 94%). That is a genuine, leakage-free, apples-to-apples improvement and it overturns my Phase-4 expectation. **But** it does not convincingly beat buy-and-hold (P = 71%), and the point estimates are unstable enough that the result must be confirmed across windows, penalties, weightings, and granularities before any claim. Phase 8 runs that test; the replace/keep recommendation waits for it.
