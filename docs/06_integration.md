# Phase 6 — Integration with the Jump Model

Module: `src/features.py`. The combined model's feature set is assembled here;
the only change versus the previous version is the **source of the Absorption
Ratio**.

## What changed — and what did not

| Component | Status |
|-----------|--------|
| Absorption-Ratio feed | **Replaced**: AR level + standardized shift now come from the 25 PIT industry groups (`data/processed/ar_pit_capwt.csv`) instead of the 11 sector ETFs |
| Jump-model features (dd10, sortino20, sortino60, ewma120) | Unchanged |
| Jump model (`core.fit_jump`, `core.forward_filter`) | Unchanged — no modification was mathematically necessary |
| Walk-forward engine (`engine.walk_forward`, train-only scaler, λ-CV, t+1 `pnl`) | Unchanged |

The new combined feature vector is `[dd10, sortino20, sortino60, ewma120, AR, dAR]` — identical in structure to the ETF version, so the model treats it identically; only the AR numbers differ.

## Verification

Running `fused_pit` through the unchanged engine:

```
fused_pit features: ['dd10','sortino20','sortino60','ewma120','AR','dAR']
shape: (1951, 6) | 2018-03-29 -> 2025-12-31
OOS regime/position series: 2020-01-02 -> 2025-12-31 | exposure 61% | switches 11
```

Concrete t+1 check in the real `engine.pnl` path:

```
signal day d         = 2020-01-16  strat return on d   = 0.0        (must be 0)
next trading day d+1 = 2020-01-17  strat return on d+1 = 0.003113 == SPY return on d+1
PASS — a regime read at close(d) is traded on d+1, never d.
```

## No new leakage introduced

- AR features are causal — machine-verified in Phase 5.
- Jump-model features are trailing EWMAs of past returns.
- The engine standardizes on training data only, infers regimes with the causal forward filter, and lags the position one day before it touches returns (established by the Stage-3 leak test and re-confirmed by the t+1 check above).

Integration is therefore leakage-neutral: swapping the AR source cannot introduce look-ahead because both the new AR and the engine are independently verified causal.

Performance — whether the constituent AR actually changes results — is measured honestly in Phase 7.
