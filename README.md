# Regime-Switching SPY Allocation: Combined Absorption Ratio and Statistical Jump Model

A no-look-ahead daily regime model that maps a binary SPY allocation (1 = fully
invested, 0 = cash) from a combination of the Kritzman et al. Absorption Ratio
(cross-sectional fragility) and the Shu-Yu-Mulvey Statistical Jump Model
(own-asset trend and downside regime). All signals use information available only
at the close of day `t` and are executed on day `t+1`. The required deliverable
holds cash at zero yield and applies no leverage; it is evaluated out-of-sample
against buy-and-hold SPY on a risk-adjusted basis.

## Objective

Equity drawdowns concentrate in turbulent, tightly coupled regimes. If such
regimes can be identified causally, exposure can be reduced before losses are
realized. The scope is a single risk asset (SPY) at daily frequency, long-or-flat,
no leverage, cash at zero yield. Success is defined by risk-adjusted performance
(Sharpe, Sortino, drawdown), not total return; a cash-capable strategy is expected
to underperform buy-and-hold in absolute return.

## Data

Two datasets span 2016-03-01 to 2025-12-31 (2,475 trading days): daily OHLCV for
14 ETFs (`data/etf_ohlcv.csv`; SPY, QQQ, RSP, and 11 GICS sector ETFs) and the
point-in-time constituent membership of the S&P 500/400/600 (the S&P 1500; placed
in `data/raw/` or via `$PIT_RAW_DIR`, git-ignored). All returns use
dividend-adjusted closes. Index membership is point-in-time and lagged one month,
removing survivorship bias. Constituent returns are split-neutralized via the
share-count change and winsorized at [-50%, +75%]; the cross-section is validated
to 0.997 correlation with SPY. After warm-up, the usable out-of-sample window
begins in 2020 and contains two stress episodes (2020, 2022), which limits
statistical power.

## Methodology

- **Absorption Ratio** (`src/models/absorption_ratio.py`): the share of total
  cross-sectional variance explained by the top 5 principal components over a
  252-day rolling covariance, expressed as a standardized shift versus a trailing
  252-day mean. Estimated on 25 GICS industry-group portfolios reconstructed
  point-in-time, because the number of raw constituents exceeds the covariance
  window and renders the sample covariance singular.
- **Statistical Jump Model** (`src/models/jump_model.py`): two-state k-means on
  four causal SPY-return features (EWM downside deviation, short and long EWM
  Sortino, EWM trend) augmented with a jump penalty that enforces regime
  persistence; fit by coordinate descent, with out-of-sample states inferred by a
  causal forward filter. The penalty is selected per training window by embargoed
  time-series cross-validation.
- **Combined model** (`src/models/fusion.py`), the required deliverable: the jump
  model re-fit on the union of the return and Absorption-Ratio features. The
  inferred bear state maps to cash (0) and all other states to full investment (1).

## Preventing look-ahead bias

The feature scaler, cluster centroids, bull/bear label, and penalty are estimated
on training data only. Features are causal by construction; out-of-sample regimes
use a forward filter refit at annual boundaries; positions are lagged one day
before contact with returns. A deliberate leakage variant was used as a diagnostic
to confirm the harness detects look-ahead. Every parameter is defined in
`config/default.yaml`.

## Results

Out-of-sample 2020-2025, locked configuration (CV-selected penalty, t+1 execution,
2 bps cost). The final column is an extension beyond the brief (see below).

| Strategy | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar |
|----------|------|-----|--------|---------|-------|--------|
| Buy & hold SPY | 15.0% | 20.7% | 0.78 | 0.96 | -33.7% | 0.45 |
| Combined (AR + JM) — deliverable | 7.1% | 12.3% | 0.62 | 0.61 | -21.8% | 0.33 |
| Jump-model only | 6.5% | 10.4% | 0.65 | 0.61 | -17.6% | 0.37 |
| JM + vol-target + cash — extension | 7.2% | 8.2% | 0.89 | 0.87 | -10.6% | 0.68 |

The combined model reduced volatility and maximum drawdown relative to
buy-and-hold but did not improve risk-adjusted return (Sharpe 0.62 vs 0.78;
difference -0.16, 95% bootstrap CI [-0.73, +0.43], P(win) 29%). The jump model
alone was comparable or marginally stronger (Sharpe 0.65) at lower exposure,
indicating the Absorption Ratio did not add measurable value within this sample.
No Sharpe difference versus buy-and-hold is statistically significant. The robust,
repeatable property is drawdown reduction, not a Sharpe edge.

## Extension beyond the brief

The final results column relaxes two assumptions of the mandate and is reported
separately. Volatility targeting replaces the binary position with continuous
sizing equal to `target_vol / trailing_20d_realized_vol`, capped at 1 (no
leverage), relaxing the binary allocation; idle capital earns the three-month
Treasury-bill rate, relaxing the zero-cash-yield assumption (the rate schedule is
an external assumption). Each lever is a pure overlay in `src/levers/`, priced
through the same no-look-ahead engine. This configuration raised the Sharpe ratio
to 0.89 and reduced maximum drawdown to -10.6%. Applying the same overlays to the
combined model produced a comparable Sharpe (0.86), reinforcing that the
Absorption Ratio remains close to neutral once volatility targeting performs the
risk-scaling.

## Stress tests

The extension configuration was perturbed one dimension at a time
(`scripts/stress_test.py`). Baseline Sharpe 0.89 vs buy-hold 0.78.

| Dimension | Range | Sharpe | Observation |
|-----------|-------|--------|-------------|
| Transaction cost | 0 - 20 bps | 0.90 - 0.78 | survives realistic cost; equals buy-hold only at 20 bps |
| Volatility window | 10 - 60 d | 0.79 - 0.89 | stable; 20d strongest |
| Target volatility | 8 - 20% | 0.90 - 0.78 | lower target raises Sharpe (de-risking), monotonic |
| Cash-rate assumption | 0% - double | 0.74 - 1.04 | ~0.15 Sharpe is cash interest; rate-regime dependent |
| Execution lag | t+1 / t+2 | 0.89 / 0.92 | robust to timing |
| k-means seeds | 3 sets | 0.85 - 0.89 | stable |

By sub-period, the configuration outperformed buy-and-hold in both stress
episodes (2020 Sharpe 1.17 vs 0.96; 2022 -0.47 vs -0.71) and underperformed in the
2023-2025 advance (0.93 vs 1.43), consistent with a risk-managed profile. The
Sharpe advantage over buy-and-hold remains statistically insignificant (+0.11,
95% CI [-0.64, +0.85]).

## Reproduction

```bash
pip install -r requirements.txt
export PIT_RAW_DIR=/path/to/raw/pit/files     # build_dataset only
python scripts/build_dataset.py               # raw S&P 1500 -> industry-group returns
python scripts/run_walkforward.py             # walk-forward -> OOS returns + positions
python scripts/evaluate.py                     # metrics, bootstrap CI, equity figure
python scripts/run_levers.py                    # extension overlays + comparison
python scripts/stress_test.py                   # stress grid for the extension config
python -m pytest tests/                          # AR, JM, features, no-look-ahead, backtest, levers
```

Deterministic given the seed in `config/default.yaml`.

## Repository structure

```
config/default.yaml     all parameters (no hyperparameter is buried in code)
src/
  config.py             typed config loader
  data/                 market data + point-in-time constituents
  features/             jump features + feature assembly
  models/               absorption_ratio, jump_model, fusion (combined model)
  backtest/             walk-forward engine + pnl
  levers/               extension overlays: vol_target, cash_yield, multi_asset
  evaluation/           metrics, bootstrap, plots
  utils/                logging, determinism
scripts/                build_dataset, run_walkforward, evaluate, run_levers, stress_test
tests/                  AR, jump model, features, no-look-ahead, backtest, levers
results/                metrics, returns, figures (regenerated); results/levers for extensions
docs/                   methodology notes
```

## Limitations

- **Short sample.** The out-of-sample window contains two stress episodes; no
  reported Sharpe difference is statistically significant, and point estimates are
  sensitive to configuration.
- **Penalty selection.** Cross-validation does not reliably tune the jump penalty
  on this sample length.
- **Marginal Absorption-Ratio value.** The Absorption Ratio adds little over the
  jump model alone within this sample.
- **Generalization.** The drawdown-reduction property held on a SPY-plus-QQQ
  application but did not generalize to the equal-weight index RSP.
- **Backfilled classifications.** Index membership is point-in-time, but GICS
  labels are backfilled in the source data, a disclosed second-order impurity.
- **Extension dependence.** The extension's cash-yield benefit depends on the
  prevailing rate environment.
