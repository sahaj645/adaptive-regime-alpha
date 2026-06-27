# Regime-Switching SPY Overlay

A no-look-ahead daily regime model that switches SPY exposure between fully
invested and cash, combining the Kritzman et al. Absorption Ratio
(cross-sectional fragility) with the Shu-Yu-Mulvey statistical jump model
(own-asset trend/downside regime). The fragility signal is built from the
point-in-time S&P 1500 constituents.

## Problem

Equity drawdowns cluster in turbulent, tightly-coupled regimes. If those regimes
can be identified causally - using only information available at the close of day
`t` - exposure can be cut for day `t+1`. The model outputs a binary regime that
maps to a SPY position (1 = invested, 0 = cash, cash earns 0); no leverage, no
shorting. The objective is risk-adjusted improvement, not beating SPY's return.

## Methodology

- **Absorption Ratio** (`src/models/absorption_ratio.py`): the share of total
  variance of a cross-section explained by its top principal components, over a
  trailing covariance window, plus a standardized 1-year shift. Built on 25 GICS
  industry-group portfolios reconstructed point-in-time from the S&P 1500.
- **Jump model** (`src/models/jump_model.py`): k-means on four return features
  plus a jump penalty for state persistence; fit by coordinate descent;
  out-of-sample inference via a causal forward filter.
- **Fusion** (`src/models/fusion.py`): the jump model fit on the union of the
  return features and the absorption-ratio features.
- **No look-ahead**: scaler, centroids, labels and the penalty are fit on TRAIN
  only; regimes are inferred causally; positions are traded `t+1`. Verified by a
  machine causality test.

## Data

`data/etf_ohlcv.csv` (SPY + sector ETFs, total-return adjusted) and the
point-in-time S&P 500/400/600 constituent files (placed in `data/raw/` or via
`$PIT_RAW_DIR`; git-ignored). Membership is monthly point-in-time (lagged one
month); returns are trading-calendar aligned, split-neutralized via the
share-count change, and winsorized. Construction is validated to 0.997
correlation with SPY.

## Pipeline

```
scripts/build_dataset.py     # raw S&P 1500 -> 25 industry-group return matrices
scripts/run_walkforward.py   # features -> walk-forward -> OOS returns (strategy + baselines)
scripts/evaluate.py          # metrics table, bootstrap CI, equity figure
```

Every parameter lives in `config/default.yaml`; no hyperparameter is buried in code.

## Results

Out-of-sample 2020-2025, canonical config (CV-selected penalty, t+1 execution, 2 bps):

| Strategy | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar |
|----------|------|-----|--------|---------|-------|--------|
| Buy & hold SPY | 15.0% | 20.7% | 0.78 | 0.96 | -33.7% | 0.45 |
| Jump-model only | 6.5% | 10.4% | 0.65 | 0.61 | -17.6% | 0.37 |
| Fused (constituent AR) | 7.1% | 12.3% | 0.62 | 0.61 | -21.8% | 0.33 |

The strategy roughly halves volatility and drawdown but **does not beat
buy-and-hold on a risk-adjusted basis** in this configuration (Sharpe difference
-0.16, 95% bootstrap CI [-0.73, +0.43], P(win) 29%). The Sharpe estimate is
unstable across reasonable choices (~0.5-0.9; see Known limitations). The robust,
repeatable property is drawdown reduction, not a Sharpe edge.

## Reproduction

```bash
pip install -r requirements.txt
export PIT_RAW_DIR=/path/to/raw/pit/files     # for build_dataset only
python scripts/build_dataset.py
python scripts/run_walkforward.py
python scripts/evaluate.py
python -m pytest tests/                        # or: see tests/
```

Deterministic given the seed in `config/default.yaml`.

## Repository structure

```
config/default.yaml         all parameters
src/
  config.py                 typed config loader
  data/                     market data + point-in-time constituents
  features/                 jump features + feature assembly
  models/                   absorption_ratio, jump_model, fusion
  backtest/                 walk-forward engine + pnl + lambda CV
  evaluation/               metrics, bootstrap, plots
  utils/                    logging, determinism
scripts/                    build_dataset, run_walkforward, evaluate
tests/                      AR, jump model, features, no-look-ahead, backtest
docs/                       data audit, methodology notes, decision walkthrough
results/                    metrics, returns, figures (regenerated)
```

## Known limitations

- **Short sample**: ~2 bear episodes (COVID, 2022). The Sharpe edge is not
  statistically distinguishable from buy-and-hold and the point estimate is
  unstable to penalty/CV choices.
- **Penalty selection**: cross-validation does not reliably tune the jump penalty
  on this sample (a robust embargoed CV did not fix it) - a data-length limit.
- **Static GICS classification**: index membership is point-in-time but GICS
  labels are backfilled in the source data (a disclosed second-order impurity).
- **Marginal AR value**: the absorption ratio adds little over the jump model
  alone, though the constituent version is robustly better than the ETF version.

## Future work

More regime cycles (longer history) are required before any edge is provable.
The largest unused levers are outside the current mandate: volatility targeting /
continuous sizing, earning the risk-free rate on cash, and a multi-asset overlay.
