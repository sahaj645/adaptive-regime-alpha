# Regime-Switching SPY Overlay — Absorption Ratio × Statistical Jump Model

A no-look-ahead daily regime filter that toggles SPY exposure between fully invested and cash, combining a cross-sectional systemic-risk measure (Kritzman et al. Absorption Ratio) with a persistence-penalized regime classifier (Shu et al. Statistical Jump Model).

---

## Motivation

Equity drawdowns are not uniformly distributed in time; they cluster in turbulent regimes characterized by elevated volatility and rising cross-asset co-movement. If those regimes can be identified causally — using only information available at the close of each day — an investor can reduce exposure ahead of the worst realizations and limit downside.

This repository tests a specific hypothesis: that *combining* a measure of market fragility (how unified the cross-section is) with a measure of an asset's own trend-and-downside dynamics produces a better regime signal than either component alone. The two inputs are deliberately orthogonal — one is cross-sectional, one is time-series — which makes the combination worth testing and the ablation informative.

The research question is framed as a falsifiable one, and the result is reported as such, including where the hypothesis fails.

---

## Research Objective

The model infers a binary market regime at the close of day `t`:

```
regime(t) ∈ {favourable, unfavourable}  ->  position(t+1) ∈ {1, 0}
```

`position = 1` is fully invested in SPY; `position = 0` is cash earning zero. The regime inferred at the close of day `t` determines the position held on day `t+1`. There is no leverage and no shorting. The intended use is a tactical overlay that manages drawdown on a long SPY allocation, not an alpha source.

---

## Methodology

**Data.** Daily total-return series (dividend-adjusted close) for 14 ETFs — SPY, QQQ, RSP and the eleven GICS sector SPDRs — from 2016-03 to 2025-12. Sector ETFs are used as the cross-sectional universe; they are survivorship-clean by construction.

**Feature engineering.** Two causal feature blocks. The jump-model block is computed from SPY log returns: an EWM downside deviation (half-life 10), EWM Sortino ratios (half-lives 20 and 60), and an EWMA return (half-life 120). The absorption-ratio block is computed from the sector cross-section: a trailing 252-day covariance matrix is eigendecomposed, and the share of total variance explained by the top two eigenvectors (~N/5) gives the Absorption Ratio, together with its standardized 15-day-versus-1-year shift.

**Signal construction.** Features are standardized using training-window statistics only. The combined ("fused") configuration concatenates the four jump-model features with the two absorption-ratio features.

**Model.** The Statistical Jump Model is k-means clustering augmented with a jump penalty `λ` that charges every state transition, producing persistent, tradeable regimes rather than daily whipsaw. Centroids are fit on training data by coordinate descent (dynamic-programming state path plus centroid update). The penalty `λ` is selected by time-series cross-validation internal to the training window. Two states are used; the unfavourable state is labelled by lower realized mean return on the training set. Out-of-sample regimes are inferred with a causal forward filter (the forward pass of the dynamic program only), which never uses future observations.

**Portfolio construction.** The favourable state maps to full SPY exposure, the unfavourable state to cash. Positions are binary, unlevered, long-only.

**Backtest.** Expanding walk-forward with annual refitting. The regime inferred at the close of `t` is executed the following day; transaction costs of 2 bps are charged per position change; cash earns zero.

**Evaluation.** Risk and return metrics are compared against buy-and-hold SPY, with block-bootstrap confidence intervals on the Sharpe differential, an in-sample "leak" benchmark to quantify the look-ahead premium, and robustness sweeps over the penalty, random seeds, costs, execution lag, alternate assets (QQQ, RSP), and absorption-ratio windows.

---

## Repository Structure

```text
sarang_alpha_project/
├── data/
│   └── etf_ohlcv.csv          # daily OHLCV + adjusted close, 14 ETFs, 2016-2025
├── src/
│   ├── core.py                # causal building blocks: features, AR, jump model, forward filter
│   ├── engine.py              # reusable no-look-ahead walk-forward engine + metrics
│   ├── absorption_ratio.py    # Absorption Ratio construction and validation plot
│   ├── jump_model.py          # in-sample jump-model regime fit and diagnostics
│   ├── walk_forward.py        # out-of-sample backbone + look-ahead leak test
│   ├── stage7_horserace.py    # main OOS comparison: buy-hold / AR / JM / fused / gate
│   ├── stage8a_robustness.py  # penalty, seed, cost and execution-lag sensitivity
│   ├── stage8b_generalize.py  # generalization to QQQ/RSP and AR-window ablation
│   └── improvement_experiment.py  # disciplined tuning ledger vs the look-ahead oracle
├── outputs/                   # generated figures and metric tables
├── requirements.txt
└── README.md
```

`data/` holds the input price series. `src/` contains the library (`core.py`, `engine.py`) and the executable experiment scripts. `outputs/` is populated by the scripts with figures and CSV metric tables.

---

## Installation

```bash
git clone <repository-url>
cd sarang_alpha_project
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Running the Project

`core.py` and `engine.py` are libraries. The experiment scripts are entry points and write their figures and tables to `outputs/`:

```bash
python src/absorption_ratio.py       # build the Absorption Ratio and validate against drawdowns
python src/jump_model.py             # in-sample regime identification and diagnostics
python src/walk_forward.py           # no-look-ahead backbone + leak test
python src/stage7_horserace.py       # main out-of-sample comparison (all configurations)
python src/stage8a_robustness.py     # robustness sweeps
python src/stage8b_generalize.py     # cross-asset generalization and AR ablation
python src/improvement_experiment.py # disciplined tuning ledger
```

The pipeline is deterministic; repeated runs reproduce the reported figures exactly.

---

## Methodological Safeguards

**Look-ahead prevention.** Every fitted transform — the feature scaler, the jump-model centroids, the bull/bear labelling, and the penalty selection — is estimated on training data only. Out-of-sample regimes use a causal forward filter that depends solely on observations up to and including day `t`. Signals are lagged before they touch returns, so a regime inferred at the close of `t` is traded on `t+1`. A deliberate in-sample "cheat" benchmark is included to quantify the look-ahead premium a careless implementation would capture.

**Survivorship bias.** The cross-sectional universe uses sector ETFs, which do not suffer constituent survivorship bias. Point-in-time S&P 1500 membership is available as an extension for a constituent-level Absorption Ratio (see Future Work).

**Expanding walk-forward validation.** Models are refit annually on an expanding window; each out-of-sample year is predicted by a model that has seen only prior data. Hyperparameter selection occurs inside the training window via nested time-series cross-validation.

**Transaction costs.** A 2 bps charge is applied per position change. Turnover is low (~2 round-trips per year), and results are insensitive to costs up to 20 bps.

**Reproducibility and determinism.** Random initializations are seeded and the multi-start fit is selected by objective value, so the entire pipeline is deterministic. Dependencies are pinned in `requirements.txt`.

---

## Results

Out-of-sample, 2020-01 to 2025-12 (6.0 years). Strategy = jump-model regime filter with next-day execution and train-internal penalty selection. Cash earns 0%.

| Metric            | Strategy | Buy & Hold SPY |
| ----------------- | -------- | -------------- |
| CAGR              | 6.9%     | 15.0%          |
| Annual Return     | 7.1%     | 16.2%          |
| Annual Volatility | 9.9%     | 20.7%          |
| Sharpe            | 0.72     | 0.78           |
| Sortino           | 0.70     | 0.96           |
| Max Drawdown      | −12.4%   | −33.7%         |
| Exposure          | 57%      | 100%           |

The strategy does not improve risk-adjusted return over buy-and-hold on this sample: the Sharpe differential is −0.06 with a 95% block-bootstrap confidence interval of [−0.84, +0.65], i.e. statistically indistinguishable from zero. The robust and repeatable effect is risk reduction — roughly a 63% smaller maximum drawdown and a halving of volatility — achieved by holding cash ~43% of the time.

An ablation across feature fusion, signal gating, and standalone use found that the Absorption Ratio added no out-of-sample value in any configuration; the jump-model component carries the regime signal. Findings generalize to QQQ but not to the equal-weight RSP. Given the short sample (effectively two bear episodes) and the number of configurations examined, reported point estimates should be interpreted with multiple-testing deflation in mind.

---

## Future Work

A constituent-level Absorption Ratio built from point-in-time GICS industry-group portfolios (rather than 11 sector ETFs) would more closely follow Kritzman et al. and may improve the systemic-risk signal. The discrete jump model could be replaced with the continuous jump model and a probability threshold tuned in-sample. Formal overfitting controls — Deflated Sharpe Ratio, White's Reality Check, and the Probability of Backtest Overfitting — should accompany any forward claim. Extending the sample to a longer history with more independent regime cycles is necessary before the Sharpe differential can be assessed with adequate power.

---

## References

- Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2010). *Principal Components as a Measure of Systemic Risk.* The Journal of Portfolio Management, 37(4), 112–126.
- Shu, Y., Yu, C., & Mulvey, J. M. (2024). *Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach.* Journal of Asset Management. arXiv:2402.05272.
- Nystrup, P., Lindström, E., & Madsen, H. (2020). *Learning Hidden Markov Models with Persistent States by Penalizing Jumps.* Expert Systems with Applications, 150, 113307.
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* The Journal of Portfolio Management, 40(5), 94–107.

---

## License

Released under the MIT License.
