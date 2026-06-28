"""Lever 3 - multi-asset regime overlay.

The same jump-model regime filter, fit independently per asset with the same
no-look-ahead walk-forward, is run across a basket (SPY/QQQ/sector ETFs). Each
asset is held only in its own favourable regime; the legs are combined
equal-weight. Diversifying the timing decision across imperfectly-correlated
assets is what lifts the basket Sharpe above any single name.

Returns OOS binary positions per asset; combination/pricing is done by the
caller so the same pnl + lever stack applies uniformly.
"""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
import pandas as pd

from data.market_data import asset_price
from features.jump_features import jump_features
from backtest.engine import walk_forward


def regime_position(cfg, ticker: str, seeds: Sequence[int],
                    fixed_lambda: int | None = None) -> pd.Series:
    """OOS binary JM position for one asset (1 = invested, 0 = cash)."""
    j, b, e = cfg.jump_model, cfg.backtest, cfg.evaluation
    price = asset_price(cfg.data.etf_file, ticker)
    feats, logret = jump_features(price, j.halflife_downside_dev, j.halflife_sortino_short,
                                  j.halflife_sortino_long, j.halflife_trend, j.burn_in)
    asset_ret = price.pct_change()
    pos, _ = walk_forward(feats, logret, asset_ret, oos_start=b.oos_start,
                          first_train_year=b.first_train_year, last_year=b.last_year,
                          n_states=j.n_states, seeds=seeds, lam_grid=j.lambda_grid,
                          cv_folds=j.cv_folds, cv_val_days=j.cv_val_days,
                          cv_embargo=j.cv_embargo, trading_days=e.trading_days,
                          fixed_lambda=fixed_lambda)
    return pos


def basket_positions(cfg, tickers: Sequence[str], seeds: Sequence[int],
                     fixed_lambda: int | None = None) -> Dict[str, pd.Series]:
    return {t: regime_position(cfg, t, seeds, fixed_lambda) for t in tickers}
