"""Lever 1 - volatility targeting (continuous sizing).

Turns a binary regime signal (0/1) into a continuously-sized position:

    size_t = base_pos_t * clip( target_vol / realized_vol_t , 0 , cap )

`realized_vol_t` is a trailing estimate that uses only returns up to and
including day t (causal). The position is *decided* at close t; the pnl engine
lags it by `trade_delay` before it touches returns, so there is no look-ahead.
Because the size now moves every day, daily turnover rises - the same pnl engine
charges cost_bps on that turnover, so the cost of the lever is captured honestly.

This deliberately breaks the binary 0/1 brief (cap may exceed 1 = leverage); it
is reported as a mandate-relaxing lever, not as the core deliverable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def realized_vol(asset_returns: pd.Series, window: int, trading_days: int) -> pd.Series:
    """Trailing annualized realized volatility, causal (data up to and incl. t)."""
    return asset_returns.rolling(window, min_periods=window).std() * np.sqrt(trading_days)


def vol_target_position(base_pos: pd.Series, asset_returns: pd.Series, *,
                        target_vol: float, cap: float, window: int,
                        trading_days: int, floor_vol: float = 1e-4) -> pd.Series:
    """Continuously-sized position from a binary base signal.

    base_pos      : 0/1 regime series indexed at decision close t (the OOS span).
    asset_returns : full-history daily returns of the traded asset; the extra
                    history warms up the trailing-vol estimate before OOS start.
    """
    rv = realized_vol(asset_returns, window, trading_days)
    scale = (target_vol / rv.clip(lower=floor_vol)).clip(lower=0.0, upper=cap)
    pos = base_pos * scale.reindex(base_pos.index)
    return pos.dropna()
