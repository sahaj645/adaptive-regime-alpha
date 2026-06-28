"""Lever 2 - earn the risk-free rate on idle cash.

The mandate sets cash yield to zero, so on the ~40-50% of days the strategy is
out of the market it earns nothing. Here the uninvested fraction (1 - held)
earns the daily risk-free rate; if the position is levered (held > 1) the
borrowed fraction (negative) PAYS that same rate as financing. This is the
honest accounting for a cash-or-leverage overlay.

    full_t = held_t * asset_ret_t + (1 - held_t) * rf_daily_t - cost_t

`held` is the position lagged by `delay` (same convention as the pnl engine), so
no look-ahead is introduced. rf comes from a piecewise-constant annual schedule.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def daily_rf(index: pd.DatetimeIndex, rf_by_year: dict, trading_days: int) -> pd.Series:
    """Per-day risk-free return from an annual schedule (compounded to daily)."""
    ann = pd.Series(index.year, index=index).map(
        {int(k): float(v) for k, v in rf_by_year.items()})
    ann = ann.ffill().bfill()                       # years outside the map -> nearest
    return (1.0 + ann) ** (1.0 / trading_days) - 1.0


def pnl_with_cash(positions: pd.Series, asset_returns: pd.Series, rf_by_year: dict, *,
                  delay: int, cost_bps: float, trading_days: int) -> pd.Series:
    """Strategy returns including risk-free carry on the uninvested fraction."""
    held = positions.shift(delay).reindex(asset_returns.index)
    turnover = held.diff().abs()
    rf = daily_rf(asset_returns.index, rf_by_year, trading_days)
    out = held * asset_returns + (1.0 - held) * rf - (cost_bps / 1e4) * turnover
    return out.loc[positions.index.min():].fillna(0.0)
