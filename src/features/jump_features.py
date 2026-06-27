"""Jump-model features: the four causal return statistics of Shu, Yu & Mulvey."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd


def jump_features(price: pd.Series, hl_downside_dev: int, hl_sortino_short: int,
                  hl_sortino_long: int, hl_trend: int, burn_in: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (features, log_returns). Each feature at t uses only data <= t."""
    r = np.log(price / price.shift(1)).dropna()
    downside = r.clip(upper=0.0)
    ewm_dd = lambda x, hl: np.sqrt((x ** 2).ewm(halflife=hl).mean())
    feats = pd.DataFrame({
        "downside_dev": ewm_dd(downside, hl_downside_dev),
        "sortino_short": r.ewm(halflife=hl_sortino_short).mean() / (ewm_dd(downside, hl_sortino_short) + 1e-8),
        "sortino_long": r.ewm(halflife=hl_sortino_long).mean() / (ewm_dd(downside, hl_sortino_long) + 1e-8),
        "trend": r.ewm(halflife=hl_trend).mean(),
    }).dropna().iloc[burn_in:]
    return feats, r
