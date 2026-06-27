"""Absorption Ratio (Kritzman, Li, Page & Rigobon, 2010). Causal: the value at
day t uses only the trailing covariance window ending at t."""
from __future__ import annotations
import numpy as np
import pandas as pd


def absorption_ratio(returns: pd.DataFrame, window: int, n_components: int,
                     shift_window: int, short_ma: int) -> pd.DataFrame:
    """Return columns [AR, dAR]: the absorption ratio and its standardized shift.

    AR_t = sum(top-n eigenvalues) / sum(all eigenvalues) of the trailing
    `window`-day covariance. dAR_t = (MA_short - MA_shift) / STD_shift, all
    trailing (Kritzman's 1-year standardized shift, decoupled from `window`).
    """
    rets = returns.dropna(how="all")
    values = rets.values
    ar = np.full(len(rets), np.nan)
    for i in range(window, len(rets) + 1):
        cov = np.cov(np.nan_to_num(values[i - window:i]), rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        ar[i - 1] = eig[:n_components].sum() / eig.sum()
    level = pd.Series(ar, index=rets.index, name="AR").dropna()
    shift = ((level.rolling(short_ma).mean() - level.rolling(shift_window).mean())
             / level.rolling(shift_window).std()).rename("dAR")
    return pd.concat([level, shift], axis=1).dropna()
