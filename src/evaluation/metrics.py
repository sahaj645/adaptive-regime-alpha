"""Performance metrics and significance testing."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd


def performance(returns: pd.Series, positions: Optional[pd.Series] = None,
                trading_days: int = 252) -> Dict[str, float]:
    r = returns.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / trading_days
    downside = r[r < 0].std() * np.sqrt(trading_days)
    mdd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    out = {
        "CAGR": cagr,
        "AnnReturn": float(r.mean() * trading_days),
        "Vol": float(r.std() * np.sqrt(trading_days)),
        "Sharpe": float(r.mean() / r.std() * np.sqrt(trading_days)) if r.std() else np.nan,
        "Sortino": float(r.mean() * trading_days / downside) if downside else np.nan,
        "MaxDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd else np.nan,
    }
    if positions is not None:
        runs = (positions.diff().fillna(0) != 0).cumsum()
        out["Exposure"] = float(positions.mean())
        out["Switches"] = int((positions.diff() != 0).sum())
        out["AvgDuration"] = float(positions.groupby(runs).size().mean())
    return out


def bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, *, block_size: int, n: int,
                          trading_days: int, seed: int) -> np.ndarray:
    """Paired stationary-block bootstrap of Sharpe(a) - Sharpe(b)."""
    a = a.dropna(); b = b.reindex(a.index).dropna(); a = a.reindex(b.index)
    A, B = a.values, b.values
    T = len(A); rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_size))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, max(1, T - block_size), n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:T]
        aa, bb = A[idx], B[idx]
        sa = aa.mean() / aa.std() * np.sqrt(trading_days) if aa.std() > 0 else 0.0
        sb = bb.mean() / bb.std() * np.sqrt(trading_days) if bb.std() > 0 else 0.0
        out[i] = sa - sb
    return out
