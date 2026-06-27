"""No-look-ahead walk-forward engine.

Guarantees: the feature scaler, jump-model centroids, bull/bear label and the
penalty are all fit on TRAIN data only; out-of-sample regimes use the causal
forward filter; positions are lagged before they touch returns (t -> t+delay).
"""
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd

from models.jump_model import fit as jm_fit, forward_filter, bear_state


def pnl(positions: pd.Series, asset_returns: pd.Series, delay: int, cost_bps: float) -> pd.Series:
    """Strategy returns over the position span: a position decided at close t earns
    from t+delay; transaction costs are charged on turnover."""
    held = positions.shift(delay).reindex(asset_returns.index)
    turnover = held.diff().abs()
    out = held * asset_returns - (cost_bps / 1e4) * turnover
    return out.loc[positions.index.min():].fillna(0.0)


def _sortino(ret: pd.Series, trading_days: int) -> float:
    ret = ret.dropna()
    if len(ret) < 20 or ret.std() == 0:
        return -1e18
    dn = ret[ret < 0].std()
    return -1e18 if (dn == 0 or np.isnan(dn)) else ret.mean() / dn * np.sqrt(trading_days)


def _fit_predict(train_feats: pd.DataFrame, all_feats: pd.DataFrame, returns: pd.Series,
                 lam: float, n_states: int, seeds: Sequence[int]) -> Tuple[np.ndarray, int]:
    mu, sd = train_feats.mean(), train_feats.std() + 1e-12
    states_tr, centroids = jm_fit(((train_feats - mu) / sd).values, lam, n_states, seeds)
    bear = bear_state(states_tr, returns.reindex(train_feats.index).values)
    states = forward_filter(((all_feats - mu) / sd).values, centroids, lam)
    return states, bear


def select_lambda(feats: pd.DataFrame, returns: pd.Series, asset_returns: pd.Series, *,
                  grid: Sequence[int], n_folds: int, val_days: int, embargo: int,
                  n_states: int, seeds: Sequence[int], trading_days: int) -> int:
    """Embargoed, multi-fold expanding CV inside the training window (no look-ahead)."""
    n = len(feats)
    best = (grid[len(grid) // 2], -np.inf)
    for lam in grid:
        scores = []
        for k in range(n_folds):
            ve, vs = n - k * val_days, n - k * val_days - val_days
            te = vs - embargo                       # purge gap (features have long memory)
            if te < 252:
                break
            states, bear = _fit_predict(feats.iloc[:te], feats.iloc[:ve], returns, lam, n_states, seeds)
            pos = pd.Series((states[vs:ve] != bear).astype(float), index=feats.index[vs:ve])
            r = pnl(pos, asset_returns, delay=1, cost_bps=0.0).reindex(pos.index).dropna()
            scores.append(_sortino(r, trading_days))
        if scores and np.mean(scores) > best[1]:
            best = (lam, float(np.mean(scores)))
    return best[0]


def walk_forward(feats: pd.DataFrame, returns: pd.Series, asset_returns: pd.Series, *,
                 oos_start: str, first_train_year: int, last_year: int, n_states: int,
                 seeds: Sequence[int], lam_grid: Sequence[int], cv_folds: int, cv_val_days: int,
                 cv_embargo: int, trading_days: int, fixed_lambda: int | None = None
                 ) -> Tuple[pd.Series, Dict[int, int]]:
    """Expanding walk-forward, annual refit. Returns (OOS positions, lambda by year)."""
    pos = pd.Series(index=feats.index, dtype=float)
    chosen: Dict[int, int] = {}
    for year in range(first_train_year + 1, last_year + 1):
        tr_end = pd.Timestamp(f"{year - 1}-12-31")
        train = feats[feats.index <= tr_end]
        if len(train) < 300:
            continue
        lam = (fixed_lambda if fixed_lambda is not None else
               select_lambda(train, returns, asset_returns, grid=lam_grid, n_folds=cv_folds,
                             val_days=cv_val_days, embargo=cv_embargo, n_states=n_states,
                             seeds=seeds, trading_days=trading_days))
        chosen[year] = lam
        upto = feats[feats.index <= pd.Timestamp(f"{year}-12-31")]
        states, bear = _fit_predict(train, upto, returns, lam, n_states, seeds)
        mask = np.asarray((upto.index > tr_end) & (upto.index <= pd.Timestamp(f"{year}-12-31")))
        pos.loc[upto.index[mask]] = (states[mask] != bear).astype(float)
    return pos.loc[oos_start:].dropna(), chosen
