"""
engine.py — reusable no-look-ahead walk-forward engine (shared by Stage 7 & 8).
Same logic as walk_forward.py / stage7, parameterised so we can vary the asset,
lambda, seeds, cost and trade delay for robustness testing.
"""
import numpy as np
import pandas as pd
import core

OOS_START, LAM_GRID = "2020-01-01", [10, 30, 50, 80]


def jm_features_from_px(px):
    """4 jump-model features from any price series (causal)."""
    r  = np.log(px / px.shift(1)).dropna()
    dr = r.clip(upper=0.0)
    ewdd = lambda x, hl: np.sqrt((x ** 2).ewm(halflife=hl).mean())
    f = pd.DataFrame({
        "dd10":      ewdd(dr, 10),
        "sortino20": r.ewm(halflife=20).mean() / (ewdd(dr, 20) + 1e-8),
        "sortino60": r.ewm(halflife=60).mean() / (ewdd(dr, 60) + 1e-8),
        "ewma120":   r.ewm(halflife=120).mean(),
    }).dropna()
    return f, r


def metrics(ret):
    ret = ret.dropna(); eq = (1 + ret).cumprod(); yrs = len(ret) / 252
    vol = ret.std() * np.sqrt(252); dn = ret[ret < 0].std() * np.sqrt(252)
    return dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, Vol=vol,
                Sharpe=ret.mean() / ret.std() * np.sqrt(252) if ret.std() else np.nan,
                Sortino=ret.mean() * 252 / dn if dn else np.nan,
                MaxDD=(eq / eq.cummax() - 1).min())


def _sortino(ret):
    ret = ret.dropna()
    if len(ret) < 20 or ret.std() == 0: return -1e9
    dn = ret[ret < 0].std()
    return -1e9 if (dn == 0 or np.isnan(dn)) else ret.mean() / dn * np.sqrt(252)


def _fit_predict(feat_tr, feat_to, r_lab, lam, seeds):
    mu_, sd_ = feat_tr.mean(), feat_tr.std() + 1e-12
    s_tr, cent = core.fit_jump(((feat_tr - mu_) / sd_).values, lam, seeds=seeds)
    bear = core.label_bear(s_tr, r_lab.reindex(feat_tr.index).values)
    states = core.forward_filter(((feat_to - mu_) / sd_).values, cent, lam)
    return states, bear


def pnl(pos, asset_ret, cost=0.0002, delay=1, oos_start=OOS_START):
    held = pos.shift(delay).reindex(asset_ret.index).fillna(0.0)
    turn = held.diff().abs().fillna(0.0)
    return (held * asset_ret - cost * turn).loc[oos_start:]


def _select_lambda(feat_tr, r_lab, asset_ret, seeds, grid=LAM_GRID):
    if len(feat_tr) < 400: return 50
    cut = feat_tr.index[-252]; inner = feat_tr[feat_tr.index < cut]
    best = (50, -1e18)
    for lam in grid:
        st, bear = _fit_predict(inner, feat_tr, r_lab, lam, seeds)
        m = np.asarray(feat_tr.index >= cut)
        pos = pd.Series((st[m] != bear).astype(float), index=feat_tr.index[m])
        sc = _sortino(pnl(pos, asset_ret, oos_start=pos.index[0]).reindex(pos.index).dropna())
        if sc > best[1]: best = (lam, sc)
    return best[0]


def walk_forward(feat, r_lab, asset_ret, lam_mode="cv", seeds=range(6), oos_start=OOS_START):
    """lam_mode = 'cv' (tune per refit) or a fixed number. Returns OOS position series."""
    pos = pd.Series(index=feat.index, dtype=float)
    for y in range(2020, 2026):
        tr_end = pd.Timestamp(f"{y-1}-12-31"); feat_tr = feat[feat.index <= tr_end]
        if len(feat_tr) < 300: continue
        lam = _select_lambda(feat_tr, r_lab, asset_ret, seeds) if lam_mode == "cv" else float(lam_mode)
        feat_to = feat[feat.index <= pd.Timestamp(f"{y}-12-31")]
        st, bear = _fit_predict(feat_tr, feat_to, r_lab, lam, seeds)
        m = np.asarray((feat_to.index > tr_end) & (feat_to.index <= pd.Timestamp(f"{y}-12-31")))
        pos.loc[feat_to.index[m]] = (st[m] != bear).astype(float)
    return pos.loc[oos_start:].dropna()


def bootstrap_sharpe_diff(ret_a, ret_b, block=21, n=2000, seed=0):
    """Paired stationary-block bootstrap of Sharpe(a) - Sharpe(b)."""
    a = ret_a.dropna(); b = ret_b.reindex(a.index).dropna(); a = a.reindex(b.index)
    A, B = a.values, b.values; T = len(A); rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block)); out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, max(1, T - block), nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        aa, bb = A[idx], B[idx]
        sa = aa.mean() / aa.std() * np.sqrt(252) if aa.std() > 0 else 0.0
        sb = bb.mean() / bb.std() * np.sqrt(252) if bb.std() > 0 else 0.0
        out[i] = sa - sb
    return out
