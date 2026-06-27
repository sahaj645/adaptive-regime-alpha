"""
lambda_cv.py -- robust jump-penalty selection.

Replaces the single-fold inner CV (which mis-tuned lambda on this short sample)
with: a PRE-SPECIFIED log-spaced grid, k expanding validation folds, and an
embargo gap between inner-train and validation so the long-memory features
(120d EWMA, 252d AR) do not leak across the boundary. All inside the training
window -> no look-ahead. The grid is fixed a priori, not chosen from OOS.
"""
import numpy as np
import pandas as pd
import core, engine

GRID = [5, 10, 20, 40, 80, 160]          # a-priori log-spaced persistence grid


def _sortino(ret):
    ret = ret.dropna()
    if len(ret) < 20 or ret.std() == 0:
        return -1e9
    dn = ret[ret < 0].std()
    return -1e9 if (dn == 0 or np.isnan(dn)) else ret.mean() / dn * np.sqrt(252)


def select_lambda(feat_tr, r, asset_ret, grid=GRID, n_folds=3, val=126, embargo=21, seeds=range(2)):
    n = len(feat_tr)
    best = (grid[len(grid) // 2], -1e18)
    for lam in grid:
        scores = []
        for k in range(n_folds):
            ve = n - k * val
            vs = ve - val
            te = vs - embargo                       # purge: train ends before validation
            if te < 252:
                break
            inner = feat_tr.iloc[:te]
            st, bear = engine._fit_predict(inner, feat_tr.iloc[:ve], r, lam, seeds)
            pos = pd.Series((st[vs:ve] != bear).astype(float), index=feat_tr.index[vs:ve])
            scores.append(_sortino(engine.pnl(pos, asset_ret, oos_start=pos.index[0]).reindex(pos.index).dropna()))
        if scores and np.mean(scores) > best[1]:
            best = (lam, np.mean(scores))
    return best[0]


def walk_forward(feat, r, asset_ret, oos_start="2020-01-01", **kw):
    pos, chosen = pd.Series(index=feat.index, dtype=float), {}
    for y in range(2020, 2026):
        tr_end = pd.Timestamp(f"{y-1}-12-31"); ftr = feat[feat.index <= tr_end]
        if len(ftr) < 300:
            continue
        lam = select_lambda(ftr, r, asset_ret, **kw); chosen[y] = lam
        fto = feat[feat.index <= pd.Timestamp(f"{y}-12-31")]
        st, bear = engine._fit_predict(ftr, fto, r, lam, seeds=range(3))
        m = np.asarray((fto.index > tr_end) & (fto.index <= pd.Timestamp(f"{y}-12-31")))
        pos.loc[fto.index[m]] = (st[m] != bear).astype(float)
    return pos.loc[oos_start:].dropna(), chosen


if __name__ == "__main__":
    import features
    _, spy, sec = core.load(); spy_ret = spy.pct_change()
    f_jm, r, _ = features.build("jm")
    f_fp, _, _ = features.build("fused_pit")
    bh = spy_ret.loc["2020-01-01":]
    print("baselines: Buy&hold Sharpe %.2f\n" % engine.metrics(bh)["Sharpe"])
    print(f"{'config':12} {'old-CV':>7} {'new-CV':>7} {'newMaxDD':>9}   new lambda by year")
    for name, feat in [("JM-only", f_jm), ("Fused-PIT", f_fp)]:
        sh_old = engine.metrics(engine.pnl(engine.walk_forward(feat, r, spy_ret, lam_mode="cv", seeds=range(3)), spy_ret))["Sharpe"]
        pos_new, chosen = walk_forward(feat, r, spy_ret)
        ret_new = engine.pnl(pos_new, spy_ret); m = engine.metrics(ret_new)
        print(f"{name:12} {sh_old:7.2f} {m['Sharpe']:7.2f} {m['MaxDD']*100:8.0f}%   {chosen}")
        if name == "Fused-PIT":
            d = engine.bootstrap_sharpe_diff(ret_new, bh, n=4000)
            print(f"             new Fused-PIT vs Buy&hold: dSharpe {m['Sharpe']-engine.metrics(bh)['Sharpe']:+.2f}  "
                  f"95%CI[{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}]  P(win){(d>0).mean()*100:.0f}%")
