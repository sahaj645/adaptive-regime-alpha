"""
Disciplined improvement ledger (JM-only on SPY, OOS 2020-2025).

Rule of the game: change ONE thing per row, justify it a priori, tune any
hyperparameter on TRAIN ONLY, and read OOS once. Never select a change because
it raised OOS Sharpe — that is fitting the test set. The 'oracle' row shows what
that cheat would falsely claim.

  A  as-submitted        : shift(1) execution, single-fold CV, grid {10,30,50,80}
  B  + strict next-day    : shift(2) execution            [CORRECTNESS — expect Sharpe down]
  C  + wider lambda search: shift(2), grid {5,10,20,30,50,80}  [a-priori: let train see low lambda]
  D  + 3-fold robust CV   : shift(2), 3 expanding folds   [a-priori: less noisy lambda pick]
  E  oracle lambda        : shift(2), lambda chosen to MAX OOS Sharpe  [LOOK-AHEAD, not real]
"""
import numpy as np, pandas as pd
import core, engine

df, spy, sec = core.load()
spy_ret = spy.pct_change()
feat, r = engine.jm_features_from_px(spy); feat = feat.iloc[150:]
OOS = "2020-01-01"
bh = spy_ret.loc[OOS:]


def sortino(ret):
    ret = ret.dropna()
    if len(ret) < 20 or ret.std() == 0: return -1e9
    dn = ret[ret < 0].std()
    return -1e9 if (dn == 0 or np.isnan(dn)) else ret.mean() / dn * np.sqrt(252)


def pick_lambda(feat_tr, grid, folds, seeds=range(2)):
    """Tune lambda by expanding-window CV INSIDE the training set only."""
    n = len(feat_tr)
    if n < 400: return 50
    best = (50, -1e18)
    for lam in grid:
        scs = []
        for fexp in range(folds):
            ve = n - fexp * 252; vs = ve - 252
            if vs < 252: break
            inner = feat_tr.iloc[:vs]
            st, bear = engine._fit_predict(inner, feat_tr.iloc[:ve], r, lam, seeds)
            validx = feat_tr.index[vs:ve]
            pos = pd.Series((st[vs:ve] != bear).astype(float), index=validx)
            scs.append(sortino(engine.pnl(pos, spy_ret, oos_start=pos.index[0]).reindex(validx).dropna()))
        if scs and np.mean(scs) > best[1]: best = (lam, np.mean(scs))
    return best[0]


def wf(feat, grid, folds, seeds=range(3)):
    pos = pd.Series(index=feat.index, dtype=float)
    for y in range(2020, 2026):
        tr_end = pd.Timestamp(f"{y-1}-12-31"); ftr = feat[feat.index <= tr_end]
        if len(ftr) < 300: continue
        lam = pick_lambda(ftr, grid, folds)
        fto = feat[feat.index <= pd.Timestamp(f"{y}-12-31")]
        st, bear = engine._fit_predict(ftr, fto, r, lam, seeds)
        m = np.asarray((fto.index > tr_end) & (fto.index <= pd.Timestamp(f"{y}-12-31")))
        pos.loc[fto.index[m]] = (st[m] != bear).astype(float)
    return pos.loc[OOS:].dropna()


def show(name, pos, delay):
    m = engine.metrics(engine.pnl(pos, spy_ret, delay=delay))
    print(f"  {name:38} Sharpe {m['Sharpe']:.2f}   Sortino {m['Sortino']:.2f}   "
          f"MaxDD {m['MaxDD']*100:4.0f}%   inv {pos.mean()*100:3.0f}%   trades {int((pos.diff()!=0).sum())}")
    return m['Sharpe']

print(f"Buy & hold SPY:  Sharpe {engine.metrics(bh)['Sharpe']:.2f}  "
      f"Sortino {engine.metrics(bh)['Sortino']:.2f}  MaxDD {engine.metrics(bh)['MaxDD']*100:.0f}%\n")

pos_base = wf(feat, [10, 30, 50, 80], 1)
pos_wide = wf(feat, [5, 10, 20, 30, 50, 80], 1)
pos_k3   = wf(feat, [5, 10, 20, 30, 50, 80], 3)

print("LEDGER (each row changes ONE thing):")
show("A as-submitted (shift1)",      pos_base, 1)
show("B +strict next-day exec",      pos_base, 2)
show("C +wider lambda search",       pos_wide, 2)
show("D +3-fold robust CV",          pos_k3,   2)

# oracle: pick lambda that maximises OOS Sharpe (look-ahead!)
orc = None
for lam in [5, 10, 20, 30, 50, 80]:
    p = engine.walk_forward(feat, r, spy_ret, lam_mode=lam, seeds=range(3))
    s = engine.metrics(engine.pnl(p, spy_ret, delay=2))['Sharpe']
    if orc is None or s > orc[1]: orc = (lam, s)
print(f"  {'E ORACLE lambda='+str(orc[0])+' (LOOK-AHEAD)':38} Sharpe {orc[1]:.2f}   <-- not achievable live")

# significance of the honest variant D vs buy-hold
retD = engine.pnl(pos_k3, spy_ret, delay=2)
d = engine.bootstrap_sharpe_diff(retD, bh, n=3000)
print(f"\nHonest (D) vs B&H Sharpe diff {engine.metrics(retD)['Sharpe']-engine.metrics(bh)['Sharpe']:+.2f}  "
      f"95% CI [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}]  P(win)={(d>0).mean()*100:.0f}%")
print(f"# legit configs tried this session ~ {len([10,30,50,80])+len([5,10,20,30,50,80])*2}+ ; deflate accordingly.")
