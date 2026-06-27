"""
Stage 8b — external validity.
  (5) generalization: run the SAME JM-only pipeline on QQQ and RSP (trade those)
  (6) does the absorption ratio EVER help? re-test the fused model across AR
      windows {126,252,504} vs JM-only on the identical window.
"""
import numpy as np, pandas as pd
import core, engine

df, spy, sec = core.load()
spy_ret = spy.pct_change()

# (5) generalization to QQQ / RSP --------------------------------------------
print("[5] generalization (JM-only, CV-tuned, trade the asset itself):")
print(f"  {'asset':5} {'strat Sharpe':>12} {'B&H Sharpe':>11} {'strat MaxDD':>12} {'B&H MaxDD':>10}")
for tk in ["SPY", "QQQ", "RSP"]:
    px   = df[df.Ticker == tk].set_index("Date")["Adj_Close"].sort_index()
    aret = px.pct_change()
    f, rr = engine.jm_features_from_px(px); f = f.iloc[150:]
    pos = engine.walk_forward(f, rr, aret, lam_mode="cv", seeds=range(3))
    ms = engine.metrics(engine.pnl(pos, aret)); mb = engine.metrics(aret.loc[engine.OOS_START:])
    print(f"  {tk:5} {ms['Sharpe']:12.2f} {mb['Sharpe']:11.2f} "
          f"{ms['MaxDD']*100:11.0f}% {mb['MaxDD']*100:9.0f}%")

# (6) does AR ever help? fused vs JM-only across AR windows -------------------
print("\n[6] fused (JM+AR) vs JM-only, across AR windows (lambda=30 fixed):")
feat_jm_full, r = engine.jm_features_from_px(spy)
for w in [126, 252, 504]:
    arf = core.ar_features(sec, window=w)
    common = feat_jm_full.index.intersection(arf.index)
    fj = feat_jm_full.loc[common]
    ff = pd.concat([fj, arf.loc[common]], axis=1)
    pj = engine.walk_forward(fj, r, spy_ret, lam_mode=30, seeds=range(3))
    pf = engine.walk_forward(ff, r, spy_ret, lam_mode=30, seeds=range(3))
    sj = engine.metrics(engine.pnl(pj, spy_ret))['Sharpe']
    sf = engine.metrics(engine.pnl(pf, spy_ret))['Sharpe']
    print(f"  AR window {w:3d}d : JM-only {sj:.2f}   fused {sf:.2f}   "
          f"{'fused better' if sf > sj else 'AR did NOT help'}")
