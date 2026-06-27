"""
Stage 8a — robustness of the JM-only winner on SPY.
  (1) lambda sensitivity (fixed lambda, no tuning) — plateau or knife-edge?
  (2) seed sensitivity (k-means init)
  (3) transaction-cost and trade-delay sensitivity
  (4) statistical significance: block-bootstrap CI on Sharpe(JM) - Sharpe(B&H)
Fixed lambda is used deliberately here so we can see sensitivity directly;
the CV-tuned version is the Stage-7 result.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import core, engine

OUT = Path(__file__).resolve().parents[1] / "outputs"
df, spy, sec = core.load()
spy_ret = spy.pct_change()
feat, r = engine.jm_features_from_px(spy); feat = feat.iloc[150:]
bh = spy_ret.loc[engine.OOS_START:]
print("Buy&hold:", {k: round(v, 3) for k, v in engine.metrics(bh).items()})

# (1) lambda sensitivity ------------------------------------------------------
print("\n[1] lambda sensitivity (fixed lambda):")
lam_rows = {}
for lam in [10, 20, 30, 50, 70, 100]:
    pos = engine.walk_forward(feat, r, spy_ret, lam_mode=lam, seeds=range(3))
    m = engine.metrics(engine.pnl(pos, spy_ret)); lam_rows[lam] = m
    print(f"  lam={lam:3d}  Sharpe {m['Sharpe']:.2f}  Sortino {m['Sortino']:.2f}  "
          f"MaxDD {m['MaxDD']*100:.0f}%  invested {pos.mean()*100:.0f}%")

# (2) seed sensitivity --------------------------------------------------------
print("\n[2] seed sensitivity (lambda=50, single seed):")
sh = []
for sd in range(6):
    pos = engine.walk_forward(feat, r, spy_ret, lam_mode=50, seeds=[sd])
    s = engine.metrics(engine.pnl(pos, spy_ret))['Sharpe']; sh.append(s)
print(f"  Sharpe across seeds: {[round(x,2) for x in sh]}  (mean {np.mean(sh):.2f}, sd {np.std(sh):.02f})")

# (3) cost & delay (reuse one position series) --------------------------------
pos50 = engine.walk_forward(feat, r, spy_ret, lam_mode=50, seeds=range(3))
print("\n[3] cost sensitivity (lambda=50):")
for c in [0, 0.0002, 0.0005, 0.001, 0.002]:
    m = engine.metrics(engine.pnl(pos50, spy_ret, cost=c))
    print(f"  cost {c*1e4:4.0f}bps  Sharpe {m['Sharpe']:.2f}  CAGR {m['CAGR']*100:.1f}%")
print("  trade delay:")
for d in [1, 2, 3]:
    m = engine.metrics(engine.pnl(pos50, spy_ret, delay=d))
    print(f"    t+{d}  Sharpe {m['Sharpe']:.2f}  CAGR {m['CAGR']*100:.1f}%  MaxDD {m['MaxDD']*100:.0f}%")

# (4) significance ------------------------------------------------------------
ret50 = engine.pnl(pos50, spy_ret)
diffs = engine.bootstrap_sharpe_diff(ret50, bh, block=21, n=3000)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print("\n[4] block-bootstrap Sharpe(JM) - Sharpe(B&H):")
print(f"  point {engine.metrics(ret50)['Sharpe']-engine.metrics(bh)['Sharpe']:+.2f}  "
      f"95% CI [{lo:+.2f}, {hi:+.2f}]  P(JM>B&H)={ (diffs>0).mean()*100:.0f}%")

# plot lambda sensitivity -----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
lams = list(lam_rows); ax.plot(lams, [lam_rows[l]['Sharpe'] for l in lams], "o-", label="JM-only Sharpe")
ax.axhline(engine.metrics(bh)['Sharpe'], color="black", ls="--", label="Buy & hold Sharpe")
ax.set_xlabel("jump penalty lambda"); ax.set_ylabel("OOS Sharpe"); ax.legend()
ax.set_title("Stage 8a — JM-only Sharpe vs lambda (robustness)")
plt.tight_layout(); plt.savefig(OUT / "stage8_lambda_sensitivity.png", dpi=110)
print("\nsaved plot -> outputs/stage8_lambda_sensitivity.png")
