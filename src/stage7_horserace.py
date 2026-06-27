"""
Stage 7 — the horse race, all through the Stage-3 no-look-ahead engine.

Five strategies, identical out-of-sample window (2020-2025), identical engine:
  1. Buy & hold SPY
  2. AR-only   : jump model on the 2 absorption-ratio features
  3. JM-only   : jump model on the 4 SPY return features
  4. Fused     : jump model on all 6 features
  5. Gate (OR) : invested if EITHER JM-only OR AR-only says bull (tests the
                 'AR for earlier re-entry' hypothesis from Stage 3)

To keep the ablation clean, all jump-model horses train on the SAME dates
(intersection of JM and AR feature availability), so the ONLY thing that
changes between them is the feature set.

No-look-ahead, enforced everywhere:
  * lambda (jump penalty) is chosen at each refit by inner time-series CV on
    TRAIN only (validate on the last 252 train days, maximise Sortino).
  * scaler, centroids and bull/bear label fit on train only; regimes inferred
    with the causal forward filter; positions traded t+1; 2 bps cost per trade.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"
OOS_START, LAM_GRID, COST = "2020-01-01", [10, 30, 50, 80], 0.0002

df, spy, sec = core.load()
jmf, r = core.jm_features(spy)
arf     = core.ar_features(sec)                 # AR, dAR (causal)
spy_ret = spy.pct_change()

common     = jmf.index.intersection(arf.index)  # fair ablation: same train dates
feat_jm    = jmf.loc[common]
feat_ar    = arf.loc[common]
feat_fused = pd.concat([feat_jm, feat_ar], axis=1).loc[common]


# ----------------------------------------------------------- helpers
def metrics(ret):
    ret = ret.dropna(); eq = (1 + ret).cumprod(); yrs = len(ret) / 252
    vol = ret.std() * np.sqrt(252); dn = ret[ret < 0].std() * np.sqrt(252)
    return dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, Vol=vol,
                Sharpe=ret.mean() / ret.std() * np.sqrt(252) if ret.std() else np.nan,
                Sortino=ret.mean() * 252 / dn if dn else np.nan,
                MaxDD=(eq / eq.cummax() - 1).min())


def pnl(pos, restrict=OOS_START):
    held = pos.shift(1).reindex(spy_ret.index).fillna(0.0)
    turn = held.diff().abs().fillna(0.0)
    out  = held * spy_ret - COST * turn
    return out.loc[restrict:] if restrict else out


def sortino(ret):
    ret = ret.dropna()
    if len(ret) < 20 or ret.std() == 0: return -1e9
    dn = ret[ret < 0].std()
    return -1e9 if (dn == 0 or np.isnan(dn)) else ret.mean() / dn * np.sqrt(252)


def fit_predict(feat_tr, feat_to, lam):
    mu_, sd_ = feat_tr.mean(), feat_tr.std() + 1e-12
    s_tr, cent = core.fit_jump(((feat_tr - mu_) / sd_).values, lam)
    bear = core.label_bear(s_tr, r.reindex(feat_tr.index).values)
    states = core.forward_filter(((feat_to - mu_) / sd_).values, cent, lam)
    return states, bear


def select_lambda(feat_tr, grid):
    if len(feat_tr) < 400: return 50
    cut = feat_tr.index[-252]
    inner_tr = feat_tr[feat_tr.index < cut]
    best = (50, -1e18)
    for lam in grid:
        states, bear = fit_predict(inner_tr, feat_tr, lam)
        mask = np.asarray(feat_tr.index >= cut)
        pos = pd.Series((states[mask] != bear).astype(float), index=feat_tr.index[mask])
        sc = sortino(pnl(pos, restrict=None).reindex(pos.index).dropna())
        if sc > best[1]: best = (lam, sc)
    return best[0]


def walk_forward(feat, grid=LAM_GRID):
    pos, lams = pd.Series(index=feat.index, dtype=float), {}
    for y in range(2020, 2026):
        tr_end = pd.Timestamp(f"{y-1}-12-31"); feat_tr = feat[feat.index <= tr_end]
        if len(feat_tr) < 300: continue
        lam = select_lambda(feat_tr, grid); lams[y] = lam
        feat_to = feat[feat.index <= pd.Timestamp(f"{y}-12-31")]
        states, bear = fit_predict(feat_tr, feat_to, lam)
        m = np.asarray((feat_to.index > tr_end) & (feat_to.index <= pd.Timestamp(f"{y}-12-31")))
        pos.loc[feat_to.index[m]] = (states[m] != bear).astype(float)
    return pos.loc[OOS_START:].dropna(), lams


# ----------------------------------------------------------- run all horses
pos_jm,  lam_jm = walk_forward(feat_jm)
pos_ar,  lam_ar = walk_forward(feat_ar)
pos_fz,  lam_fz = walk_forward(feat_fused)
idx = pos_jm.index.intersection(pos_ar.index)
pos_gate = ((pos_jm.reindex(idx) == 1) | (pos_ar.reindex(idx) == 1)).astype(float)

rets = {"Buy & hold SPY": spy_ret.loc[OOS_START:],
        "AR-only":        pnl(pos_ar),
        "JM-only":        pnl(pos_jm),
        "Fused":          pnl(pos_fz),
        "Gate (OR)":      pnl(pos_gate)}
inv = {"AR-only": pos_ar.mean(), "JM-only": pos_jm.mean(),
       "Fused": pos_fz.mean(), "Gate (OR)": pos_gate.mean()}

tbl = pd.DataFrame({k: metrics(v) for k, v in rets.items()}).T
fmt = tbl.copy()
for c in ["CAGR", "Vol", "MaxDD"]: fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
for c in ["Sharpe", "Sortino"]:    fmt[c] = fmt[c].round(2)
print("OOS:", OOS_START, "->", rets["Buy & hold SPY"].index.max().date())
print("chosen lambda by year — JM:", lam_jm, "AR:", lam_ar, "Fused:", lam_fz)
print("time invested:", {k: f"{v*100:.0f}%" for k, v in inv.items()})
print(fmt.to_string())
tbl.to_csv(OUT / "stage7_metrics.csv")

# ----------------------------------------------------------- equity plot
fig, ax = plt.subplots(figsize=(13, 6.5))
colors = {"Buy & hold SPY": "black", "AR-only": "tab:green", "JM-only": "tab:blue",
          "Fused": "tab:purple", "Gate (OR)": "tab:orange"}
for k, rr in rets.items():
    eq = (1 + rr).cumprod()
    ax.plot(eq.index, eq.values, label=k, lw=1.6 if k in ("Fused", "Buy & hold SPY") else 1.1,
            color=colors[k])
ax.set_yscale("log"); ax.set_title("Stage 7 — out-of-sample growth of $1 (2020-2025), all horses")
ax.legend(); plt.tight_layout(); plt.savefig(OUT / "stage7_horserace.png", dpi=110)
print("saved plot -> outputs/stage7_horserace.png")
