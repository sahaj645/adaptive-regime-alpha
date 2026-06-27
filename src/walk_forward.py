"""
walk_forward.py — the no-look-ahead engine (Stage 3) + a validation run.

The timing contract (this is the part that was getting marked down):
  1. Features at the close of day t use only data <= t        (core.py guarantees this)
  2. Every fitted transform -- the feature scaler, the jump-model centroids,
     the bull/bear labelling -- is fit on TRAIN data only, then frozen.
  3. Regimes out-of-sample are inferred with the CAUSAL forward filter
     (core.forward_filter), never the future-peeking backward pass.
  4. A regime read at the close of day t is traded on t+1:  pos.shift(1) * ret.

Protocol: expanding walk-forward. Refit at each year-end; the model fit on data
through Dec-31 of year Y trades the whole of year Y+1 out-of-sample. First train
ends 2019-12-31, so the out-of-sample track is 2020-2025 (COVID, 2022, etc.).

This script validates the engine on the JUMP-MODEL-ONLY configuration and runs a
LEAK TEST: the same model fit in-sample (peeking) to show how much look-ahead
would have flattered the numbers. Stage 7 reuses this engine for all 5 horses.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"
LAM, OOS_START = 50, "2020-01-01"

df, spy, sec = core.load()
feat, r = core.jm_features(spy)
feat = feat.iloc[150:]                      # burn-in for the 120d EWMA
spy_ret = spy.pct_change()


# ------------------------------------------------------------------ engine
def walk_forward(feat, r, lam, oos_start=OOS_START, first_train_year=2019, last_year=2025):
    """Return a daily position series (1=invested, 0=cash) at SIGNAL day t, OOS."""
    pos = pd.Series(index=feat.index, dtype=float)
    for y in range(first_train_year + 1, last_year + 1):           # 2020..2025
        tr_end = pd.Timestamp(f"{y-1}-12-31")
        feat_tr = feat[feat.index <= tr_end]
        if len(feat_tr) < 250:
            continue
        # --- fit on TRAIN only: scaler, centroids, bull/bear label ---
        mu_, sd_ = feat_tr.mean(), feat_tr.std() + 1e-12
        Xtr = ((feat_tr - mu_) / sd_).values
        s_tr, cent = core.fit_jump(Xtr, lam)
        bear = core.label_bear(s_tr, r.reindex(feat_tr.index).values)
        # --- predict the OOS year causally (forward filter over all data <= t) ---
        feat_to_y = feat[feat.index <= pd.Timestamp(f"{y}-12-31")]
        Xall = ((feat_to_y - mu_) / sd_).values
        states = core.forward_filter(Xall, cent, lam)              # causal
        seg_mask = np.asarray((feat_to_y.index > tr_end) &
                              (feat_to_y.index <= pd.Timestamp(f"{y}-12-31")))
        invested = (states[seg_mask] != bear).astype(float)        # bull -> 1, bear -> 0
        pos.loc[feat_to_y.index[seg_mask]] = invested
    return pos.loc[oos_start:].dropna()


def in_sample_cheat(feat, r, lam, oos_start=OOS_START):
    """LEAK: fit & SMOOTH on the full sample (peeks at the future), same trade rule."""
    mu_, sd_ = feat.mean(), feat.std() + 1e-12
    X = ((feat - mu_) / sd_).values
    s, cent = core.fit_jump(X, lam)                               # full-sample fit
    bear = core.label_bear(s, r.reindex(feat.index).values)
    pos = pd.Series((s != bear).astype(float), index=feat.index)
    return pos.loc[oos_start:].dropna()


# ------------------------------------------------------------------ metrics
def metrics(ret):
    ret = ret.dropna(); eq = (1 + ret).cumprod(); yrs = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol  = ret.std() * np.sqrt(252)
    shp  = ret.mean() / ret.std() * np.sqrt(252) if ret.std() else np.nan
    dn   = ret[ret < 0].std() * np.sqrt(252)
    srt  = ret.mean() * 252 / dn if dn else np.nan
    mdd  = (eq / eq.cummax() - 1).min()
    return dict(CAGR=cagr, Vol=vol, Sharpe=shp, Sortino=srt, MaxDD=mdd)


def strat_returns(pos):
    held = pos.shift(1).reindex(spy_ret.index).fillna(0.0)        # trade NEXT day
    return (held * spy_ret).loc[OOS_START:]


# ------------------------------------------------------------------ run
pos_oos   = walk_forward(feat, r, LAM)
pos_cheat = in_sample_cheat(feat, r, LAM)

ret_oos   = strat_returns(pos_oos)
ret_cheat = strat_returns(pos_cheat)
ret_bh    = spy_ret.loc[OOS_START:]

rows = {"JM walk-forward (honest OOS)": metrics(ret_oos),
        "JM in-sample (LEAK / cheat)":  metrics(ret_cheat),
        "Buy & hold SPY":               metrics(ret_bh)}
tbl = pd.DataFrame(rows).T
tbl_fmt = tbl.copy()
for c in ["CAGR", "Vol", "MaxDD"]: tbl_fmt[c] = (tbl_fmt[c] * 100).round(1).astype(str) + "%"
for c in ["Sharpe", "Sortino"]:    tbl_fmt[c] = tbl_fmt[c].round(2)
print("OOS window:", OOS_START, "->", ret_bh.index.max().date(), f"({len(ret_bh)} days)")
print("Time invested (honest):", f"{pos_oos.mean()*100:.0f}%   switches: {int((pos_oos.diff()!=0).sum())}")
print(tbl_fmt.to_string())

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(figsize=(13, 6))
for lab, rr, c in [("Buy & hold SPY", ret_bh, "black"),
                   ("JM walk-forward (honest OOS)", ret_oos, "tab:blue"),
                   ("JM in-sample (leak)", ret_cheat, "tab:red")]:
    ax.plot((1 + rr).cumprod().index, (1 + rr).cumprod().values, label=lab, lw=1.3,
            color=c, ls=("--" if "leak" in lab else "-"))
ax.set_yscale("log"); ax.set_title("Stage-3 backbone check: honest OOS vs leaked vs buy-and-hold (growth of $1)")
ax.legend(); plt.tight_layout(); plt.savefig(OUT / "stage3_backbone_check.png", dpi=110)
print("saved plot -> outputs/stage3_backbone_check.png")
