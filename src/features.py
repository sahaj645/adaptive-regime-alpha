"""
features.py — model feature assembly (Phase 6 integration).

The ONLY thing that changes between the old and new combined model is the
Absorption-Ratio feed:
    fused_etf : 4 jump-model features + AR level + dAR  (from the 11 sector ETFs)
    fused_pit : 4 jump-model features + AR level + dAR  (from the 25 PIT industry groups)
The jump-model features and the walk-forward engine are imported unchanged.
"""
from pathlib import Path
import pandas as pd
import core
import engine

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def build(which: str = "fused_pit"):
    """Return (feature_df, spy_log_returns, spy_price). All columns are causal."""
    _, spy, sec = core.load()
    jm, r = engine.jm_features_from_px(spy)
    jm = jm.iloc[150:]                                   # 120d-EWMA burn-in
    if which == "jm":
        feat = jm
    elif which == "fused_etf":
        feat = jm.join(core.ar_features(sec), how="inner")            # ETF AR
    elif which == "fused_pit":
        ar = pd.read_csv(PROC / "ar_pit_capwt.csv", index_col=0, parse_dates=True)
        feat = jm.join(ar, how="inner")                               # constituent AR
    elif which == "ar_pit":
        feat = pd.read_csv(PROC / "ar_pit_capwt.csv", index_col=0, parse_dates=True)
    else:
        raise ValueError(f"unknown feature set: {which}")
    return feat.dropna(), r, spy


if __name__ == "__main__":
    feat, r, spy = build("fused_pit")
    spy_ret = spy.pct_change()
    print("fused_pit features:", list(feat.columns))
    print("shape:", feat.shape, "|", feat.index.min().date(), "->", feat.index.max().date())

    # run through the UNCHANGED engine
    pos = engine.walk_forward(feat, r, spy_ret, lam_mode="cv", seeds=range(3))
    print("OOS regime/position series: %s -> %s | exposure %.0f%% | switches %d"
          % (pos.index.min().date(), pos.index.max().date(), pos.mean() * 100, int((pos.diff() != 0).sum())))

    # --- concrete t+1 verification in the real engine.pnl code path ---
    probe = pd.Series(0.0, index=spy_ret.loc["2020-01-01":].index)
    d = probe.index[10]
    probe.loc[d] = 1.0                                   # invest on signal day d only
    pl = engine.pnl(probe, spy_ret, cost=0.0, oos_start=probe.index[0])
    i = pl.index.get_loc(d)
    print("\nt+1 contract check:")
    print("  signal day d           =", d.date(), " strat return on d   =", round(pl.loc[d], 6), "(must be 0)")
    print("  next trading day d+1   =", pl.index[i + 1].date(),
          " strat return on d+1 =", round(pl.iloc[i + 1], 6),
          "==", round(spy_ret.loc[pl.index[i + 1]], 6), "(SPY return on d+1)")
    assert pl.loc[d] == 0.0, "leak: signal earns on its own day"
    print("PASS — a regime read at close(d) is traded on d+1, never d.")
