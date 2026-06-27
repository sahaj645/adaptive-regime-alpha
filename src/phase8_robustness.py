"""
Phase 8 — robustness of the constituent-AR advantage.

Question: is Fused-PIT > Fused-ETF a real effect or a favourable draw? Sweep the
AR construction (covariance window, PCA dimension, weighting, GICS granularity)
at a FIXED jump penalty (isolates the AR from lambda-CV noise) and check whether
the constituent AR keeps beating the ETF AR. Plus crisis/bull subperiods.
"""
from pathlib import Path
import numpy as np, pandas as pd
import core, engine
from absorption_ratio_pit import absorption_ratio

ROOT = Path(__file__).resolve().parents[1]; PROC = ROOT / "data" / "processed"
LAM, SEEDS = 30, range(2)                       # fixed lambda; seed-invariant
_, spy, sec = core.load(); spy_ret = spy.pct_change()
jm, r = engine.jm_features_from_px(spy); jm = jm.iloc[150:]


def run(feat):
    pos = engine.walk_forward(feat.dropna(), r, spy_ret, lam_mode=LAM, seeds=SEEDS)
    ret = engine.pnl(pos, spy_ret)
    m = engine.metrics(ret)
    return m["Sharpe"], m["MaxDD"], pos.mean(), ret


# --- baselines at the same fixed lambda ---
sh_bh = engine.metrics(spy_ret.loc["2020-01-01":])["Sharpe"]
sh_jm = run(jm)[0]
sh_fe, dd_fe, _, _ = run(jm.join(core.ar_features(sec), how="inner"))
print(f"baselines @lambda={LAM}:  Buy&hold {sh_bh:.2f}   JM-only {sh_jm:.2f}   Fused-ETF {sh_fe:.2f}\n")

cap = pd.read_csv(PROC / "ig_returns_capwt.csv", index_col=0, parse_dates=True)
eqw = pd.read_csv(PROC / "ig_returns_eqwt.csv", index_col=0, parse_dates=True)

configs = [
    ("cap 25grp  w252 n5  (base)", lambda: absorption_ratio(cap, 252, 5)),
    ("cap 25grp  w126 n5",         lambda: absorption_ratio(cap, 126, 5)),
    ("cap 25grp  w500 n5",         lambda: absorption_ratio(cap, 500, 5)),
    ("cap 25grp  w252 n3",         lambda: absorption_ratio(cap, 252, 3)),
    ("cap 25grp  w252 n8",         lambda: absorption_ratio(cap, 252, 8)),
    ("eq  25grp  w252 n5",         lambda: absorption_ratio(eqw, 252, 5)),
]
try:
    capi = pd.read_csv(PROC / "ig_returns_capwt_industry.csv", index_col=0, parse_dates=True)
    configs.append(("cap 73ind  w252 n15", lambda: absorption_ratio(capi, 252, 15)))
except FileNotFoundError:
    print("(73-industry matrix not found — skipping granularity test)")

print(f"Fused-PIT robustness (Fused-ETF baseline = {sh_fe:.2f}):")
beats, base_ret = 0, None
for label, make_ar in configs:
    sh, dd, expo, ret = run(jm.join(make_ar(), how="inner"))
    if base_ret is None:
        base_ret = ret
    win = sh > sh_fe
    beats += win
    print(f"  {label:26} Sharpe {sh:.2f}   MaxDD {dd*100:4.0f}%   expo {expo*100:3.0f}%   {'> ETF' if win else '<= ETF'}")
print(f"\nFused-PIT beats Fused-ETF in {beats}/{len(configs)} configurations")

print("\nSubperiod Sharpe (base Fused-PIT vs Buy&hold):")
for name, (a, b) in {"COVID 2020": ("2020-01-01", "2020-12-31"),
                     "2022 bear": ("2022-01-01", "2022-12-31"),
                     "bull 2023-24": ("2023-01-01", "2024-12-31")}.items():
    sp = engine.metrics(base_ret.loc[a:b])["Sharpe"]
    sb = engine.metrics(spy_ret.loc[a:b])["Sharpe"]
    print(f"  {name:14} PIT {sp:+.2f}   B&H {sb:+.2f}")
