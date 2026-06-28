#!/usr/bin/env python3
"""Stress test the submission config: JM-only (SPY) + vol-target + cash yield.

Baseline config: target_vol 10%, cap 1.0 (no leverage), 20d vol window, t+1
execution, 2bps cost, T-bill cash schedule. Each test varies ONE dimension and
re-prices through the same no-look-ahead engine. Writes
results/levers/stress_submission.csv and stress_submission.png.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Config
from data.market_data import asset_price
from backtest.engine import pnl
from levers.vol_target import vol_target_position
from levers.cash_yield import pnl_with_cash
from evaluation.metrics import performance, bootstrap_sharpe_diff
from evaluation.plots import equity_curves
from utils.logging import get_logger

cfg = Config.load("config/default.yaml")
b, e, L = cfg.backtest, cfg.evaluation, cfg.levers
TD = e.trading_days
res = Path(cfg.results_dir); out = res / "levers"
log = get_logger("stress_test")

spy = asset_price(cfg.data.etf_file, cfg.data.trade_asset).pct_change()
bench = pd.read_csv(res / "oos_returns.csv", index_col=0, parse_dates=True)["BuyHold"]
idx = bench.index
jm = pd.read_csv(res / "positions_jm.csv", index_col=0, parse_dates=True)["position"]

# Baseline knobs
BASE = dict(target_vol=0.10, cap=1.0, window=20, delay=b.trade_delay,
            cost_bps=b.cost_bps, rf=L.risk_free_by_year)


def config_return(pos=jm, *, target_vol, cap, window, delay, cost_bps, rf):
    vtp = vol_target_position(pos, spy, target_vol=target_vol, cap=cap,
                              window=window, trading_days=TD)
    r = pnl_with_cash(vtp, spy, rf, delay=delay, cost_bps=cost_bps, trading_days=TD)
    return r.reindex(idx).fillna(0.0)


def m(r):
    p = performance(r, None, TD)
    return {k: p[k] for k in ("CAGR", "Vol", "Sharpe", "Sortino", "MaxDD")}


rows = []
def add(test, variant, r):
    rows.append({"Test": test, "Variant": variant, **m(r)})

base_r = config_return(**BASE)
sh_bh = performance(bench, None, TD)["Sharpe"]
add("Baseline", "target10/cap1.0/20d/t+1/2bps/Tbill", base_r)
add("Reference", "Buy & hold SPY", bench)

# 1. Transaction-cost sensitivity (the big one - daily rebalancing)
for c in [0, 2, 5, 10, 20]:
    add("1. Cost (bps)", f"{c} bps", config_return(**{**BASE, "cost_bps": c}))

# 2. Vol-estimation window
for w in [10, 20, 40, 60]:
    add("2. Vol window (days)", f"{w}d", config_return(**{**BASE, "window": w}))

# 3. Target vol
for tv in [0.08, 0.10, 0.15, 0.20]:
    add("3. Target vol", f"{int(tv*100)}%", config_return(**{**BASE, "target_vol": tv}))

# 4. Cash-rate ASSUMPTION sensitivity
rf0 = {y: 0.0 for y in L.risk_free_by_year}
rf_half = {y: v * 0.5 for y, v in L.risk_free_by_year.items()}
rf_2x = {y: v * 2.0 for y, v in L.risk_free_by_year.items()}
rf_flat = {y: 0.03 for y in L.risk_free_by_year}
for tag, rf in [("0% (no lever-2)", rf0), ("half", rf_half),
                ("T-bill (base)", L.risk_free_by_year), ("double", rf_2x), ("flat 3%", rf_flat)]:
    add("4. Cash-rate assumption", tag, config_return(**{**BASE, "rf": rf}))

# 5. Execution lag
for d in [1, 2]:
    add("5. Execution lag", f"t+{d}", config_return(**{**BASE, "delay": d}))

# 6. Seed stability (refit JM base with alternate k-means seed sets)
for tag in ["", "seeda", "seedb"]:
    f = (res / "positions_jm.csv") if tag == "" else (out / f"positions_jm_{tag}.csv")
    pos = pd.read_csv(f, index_col=0, parse_dates=True)["position"]
    label = "seeds 0-2 (base)" if tag == "" else ("seeds 10-12" if tag == "seeda" else "seeds 20-22")
    add("6. Seed set", label, config_return(pos, **BASE))

tbl = pd.DataFrame(rows)
tbl.to_csv(out / "stress_submission.csv", index=False)
print(tbl.round(3).to_string(index=False))

# --- significance + subperiods on the baseline config ---
d = bootstrap_sharpe_diff(base_r, bench, block_size=e.block_size, n=e.bootstrap_n,
                          trading_days=TD, seed=cfg.seed)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\nBaseline Sharpe {m(base_r)['Sharpe']:.2f} vs B&H {sh_bh:.2f}: "
      f"dSharpe {m(base_r)['Sharpe']-sh_bh:+.2f}, 95%CI[{lo:+.2f},{hi:+.2f}], P(win){(d>0).mean()*100:.0f}%")

print("\nSubperiod stability (Sharpe, config vs buy-hold):")
periods = {"COVID 2020-21": ("2020-01-01", "2021-12-31"),
           "Bear 2022": ("2022-01-01", "2022-12-31"),
           "Bull 2023-25": ("2023-01-01", "2025-12-31")}
for name, (s, e_) in periods.items():
    cr, bb = base_r.loc[s:e_], bench.loc[s:e_]
    print(f"  {name:14s}  config {performance(cr,None,TD)['Sharpe']:+.2f}   "
          f"buy-hold {performance(bb,None,TD)['Sharpe']:+.2f}   "
          f"config MaxDD {performance(cr,None,TD)['MaxDD']*100:+.0f}%")

equity_curves({"Buy & hold SPY": bench, "Submission config (JM+vt+cash)": base_r,
               "Config @ 10bps cost": config_return(**{**BASE, "cost_bps": 10})},
              out / "stress_submission.png",
              "Submission config stress: equity vs buy-hold", highlight=("Submission config (JM+vt+cash)",))
log.info("wrote %s", out / "stress_submission.csv")
