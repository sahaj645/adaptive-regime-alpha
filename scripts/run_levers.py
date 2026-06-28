#!/usr/bin/env python3
"""Lever sweep + consolidated final results.

Levers (each a pure overlay in src/levers/, priced through the same
no-look-ahead engine: t+1 execution, cost on turnover):
  1 vol targeting  - continuous sizing by trailing realized vol (cap = leverage)
  2 cash yield     - risk-free carry on the uninvested fraction (financing if levered)
  3 multi-asset    - same JM regime per asset (SPY/QQQ/sectors), combined equal-weight

Writes results/levers/{lever1_vol_target.csv, final_summary.csv,
positions_basket.csv, final_equity.png}.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Config
from data.market_data import asset_price
from backtest.engine import pnl
from levers.vol_target import vol_target_position
from levers.cash_yield import pnl_with_cash
from levers.multi_asset import basket_positions
from evaluation.metrics import performance, bootstrap_sharpe_diff
from evaluation.plots import equity_curves
from utils.logging import get_logger

BASES = {"JM_only": "positions_jm.csv", "Fused_PIT": "positions_fused.csv"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    cfg = Config.load(ap.parse_args().config)
    log = get_logger("run_levers")
    b, e, L = cfg.backtest, cfg.evaluation, cfg.levers
    td = e.trading_days
    seeds = tuple(range(cfg.jump_model.n_seeds))
    res = Path(cfg.results_dir); out = res / "levers"; out.mkdir(parents=True, exist_ok=True)

    spy_ret = asset_price(cfg.data.etf_file, cfg.data.trade_asset).pct_change()
    oos = pd.read_csv(res / "oos_returns.csv", index_col=0, parse_dates=True)
    bench = oos["BuyHold"]
    idx = bench.index

    def vt(pos, asset_ret, tv, cap):
        return vol_target_position(pos, asset_ret, target_vol=tv, cap=cap,
                                   window=L.vol_window, trading_days=td)

    def align(r):
        return r.reindex(idx).fillna(0.0)

    # ---------- Lever 1 sweep on the saved SPY bases (kept for the appendix) ----
    rows1 = [{"Base": "-", "Config": "BuyHold", **performance(bench, None, td)}]
    for name, f in BASES.items():
        bp = pd.read_csv(res / f, index_col=0, parse_dates=True)["position"]
        rows1.append({"Base": name, "Config": f"{name} (binary)",
                      **performance(align(pnl(bp, spy_ret, b.trade_delay, b.cost_bps)), bp, td)})
        for tv in L.target_vols:
            for cap in L.caps:
                r = align(pnl(vt(bp, spy_ret, tv, cap), spy_ret, b.trade_delay, b.cost_bps))
                rows1.append({"Base": name, "Config": f"{name} vt{int(tv*100)}_cap{cap}",
                              **performance(r, None, td)})
    pd.DataFrame(rows1).to_csv(out / "lever1_vol_target.csv", index=False)

    # ---------- Multi-asset regime positions (cached) --------------------------
    cache = out / "positions_basket.csv"
    if cache.exists():
        bpos = pd.read_csv(cache, index_col=0, parse_dates=True)
        basket = {c: bpos for c, bpos in bpos.items()} if False else \
                 {c: bpos.dropna() for c, bpos in bpos.items()}
    else:
        log.info("fitting regime model per asset for %d names ...", len(L.multi_asset))
        basket = basket_positions(cfg, L.multi_asset, seeds)
        pd.DataFrame(basket).to_csv(cache)
    asset_rets = {t: asset_price(cfg.data.etf_file, t).pct_change() for t in L.multi_asset}

    # Sleeve pricer: per-asset stack, then equal-weight average across the basket
    def basket_return(tickers, *, lever_vt=None, cash=False):
        legs = []
        for t in tickers:
            p = basket[t].dropna()
            ar = asset_rets[t]
            pos = vt(p, ar, *lever_vt) if lever_vt else p
            if cash:
                leg = pnl_with_cash(pos, ar, L.risk_free_by_year, delay=b.trade_delay,
                                    cost_bps=b.cost_bps, trading_days=td)
            else:
                leg = pnl(pos, ar, b.trade_delay, b.cost_bps)
            legs.append(align(leg))
        return pd.concat(legs, axis=1).mean(axis=1)

    def avg_exposure(tickers):
        return pd.concat([basket[t].reindex(idx) for t in tickers], axis=1).mean(axis=1)

    # ---------- Cumulative stack on the SPY JM base ----------------------------
    jm = pd.read_csv(res / "positions_jm.csv", index_col=0, parse_dates=True)["position"]
    NV = (0.10, 1.0)          # no-leverage vol-target config used through the stack
    full = L.multi_asset
    qqq = ["QQQ"]

    def cash_price(pos, ar):
        return align(pnl_with_cash(pos, ar, L.risk_free_by_year, delay=b.trade_delay,
                                   cost_bps=b.cost_bps, trading_days=td))

    stack = {
        "Buy & hold SPY":                 align(bench),
        "L0  JM binary (mandate)":        align(pnl(jm, spy_ret, b.trade_delay, b.cost_bps)),
        "L2  JM + cash yield":            cash_price(jm, spy_ret),
        "L1  JM + vol-target":            align(pnl(vt(jm, spy_ret, *NV), spy_ret, b.trade_delay, b.cost_bps)),
        "L1+L2  JM + vt + cash":          cash_price(vt(jm, spy_ret, *NV), spy_ret),
        "L3  QQQ regime (binary)":        basket_return(qqq),
        "L3  multi-asset basket (binary)":basket_return(full),
        "L3+L2  basket + cash":           basket_return(full, cash=True),
        "FULL  basket + vt + cash":       basket_return(full, lever_vt=NV, cash=True),
    }
    rows = [{"Strategy": k, **performance(v, None, td)} for k, v in stack.items()]
    summ = pd.DataFrame(rows)[["Strategy", "CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar"]]

    # ---------- find the best full-stack config across the lever-1 grid --------
    best = None
    for tv in L.target_vols:
        for cap in L.caps:
            r = basket_return(full, lever_vt=(tv, cap), cash=True)
            s = performance(r, None, td)["Sharpe"]
            if best is None or s > best[0]:
                best = (s, tv, cap, r)
    bs, btv, bcap, br = best
    summ = pd.concat([summ, pd.DataFrame([{"Strategy":
            f"FULL  basket best (vt{int(btv*100)}_cap{bcap})", **performance(br, None, td)}])],
            ignore_index=True)[["Strategy","CAGR","Vol","Sharpe","Sortino","MaxDD","Calmar"]]
    summ.to_csv(out / "final_summary.csv", index=False)

    print(summ.round(3).to_string(index=False))
    sh_bh = performance(bench, None, td)["Sharpe"]
    d = bootstrap_sharpe_diff(br, bench, block_size=e.block_size, n=e.bootstrap_n,
                              trading_days=td, seed=cfg.seed)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\nBuy&hold Sharpe {sh_bh:.2f}. Best full stack Sharpe {bs:.2f} "
          f"(vt{int(btv*100)}_cap{bcap}); vs B&H {bs-sh_bh:+.2f}, "
          f"95%CI[{lo:+.2f},{hi:+.2f}], P(win){(d>0).mean()*100:.0f}%")

    fig_curves = {
        "Buy & hold SPY": bench,
        "JM binary (mandate)": stack["L0  JM binary (mandate)"],
        "JM + vt + cash": stack["L1+L2  JM + vt + cash"],
        "Multi-asset basket + cash": stack["L3+L2  basket + cash"],
        f"FULL best (vt{int(btv*100)}_cap{bcap})": br,
    }
    equity_curves(fig_curves, out / "final_equity.png",
                  "Lever stack vs buy-and-hold (OOS 2020-2025)",
                  highlight=("Buy & hold SPY", f"FULL best (vt{int(btv*100)}_cap{bcap})"))
    log.info("wrote %s", out / "final_summary.csv")


if __name__ == "__main__":
    main()
