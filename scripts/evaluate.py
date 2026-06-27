#!/usr/bin/env python3
"""Compute the performance table, bootstrap significance, and equity figure from
the OOS returns produced by run_walkforward."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Config
from evaluation.metrics import performance, bootstrap_sharpe_diff
from evaluation.plots import equity_curves
from utils.logging import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    cfg = Config.load(ap.parse_args().config)
    log = get_logger("evaluate"); res = Path(cfg.results_dir); e = cfg.evaluation
    rets = pd.read_csv(res / "oos_returns.csv", index_col=0, parse_dates=True)
    pos = pd.read_csv(res / "positions_fused.csv", index_col=0, parse_dates=True)["position"]

    table = pd.DataFrame({c: performance(rets[c], pos if c == "Fused_PIT" else None, e.trading_days)
                          for c in rets.columns}).T
    table.to_csv(res / "metrics.csv")
    print(table.round(3).to_string())

    d = bootstrap_sharpe_diff(rets["Fused_PIT"], rets["BuyHold"], block_size=e.block_size,
                              n=e.bootstrap_n, trading_days=e.trading_days, seed=cfg.seed)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\nFused-PIT vs BuyHold: dSharpe "
          f"{table.loc['Fused_PIT','Sharpe'] - table.loc['BuyHold','Sharpe']:+.2f}"
          f"  95%CI[{lo:+.2f},{hi:+.2f}]  P(win){(d > 0).mean() * 100:.0f}%")
    equity_curves({c: rets[c] for c in rets.columns}, res / "equity_curves.png",
                  "Out-of-sample growth of $1 (2020-2025)", highlight=("Fused_PIT", "BuyHold"))
    log.info("wrote %s", res / "metrics.csv")


if __name__ == "__main__":
    main()
