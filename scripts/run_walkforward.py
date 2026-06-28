#!/usr/bin/env python3
"""Run the no-look-ahead walk-forward; write OOS daily returns for the combined
model and its baselines (buy-and-hold, jump-model-only) to results/."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Config
from data.market_data import asset_price
from features.pipeline import jump_feature_set, fused_feature_set
from backtest.engine import walk_forward, pnl
from utils.logging import get_logger
from utils.determinism import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    cfg = Config.load(ap.parse_args().config)
    set_seed(cfg.seed); log = get_logger("run_walkforward")
    b, j, e = cfg.backtest, cfg.jump_model, cfg.evaluation
    seeds = tuple(range(j.n_seeds))
    asset_ret = asset_price(cfg.data.etf_file, cfg.data.trade_asset).pct_change()

    returns_matrix = pd.read_csv(Path(cfg.data.processed_dir) / "ig_returns_capwt.csv",
                                 index_col=0, parse_dates=True)
    fused, logret, _ = fused_feature_set(cfg, returns_matrix)
    jm, _, _ = jump_feature_set(cfg)

    def run(feats):
        pos, lam = walk_forward(feats, logret, asset_ret, oos_start=b.oos_start,
                                first_train_year=b.first_train_year, last_year=b.last_year,
                                n_states=j.n_states, seeds=seeds, lam_grid=j.lambda_grid,
                                cv_folds=j.cv_folds, cv_val_days=j.cv_val_days,
                                cv_embargo=j.cv_embargo, trading_days=e.trading_days)
        return pos, pnl(pos, asset_ret, b.trade_delay, b.cost_bps), lam

    pos_f, ret_f, lam_f = run(fused)
    pos_j, ret_j, lam_j = run(jm)
    bench = asset_ret.loc[b.oos_start:]

    res = Path(cfg.results_dir); res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"BuyHold": bench, "JM_only": ret_j, "Fused_PIT": ret_f}).to_csv(res / "oos_returns.csv")
    pos_f.to_frame("position").to_csv(res / "positions_fused.csv")
    pos_j.to_frame("position").to_csv(res / "positions_jm.csv")
    (res / "lambda_by_year.json").write_text(json.dumps({"fused": lam_f, "jm_only": lam_j}, default=int, indent=2))
    log.info("wrote %s", res / "oos_returns.csv")


if __name__ == "__main__":
    main()
