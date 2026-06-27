#!/usr/bin/env python3
"""Build point-in-time GICS portfolio return matrices from the S&P 1500 raw
files. Raw directory comes from config (override with $PIT_RAW_DIR)."""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Config
from data.constituents import build_portfolios
from utils.logging import get_logger
from utils.determinism import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    cfg = Config.load(ap.parse_args().config)
    set_seed(cfg.seed)
    log = get_logger("build_dataset")
    raw_dir = os.environ.get("PIT_RAW_DIR", cfg.data.raw_pit_dir)
    c = cfg.constituents
    cap, eq = build_portfolios(raw_dir, cfg.data.etf_file, c.gics_level, c.split_threshold,
                               c.winsor_low, c.winsor_high, c.max_gap_days)
    out = Path(cfg.data.processed_dir); out.mkdir(parents=True, exist_ok=True)
    cap.to_csv(out / "ig_returns_capwt.csv")
    eq.to_csv(out / "ig_returns_eqwt.csv")
    log.info("wrote %s and equal-weight counterpart", out / "ig_returns_capwt.csv")


if __name__ == "__main__":
    main()
