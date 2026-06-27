"""Point-in-time GICS portfolio return matrices from the S&P 1500 constituent
files. Survivorship-free (monthly PIT membership, lagged one month), trading-
calendar aligned, split-neutralized via the share-count change, winsorized."""
from __future__ import annotations
import glob
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from utils.logging import get_logger
from data.market_data import trading_calendar, asset_price

log = get_logger(__name__)
_TAGS = ("sp500", "sp400", "sp600")


def _find(raw_dir: str | Path, tag: str) -> str:
    hits = sorted(glob.glob(str(Path(raw_dir) / f"{tag}_pit_*.csv")))
    if not hits:
        raise FileNotFoundError(f"{tag}_pit_*.csv not found in {raw_dir}")
    return hits[0]


def build_portfolios(raw_dir: str | Path, etf_file: str | Path, gics_level: str,
                     split_threshold: float, winsor_low: float, winsor_high: float,
                     max_gap_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cap_weighted, equal_weighted) daily return matrices: dates x GICS bucket."""
    cal = trading_calendar(etf_file)
    cols = ["ID", "DATE", "Price", "Market_Cap", "Shares_Out", gics_level, "PIT_Member_Date"]
    dtype = {"ID": "category", gics_level: "category"}

    panels = []
    for tag in _TAGS:
        df = pd.read_csv(_find(raw_dir, tag), usecols=cols, dtype=dtype,
                         parse_dates=["DATE", "PIT_Member_Date"])
        panels.append(df[df.Price.notna() & df.DATE.isin(cal)])
    p = pd.concat(panels, ignore_index=True).dropna(subset=[gics_level])
    p["mp"] = p.DATE.dt.to_period("M")
    log.info("constituent panel: %d active member-days", len(p))

    # point-in-time membership, lagged one month (member in M -> eligible in M+1)
    mem = p[["ID", "mp"]].drop_duplicates()
    p = p.merge(mem.assign(mp=mem.mp + 1, eligible=True), on=["ID", "mp"], how="left")
    p["eligible"] = p["eligible"].fillna(False)

    # split-adjusted, winsorized trading-day returns
    p = p.sort_values(["ID", "DATE"])
    g = p.groupby("ID", observed=True)
    prev_price, prev_cap, prev_sh = g.Price.shift(), g.Market_Cap.shift(), g.Shares_Out.shift()
    gap = g.DATE.diff().dt.days
    is_split = (p.Shares_Out / prev_sh - 1.0).abs() > split_threshold
    ret = np.where(is_split, p.Market_Cap / prev_cap - 1.0, p.Price / prev_price - 1.0)
    p["r"] = np.clip(ret, winsor_low, winsor_high)
    p["w"] = prev_cap
    valid = p.eligible & prev_price.notna() & np.isfinite(p["r"]) & (gap <= max_gap_days)
    v = p.loc[valid, ["DATE", gics_level, "r", "w"]].copy()

    grp = v.groupby(["DATE", gics_level], observed=True)
    v["wr"] = v.w * v.r
    cap = (grp.wr.sum() / grp.w.sum()).unstack()
    eq = grp.r.mean().unstack()

    # construction sanity: cap-weighted aggregate must track the market
    gcap = grp.w.sum().unstack()
    mkt = (cap * gcap).sum(1) / gcap.sum(1)
    spy_ret = asset_price(etf_file, "SPY").pct_change()
    common = mkt.index.intersection(spy_ret.dropna().index)
    corr = float(np.corrcoef(mkt.loc[common], spy_ret.loc[common])[0, 1])
    log.info("%d buckets, %d days, corr(cap-weighted market, SPY)=%.3f", cap.shape[1], cap.shape[0], corr)
    if corr < 0.95:
        raise ValueError(f"construction sanity failed: corr {corr:.3f} < 0.95")
    return cap, eq
