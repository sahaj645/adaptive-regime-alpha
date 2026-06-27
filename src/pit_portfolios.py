"""
pit_portfolios.py -- daily point-in-time GICS portfolio return matrices from the
S&P 1500 point-in-time constituent files.

Outputs (cached, git-ignored):
    data/processed/ig_returns_capwt{suffix}.csv   # market-cap weighted (primary)
    data/processed/ig_returns_eqwt{suffix}.csv    # equal weighted      (robustness)
    index = NYSE trading date, columns = GICS buckets at the chosen granularity.

No-look-ahead: trading-calendar aligned; membership lagged one month (member in
M -> eligible in M+1); GICS classification is static in source (documented).
Corporate actions: raw Price is split-unadjusted (Market_Cap = Price x Shares);
on >10% Shares_Out changes the split-neutral Market_Cap return is used; returns
winsorized to [-50%, +75%].

Env: PIT_RAW_DIR (raw files dir), PIT_GROUP_COL (GICS level), PIT_OUT_SUFFIX.
"""
import os, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("PIT_RAW_DIR", ROOT / "data" / "raw"))
PROC    = ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)
ETF     = ROOT / "data" / "etf_ohlcv.csv"
WIN_LO, WIN_HI, SPLIT_THR = -0.50, 0.75, 0.10


def _find(tag):
    hits = sorted(glob.glob(str(RAW_DIR / f"{tag}_pit_*.csv")))
    if not hits:
        raise FileNotFoundError(f"{tag}_pit_*.csv not found in {RAW_DIR}")
    return hits[0]


def build(group_col="GICS_Ind_Group", suffix=""):
    tset = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_csv(ETF, usecols=["Date"]).Date.unique())))
    cols = ["ID", "DATE", "Price", "Market_Cap", "Shares_Out", group_col, "PIT_Member_Date"]
    dt   = {"ID": "category", group_col: "category"}

    panels = []
    for tag in ["sp500", "sp400", "sp600"]:
        df = pd.read_csv(_find(tag), usecols=cols, dtype=dt, parse_dates=["DATE", "PIT_Member_Date"])
        df = df[df.Price.notna() & df.DATE.isin(tset)]
        panels.append(df)
    p = pd.concat(panels, ignore_index=True).dropna(subset=[group_col])
    del panels
    p["mp"] = p.DATE.dt.to_period("M")
    print(f"panel: {len(p):,} active member-days on the trading calendar")

    mem = p[["ID", "mp"]].drop_duplicates()
    elig = mem.assign(mp=mem.mp + 1, eligible=True)
    p = p.merge(elig, on=["ID", "mp"], how="left")
    p["eligible"] = p["eligible"].fillna(False)

    p = p.sort_values(["ID", "DATE"])
    g = p.groupby("ID", observed=True)
    pPrice, pCap, pSh = g.Price.shift(), g.Market_Cap.shift(), g.Shares_Out.shift()
    gap = g.DATE.diff().dt.days
    raw_r = p.Price / pPrice - 1.0
    cap_r = p.Market_Cap / pCap - 1.0
    is_split = (p.Shares_Out / pSh - 1.0).abs() > SPLIT_THR
    p["r"] = np.where(is_split, cap_r, raw_r)
    p["r"] = p["r"].clip(WIN_LO, WIN_HI)
    p["w"] = pCap
    valid = p.eligible & pPrice.notna() & np.isfinite(p["r"]) & (gap <= 7)
    v = p.loc[valid, ["DATE", group_col, "r", "w"]].copy()
    print(f"valid stock-day returns: {len(v):,}")

    grp = v.groupby(["DATE", group_col], observed=True)
    v["wr"] = v.w * v.r
    cap = (grp.wr.sum() / grp.w.sum()).unstack()
    eq  = grp.r.mean().unstack()
    cap.to_csv(PROC / f"ig_returns_capwt{suffix}.csv")
    eq.to_csv(PROC / f"ig_returns_eqwt{suffix}.csv")

    gcap = grp.w.sum().unstack()
    mkt = (cap * gcap).sum(1) / gcap.sum(1)
    spy = pd.read_csv(ETF)[lambda d: d.Ticker == "SPY"].set_index("Date")["Adj_Close"]
    spy.index = pd.to_datetime(spy.index); spy_ret = spy.sort_index().pct_change()
    common = mkt.index.intersection(spy_ret.dropna().index)
    corr = np.corrcoef(mkt.loc[common], spy_ret.loc[common])[0, 1]
    print(f"\nmatrix: {cap.shape[0]} days x {cap.shape[1]} buckets, {cap.index.min().date()}..{cap.index.max().date()}")
    print(f"non-null cells: cap={cap.notna().mean().mean()*100:.1f}%  eq={eq.notna().mean().mean()*100:.1f}%")
    print(f"SANITY corr(cap-weighted market, SPY daily return) = {corr:.3f}")
    return cap, eq


if __name__ == "__main__":
    build(os.environ.get("PIT_GROUP_COL", "GICS_Ind_Group"),
          os.environ.get("PIT_OUT_SUFFIX", ""))
