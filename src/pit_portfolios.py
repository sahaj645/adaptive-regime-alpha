"""
pit_portfolios.py — daily point-in-time GICS Industry-Group portfolio returns
from the S&P 1500 point-in-time constituent files.

Outputs (cached, git-ignored):
    data/processed/ig_returns_capwt.csv   # market-cap weighted  (primary)
    data/processed/ig_returns_eqwt.csv    # equal weighted       (robustness)
    index = NYSE trading date, columns = 25 GICS industry groups.

No-look-ahead construction:
  * Trading-calendar aligned: returns are trading-day to trading-day (the raw
    files are on a calendar-day grid; non-trading days are dropped).
  * Membership lagged one month: a name contributes in month M only if it was a
    point-in-time member in month M-1 (PIT_Member_Date snapshots are month-end),
    so no within-month future membership is used.
  * GICS classification is time-invariant in the source data (documented
    impurity, see docs/02); membership itself is point-in-time.

Corporate actions: raw Price is split-unadjusted (Market_Cap = Price x Shares
exactly). On days with a >10% Shares_Out change (splits / major actions) the
split-neutral Market_Cap return is used; all returns are winsorized to
[-50%, +75%] as a backstop against residual data errors.

Configuration: set env var PIT_RAW_DIR to the folder holding the raw
sp{500,400,600}_pit_*.csv files (defaults to data/raw/).
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


def build():
    tset = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_csv(ETF, usecols=["Date"]).Date.unique())))
    cols = ["ID", "DATE", "Price", "Market_Cap", "Shares_Out", "GICS_Ind_Group", "PIT_Member_Date"]
    dt   = {"ID": "category", "GICS_Ind_Group": "category"}

    panels = []
    for tag in ["sp500", "sp400", "sp600"]:
        df = pd.read_csv(_find(tag), usecols=cols, dtype=dt, parse_dates=["DATE", "PIT_Member_Date"])
        df = df[df.Price.notna() & df.DATE.isin(tset)]
        panels.append(df)
    p = pd.concat(panels, ignore_index=True).dropna(subset=["GICS_Ind_Group"])
    del panels
    p["mp"] = p.DATE.dt.to_period("M")
    print(f"panel: {len(p):,} active member-days on the trading calendar")

    # --- point-in-time membership, lagged one month (member in M -> eligible in M+1) ---
    mem = p[["ID", "mp"]].drop_duplicates()
    elig = mem.assign(mp=mem.mp + 1, eligible=True)
    p = p.merge(elig, on=["ID", "mp"], how="left")
    p["eligible"] = p["eligible"].fillna(False)

    # --- split-adjusted, winsorized trading-day returns ---
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
    v = p.loc[valid, ["DATE", "GICS_Ind_Group", "r", "w"]].copy()
    print(f"valid stock-day returns: {len(v):,}")

    # --- aggregate to industry-group portfolios ---
    grp = v.groupby(["DATE", "GICS_Ind_Group"], observed=True)
    v["wr"] = v.w * v.r
    cap = (grp.wr.sum() / grp.w.sum()).unstack()
    eq  = grp.r.mean().unstack()
    cap.to_csv(PROC / "ig_returns_capwt.csv")
    eq.to_csv(PROC / "ig_returns_eqwt.csv")

    # --- sanity: cap-weighted market vs SPY ---
    gcap = grp.w.sum().unstack()                                  # group total cap
    mkt = (cap * gcap).sum(1) / gcap.sum(1)                       # cap-weighted across groups
    spy = pd.read_csv(ETF)[lambda d: d.Ticker == "SPY"].set_index("Date")["Adj_Close"]
    spy.index = pd.to_datetime(spy.index); spy_ret = spy.sort_index().pct_change()
    common = mkt.index.intersection(spy_ret.dropna().index)
    corr = np.corrcoef(mkt.loc[common], spy_ret.loc[common])[0, 1]
    print(f"\ncap-weighted matrix: {cap.shape[0]} days x {cap.shape[1]} groups, "
          f"{cap.index.min().date()}..{cap.index.max().date()}")
    print(f"non-null cells: cap={cap.notna().mean().mean()*100:.1f}%  eq={eq.notna().mean().mean()*100:.1f}%")
    print(f"SANITY corr(cap-weighted 25-group market, SPY daily return) = {corr:.3f}")
    return cap, eq


if __name__ == "__main__":
    build()
