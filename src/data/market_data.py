"""Market data access: ETF prices, sector returns, the NYSE trading calendar."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import pandas as pd


def _read(etf_file: str | Path) -> pd.DataFrame:
    return pd.read_csv(etf_file, parse_dates=["Date"])


def asset_price(etf_file: str | Path, ticker: str) -> pd.Series:
    """Dividend-adjusted close for one ETF, indexed by date."""
    df = _read(etf_file)
    return df[df.Ticker == ticker].set_index("Date")["Adj_Close"].sort_index()


def sector_returns(etf_file: str | Path, exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Daily returns of the GICS sector ETFs (a clean, ready-made cross-section)."""
    df = _read(etf_file)
    px = (df[df.ETF_Group == "Sector ETF"]
          .pivot(index="Date", columns="Ticker", values="Adj_Close")
          .sort_index().drop(columns=list(exclude), errors="ignore"))
    return px.pct_change().iloc[1:]


def trading_calendar(etf_file: str | Path) -> pd.DatetimeIndex:
    df = pd.read_csv(etf_file, usecols=["Date"], parse_dates=["Date"])
    return pd.DatetimeIndex(sorted(df.Date.unique()))
