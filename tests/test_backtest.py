import numpy as np, pandas as pd
from backtest.engine import pnl


def test_t_plus_one_and_costs():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    ret = pd.Series(np.full(10, 0.01), index=idx)
    pos = pd.Series(0.0, index=idx); pos.iloc[2] = 1.0       # invested signalled at close of day 2
    out = pnl(pos, ret, delay=1, cost_bps=0.0)
    assert out.iloc[2] == 0.0                                # nothing on the signal day
    assert np.isclose(out.iloc[3], 0.01)                     # earns the NEXT day's return
    out_cost = pnl(pos, ret, delay=1, cost_bps=10.0)
    assert out_cost.iloc[3] < out.iloc[3]                    # turnover cost reduces it
