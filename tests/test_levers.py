"""No-look-ahead and sanity tests for the vol-targeting lever."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from levers.vol_target import vol_target_position, realized_vol


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    ret = pd.Series(rng.normal(0, 0.01, n), index=idx)
    base = pd.Series(1.0, index=idx)            # always-on signal isolates the lever
    return ret, base


def test_no_lookahead_future_returns_dont_change_past_positions():
    ret, base = _data()
    cut = ret.index[300]
    pos_a = vol_target_position(base, ret, target_vol=0.15, cap=1.5, window=20, trading_days=252)
    ret2 = ret.copy()
    ret2.loc[ret2.index > cut] *= 5.0           # blow up the future only
    pos_b = vol_target_position(base, ret2, target_vol=0.15, cap=1.5, window=20, trading_days=252)
    past = pos_a.index[pos_a.index <= cut]
    assert np.allclose(pos_a.loc[past].values, pos_b.loc[past].values), "future leaked into past"


def test_cap_is_respected():
    ret, base = _data()
    pos = vol_target_position(base, ret, target_vol=0.50, cap=1.2, window=20, trading_days=252)
    assert pos.max() <= 1.2 + 1e-9


def test_zero_base_stays_flat():
    ret, base = _data()
    pos = vol_target_position(base * 0.0, ret, target_vol=0.15, cap=2.0, window=20, trading_days=252)
    assert (pos.abs() < 1e-12).all()


def test_realized_vol_is_trailing():
    ret, _ = _data()
    rv = realized_vol(ret, 20, 252)
    assert rv.iloc[:19].isna().all() and rv.iloc[19:].notna().all()
