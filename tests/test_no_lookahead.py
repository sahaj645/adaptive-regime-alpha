"""
Machine verification of no-look-ahead in the Absorption Ratio.

Claim: AR(t) and dAR(t) depend only on data up to and including day t.
Test : corrupt every observation AFTER a cut date T with random garbage,
       recompute, and assert that every AR/dAR value at t <= T is unchanged
       (bit-for-bit), while values at t > T do change (so the test is live).

Run:  python tests/test_no_lookahead.py    (also importable by pytest)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from absorption_ratio_pit import absorption_ratio  # noqa: E402


def test_absorption_ratio_is_causal():
    cap = pd.read_csv(ROOT / "data" / "processed" / "ig_returns_capwt.csv",
                      index_col=0, parse_dates=True)
    base = absorption_ratio(cap, window=252, n_eig=5)

    T = cap.index[1500]                                  # cut date (~mid-sample)
    rng = np.random.default_rng(0)
    corrupt = cap.copy()
    fut = corrupt.index > T
    corrupt.loc[fut] = rng.normal(0, 0.25, corrupt.loc[fut].shape)   # garbage future
    pert = absorption_ratio(corrupt, window=252, n_eig=5)

    past = base.index[base.index <= T]
    future = base.index[base.index > T]

    same_past = np.allclose(base.loc[past].values, pert.reindex(past).values, atol=1e-12, rtol=0)
    changed_future = not np.allclose(base.loc[future].values, pert.reindex(future).values, atol=1e-9)

    print(f"corrupted all observations after {T.date()}")
    print(f"  AR/dAR identical for all t <= T : {same_past}   ({len(past)} dates)")
    print(f"  AR/dAR changed for some t  > T  : {changed_future}   ({len(future)} dates)")
    assert same_past, "LOOK-AHEAD DETECTED: a past AR value moved when the future changed"
    assert changed_future, "test inert: corruption had no effect on future AR"
    return same_past and changed_future


if __name__ == "__main__":
    ok = test_absorption_ratio_is_causal()
    print("PASS — AR(t) depends only on data <= t" if ok else "FAIL")
