import numpy as np, pandas as pd
from models.absorption_ratio import absorption_ratio


def test_absorption_ratio_is_causal():
    rng = np.random.default_rng(0); T, N = 700, 8
    idx = pd.date_range("2018-01-01", periods=T, freq="B")
    R = pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=idx)
    base = absorption_ratio(R, 252, 2, 252, 15)
    cut = idx[450]
    R2 = R.copy(); R2.loc[R2.index > cut] = rng.normal(0, 0.1, R2.loc[R2.index > cut].shape)
    pert = absorption_ratio(R2, 252, 2, 252, 15)
    past = base.index[base.index <= cut]
    assert np.allclose(base.loc[past].values, pert.reindex(past).values, atol=1e-12)
