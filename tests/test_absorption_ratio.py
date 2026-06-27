import numpy as np, pandas as pd
from models.absorption_ratio import absorption_ratio


def test_bounds_and_concentration():
    rng = np.random.default_rng(0); T, N = 600, 10
    idx = pd.date_range("2018-01-01", periods=T, freq="B")
    common = rng.normal(0, 0.01, (T, 1))
    correlated = pd.DataFrame(common + rng.normal(0, 0.002, (T, N)), index=idx)
    independent = pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=idx)
    ar_c = absorption_ratio(correlated, 252, 2, 252, 15)["AR"]
    ar_i = absorption_ratio(independent, 252, 2, 252, 15)["AR"]
    assert ((ar_c >= 0) & (ar_c <= 1)).all()
    assert ar_c.mean() > ar_i.mean()      # a correlated cross-section absorbs more variance
