import numpy as np, pandas as pd
from features.jump_features import jump_features


def test_columns_and_causality():
    idx = pd.date_range("2017-01-01", periods=800, freq="B")
    price = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 800)), index=idx)
    feats, _ = jump_features(price, 10, 20, 60, 120, burn_in=150)
    assert list(feats.columns) == ["downside_dev", "sortino_short", "sortino_long", "trend"]
    assert feats.notna().all().all()
    future = pd.Series([price.iloc[-1] * 1.5], index=[idx[-1] + pd.Timedelta(days=3)])
    feats2, _ = jump_features(pd.concat([price, future]), 10, 20, 60, 120, 150)
    common = feats.index.intersection(feats2.index)
    assert np.allclose(feats.loc[common].values, feats2.loc[common].values)
