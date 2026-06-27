"""Assemble model feature sets from configuration. The only place feature sets
are composed; downstream code just consumes the resulting frames."""
from __future__ import annotations
from typing import Tuple
import pandas as pd

from data.market_data import asset_price
from features.jump_features import jump_features
from models.absorption_ratio import absorption_ratio
from models.fusion import fuse


def jump_feature_set(cfg) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    price = asset_price(cfg.data.etf_file, cfg.data.trade_asset)
    j = cfg.jump_model
    feats, logret = jump_features(price, j.halflife_downside_dev, j.halflife_sortino_short,
                                  j.halflife_sortino_long, j.halflife_trend, j.burn_in)
    return feats, logret, price


def absorption_feature_set(cfg, returns_matrix: pd.DataFrame) -> pd.DataFrame:
    a = cfg.absorption_ratio
    return absorption_ratio(returns_matrix, a.window, a.n_components, a.shift_window, a.short_ma)


def fused_feature_set(cfg, returns_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    feats, logret, price = jump_feature_set(cfg)
    arf = absorption_feature_set(cfg, returns_matrix)
    return fuse(feats, arf), logret, price
