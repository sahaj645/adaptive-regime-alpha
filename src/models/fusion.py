"""Feature fusion: the combined model is the jump model fit on jump-model
features joined with the absorption-ratio features (level + standardized shift)."""
from __future__ import annotations
import pandas as pd


def fuse(jump_features: pd.DataFrame, ar_features: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on date; the union of columns is the combined feature set."""
    return jump_features.join(ar_features, how="inner").dropna()
