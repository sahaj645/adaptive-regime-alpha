"""Result figures. Pure rendering from precomputed series - no model logic here."""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def equity_curves(curves: Dict[str, pd.Series], path: str | Path, title: str,
                  highlight: tuple = ()) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for label, ret in curves.items():
        eq = (1 + ret.dropna()).cumprod()
        ax.plot(eq.index, eq.values, label=label, lw=1.8 if label in highlight else 1.1)
    ax.set_yscale("log"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
