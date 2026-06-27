"""Typed configuration loaded from a single YAML file. No parameter is defined
anywhere else in the codebase."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataCfg:
    etf_file: str; raw_pit_dir: str; processed_dir: str
    trade_asset: str; benchmark_asset: str; exclude_etfs: List[str]


@dataclass(frozen=True)
class ConstituentsCfg:
    gics_level: str; weighting: str; split_threshold: float
    winsor_low: float; winsor_high: float; max_gap_days: int


@dataclass(frozen=True)
class AbsorptionRatioCfg:
    window: int; n_components: int; shift_window: int; short_ma: int


@dataclass(frozen=True)
class JumpModelCfg:
    burn_in: int; n_states: int
    halflife_downside_dev: int; halflife_sortino_short: int
    halflife_sortino_long: int; halflife_trend: int
    lambda_grid: List[int]; cv_folds: int; cv_val_days: int
    cv_embargo: int; n_seeds: int


@dataclass(frozen=True)
class BacktestCfg:
    oos_start: str; first_train_year: int; last_year: int
    trade_delay: int; cost_bps: float


@dataclass(frozen=True)
class EvaluationCfg:
    bootstrap_n: int; block_size: int; trading_days: int


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataCfg
    constituents: ConstituentsCfg
    absorption_ratio: AbsorptionRatioCfg
    jump_model: JumpModelCfg
    backtest: BacktestCfg
    evaluation: EvaluationCfg
    results_dir: str

    @staticmethod
    def load(path: str | Path = ROOT / "config" / "default.yaml") -> "Config":
        d = yaml.safe_load(Path(path).read_text())
        return Config(
            seed=d["seed"],
            data=DataCfg(**d["data"]),
            constituents=ConstituentsCfg(**d["constituents"]),
            absorption_ratio=AbsorptionRatioCfg(**d["absorption_ratio"]),
            jump_model=JumpModelCfg(**d["jump_model"]),
            backtest=BacktestCfg(**d["backtest"]),
            evaluation=EvaluationCfg(**d["evaluation"]),
            results_dir=d["paths"]["results_dir"],
        )

    def path(self, *parts: str) -> Path:
        return ROOT.joinpath(*parts)
