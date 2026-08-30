"""Domain models for the trend-following backtesting system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkType(StrEnum):
    NONE = "none"
    EQUAL_WEIGHT = "equal_weight"


class DataDownloadRequest(BaseModel):
    source: str
    symbols: list[str]
    start_date: date
    end_date: date | None = None
    overwrite: bool = False
    timeframe: str = "1d"
    quote_currency: str = "USDT"


class BacktestDataSelection(BaseModel):
    source: str
    symbols: list[str]
    start_date: date
    end_date: date | None = None
    timeframe: str = "1d"


class MomentumParams(BaseModel):
    lookbacks: list[int] = Field(default=[5, 10, 21, 42])
    allow_short: bool = True

    @field_validator("lookbacks")
    @classmethod
    def validate_lookbacks(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("At least one lookback period is required")
        if any(lb <= 0 for lb in v):
            raise ValueError("All lookback periods must be positive")
        return v


class VolatilityParams(BaseModel):
    vol_window: int = 21
    ann_factor: int = 365

    @field_validator("vol_window")
    @classmethod
    def validate_vol_window(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("vol_window must be positive")
        return v

    @field_validator("ann_factor")
    @classmethod
    def validate_ann_factor(cls, v: int) -> int:
        if v not in (252, 365):
            raise ValueError("ann_factor must be 252 or 365")
        return v


class RiskParams(BaseModel):
    target_portfolio_vol: float = 0.10
    max_gross_leverage: float = 1.0

    @field_validator("target_portfolio_vol")
    @classmethod
    def validate_target_vol(cls, v: float) -> float:
        if v <= 0 or v > 2.0:
            raise ValueError("target_portfolio_vol must be in (0, 2.0]")
        return v

    @field_validator("max_gross_leverage")
    @classmethod
    def validate_max_leverage(cls, v: float) -> float:
        if v <= 0 or v > 10.0:
            raise ValueError("max_gross_leverage must be in (0, 10.0]")
        return v


class ExecutionParams(BaseModel):
    taker_fee_pct: float = 0.001
    maker_fee_pct: float = 0.0005
    slippage_pct: float = 0.0005
    rebalance_threshold: float = 0.01

    @field_validator("taker_fee_pct", "maker_fee_pct", "slippage_pct")
    @classmethod
    def validate_pct_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Fee and slippage percentages must be non-negative")
        return v

    @field_validator("rebalance_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v < 0 or v > 1.0:
            raise ValueError("rebalance_threshold must be in [0, 1.0]")
        return v


class BacktestParams(BaseModel):
    min_history: int = 60
    benchmark: BenchmarkType = BenchmarkType.EQUAL_WEIGHT
    covariance_window: int = 60
    covariance_shrinkage: float = 0.1

    @field_validator("min_history")
    @classmethod
    def validate_min_history(cls, v: int) -> int:
        if v < 0:
            raise ValueError("min_history must be non-negative")
        return v

    @field_validator("covariance_window")
    @classmethod
    def validate_covariance_window(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("covariance_window must be positive")
        return v

    @field_validator("covariance_shrinkage")
    @classmethod
    def validate_covariance_shrinkage(cls, v: float) -> float:
        if v < 0 or v > 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1.0]")
        return v


class BacktestRequest(BaseModel):
    data_selection: BacktestDataSelection
    momentum: MomentumParams
    volatility: VolatilityParams
    risk: RiskParams
    execution: ExecutionParams
    backtest: BacktestParams


class DatasetMetadata(BaseModel):
    source: str
    symbol: str
    timeframe: str
    start_date: date
    end_date: date
    rows: int
    downloaded_at: str
    last_updated: str
    file_path: str


# ---------------------------------------------------------------------------
# Universe configuration
# ---------------------------------------------------------------------------


class UniverseMode(StrEnum):
    STATIC = "static"
    DYNAMIC_TOP_N = "dynamic_top_n"


class UniverseConfig(BaseModel):
    """Configuration for historical universe reconstruction."""

    mode: UniverseMode = Field(
        default=UniverseMode.STATIC,
        description="Universe selection mode: static list or dynamic top-N by liquidity.",
    )
    symbols: list[str] = Field(
        default_factory=list,
        description="Static symbol list (used when mode=static).",
    )
    top_n: int = Field(
        default=50,
        description="Number of assets in dynamic top-N universe.",
    )
    liquidity_window: int = Field(
        default=21,
        description="Trailing window in days for rolling dollar volume.",
    )
    rebalance_frequency: str = Field(
        default="monthly",
        description="Reconstitution frequency: 'monthly' only for now.",
    )
    rebalance_day: int = Field(
        default=1,
        description="Day of month for universe reconstitution.",
    )
    exclude_stablecoins: bool = Field(
        default=True,
        description="Whether to exclude stablecoin assets from universe.",
    )
    min_volume_days: int = Field(
        default=15,
        description=(
            "Minimum number of valid volume observations in the trailing "
            "window for an asset to be ranked."
        ),
    )

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_n must be positive")
        return v

    @field_validator("liquidity_window")
    @classmethod
    def validate_liquidity_window(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("liquidity_window must be positive")
        return v

    @field_validator("rebalance_frequency")
    @classmethod
    def validate_rebalance_frequency(cls, v: str) -> str:
        if v not in ("monthly",):
            raise ValueError("rebalance_frequency must be 'monthly'")
        return v

    @field_validator("rebalance_day")
    @classmethod
    def validate_rebalance_day(cls, v: int) -> int:
        if v < 1 or v > 28:
            raise ValueError("rebalance_day must be between 1 and 28")
        return v

    @field_validator("min_volume_days")
    @classmethod
    def validate_min_volume_days(cls, v: int) -> int:
        if v < 0:
            raise ValueError("min_volume_days must be non-negative")
        return v


# ---------------------------------------------------------------------------
# Backtest universe diagnostics
# ---------------------------------------------------------------------------


class UniverseSnapshot(BaseModel):
    """Snapshot of universe state at a single rebalance date."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    date: pd.Timestamp
    members: list[str]
    rankings: dict[str, int]
    dollar_volumes_21d: dict[str, float]
    entries: list[str]
    exits: list[str]


class BacktestResult(BaseModel):
    stats: dict[str, float]
    returns: object
    gross_returns: object
    positions: object
    executed_weights: object
    turnover: object
    costs: object
    benchmark_returns: object | None
    metadata: dict[str, str]


def load_defaults(config_path: Path | None = None) -> dict:
    """Load default configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "defaults.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_paths_config(config_path: Path | None = None) -> dict[str, str]:
    """Load local path configuration from paths.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "paths.yaml"
    if not config_path.exists():
        return {"data_dir": "./data", "output_dir": "./output"}
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return {
        "data_dir": cfg.get("data_dir", "./data"),
        "output_dir": cfg.get("output_dir", "./output"),
    }


# ---------------------------------------------------------------------------
# Walk-Forward Optimization models
# ---------------------------------------------------------------------------


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward out-of-sample validation."""

    train_window: int = Field(default=756, description="Training window length in bars.")
    test_window: int = Field(default=126, description="Test window length in bars.")
    step: int = Field(default=126, description="Step size in bars between folds.")
    minimum_training_bars: int = Field(
        default=126,
        description="Minimum training bars required before a fold is valid.",
    )
    minimum_training_observations: int = Field(
        default=126,
        description="Minimum post-warmup return observations used for parameter selection.",
    )
    minimum_training_trades: int = Field(
        default=1,
        description="Minimum number of executed trades required for a candidate to be eligible.",
    )
    sharpe_tie_tolerance: float = Field(
        default=0.02,
        description=(
            "Training Sharpe difference treated as"
            " statistically indistinguishable for tie-breaking."
        ),
    )
    allow_short: bool = Field(default=True, description="Whether to allow short positions.")
    ann_factor: int = Field(default=365, description="Annualization factor (252 or 365).")
    target_portfolio_vol: float = Field(default=0.10)
    max_gross_leverage: float = Field(default=1.0)
    taker_fee_pct: float = Field(default=0.001)
    slippage_pct: float = Field(default=0.0005)
    min_history: int = Field(default=60)

    @field_validator("ann_factor")
    @classmethod
    def validate_ann_factor(cls, v: int) -> int:
        if v not in (252, 365):
            raise ValueError("ann_factor must be 252 or 365")
        return v

    @field_validator("train_window", "test_window", "step")
    @classmethod
    def validate_positive_windows(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("WFO window and step values must be positive")
        return v

    @field_validator("minimum_training_bars", "minimum_training_observations")
    @classmethod
    def validate_non_negative_counts(cls, v: int) -> int:
        if v < 0:
            raise ValueError("training observation requirements must be non-negative")
        return v

    @field_validator("minimum_training_trades")
    @classmethod
    def validate_training_trades(cls, v: int) -> int:
        if v < 0:
            raise ValueError("minimum_training_trades must be non-negative")
        return v

    @field_validator("sharpe_tie_tolerance")
    @classmethod
    def validate_sharpe_tolerance(cls, v: float) -> float:
        if v < 0:
            raise ValueError("sharpe_tie_tolerance must be non-negative")
        return v

    @field_validator("min_history")
    @classmethod
    def validate_min_history(cls, v: int) -> int:
        if v < 0:
            raise ValueError("min_history must be non-negative")
        return v


class ParameterGrid(BaseModel):
    """Explicit parameter search grid for WFO optimization."""

    lookbacks: list[list[int]] = Field(
        default=[[5, 10, 21, 42], [10, 21, 42, 84], [21, 42, 84, 126]],
    )
    vol_window: list[int] = Field(default=[20, 40, 60])
    covariance_window: list[int] = Field(default=[40, 60, 120])
    covariance_shrinkage: list[float] = Field(default=[0.0, 0.1, 0.25, 0.5])
    rebalance_threshold: list[float] = Field(default=[0.0, 0.005, 0.01])


@dataclass(frozen=True)
class WalkForwardFold:
    """Immutable definition of a single walk-forward fold."""

    fold_index: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int

    @property
    def train_length(self) -> int:
        return self.train_end_idx - self.train_start_idx

    @property
    def test_length(self) -> int:
        return self.test_end_idx - self.test_start_idx


@dataclass(frozen=True)
class FoldResult:
    """Immutable result of a single walk-forward fold."""

    fold: WalkForwardFold
    selected_parameters: dict[str, object]
    oos_returns: pd.Series
    oos_equity: pd.Series
    oos_gross_returns: pd.Series
    oos_turnover: pd.Series
    oos_costs: pd.Series
    oos_positions: pd.DataFrame
    training_sharpe: float
    training_metrics: dict[str, float]
    oos_metrics: dict[str, float]


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward validation report."""

    folds: list[FoldResult]
    stitched_oos_returns: pd.Series
    stitched_oos_equity: pd.Series
    aggregate_metrics: dict[str, float]
    parameter_stability: dict[str, dict[object, int]]
    per_fold_summary: list[dict[str, object]]
