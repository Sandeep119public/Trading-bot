"""Domain models for the trend-following backtesting system."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class BenchmarkType(StrEnum):
    NONE = "none"
    EQUAL_WEIGHT = "equal_weight"


class DataDownloadRequest(BaseModel):
    source: str
    symbols: list[str]
    start_date: date
    end_date: date | None = None
    overwrite: bool = False


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
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    rebalance_threshold: float = 0.01

    @field_validator("fee_bps", "slippage_bps")
    @classmethod
    def validate_bps(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Fee and slippage bps must be non-negative")
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

    @field_validator("min_history")
    @classmethod
    def validate_min_history(cls, v: int) -> int:
        if v < 0:
            raise ValueError("min_history must be non-negative")
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


class BacktestResult(BaseModel):
    stats: dict[str, float]
    returns: object  # pd.Series
    positions: object  # pd.DataFrame
    executed_weights: object  # pd.DataFrame
    turnover: object  # pd.Series
    costs: object  # pd.Series
    benchmark_returns: object | None  # pd.Series or None
    metadata: dict[str, str]


def load_defaults(config_path: Path | None = None) -> dict:
    """Load default configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "defaults.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
