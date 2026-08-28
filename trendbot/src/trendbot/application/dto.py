"""Data transfer objects for the application layer."""

from __future__ import annotations

from pydantic import BaseModel

from trendbot.domain.models import (
    BacktestDataSelection,
    BacktestParams,
    BacktestRequest,
    BacktestResult,
    DataDownloadRequest,
    ExecutionParams,
    MomentumParams,
    RiskParams,
    VolatilityParams,
)


class DownloadResult(BaseModel):
    success: bool
    message: str
    symbols_processed: list[str]
    symbols_failed: list[str]


class BacktestDTO(BaseModel):
    request: BacktestRequest
    result: BacktestResult | None = None
    error: str | None = None


__all__ = [
    "DownloadResult",
    "BacktestDTO",
    "BacktestRequest",
    "BacktestResult",
    "DataDownloadRequest",
    "MomentumParams",
    "VolatilityParams",
    "RiskParams",
    "ExecutionParams",
    "BacktestParams",
    "BacktestDataSelection",
]
