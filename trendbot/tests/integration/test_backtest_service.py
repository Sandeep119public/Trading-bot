"""Integration tests for backtest service."""

import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trendbot.application.backtest_service import BacktestService
from trendbot.domain.models import (
    BacktestDataSelection,
    BacktestParams,
    BacktestRequest,
    ExecutionParams,
    MomentumParams,
    RiskParams,
    VolatilityParams,
)
from trendbot.infrastructure.repositories.parquet_price_repository import ParquetPriceRepository


@pytest.fixture
def backtest_service():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = ParquetPriceRepository(tmpdir)

        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        np.random.seed(42)
        for sym in ["A", "B"]:
            prices = 100 + np.cumsum(np.random.randn(200) * 2)
            df = pd.DataFrame({"close": prices}, index=dates)
            df.index.name = "date"
            repo.save_prices("test", sym, "1d", df)

        svc = BacktestService(repo)
        yield svc


def test_full_backtest(backtest_service):
    request = BacktestRequest(
        data_selection=BacktestDataSelection(
            source="test",
            symbols=["A", "B"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 7, 19),
        ),
        momentum=MomentumParams(lookbacks=[5, 10], allow_short=True),
        volatility=VolatilityParams(vol_window=21, ann_factor=365),
        risk=RiskParams(target_portfolio_vol=0.10, max_gross_leverage=1.0),
        execution=ExecutionParams(taker_fee_pct=0.001, slippage_pct=0.0005, rebalance_threshold=0.01),
        backtest=BacktestParams(min_history=30, benchmark="equal_weight"),
    )

    dto = backtest_service.run(request)
    assert dto.error is None
    assert dto.result is not None
    assert "total_return" in dto.result.stats
    assert len(dto.result.returns) > 0


def test_backtest_no_data(backtest_service):
    request = BacktestRequest(
        data_selection=BacktestDataSelection(
            source="test",
            symbols=["NONEXISTENT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 7, 19),
        ),
        momentum=MomentumParams(lookbacks=[5], allow_short=True),
        volatility=VolatilityParams(vol_window=21, ann_factor=365),
        risk=RiskParams(target_portfolio_vol=0.10, max_gross_leverage=1.0),
        execution=ExecutionParams(taker_fee_pct=0.001, slippage_pct=0.0005, rebalance_threshold=0.01),
        backtest=BacktestParams(min_history=30, benchmark="none"),
    )

    dto = backtest_service.run(request)
    assert dto.error is not None
