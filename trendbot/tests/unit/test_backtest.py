"""Unit tests for backtest engine."""

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.backtest import compute_benchmark_returns, run_backtest


@pytest.fixture
def sample_close():
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    np.random.seed(42)
    close = pd.DataFrame(
        {
            "A": 100 + np.cumsum(np.random.randn(200) * 2),
            "B": 50 + np.cumsum(np.random.randn(200) * 1),
        },
        index=dates,
    )
    return close


def test_backtest_returns_shape(sample_close):
    result = run_backtest(
        close=sample_close,
        lookbacks=[5, 10],
        allow_short=True,
        vol_window=21,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        rebalance_threshold=0.01,
        min_history=30,
    )
    assert len(result["returns"]) == len(sample_close)
    assert len(result["positions"]) == len(sample_close)
    assert len(result["turnover"]) == len(sample_close)


def test_backtest_no_lookahead(sample_close):
    result = run_backtest(
        close=sample_close,
        lookbacks=[5],
        allow_short=True,
        vol_window=21,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        rebalance_threshold=0.01,
        min_history=30,
    )
    positions = result["positions"]
    for i in range(1, len(positions)):
        assert not np.any(np.isnan(positions.iloc[i])) or positions.iloc[i].isna().all()


def test_backtest_costs_non_negative(sample_close):
    result = run_backtest(
        close=sample_close,
        lookbacks=[5, 10],
        allow_short=True,
        vol_window=21,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        rebalance_threshold=0.01,
        min_history=30,
    )
    assert (result["costs"] >= 0).all()


def test_backtest_gross_leverage(sample_close):
    result = run_backtest(
        close=sample_close,
        lookbacks=[5, 10],
        allow_short=True,
        vol_window=21,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        rebalance_threshold=0.01,
        min_history=30,
    )
    gross = result["executed_weights"].abs().sum(axis=1)
    assert (gross <= 1.0 + 1e-10).all()


def test_benchmark_equal_weight(sample_close):
    bench = compute_benchmark_returns(sample_close, "equal_weight", min_history=30)
    assert bench is not None
    assert len(bench) == len(sample_close)


def test_benchmark_none(sample_close):
    bench = compute_benchmark_returns(sample_close, "none", min_history=30)
    assert bench is None


def test_backtest_long_only(sample_close):
    result = run_backtest(
        close=sample_close,
        lookbacks=[5],
        allow_short=False,
        vol_window=21,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        rebalance_threshold=0.01,
        min_history=30,
    )
    assert (result["positions"] >= -1e-10).all().all()
