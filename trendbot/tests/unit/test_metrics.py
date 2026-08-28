"""Unit tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.metrics import (
    compute_drawdown_series,
    compute_metrics,
    compute_monthly_returns,
)


@pytest.fixture
def sample_returns():
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    return returns


@pytest.fixture
def sample_positions():
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)
    positions = pd.DataFrame(
        {"A": np.random.randn(252) * 0.3, "B": np.random.randn(252) * 0.2},
        index=dates,
    )
    return positions


def test_metrics_keys(sample_returns, sample_positions):
    turnover = pd.Series(0.01, index=sample_returns.index)
    costs = pd.Series(0.001, index=sample_returns.index)
    metrics = compute_metrics(sample_returns, sample_positions, turnover, costs)
    expected_keys = [
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "daily_win_rate",
        "avg_gross_exposure",
        "avg_daily_turnover",
        "total_cost_drag",
        "total_gross_return",
        "total_net_return",
        "fee_drag_pct",
    ]
    for key in expected_keys:
        assert key in metrics


def test_metrics_values_valid(sample_returns, sample_positions):
    turnover = pd.Series(0.01, index=sample_returns.index)
    costs = pd.Series(0.001, index=sample_returns.index)
    metrics = compute_metrics(sample_returns, sample_positions, turnover, costs)
    assert isinstance(metrics["total_return"], float)
    assert isinstance(metrics["sharpe_ratio"], float)
    assert metrics["max_drawdown"] <= 0
    assert 0 <= metrics["daily_win_rate"] <= 1


def test_drawdown_series(sample_returns):
    dd = compute_drawdown_series(sample_returns)
    assert len(dd) == len(sample_returns)
    assert dd.max() <= 0
    assert dd.min() >= -1


def test_monthly_returns(sample_returns):
    monthly = compute_monthly_returns(sample_returns)
    assert monthly.shape[0] > 0
    assert monthly.shape[1] <= 12


def test_empty_returns():
    dates = pd.date_range("2023-01-01", periods=0, freq="D")
    empty = pd.Series(dtype=float, index=dates)
    positions = pd.DataFrame(dtype=float, index=dates)
    turnover = pd.Series(dtype=float, index=dates)
    costs = pd.Series(dtype=float, index=dates)
    metrics = compute_metrics(empty, positions, turnover, costs)
    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
