"""Unit tests for signal generation."""

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.signals import compute_momentum_signals


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = pd.DataFrame(
        {
            "A": 100 + np.cumsum(np.random.randn(100) * 2),
            "B": 50 + np.cumsum(np.random.randn(100) * 1),
        },
        index=dates,
    )
    return prices


def test_signal_shape(sample_prices):
    signals = compute_momentum_signals(sample_prices, [5, 10], allow_short=True)
    assert signals.shape == sample_prices.shape


def test_signal_values_range(sample_prices):
    signals = compute_momentum_signals(sample_prices, [5, 10], allow_short=True)
    assert signals.max().max() <= 1.0
    assert signals.min().min() >= -1.0


def test_long_only_clipping(sample_prices):
    signals = compute_momentum_signals(sample_prices, [5, 10], allow_short=False)
    assert signals.min().min() >= 0.0


def test_single_lookback(sample_prices):
    signals = compute_momentum_signals(sample_prices, [5], allow_short=True)
    assert signals.max().max() <= 1.0
    assert signals.min().min() >= -1.0


def test_multiple_lookbacks(sample_prices):
    signals = compute_momentum_signals(sample_prices, [5, 10, 21], allow_short=True)
    expected_range = (-1.0, 1.0)
    assert signals.max().max() <= expected_range[1]
    assert signals.min().min() >= expected_range[0]


def test_nan_handling():
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = pd.DataFrame(
        {"A": [np.nan] * 10 + list(range(40))},
        index=dates,
    )
    signals = compute_momentum_signals(prices, [5], allow_short=True)
    assert signals.shape == prices.shape
