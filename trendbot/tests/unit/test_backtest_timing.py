"""Deterministic tests for causal backtest timing."""

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.backtest import run_backtest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deterministic_prices():
    """Simple deterministic price series with clear trends."""
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    np.random.seed(99)
    # Asset A: steady uptrend
    # Asset B: flat then dip
    a_returns = np.concatenate([np.full(60, 0.002), np.full(60, -0.001)])
    b_returns = np.concatenate([np.full(60, 0.001), np.full(60, 0.0005)])
    a_prices = 100 * np.cumprod(1 + a_returns)
    b_prices = 50 * np.cumprod(1 + b_returns)
    return pd.DataFrame({"A": a_prices, "B": b_prices}, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_future_price_change_cannot_change_current_position(deterministic_prices):
    """Changing a price at t must not alter positions at t-1 or earlier."""
    close = deterministic_prices
    params = dict(
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
        covariance_window=60,
        covariance_shrinkage=0.1,
    )

    result_original = run_backtest(close=close, **params)

    # Modify the last price of asset A by a large factor
    close_modified = close.copy()
    close_modified.iloc[-1, close_modified.columns.get_loc("A")] *= 10.0

    result_modified = run_backtest(close=close_modified, **params)

    # All positions before the modified bar must be identical
    n_bars = len(close)
    # The modified bar is the last one; positions at n_bars-2 and earlier
    # must be the same (position at n_bars-2 depends only on data up to n_bars-2)
    pd.testing.assert_frame_equal(
        result_original["positions"].iloc[: n_bars - 2],
        result_modified["positions"].iloc[: n_bars - 2],
    )
    pd.testing.assert_frame_equal(
        result_original["executed_weights"].iloc[: n_bars - 2],
        result_modified["executed_weights"].iloc[: n_bars - 2],
    )


def test_position_earned_next_bar():
    """The return earned at t+1 must equal position[t] * return[t+1]."""
    # Create a scenario with explicit known prices
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    np.random.seed(7)
    prices = pd.DataFrame(
        {
            "A": 100 + np.cumsum(np.random.randn(80) * 0.5),
            "B": 50 + np.cumsum(np.random.randn(80) * 0.3),
        },
        index=dates,
    )

    result = run_backtest(
        close=prices,
        lookbacks=[5],
        allow_short=True,
        vol_window=21,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.0,
        slippage_pct=0.0,
        rebalance_threshold=0.0,
        min_history=30,
        covariance_window=60,
        covariance_shrinkage=0.1,
    )

    positions = result["positions"]
    gross_returns = result["gross_returns"]
    daily_rets = (prices / prices.shift(1) - 1).fillna(0.0)

    # For every bar where there is a position, gross_return[t] = position[t-1] * daily_return[t]
    for i in range(1, len(positions)):
        pos_prev = positions.iloc[i - 1].values
        ret_today = daily_rets.iloc[i].values
        expected_gross = float(np.dot(pos_prev, ret_today))
        if abs(expected_gross) > 1e-15:
            assert np.isclose(gross_returns.iloc[i], expected_gross, atol=1e-10), (
                f"Bar {i}: gross_return={gross_returns.iloc[i]:.8f} "
                f"!= position * return = {expected_gross:.8f}"
            )


def test_no_turnover_before_warmup(deterministic_prices):
    """Turnover and costs must be zero before required_history."""
    result = run_backtest(
        close=deterministic_prices,
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
        covariance_window=60,
        covariance_shrinkage=0.1,
    )

    # required_history = max(30, 10, 21, 60) = 60
    required_history = 60

    assert (result["turnover"].iloc[:required_history] == 0).all()
    assert (result["costs"].iloc[:required_history] == 0).all()
    # Positions are also zero before warmup
    assert (result["positions"].iloc[:required_history] == 0).all().all()
    # Returns are zero before warmup
    assert (result["returns"].iloc[:required_history] == 0).all()
    assert (result["gross_returns"].iloc[:required_history] == 0).all()


def test_zero_signal_produces_zero_position():
    """When all signals are zero, positions must remain zero."""
    # Constant flat prices -> signals will be zero after warmup
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    flat = pd.DataFrame(
        {"A": np.full(120, 100.0), "B": np.full(120, 50.0)},
        index=dates,
    )

    result = run_backtest(
        close=flat,
        lookbacks=[5, 10],
        allow_short=True,
        vol_window=21,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        rebalance_threshold=0.0,
        min_history=30,
        covariance_window=60,
        covariance_shrinkage=0.1,
    )

    # With flat prices, signals are zero -> positions should be zero
    required_history = 60
    assert (result["positions"].iloc[required_history:] == 0).all().all()
    assert (result["turnover"].iloc[required_history:] == 0).all()
    assert (result["costs"].iloc[required_history:] == 0).all()


def test_rebalance_threshold_does_not_trade():
    """When target weight changes by less than threshold, no trade occurs."""
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    np.random.seed(42)
    # Slowly drifting prices -> very small weight changes bar-to-bar
    drift = np.concatenate([np.full(120, 0.0001)])
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + drift),
            "B": 50 * np.cumprod(1 + drift * 0.5),
        },
        index=dates,
    )

    # Use a very high rebalance threshold so nothing ever triggers
    result = run_backtest(
        close=prices,
        lookbacks=[5, 10],
        allow_short=True,
        vol_window=21,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        rebalance_threshold=0.5,  # Very high threshold
        min_history=30,
        covariance_window=60,
        covariance_shrinkage=0.1,
    )

    required_history = 60
    # Once positions are established, turnover should be zero after the first
    # trade because the threshold is so high
    turnover_post = result["turnover"].iloc[required_history + 1 :]
    # Allow at most a handful of small trades from the initial position setup
    assert (turnover_post == 0).sum() / len(turnover_post) > 0.95
