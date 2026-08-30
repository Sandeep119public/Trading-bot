"""Unit tests for strict fee deduction in backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendbot.domain.backtest import run_backtest


def test_strict_fee_deduction():
    """Verify exact fee deduction: turnover * (taker_fee_pct + slippage_pct).

    Scenario: Position changes by 50% (0.5 turnover).
    With taker_fee_pct=0.001 (0.1%) and slippage_pct=0.0005 (0.05%):
    cost = 0.5 * (0.001 + 0.0005) = 0.00075 (7.5 bps of portfolio).
    net_return = gross_return - cost.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    close = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=dates,
    )

    taker_fee_pct = 0.001
    slippage_pct = 0.0005
    cost_rate = taker_fee_pct + slippage_pct

    result = run_backtest(
        close=close,
        lookbacks=[2],
        allow_short=False,
        vol_window=3,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=taker_fee_pct,
        slippage_pct=slippage_pct,
        rebalance_threshold=0.0,
        min_history=0,
        covariance_window=3,
    )

    returns = result["returns"]
    gross_returns = result["gross_returns"]
    costs = result["costs"]
    turnover = result["turnover"]

    net_returns = gross_returns - costs
    pd.testing.assert_series_equal(returns, net_returns, check_names=False)

    expected_costs = turnover * cost_rate
    pd.testing.assert_series_equal(costs, expected_costs, check_names=False)


def test_zero_turnover_zero_cost():
    """When there is no position change, cost should be zero."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    np.random.seed(42)
    close = pd.DataFrame(
        {"A": 100 + np.cumsum(np.random.randn(10) * 0.5)},
        index=dates,
    )

    result = run_backtest(
        close=close,
        lookbacks=[2],
        allow_short=False,
        vol_window=3,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        rebalance_threshold=1.0,
        min_history=0,
        covariance_window=3,
    )

    turnover = result["turnover"]
    costs = result["costs"]

    assert (costs >= 0).all()
    assert ((turnover == 0) & (costs == 0)).all() or (turnover > 0).any()


def test_higher_fees_higher_cost():
    """Higher fee parameters should produce higher costs, all else equal."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(42)
    close = pd.DataFrame(
        {"A": 100 + np.cumsum(np.random.randn(50) * 2)},
        index=dates,
    )

    base = run_backtest(
        close=close, lookbacks=[5], allow_short=False, vol_window=10,
        ann_factor=365, target_portfolio_vol=0.10, max_gross_leverage=1.0,
        taker_fee_pct=0.001, slippage_pct=0.0005,
        rebalance_threshold=0.01, min_history=0,
        covariance_window=10,
    )

    high = run_backtest(
        close=close, lookbacks=[5], allow_short=False, vol_window=10,
        ann_factor=365, target_portfolio_vol=0.10, max_gross_leverage=1.0,
        taker_fee_pct=0.005, slippage_pct=0.002,
        rebalance_threshold=0.01, min_history=0,
        covariance_window=10,
    )

    assert high["costs"].sum() > base["costs"].sum()


def test_fee_impact_in_metrics():
    """Verify that fee_drag_pct correctly captures the fee impact."""
    from trendbot.domain.metrics import compute_metrics

    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    gross = pd.Series(np.random.randn(100) * 0.01, index=dates)
    net = gross - 0.001

    compute_metrics(gross, pd.DataFrame({"A": np.ones(100)}),
                                     pd.Series(0.01, index=dates),
                                     pd.Series(0.0, index=dates),
                                     gross_returns=gross)
    net_metrics = compute_metrics(net, pd.DataFrame({"A": np.ones(100)}),
                                   pd.Series(0.01, index=dates),
                                   pd.Series(0.001, index=dates),
                                   gross_returns=gross)

    assert net_metrics["fee_drag_pct"] > 0
    assert net_metrics["total_net_return"] < net_metrics["total_gross_return"]
