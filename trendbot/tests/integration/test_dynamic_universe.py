"""Integration tests for dynamic universe with backtest engine.

Tests the full pipeline: universe construction → backtest with universe →
correct liquidation trades, entry handling, and P&L attribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.backtest import run_backtest
from trendbot.domain.models import UniverseConfig, UniverseMode
from trendbot.domain.universe import (
    compute_universe_schedule,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_asset_backtest_data():
    """3 assets with varying volumes across 2 months."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    np.random.seed(42)
    close = pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(60) * 0.5),
        "B": 80 + np.cumsum(np.random.randn(60) * 0.5),
        "C": 60 + np.cumsum(np.random.randn(60) * 0.5),
    }, index=dates)
    volume = pd.DataFrame({
        "A": np.full(60, 1000.0),
        "B": np.full(60, 900.0),
        "C": np.full(60, 800.0),
    }, index=dates)
    return close, volume


@pytest.fixture
def universe_exit_data():
    """Asset B exits universe after month 1."""
    dates = pd.date_range("2022-12-01", periods=90, freq="D")
    np.random.seed(42)
    close = pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(90) * 0.5),
        "B": 80 + np.cumsum(np.random.randn(90) * 0.5),
        "C": 60 + np.cumsum(np.random.randn(90) * 0.5),
    }, index=dates)
    volume = pd.DataFrame({
        "A": np.full(90, 1000.0),
        "B": np.concatenate([np.full(32, 900.0), np.full(58, 10.0)]),
        "C": np.full(90, 800.0),
    }, index=dates)
    return close, volume


@pytest.fixture
def universe_entry_data():
    """Asset D enters universe in month 2 (not present in month 1)."""
    dates = pd.date_range("2022-12-01", periods=90, freq="D")
    np.random.seed(42)
    close = pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(90) * 0.5),
        "B": 80 + np.cumsum(np.random.randn(90) * 0.5),
        "D": np.concatenate([
            np.full(32, np.nan),
            60 + np.cumsum(np.random.randn(58) * 0.5),
        ]),
    }, index=dates)
    volume = pd.DataFrame({
        "A": np.full(90, 1000.0),
        "B": np.full(90, 900.0),
        "D": np.concatenate([np.full(32, np.nan), np.full(58, 800.0)]),
    }, index=dates)
    return close, volume


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestDynamicUniverseBacktest:
    def test_backtest_with_universe_schedule_returns_valid(
        self, multi_asset_backtest_data,
    ):
        close, volume = multi_asset_backtest_data
        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=2,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        result = run_backtest(
            close=close,
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
            universe_schedule=schedule,
        )
        assert len(result["returns"]) == len(close)
        assert result["returns"].notna().sum() > 0

    def test_backtest_without_schedule_backward_compatible(
        self, multi_asset_backtest_data,
    ):
        close, volume = multi_asset_backtest_data
        result_static = run_backtest(
            close=close,
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
        result_dynamic = run_backtest(
            close=close,
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
            universe_schedule=None,
        )
        pd.testing.assert_series_equal(
            result_static["returns"], result_dynamic["returns"],
        )

    def test_universe_exit_generates_liquidation(
        self, universe_exit_data,
    ):
        close, volume = universe_exit_data
        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=2,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        result = run_backtest(
            close=close,
            lookbacks=[5, 10],
            allow_short=True,
            vol_window=21,
            ann_factor=365,
            target_portfolio_vol=0.10,
            max_gross_leverage=1.0,
            taker_fee_pct=0.001,
            slippage_pct=0.0005,
            rebalance_threshold=0.00,
            min_history=10,
            universe_schedule=schedule,
        )

        positions = result["positions"]
        turnover = result["turnover"]
        assert len(positions) == len(close)
        assert turnover.sum() > 0

    def test_universe_entry_respects_history(
        self, universe_entry_data,
    ):
        close, volume = universe_entry_data
        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=2,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        result = run_backtest(
            close=close,
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
            universe_schedule=schedule,
        )

        positions = result["positions"]
        d_col = "D" in positions.columns
        if d_col:
            early_positions = positions.iloc[:31]["D"]
            assert (early_positions == 0).all()

    def test_costs_non_negative(self, multi_asset_backtest_data):
        close, volume = multi_asset_backtest_data
        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=2,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        result = run_backtest(
            close=close,
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
            universe_schedule=schedule,
        )
        assert (result["costs"] >= 0).all()


# ---------------------------------------------------------------------------
# End-to-end synthetic universe lifecycle test
# ---------------------------------------------------------------------------


class TestEndToEndUniverseLifecycle:
    def test_three_month_universe_transitions(self):
        dates = pd.date_range("2022-12-01", periods=120, freq="D")
        np.random.seed(42)
        close = pd.DataFrame({
            "A": 100 + np.cumsum(np.random.randn(120) * 0.5),
            "B": 80 + np.cumsum(np.random.randn(120) * 0.5),
            "C": 60 + np.cumsum(np.random.randn(120) * 0.5),
            "D": 40 + np.cumsum(np.random.randn(120) * 0.5),
            "E": 20 + np.cumsum(np.random.randn(120) * 0.5),
        }, index=dates)
        volume = pd.DataFrame({
            "A": np.concatenate([np.full(62, 1000), np.full(58, 100)]),
            "B": np.full(120, 900.0),
            "C": np.concatenate([np.full(92, 800), np.full(28, 1200)]),
            "D": np.concatenate([np.full(32, 200), np.full(30, 1100), np.full(58, 200)]),
            "E": np.full(120, 100.0),
        }, index=dates)

        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=3,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        jan = pd.Timestamp("2023-01-10")
        feb = pd.Timestamp("2023-02-10")
        mar = pd.Timestamp("2023-03-10")

        jan_u = set(schedule.get(jan, []))
        feb_u = set(schedule.get(feb, []))
        mar_u = set(schedule.get(mar, []))

        assert len(jan_u) == 3
        assert len(feb_u) == 3
        assert len(mar_u) == 3

        assert "A" in jan_u

        result = run_backtest(
            close=close,
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
            universe_schedule=schedule,
        )

        assert len(result["returns"]) == 120
        assert result["turnover"].sum() > 0
        assert (result["costs"] >= 0).all()
        assert (result["gross_returns"] - result["returns"] - result["costs"]).abs().max() < 1e-10
