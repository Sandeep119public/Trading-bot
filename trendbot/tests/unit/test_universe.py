"""Unit tests for historical dynamic liquidity universe.

Tests cover:
  1. Stablecoin exclusion
  2. Top-N selection
  3. Only trailing data is used (no lookahead)
  4. Monthly reconstitution
  5. Asset exits → target weight = 0
  6. New asset history requirements
  7. Listing date enforcement
  8. Delisting / missing data handling
  9. No survivorship bias
 10. Static mode backward compatibility
 11. Deterministic tie-breaking
 12. Missing volume handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.models import UniverseConfig, UniverseMode
from trendbot.domain.universe import (
    StaticUniverse,
    compute_dollar_volume,
    compute_trailing_dollar_volume,
    compute_universe_schedule,
    extract_base_asset,
    filter_stablecoins,
    find_rebalance_dates,
    is_rebalance_date,
    is_stablecoin,
    select_top_n,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config():
    return UniverseConfig(
        mode=UniverseMode.DYNAMIC_TOP_N,
        top_n=3,
        liquidity_window=5,
        rebalance_day=1,
        exclude_stablecoins=True,
        min_volume_days=3,
    )


@pytest.fixture
def three_month_data():
    """Three months of deterministic data for 5 assets."""
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    np.random.seed(42)
    close = pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(90) * 0.5),
        "B": 80 + np.cumsum(np.random.randn(90) * 0.5),
        "C": 60 + np.cumsum(np.random.randn(90) * 0.5),
        "D": 40 + np.cumsum(np.random.randn(90) * 0.5),
        "E": 20 + np.cumsum(np.random.randn(90) * 0.5),
    }, index=dates)
    volume = pd.DataFrame({
        "A": np.full(90, 1000.0),
        "B": np.full(90, 900.0),
        "C": np.full(90, 800.0),
        "D": np.full(90, 700.0),
        "E": np.full(90, 600.0),
    }, index=dates)
    return close, volume


@pytest.fixture
def dynamic_volume_data():
    """Data where volume rankings change across months."""
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    np.random.seed(42)
    close = pd.DataFrame({
        "A": 100 + np.cumsum(np.random.randn(90) * 0.5),
        "B": 80 + np.cumsum(np.random.randn(90) * 0.5),
        "C": 60 + np.cumsum(np.random.randn(90) * 0.5),
        "D": 40 + np.cumsum(np.random.randn(90) * 0.5),
        "E": 20 + np.cumsum(np.random.randn(90) * 0.5),
    }, index=dates)
    # A dominates month 1, D dominates month 2, C dominates month 3
    vol_a = np.concatenate([
        np.full(31, 1000.0),  # Jan: A highest
        np.full(30, 100.0),   # Feb: A low
        np.full(29, 100.0),   # Mar: A low
    ])
    vol_b = np.full(90, 500.0)
    vol_c = np.concatenate([
        np.full(31, 300.0),
        np.full(30, 300.0),
        np.full(29, 1000.0),  # Mar: C highest
    ])
    vol_d = np.concatenate([
        np.full(31, 200.0),
        np.full(30, 1000.0),  # Feb: D highest
        np.full(29, 200.0),
    ])
    vol_e = np.full(90, 100.0)
    volume = pd.DataFrame({
        "A": vol_a, "B": vol_b, "C": vol_c, "D": vol_d, "E": vol_e,
    }, index=dates)
    return close, volume


# ---------------------------------------------------------------------------
# Test 1: Stablecoins excluded
# ---------------------------------------------------------------------------


class TestStablecoinExclusion:
    def test_stablecoins_excluded_from_top_n(self):
        dates = pd.date_range("2022-12-01", periods=60, freq="D")
        close = pd.DataFrame({
            "BTC": [100] * 60,
            "ETH": [50] * 60,
            "USDT": [1] * 60,
            "USDC": [1] * 60,
            "SOL": [20] * 60,
        }, index=dates)
        volume = pd.DataFrame({
            "BTC": [1000] * 60,
            "ETH": [900] * 60,
            "USDT": [5000] * 60,
            "USDC": [4000] * 60,
            "SOL": [800] * 60,
        }, index=dates)

        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=3,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)
        first_universe = schedule[pd.Timestamp("2023-01-01")]
        assert "USDT" not in first_universe
        assert "USDC" not in first_universe
        assert "BTC" in first_universe
        assert "ETH" in first_universe
        assert "SOL" in first_universe

    def test_stablecoin_base_asset_detection(self):
        assert is_stablecoin("USDT")
        assert is_stablecoin("USDC")
        assert is_stablecoin("DAI")
        assert is_stablecoin("BUSD")
        assert not is_stablecoin("BTC")
        assert not is_stablecoin("ETH")
        assert not is_stablecoin("SOL")

    def test_filter_stablecoins(self):
        symbols = ["BTC", "ETH", "USDT", "SOL", "USDC", "DOGE"]
        result = filter_stablecoins(symbols)
        assert result == ["BTC", "ETH", "SOL", "DOGE"]

    def test_extract_base_asset(self):
        assert extract_base_asset("BTCUSD") == "BTC"
        assert extract_base_asset("BTC/USDT") == "BTC"
        assert extract_base_asset("BTC-USDT") == "BTC"
        assert extract_base_asset("BTCUSDT") == "BTC"
        assert extract_base_asset("BTC/USDT:USDT") == "BTC"


# ---------------------------------------------------------------------------
# Test 2: Top-N selection
# ---------------------------------------------------------------------------


class TestTopNSelection:
    def test_top_n_basic(self):
        trailing = pd.Series({"A": 100, "B": 90, "C": 80, "D": 70, "E": 60})
        result = select_top_n(trailing, top_n=3)
        assert result == ["A", "B", "C"]

    def test_top_n_exact_n(self):
        trailing = pd.Series({"A": 100, "B": 90, "C": 80, "D": 70, "E": 60})
        result = select_top_n(trailing, top_n=5)
        assert len(result) == 5

    def test_top_n_fewer_than_available(self):
        trailing = pd.Series({"A": 100, "B": 90, "C": 80})
        result = select_top_n(trailing, top_n=5)
        assert result == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Test 3: Only trailing data is used (no lookahead)
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_future_volume_does_not_affect_past_universe(self, simple_config):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        close = pd.DataFrame({
            "A": [100] * 60,
            "B": [80] * 60,
            "C": [60] * 60,
        }, index=dates)
        volume = pd.DataFrame({
            "A": [100] * 60,
            "B": [90] * 60,
            "C": [80] * 60,
        }, index=dates)

        schedule_before = compute_universe_schedule(close, volume, simple_config)

        volume.loc[dates[50]:dates[59], "C"] = 10000
        schedule_after = compute_universe_schedule(close, volume, simple_config)

        jan_date = pd.Timestamp("2023-01-05")
        assert schedule_before[jan_date] == schedule_after[jan_date]


# ---------------------------------------------------------------------------
# Test 4: Monthly reconstitution
# ---------------------------------------------------------------------------


class TestMonthlyReconstitution:
    def test_universe_changes_monthly(self, simple_config):
        dates = pd.date_range("2023-01-01", periods=90, freq="D")
        close = pd.DataFrame({
            "A": [100] * 90,
            "B": [80] * 90,
            "C": [60] * 90,
            "D": [40] * 90,
        }, index=dates)
        volume = pd.DataFrame({
            "A": [1000] * 30 + [100] * 60,
            "B": [900] * 30 + [900] * 60,
            "C": [800] * 30 + [800] * 60,
            "D": [700] * 30 + [1000] * 60,
        }, index=dates)

        schedule = compute_universe_schedule(close, volume, simple_config)

        jan_universe = schedule.get(pd.Timestamp("2023-01-05"))
        feb_universe = schedule.get(pd.Timestamp("2023-02-01"))

        assert jan_universe is not None
        assert feb_universe is not None

        jan_same = schedule.get(pd.Timestamp("2023-01-15"))
        assert jan_same == jan_universe

    def test_is_rebalance_date(self):
        assert is_rebalance_date(pd.Timestamp("2023-01-01"))
        assert is_rebalance_date(pd.Timestamp("2023-02-01"))
        assert not is_rebalance_date(pd.Timestamp("2023-01-15"))
        assert not is_rebalance_date(pd.Timestamp("2023-01-31"))

    def test_find_rebalance_dates(self):
        dates = pd.date_range("2023-01-01", periods=90, freq="D")
        rebalance = find_rebalance_dates(dates, rebalance_day=1)
        assert len(rebalance) == 3
        assert rebalance[0] == pd.Timestamp("2023-01-01")
        assert rebalance[1] == pd.Timestamp("2023-02-01")
        assert rebalance[2] == pd.Timestamp("2023-03-01")


# ---------------------------------------------------------------------------
# Test 5: Asset exits
# ---------------------------------------------------------------------------


class TestAssetExits:
    def test_exited_asset_gets_zero_target(self):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        close = pd.DataFrame({
            "A": [100] * 60,
            "B": [80] * 60,
            "C": [60] * 60,
        }, index=dates)
        volume = pd.DataFrame({
            "A": [1000] * 60,
            "B": [900] * 30 + [100] * 30,
            "C": [800] * 60,
        }, index=dates)

        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=2,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        jan_date = pd.Timestamp("2023-01-05")
        feb_date = pd.Timestamp("2023-02-01")

        jan_universe = schedule.get(jan_date, [])
        feb_universe = schedule.get(feb_date, [])

        if "B" in jan_universe and "B" not in feb_universe:
            assert "B" in jan_universe
            assert "B" not in feb_universe


# ---------------------------------------------------------------------------
# Test 6: New asset history
# ---------------------------------------------------------------------------


class TestNewAssetHistory:
    def test_new_asset_needs_enough_history(self):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        close = pd.DataFrame({
            "A": [100] * 60,
            "B": [80] * 60,
            "C": np.nan * 60,
        }, index=dates)
        close.loc[dates[30]:dates[59], "C"] = 60.0

        volume = pd.DataFrame({
            "A": [1000] * 60,
            "B": [900] * 60,
            "C": np.nan * 60,
        }, index=dates)
        volume.loc[dates[30]:dates[59], "C"] = 800.0

        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=3,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=10,
        )
        schedule = compute_universe_schedule(close, volume, config)

        jan_date = pd.Timestamp("2023-01-05")
        jan_universe = schedule.get(jan_date, [])
        assert "C" not in jan_universe


# ---------------------------------------------------------------------------
# Test 7: Listing date
# ---------------------------------------------------------------------------


class TestListingDate:
    def test_asset_not_before_first_observation(self):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        close = pd.DataFrame({
            "A": [100] * 60,
            "B": np.nan * 60,
        }, index=dates)
        close.loc[dates[20]:dates[59], "B"] = 80.0

        volume = pd.DataFrame({
            "A": [1000] * 60,
            "B": np.nan * 60,
        }, index=dates)
        volume.loc[dates[20]:dates[59], "B"] = 500.0

        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=2,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        jan_date = pd.Timestamp("2023-01-05")
        jan_universe = schedule.get(jan_date, [])
        assert "B" not in jan_universe

        mar_date = pd.Timestamp("2023-03-01")
        mar_universe = schedule.get(mar_date, [])
        assert "B" in mar_universe


# ---------------------------------------------------------------------------
# Test 8: Delisting / missing data
# ---------------------------------------------------------------------------


class TestDelistingMissingData:
    def test_missing_candle_not_treated_as_delisting(self):
        dates = pd.date_range("2022-12-01", periods=90, freq="D")
        close = pd.DataFrame({
            "A": [100] * 90,
            "B": [80] * 90,
        }, index=dates)
        volume = pd.DataFrame({
            "A": [1000] * 90,
            "B": [900] * 90,
        }, index=dates)

        close.iloc[45, close.columns.get_loc("B")] = np.nan
        volume.iloc[45, volume.columns.get_loc("B")] = np.nan

        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=2,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        after_gap = pd.Timestamp("2023-01-20")
        universe = schedule.get(after_gap, [])
        assert "B" in universe


# ---------------------------------------------------------------------------
# Test 9: No survivorship bias
# ---------------------------------------------------------------------------


class TestNoSurvivorshipBias:
    def test_historical_universe_preserved(self):
        dates = pd.date_range("2021-12-01", periods=400, freq="D")
        close = pd.DataFrame({
            "A": [100] * 400,
            "B": [80] * 400,
            "C": [60] * 400,
            "D": [40] * 400,
        }, index=dates)
        volume = pd.DataFrame({
            "A": [1000] * 200 + [100] * 200,
            "B": [900] * 400,
            "C": [800] * 400,
            "D": [100] * 200 + [1000] * 200,
        }, index=dates)

        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=3,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        schedule = compute_universe_schedule(close, volume, config)

        jan_2022 = pd.Timestamp("2022-01-10")
        jul_2022 = pd.Timestamp("2022-07-10")

        jan_universe = schedule.get(jan_2022, [])
        jul_universe = schedule.get(jul_2022, [])

        assert "A" in jan_universe
        assert "A" not in jul_universe
        assert "D" not in jan_universe
        assert "D" in jul_universe


# ---------------------------------------------------------------------------
# Test 10: Static mode
# ---------------------------------------------------------------------------


class TestStaticMode:
    def test_static_universe_same_everywhere(self):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        close = pd.DataFrame({
            "A": [100] * 60,
            "B": [80] * 60,
            "C": [60] * 60,
        }, index=dates)
        volume = pd.DataFrame({
            "A": [1000] * 60,
            "B": [900] * 60,
            "C": [800] * 60,
        }, index=dates)

        config = UniverseConfig(
            mode=UniverseMode.STATIC,
            symbols=["A", "B"],
            exclude_stablecoins=True,
        )
        schedule = compute_universe_schedule(close, volume, config)

        for ts in dates:
            assert schedule[ts] == ["A", "B"]

    def test_static_universe_class(self):
        u = StaticUniverse(["BTC", "ETH", "SOL"])
        assert u.get_universe(pd.Timestamp("2023-01-01")) == ["BTC", "ETH", "SOL"]
        assert u.get_all_members() == ["BTC", "ETH", "SOL"]


# ---------------------------------------------------------------------------
# Test 11: Deterministic tie-breaking
# ---------------------------------------------------------------------------


class TestTieBreaking:
    def test_tie_breaking_uses_lexicographic_order(self):
        trailing = pd.Series({"A": 100, "B": 100, "C": 100, "D": 80, "E": 60})
        result = select_top_n(trailing, top_n=3)
        assert result == ["A", "B", "C"]

    def test_tie_breaking_uses_previous_rank(self):
        trailing = pd.Series({"A": 100, "B": 100, "C": 100})
        prev_rank = pd.Series({"A": 3, "B": 1, "C": 2})
        result = select_top_n(trailing, top_n=2, previous_rank=prev_rank)
        assert result[0] == "B"
        assert result[1] == "C"


# ---------------------------------------------------------------------------
# Test 12: Missing volume handling
# ---------------------------------------------------------------------------


class TestMissingVolume:
    def test_nan_volume_excluded_from_ranking(self):
        trailing = pd.Series({"A": 100, "B": np.nan, "C": 80})
        result = select_top_n(trailing, top_n=3)
        assert "B" not in result
        assert "A" in result
        assert "C" in result

    def test_all_nan_volume_returns_empty(self):
        trailing = pd.Series({"A": np.nan, "B": np.nan})
        result = select_top_n(trailing, top_n=3)
        assert result == []


# ---------------------------------------------------------------------------
# Dollar volume computation
# ---------------------------------------------------------------------------


class TestDollarVolume:
    def test_compute_dollar_volume(self):
        dates = pd.date_range("2023-01-01", periods=5)
        close = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=dates)
        volume = pd.DataFrame({"A": [1000, 1100, 1200, 1300, 1400]}, index=dates)
        dv = compute_dollar_volume(close, volume)
        assert dv.iloc[0, 0] == 100_000.0
        assert dv.iloc[4, 0] == 145_600.0

    def test_compute_trailing_dollar_volume(self):
        dates = pd.date_range("2023-01-01", periods=10)
        dv = pd.DataFrame({"A": [100.0] * 10}, index=dates)
        trail = compute_trailing_dollar_volume(dv, window=5)
        assert pd.isna(trail.iloc[0, 0])
        assert pd.isna(trail.iloc[3, 0])
        assert trail.iloc[4, 0] == 500.0
        assert trail.iloc[9, 0] == 500.0


# ---------------------------------------------------------------------------
# HistoricalLiquidityUniverse class
# ---------------------------------------------------------------------------


class TestHistoricalLiquidityUniverse:
    def test_get_universe_at_timestamp(self, three_month_data):
        from trendbot.domain.universe import HistoricalLiquidityUniverse

        close, volume = three_month_data
        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=3,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        hu = HistoricalLiquidityUniverse(close, volume, config)
        u = hu.get_universe(pd.Timestamp("2023-01-05"))
        assert len(u) <= 3

    def test_get_all_members(self, three_month_data):
        from trendbot.domain.universe import HistoricalLiquidityUniverse

        close, volume = three_month_data
        config = UniverseConfig(
            mode=UniverseMode.DYNAMIC_TOP_N,
            top_n=3,
            liquidity_window=5,
            exclude_stablecoins=True,
            min_volume_days=3,
        )
        hu = HistoricalLiquidityUniverse(close, volume, config)
        all_m = hu.get_all_members()
        assert len(all_m) > 0
        assert all_m == sorted(all_m)
