"""Integration tests for CCXT data provider with pagination."""

from __future__ import annotations

from datetime import date

import pytest


class MockCcxtExchange:
    """Mock CCXT exchange that returns max 500 candles per call.

    Simulates a data source with 1500 total candles of data.
    """

    def __init__(self, max_candles: int = 500, total_data: int = 1500):
        self.max_candles = max_candles
        self.total_data = total_data
        self.markets = {"BTC/USDT": {}, "ETH/USDT": {}}
        self._call_count = 0
        self._base_ts = 1577836800000  # 2020-01-01 UTC

    def load_markets(self):
        pass

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        """Return candles starting from `since`, respecting max_candles and total_data."""
        self._call_count += 1
        effective_limit = min(limit or self.max_candles, self.max_candles)

        start_ts = since if since is not None else self._base_ts
        offset = (start_ts - self._base_ts) // 86400000

        remaining = self.total_data - offset
        if remaining <= 0:
            return []

        n = min(effective_limit, remaining)
        candles = []
        for i in range(n):
            ts = start_ts + i * 86400000
            candles.append([
                ts,
                100.0 + (offset + i) * 0.1,
                101.0 + (offset + i) * 0.1,
                99.0 + (offset + i) * 0.1,
                100.5 + (offset + i) * 0.1,
                1000.0,
            ])
        return candles


def test_ccxt_pagination_fetches_all_candles():
    """Verify that ccxt_provider paginates through multiple API calls
    to fetch more candles than a single request allows."""
    from trendbot.infrastructure.data_providers.ccxt_provider import (
        BINANCE_MAX_CANDLES,
        CcxtProvider,
    )

    mock_exchange = MockCcxtExchange(max_candles=BINANCE_MAX_CANDLES, total_data=2500)

    provider = CcxtProvider.__new__(CcxtProvider)
    provider.quote_currency = "USDT"
    provider.timeframe = "1d"
    provider._exchange = mock_exchange

    df = provider.fetch_daily_close_prices(
        symbol="BTC",
        start_date=date(2020, 1, 1),
        end_date=date(2027, 1, 1),
    )

    assert len(df) > BINANCE_MAX_CANDLES, (
        f"Expected >{BINANCE_MAX_CANDLES} candles, got {len(df)}"
    )
    assert mock_exchange._call_count >= 3, (
        f"Expected >=3 API calls for pagination, got {mock_exchange._call_count}"
    )
    assert len(df.columns) == 1
    assert df.columns[0] == "BTC"
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing


def test_ccxt_single_page_fetch():
    """When data fits in one page, only one API call is made."""
    from trendbot.infrastructure.data_providers.ccxt_provider import CcxtProvider

    mock_exchange = MockCcxtExchange(max_candles=500)

    provider = CcxtProvider.__new__(CcxtProvider)
    provider.quote_currency = "USDT"
    provider.timeframe = "1d"
    provider._exchange = mock_exchange

    df = provider.fetch_daily_close_prices(
        symbol="BTC",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )

    assert len(df) <= 10
    assert mock_exchange._call_count == 1


def test_ccxt_symbol_mapping():
    """Verify that various symbol formats are mapped to Binance pair format."""
    from trendbot.infrastructure.data_providers.ccxt_provider import CcxtProvider

    normalize = CcxtProvider._normalize_symbol

    # Plain base format
    assert normalize("BTC") == "BTC/USDT"
    assert normalize("eth") == "ETH/USDT"

    # Yahoo Finance format
    assert normalize("BTC-USD") == "BTC/USDT"
    assert normalize("ETH-USD") == "ETH/USDT"
    assert normalize("SOL-USD") == "SOL/USDT"

    # Already in CCXT format
    assert normalize("BTC/USDC") == "BTC/USDC"
    assert normalize("SOL/USDC") == "SOL/USDC"
    assert normalize("btc/usdt") == "BTC/USDT"

    # Custom quote currency
    assert normalize("SOL", "USDC") == "SOL/USDC"


def test_ccxt_empty_response_raises():
    """Verify that empty API response raises ValueError."""
    from trendbot.infrastructure.data_providers.ccxt_provider import CcxtProvider

    mock_exchange = MockCcxtExchange(max_candles=500)
    mock_exchange.fetch_ohlcv = lambda *args, **kwargs: []

    provider = CcxtProvider.__new__(CcxtProvider)
    provider.quote_currency = "USDT"
    provider.timeframe = "1d"
    provider._exchange = mock_exchange

    with pytest.raises(ValueError, match="No data returned"):
        provider.fetch_daily_close_prices(
            symbol="BTC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
        )
