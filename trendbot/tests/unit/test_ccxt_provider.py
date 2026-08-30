"""Unit tests for CCXT provider symbol mapping and Binance request construction."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from trendbot.infrastructure.data_providers.ccxt_provider import (
    BINANCE_MAX_CANDLES,
    CcxtProvider,
)


class TestNormalizeSymbol:
    """Tests for CcxtProvider._normalize_symbol."""

    def test_plain_base_default_quote(self):
        assert CcxtProvider._normalize_symbol("BTC") == "BTC/USDT"
        assert CcxtProvider._normalize_symbol("ETH") == "ETH/USDT"
        assert CcxtProvider._normalize_symbol("SOL") == "SOL/USDT"

    def test_plain_base_custom_quote(self):
        assert CcxtProvider._normalize_symbol("BTC", "USDC") == "BTC/USDC"
        assert CcxtProvider._normalize_symbol("SOL", "BUSD") == "SOL/BUSD"

    def test_yahoo_finance_format(self):
        assert CcxtProvider._normalize_symbol("BTC-USD") == "BTC/USDT"
        assert CcxtProvider._normalize_symbol("ETH-USD") == "ETH/USDT"
        assert CcxtProvider._normalize_symbol("SOL-USD") == "SOL/USDT"

    def test_yahoo_finance_non_usd_quote(self):
        assert CcxtProvider._normalize_symbol("BTC-EUR") == "BTC/EUR"

    def test_already_ccxt_format(self):
        assert CcxtProvider._normalize_symbol("BTC/USDT") == "BTC/USDT"
        assert CcxtProvider._normalize_symbol("BTC/USDC") == "BTC/USDC"

    def test_case_insensitive(self):
        assert CcxtProvider._normalize_symbol("btc") == "BTC/USDT"
        assert CcxtProvider._normalize_symbol("btc-usd") == "BTC/USDT"
        assert CcxtProvider._normalize_symbol("btc/usdt") == "BTC/USDT"

    def test_whitespace_handling(self):
        assert CcxtProvider._normalize_symbol("  BTC  ") == "BTC/USDT"
        assert CcxtProvider._normalize_symbol(" BTC-USD ") == "BTC/USDT"


class TestSanitizeForPath:
    """Tests for CcxtProvider._sanitize_for_path."""

    def test_slash_replaced_with_dash(self):
        assert CcxtProvider._sanitize_for_path("BTC/USDT") == "BTC-USDT"

    def test_no_slash_unchanged(self):
        assert CcxtProvider._sanitize_for_path("BTC") == "BTC"
        assert CcxtProvider._sanitize_for_path("BTC-USD") == "BTC-USD"

    def test_multiple_slashes(self):
        assert CcxtProvider._sanitize_for_path("A/B/C") == "A-B-C"


class TestNormalizeIntegration:
    """Integration tests verifying normalize -> sanitize round-trip."""

    def test_yahoo_to_parquet_path(self):
        """BTC-USD -> BTC/USDT for API, BTC-USD for storage."""
        pair = CcxtProvider._normalize_symbol("BTC-USD")
        storage = CcxtProvider._sanitize_for_path("BTC-USD")
        assert pair == "BTC/USDT"
        assert storage == "BTC-USD"
        assert "/" not in storage

    def test_bare_to_parquet_path(self):
        """BTC -> BTC/USDT for API, BTC for storage."""
        pair = CcxtProvider._normalize_symbol("BTC")
        storage = CcxtProvider._sanitize_for_path("BTC")
        assert pair == "BTC/USDT"
        assert storage == "BTC"

    def test_ccxt_to_parquet_path(self):
        """BTC/USDT -> BTC/USDT for API, BTC-USDT for storage."""
        pair = CcxtProvider._normalize_symbol("BTC/USDT")
        storage = CcxtProvider._sanitize_for_path("BTC/USDT")
        assert pair == "BTC/USDT"
        assert storage == "BTC-USDT"


class TestBinanceRequestConstruction:
    """Regression tests for Binance API parameter handling."""

    @staticmethod
    def _provider_with_mock_exchange() -> CcxtProvider:
        provider = CcxtProvider.__new__(CcxtProvider)
        provider._exchange = Mock()
        return provider

    def test_since_and_limit_are_not_duplicated_in_params(self):
        provider = self._provider_with_mock_exchange()
        candles = [[1000, 1, 1, 1, 1, 1]]
        provider._exchange.fetch_ohlcv.return_value = candles

        result = provider._fetch_with_retry(
            "BTC/USDT",
            "1d",
            since_ms=1000,
            limit=1000,
            params={"endTime": 2000},
            max_retries=1,
        )

        assert result == candles
        provider._exchange.fetch_ohlcv.assert_called_once_with(
            symbol="BTC/USDT",
            timeframe="1d",
            since=1000,
            limit=1000,
            params={"endTime": 2000},
        )

    def test_no_end_time_when_no_end_date(self):
        provider = self._provider_with_mock_exchange()
        provider._exchange.fetch_ohlcv.return_value = []

        provider._fetch_paginated(
            "BTC/USDT",
            "1d",
            since_ms=1000,
            end_ms=None,
        )

        provider._exchange.fetch_ohlcv.assert_called_once_with(
            symbol="BTC/USDT",
            timeframe="1d",
            since=1000,
            limit=BINANCE_MAX_CANDLES,
            params={},
        )

    def test_end_time_uses_binance_end_time_parameter(self):
        provider = self._provider_with_mock_exchange()
        provider._exchange.fetch_ohlcv.return_value = []

        provider._fetch_paginated(
            "BTC/USDT",
            "1d",
            since_ms=1000,
            end_ms=2000,
        )

        provider._exchange.fetch_ohlcv.assert_called_once_with(
            symbol="BTC/USDT",
            timeframe="1d",
            since=1000,
            limit=BINANCE_MAX_CANDLES,
            params={"endTime": 2000},
        )

    def test_pagination_advances_after_full_page(self):
        provider = self._provider_with_mock_exchange()
        first_page = [
            [1000, 1, 1, 1, 1, 1],
            [2000, 1, 1, 1, 1, 1],
        ]
        second_page = [[3000, 1, 1, 1, 1, 1]]
        provider._exchange.fetch_ohlcv.side_effect = [first_page, second_page]

        # Temporarily use a tiny page size to exercise pagination deterministically.
        import trendbot.infrastructure.data_providers.ccxt_provider as module

        original_limit = module.BINANCE_MAX_CANDLES
        module.BINANCE_MAX_CANDLES = 2
        try:
            result = provider._fetch_paginated(
                "BTC/USDT",
                "1d",
                since_ms=1000,
                end_ms=4000,
            )
        finally:
            module.BINANCE_MAX_CANDLES = original_limit

        assert result == first_page + second_page
        assert provider._exchange.fetch_ohlcv.call_count == 2
        first_call = provider._exchange.fetch_ohlcv.call_args_list[0]
        second_call = provider._exchange.fetch_ohlcv.call_args_list[1]
        assert first_call.kwargs["since"] == 1000
        assert second_call.kwargs["since"] == 2001
        assert first_call.kwargs["params"] == {"endTime": 4000}
        assert second_call.kwargs["params"] == {"endTime": 4000}

    def test_retry_exhaustion_raises_runtime_error(self):
        provider = self._provider_with_mock_exchange()
        provider._exchange.fetch_ohlcv.side_effect = RuntimeError("API failure")

        with pytest.raises(RuntimeError, match="Failed to fetch BTC/USDT after 2 retries"):
            provider._fetch_with_retry(
                "BTC/USDT",
                "1d",
                since_ms=1000,
                limit=1000,
                params={},
                max_retries=2,
            )
