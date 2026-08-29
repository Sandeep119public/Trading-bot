"""Unit tests for CCXT provider symbol mapping and sanitization."""

from __future__ import annotations

import pytest

from trendbot.infrastructure.data_providers.ccxt_provider import CcxtProvider


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
