"""CCXT data provider implementation for Binance exchange data."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd

from trendbot.application.ports import DataProvider

logger = logging.getLogger(__name__)

# Binance limits OHLCV responses to 1000-1500 candles per request
BINANCE_MAX_CANDLES = 1000

TIMEFRAME_MAP = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def _symbol_to_binance_pair(symbol: str, quote_currency: str = "USDT") -> str:
    """Map user inputs to valid CCXT/Binance symbols.

    Handles: BTC-USD (Yahoo), BTC (bare), BTC/USDT (already correct).
    """
    symbol = symbol.upper().strip()

    # 1. Handle Yahoo Finance format (e.g., BTC-USD -> BTC/USDT)
    if "-" in symbol:
        parts = symbol.split("-")
        base = parts[0]
        quote = parts[1]
        # Binance uses USDT instead of USD for fiat-pegged pairs
        if quote == "USD":
            quote = "USDT"
        return f"{base}/{quote}"

    # 2. Handle plain base format (e.g., BTC -> BTC/USDT)
    if "/" not in symbol:
        return f"{symbol}/{quote_currency.upper()}"

    # 3. Already in CCXT format (e.g., BTC/USDT)
    return symbol


class CcxtProvider(DataProvider):
    """Data provider using CCXT for Binance exchange data.

    Supports paginated OHLCV fetching to handle Binance API limits
    (max 1000-1500 candles per request).
    """

    def __init__(
        self,
        quote_currency: str = "USDT",
        timeframe: str = "1d",
        rate_limit: bool = True,
    ) -> None:
        """Initialize the CCXT provider.

        Args:
            quote_currency: Quote currency for pair construction (default USDT).
            timeframe: Default OHLCV timeframe (default 1d).
            rate_limit: Whether to enable CCXT rate limiting (recommended).
        """
        self.quote_currency = quote_currency
        self.timeframe = timeframe

        try:
            import ccxt

            exchange_class = getattr(ccxt, "binance")
            self._exchange = exchange_class(
                {"enableRateLimit": rate_limit, "options": {"defaultType": "spot"}}
            )
        except ImportError:
            raise ImportError(
                "ccxt is required for Binance data. Install with: pip install ccxt"
            )
        except AttributeError:
            raise ValueError("Exchange 'binance' not found in ccxt")

    def fetch_daily_close_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Fetch daily close prices from Binance with automatic pagination.

        Args:
            symbol: Ticker symbol (e.g., 'BTC', 'ETH', 'BTC/USDT').
            start_date: Start date for data.
            end_date: End date for data (None for latest).

        Returns:
            DataFrame with close prices (index=date, columns=['close']).

        Raises:
            ValueError: If no data is returned for the symbol.
            RuntimeError: If the Binance API request fails.
        """
        pair = _symbol_to_binance_pair(symbol, self.quote_currency)
        timeframe = self.timeframe

        since_ms = int(datetime.combine(start_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        ).timestamp() * 1000)

        end_ms = None
        if end_date is not None:
            end_ms = int(datetime.combine(end_date, datetime.max.time()).replace(
                tzinfo=timezone.utc
            ).timestamp() * 1000)

        all_ohlcv = self._fetch_paginated(pair, timeframe, since_ms, end_ms)

        if not all_ohlcv:
            raise ValueError(f"No data returned for symbol '{symbol}' on Binance")

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.set_index("date")
        df = df[["close"]]

        if end_ms is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        logger.info(
            "Fetched %d rows for %s from Binance (pair=%s, tf=%s)",
            len(df), symbol, pair, timeframe,
        )
        return df

    def _fetch_paginated(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        end_ms: int | None,
    ) -> list[list]:
        """Fetch OHLCV data with pagination to handle API limits.

        Loops through requests using the `since` parameter until all
        candles are fetched or the end timestamp is reached.

        Args:
            pair: Trading pair (e.g., 'BTC/USDT').
            timeframe: OHLCV timeframe.
            since_ms: Start timestamp in milliseconds.
            end_ms: End timestamp in milliseconds (None for latest).

        Returns:
            List of OHLCV candles [timestamp, open, high, low, close, volume].
        """
        all_candles: list[list] = []
        current_since = since_ms
        max_retries = 3

        while True:
            limit = BINANCE_MAX_CANDLES
            params = {"since": current_since, "limit": limit}
            if end_ms is not None:
                params["until"] = end_ms

            candles = self._fetch_with_retry(pair, timeframe, params, max_retries)

            if not candles:
                break

            all_candles.extend(candles)

            last_timestamp = candles[-1][0]

            if end_ms is not None and last_timestamp >= end_ms:
                break

            if len(candles) < limit:
                break

            current_since = last_timestamp + 1

        return all_candles

    def _fetch_with_retry(
        self,
        pair: str,
        timeframe: str,
        params: dict,
        max_retries: int,
    ) -> list[list]:
        """Fetch OHLCV with retry logic for transient failures.

        Args:
            pair: Trading pair.
            timeframe: OHLCV timeframe.
            params: Fetch parameters (since, limit, until).
            max_retries: Maximum number of retry attempts.

        Returns:
            List of OHLCV candles.

        Raises:
            RuntimeError: If all retries fail.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                candles = self._exchange.fetch_ohlcv(
                    symbol=pair,
                    timeframe=timeframe,
                    since=params.get("since"),
                    limit=params.get("limit", BINANCE_MAX_CANDLES),
                    params={
                        k: v
                        for k, v in params.items()
                        if k in ("since", "limit", "until")
                    },
                )
                return candles
            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt + 1, max_retries, pair, e,
                )
                if attempt < max_retries - 1:
                    import time
                    time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(
            f"Failed to fetch {pair} after {max_retries} retries: {last_error}"
        )

    def validate_symbol(self, symbol: str) -> bool:
        """Validate that a symbol exists on Binance.

        Args:
            symbol: Symbol to validate (e.g., 'BTC' or 'BTC/USDT').

        Returns:
            True if symbol is valid on Binance, False otherwise.
        """
        try:
            pair = _symbol_to_binance_pair(symbol, self.quote_currency)
            self._exchange.load_markets()
            return pair in self._exchange.markets
        except Exception:
            return False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data with configurable timeframe.

        This is a convenience method that allows fetching data with
        any supported timeframe, not just daily close prices.

        Args:
            symbol: Ticker symbol (e.g., 'BTC', 'BTC/USDT').
            timeframe: OHLCV timeframe ('1h', '4h', '1d').
            start_date: Start date for data (None for earliest available).
            end_date: End date for data (None for latest).

        Returns:
            DataFrame with OHLCV columns (index=date).
        """
        tf = TIMEFRAME_MAP.get(timeframe, timeframe)
        pair = _symbol_to_binance_pair(symbol, self.quote_currency)

        since_ms = None
        if start_date is not None:
            since_ms = int(datetime.combine(start_date, datetime.min.time()).replace(
                tzinfo=timezone.utc
            ).timestamp() * 1000)

        end_ms = None
        if end_date is not None:
            end_ms = int(datetime.combine(end_date, datetime.max.time()).replace(
                tzinfo=timezone.utc
            ).timestamp() * 1000)

        all_candles = self._fetch_paginated(pair, tf, since_ms or 0, end_ms)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.set_index("date")
        df = df.drop(columns=["timestamp"])

        if end_ms is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        return df
