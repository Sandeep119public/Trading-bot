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


class CcxtProvider(DataProvider):
    """Data provider using CCXT for Binance exchange data.

    Supports paginated OHLCV fetching to handle Binance API limits.
    """

    def __init__(
        self,
        quote_currency: str = "USDT",
        timeframe: str = "1d",
        rate_limit: bool = True,
    ) -> None:
        """Initialize the CCXT provider."""
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

    @staticmethod
    def _normalize_symbol(symbol: str, quote_currency: str = "USDT") -> str:
        """Convert user inputs into valid CCXT/Binance symbols."""
        symbol = symbol.upper().strip()

        if "-" in symbol:
            parts = symbol.split("-")
            base = parts[0]
            quote = parts[1]
            if quote == "USD":
                quote = "USDT"
            return f"{base}/{quote}"

        if "/" not in symbol:
            return f"{symbol}/{quote_currency.upper()}"

        return symbol

    @staticmethod
    def _sanitize_for_path(symbol: str) -> str:
        """Sanitize a symbol for use in file system paths."""
        return symbol.replace("/", "-")

    def fetch_daily_close_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Fetch OHLCV closes from Binance with automatic pagination."""
        pair = self._normalize_symbol(symbol, self.quote_currency)
        timeframe = self.timeframe

        since_ms = int(
            datetime.combine(start_date, datetime.min.time())
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )

        end_ms = None
        if end_date is not None:
            end_ms = int(
                datetime.combine(end_date, datetime.max.time())
                .replace(tzinfo=timezone.utc)
                .timestamp()
                * 1000
            )

        all_ohlcv = self._fetch_paginated(pair, timeframe, since_ms, end_ms)

        if not all_ohlcv:
            raise ValueError(f"No data returned for symbol '{symbol}' on Binance")

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["date"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        ).dt.tz_localize(None)
        df = df.set_index("date")[["close"]]

        if end_ms is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        df = df[~df.index.duplicated(keep="first")].sort_index()
        df.columns = [symbol]

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
        """Fetch OHLCV pages using CCXT's standard since/limit arguments.

        ``since`` and ``limit`` are passed as CCXT function arguments. Binance
        accepts the optional range terminator as ``endTime``. Do not duplicate
        ``since`` or ``limit`` inside ``params`` because CCXT would send them
        twice to Binance and Binance rejects the request with error -1104.
        """
        all_candles: list[list] = []
        current_since = since_ms
        max_retries = 3

        while True:
            limit = BINANCE_MAX_CANDLES
            params = {"endTime": end_ms} if end_ms is not None else {}

            candles = self._fetch_with_retry(
                pair,
                timeframe,
                current_since,
                limit,
                params,
                max_retries,
            )

            if not candles:
                break

            all_candles.extend(candles)
            last_timestamp = candles[-1][0]

            if end_ms is not None and last_timestamp >= end_ms:
                break

            if len(candles) < limit:
                break

            next_since = last_timestamp + 1
            if next_since <= current_since:
                raise RuntimeError(
                    f"Binance pagination stalled for {pair}: "
                    f"current_since={current_since}, last_timestamp={last_timestamp}"
                )
            current_since = next_since

        return all_candles

    def _fetch_with_retry(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        limit: int,
        params: dict,
        max_retries: int,
    ) -> list[list]:
        """Fetch one OHLCV page with retry logic.

        ``since_ms`` and ``limit`` are deliberately passed only through the
        named CCXT arguments. ``params`` contains only exchange-specific
        parameters such as Binance's ``endTime``.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                candles = self._exchange.fetch_ohlcv(
                    symbol=pair,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=limit,
                    params=params,
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
                    time.sleep(min(2**attempt, 10))

        raise RuntimeError(
            f"Failed to fetch {pair} after {max_retries} retries: {last_error}"
        )

    def validate_symbol(self, symbol: str) -> bool:
        """Validate that a symbol exists on Binance."""
        try:
            pair = self._normalize_symbol(symbol, self.quote_currency)
            self._exchange.load_markets()
            return pair in self._exchange.markets
        except Exception:
            return False
