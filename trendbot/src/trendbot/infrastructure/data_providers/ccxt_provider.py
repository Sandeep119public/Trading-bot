"""CCXT data providers for Binance spot and USD-M futures data."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import pandas as pd

from trendbot.application.ports import DataProvider

logger = logging.getLogger(__name__)

BINANCE_MAX_CANDLES = 1000
TIMEFRAME_MAP = {"1h": "1h", "4h": "4h", "1d": "1d"}


class CcxtProvider(DataProvider):
    """Data provider using CCXT for Binance spot market data."""

    def __init__(
        self,
        quote_currency: str = "USDT",
        timeframe: str = "1d",
        rate_limit: bool = True,
    ) -> None:
        self.quote_currency = quote_currency.upper()
        self.timeframe = self._validate_timeframe(timeframe)
        self._exchange = self._create_exchange("spot", rate_limit)

    @staticmethod
    def _validate_timeframe(timeframe: str) -> str:
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported Binance timeframe '{timeframe}'. "
                f"Supported: {', '.join(TIMEFRAME_MAP)}"
            )
        return TIMEFRAME_MAP[timeframe]

    @staticmethod
    def _create_exchange(market_type: str, rate_limit: bool):
        try:
            import ccxt
            exchange_class = getattr(ccxt, "binance")
            options = {"defaultType": market_type}
            if market_type == "future":
                options["defaultSubType"] = "linear"
            return exchange_class({
                "enableRateLimit": rate_limit,
                "options": options,
            })
        except ImportError as exc:
            raise ImportError(
                "ccxt is required for Binance data."
                " Install with: pip install ccxt"
            ) from exc
        except AttributeError as exc:
            raise ValueError(
                "Exchange 'binance' not found in ccxt"
            ) from exc

    @staticmethod
    def _normalize_symbol(symbol: str, quote_currency: str = "USDT") -> str:
        value = symbol.upper().strip()
        if "/" in value:
            return value
        if "-" in value:
            base, quote = value.split("-", 1)
            if quote == "USD":
                quote = "USDT"
            return f"{base}/{quote}"
        for suffix in ("USDT", "USDC", "BUSD", "BTC", "ETH"):
            if value.endswith(suffix) and len(value) > len(suffix):
                return f"{value[:-len(suffix)]}/{suffix}"
        return f"{value}/{quote_currency.upper()}"

    @staticmethod
    def _sanitize_for_path(symbol: str) -> str:
        return symbol.replace("/", "-")

    def _fetch_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Fetch raw OHLCV data and return a DataFrame with close + volume."""
        pair = self._normalize_symbol(symbol, self.quote_currency)
        since_ms = int(
            datetime.combine(start_date, datetime.min.time())
            .replace(tzinfo=UTC)
            .timestamp()
            * 1000
        )
        end_ms = (
            None
            if end_date is None
            else int(
                datetime.combine(end_date, datetime.max.time())
                .replace(tzinfo=UTC)
                .timestamp()
                * 1000
            )
        )
        all_ohlcv = self._fetch_paginated(
            pair, self.timeframe, since_ms, end_ms
        )
        if not all_ohlcv:
            raise ValueError(
                f"No data returned for symbol '{symbol}' on Binance"
            )
        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["date"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        ).dt.tz_localize(None)
        df = df.set_index("date")[["close", "volume"]]
        if end_ms is not None:
            df = df[df.index <= pd.Timestamp(end_date)]
        df = df[~df.index.duplicated(keep="first")].sort_index()
        return df

    def fetch_daily_close_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date | None,
    ) -> pd.DataFrame:
        df = self._fetch_ohlcv(symbol, start_date, end_date)
        return df[["close"]].rename(columns={"close": symbol.upper()})

    def fetch_daily_volume(
        self,
        symbol: str,
        start_date: date,
        end_date: date | None,
    ) -> pd.DataFrame:
        df = self._fetch_ohlcv(symbol, start_date, end_date)
        return df[["volume"]].rename(columns={"volume": symbol.upper()})

    def _fetch_paginated(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        end_ms: int | None,
    ) -> list[list]:
        all_candles: list[list] = []
        current_since = since_ms
        while True:
            params = (
                {"endTime": end_ms} if end_ms is not None else {}
            )
            candles = self._fetch_with_retry(
                pair, timeframe, current_since,
                BINANCE_MAX_CANDLES, params, 3,
            )
            if not candles:
                break
            all_candles.extend(candles)
            last_timestamp = candles[-1][0]
            if end_ms is not None and last_timestamp >= end_ms:
                break
            if len(candles) < BINANCE_MAX_CANDLES:
                break
            next_since = last_timestamp + 1
            if next_since <= current_since:
                raise RuntimeError(
                    f"Binance pagination stalled for {pair}"
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
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self._exchange.fetch_ohlcv(
                    symbol=pair,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=limit,
                    params=params,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt + 1, max_retries, pair, exc,
                )
                if attempt < max_retries - 1:
                    import time
                    time.sleep(min(2**attempt, 10))
        raise RuntimeError(
            f"Failed to fetch {pair} after"
            f" {max_retries} retries: {last_error}"
        )

    def validate_symbol(self, symbol: str) -> bool:
        try:
            pair = self._normalize_symbol(symbol, self.quote_currency)
            self._exchange.load_markets()
            return pair in self._exchange.markets
        except Exception:
            return False


class BinanceFuturesProvider(CcxtProvider):
    """Binance USD-M perpetual/futures provider using CCXT linear markets."""

    def __init__(
        self,
        quote_currency: str = "USDT",
        timeframe: str = "1d",
        rate_limit: bool = True,
    ) -> None:
        self.quote_currency = quote_currency.upper()
        self.timeframe = self._validate_timeframe(timeframe)
        self._exchange = self._create_exchange("future", rate_limit)

    @staticmethod
    def _normalize_symbol(
        symbol: str, quote_currency: str = "USDT"
    ) -> str:
        value = symbol.upper().strip()
        if "/" in value:
            base, quote = value.split("/", 1)
            quote = quote.split(":", 1)[0]
            if quote in {"USDT", "USDC"}:
                return f"{base}/{quote}:{quote}"
            return f"{base}/{quote}:USDT"
        if "-" in value:
            base, quote = value.split("-", 1)
            if quote in {"USD", "USDT", "USDC"}:
                settle = "USDT" if quote == "USD" else quote
                return f"{base}/{settle}:{settle}"
        for suffix in ("USDT", "USDC"):
            if value.endswith(suffix) and len(value) > len(suffix):
                return f"{value[:-len(suffix)]}/{suffix}:{suffix}"
        return f"{value}/{quote_currency}:USDT"

    def validate_symbol(self, symbol: str) -> bool:
        try:
            pair = self._normalize_symbol(symbol, self.quote_currency)
            self._exchange.load_markets()
            market = self._exchange.markets.get(pair)
            return bool(market and (market.get("swap") or market.get("future")))
        except Exception:
            return False
