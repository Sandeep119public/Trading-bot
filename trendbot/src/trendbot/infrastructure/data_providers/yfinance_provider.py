"""YFinance data provider implementation."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from trendbot.application.ports import DataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(DataProvider):
    """Data provider using yfinance for Yahoo Finance data."""

    def fetch_daily_close_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Fetch daily close prices from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., 'BTC-USD', 'SPY').
            start_date: Start date for data.
            end_date: End date for data (None for latest).

        Returns:
            DataFrame with close prices.

        Raises:
            ValueError: If no data is returned for the symbol.
        """
        ticker = yf.Ticker(symbol)
        end_str = end_date.isoformat() if end_date else None
        df = ticker.history(start=start_date.isoformat(), end=end_str, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}'")

        close = df[["Close"]].copy()
        close.columns = ["close"]
        close.index.name = "date"
        close.index = pd.to_datetime(close.index).tz_localize(None)

        logger.info("Fetched %d rows for %s", len(close), symbol)
        return close

    def validate_symbol(self, symbol: str) -> bool:
        """Validate that a symbol exists in Yahoo Finance.

        Args:
            symbol: Ticker symbol to validate.

        Returns:
            True if symbol is valid, False otherwise.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return bool(info and "symbol" in info)
        except Exception:
            return False
