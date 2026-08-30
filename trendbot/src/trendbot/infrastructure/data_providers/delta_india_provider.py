"""Delta Exchange India public market-data provider.

Supports perpetual/futures OHLC candles from Delta's India production API.
No API key is required for historical public candles.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime

import pandas as pd
import requests

from trendbot.application.ports import DataProvider

logger = logging.getLogger(__name__)

DELTA_BASE_URL = "https://api.india.delta.exchange"
DELTA_MAX_CANDLES = 1900
DELTA_RESOLUTIONS = {"1h": "1h", "4h": "4h", "1d": "1d"}


class DeltaIndiaProvider(DataProvider):
    """Fetch public perpetual/futures market data from Delta Exchange India."""

    def __init__(self, timeframe: str = "1d", rate_limit_seconds: float = 0.15) -> None:
        if timeframe not in DELTA_RESOLUTIONS:
            raise ValueError(
                f"Unsupported Delta timeframe '{timeframe}'. "
                f"Supported: {', '.join(DELTA_RESOLUTIONS)}"
            )
        self.timeframe = timeframe
        self.rate_limit_seconds = max(0.0, rate_limit_seconds)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalize common user inputs to Delta contract symbols."""
        value = symbol.strip().upper()
        if "/" in value:
            base, quote = value.split("/", 1)
            if quote in {"USD", "USDT"}:
                return f"{base}USD"
        if "-" in value:
            parts = value.split("-")
            if len(parts) == 2 and parts[1] in {"USD", "USDT"}:
                return f"{parts[0]}USD"
        if value.endswith("USDT"):
            return f"{value[:-4]}USD"
        return value

    def fetch_daily_close_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Fetch OHLC candles and return a close-price DataFrame."""
        contract = self.normalize_symbol(symbol)
        start = int(
            datetime.combine(start_date, datetime.min.time())
            .replace(tzinfo=UTC)
            .timestamp()
        )
        end = (
            int(
                datetime.combine(end_date, datetime.max.time())
                .replace(tzinfo=UTC)
                .timestamp()
            )
            if end_date is not None
            else int(datetime.now(UTC).timestamp())
        )

        rows: list[dict] = []
        cursor = start
        session = requests.Session()

        while cursor <= end:
            params = {
                "resolution": DELTA_RESOLUTIONS[self.timeframe],
                "symbol": contract,
                "start": cursor,
                "end": end,
            }
            response = session.get(
                f"{DELTA_BASE_URL}/v2/history/candles",
                params=params,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success", False):
                raise RuntimeError(f"Delta API returned unsuccessful response: {payload}")

            batch = payload.get("result", [])
            if not batch:
                break
            rows.extend(batch)
            last_time = max(int(item["time"]) for item in batch)
            if last_time >= end or len(batch) < DELTA_MAX_CANDLES:
                break
            next_cursor = last_time + 1
            if next_cursor <= cursor:
                raise RuntimeError(f"Delta pagination stalled for {contract}")
            cursor = next_cursor
            if self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds)

        if not rows:
            raise ValueError(f"No data returned for Delta contract '{contract}'")

        frame = pd.DataFrame(rows)
        required = {"time", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Delta candle response missing fields: {sorted(missing)}")
        frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "close"])
        frame = frame.set_index("timestamp").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        if end_date is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end_date, tz="UTC")]
        frame.index = frame.index.tz_localize(None)
        return frame[["close"]].rename(columns={"close": symbol.upper()})

    def validate_symbol(self, symbol: str) -> bool:
        """Validate that a Delta product exists and is a futures/perpetual contract."""
        contract = self.normalize_symbol(symbol)
        try:
            response = requests.get(
                f"{DELTA_BASE_URL}/v2/tickers/{contract}",
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if response.status_code != 200:
                return False
            payload = response.json()
            product = payload.get("result", {})
            contract_type = str(product.get("contract_type", ""))
            return contract_type in {"perpetual_futures", "futures"}
        except Exception:
            return False
