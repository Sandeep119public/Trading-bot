from datetime import date

import pandas as pd

from trendbot.infrastructure.data_providers.ccxt_provider import (
    BinanceFuturesProvider,
    CcxtProvider,
)
from trendbot.infrastructure.data_providers.delta_india_provider import DeltaIndiaProvider


def test_binance_futures_normalizes_common_symbols() -> None:
    assert BinanceFuturesProvider._normalize_symbol("BTCUSDT") == "BTC/USDT:USDT"
    assert BinanceFuturesProvider._normalize_symbol("BTC/USDT") == "BTC/USDT:USDT"
    assert BinanceFuturesProvider._normalize_symbol("BTC-USD") == "BTC/USDT:USDT"
    assert BinanceFuturesProvider._normalize_symbol("ETHUSDT") == "ETH/USDT:USDT"


def test_binance_spot_normalization_remains_compatible() -> None:
    assert CcxtProvider._normalize_symbol("BTCUSDT") == "BTC/USDT"
    assert CcxtProvider._normalize_symbol("BTC-USD") == "BTC/USDT"


def test_delta_normalizes_common_symbols() -> None:
    assert DeltaIndiaProvider.normalize_symbol("BTCUSD") == "BTCUSD"
    assert DeltaIndiaProvider.normalize_symbol("BTC/USDT") == "BTCUSD"
    assert DeltaIndiaProvider.normalize_symbol("BTC-USDT") == "BTCUSD"
    assert DeltaIndiaProvider.normalize_symbol("BTCUSDT") == "BTCUSD"


def test_delta_candle_response_is_close_series(monkeypatch) -> None:
    provider = DeltaIndiaProvider(timeframe="1d")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "success": True,
                "result": [
                    {
                        "time": 1704067200,
                        "open": 100,
                        "high": 110,
                        "low": 90,
                        "close": 105,
                        "volume": 1,
                    },
                    {
                        "time": 1704153600,
                        "open": 105,
                        "high": 115,
                        "low": 95,
                        "close": 110,
                        "volume": 2,
                    },
                ],
            }

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        "trendbot.infrastructure.data_providers.delta_india_provider.requests.Session",
        lambda: FakeSession(),
    )
    frame = provider.fetch_daily_close_prices("BTCUSD", date(2024, 1, 1), date(2024, 1, 2))
    assert list(frame.columns) == ["BTCUSD"]
    assert frame["BTCUSD"].tolist() == [105, 110]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is None
