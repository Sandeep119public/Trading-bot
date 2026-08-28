"""Integration tests for data service."""

import tempfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trendbot.application.data_service import DataService
from trendbot.domain.models import DataDownloadRequest
from trendbot.infrastructure.repositories.parquet_price_repository import ParquetPriceRepository


class FakeProvider:
    """Fake data provider for testing."""

    VALID_SYMBOLS = {"A", "B", "C"}

    def fetch_daily_close_prices(self, symbol, start_date, end_date):
        if symbol not in self.VALID_SYMBOLS:
            raise ValueError(f"Invalid symbol: {symbol}")
        dates = pd.date_range(start_date, end_date or date.today(), freq="D")
        np.random.seed(hash(symbol) % 2**31)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({"close": prices}, index=dates)
        df.index.name = "date"
        return df

    def validate_symbol(self, symbol):
        return symbol in self.VALID_SYMBOLS


@pytest.fixture
def data_service():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = ParquetPriceRepository(tmpdir)
        provider = FakeProvider()
        svc = DataService(provider, repo)
        yield svc


def test_download_and_list(data_service):
    request = DataDownloadRequest(
        source="test",
        symbols=["A", "B"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 6, 30),
    )
    result = data_service.download_data(request)
    assert result.success
    assert len(result.symbols_processed) == 2

    datasets = data_service.list_datasets()
    assert len(datasets) == 2


def test_load_prices(data_service):
    request = DataDownloadRequest(
        source="test",
        symbols=["A"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 6, 30),
    )
    data_service.download_data(request)

    prices = data_service.load_prices(
        source="test",
        symbols=["A"],
        timeframe="1d",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 6, 30),
    )
    assert not prices.empty
    assert "A" in prices.columns


def test_delete_dataset(data_service):
    request = DataDownloadRequest(
        source="test",
        symbols=["A"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 6, 30),
    )
    data_service.download_data(request)
    assert len(data_service.list_datasets()) == 1

    data_service.delete_dataset("test", "A")
    assert len(data_service.list_datasets()) == 0


def test_empty_download(data_service):
    request = DataDownloadRequest(
        source="test",
        symbols=["INVALID"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 6, 30),
    )
    result = data_service.download_data(request)
    assert not result.success
    assert len(result.symbols_failed) == 1
