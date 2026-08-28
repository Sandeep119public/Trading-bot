"""Ports and interfaces for the application layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from trendbot.domain.models import DatasetMetadata


class DataProvider(ABC):
    """Interface for external data providers."""

    @abstractmethod
    def fetch_daily_close_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Fetch daily close prices for a symbol."""
        ...

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate that a symbol exists in the data source."""
        ...


class PriceRepository(ABC):
    """Interface for local price data persistence."""

    @abstractmethod
    def save_prices(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        """Save price data to local storage."""
        ...

    @abstractmethod
    def load_prices(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start_date: date | None,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Load price data from local storage."""
        ...

    @abstractmethod
    def list_datasets(self) -> list[DatasetMetadata]:
        """List all stored datasets."""
        ...

    @abstractmethod
    def delete_dataset(
        self,
        source: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Delete a stored dataset."""
        ...

    @abstractmethod
    def dataset_exists(
        self,
        source: str,
        symbol: str,
        timeframe: str,
    ) -> bool:
        """Check if a dataset exists locally."""
        ...
