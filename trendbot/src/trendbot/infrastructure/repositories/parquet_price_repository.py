"""Parquet-based price data repository."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from trendbot.application.ports import PriceRepository
from trendbot.domain.models import DatasetMetadata

logger = logging.getLogger(__name__)


class ParquetPriceRepository(PriceRepository):
    """Repository storing price data as Parquet files with JSON metadata."""

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._raw_dir = self._data_dir / "raw"
        self._meta_dir = self._data_dir / "metadata"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, source: str, symbol: str, timeframe: str) -> Path:
        return self._raw_dir / source / f"{symbol}_{timeframe}.parquet"

    def _meta_path(self, source: str, symbol: str, timeframe: str) -> Path:
        return self._meta_dir / f"{source}_{symbol}_{timeframe}.json"

    def save_prices(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        """Save price data as Parquet with metadata JSON.

        Args:
            source: Data source name.
            symbol: Ticker symbol.
            timeframe: Data timeframe.
            df: DataFrame with price data.
        """
        path = self._parquet_path(source, symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

        now = datetime.now().isoformat()
        meta = DatasetMetadata(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index.min().date() if len(df) > 0 else date.today(),
            end_date=df.index.max().date() if len(df) > 0 else date.today(),
            rows=len(df),
            downloaded_at=now,
            last_updated=now,
            file_path=str(path),
        )

        meta_path = self._meta_path(source, symbol, timeframe)
        with open(meta_path, "w") as f:
            json.dump(meta.model_dump(), f, indent=2, default=str)

        logger.info("Saved %s/%s/%s: %d rows", source, symbol, timeframe, len(df))

    def load_prices(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start_date: date | None,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Load price data from Parquet file.

        Args:
            source: Data source name.
            symbol: Ticker symbol.
            timeframe: Data timeframe.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            DataFrame with price data.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        path = self._parquet_path(source, symbol, timeframe)
        if not path.exists():
            raise FileNotFoundError(f"No data found for {source}/{symbol}/{timeframe}")

        df = pd.read_parquet(path)

        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df

    def list_datasets(self) -> list[DatasetMetadata]:
        """List all stored datasets by reading metadata files."""
        datasets: list[DatasetMetadata] = []

        for meta_file in self._meta_dir.glob("*.json"):
            try:
                with open(meta_file) as f:
                    data = json.load(f)
                datasets.append(DatasetMetadata(**data))
            except Exception as e:
                logger.warning("Failed to read metadata %s: %s", meta_file, e)

        return sorted(datasets, key=lambda d: (d.source, d.symbol))

    def delete_dataset(
        self,
        source: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Delete a stored dataset and its metadata."""
        parquet = self._parquet_path(source, symbol, timeframe)
        meta = self._meta_path(source, symbol, timeframe)

        if parquet.exists():
            parquet.unlink()
        if meta.exists():
            meta.unlink()

        logger.info("Deleted %s/%s/%s", source, symbol, timeframe)

    def dataset_exists(
        self,
        source: str,
        symbol: str,
        timeframe: str,
    ) -> bool:
        """Check if a dataset exists locally."""
        return self._parquet_path(source, symbol, timeframe).exists()
