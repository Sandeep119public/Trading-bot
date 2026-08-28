"""Application service for data management."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from trendbot.application.dto import DownloadResult
from trendbot.application.ports import DataProvider, PriceRepository
from trendbot.domain.models import DataDownloadRequest, DatasetMetadata

logger = logging.getLogger(__name__)


class DataService:
    """Service for downloading and managing price data."""

    def __init__(self, provider: DataProvider, repository: PriceRepository) -> None:
        self._provider = provider
        self._repository = repository

    def download_data(self, request: DataDownloadRequest) -> DownloadResult:
        """Download data from provider and store locally.

        Args:
            request: Data download request with source, symbols, dates.

        Returns:
            DownloadResult with success status and details.
        """
        processed: list[str] = []
        failed: list[str] = []

        for symbol in request.symbols:
            try:
                if not request.overwrite and self._repository.dataset_exists(
                    request.source, symbol, "1d"
                ):
                    logger.info("Skipping %s (already exists, overwrite=False)", symbol)
                    processed.append(symbol)
                    continue

                df = self._provider.fetch_daily_close_prices(
                    symbol, request.start_date, request.end_date
                )

                if df.empty:
                    logger.warning("Empty data for %s", symbol)
                    failed.append(f"{symbol}: empty data")
                    continue

                self._repository.save_prices(request.source, symbol, "1d", df)
                processed.append(symbol)
                logger.info("Downloaded %s: %d rows", symbol, len(df))

            except Exception as e:
                logger.error("Failed to download %s: %s", symbol, e)
                failed.append(f"{symbol}: {e}")

        return DownloadResult(
            success=len(failed) == 0,
            message=f"Processed {len(processed)}, failed {len(failed)}",
            symbols_processed=processed,
            symbols_failed=failed,
        )

    def list_datasets(self) -> list[DatasetMetadata]:
        """List all locally stored datasets."""
        return self._repository.list_datasets()

    def load_prices(
        self,
        source: str,
        symbols: list[str],
        timeframe: str,
        start_date: date | None,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Load price panel for multiple symbols.

        Args:
            source: Data source name.
            symbols: List of ticker symbols.
            timeframe: Data timeframe.
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            DataFrame with close prices (index=date, columns=symbols).

        Raises:
            ValueError: If no data could be loaded.
        """
        frames: dict[str, pd.Series] = {}

        for symbol in symbols:
            try:
                df = self._repository.load_prices(source, symbol, timeframe, start_date, end_date)
                if not df.empty and "close" in df.columns:
                    frames[symbol] = df["close"]
                else:
                    logger.warning("No close data for %s", symbol)
            except Exception as e:
                logger.warning("Failed to load %s: %s", symbol, e)

        if not frames:
            raise ValueError("No price data could be loaded for the requested universe")

        panel = pd.DataFrame(frames)
        panel = panel.dropna(how="all")
        return panel

    def delete_dataset(self, source: str, symbol: str, timeframe: str = "1d") -> None:
        """Delete a stored dataset."""
        self._repository.delete_dataset(source, symbol, timeframe)
