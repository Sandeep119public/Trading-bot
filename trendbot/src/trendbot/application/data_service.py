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

    def _provider_for(self, request: DataDownloadRequest) -> DataProvider:
        """Resolve the concrete provider for a download request."""
        timeframe = getattr(request, "timeframe", "1d") or "1d"
        if request.source == "binance":
            from trendbot.infrastructure.data_providers.ccxt_provider import CcxtProvider
            return CcxtProvider(
                quote_currency=getattr(request, "quote_currency", "USDT"),
                timeframe=timeframe,
            )
        if request.source == "binance_futures":
            from trendbot.infrastructure.data_providers.ccxt_provider import BinanceFuturesProvider
            return BinanceFuturesProvider(
                quote_currency=getattr(request, "quote_currency", "USDT"),
                timeframe=timeframe,
            )
        if request.source == "delta_india":
            from trendbot.infrastructure.data_providers.delta_india_provider import (
                DeltaIndiaProvider,
            )
            return DeltaIndiaProvider(timeframe=timeframe)
        if request.source == "yfinance":
            return self._provider
        return self._provider

    def download_data(self, request: DataDownloadRequest) -> DownloadResult:
        """Download data from the selected market-data provider and store it locally."""
        processed: list[str] = []
        failed: list[str] = []
        timeframe = getattr(request, "timeframe", "1d") or "1d"
        provider = self._provider_for(request)

        for symbol in request.symbols:
            try:
                normalized = symbol.strip().upper()
                if not request.overwrite and self._repository.dataset_exists(
                    request.source, normalized, timeframe
                ):
                    logger.info("Skipping %s (already exists, overwrite=False)", normalized)
                    processed.append(normalized)
                    continue

                df = provider.fetch_daily_close_prices(
                    normalized, request.start_date, request.end_date
                )
                if df.empty:
                    logger.warning("Empty data for %s", normalized)
                    failed.append(f"{normalized}: empty data")
                    continue

                self._repository.save_prices(request.source, normalized, timeframe, df)

                try:
                    vol_df = provider.fetch_daily_volume(
                        normalized, request.start_date, request.end_date
                    )
                    if not vol_df.empty:
                        self._repository.save_volume(
                            request.source, normalized, timeframe, vol_df
                        )
                except (AttributeError, NotImplementedError):
                    pass

                processed.append(normalized)
                logger.info(
                    "Downloaded %s: %d rows from %s",
                    normalized,
                    len(df),
                    request.source,
                )
            except Exception as e:
                logger.error("Failed to download %s from %s: %s", symbol, request.source, e)
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
        """Load close-price panel for multiple stored symbols."""
        frames: dict[str, pd.Series] = {}
        for symbol in symbols:
            try:
                df = self._repository.load_prices(source, symbol, timeframe, start_date, end_date)
                if df.empty:
                    logger.warning("No close data for %s", symbol)
                elif "close" in df.columns:
                    frames[symbol] = df["close"]
                elif symbol.upper() in df.columns:
                    frames[symbol] = df[symbol.upper()]
                elif len(df.columns) == 1:
                    frames[symbol] = df.iloc[:, 0]
                else:
                    logger.warning("No close data for %s", symbol)
            except Exception as e:
                logger.warning("Failed to load %s: %s", symbol, e)

        if not frames:
            raise ValueError("No price data could be loaded for the requested universe")
        return pd.DataFrame(frames).dropna(how="all")

    def load_volume_data(
        self,
        source: str,
        symbols: list[str],
        timeframe: str,
        start_date: date | None,
        end_date: date | None,
    ) -> pd.DataFrame:
        """Load base-volume panel for multiple stored symbols."""
        frames: dict[str, pd.Series] = {}
        for symbol in symbols:
            try:
                df = self._repository.load_volume(
                    source, symbol, timeframe, start_date, end_date
                )
                if df.empty:
                    logger.warning("No volume data for %s", symbol)
                elif "volume" in df.columns:
                    frames[symbol] = df["volume"]
                elif symbol.upper() in df.columns:
                    frames[symbol] = df[symbol.upper()]
                elif len(df.columns) == 1:
                    frames[symbol] = df.iloc[:, 0]
                else:
                    logger.warning("No volume data for %s", symbol)
            except Exception as e:
                logger.warning("Failed to load volume for %s: %s", symbol, e)

        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames).dropna(how="all")

    def delete_dataset(self, source: str, symbol: str, timeframe: str = "1d") -> None:
        """Delete a stored dataset."""
        self._repository.delete_dataset(source, symbol, timeframe)
