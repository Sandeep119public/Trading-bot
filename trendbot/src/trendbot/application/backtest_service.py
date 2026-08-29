"""Application service for running backtests."""

from __future__ import annotations

import logging

import pandas as pd

from trendbot.application.dto import BacktestDTO
from trendbot.application.ports import PriceRepository
from trendbot.domain.backtest import compute_benchmark_returns, run_backtest
from trendbot.domain.metrics import compute_metrics
from trendbot.domain.models import BacktestRequest, BacktestResult

logger = logging.getLogger(__name__)


class BacktestService:
    """Service for coordinating backtest execution."""

    def __init__(self, repository: PriceRepository) -> None:
        self._repository = repository

    def run(self, request: BacktestRequest) -> BacktestDTO:
        """Execute a full backtest.

        Args:
            request: Complete backtest configuration.

        Returns:
            BacktestDTO with result or error.
        """
        try:
            close = self._load_data(request)
            result = self._execute_backtest(request, close)
            return BacktestDTO(request=request, result=result)
        except Exception as e:
            logger.error("Backtest failed: %s", e)
            return BacktestDTO(request=request, error=str(e))

    def _load_data(self, request: BacktestRequest) -> pd.DataFrame:
        """Load price data for the backtest.

        Raises:
            ValueError: If no data found, or if assets have missing/misaligned data.
        """
        ds = request.data_selection
        frames: dict[str, pd.Series] = {}
        missing_assets: list[str] = []

        for symbol in ds.symbols:
            try:
                df = self._repository.load_prices(
                    ds.source, symbol, ds.timeframe, ds.start_date, ds.end_date
                )
                if not df.empty and "close" in df.columns:
                    frames[symbol] = df["close"]
                else:
                    missing_assets.append(symbol)
            except Exception as e:
                logger.warning("Skipping %s: %s", symbol, e)
                missing_assets.append(symbol)

        if not frames:
            raise ValueError("No valid price data found for the selected universe")

        close = pd.DataFrame(frames).dropna(how="all")

        # Data integrity check: verify all assets have data for the full date range
        issues: list[str] = []

        if missing_assets:
            issues.append(f"No data loaded for: {', '.join(missing_assets)}")

        if ds.start_date is not None and len(close) > 0:
            actual_start = close.index[0].date()
            if actual_start > ds.start_date:
                issues.append(
                    f"Data starts at {actual_start}, but backtest starts at {ds.start_date}"
                )

        if ds.end_date is not None and len(close) > 0:
            actual_end = close.index[-1].date()
            if actual_end < ds.end_date:
                issues.append(
                    f"Data ends at {actual_end}, but backtest ends at {ds.end_date}"
                )

        # Check for assets with significant gaps (>20% missing bars)
        if len(close) > 0:
            expected_bars = len(close)
            for col in close.columns:
                coverage = close[col].notna().sum() / expected_bars
                if coverage < 0.8:
                    issues.append(
                        f"Asset '{col}' has only {coverage:.0%} data coverage "
                        f"({close[col].notna().sum()}/{expected_bars} bars)"
                    )

        # Check for misaligned dates across assets
        if close.shape[1] > 1:
            date_sets = {col: set(close[col].dropna().index) for col in close.columns}
            reference = date_sets[list(date_sets.keys())[0]]
            for col, dates in date_sets.items():
                missing_dates = reference - dates
                if len(missing_dates) > 0:
                    sample = sorted(missing_dates)[:3]
                    sample_str = ", ".join(str(d.date()) for d in sample)
                    issues.append(
                        f"Asset '{col}' missing {len(missing_dates)} dates "
                        f"(e.g., {sample_str})"
                    )

        if issues:
            raise ValueError(
                "Data integrity check failed:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
            )

        return close

    def _execute_backtest(
        self,
        request: BacktestRequest,
        close: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest engine and compute results."""
        mom = request.momentum
        vol = request.volatility
        risk = request.risk
        exec_ = request.execution
        bt = request.backtest

        bt_result = run_backtest(
            close=close,
            lookbacks=mom.lookbacks,
            allow_short=mom.allow_short,
            vol_window=vol.vol_window,
            ann_factor=vol.ann_factor,
            target_portfolio_vol=risk.target_portfolio_vol,
            max_gross_leverage=risk.max_gross_leverage,
            taker_fee_pct=exec_.taker_fee_pct,
            slippage_pct=exec_.slippage_pct,
            rebalance_threshold=exec_.rebalance_threshold,
            min_history=bt.min_history,
        )

        bench_returns = compute_benchmark_returns(close, bt.benchmark.value, bt.min_history)

        metrics = compute_metrics(
            returns=bt_result["returns"],
            positions=bt_result["positions"],
            turnover=bt_result["turnover"],
            costs=bt_result["costs"],
            ann_factor=vol.ann_factor,
            gross_returns=bt_result["gross_returns"],
            min_history=bt.min_history,
        )

        return BacktestResult(
            stats=metrics,
            returns=bt_result["returns"],
            gross_returns=bt_result["gross_returns"],
            positions=bt_result["positions"],
            executed_weights=bt_result["executed_weights"],
            turnover=bt_result["turnover"],
            costs=bt_result["costs"],
            benchmark_returns=bench_returns,
            metadata={
                "universe": ", ".join(close.columns.tolist()),
                "start_date": str(close.index[0].date()),
                "end_date": str(close.index[-1].date()),
                "n_bars": str(len(close)),
                "n_assets": str(close.shape[1]),
            },
        )
