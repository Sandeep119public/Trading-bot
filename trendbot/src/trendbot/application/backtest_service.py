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
        """Load price data for the backtest."""
        ds = request.data_selection
        frames: dict[str, pd.Series] = {}

        for symbol in ds.symbols:
            try:
                df = self._repository.load_prices(
                    ds.source, symbol, ds.timeframe, ds.start_date, ds.end_date
                )
                if not df.empty and "close" in df.columns:
                    frames[symbol] = df["close"]
            except Exception as e:
                logger.warning("Skipping %s: %s", symbol, e)

        if not frames:
            raise ValueError("No valid price data found for the selected universe")

        close = pd.DataFrame(frames).dropna(how="all")
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
