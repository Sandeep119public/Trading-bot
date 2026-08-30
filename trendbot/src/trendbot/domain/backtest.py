"""Backtest simulation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendbot.domain.portfolio import construct_target_portfolio
from trendbot.domain.signals import compute_momentum_signals
from trendbot.domain.sizing import compute_asset_volatility


def run_backtest(
    close: pd.DataFrame,
    lookbacks: list[int],
    allow_short: bool,
    vol_window: int,
    ann_factor: int,
    target_portfolio_vol: float,
    max_gross_leverage: float,
    taker_fee_pct: float,
    slippage_pct: float,
    rebalance_threshold: float,
    min_history: int,
    covariance_window: int = 60,
    covariance_shrinkage: float = 0.1,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run the multi-horizon trend-following backtest causally."""
    columns = close.columns.tolist()
    n_bars = len(close)
    daily_returns = (close / close.shift(1) - 1).fillna(0.0)

    required_history = max(
        min_history,
        max(lookbacks) if lookbacks else 1,
        vol_window,
        covariance_window,
    )

    ret_arr = np.zeros(n_bars, dtype=np.float64)
    gross_ret_arr = np.zeros(n_bars, dtype=np.float64)
    positions_arr = np.zeros((n_bars, len(columns)), dtype=np.float64)
    executed_arr = np.zeros((n_bars, len(columns)), dtype=np.float64)
    turnover_arr = np.zeros(n_bars, dtype=np.float64)
    costs_arr = np.zeros(n_bars, dtype=np.float64)

    previous_position = pd.Series(0.0, index=columns)

    for i in range(required_history, n_bars):
        # Information available at t. Covariance intentionally uses returns
        # through t-1; the close at t is used for signal and current vol.
        history_start = max(0, i - covariance_window)
        historical_returns = daily_returns.iloc[history_start:i]
        price_history = close.iloc[: i + 1]

        signal_df = compute_momentum_signals(price_history, lookbacks, allow_short)
        signals = signal_df.iloc[-1].reindex(columns).fillna(0.0)

        asset_vol_df = compute_asset_volatility(price_history, vol_window, ann_factor)
        asset_vols = asset_vol_df.iloc[-1].reindex(columns)

        valid = (
            asset_vols.notna()
            & np.isfinite(asset_vols)
            & (asset_vols > 0)
            & signals.notna()
        )
        signals = signals.where(valid, 0.0)
        asset_vols = asset_vols.where(valid)

        target = construct_target_portfolio(
            returns_history=historical_returns,
            asset_vols=asset_vols,
            signals=signals,
            target_vol=target_portfolio_vol,
            max_gross_leverage=max_gross_leverage,
            max_asset_weight=1.0,
            cov_shrinkage=covariance_shrinkage,
            fallback_vols=asset_vols,
            ann_factor=ann_factor,
        ).reindex(columns).fillna(0.0)

        diff = (target - previous_position).abs()
        execute_mask = diff > rebalance_threshold
        new_position = previous_position.where(~execute_mask, target)

        gross = new_position.abs().sum()
        if gross > max_gross_leverage:
            new_position *= max_gross_leverage / gross

        trade_size = (new_position - previous_position).abs()
        day_turnover = float(trade_size.sum())
        trading_cost = day_turnover * (taker_fee_pct + slippage_pct)

        turnover_arr[i] = day_turnover
        costs_arr[i] = trading_cost
        executed_arr[i] = new_position.values
        positions_arr[i] = new_position.values
        previous_position = new_position

    # Position at t-1 earns the market return observed at t. Trading costs at t
    # are charged at the t execution timestamp, so cost attribution is explicit.
    for i in range(required_history + 1, n_bars):
        gross_ret_arr[i] = float(
            np.dot(positions_arr[i - 1], daily_returns.iloc[i].values)
        )
        ret_arr[i] = gross_ret_arr[i] - costs_arr[i]

    idx = close.index
    return {
        "returns": pd.Series(ret_arr, index=idx, name="returns"),
        "gross_returns": pd.Series(gross_ret_arr, index=idx, name="gross_returns"),
        "positions": pd.DataFrame(positions_arr, index=idx, columns=columns),
        "executed_weights": pd.DataFrame(executed_arr, index=idx, columns=columns),
        "turnover": pd.Series(turnover_arr, index=idx, name="turnover"),
        "costs": pd.Series(costs_arr, index=idx, name="costs"),
    }


def compute_benchmark_returns(
    close: pd.DataFrame,
    benchmark_type: str,
    min_history: int,
) -> pd.Series | None:
    """Compute benchmark returns."""
    if benchmark_type == "none":
        return None

    coverage = close.notna().mean()
    well_covered = coverage[coverage > 0.8].index
    filtered_close = close[well_covered]

    daily_returns = filtered_close / filtered_close.shift(1) - 1
    bench = daily_returns.mean(axis=1)

    n_bars = len(close)
    history_mask = np.zeros(n_bars, dtype=bool)
    if min_history > 0 and n_bars > min_history:
        history_mask[min_history:] = True
    else:
        history_mask[:] = True

    return pd.Series(np.where(history_mask, bench.values, 0.0), index=close.index)
