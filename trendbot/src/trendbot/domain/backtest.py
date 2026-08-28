"""Backtest simulation engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendbot.domain.signals import compute_momentum_signals
from trendbot.domain.sizing import (
    apply_leverage_cap,
    compute_asset_volatility,
    compute_raw_weights,
)


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
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run the multi-horizon trend-following backtest.

    Args:
        close: DataFrame of daily close prices (index=date, columns=assets).
        lookbacks: Momentum lookback periods.
        allow_short: Whether to allow short positions.
        vol_window: Rolling volatility window.
        ann_factor: Annualization factor.
        target_portfolio_vol: Target portfolio volatility.
        max_gross_leverage: Maximum gross leverage.
        taker_fee_pct: Taker fee as a decimal fraction (e.g. 0.001 = 0.1%).
        slippage_pct: Slippage as a decimal fraction (e.g. 0.0005 = 0.05%).
        rebalance_threshold: Minimum weight change to trigger rebalance.
        min_history: Minimum bars before trading starts.

    Returns:
        Dictionary with keys: returns, positions, executed_weights, turnover, costs.
    """
    daily_returns = close / close.shift(1) - 1

    score_norm = compute_momentum_signals(close, lookbacks, allow_short)
    asset_vol = compute_asset_volatility(close, vol_window, ann_factor)

    num_tradable = (asset_vol > 0).sum(axis=1)
    num_tradable = num_tradable.replace(0, np.nan)

    raw_weights = compute_raw_weights(score_norm, asset_vol, target_portfolio_vol, num_tradable)
    target_weights = apply_leverage_cap(raw_weights, max_gross_leverage)

    n_bars, n_assets = close.shape
    cols = close.columns.tolist()
    idx = close.index

    target_arr = target_weights.values.astype(np.float64)
    executed_arr = np.zeros((n_bars, n_assets), dtype=np.float64)

    for i in range(1, n_bars):
        prev = executed_arr[i - 1]
        tgt = target_arr[i]

        diff = np.abs(tgt - prev)
        exec_row = np.where(diff > rebalance_threshold, tgt, prev)

        gross = np.abs(exec_row).sum()
        if gross > max_gross_leverage and gross > 0:
            exec_row = exec_row * max_gross_leverage / gross

        executed_arr[i] = exec_row

    positions_arr = np.roll(executed_arr, 1, axis=0)
    positions_arr[0] = 0.0

    diff = np.diff(positions_arr, axis=0, prepend=np.zeros((1, n_assets)))
    turnover_arr = np.abs(diff).sum(axis=1)
    turnover_arr[0] = np.abs(positions_arr[0]).sum()

    cost_rate = taker_fee_pct + slippage_pct
    costs_arr = turnover_arr * cost_rate

    gross_ret_arr = (positions_arr * daily_returns.values).sum(axis=1)
    ret_arr = gross_ret_arr - costs_arr

    history_mask = np.zeros(n_bars, dtype=bool)
    if min_history > 0 and n_bars > min_history:
        history_mask[min_history:] = True
    else:
        history_mask[:] = True

    ret_arr = np.where(history_mask, ret_arr, 0.0)
    gross_ret_arr = np.where(history_mask, gross_ret_arr, 0.0)
    positions_arr = np.where(history_mask[:, None], positions_arr, 0.0)
    executed_arr = np.where(history_mask[:, None], executed_arr, 0.0)
    turnover_arr = np.where(history_mask, turnover_arr, 0.0)
    costs_arr = np.where(history_mask, costs_arr, 0.0)

    returns = pd.Series(ret_arr, index=idx, name="returns")
    gross_returns = pd.Series(gross_ret_arr, index=idx, name="gross_returns")
    positions = pd.DataFrame(positions_arr, index=idx, columns=cols)
    executed_weights = pd.DataFrame(executed_arr, index=idx, columns=cols)
    turnover = pd.Series(turnover_arr, index=idx, name="turnover")
    costs = pd.Series(costs_arr, index=idx, name="costs")

    return {
        "returns": returns,
        "gross_returns": gross_returns,
        "positions": positions,
        "executed_weights": executed_weights,
        "turnover": turnover,
        "costs": costs,
    }


def compute_benchmark_returns(
    close: pd.DataFrame,
    benchmark_type: str,
    min_history: int,
) -> pd.Series | None:
    """Compute benchmark returns.

    Args:
        close: DataFrame of daily close prices.
        benchmark_type: Type of benchmark ('none', 'equal_weight').
        min_history: Minimum bars before trading starts.

    Returns:
        Series of benchmark returns or None.
    """
    if benchmark_type == "none":
        return None

    daily_returns = close / close.shift(1) - 1
    bench = daily_returns.mean(axis=1)

    n_bars = len(close)
    history_mask = np.zeros(n_bars, dtype=bool)
    if min_history > 0 and n_bars > min_history:
        history_mask[min_history:] = True
    else:
        history_mask[:] = True

    return pd.Series(np.where(history_mask, bench.values, 0.0), index=close.index)
