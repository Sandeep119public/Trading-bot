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
    """Run the multi-horizon trend-following backtest.

    Uses a strict causal event-driven loop.  At each timestamp ``t`` the engine
    uses only information available at or before ``t`` to determine the target
    portfolio.  The executed position is held from ``t`` to ``t+1`` and earns
    the return observed at ``t+1``.

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
        covariance_window: Lookback window for covariance estimation.
        covariance_shrinkage: Shrinkage intensity for covariance estimation.

    Returns:
        Dictionary with keys: returns, gross_returns, positions, executed_weights,
        turnover, costs.
    """
    columns = close.columns.tolist()
    n_bars = len(close)

    daily_returns = (close / close.shift(1) - 1).fillna(0.0)

    # --- Warmup gate ---------------------------------------------------------
    required_history = max(
        min_history,
        max(lookbacks) if lookbacks else 1,
        vol_window,
        covariance_window,
    )

    # --- Allocate output arrays -----------------------------------------------
    ret_arr = np.zeros(n_bars, dtype=np.float64)
    gross_ret_arr = np.zeros(n_bars, dtype=np.float64)
    positions_arr = np.zeros((n_bars, len(columns)), dtype=np.float64)
    executed_arr = np.zeros((n_bars, len(columns)), dtype=np.float64)
    turnover_arr = np.zeros(n_bars, dtype=np.float64)
    costs_arr = np.zeros(n_bars, dtype=np.float64)

    previous_position = pd.Series(0.0, index=columns)

    # --- Event-driven loop ----------------------------------------------------
    for i in range(required_history, n_bars):
        # 1. INFORMATION SET (strictly up to i)
        history_start = max(0, i - covariance_window)
        historical_returns = daily_returns.iloc[history_start:i]

        price_history = close.iloc[: i + 1]

        # 2. SIGNAL
        signal_df = compute_momentum_signals(price_history, lookbacks, allow_short)
        signals = signal_df.iloc[-1].reindex(columns).fillna(0.0)

        # 3. INDIVIDUAL VOLATILITY
        asset_vol_df = compute_asset_volatility(price_history, vol_window, ann_factor)
        asset_vols = asset_vol_df.iloc[-1].reindex(columns)

        # Remove assets for which risk estimates don't exist
        valid = (
            asset_vols.notna()
            & np.isfinite(asset_vols)
            & (asset_vols > 0)
            & signals.notna()
        )
        signals = signals.where(valid, 0.0)
        asset_vols = asset_vols.where(valid)

        # 4. TARGET PORTFOLIO
        target = construct_target_portfolio(
            returns_history=historical_returns,
            asset_vols=asset_vols,
            signals=signals,
            target_vol=target_portfolio_vol,
            max_gross_leverage=max_gross_leverage,
            max_asset_weight=1.0,
            cov_shrinkage=covariance_shrinkage,
            fallback_vols=asset_vols,
        )
        target = target.reindex(columns).fillna(0.0)

        # 5. EXECUTION DECISION
        diff = (target - previous_position).abs()
        execute_mask = diff > rebalance_threshold
        new_position = previous_position.where(~execute_mask, target)

        # Hard leverage safeguard
        gross = new_position.abs().sum()
        if gross > max_gross_leverage:
            new_position *= max_gross_leverage / gross

        # 6. TRADE ACCOUNTING
        trade_size = (new_position - previous_position).abs()
        day_turnover = trade_size.sum()
        trading_cost = day_turnover * (taker_fee_pct + slippage_pct)

        turnover_arr[i] = day_turnover
        costs_arr[i] = trading_cost

        # 7. POSITION HELD FROM i -> i+1
        executed_arr[i] = new_position.values
        positions_arr[i] = new_position.values
        previous_position = new_position

    # --- Realized returns (position at t earns return at t+1) -----------------
    for i in range(required_history, n_bars):
        if i > 0:
            pos = positions_arr[i - 1]
            ret = daily_returns.iloc[i].values
            gross_ret_arr[i] = float(np.dot(pos, ret))
        ret_arr[i] = gross_ret_arr[i] - costs_arr[i]

    # First bar always earns zero (no prior position)
    # Bars before required_history stay at zero (no strategy)

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
