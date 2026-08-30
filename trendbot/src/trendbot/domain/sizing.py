"""Position sizing logic for the trend-following strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_asset_volatility(
    close: pd.DataFrame,
    vol_window: int,
    ann_factor: int,
) -> pd.DataFrame:
    """Compute annualized rolling volatility for each asset.

    Args:
        close: DataFrame of daily close prices.
        vol_window: Rolling window for standard deviation.
        ann_factor: Annualization factor (252 or 365).

    Returns:
        DataFrame of annualized volatility per asset.
    """
    daily_returns = close / close.shift(1) - 1
    asset_vol = daily_returns.rolling(window=vol_window, min_periods=vol_window).std() * np.sqrt(
        ann_factor
    )
    return asset_vol


def compute_raw_weights(
    score_norm: pd.DataFrame,
    asset_vol: pd.DataFrame,
    target_portfolio_vol: float,
    num_assets: pd.Series,
) -> pd.DataFrame:
    """Compute raw target weights before leverage capping.

    Args:
        score_norm: Normalized momentum scores.
        asset_vol: Annualized asset volatility.
        target_portfolio_vol: Target portfolio volatility.
        num_assets: Number of tradable assets per bar.

    Returns:
        DataFrame of raw target weights.
    """
    risk_budget = target_portfolio_vol / num_assets.replace(0, np.nan)
    risk_budget_aligned = risk_budget.reindex(score_norm.index)
    raw = score_norm.multiply(risk_budget_aligned, axis=0) / asset_vol.replace(0, np.nan)
    return raw.fillna(0.0)


def apply_leverage_cap(
    weights: pd.DataFrame,
    max_gross_leverage: float,
) -> pd.DataFrame:
    """Scale weights to respect gross leverage cap.

    Args:
        weights: DataFrame of target weights.
        max_gross_leverage: Maximum allowed gross exposure.

    Returns:
        DataFrame of capped weights.
    """
    gross = weights.abs().sum(axis=1)
    safe_gross = gross.replace(0, np.nan)
    scale = np.where(safe_gross > max_gross_leverage, max_gross_leverage / safe_gross, 1.0)
    scale = np.nan_to_num(scale, nan=0.0)
    return weights.multiply(scale, axis=0)


def apply_rebalance_threshold(
    target: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Apply rebalance threshold - keep current weight if change is small.

    Args:
        target: Target weights DataFrame.
        current: Current weights DataFrame.
        threshold: Minimum weight change to trigger rebalance.

    Returns:
        DataFrame of executed weights.
    """
    diff = (target - current).abs()
    mask = diff > threshold
    executed = target.where(mask, current)
    return executed
