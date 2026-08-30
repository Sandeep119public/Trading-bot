"""Orchestration layer for covariance-aware target portfolio construction."""

from __future__ import annotations

import pandas as pd

from trendbot.domain.constraints import apply_constraints
from trendbot.domain.covariance import estimate_covariance
from trendbot.domain.risk import calculate_portfolio_volatility, calculate_volatility_scalar


def construct_target_portfolio(
    returns_history: pd.DataFrame,
    asset_vols: pd.Series,
    signals: pd.Series,
    target_vol: float = 0.10,
    max_gross_leverage: float = 2.0,
    max_asset_weight: float = 1.0,
    cov_shrinkage: float = 0.1,
    fallback_vols: pd.Series | None = None,
) -> pd.Series:
    """Construct the mathematically correct target portfolio.

    Executes the mandated sequence:
        1. historical returns -> covariance estimate
        2. base weights = inverse-vol normalized to 100% gross invested
        3. raw weights = base weights * signals
        4. portfolio vol from raw weights and covariance
        5. target-vol scalar = target_vol / portfolio_vol
        6. apply hard constraints

    CONTRACT: ``returns_history`` must be ALREADY sliced up to t-1 by the caller.
    This function must not perform any time-series slicing itself.

    Args:
        returns_history: Historical returns DataFrame strictly up to t-1.
        asset_vols: Individual asset volatilities at t (same frequency as cov_matrix).
        signals: Directional signals at t in [-1.0, 1.0].
        target_vol: Target portfolio volatility (same frequency as cov_matrix).
        max_gross_leverage: Maximum gross exposure.
        max_asset_weight: Maximum absolute weight per asset.
        cov_shrinkage: Shrinkage intensity for covariance estimation.
        fallback_vols: Asset volatilities for covariance fallback when history is short.

    Returns:
        Series of final constrained target weights.
    """
    # 1. Historical returns -> covariance estimate
    cov_matrix = estimate_covariance(
        returns_history, shrinkage=cov_shrinkage, fallback_vols=fallback_vols
    )

    # 2. Base weights: inverse-volatility normalized to sum to 1.0 (100% gross)
    inv_vols = 1.0 / asset_vols.replace(0, 1e-8)
    base_weights = inv_vols / inv_vols.sum()

    # 3. Raw signed weights
    raw_weights = base_weights * signals.reindex(base_weights.index).fillna(0.0)

    # 4. Portfolio volatility
    port_vol = calculate_portfolio_volatility(raw_weights, cov_matrix)

    # 5. Target-vol scalar
    scalar = calculate_volatility_scalar(port_vol, target_vol)

    # 6. Constraints
    scaled_weights = raw_weights * scalar
    final_weights = apply_constraints(scaled_weights, max_gross_leverage, max_asset_weight)

    return final_weights
