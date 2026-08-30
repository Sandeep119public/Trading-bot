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
    ann_factor: int = 365,
) -> pd.Series:
    """Construct the covariance-aware target portfolio.

    All risk quantities are annualized: ``asset_vols``, covariance, portfolio
    volatility, and ``target_vol`` must use the same units.
    """
    cov_matrix = estimate_covariance(
        returns_history,
        shrinkage=cov_shrinkage,
        fallback_vols=fallback_vols,
        ann_factor=ann_factor,
    )

    inv_vols = 1.0 / asset_vols.replace(0, 1e-8)
    base_weights = inv_vols / inv_vols.sum()

    raw_weights = base_weights * signals.reindex(base_weights.index).fillna(0.0)

    port_vol = calculate_portfolio_volatility(raw_weights, cov_matrix)
    scalar = calculate_volatility_scalar(port_vol, target_vol)

    scaled_weights = raw_weights * scalar
    return apply_constraints(
        scaled_weights,
        max_gross_leverage,
        max_asset_weight,
    )
