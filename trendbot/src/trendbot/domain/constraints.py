"""Hard risk constraints for portfolio weights."""

from __future__ import annotations

import pandas as pd


def apply_constraints(
    weights: pd.Series,
    max_gross_leverage: float = 2.0,
    max_asset_weight: float = 1.0,
) -> pd.Series:
    """Apply hard constraints while preserving relative signal direction.

    1. Per-asset cap via clipping.
    2. Gross leverage cap via proportional scaling.

    Args:
        weights: Raw scaled portfolio weights.
        max_gross_leverage: Maximum allowed gross exposure (sum of absolute weights).
        max_asset_weight: Maximum absolute weight per asset.

    Returns:
        Constrained weights as a Series.
    """
    # 1. Per-asset cap
    capped_weights = weights.clip(lower=-max_asset_weight, upper=max_asset_weight)

    # 2. Gross leverage cap
    current_gross = capped_weights.abs().sum()

    if current_gross > max_gross_leverage:
        scale_factor = max_gross_leverage / current_gross
        capped_weights = capped_weights * scale_factor

    return capped_weights
