"""Pure linear algebra for portfolio volatility and target-vol scalar."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_portfolio_volatility(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Calculate portfolio volatility: sqrt(w^T * Sigma * w).

    CONTRACT: ``weights`` and ``cov_matrix`` must be on the exact same frequency
    (e.g., both daily or both annualized).

    Args:
        weights: Portfolio weight vector.
        cov_matrix: Covariance matrix.

    Returns:
        Portfolio volatility (annualized or daily, matching inputs).
    """
    aligned_weights = weights.reindex(cov_matrix.index).fillna(0.0).values
    cov_values = cov_matrix.values

    # Verify alignment: weights and covariance must represent the same assets
    # in the same order (after reindex).  This is a hard assertion to catch
    # silent ordering bugs.
    assert cov_matrix.index.equals(cov_matrix.columns), (
        "Covariance matrix index and columns must match"
    )

    variance = float(np.dot(aligned_weights.T, np.dot(cov_values, aligned_weights)))

    # Floating point safeguard
    if variance < 0:
        variance = 0.0

    return np.sqrt(variance)


def calculate_volatility_scalar(current_vol: float, target_vol: float) -> float:
    """Calculate the leverage multiplier to hit the target volatility.

    Args:
        current_vol: Current portfolio volatility.
        target_vol: Desired portfolio volatility.

    Returns:
        Scalar multiplier. Returns 0.0 for flat/dead assets to prevent division by zero.
    """
    if current_vol <= 1e-8:
        return 0.0
    return target_vol / current_vol
