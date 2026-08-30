"""Covariance matrix estimation with numerical safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_covariance(
    returns: pd.DataFrame,
    shrinkage: float = 0.1,
    fallback_vols: pd.Series | None = None,
) -> pd.DataFrame:
    """Estimate the covariance matrix from a returns DataFrame.

    CONTRACT: The caller MUST ensure ``returns`` contains only information
    available at or before time t-1 to prevent lookahead bias.

    Args:
        returns: DataFrame of daily returns (index=date, columns=assets).
        shrinkage: Shrinkage intensity towards the diagonal (0 = none, 1 = pure diagonal).
        fallback_vols: Asset volatilities used to build a diagonal covariance
            matrix when history is insufficient. If None, a default variance
            of 0.0001 is used.

    Returns:
        Positive-definite covariance matrix as a DataFrame.
    """
    clean_returns = returns.dropna()

    if clean_returns.empty or len(clean_returns) < 5:
        if fallback_vols is not None:
            aligned_vols = fallback_vols.reindex(returns.columns).fillna(0.01)
            variances = aligned_vols.values ** 2
            return pd.DataFrame(
                np.diag(variances),
                index=returns.columns,
                columns=returns.columns,
            )
        default_var = 0.01 ** 2
        return pd.DataFrame(
            np.eye(len(returns.columns)) * default_var,
            index=returns.columns,
            columns=returns.columns,
        )

    sample_cov = clean_returns.cov().values.copy()

    if shrinkage > 0:
        diag = np.diag(np.diag(sample_cov))
        sample_cov = (1 - shrinkage) * sample_cov + shrinkage * diag

    # Force positive definiteness via nugget effect
    try:
        np.linalg.cholesky(sample_cov)
    except np.linalg.LinAlgError:
        epsilon = 1e-5
        sample_cov += np.eye(len(sample_cov)) * epsilon

    return pd.DataFrame(sample_cov, index=returns.columns, columns=returns.columns)
