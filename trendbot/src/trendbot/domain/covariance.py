"""Covariance matrix estimation with numerical safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_covariance(
    returns: pd.DataFrame,
    shrinkage: float = 0.1,
    fallback_vols: pd.Series | None = None,
    ann_factor: int = 365,
) -> pd.DataFrame:
    """Estimate an annualized covariance matrix from daily returns.

    The caller must provide a history containing no future information. The
    returned covariance matrix is always annualized, matching annualized asset
    volatilities and annualized target volatility.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1")
    if ann_factor <= 0:
        raise ValueError("ann_factor must be positive")

    columns = returns.columns
    clean_returns = returns.loc[:, columns].dropna(how="any")

    if clean_returns.empty or len(clean_returns) < 5:
        if fallback_vols is not None:
            aligned_vols = fallback_vols.reindex(columns).fillna(0.01)
        else:
            aligned_vols = pd.Series(0.01, index=columns, dtype=float)

        aligned_vols = aligned_vols.astype(float).clip(lower=1e-8)
        variances = aligned_vols.to_numpy() ** 2
        return pd.DataFrame(np.diag(variances), index=columns, columns=columns)

    # Input returns are daily. Convert the sample covariance to annual units.
    sample_cov = clean_returns.cov().to_numpy(dtype=float) * float(ann_factor)

    if shrinkage > 0.0:
        diagonal = np.diag(np.diag(sample_cov))
        sample_cov = (1.0 - shrinkage) * sample_cov + shrinkage * diagonal

    # Stabilize the matrix while preserving its scale.
    sample_cov = (sample_cov + sample_cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)
    floor = max(float(np.max(np.diag(sample_cov))) * 1e-10, 1e-12)
    eigenvalues = np.maximum(eigenvalues, floor)
    stable_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    stable_cov = (stable_cov + stable_cov.T) / 2.0

    return pd.DataFrame(stable_cov, index=columns, columns=columns)
