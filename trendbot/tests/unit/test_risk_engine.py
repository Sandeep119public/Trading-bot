"""Deterministic tests for the covariance-aware risk engine."""

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.covariance import estimate_covariance
from trendbot.domain.risk import calculate_portfolio_volatility, calculate_volatility_scalar
from trendbot.domain.constraints import apply_constraints
from trendbot.domain.portfolio import construct_target_portfolio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_returns():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    base = np.random.normal(0, 0.01, 100)
    asset_a = base + np.random.normal(0, 0.005, 100)
    asset_b = base + np.random.normal(0, 0.005, 100)  # Highly correlated with A
    asset_c = np.random.normal(0, 0.02, 100)  # Uncorrelated
    return pd.DataFrame({"A": asset_a, "B": asset_b, "C": asset_c}, index=dates)


@pytest.fixture
def mock_inputs(mock_returns):
    asset_vols = pd.Series({"A": 0.01, "B": 0.01, "C": 0.02})
    signals = pd.Series({"A": 1.0, "B": -1.0, "C": 0.5})
    return {
        "returns_history": mock_returns,
        "asset_vols": asset_vols,
        "signals": signals,
        "target_vol": 0.005,
        "max_gross_leverage": 2.0,
        "max_asset_weight": 0.5,
        "cov_shrinkage": 0.1,
    }


# ---------------------------------------------------------------------------
# Component Tests
# ---------------------------------------------------------------------------

def test_covariance_stability(mock_returns):
    """Singular matrices (perfectly correlated assets) must not crash."""
    returns = mock_returns.copy()
    returns["D"] = returns["A"] * 2.0  # Perfectly correlated -> singular

    cov = estimate_covariance(returns, shrinkage=0.0)

    # Must be positive definite (Cholesky succeeds)
    try:
        np.linalg.cholesky(cov.values)
    except np.linalg.LinAlgError:
        pytest.fail("Covariance matrix is not positive definite")


def test_covariance_shrinkage(mock_returns):
    """Shrinkage must pull the matrix towards the diagonal."""
    cov_no_shrink = estimate_covariance(mock_returns, shrinkage=0.0)
    cov_shrink = estimate_covariance(mock_returns, shrinkage=0.5)

    # Off-diagonal elements should be smaller with more shrinkage
    off_diag_no = np.abs(cov_no_shrink.values).sum() - np.trace(np.abs(cov_no_shrink.values))
    off_diag_shrink = np.abs(cov_shrink.values).sum() - np.trace(np.abs(cov_shrink.values))
    assert off_diag_shrink < off_diag_no


def test_covariance_insufficient_history():
    """Fewer than 5 rows must return a diagonal matrix using default variance."""
    short_returns = pd.DataFrame(
        {"A": [0.01, 0.02], "B": [0.03, 0.04]},
        index=pd.date_range("2023-01-01", periods=2),
    )
    cov = estimate_covariance(short_returns)
    expected = np.eye(2) * (0.01 ** 2)
    np.testing.assert_array_almost_equal(cov.values, expected)


def test_risk_math():
    """Core linear algebra: w^T * Sigma * w with a known 2x2 case."""
    weights = pd.Series({"A": 0.5, "B": 0.5})
    cov = pd.DataFrame({"A": [0.04, 0.01], "B": [0.01, 0.04]}, index=["A", "B"])

    # Expected: sqrt(0.5^2*0.04 + 0.5^2*0.04 + 2*0.5*0.5*0.01) = sqrt(0.025)
    vol = calculate_portfolio_volatility(weights, cov)
    assert np.isclose(vol, np.sqrt(0.025))


def test_risk_zero_weights():
    """Zero weights must produce zero volatility."""
    weights = pd.Series({"A": 0.0, "B": 0.0})
    cov = pd.DataFrame({"A": [0.04, 0.01], "B": [0.01, 0.04]}, index=["A", "B"])
    assert calculate_portfolio_volatility(weights, cov) == 0.0


def test_risk_missing_weight_defaults_to_zero():
    """Missing asset in weights defaults to 0."""
    weights = pd.Series({"A": 1.0})  # B is missing
    cov = pd.DataFrame({"A": [0.04, 0.01], "B": [0.01, 0.04]}, index=["A", "B"])
    vol = calculate_portfolio_volatility(weights, cov)
    assert np.isclose(vol, np.sqrt(0.04))


def test_volatility_scalar():
    assert calculate_volatility_scalar(0.02, 0.10) == pytest.approx(5.0)


def test_volatility_scalar_flat_asset():
    assert calculate_volatility_scalar(0.0, 0.10) == 0.0
    assert calculate_volatility_scalar(1e-10, 0.10) == 0.0


def test_constraints_scaling():
    """Gross leverage cap must scale proportionally without flipping signs."""
    weights = pd.Series({"A": 1.2, "B": -0.8, "C": 0.5})  # Gross = 2.5

    capped = apply_constraints(weights, max_gross_leverage=2.0, max_asset_weight=1.0)

    # A clipped from 1.2 -> 1.0; gross becomes 1.0 + 0.8 + 0.5 = 2.3
    # Scaled down by 2.0 / 2.3
    assert np.isclose(capped.abs().sum(), 2.0)
    assert capped["B"] < 0  # Sign preserved
    assert capped["A"] <= 1.0 + 1e-9
    assert capped["C"] <= 1.0 + 1e-9


def test_constraints_below_cap():
    """Weights below the cap should pass through unchanged."""
    weights = pd.Series({"A": 0.3, "B": -0.2})
    capped = apply_constraints(weights, max_gross_leverage=2.0, max_asset_weight=1.0)
    pd.testing.assert_series_equal(capped, weights)


def test_constraints_asset_clip_before_gross():
    """Asset clipping happens before gross scaling."""
    weights = pd.Series({"A": 1.5, "B": -1.5})
    capped = apply_constraints(weights, max_gross_leverage=3.0, max_asset_weight=1.0)

    # After clip: A=1.0, B=-1.0, gross=2.0 (< 3.0), no scaling
    assert np.isclose(capped["A"], 1.0)
    assert np.isclose(capped["B"], -1.0)
    assert np.isclose(capped.abs().sum(), 2.0)


# ---------------------------------------------------------------------------
# Integration / Orchestration Test
# ---------------------------------------------------------------------------

def test_portfolio_orchestration_sequence(mock_inputs):
    """Verify the full sequence matches manual step-by-step calculation."""
    final_weights = construct_target_portfolio(**mock_inputs)

    # 1. Constraints respected
    assert final_weights.abs().sum() <= mock_inputs["max_gross_leverage"] + 1e-6
    assert final_weights.abs().max() <= mock_inputs["max_asset_weight"] + 1e-6

    # 2. Signal directions respected
    assert final_weights["A"] >= 0
    assert final_weights["B"] <= 0
    assert final_weights["C"] >= 0

    # 3. Independent manual calculation
    cov = estimate_covariance(mock_inputs["returns_history"], shrinkage=0.1)
    inv_vols = 1.0 / mock_inputs["asset_vols"]
    base = inv_vols / inv_vols.sum()
    raw = base * mock_inputs["signals"]
    port_vol = calculate_portfolio_volatility(raw, cov)
    scalar = calculate_volatility_scalar(port_vol, mock_inputs["target_vol"])
    expected_scaled = raw * scalar
    expected_final = apply_constraints(
        expected_scaled,
        mock_inputs["max_gross_leverage"],
        mock_inputs["max_asset_weight"],
    )

    pd.testing.assert_series_equal(final_weights, expected_final, check_names=False)
