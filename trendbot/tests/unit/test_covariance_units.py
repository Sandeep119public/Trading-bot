"""Regression tests for annualized covariance units and causal risk accounting.

These tests exist to prevent unit-consistency regressions.  If anyone removes
the ``* ann_factor`` from ``estimate_covariance``, these tests will fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.backtest import run_backtest
from trendbot.domain.constraints import apply_constraints
from trendbot.domain.covariance import estimate_covariance
from trendbot.domain.portfolio import construct_target_portfolio
from trendbot.domain.risk import calculate_portfolio_volatility, calculate_volatility_scalar

# ---------------------------------------------------------------------------
# 1. Covariance annualization regression
# ---------------------------------------------------------------------------


class TestCovarianceAnnualization:
    """Verify that sample covariance is multiplied by ann_factor."""

    def test_annualization_occurs(self):
        """If * ann_factor is removed from estimate_covariance, this test fails."""
        ann_factor = 365
        np.random.seed(123)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        daily_returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 50),
                "B": np.random.normal(0, 0.015, 50),
            },
            index=dates,
        )

        expected_daily_cov = daily_returns.cov()
        expected_annual_cov = expected_daily_cov * ann_factor

        actual = estimate_covariance(
            daily_returns,
            shrinkage=0.0,
            ann_factor=ann_factor,
        )

        np.testing.assert_allclose(
            actual.values,
            expected_annual_cov.values,
            rtol=1e-10,
            err_msg="Covariance is not annualized — * ann_factor may have been removed",
        )

    def test_annualization_252_vs_365(self):
        """Different ann_factor values must produce proportionally different results."""
        np.random.seed(456)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        daily_returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 50),
                "B": np.random.normal(0, 0.015, 50),
            },
            index=dates,
        )

        cov_252 = estimate_covariance(daily_returns, shrinkage=0.0, ann_factor=252)
        cov_365 = estimate_covariance(daily_returns, shrinkage=0.0, ann_factor=365)

        # ratio of any element should be 365/252
        np.testing.assert_allclose(
            cov_365.values / cov_252.values,
            365.0 / 252.0,
            rtol=1e-10,
        )

    def test_shrinkage_preserves_annualization(self):
        """Shrinkage must not strip the annualization factor."""
        ann_factor = 365
        np.random.seed(789)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        daily_returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 50),
                "B": np.random.normal(0, 0.015, 50),
            },
            index=dates,
        )

        cov_no_shrink = estimate_covariance(
            daily_returns, shrinkage=0.0, ann_factor=ann_factor
        )
        cov_shrink = estimate_covariance(
            daily_returns, shrinkage=0.5, ann_factor=ann_factor
        )

        # Diagonal elements are preserved under shrinkage
        np.testing.assert_allclose(
            np.diag(cov_shrink.values),
            np.diag(cov_no_shrink.values),
            rtol=1e-10,
        )

        # Off-diagonals are shrunk towards zero
        off_diag = cov_shrink.values - np.diag(np.diag(cov_shrink.values))
        assert np.abs(off_diag).max() < np.abs(
            cov_no_shrink.values - np.diag(np.diag(cov_no_shrink.values))
        ).max()


# ---------------------------------------------------------------------------
# 2. Fallback covariance units
# ---------------------------------------------------------------------------


class TestFallbackCovarianceUnits:
    """Fallback covariance must be in annualized variance units."""

    def test_fallback_diagonal_variance(self):
        """fallback_vols**2 must appear on the diagonal, off-diag = 0."""
        fallback_vols = pd.Series({"A": 0.20, "B": 0.30})
        short_returns = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, 0.04]},
            index=pd.date_range("2023-01-01", periods=2),
        )

        cov = estimate_covariance(
            short_returns,
            shrinkage=0.0,
            fallback_vols=fallback_vols,
            ann_factor=365,
        )

        expected = pd.DataFrame(
            np.diag([0.20**2, 0.30**2]),
            index=["A", "B"],
            columns=["A", "B"],
        )
        pd.testing.assert_frame_equal(cov, expected)

    def test_fallback_ignores_ann_factor(self):
        """fallback_vols are already annualized; ann_factor must NOT scale them."""
        fallback_vols = pd.Series({"A": 0.20, "B": 0.30})
        short_returns = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, 0.04]},
            index=pd.date_range("2023-01-01", periods=2),
        )

        cov_252 = estimate_covariance(
            short_returns,
            shrinkage=0.0,
            fallback_vols=fallback_vols,
            ann_factor=252,
        )
        cov_365 = estimate_covariance(
            short_returns,
            shrinkage=0.0,
            fallback_vols=fallback_vols,
            ann_factor=365,
        )

        pd.testing.assert_frame_equal(cov_252, cov_365)

    def test_fallback_default_variance(self):
        """No fallback_vols => default annualized variance of 0.01^2."""
        short_returns = pd.DataFrame(
            {"A": [0.01], "B": [0.02]},
            index=pd.date_range("2023-01-01", periods=1),
        )

        cov = estimate_covariance(short_returns, shrinkage=0.0, ann_factor=365)
        expected_var = 0.01**2
        np.testing.assert_allclose(np.diag(cov.values), expected_var)


# ---------------------------------------------------------------------------
# 3. Portfolio volatility exact calculation
# ---------------------------------------------------------------------------


class TestPortfolioVolatilityCalculation:
    """Verify sqrt(w^T @ annual_cov @ w) with known cases."""

    def test_known_2x2_case(self):
        """Hand-calculated 2-asset portfolio volatility."""
        annual_cov = pd.DataFrame(
            {"A": [0.04, 0.01], "B": [0.01, 0.04]},
            index=["A", "B"],
        )
        weights = pd.Series({"A": 0.5, "B": 0.5})

        vol = calculate_portfolio_volatility(weights, annual_cov)
        # variance = 0.25*0.04 + 0.25*0.04 + 2*0.25*0.01 = 0.01 + 0.01 + 0.005 = 0.025
        assert np.isclose(vol, np.sqrt(0.025))

    def test_single_asset(self):
        """Single asset portfolio: vol = asset_vol."""
        cov = pd.DataFrame({"A": [0.04]}, index=["A"])
        weights = pd.Series({"A": 1.0})
        vol = calculate_portfolio_volatility(weights, cov)
        assert np.isclose(vol, 0.20)

    def test_perfect_hedge(self):
        """Two perfectly negatively correlated assets can reduce vol below either individual."""
        cov = pd.DataFrame(
            {"A": [0.04, -0.039], "B": [-0.039, 0.04]},
            index=["A", "B"],
        )
        weights = pd.Series({"A": 0.5, "B": 0.5})
        vol = calculate_portfolio_volatility(weights, cov)
        # variance = 0.25*0.04 + 0.25*0.04 + 2*0.25*(-0.039) = 0.02 - 0.0195 = 0.0005
        assert np.isclose(vol, np.sqrt(0.0005), rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. Volatility targeting chain with annualized units
# ---------------------------------------------------------------------------


class TestVolatilityTargetingChain:
    """Verify that the full chain uses consistent annualized units."""

    def test_scalar_is_annual_ratio(self):
        """target_vol / port_vol must be a ratio of two annualized quantities."""
        # Annualized covariance
        cov = pd.DataFrame(
            {"A": [0.04, 0.01], "B": [0.01, 0.09]},
            index=["A", "B"],
        )
        weights = pd.Series({"A": 0.5, "B": 0.5})

        port_vol = calculate_portfolio_volatility(weights, cov)
        target_vol = 0.10  # 10% annualized

        scalar = calculate_volatility_scalar(port_vol, target_vol)

        # Verify the scalar * port_vol = target_vol
        assert np.isclose(scalar * port_vol, target_vol)

    def test_annualized_consistency_in_portfolio_construction(self):
        """The full construct_target_portfolio chain must use annualized units."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        daily_returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 100),
                "B": np.random.normal(0, 0.015, 100),
            },
            index=dates,
        )

        ann_factor = 365
        asset_vols = pd.Series({"A": 0.20, "B": 0.30})  # annualized
        signals = pd.Series({"A": 1.0, "B": -1.0})
        target_vol = 0.10  # annualized

        final = construct_target_portfolio(
            returns_history=daily_returns,
            asset_vols=asset_vols,
            signals=signals,
            target_vol=target_vol,
            max_gross_leverage=2.0,
            max_asset_weight=1.0,
            cov_shrinkage=0.0,
            ann_factor=ann_factor,
        )

        # Verify constraints
        assert final.abs().sum() <= 2.0 + 1e-6
        assert final.abs().max() <= 1.0 + 1e-6

        # Manually reproduce
        cov = estimate_covariance(daily_returns, shrinkage=0.0, ann_factor=ann_factor)
        inv_vols = 1.0 / asset_vols
        base = inv_vols / inv_vols.sum()
        raw = base * signals
        port_vol = calculate_portfolio_volatility(raw, cov)
        scalar = calculate_volatility_scalar(port_vol, target_vol)
        expected = apply_constraints(raw * scalar, 2.0, 1.0)

        pd.testing.assert_series_equal(final, expected, check_names=False)


# ---------------------------------------------------------------------------
# 5. Strong no-lookahead test
# ---------------------------------------------------------------------------


class TestNoLookahead:
    """Future data must never propagate backward."""

    def test_future_mutation_does_not_affect_past(self):
        """Changing a future price must not alter any position or return before it."""
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        np.random.seed(99)
        a_ret = np.concatenate([np.full(60, 0.002), np.full(60, -0.001)])
        b_ret = np.concatenate([np.full(60, 0.001), np.full(60, 0.0005)])
        a_prices = 100 * np.cumprod(1 + a_ret)
        b_prices = 50 * np.cumprod(1 + b_ret)
        close = pd.DataFrame({"A": a_prices, "B": b_prices}, index=dates)

        params = dict(
            lookbacks=[5, 10],
            allow_short=True,
            vol_window=21,
            ann_factor=365,
            target_portfolio_vol=0.10,
            max_gross_leverage=1.0,
            taker_fee_pct=0.001,
            slippage_pct=0.0005,
            rebalance_threshold=0.01,
            min_history=30,
            covariance_window=60,
            covariance_shrinkage=0.1,
        )

        result_orig = run_backtest(close=close, **params)

        close_mod = close.copy()
        # Mutate the very last price dramatically
        close_mod.iloc[-1, close_mod.columns.get_loc("A")] *= 100.0
        result_mod = run_backtest(close=close_mod, **params)

        n = len(close)
        # Everything before the mutated bar must be identical
        cutoff = n - 2  # position at n-2 doesn't see bar n-1

        pd.testing.assert_frame_equal(
            result_orig["positions"].iloc[:cutoff],
            result_mod["positions"].iloc[:cutoff],
        )
        pd.testing.assert_frame_equal(
            result_orig["executed_weights"].iloc[:cutoff],
            result_mod["executed_weights"].iloc[:cutoff],
        )
        pd.testing.assert_series_equal(
            result_orig["turnover"].iloc[:cutoff],
            result_mod["turnover"].iloc[:cutoff],
        )
        pd.testing.assert_series_equal(
            result_orig["costs"].iloc[:cutoff],
            result_mod["costs"].iloc[:cutoff],
        )
        pd.testing.assert_series_equal(
            result_orig["gross_returns"].iloc[:cutoff],
            result_mod["gross_returns"].iloc[:cutoff],
        )
        pd.testing.assert_series_equal(
            result_orig["returns"].iloc[:cutoff],
            result_mod["returns"].iloc[:cutoff],
        )


# ---------------------------------------------------------------------------
# 6. Next-bar return accounting
# ---------------------------------------------------------------------------


class TestNextBarReturnAccounting:
    """position[t] must earn return[t+1], not return[t]."""

    def test_deterministic_next_bar(self):
        """With a known position and known returns, verify the attribution."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(7)
        prices = pd.DataFrame(
            {
                "A": 100 + np.cumsum(np.random.randn(100) * 0.5),
                "B": 50 + np.cumsum(np.random.randn(100) * 0.3),
            },
            index=dates,
        )

        result = run_backtest(
            close=prices,
            lookbacks=[5],
            allow_short=True,
            vol_window=21,
            ann_factor=365,
            target_portfolio_vol=0.10,
            max_gross_leverage=1.0,
            taker_fee_pct=0.0,
            slippage_pct=0.0,
            rebalance_threshold=0.0,
            min_history=30,
            covariance_window=60,
            covariance_shrinkage=0.1,
        )

        positions = result["positions"]
        gross_returns = result["gross_returns"]
        daily_rets = (prices / prices.shift(1) - 1).fillna(0.0)

        for i in range(1, len(positions)):
            pos_prev = positions.iloc[i - 1].values
            ret_today = daily_rets.iloc[i].values
            expected_gross = float(np.dot(pos_prev, ret_today))
            if abs(expected_gross) > 1e-15:
                assert np.isclose(
                    gross_returns.iloc[i], expected_gross, atol=1e-10
                ), (
                    f"Bar {i}: gross_return={gross_returns.iloc[i]:.8f} "
                    f"!= position[t-1]*return[t] = {expected_gross:.8f}"
                )


# ---------------------------------------------------------------------------
# 7. Warmup gate enforcement
# ---------------------------------------------------------------------------


class TestWarmupGate:
    """No positions, turnover, costs, or returns before required_history."""

    def test_warmup_is_zero(self):
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        np.random.seed(42)
        a_ret = np.concatenate([np.full(60, 0.002), np.full(60, -0.001)])
        b_ret = np.concatenate([np.full(60, 0.001), np.full(60, 0.0005)])
        close = pd.DataFrame(
            {
                "A": 100 * np.cumprod(1 + a_ret),
                "B": 50 * np.cumprod(1 + b_ret),
            },
            index=dates,
        )

        result = run_backtest(
            close=close,
            lookbacks=[5, 10],
            allow_short=True,
            vol_window=21,
            ann_factor=365,
            target_portfolio_vol=0.10,
            max_gross_leverage=1.0,
            taker_fee_pct=0.001,
            slippage_pct=0.0005,
            rebalance_threshold=0.01,
            min_history=30,
            covariance_window=60,
            covariance_shrinkage=0.1,
        )

        # required_history = max(30, 10, 21, 60) = 60
        required_history = 60

        assert (result["turnover"].iloc[:required_history] == 0).all()
        assert (result["costs"].iloc[:required_history] == 0).all()
        assert (result["positions"].iloc[:required_history] == 0).all().all()
        assert (result["returns"].iloc[:required_history] == 0).all()
        assert (result["gross_returns"].iloc[:required_history] == 0).all()


# ---------------------------------------------------------------------------
# 8. Missing data handling
# ---------------------------------------------------------------------------


class TestMissingData:
    """NaN / missing prices must not produce NaN weights or infinite leverage."""

    def test_nan_prices_do_not_corrupt_output(self):
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        np.random.seed(42)
        a = 100 + np.cumsum(np.random.randn(120) * 2)
        b = 50 + np.cumsum(np.random.randn(120) * 1)
        # Inject NaN in the middle of asset B
        b[60:65] = np.nan
        close = pd.DataFrame({"A": a, "B": b}, index=dates)

        result = run_backtest(
            close=close,
            lookbacks=[5],
            allow_short=True,
            vol_window=21,
            ann_factor=365,
            target_portfolio_vol=0.10,
            max_gross_leverage=1.0,
            taker_fee_pct=0.001,
            slippage_pct=0.0005,
            rebalance_threshold=0.01,
            min_history=30,
            covariance_window=60,
            covariance_shrinkage=0.1,
        )

        # No NaN in final outputs
        assert not result["returns"].isna().any()
        assert not result["positions"].isna().any().any()
        assert not result["turnover"].isna().any()
        assert not result["costs"].isna().any()

        # No infinite values
        assert np.isfinite(result["returns"].values).all()
        assert np.isfinite(result["positions"].values).all()


# ---------------------------------------------------------------------------
# 9. Positive-definite stabilization preserves units
# ---------------------------------------------------------------------------


class TestPositiveDefiniteStabilization:
    """Eigenvalue flooring must not distort normal covariance matrices."""

    def test_normal_matrix_unaffected(self):
        """A well-conditioned matrix should pass through without modification."""
        np.random.seed(321)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 50),
                "B": np.random.normal(0, 0.015, 50),
            },
            index=dates,
        )

        cov = estimate_covariance(returns, shrinkage=0.0, ann_factor=365)
        expected = returns.cov() * 365

        np.testing.assert_allclose(cov.values, expected.values, rtol=1e-10)

    def test_perfectly_correlated_positive_definite(self):
        """Perfectly correlated assets must still produce a PD matrix."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        np.random.seed(654)
        a = np.random.normal(0, 0.01, 50)
        returns = pd.DataFrame({"A": a, "B": a * 2.0}, index=dates)

        cov = estimate_covariance(returns, shrinkage=0.0, ann_factor=365)

        try:
            np.linalg.cholesky(cov.values)
        except np.linalg.LinAlgError:
            pytest.fail("Covariance matrix for perfectly correlated assets is not PD")

    def test_cholesky_succeeds(self):
        """Any covariance output must be Cholesky-decomposable."""
        np.random.seed(111)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 100),
                "B": np.random.normal(0, 0.015, 100),
                "C": np.random.normal(0, 0.02, 100),
            },
            index=dates,
        )

        for shrinkage in [0.0, 0.1, 0.5]:
            cov = estimate_covariance(returns, shrinkage=shrinkage, ann_factor=365)
            try:
                np.linalg.cholesky(cov.values)
            except np.linalg.LinAlgError:
                pytest.fail(f"Cholesky failed with shrinkage={shrinkage}")


# ---------------------------------------------------------------------------
# 10. Zero signal produces zero weight
# ---------------------------------------------------------------------------


class TestZeroSignal:
    """Zero signal must produce zero target weights regardless of vol/cov."""

    def test_all_zero_signals(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        daily_returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 100),
                "B": np.random.normal(0, 0.015, 100),
            },
            index=dates,
        )

        asset_vols = pd.Series({"A": 0.20, "B": 0.30})
        signals = pd.Series({"A": 0.0, "B": 0.0})

        final = construct_target_portfolio(
            returns_history=daily_returns,
            asset_vols=asset_vols,
            signals=signals,
            target_vol=0.10,
            max_gross_leverage=2.0,
            max_asset_weight=1.0,
            cov_shrinkage=0.1,
            ann_factor=365,
        )

        np.testing.assert_allclose(final.values, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 11. Gross leverage never exceeds cap
# ---------------------------------------------------------------------------


class TestLeverageCap:
    """Final weights must always respect max_gross_leverage."""

    def test_leverage_respected_with_extreme_signals(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        daily_returns = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 100),
                "B": np.random.normal(0, 0.015, 100),
            },
            index=dates,
        )

        asset_vols = pd.Series({"A": 0.20, "B": 0.30})
        signals = pd.Series({"A": 1.0, "B": 1.0})
        max_gross = 0.5

        final = construct_target_portfolio(
            returns_history=daily_returns,
            asset_vols=asset_vols,
            signals=signals,
            target_vol=0.10,
            max_gross_leverage=max_gross,
            max_asset_weight=1.0,
            cov_shrinkage=0.1,
            ann_factor=365,
        )

        assert final.abs().sum() <= max_gross + 1e-6


# ---------------------------------------------------------------------------
# 12. Transaction-cost timing
# ---------------------------------------------------------------------------


class TestTransactionCostTiming:
    """Costs must be paid at execution time, not at P&L time."""

    def test_costs_non_negative(self):
        """Costs must never be negative."""
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        np.random.seed(42)
        a_ret = np.concatenate([np.full(60, 0.002), np.full(60, -0.001)])
        b_ret = np.concatenate([np.full(60, 0.001), np.full(60, 0.0005)])
        close = pd.DataFrame(
            {
                "A": 100 * np.cumprod(1 + a_ret),
                "B": 50 * np.cumprod(1 + b_ret),
            },
            index=dates,
        )

        result = run_backtest(
            close=close,
            lookbacks=[5],
            allow_short=True,
            vol_window=21,
            ann_factor=365,
            target_portfolio_vol=0.10,
            max_gross_leverage=1.0,
            taker_fee_pct=0.01,
            slippage_pct=0.005,
            rebalance_threshold=0.0,
            min_history=30,
            covariance_window=60,
            covariance_shrinkage=0.1,
        )

        assert (result["costs"] >= -1e-15).all()

    def test_costs_paid_at_same_bar_as_turnover(self):
        """Costs at bar i must be zero when turnover at bar i is zero."""
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        np.random.seed(42)
        a_ret = np.concatenate([np.full(60, 0.002), np.full(60, -0.001)])
        b_ret = np.concatenate([np.full(60, 0.001), np.full(60, 0.0005)])
        close = pd.DataFrame(
            {
                "A": 100 * np.cumprod(1 + a_ret),
                "B": 50 * np.cumprod(1 + b_ret),
            },
            index=dates,
        )

        result = run_backtest(
            close=close,
            lookbacks=[5],
            allow_short=True,
            vol_window=21,
            ann_factor=365,
            target_portfolio_vol=0.10,
            max_gross_leverage=1.0,
            taker_fee_pct=0.01,
            slippage_pct=0.005,
            rebalance_threshold=0.0,
            min_history=30,
            covariance_window=60,
            covariance_shrinkage=0.1,
        )

        turnover = result["turnover"]
        costs = result["costs"]
        zero_turnover = turnover == 0
        assert (costs[zero_turnover] == 0).all()
