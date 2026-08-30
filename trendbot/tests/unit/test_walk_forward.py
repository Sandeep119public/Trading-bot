"""Comprehensive test suite for walk-forward out-of-sample validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.models import (
    FoldResult,
    ParameterGrid,
    WalkForwardConfig,
    WalkForwardFold,
)
from trendbot.domain.walk_forward import (
    compute_parameter_stability,
    enumerate_parameter_combinations,
    generate_folds,
    run_oos_fold,
    run_walk_forward,
    select_parameters,
    stitch_oos_returns,
    validate_no_overlapping_oos,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Default walk-forward configuration."""
    return WalkForwardConfig(
        train_window=80,
        test_window=30,
        step=30,
        minimum_training_bars=60,
        minimum_training_observations=0,
        minimum_training_trades=0,
        ann_factor=365,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        allow_short=True,
        min_history=30,
    )


@pytest.fixture
def small_grid():
    """Small parameter grid for fast testing."""
    return ParameterGrid(
        lookbacks=[[5], [10]],
        vol_window=[21],
        covariance_window=[60],
        covariance_shrinkage=[0.1],
        rebalance_threshold=[0.01],
    )


@pytest.fixture
def large_grid():
    """Larger parameter grid for combinatorial tests."""
    return ParameterGrid(
        lookbacks=[[5], [10], [20]],
        vol_window=[10, 21],
        covariance_window=[40, 60],
        covariance_shrinkage=[0.0, 0.1],
        rebalance_threshold=[0.0, 0.01],
    )


@pytest.fixture
def synthetic_close():
    """Synthetic close prices for testing (500 bars)."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    a_ret = np.random.normal(0.0005, 0.01, n)
    b_ret = np.random.normal(0.0003, 0.015, n)
    close = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + a_ret),
            "B": 50 * np.cumprod(1 + b_ret),
        },
        index=dates,
    )
    return close


@pytest.fixture
def synthetic_close_short():
    """Synthetic close prices with fewer bars (150)."""
    np.random.seed(123)
    n = 150
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    a_ret = np.random.normal(0.001, 0.01, n)
    b_ret = np.random.normal(-0.0005, 0.02, n)
    close = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + a_ret),
            "B": 50 * np.cumprod(1 + b_ret),
        },
        index=dates,
    )
    return close


# ---------------------------------------------------------------------------
# 1. Fold generation and validation
# ---------------------------------------------------------------------------


class TestFoldGeneration:
    """Test fold generation from config."""

    def test_generate_folds_basic(self, default_config, synthetic_close):
        folds = generate_folds(len(synthetic_close), default_config)
        assert len(folds) > 0
        for f in folds:
            assert isinstance(f, WalkForwardFold)
            assert (
                f.train_end_idx - f.train_start_idx
                == default_config.train_window
            )
            assert (
                f.test_end_idx - f.test_start_idx
                == default_config.test_window
            )

    def test_fold_indices_are_contiguous(self, default_config, synthetic_close):
        folds = generate_folds(len(synthetic_close), default_config)
        for f in folds:
            assert f.test_start_idx == f.train_end_idx

    def test_fold_indices_within_bounds(
        self, default_config, synthetic_close,
    ):
        folds = generate_folds(len(synthetic_close), default_config)
        for f in folds:
            assert f.train_start_idx >= 0
            assert f.test_end_idx <= len(synthetic_close)

    def test_no_overlapping_folds(self, default_config, synthetic_close):
        folds = generate_folds(len(synthetic_close), default_config)
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                assert (
                    folds[i].test_end_idx <= folds[j].test_start_idx
                    or folds[j].test_end_idx <= folds[i].test_start_idx
                )

    def test_minimum_training_bars(self, default_config, synthetic_close):
        folds = generate_folds(len(synthetic_close), default_config)
        for f in folds:
            train_len = f.train_end_idx - f.train_start_idx
            assert train_len >= default_config.minimum_training_bars

    def test_insufficient_data_raises(self):
        config = WalkForwardConfig(
            train_window=100,
            test_window=50,
            step=50,
            minimum_training_bars=100,
        )
        with pytest.raises(ValueError, match="too short"):
            generate_folds(50, config)

    def test_single_fold(self):
        config = WalkForwardConfig(
            train_window=100,
            test_window=50,
            step=50,
            minimum_training_bars=100,
        )
        # start=0: 0+100+50=150 <= 150 → fold; start=50: 200 > 150 → stop
        folds = generate_folds(150, config)
        assert len(folds) == 1

    def test_multiple_folds(self):
        config = WalkForwardConfig(
            train_window=100,
            test_window=50,
            step=50,
            minimum_training_bars=100,
        )
        # start goes 0,50,...,350 (each +150 <= 500) → 8 folds
        folds = generate_folds(500, config)
        assert len(folds) == 8

    def test_fold_index_sequential(self, default_config, synthetic_close):
        folds = generate_folds(len(synthetic_close), default_config)
        for i, f in enumerate(folds):
            assert f.fold_index == i


# ---------------------------------------------------------------------------
# 2. OOS overlap validation
# ---------------------------------------------------------------------------


class TestOOSOverlapValidation:
    """Test that overlapping OOS periods are detected."""

    def test_no_overlap_passes(self):
        folds = [
            WalkForwardFold(
                fold_index=0, train_start_idx=0, train_end_idx=100,
                test_start_idx=100, test_end_idx=150,
            ),
            WalkForwardFold(
                fold_index=1, train_start_idx=50, train_end_idx=150,
                test_start_idx=150, test_end_idx=200,
            ),
        ]
        validate_no_overlapping_oos(folds)

    def test_overlap_detected(self):
        folds = [
            WalkForwardFold(
                fold_index=0, train_start_idx=0, train_end_idx=100,
                test_start_idx=100, test_end_idx=160,
            ),
            WalkForwardFold(
                fold_index=1, train_start_idx=50, train_end_idx=150,
                test_start_idx=150, test_end_idx=200,
            ),
        ]
        with pytest.raises(ValueError, match="overlaps"):
            validate_no_overlapping_oos(folds)

    def test_adjacent_oos_passes(self):
        folds = [
            WalkForwardFold(
                fold_index=0, train_start_idx=0, train_end_idx=100,
                test_start_idx=100, test_end_idx=150,
            ),
            WalkForwardFold(
                fold_index=1, train_start_idx=50, train_end_idx=150,
                test_start_idx=150, test_end_idx=200,
            ),
        ]
        validate_no_overlapping_oos(folds)


# ---------------------------------------------------------------------------
# 3. Parameter enumeration
# ---------------------------------------------------------------------------


class TestParameterEnumeration:
    """Test parameter grid enumeration."""

    def test_single_combination(self):
        grid = ParameterGrid(
            lookbacks=[[5]],
            vol_window=[21],
            covariance_window=[60],
            covariance_shrinkage=[0.1],
            rebalance_threshold=[0.01],
        )
        combos = enumerate_parameter_combinations(grid)
        assert len(combos) == 1
        assert combos[0]["lookbacks"] == [5]
        assert combos[0]["vol_window"] == 21

    def test_multiple_combinations(self):
        grid = ParameterGrid(
            lookbacks=[[5], [10]],
            vol_window=[21],
            covariance_window=[60],
            covariance_shrinkage=[0.1],
            rebalance_threshold=[0.01],
        )
        combos = enumerate_parameter_combinations(grid)
        assert len(combos) == 2

    def test_full_cartesian_product(self, large_grid):
        combos = enumerate_parameter_combinations(large_grid)
        expected = 3 * 2 * 2 * 2 * 2
        assert len(combos) == expected

    def test_combinations_have_all_keys(self, large_grid):
        combos = enumerate_parameter_combinations(large_grid)
        for combo in combos:
            assert "lookbacks" in combo
            assert "vol_window" in combo
            assert "covariance_window" in combo
            assert "covariance_shrinkage" in combo
            assert "rebalance_threshold" in combo

    def test_lookbacks_are_lists(self, large_grid):
        combos = enumerate_parameter_combinations(large_grid)
        for combo in combos:
            assert isinstance(combo["lookbacks"], list)


# ---------------------------------------------------------------------------
# 4. Parameter selection
# ---------------------------------------------------------------------------


class TestParameterSelection:
    """Test parameter selection on training data."""

    def test_select_parameters_returns_valid(
        self, synthetic_close_short, small_grid, default_config,
    ):
        train_close = synthetic_close_short.iloc[:80]
        params, sharpe, metrics = select_parameters(
            train_close, small_grid, default_config,
        )
        assert isinstance(params, dict)
        assert "lookbacks" in params
        assert isinstance(sharpe, float)
        assert isinstance(metrics, dict)

    def test_select_parameters_returns_best(
        self, synthetic_close_short, small_grid, default_config,
    ):
        train_close = synthetic_close_short.iloc[:80]
        params, sharpe, _ = select_parameters(
            train_close, small_grid, default_config,
        )
        assert np.isfinite(sharpe)

    def test_select_parameters_no_invalid(
        self, synthetic_close_short, small_grid, default_config,
    ):
        train_close = synthetic_close_short.iloc[:80]
        params, sharpe, metrics = select_parameters(
            train_close, small_grid, default_config,
        )
        assert np.isfinite(sharpe)
        assert np.isfinite(metrics["total_return"])

    def test_select_parameters_different_grids(
        self, synthetic_close_short, default_config,
    ):
        train_close = synthetic_close_short.iloc[:80]
        grid1 = ParameterGrid(
            lookbacks=[[5]], vol_window=[21], covariance_window=[60],
            covariance_shrinkage=[0.1], rebalance_threshold=[0.01],
        )
        grid2 = ParameterGrid(
            lookbacks=[[10]], vol_window=[21], covariance_window=[60],
            covariance_shrinkage=[0.1], rebalance_threshold=[0.01],
        )
        params1, _, _ = select_parameters(
            train_close, grid1, default_config,
        )
        params2, _, _ = select_parameters(
            train_close, grid2, default_config,
        )
        assert params1["lookbacks"] == [5]
        assert params2["lookbacks"] == [10]


# ---------------------------------------------------------------------------
# 5. OOS fold execution
# ---------------------------------------------------------------------------


class TestOOSFoldExecution:
    """Test OOS fold execution with frozen parameters."""

    def test_oos_fold_returns_fold_result(
        self, synthetic_close_short, small_grid, default_config,
    ):
        train_close = synthetic_close_short.iloc[:80]
        params, _, _ = select_parameters(
            train_close, small_grid, default_config,
        )
        fold = WalkForwardFold(
            fold_index=0,
            train_start_idx=0,
            train_end_idx=80,
            test_start_idx=80,
            test_end_idx=110,
        )
        full_close = synthetic_close_short.iloc[:110]
        result = run_oos_fold(
            full_close, fold, params, default_config,
        )
        assert isinstance(result, FoldResult)
        assert len(result.oos_returns) == 30

    def test_oos_fold_preserves_params(
        self, synthetic_close_short, small_grid, default_config,
    ):
        train_close = synthetic_close_short.iloc[:80]
        params, _, _ = select_parameters(
            train_close, small_grid, default_config,
        )
        fold = WalkForwardFold(
            fold_index=0,
            train_start_idx=0,
            train_end_idx=80,
            test_start_idx=80,
            test_end_idx=110,
        )
        full_close = synthetic_close_short.iloc[:110]
        result = run_oos_fold(
            full_close, fold, params, default_config,
        )
        assert result.selected_parameters == params

    def test_oos_fold_metrics_valid(
        self, synthetic_close_short, small_grid, default_config,
    ):
        train_close = synthetic_close_short.iloc[:80]
        params, _, _ = select_parameters(
            train_close, small_grid, default_config,
        )
        fold = WalkForwardFold(
            fold_index=0,
            train_start_idx=0,
            train_end_idx=80,
            test_start_idx=80,
            test_end_idx=110,
        )
        full_close = synthetic_close_short.iloc[:110]
        result = run_oos_fold(
            full_close, fold, params, default_config,
        )
        assert np.isfinite(result.oos_metrics["total_return"])
        assert np.isfinite(result.oos_metrics["sharpe_ratio"])

    def test_oos_returns_are_frozen(
        self, synthetic_close_short, small_grid, default_config,
    ):
        """OOS returns with same params should be deterministic."""
        train_close = synthetic_close_short.iloc[:80]
        params, _, _ = select_parameters(
            train_close, small_grid, default_config,
        )
        fold = WalkForwardFold(
            fold_index=0,
            train_start_idx=0,
            train_end_idx=80,
            test_start_idx=80,
            test_end_idx=110,
        )
        full_close = synthetic_close_short.iloc[:110]
        result1 = run_oos_fold(
            full_close, fold, params, default_config,
        )
        result2 = run_oos_fold(
            full_close, fold, params, default_config,
        )
        pd.testing.assert_series_equal(
            result1.oos_returns, result2.oos_returns,
        )


# ---------------------------------------------------------------------------
# 6. Return stitching
# ---------------------------------------------------------------------------


class TestReturnStitching:
    """Test stitching of OOS returns."""

    def test_stitch_empty(self):
        returns, equity = stitch_oos_returns([])
        assert len(returns) == 0
        assert len(equity) == 0

    def test_stitch_single_fold(
        self, synthetic_close_short, small_grid, default_config,
    ):
        train_close = synthetic_close_short.iloc[:80]
        params, _, _ = select_parameters(
            train_close, small_grid, default_config,
        )
        fold = WalkForwardFold(
            fold_index=0, train_start_idx=0, train_end_idx=80,
            test_start_idx=80, test_end_idx=110,
        )
        full_close = synthetic_close_short.iloc[:110]
        result = run_oos_fold(
            full_close, fold, params, default_config,
        )
        stitched, _equity = stitch_oos_returns([result])
        pd.testing.assert_series_equal(
            stitched, result.oos_returns,
        )

    def test_stitch_preserves_order(
        self, synthetic_close_short, small_grid, default_config,
    ):
        """Stitched returns must be in fold order."""
        train_close = synthetic_close_short.iloc[:80]
        params, _, _ = select_parameters(
            train_close, small_grid, default_config,
        )

        results = []
        for i in range(2):
            fold = WalkForwardFold(
                fold_index=i,
                train_start_idx=i * 40,
                train_end_idx=i * 40 + 80,
                test_start_idx=i * 40 + 80,
                test_end_idx=i * 40 + 110,
            )
            full_close = synthetic_close_short.iloc[
                i * 40 : i * 40 + 110
            ]
            results.append(
                run_oos_fold(
                    full_close, fold, params, default_config,
                )
            )

        stitched, _equity = stitch_oos_returns(results)
        expected_idx = pd.concat(
            [r.oos_returns for r in results],
        ).index
        pd.testing.assert_index_equal(
            stitched.index, expected_idx,
        )

    def test_stitch_equity_curve(self):
        """Equity curve must be cumulative product of (1 + returns)."""
        r1 = pd.Series(
            [0.01, 0.02, -0.01],
            index=pd.date_range("2023-01-01", periods=3),
        )
        r2 = pd.Series(
            [0.005, 0.01],
            index=pd.date_range("2023-01-04", periods=2),
        )
        fake_result1 = FoldResult(
            fold=WalkForwardFold(0, 0, 10, 10, 13),
            selected_parameters={"lookbacks": [5]},
            oos_returns=r1,
            oos_equity=(1 + r1).cumprod(),
            oos_gross_returns=r1.copy(),
            oos_turnover=pd.Series([0.0] * 3),
            oos_costs=pd.Series([0.0] * 3),
            oos_positions=pd.DataFrame({"A": [0.5] * 3}),
            training_sharpe=0.5,
            training_metrics={},
            oos_metrics={},
        )
        fake_result2 = FoldResult(
            fold=WalkForwardFold(1, 5, 15, 15, 18),
            selected_parameters={"lookbacks": [5]},
            oos_returns=r2,
            oos_equity=(1 + r2).cumprod(),
            oos_gross_returns=r2.copy(),
            oos_turnover=pd.Series([0.0] * 2),
            oos_costs=pd.Series([0.0] * 2),
            oos_positions=pd.DataFrame({"A": [0.5] * 2}),
            training_sharpe=0.6,
            training_metrics={},
            oos_metrics={},
        )
        stitched, equity = stitch_oos_returns(
            [fake_result1, fake_result2],
        )
        expected = (1 + stitched).cumprod()
        pd.testing.assert_series_equal(equity, expected)


# ---------------------------------------------------------------------------
# 7. Parameter stability
# ---------------------------------------------------------------------------


class TestParameterStability:
    """Test parameter stability computation."""

    def test_stability_empty(self):
        assert compute_parameter_stability([]) == {}

    def test_stability_single_param(self):
        fr1 = FoldResult(
            fold=WalkForwardFold(0, 0, 10, 10, 20),
            selected_parameters={"lookbacks": [5], "vol_window": 21},
            oos_returns=pd.Series([0.01]),
            oos_equity=pd.Series([1.01]),
            oos_gross_returns=pd.Series([0.01]),
            oos_turnover=pd.Series([0.0]),
            oos_costs=pd.Series([0.0]),
            oos_positions=pd.DataFrame(),
            training_sharpe=0.5,
            training_metrics={},
            oos_metrics={},
        )
        fr2 = FoldResult(
            fold=WalkForwardFold(1, 5, 15, 15, 25),
            selected_parameters={"lookbacks": [10], "vol_window": 21},
            oos_returns=pd.Series([0.02]),
            oos_equity=pd.Series([1.02]),
            oos_gross_returns=pd.Series([0.02]),
            oos_turnover=pd.Series([0.0]),
            oos_costs=pd.Series([0.0]),
            oos_positions=pd.DataFrame(),
            training_sharpe=0.6,
            training_metrics={},
            oos_metrics={},
        )
        stability = compute_parameter_stability([fr1, fr2])
        # lists are converted to tuples for hashability
        assert stability["lookbacks"][(5,)] == 1
        assert stability["lookbacks"][(10,)] == 1
        assert stability["vol_window"][21] == 2

    def test_stability_multiple_params(self):
        fr = FoldResult(
            fold=WalkForwardFold(0, 0, 10, 10, 20),
            selected_parameters={
                "lookbacks": [5],
                "vol_window": 21,
                "covariance_window": 60,
            },
            oos_returns=pd.Series([0.01]),
            oos_equity=pd.Series([1.01]),
            oos_gross_returns=pd.Series([0.01]),
            oos_turnover=pd.Series([0.0]),
            oos_costs=pd.Series([0.0]),
            oos_positions=pd.DataFrame(),
            training_sharpe=0.5,
            training_metrics={},
            oos_metrics={},
        )
        stability = compute_parameter_stability([fr])
        assert len(stability) == 3


# ---------------------------------------------------------------------------
# 8. Walk-forward integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    """Integration tests for the complete walk-forward engine."""

    def test_run_walk_forward_returns_report(
        self, synthetic_close_short, small_grid, default_config,
    ):
        report = run_walk_forward(
            synthetic_close_short, small_grid, default_config,
        )
        assert hasattr(report, "folds")
        assert hasattr(report, "stitched_oos_returns")
        assert hasattr(report, "stitched_oos_equity")
        assert hasattr(report, "aggregate_metrics")
        assert hasattr(report, "parameter_stability")
        assert hasattr(report, "per_fold_summary")

    def test_run_walk_forward_has_folds(
        self, synthetic_close_short, small_grid, default_config,
    ):
        report = run_walk_forward(
            synthetic_close_short, small_grid, default_config,
        )
        assert len(report.folds) >= 1

    def test_run_walk_forward_oos_returns_not_empty(
        self, synthetic_close_short, small_grid, default_config,
    ):
        report = run_walk_forward(
            synthetic_close_short, small_grid, default_config,
        )
        assert len(report.stitched_oos_returns) > 0

    def test_run_walk_forward_aggregate_metrics_valid(
        self, synthetic_close_short, small_grid, default_config,
    ):
        report = run_walk_forward(
            synthetic_close_short, small_grid, default_config,
        )
        assert np.isfinite(report.aggregate_metrics["total_return"])
        assert np.isfinite(report.aggregate_metrics["sharpe_ratio"])

    def test_run_walk_forward_per_fold_summary(
        self, synthetic_close_short, small_grid, default_config,
    ):
        report = run_walk_forward(
            synthetic_close_short, small_grid, default_config,
        )
        assert len(report.per_fold_summary) == len(report.folds)
        for fold in report.per_fold_summary:
            assert "fold" in fold
            assert "oos_sharpe" in fold
            assert "oos_total_return" in fold

    def test_run_walk_forward_equity_starts_near_one(
        self, synthetic_close_short, small_grid, default_config,
    ):
        report = run_walk_forward(
            synthetic_close_short, small_grid, default_config,
        )
        first_eq = report.stitched_oos_equity.iloc[0]
        assert 0.9 < first_eq < 1.1

    def test_run_walk_forward_causal(
        self, synthetic_close_short, small_grid, default_config,
    ):
        """Verify that future data cannot leak into parameter selection."""
        report = run_walk_forward(
            synthetic_close_short, small_grid, default_config,
        )
        for fr in report.folds:
            assert fr.fold.test_start_idx >= fr.fold.train_end_idx

    def test_run_walk_forward_different_grid_sizes(
        self, synthetic_close_short, default_config,
    ):
        """Different grid sizes should produce different fold results."""
        grid1 = ParameterGrid(
            lookbacks=[[5]], vol_window=[21],
            covariance_window=[60],
            covariance_shrinkage=[0.1],
            rebalance_threshold=[0.01],
        )
        grid2 = ParameterGrid(
            lookbacks=[[10]], vol_window=[21],
            covariance_window=[60],
            covariance_shrinkage=[0.1],
            rebalance_threshold=[0.01],
        )
        report1 = run_walk_forward(
            synthetic_close_short, grid1, default_config,
        )
        report2 = run_walk_forward(
            synthetic_close_short, grid2, default_config,
        )
        assert report1 is not report2


# ---------------------------------------------------------------------------
# 9. Edge cases and robustness
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_minimum_data_for_one_fold(self):
        config = WalkForwardConfig(
            train_window=100,
            test_window=50,
            step=50,
            minimum_training_bars=100,
        )
        folds = generate_folds(150, config)
        assert len(folds) == 1

    def test_step_equal_to_test_window_no_overlap(self):
        config = WalkForwardConfig(
            train_window=100,
            test_window=50,
            step=50,
            minimum_training_bars=100,
        )
        folds = generate_folds(500, config)
        validate_no_overlapping_oos(folds)

    def test_large_lookbacks(self, synthetic_close_short, default_config):
        """Large lookbacks should still work with sufficient data."""
        grid = ParameterGrid(
            lookbacks=[[20, 50]],
            vol_window=[21],
            covariance_window=[60],
            covariance_shrinkage=[0.1],
            rebalance_threshold=[0.01],
        )
        train_close = synthetic_close_short.iloc[:80]
        params, sharpe, _ = select_parameters(
            train_close, grid, default_config,
        )
        assert np.isfinite(sharpe)

    def test_single_asset(self):
        """Single asset should work."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        close = pd.DataFrame(
            {
                "A": 100 * np.cumprod(
                    1 + np.random.normal(0.001, 0.01, n),
                ),
            },
            index=dates,
        )
        config = WalkForwardConfig(
            train_window=80,
            test_window=30,
            step=30,
            minimum_training_bars=60,
            ann_factor=365,
        )
        folds = generate_folds(len(close), config)
        assert len(folds) > 0

    def test_zero_returns(self):
        """Zero returns should not crash."""
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        close = pd.DataFrame(
            {"A": np.full(n, 100.0), "B": np.full(n, 50.0)},
            index=dates,
        )
        config = WalkForwardConfig(
            train_window=80,
            test_window=30,
            step=30,
            minimum_training_bars=60,
            ann_factor=365,
        )
        folds = generate_folds(len(close), config)
        assert len(folds) > 0
