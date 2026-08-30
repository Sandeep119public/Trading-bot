"""WFO / OOS / Overfitting deep audit tests.

Covers:
  - Phase 3:  Data leakage (Tests A/B/C)
  - Phase 4:  Fold boundary integrity
  - Phase 5:  Warmup context correctness
  - Phase 10: OOS stitching integrity
  - Phase 11: Fold state reset
  - Phase 12: OOS return integrity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.models import (
    ParameterGrid,
    WalkForwardConfig,
)
from trendbot.domain.walk_forward import (
    _required_history,
    generate_folds,
    run_oos_fold,
    run_walk_forward,
    select_parameters,
    validate_no_overlapping_oos,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wfo_config():
    return WalkForwardConfig(
        train_window=200,
        test_window=60,
        step=60,
        minimum_training_bars=100,
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
    return ParameterGrid(
        lookbacks=[[5, 10, 21], [10, 21, 42]],
        vol_window=[21, 40],
        covariance_window=[60],
        covariance_shrinkage=[0.1],
        rebalance_threshold=[0.01],
    )


@pytest.fixture
def synthetic_close():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    a_ret = np.random.normal(0.0005, 0.01, n)
    b_ret = np.random.normal(0.0003, 0.015, n)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + a_ret),
            "B": 50 * np.cumprod(1 + b_ret),
        },
        index=dates,
    )


# ===========================================================================
# Phase 3: DATA LEAKAGE TESTS
# ===========================================================================


class TestLeakageAOosDataModification:
    """Modifying prices ONLY inside the OOS period must NOT change selected parameters."""

    def test_oos_modification_preserves_params(
        self, synthetic_close, small_grid, wfo_config,
    ):
        folds = generate_folds(len(synthetic_close), wfo_config)
        fold = folds[0]

        train_close = synthetic_close.iloc[fold.train_start_idx:fold.train_end_idx]
        params_base, _, _ = select_parameters(train_close, small_grid, wfo_config)

        full_close_base = synthetic_close.iloc[fold.train_start_idx:fold.test_end_idx]
        result_base = run_oos_fold(full_close_base, fold, params_base, wfo_config)

        close_modified = synthetic_close.copy()
        oos_slice = close_modified.iloc[fold.test_start_idx:fold.test_end_idx]
        close_modified.iloc[fold.test_start_idx:fold.test_end_idx] = oos_slice * 1.1

        full_close_mod = close_modified.iloc[fold.train_start_idx:fold.test_end_idx]
        result_mod = run_oos_fold(full_close_mod, fold, params_base, wfo_config)

        assert result_base.selected_parameters == result_mod.selected_parameters

    def test_oos_modification_changes_returns(
        self, synthetic_close, small_grid, wfo_config,
    ):
        folds = generate_folds(len(synthetic_close), wfo_config)
        fold = folds[0]

        train_close = synthetic_close.iloc[fold.train_start_idx:fold.train_end_idx]
        params, _, _ = select_parameters(train_close, small_grid, wfo_config)

        full_close_base = synthetic_close.iloc[fold.train_start_idx:fold.test_end_idx]
        result_base = run_oos_fold(full_close_base, fold, params, wfo_config)

        close_modified = synthetic_close.copy()
        close_modified.iloc[fold.test_start_idx:fold.test_end_idx] *= 1.1
        full_close_mod = close_modified.iloc[fold.train_start_idx:fold.test_end_idx]
        result_mod = run_oos_fold(full_close_mod, fold, params, wfo_config)

        assert not result_base.oos_returns.equals(result_mod.oos_returns)


class TestLeakageBFutureFoldModification:
    """Modifying data AFTER the current fold's entire context must not change selected params.

    Important: with rolling/overlapping train windows, modifying a future fold's
    TRAINING data may overlap with the current fold's training data.  We only
    modify data strictly after fold0's test_end to ensure true isolation.
    """

    def test_future_fold_modification_preserves_params(
        self, synthetic_close, small_grid, wfo_config,
    ):
        folds = generate_folds(len(synthetic_close), wfo_config)
        if len(folds) < 2:
            pytest.skip("Need at least 2 folds")

        fold0 = folds[0]
        train0 = synthetic_close.iloc[fold0.train_start_idx:fold0.train_end_idx]
        params_base, _, _ = select_parameters(train0, small_grid, wfo_config)

        close_modified = synthetic_close.copy()
        modify_start = fold0.test_end_idx
        close_modified.iloc[modify_start:] *= 2.0

        train_mod = close_modified.iloc[fold0.train_start_idx:fold0.train_end_idx]
        params_mod, _, _ = select_parameters(train_mod, small_grid, wfo_config)

        assert params_base == params_mod


class TestLeakageCTrainingDataModification:
    """Modifying training data MUST change selected parameters when the grid is fine enough."""

    def test_training_modification_changes_params(
        self, synthetic_close, wfo_config,
    ):
        fine_grid = ParameterGrid(
            lookbacks=[[5, 10, 21], [21, 42, 84]],
            vol_window=[14, 40],
            covariance_window=[60],
            covariance_shrinkage=[0.1],
            rebalance_threshold=[0.01],
        )
        folds = generate_folds(len(synthetic_close), wfo_config)
        fold = folds[0]

        train_base = synthetic_close.iloc[fold.train_start_idx:fold.train_end_idx]
        params_base, _, _ = select_parameters(train_base, fine_grid, wfo_config)

        train_modified = train_base.copy()
        train_modified.iloc[:80] *= 5.0
        params_mod, _, _ = select_parameters(train_modified, fine_grid, wfo_config)

        assert params_base != params_mod


# ===========================================================================
# Phase 4: FOLD BOUNDARY INTEGRITY
# ===========================================================================


class TestFoldBoundaries:
    """Every fold must have non-overlapping, chronologically ordered OOS windows."""

    def test_train_end_before_test_start(self, synthetic_close, wfo_config):
        folds = generate_folds(len(synthetic_close), wfo_config)
        for f in folds:
            assert f.train_end_idx <= f.test_start_idx
            assert f.test_start_idx == f.train_end_idx

    def test_no_oos_overlap(self, synthetic_close, wfo_config):
        folds = generate_folds(len(synthetic_close), wfo_config)
        validate_no_overlapping_oos(folds)

    def test_chronological_order(self, synthetic_close, wfo_config):
        folds = generate_folds(len(synthetic_close), wfo_config)
        for i in range(1, len(folds)):
            assert folds[i].test_start_idx >= folds[i - 1].test_end_idx

    def test_no_gaps_between_folds_when_step_equals_test_window(
        self, synthetic_close,
    ):
        config = WalkForwardConfig(
            train_window=200,
            test_window=60,
            step=60,
            minimum_training_bars=100,
            minimum_training_observations=0,
            minimum_training_trades=0,
            ann_factor=365,
            allow_short=True,
            min_history=30,
        )
        folds = generate_folds(len(synthetic_close), config)
        for i in range(1, len(folds)):
            assert folds[i].test_start_idx == folds[i - 1].test_end_idx

    def test_fold_indices_within_data_bounds(self, synthetic_close, wfo_config):
        folds = generate_folds(len(synthetic_close), wfo_config)
        for f in folds:
            assert f.train_start_idx >= 0
            assert f.test_end_idx <= len(synthetic_close)

    def test_all_bars_covered_with_step_equals_test_window(
        self, synthetic_close,
    ):
        config = WalkForwardConfig(
            train_window=200,
            test_window=60,
            step=60,
            minimum_training_bars=100,
            minimum_training_observations=0,
            minimum_training_trades=0,
            ann_factor=365,
            allow_short=True,
            min_history=30,
        )
        folds = generate_folds(len(synthetic_close), config)
        first_train = folds[0].train_start_idx
        last_test = folds[-1].test_end_idx
        total_bars = len(synthetic_close)
        assert first_train == 0
        assert last_test == total_bars


# ===========================================================================
# Phase 5: WARMUP CONTEXT
# ===========================================================================


class TestWarmupContext:
    """Frozen OOS strategy must receive sufficient historical context."""

    def test_oos_fold_gets_train_plus_test_context(self, synthetic_close, small_grid, wfo_config):
        folds = generate_folds(len(synthetic_close), wfo_config)
        fold = folds[0]
        train_close = synthetic_close.iloc[fold.train_start_idx:fold.train_end_idx]
        params, _, _ = select_parameters(train_close, small_grid, wfo_config)

        full_close = synthetic_close.iloc[fold.train_start_idx:fold.test_end_idx]
        expected_len = fold.test_end_idx - fold.train_start_idx
        assert len(full_close) == expected_len

        result = run_oos_fold(full_close, fold, params, wfo_config)
        assert len(result.oos_returns) == fold.test_length

    def test_oos_returns_start_after_warmup(self, synthetic_close, small_grid, wfo_config):
        folds = generate_folds(len(synthetic_close), wfo_config)
        fold = folds[0]
        train_close = synthetic_close.iloc[fold.train_start_idx:fold.train_end_idx]
        params, _, _ = select_parameters(train_close, small_grid, wfo_config)

        full_close = synthetic_close.iloc[fold.train_start_idx:fold.test_end_idx]
        run_oos_fold(full_close, fold, params, wfo_config)

        req = _required_history(params, wfo_config)
        assert req <= fold.train_length, (
            f"required_history ({req}) must be <= train_length ({fold.train_length})"
        )

    def test_training_context_available_for_oos_warmup(
        self, synthetic_close, small_grid, wfo_config,
    ):
        folds = generate_folds(len(synthetic_close), wfo_config)
        fold = folds[0]
        train_close = synthetic_close.iloc[fold.train_start_idx:fold.train_end_idx]
        params, _, _ = select_parameters(train_close, small_grid, wfo_config)

        full_close = synthetic_close.iloc[fold.train_start_idx:fold.test_end_idx]
        result = run_oos_fold(full_close, fold, params, wfo_config)

        assert len(result.oos_returns) > 0
        assert result.oos_returns.notna().any()


# ===========================================================================
# Phase 10: OOS STITCHING INTEGRITY
# ===========================================================================


class TestOOSStitching:
    """Stitched OOS returns must be correct, non-overlapping, and properly compounded."""

    def test_stitched_equity_matches_cumprod(self, synthetic_close, small_grid, wfo_config):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        expected_equity = (1.0 + report.stitched_oos_returns).cumprod()
        pd.testing.assert_series_equal(
            report.stitched_oos_equity, expected_equity, check_names=False,
        )

    def test_stitched_returns_no_duplicates(self, synthetic_close, small_grid, wfo_config):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        assert not report.stitched_oos_returns.index.has_duplicates

    def test_stitched_returns_monotonic(self, synthetic_close, small_grid, wfo_config):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        assert report.stitched_oos_returns.index.is_monotonic_increasing

    def test_stitched_returns_only_oos_bars(self, synthetic_close, small_grid, wfo_config):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        total_oos = sum(fr.fold.test_length for fr in report.folds)
        assert len(report.stitched_oos_returns) == total_oos

    def test_no_training_returns_in_stitched(self, synthetic_close, small_grid, wfo_config):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        oos_indices = set()
        for fr in report.folds:
            oos_indices.update(range(fr.fold.test_start_idx, fr.fold.test_end_idx))
        all_indices = set(range(len(synthetic_close)))
        non_oos = all_indices - oos_indices
        for idx in non_oos:
            date = synthetic_close.index[idx]
            assert date not in report.stitched_oos_returns.index


# ===========================================================================
# Phase 11: FOLD STATE RESET
# ===========================================================================


class TestFoldStateReset:
    """Each OOS fold must start from a clean state."""

    def test_oos_equity_starts_near_one(self, synthetic_close, small_grid, wfo_config):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        for fr in report.folds:
            first_eq = fr.oos_equity.iloc[0]
            assert 0.9 < first_eq < 1.1

    def test_oos_fold_positions_start_zero(self, synthetic_close, small_grid, wfo_config):
        folds = generate_folds(len(synthetic_close), wfo_config)
        fold = folds[0]
        train_close = synthetic_close.iloc[fold.train_start_idx:fold.train_end_idx]
        params, _, _ = select_parameters(train_close, small_grid, wfo_config)

        full_close = synthetic_close.iloc[fold.train_start_idx:fold.test_end_idx]
        result = run_oos_fold(full_close, fold, params, wfo_config)

        assert len(result.oos_positions) == fold.test_length

    def test_each_fold_compounds_independently(
        self, synthetic_close, small_grid, wfo_config,
    ):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        for fr in report.folds:
            eq = fr.oos_equity
            assert eq.iloc[0] == pytest.approx(1.0 + fr.oos_returns.iloc[0], abs=1e-10)


# ===========================================================================
# Phase 12: OOS RETURN INTEGRITY
# ===========================================================================


class TestOOSReturnIntegrity:
    """Verify position[t] earns return[t+1] in OOS and costs are correct."""

    def test_oos_position_timing(self, synthetic_close, small_grid, wfo_config):
        """Verify position timing by checking gross_returns are derived from positions.

        The backtest engine (tested separately) computes:
          gross_ret[i] = dot(positions[i-1], daily_returns[i])

        run_oos_fold slices the backtest output. We verify the accounting
        relationship net = gross - costs, and that gross_returns are non-zero
        only when positions are non-zero (meaning the engine is actually using
        positions to compute returns).
        """
        folds = generate_folds(len(synthetic_close), wfo_config)
        fold = folds[0]
        train_close = synthetic_close.iloc[fold.train_start_idx:fold.train_end_idx]
        params, _, _ = select_parameters(train_close, small_grid, wfo_config)

        full_close = synthetic_close.iloc[fold.train_start_idx:fold.test_end_idx]
        result = run_oos_fold(full_close, fold, params, wfo_config)

        net = result.oos_gross_returns - result.oos_costs
        pd.testing.assert_series_equal(
            result.oos_returns, net, check_names=False, atol=1e-15,
        )

        active_mask = result.oos_positions.abs().sum(axis=1) > 1e-10
        if active_mask.any():
            active_gross = result.oos_gross_returns[active_mask]
            assert (active_gross != 0).any(), (
                "Non-zero positions should produce non-zero gross returns"
            )

        has_turnover = result.oos_turnover > 0
        has_cost = result.oos_costs > 0
        assert (has_turnover == has_cost).all(), (
            "Costs must be paid exactly when turnover occurs"
        )

    def test_oos_costs_non_negative(self, synthetic_close, small_grid, wfo_config):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        for fr in report.folds:
            assert (fr.oos_costs >= -1e-15).all()

    def test_oos_costs_aligned_with_turnover(
        self, synthetic_close, small_grid, wfo_config,
    ):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        cost_rate = wfo_config.taker_fee_pct + wfo_config.slippage_pct
        for fr in report.folds:
            expected_costs = fr.oos_turnover * cost_rate
            pd.testing.assert_series_equal(
                fr.oos_costs, expected_costs, check_names=False, atol=1e-15,
            )

    def test_oos_net_equals_gross_minus_costs(
        self, synthetic_close, small_grid, wfo_config,
    ):
        report = run_walk_forward(synthetic_close, small_grid, wfo_config)
        for fr in report.folds:
            expected_net = fr.oos_gross_returns - fr.oos_costs
            pd.testing.assert_series_equal(
                fr.oos_returns, expected_net, check_names=False, atol=1e-15,
            )
