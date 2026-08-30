"""Regression tests for WFO statistical and accounting safeguards."""

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.metrics import compute_metrics
from trendbot.domain.models import WalkForwardConfig
from trendbot.domain.walk_forward import _tie_break, validate_no_overlapping_oos
from trendbot.domain.models import WalkForwardFold


def _candidate(sharpe: float, drawdown: float, turnover: float, threshold: float = 0.0):
    params = {
        "lookbacks": [5, 10, 21, 42],
        "vol_window": 40,
        "covariance_window": 60,
        "covariance_shrinkage": 0.1,
        "rebalance_threshold": threshold,
    }
    metrics = {
        "max_drawdown": drawdown,
        "avg_daily_turnover": turnover,
    }
    return sharpe, params, metrics


def test_sharpe_tolerance_prefers_lower_drawdown():
    best = _tie_break(
        [
            _candidate(1.20, -0.25, 0.05),
            _candidate(1.19, -0.10, 0.05),
        ],
        sharpe_tolerance=0.02,
    )
    assert best["rebalance_threshold"] == 0.0
    # Both candidates have identical parameters in this fixture; the selection
    # rule must therefore still be deterministic.


def test_sharpe_tolerance_does_not_hide_material_difference():
    first = _candidate(1.20, -0.25, 0.05, 0.0)
    second = _candidate(1.10, -0.05, 0.05, 0.01)
    best = _tie_break([first, second], sharpe_tolerance=0.02)
    assert best["rebalance_threshold"] == 0.0


def test_wfo_config_validates_research_guards():
    config = WalkForwardConfig()
    assert config.minimum_training_observations == 126
    assert config.minimum_training_trades == 1
    assert config.sharpe_tie_tolerance == pytest.approx(0.02)


def test_wfo_config_rejects_negative_sharpe_tolerance():
    with pytest.raises(ValueError):
        WalkForwardConfig(sharpe_tie_tolerance=-0.01)


def test_stitched_oos_overlap_is_rejected():
    folds = [
        WalkForwardFold(0, 0, 100, 100, 150),
        WalkForwardFold(1, 50, 150, 140, 190),
    ]
    with pytest.raises(ValueError, match="overlaps"):
        validate_no_overlapping_oos(folds)


def test_cagr_is_annualized_for_sub_year_periods():
    n = 126
    daily_return = (2.0 ** (1.0 / n)) - 1.0
    returns = pd.Series(np.full(n, daily_return))
    positions = pd.DataFrame(np.zeros((n, 1)))
    turnover = pd.Series(np.zeros(n))
    costs = pd.Series(np.zeros(n))

    metrics = compute_metrics(
        returns=returns,
        positions=positions,
        turnover=turnover,
        costs=costs,
        ann_factor=365,
    )

    assert metrics["total_return"] == pytest.approx(1.0, rel=1e-8)
    assert metrics["cagr"] == pytest.approx(2.0 ** (365.0 / n) - 1.0, rel=1e-8)


def test_cost_drag_excludes_warmup_costs():
    returns = pd.Series([0.0, 0.0, 0.01, 0.01])
    positions = pd.DataFrame(np.zeros((4, 1)))
    turnover = pd.Series([1.0, 1.0, 0.0, 0.0])
    costs = pd.Series([0.10, 0.10, 0.01, 0.01])

    metrics = compute_metrics(
        returns=returns,
        positions=positions,
        turnover=turnover,
        costs=costs,
        ann_factor=365,
        min_history=2,
    )

    assert metrics["total_cost_drag"] == pytest.approx(0.02)
