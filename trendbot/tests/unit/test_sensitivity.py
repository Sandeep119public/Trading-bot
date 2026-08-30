"""Unit tests for parameter sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.sensitivity import (
    _scale_lookbacks,
    _format_lookback_label,
    compute_plateau_metrics,
    run_sensitivity_analysis,
)


@pytest.fixture
def sample_close():
    """Generate sample price data for testing."""
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "A": 100 + np.cumsum(np.random.randn(300) * 2),
            "B": 50 + np.cumsum(np.random.randn(300) * 1),
        },
        index=dates,
    )


def test_scale_lookbacks():
    assert _scale_lookbacks([5, 10, 21], 1.0) == [5, 10, 21]
    assert _scale_lookbacks([5, 10, 21], 2.0) == [10, 20, 42]
    assert _scale_lookbacks([5, 10, 21], 0.5) == [2, 5, 10]
    assert _scale_lookbacks([2, 4], 0.1) == [2, 2]


def test_format_lookback_label():
    assert _format_lookback_label([5, 10, 21]) == "[5,10,21]"
    assert _format_lookback_label([10]) == "[10]"


def test_run_sensitivity_analysis_returns_dataframe(sample_close):
    results = run_sensitivity_analysis(
        close=sample_close,
        base_lookbacks=[5, 10, 21],
        base_vol_window=21,
        base_target_vol=0.10,
        base_ann_factor=365,
        base_max_leverage=1.0,
        base_taker_fee_pct=0.001,
        base_slippage_pct=0.0005,
        base_rebalance_threshold=0.01,
        base_min_history=30,
        allow_short=True,
        lookback_multipliers=[0.5, 1.0],
        vol_windows=[14, 21],
        target_vols=[0.05, 0.10],
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 8  # 2 x 2 x 2
    assert "sharpe_ratio" in results.columns
    assert "cagr" in results.columns
    assert "lookback_config" in results.columns
    assert "vol_window" in results.columns
    assert "target_vol" in results.columns


def test_plateau_metrics_robust():
    df = pd.DataFrame({
        "lookback_config": ["[5,10]", "[5,10]", "[5,10]", "[5,10]"],
        "vol_window": [14, 21, 14, 21],
        "target_vol": [0.05, 0.05, 0.10, 0.10],
        "sharpe_ratio": [1.0, 0.95, 0.9, 0.85],
        "cagr": [0.1, 0.09, 0.08, 0.07],
    })
    result = compute_plateau_metrics(df)
    assert result["plateau_found"] is True
    assert result["plateau_pct"] == 100.0


def test_plateau_metrics_overfitted():
    df = pd.DataFrame({
        "lookback_config": ["[5,10]", "[5,10]", "[5,10]", "[5,10]", "[5,10]"],
        "vol_window": [14, 21, 30, 14, 21],
        "target_vol": [0.05, 0.05, 0.05, 0.10, 0.10],
        "sharpe_ratio": [1.0, 0.5, 0.3, 0.2, 0.1],
        "cagr": [0.1, 0.05, 0.03, 0.02, 0.01],
    })
    result = compute_plateau_metrics(df)
    assert result["plateau_found"] is False
    assert result["plateau_pct"] == 20.0


def test_plateau_metrics_empty():
    df = pd.DataFrame()
    result = compute_plateau_metrics(df)
    assert result["plateau_found"] is False
