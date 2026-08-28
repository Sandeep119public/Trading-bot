"""Unit tests for position sizing."""

import numpy as np
import pandas as pd
import pytest

from trendbot.domain.sizing import (
    apply_leverage_cap,
    apply_rebalance_threshold,
    compute_asset_volatility,
    compute_raw_weights,
)


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = pd.DataFrame(
        {
            "A": 100 + np.cumsum(np.random.randn(100) * 2),
            "B": 50 + np.cumsum(np.random.randn(100) * 1),
        },
        index=dates,
    )
    score = pd.DataFrame(
        {"A": 0.5, "B": -0.3},
        index=dates,
    )
    return close, score


def test_asset_volatility_shape(sample_data):
    close, _ = sample_data
    vol = compute_asset_volatility(close, vol_window=21, ann_factor=365)
    assert vol.shape == close.shape


def test_asset_volatility_positive(sample_data):
    close, _ = sample_data
    vol = compute_asset_volatility(close, vol_window=21, ann_factor=365)
    valid_vol = vol.dropna()
    assert (valid_vol >= 0).all().all()


def test_leverage_cap():
    weights = pd.DataFrame(
        {"A": [0.6, 1.2], "B": [0.5, 0.8]},
        index=[0, 1],
    )
    capped = apply_leverage_cap(weights, max_gross_leverage=1.0)
    gross = capped.abs().sum(axis=1)
    assert (gross <= 1.0 + 1e-10).all()


def test_leverage_cap_no_change():
    weights = pd.DataFrame(
        {"A": [0.3, 0.4], "B": [0.2, 0.3]},
        index=[0, 1],
    )
    capped = apply_leverage_cap(weights, max_gross_leverage=2.0)
    pd.testing.assert_frame_equal(weights, capped)


def test_rebalance_threshold_no_change():
    target = pd.DataFrame({"A": [0.5], "B": [0.3]})
    current = pd.DataFrame({"A": [0.5], "B": [0.3]})
    executed = apply_rebalance_threshold(target, current, threshold=0.01)
    pd.testing.assert_frame_equal(executed, current)


def test_rebalance_threshold_full_change():
    target = pd.DataFrame({"A": [0.8], "B": [0.1]})
    current = pd.DataFrame({"A": [0.2], "B": [0.7]})
    executed = apply_rebalance_threshold(target, current, threshold=0.01)
    pd.testing.assert_frame_equal(executed, target)


def test_rebalance_threshold_partial():
    target = pd.DataFrame({"A": [0.5], "B": [0.3]})
    current = pd.DataFrame({"A": [0.49], "B": [0.7]})
    executed = apply_rebalance_threshold(target, current, threshold=0.05)
    assert executed["A"].iloc[0] == pytest.approx(0.49)
    assert executed["B"].iloc[0] == pytest.approx(0.3)


def test_raw_weights():
    score = pd.DataFrame({"A": [0.5], "B": [-0.3]})
    vol = pd.DataFrame({"A": [0.2], "B": [0.15]})
    num_assets = pd.Series([2], index=score.index)
    raw = compute_raw_weights(score, vol, target_portfolio_vol=0.10, num_assets=num_assets)
    assert raw.shape == score.shape
    assert not raw.isna().all().all()
