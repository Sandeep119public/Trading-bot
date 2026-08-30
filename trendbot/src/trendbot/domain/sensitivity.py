"""Parameter sensitivity analysis engine.

Pure domain functions for walk-forward and heatmap analysis to detect overfitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from trendbot.domain.backtest import run_backtest
from trendbot.domain.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensitivityResult:
    """Single parameter combination result."""

    lookbacks: list[int]
    vol_window: int
    target_portfolio_vol: float
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    total_return: float
    avg_turnover: float


def run_sensitivity_analysis(
    close: pd.DataFrame,
    base_lookbacks: list[int],
    base_vol_window: int,
    base_target_vol: float,
    base_ann_factor: int,
    base_max_leverage: float,
    base_taker_fee_pct: float,
    base_slippage_pct: float,
    base_rebalance_threshold: float,
    base_min_history: int,
    allow_short: bool,
    lookback_multipliers: list[float] | None = None,
    vol_windows: list[int] | None = None,
    target_vols: list[float] | None = None,
) -> pd.DataFrame:
    """Run sensitivity analysis across parameter combinations.

    Tests every combination of lookback multipliers, volatility windows,
    and target portfolio vols. Returns a DataFrame of results suitable
    for heatmap visualization.

    Args:
        close: DataFrame of daily close prices (index=date, columns=assets).
        base_lookbacks: Base momentum lookback periods.
        base_vol_window: Base volatility window.
        base_target_vol: Base target portfolio volatility.
        base_ann_factor: Annualization factor (252 or 365).
        base_max_leverage: Maximum gross leverage.
        base_taker_fee_pct: Taker fee as decimal fraction.
        base_slippage_pct: Slippage as decimal fraction.
        base_rebalance_threshold: Minimum weight change to trigger rebalance.
        base_min_history: Minimum bars before trading starts.
        allow_short: Whether to allow short positions.
        lookback_multipliers: Multipliers to apply to base lookbacks (e.g., [0.5, 1.0, 2.0]).
        vol_windows: List of volatility windows to test.
        target_vols: List of target portfolio vols to test.

    Returns:
        DataFrame with columns: lookback_config, vol_window, target_vol,
        sharpe_ratio, cagr, max_drawdown, total_return, avg_turnover.
    """
    if lookback_multipliers is None:
        lookback_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
    if vol_windows is None:
        vol_windows = [14, 21, 30, 42, 63]
    if target_vols is None:
        target_vols = [0.05, 0.10, 0.15]

    results: list[dict] = []

    total_combos = len(lookback_multipliers) * len(vol_windows) * len(target_vols)
    logger.info("Running sensitivity analysis: %d combinations", total_combos)

    for lb_mult in lookback_multipliers:
        scaled_lookbacks = _scale_lookbacks(base_lookbacks, lb_mult)
        lb_label = _format_lookback_label(scaled_lookbacks)

        for vol_win in vol_windows:
            for tgt_vol in target_vols:
                try:
                    bt_result = run_backtest(
                        close=close,
                        lookbacks=scaled_lookbacks,
                        allow_short=allow_short,
                        vol_window=vol_win,
                        ann_factor=base_ann_factor,
                        target_portfolio_vol=tgt_vol,
                        max_gross_leverage=base_max_leverage,
                        taker_fee_pct=base_taker_fee_pct,
                        slippage_pct=base_slippage_pct,
                        rebalance_threshold=base_rebalance_threshold,
                        min_history=base_min_history,
                    )

                    metrics = compute_metrics(
                        returns=bt_result["returns"],
                        positions=bt_result["positions"],
                        turnover=bt_result["turnover"],
                        costs=bt_result["costs"],
                        ann_factor=base_ann_factor,
                        gross_returns=bt_result["gross_returns"],
                        min_history=base_min_history,
                    )

                    results.append({
                        "lookback_config": lb_label,
                        "lookback_multiplier": lb_mult,
                        "vol_window": vol_win,
                        "target_vol": tgt_vol,
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "cagr": metrics["cagr"],
                        "max_drawdown": metrics["max_drawdown"],
                        "total_return": metrics["total_return"],
                        "avg_turnover": metrics["avg_daily_turnover"],
                    })
                except Exception as e:
                    logger.warning(
                        "Failed for lb_mult=%.2f, vol=%d, tgt_vol=%.2f: %s",
                        lb_mult, vol_win, tgt_vol, e,
                    )

    df = pd.DataFrame(results)
    logger.info("Sensitivity analysis complete: %d successful runs", len(df))
    return df


def _scale_lookbacks(base_lookbacks: list[int], multiplier: float) -> list[int]:
    """Scale lookback periods by a multiplier, ensuring minimum of 2."""
    return [max(2, int(round(lb * multiplier))) for lb in base_lookbacks]


def _format_lookback_label(lookbacks: list[int]) -> str:
    """Create a compact label for a lookback configuration."""
    return "[" + ",".join(str(lb) for lb in lookbacks) + "]"


def compute_plateau_metrics(results_df: pd.DataFrame) -> dict:
    """Identify the robust parameter plateau region.

    The plateau is the contiguous region where Sharpe ratios are within
    80% of the maximum, indicating parameters that are not overfitted.

    Args:
        results_df: Output from run_sensitivity_analysis.

    Returns:
        Dictionary with plateau analysis summary.
    """
    if results_df.empty:
        return {"plateau_found": False, "plateau_pct": 0.0, "max_sharpe": 0.0}

    max_sharpe = results_df["sharpe_ratio"].max()
    if max_sharpe <= 0:
        return {"plateau_found": False, "plateau_pct": 0.0, "max_sharpe": float(max_sharpe)}

    threshold = max_sharpe * 0.8
    plateau = results_df[results_df["sharpe_ratio"] >= threshold]
    plateau_pct = len(plateau) / len(results_df) * 100

    return {
        "plateau_found": plateau_pct > 20.0,
        "plateau_pct": round(plateau_pct, 1),
        "max_sharpe": round(float(max_sharpe), 3),
        "threshold_sharpe": round(float(threshold), 3),
        "plateau_configs": plateau["lookback_config"].unique().tolist(),
        "plateau_vol_windows": sorted(plateau["vol_window"].unique().tolist()),
    }
