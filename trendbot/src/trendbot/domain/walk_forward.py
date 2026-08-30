"""Walk-Forward Out-of-Sample Validation engine.

This module implements rigorous walk-forward validation where:

1. Parameters are selected using ONLY training data.
2. Selected parameters are frozen before the OOS period.
3. OOS returns are stitched into a single equity curve.
4. No future information ever influences parameter selection.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter

import pandas as pd

from trendbot.domain.backtest import run_backtest
from trendbot.domain.metrics import compute_metrics
from trendbot.domain.models import (
    FoldResult,
    ParameterGrid,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def generate_folds(
    n_bars: int,
    config: WalkForwardConfig,
) -> list[WalkForwardFold]:
    """Generate non-overlapping train/test folds.

    Uses bar indices.  The training window is ``[train_start, train_end)``
    and the test window is ``[test_start, test_end)``.

    Args:
        n_bars: Total number of bars in the dataset.
        config: Walk-forward configuration.

    Returns:
        List of WalkForwardFold objects.

    Raises:
        ValueError: If the dataset is too short for any fold.
    """
    folds: list[WalkForwardFold] = []
    fold_index = 0
    start = 0

    while start + config.train_window + config.test_window <= n_bars:
        train_start = start
        train_end = start + config.train_window
        test_start = train_end
        test_end = test_start + config.test_window

        if train_end - train_start >= config.minimum_training_bars:
            folds.append(
                WalkForwardFold(
                    fold_index=fold_index,
                    train_start_idx=train_start,
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                )
            )
            fold_index += 1

        start += config.step

    if not folds:
        raise ValueError(
            f"Dataset has {n_bars} bars which is too short for "
            f"train_window={config.train_window}, test_window={config.test_window}."
        )

    return folds


def validate_no_overlapping_oos(folds: list[WalkForwardFold]) -> None:
    """Verify that OOS test periods do not overlap.

    Args:
        folds: List of walk-forward folds.

    Raises:
        ValueError: If any two OOS periods overlap.
    """
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            a_start = folds[i].test_start_idx
            a_end = folds[i].test_end_idx
            b_start = folds[j].test_start_idx
            b_end = folds[j].test_end_idx
            if a_start < b_end and b_start < a_end:
                raise ValueError(
                    f"Fold {i} OOS [{a_start}, {a_end}) overlaps with "
                    f"fold {j} OOS [{b_start}, {b_end})."
                )


# ---------------------------------------------------------------------------
# Parameter enumeration
# ---------------------------------------------------------------------------


def enumerate_parameter_combinations(
    grid: ParameterGrid,
) -> list[dict[str, object]]:
    """Enumerate all parameter combinations from the grid.

    Args:
        grid: Parameter search grid.

    Returns:
        List of parameter dictionaries.
    """
    keys = [
        "lookbacks",
        "vol_window",
        "covariance_window",
        "covariance_shrinkage",
        "rebalance_threshold",
    ]
    values = [
        grid.lookbacks,
        grid.vol_window,
        grid.covariance_window,
        grid.covariance_shrinkage,
        grid.rebalance_threshold,
    ]

    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


# ---------------------------------------------------------------------------
# Training evaluation
# ---------------------------------------------------------------------------


def _run_backtest_safely(
    close: pd.DataFrame,
    params: dict[str, object],
    config: WalkForwardConfig,
) -> dict[str, pd.DataFrame | pd.Series] | None:
    """Run a backtest with given parameters, returning None on failure."""
    try:
        return run_backtest(
            close=close,
            lookbacks=params["lookbacks"],
            allow_short=config.allow_short,
            vol_window=params["vol_window"],
            ann_factor=config.ann_factor,
            target_portfolio_vol=config.target_portfolio_vol,
            max_gross_leverage=config.max_gross_leverage,
            taker_fee_pct=config.taker_fee_pct,
            slippage_pct=config.slippage_pct,
            rebalance_threshold=params["rebalance_threshold"],
            min_history=config.min_history,
            covariance_window=params["covariance_window"],
            covariance_shrinkage=params["covariance_shrinkage"],
        )
    except Exception as e:
        logger.warning("Backtest failed for params %s: %s", params, e)
        return None


def _compute_training_sharpe(
    result: dict[str, pd.DataFrame | pd.Series],
    ann_factor: int,
    min_history: int,
) -> float | None:
    """Compute training Sharpe ratio from backtest result.

    Returns None if the result has insufficient trading activity.
    """
    metrics = compute_metrics(
        returns=result["returns"],
        positions=result["positions"],
        turnover=result["turnover"],
        costs=result["costs"],
        ann_factor=ann_factor,
        gross_returns=result["gross_returns"],
        min_history=min_history,
    )

    sharpe = metrics["sharpe_ratio"]
    total_return = metrics["total_return"]

    # Require at least some trading activity
    if abs(total_return) < 1e-15 and metrics["avg_daily_turnover"] < 1e-10:
        return None

    return sharpe


def _tie_break(
    candidates: list[tuple[float, dict[str, object], dict[str, float]]],
) -> dict[str, object]:
    """Deterministic tie-breaking for parameter selection.

    Tie-breaking order:
    1. Highest training Sharpe
    2. Lowest max drawdown (within tolerance)
    3. Lowest turnover (within tolerance)
    4. Lowest rebalance_threshold (simpler)
    5. Smallest vol_window (simpler)
    6. Smallest covariance_window (simpler)

    Args:
        candidates: List of (sharpe, params, metrics) tuples, sorted by sharpe desc.

    Returns:
        Best parameter combination.
    """
    if len(candidates) == 1:
        return candidates[0][1]

    sharpe_tol = 1e-6

    # Group by sharpe tolerance
    best_sharpe = candidates[0][0]
    tied = [c for c in candidates if abs(c[0] - best_sharpe) < sharpe_tol]

    if len(tied) == 1:
        return tied[0][1]

    # Tie-break on max drawdown (less negative is better)
    tied.sort(key=lambda x: x[2]["max_drawdown"], reverse=True)
    best_dd = tied[0][2]["max_drawdown"]
    dd_tol = 1e-6
    still_tied = [c for c in tied if abs(c[2]["max_drawdown"] - best_dd) < dd_tol]

    if len(still_tied) == 1:
        return still_tied[0][1]

    # Tie-break on turnover
    still_tied.sort(key=lambda x: x[2]["avg_daily_turnover"])
    best_turnover = still_tied[0][2]["avg_daily_turnover"]
    to_tol = 1e-10
    final = [c for c in still_tied if abs(c[2]["avg_daily_turnover"] - best_turnover) < to_tol]

    if len(final) == 1:
        return final[0][1]

    # Tie-break on simplicity
    final.sort(
        key=lambda x: (
            x[1]["rebalance_threshold"],
            x[1]["vol_window"],
            x[1]["covariance_window"],
        )
    )
    return final[0][1]


# ---------------------------------------------------------------------------
# Core WFO engine
# ---------------------------------------------------------------------------


def select_parameters(
    train_close: pd.DataFrame,
    grid: ParameterGrid,
    config: WalkForwardConfig,
) -> tuple[dict[str, object], float, dict[str, float]]:
    """Select optimal parameters using ONLY training data.

    Args:
        train_close: Training period close prices.
        grid: Parameter search grid.
        config: Walk-forward configuration.

    Returns:
        Tuple of (selected_parameters, training_sharpe, training_metrics).

    Raises:
        ValueError: If no parameter combination produces valid results.
    """
    combos = enumerate_parameter_combinations(grid)
    results: list[tuple[float, dict[str, object], dict[str, float]]] = []

    for params in combos:
        bt = _run_backtest_safely(train_close, params, config)
        if bt is None:
            continue

        sharpe = _compute_training_sharpe(bt, config.ann_factor, config.min_history)
        if sharpe is None:
            continue

        metrics = compute_metrics(
            returns=bt["returns"],
            positions=bt["positions"],
            turnover=bt["turnover"],
            costs=bt["costs"],
            ann_factor=config.ann_factor,
            gross_returns=bt["gross_returns"],
            min_history=config.min_history,
        )

        results.append((sharpe, params, metrics))

    if not results:
        raise ValueError(
            "No parameter combination produced valid training results. "
            "Consider widening the parameter grid or relaxing constraints."
        )

    results.sort(key=lambda x: x[0], reverse=True)
    best_sharpe, best_params, best_metrics = results[0]

    # Apply tie-breaking
    best_params = _tie_break(results)

    # Re-compute metrics for the tie-broken winner
    bt = _run_backtest_safely(train_close, best_params, config)
    if bt is not None:
        computed = _compute_training_sharpe(
            bt, config.ann_factor, config.min_history
        )
        best_sharpe = computed or best_sharpe
        best_metrics = compute_metrics(
            returns=bt["returns"],
            positions=bt["positions"],
            turnover=bt["turnover"],
            costs=bt["costs"],
            ann_factor=config.ann_factor,
            gross_returns=bt["gross_returns"],
            min_history=config.min_history,
        )

    return best_params, best_sharpe, best_metrics


def run_oos_fold(
    full_close: pd.DataFrame,
    fold: WalkForwardFold,
    selected_params: dict[str, object],
    config: WalkForwardConfig,
) -> FoldResult:
    """Run a single OOS fold with frozen parameters.

    The backtest runs on data from fold.train_start to fold.test_end,
    but only the OOS portion [test_start, test_end) is extracted.

    Args:
        full_close: Full dataset (train_start through test_end).
        fold: Fold definition with bar indices.
        selected_params: Parameters frozen during training.
        config: Walk-forward configuration.

    Returns:
        FoldResult with OOS returns and metrics.
    """
    # Run backtest on full context (train + test)
    bt = run_backtest(
        close=full_close,
        lookbacks=selected_params["lookbacks"],
        allow_short=config.allow_short,
        vol_window=selected_params["vol_window"],
        ann_factor=config.ann_factor,
        target_portfolio_vol=config.target_portfolio_vol,
        max_gross_leverage=config.max_gross_leverage,
        taker_fee_pct=config.taker_fee_pct,
        slippage_pct=config.slippage_pct,
        rebalance_threshold=selected_params["rebalance_threshold"],
        min_history=config.min_history,
        covariance_window=selected_params["covariance_window"],
        covariance_shrinkage=selected_params["covariance_shrinkage"],
    )

    # Extract OOS slice — convert absolute indices to relative within full_close
    relative_test_start = fold.test_start_idx - fold.train_start_idx
    relative_test_end = fold.test_end_idx - fold.train_start_idx
    relative_train_end = fold.train_end_idx - fold.train_start_idx

    oos_returns = bt["returns"].iloc[relative_test_start:relative_test_end].copy()
    oos_gross_returns = bt["gross_returns"].iloc[relative_test_start:relative_test_end].copy()
    oos_turnover = bt["turnover"].iloc[relative_test_start:relative_test_end].copy()
    oos_costs = bt["costs"].iloc[relative_test_start:relative_test_end].copy()
    oos_positions = bt["positions"].iloc[relative_test_start:relative_test_end].copy()

    # Build OOS equity curve from returns
    oos_equity = (1 + oos_returns).cumprod()

    # Compute OOS metrics (min_history=0 since we already sliced the OOS period)
    oos_metrics = compute_metrics(
        returns=oos_returns,
        positions=oos_positions,
        turnover=oos_turnover,
        costs=oos_costs,
        ann_factor=config.ann_factor,
        gross_returns=oos_gross_returns,
        min_history=0,
    )

    # Compute training metrics from the backtest result's returns up to OOS start
    train_returns = bt["returns"].iloc[:relative_train_end]
    train_positions = bt["positions"].iloc[:relative_train_end]
    train_turnover = bt["turnover"].iloc[:relative_train_end]
    train_costs = bt["costs"].iloc[:relative_train_end]
    train_gross = bt["gross_returns"].iloc[:relative_train_end]

    training_metrics = compute_metrics(
        returns=train_returns,
        positions=train_positions,
        turnover=train_turnover,
        costs=train_costs,
        ann_factor=config.ann_factor,
        gross_returns=train_gross,
        min_history=config.min_history,
    )

    return FoldResult(
        fold=fold,
        selected_parameters=selected_params,
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        oos_gross_returns=oos_gross_returns,
        oos_turnover=oos_turnover,
        oos_costs=oos_costs,
        oos_positions=oos_positions,
        training_sharpe=training_metrics["sharpe_ratio"],
        training_metrics=training_metrics,
        oos_metrics=oos_metrics,
    )


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


def stitch_oos_returns(fold_results: list[FoldResult]) -> tuple[pd.Series, pd.Series]:
    """Concatenate non-overlapping OOS returns into a single series.

    Args:
        fold_results: List of FoldResult objects with OOS returns.

    Returns:
        Tuple of (stitched_returns, stitched_equity).
    """
    if not fold_results:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    all_returns = [fr.oos_returns for fr in fold_results]
    stitched = pd.concat(all_returns)
    stitched_equity = (1 + stitched).cumprod()

    return stitched, stitched_equity


def compute_parameter_stability(
    fold_results: list[FoldResult],
) -> dict[str, dict[object, int]]:
    """Count how often each parameter value is selected across folds.

    Args:
        fold_results: List of FoldResult objects.

    Returns:
        Dictionary mapping parameter names to value-count dictionaries.
    """
    if not fold_results:
        return {}

    param_keys = fold_results[0].selected_parameters.keys()
    stability: dict[str, dict[object, int]] = {}

    for key in param_keys:
        counter: Counter[object] = Counter()
        for fr in fold_results:
            val = fr.selected_parameters[key]
            # Convert list to tuple for hashability
            counter[tuple(val) if isinstance(val, list) else val] += 1
        stability[key] = dict(counter)

    return stability


# ---------------------------------------------------------------------------
# Full WFO runner
# ---------------------------------------------------------------------------


def run_walk_forward(
    close: pd.DataFrame,
    grid: ParameterGrid,
    config: WalkForwardConfig,
) -> WalkForwardReport:
    """Run the complete walk-forward out-of-sample validation.

    This is the primary entry point.  It:
    1. Generates folds from the dataset.
    2. For each fold, selects parameters on TRAIN data only.
    3. Runs OOS with frozen parameters.
    4. Stitches OOS returns into a single equity curve.
    5. Computes aggregate and per-fold metrics.

    Args:
        close: Full dataset of close prices.
        grid: Parameter search grid.
        config: Walk-forward configuration.

    Returns:
        WalkForwardReport with all results.
    """
    n_bars = len(close)
    folds = generate_folds(n_bars, config)
    validate_no_overlapping_oos(folds)

    logger.info(
        "WFO: %d folds, train=%d, test=%d, step=%d, total_bars=%d",
        len(folds), config.train_window, config.test_window,
        config.step, n_bars,
    )

    fold_results: list[FoldResult] = []

    for fold in folds:
        logger.info(
            "Fold %d: train=[%d, %d) test=[%d, %d)",
            fold.fold_index,
            fold.train_start_idx, fold.train_end_idx,
            fold.test_start_idx, fold.test_end_idx,
        )

        # Extract training data
        train_close = close.iloc[fold.train_start_idx : fold.train_end_idx]

        # Select parameters using ONLY training data
        selected_params, train_sharpe, train_metrics = select_parameters(
            train_close, grid, config,
        )

        logger.info("Fold %d: selected params=%s, train_sharpe=%.3f",
                     fold.fold_index, selected_params, train_sharpe)

        # Run OOS with frozen parameters
        full_close = close.iloc[fold.train_start_idx : fold.test_end_idx]
        fold_result = run_oos_fold(full_close, fold, selected_params, config)

        fold_results.append(fold_result)

    # Stitch OOS returns
    stitched_returns, stitched_equity = stitch_oos_returns(fold_results)

    # Compute aggregate metrics
    aggregate_metrics = compute_metrics(
        returns=stitched_returns,
        positions=pd.concat([fr.oos_positions for fr in fold_results]),
        turnover=pd.concat([fr.oos_turnover for fr in fold_results]),
        costs=pd.concat([fr.oos_costs for fr in fold_results]),
        ann_factor=config.ann_factor,
        gross_returns=pd.concat([fr.oos_gross_returns for fr in fold_results]),
        min_history=0,
    )

    # Parameter stability
    param_stability = compute_parameter_stability(fold_results)

    # Per-fold summary
    per_fold_summary = []
    for fr in fold_results:
        per_fold_summary.append({
            "fold": fr.fold.fold_index,
            "train_range": f"[{fr.fold.train_start_idx}, {fr.fold.train_end_idx})",
            "test_range": f"[{fr.fold.test_start_idx}, {fr.fold.test_end_idx})",
            "selected_params": fr.selected_parameters,
            "training_sharpe": fr.training_sharpe,
            "oos_sharpe": fr.oos_metrics["sharpe_ratio"],
            "oos_cagr": fr.oos_metrics["cagr"],
            "oos_max_drawdown": fr.oos_metrics["max_drawdown"],
            "oos_total_return": fr.oos_metrics["total_return"],
        })

    return WalkForwardReport(
        folds=fold_results,
        stitched_oos_returns=stitched_returns,
        stitched_oos_equity=stitched_equity,
        aggregate_metrics=aggregate_metrics,
        parameter_stability=param_stability,
        per_fold_summary=per_fold_summary,
    )
