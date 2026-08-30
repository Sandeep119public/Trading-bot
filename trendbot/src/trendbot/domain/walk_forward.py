"""Rigorous walk-forward out-of-sample validation.

The WFO engine enforces three boundaries:

* parameter selection sees training data only;
* selected parameters are immutable during OOS execution;
* OOS scoring uses only the designated test interval.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from typing import cast

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


def generate_folds(n_bars: int, config: WalkForwardConfig) -> list[WalkForwardFold]:
    """Generate fixed-length, non-overlapping train/test folds."""
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
    """Reject overlapping OOS intervals and malformed fold ordering."""
    ordered = sorted(folds, key=lambda f: f.test_start_idx)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.test_end_idx > current.test_start_idx:
            raise ValueError(
                f"Fold {current.fold_index} OOS [{current.test_start_idx}, "
                f"{current.test_end_idx}) overlaps fold {previous.fold_index} "
                f"OOS [{previous.test_start_idx}, {previous.test_end_idx})."
            )


def _validate_close(close: pd.DataFrame) -> None:
    """Validate input data before optimization."""
    if close.empty:
        raise ValueError("WFO requires a non-empty close-price DataFrame")
    if not close.index.is_monotonic_increasing:
        raise ValueError("WFO close index must be sorted in increasing time order")
    if close.index.has_duplicates:
        raise ValueError("WFO close index must not contain duplicate timestamps")
    if close.columns.empty:
        raise ValueError("WFO requires at least one asset column")


def enumerate_parameter_combinations(grid: ParameterGrid) -> list[dict[str, object]]:
    """Enumerate the explicit parameter grid deterministically."""
    keys = (
        "lookbacks", "vol_window", "covariance_window",
        "covariance_shrinkage", "rebalance_threshold",
    )
    values = (
        grid.lookbacks, grid.vol_window, grid.covariance_window,
        grid.covariance_shrinkage, grid.rebalance_threshold,
    )
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _required_history(params: dict[str, object], config: WalkForwardConfig) -> int:
    """Return the exact warmup gate used by the production backtest."""
    lookbacks = cast(list[int], params["lookbacks"])
    return max(
        config.min_history,
        max(lookbacks, default=1),
        int(params["vol_window"]),
        int(params["covariance_window"]),
    )


def _active_slice(
    result: dict[str, pd.DataFrame | pd.Series],
    params: dict[str, object],
    config: WalkForwardConfig,
) -> slice:
    """Return the first bar that can contain a realized strategy return."""
    del result
    return slice(_required_history(params, config) + 1, None)


def _run_backtest_safely(
    close: pd.DataFrame,
    params: dict[str, object],
    config: WalkForwardConfig,
) -> dict[str, pd.DataFrame | pd.Series] | None:
    """Run a candidate backtest; invalid candidates are excluded explicitly."""
    try:
        return run_backtest(
            close=close,
            lookbacks=cast(list[int], params["lookbacks"]),
            allow_short=config.allow_short,
            vol_window=int(params["vol_window"]),
            ann_factor=config.ann_factor,
            target_portfolio_vol=config.target_portfolio_vol,
            max_gross_leverage=config.max_gross_leverage,
            taker_fee_pct=config.taker_fee_pct,
            slippage_pct=config.slippage_pct,
            rebalance_threshold=float(params["rebalance_threshold"]),
            min_history=config.min_history,
            covariance_window=int(params["covariance_window"]),
            covariance_shrinkage=float(params["covariance_shrinkage"]),
        )
    except Exception as exc:
        logger.warning("Candidate rejected for params %s: %s", params, exc)
        return None


def _held_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """Align exposure with the return it actually earns."""
    return positions.shift(1).fillna(0.0)


def _training_metrics(
    result: dict[str, pd.DataFrame | pd.Series],
    params: dict[str, object],
    config: WalkForwardConfig,
) -> dict[str, float]:
    """Compute training metrics using causal return/exposure alignment."""
    active = _active_slice(result, params, config)
    returns = cast(pd.Series, result["returns"]).iloc[active]
    gross_returns = cast(pd.Series, result["gross_returns"]).iloc[active]
    turnover = cast(pd.Series, result["turnover"]).iloc[active]
    costs = cast(pd.Series, result["costs"]).iloc[active]
    positions = _held_positions(cast(pd.DataFrame, result["positions"])).iloc[active]
    return compute_metrics(
        returns=returns,
        positions=positions,
        turnover=turnover,
        costs=costs,
        ann_factor=config.ann_factor,
        gross_returns=gross_returns,
        min_history=0,
    )


def _candidate_is_eligible(
    result: dict[str, pd.DataFrame | pd.Series],
    params: dict[str, object],
    config: WalkForwardConfig,
) -> bool:
    """Require enough realized observations and actual trades."""
    active = _active_slice(result, params, config)
    returns = cast(pd.Series, result["returns"]).iloc[active]
    turnover = cast(pd.Series, result["turnover"]).iloc[active]
    observations = int(returns.notna().sum())
    trades = int((turnover > 0.0).sum())
    return (
        observations >= config.minimum_training_observations
        and trades >= config.minimum_training_trades
    )


def _compute_training_sharpe(
    result: dict[str, pd.DataFrame | pd.Series],
    ann_factor: int,
    min_history: int,
) -> float | None:
    """Backward-compatible helper retained for existing callers/tests."""
    metrics = compute_metrics(
        returns=cast(pd.Series, result["returns"]),
        positions=_held_positions(cast(pd.DataFrame, result["positions"])),
        turnover=cast(pd.Series, result["turnover"]),
        costs=cast(pd.Series, result["costs"]),
        ann_factor=ann_factor,
        gross_returns=cast(pd.Series, result["gross_returns"]),
        min_history=min_history,
    )
    if abs(metrics["total_return"]) < 1e-15 and metrics["avg_daily_turnover"] < 1e-10:
        return None
    return metrics["sharpe_ratio"]


def _tie_break(
    candidates: list[tuple[float, dict[str, object], dict[str, float]]],
    sharpe_tolerance: float = 0.02,
) -> dict[str, object]:
    """Prefer robust candidates when Sharpe differences are practically tiny."""
    if not candidates:
        raise ValueError("Cannot tie-break an empty candidate set")
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    best_sharpe = candidates[0][0]
    tied = [c for c in candidates if best_sharpe - c[0] <= sharpe_tolerance]
    best_dd = max(c[2]["max_drawdown"] for c in tied)
    tied = [c for c in tied if best_dd - c[2]["max_drawdown"] <= 1e-4]
    best_turnover = min(c[2]["avg_daily_turnover"] for c in tied)
    tied = [c for c in tied if c[2]["avg_daily_turnover"] - best_turnover <= 1e-6]
    tied.sort(
        key=lambda c: (
            float(c[1]["rebalance_threshold"]),
            int(c[1]["vol_window"]),
            int(c[1]["covariance_window"]),
            float(c[1]["covariance_shrinkage"]),
            tuple(cast(list[int], c[1]["lookbacks"])),
        )
    )
    return tied[0][1]


def select_parameters(
    train_close: pd.DataFrame,
    grid: ParameterGrid,
    config: WalkForwardConfig,
) -> tuple[dict[str, object], float, dict[str, float]]:
    """Select parameters using training data only."""
    if len(train_close) < config.minimum_training_bars:
        raise ValueError(
            f"Training data has {len(train_close)} bars, below "
            f"minimum_training_bars={config.minimum_training_bars}."
        )
    results: list[tuple[float, dict[str, object], dict[str, float]]] = []
    for params in enumerate_parameter_combinations(grid):
        if _required_history(params, config) + 1 >= len(train_close):
            continue
        bt = _run_backtest_safely(train_close, params, config)
        if bt is None or not _candidate_is_eligible(bt, params, config):
            continue
        metrics = _training_metrics(bt, params, config)
        sharpe = metrics["sharpe_ratio"]
        if not pd.notna(sharpe):
            continue
        results.append((float(sharpe), params, metrics))
    if not results:
        raise ValueError(
            "No parameter combination produced valid training results. "
            "Increase training data or relax minimum observation/trade constraints."
        )
    best_params = _tie_break(results, config.sharpe_tie_tolerance)
    winner = next(r for r in results if r[1] == best_params)
    return best_params, winner[0], winner[2]


def run_oos_fold(
    full_close: pd.DataFrame,
    fold: WalkForwardFold,
    selected_params: dict[str, object],
    config: WalkForwardConfig,
) -> FoldResult:
    """Run frozen parameters over train context plus OOS period."""
    expected_len = fold.test_end_idx - fold.train_start_idx
    if len(full_close) != expected_len:
        raise ValueError(
            "run_oos_fold requires full_close to span exactly "
            "train_start_idx:test_end_idx"
        )
    bt = run_backtest(
        close=full_close,
        lookbacks=cast(list[int], selected_params["lookbacks"]),
        allow_short=config.allow_short,
        vol_window=int(selected_params["vol_window"]),
        ann_factor=config.ann_factor,
        target_portfolio_vol=config.target_portfolio_vol,
        max_gross_leverage=config.max_gross_leverage,
        taker_fee_pct=config.taker_fee_pct,
        slippage_pct=config.slippage_pct,
        rebalance_threshold=float(selected_params["rebalance_threshold"]),
        min_history=config.min_history,
        covariance_window=int(selected_params["covariance_window"]),
        covariance_shrinkage=float(selected_params["covariance_shrinkage"]),
    )

    relative_test_start = fold.test_start_idx - fold.train_start_idx
    relative_test_end = fold.test_end_idx - fold.train_start_idx
    relative_train_end = fold.train_end_idx - fold.train_start_idx

    all_returns = cast(pd.Series, bt["returns"])
    all_gross = cast(pd.Series, bt["gross_returns"])
    all_turnover = cast(pd.Series, bt["turnover"])
    all_costs = cast(pd.Series, bt["costs"])
    all_positions = cast(pd.DataFrame, bt["positions"])

    oos_returns = all_returns.iloc[relative_test_start:relative_test_end].copy()
    oos_gross_returns = all_gross.iloc[relative_test_start:relative_test_end].copy()
    oos_turnover = all_turnover.iloc[relative_test_start:relative_test_end].copy()
    oos_costs = all_costs.iloc[relative_test_start:relative_test_end].copy()
    oos_positions = _held_positions(all_positions).iloc[
        relative_test_start:relative_test_end
    ].copy()
    oos_equity = (1.0 + oos_returns).cumprod()

    oos_metrics = compute_metrics(
        returns=oos_returns,
        positions=oos_positions,
        turnover=oos_turnover,
        costs=oos_costs,
        ann_factor=config.ann_factor,
        gross_returns=oos_gross_returns,
        min_history=0,
    )

    train_returns = all_returns.iloc[:relative_train_end]
    train_gross = all_gross.iloc[:relative_train_end]
    train_turnover = all_turnover.iloc[:relative_train_end]
    train_costs = all_costs.iloc[:relative_train_end]
    train_positions = _held_positions(all_positions).iloc[:relative_train_end]
    training_metrics = compute_metrics(
        returns=train_returns,
        positions=train_positions,
        turnover=train_turnover,
        costs=train_costs,
        ann_factor=config.ann_factor,
        gross_returns=train_gross,
        min_history=0,
    )

    return FoldResult(
        fold=fold,
        selected_parameters=dict(selected_params),
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


def stitch_oos_returns(fold_results: list[FoldResult]) -> tuple[pd.Series, pd.Series]:
    """Concatenate OOS returns and reject duplicate timestamps."""
    if not fold_results:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    stitched = pd.concat([fr.oos_returns for fr in fold_results])
    if stitched.index.has_duplicates:
        raise ValueError("Stitched OOS returns contain duplicate timestamps")
    if not stitched.index.is_monotonic_increasing:
        stitched = stitched.sort_index()
    return stitched, (1.0 + stitched).cumprod()


def compute_parameter_stability(
    fold_results: list[FoldResult],
) -> dict[str, dict[object, int]]:
    """Count parameter selections across folds."""
    if not fold_results:
        return {}
    stability: dict[str, dict[object, int]] = {}
    for key in fold_results[0].selected_parameters:
        counter: Counter[object] = Counter()
        for fr in fold_results:
            value = fr.selected_parameters[key]
            counter[tuple(value) if isinstance(value, list) else value] += 1
        stability[key] = dict(counter)
    return stability


def _aggregate_metrics(
    fold_results: list[FoldResult], config: WalkForwardConfig
) -> dict[str, float]:
    """Compute metrics from the stitched OOS series, never fold averages."""
    return compute_metrics(
        returns=pd.concat([fr.oos_returns for fr in fold_results]),
        positions=pd.concat([fr.oos_positions for fr in fold_results]),
        turnover=pd.concat([fr.oos_turnover for fr in fold_results]),
        costs=pd.concat([fr.oos_costs for fr in fold_results]),
        ann_factor=config.ann_factor,
        gross_returns=pd.concat([fr.oos_gross_returns for fr in fold_results]),
        min_history=0,
    )


def run_walk_forward(
    close: pd.DataFrame,
    grid: ParameterGrid,
    config: WalkForwardConfig,
) -> WalkForwardReport:
    """Run complete training-only optimization and frozen OOS validation."""
    _validate_close(close)
    folds = generate_folds(len(close), config)
    validate_no_overlapping_oos(folds)
    fold_results: list[FoldResult] = []

    for fold in folds:
        train_close = close.iloc[fold.train_start_idx:fold.train_end_idx]
        selected_params, train_sharpe, _ = select_parameters(
            train_close, grid, config
        )
        logger.info(
            "Fold %d: selected=%s train_sharpe=%.4f",
            fold.fold_index,
            selected_params,
            train_sharpe,
        )
        full_close = close.iloc[fold.train_start_idx:fold.test_end_idx]
        fold_results.append(run_oos_fold(full_close, fold, selected_params, config))

    stitched_returns, stitched_equity = stitch_oos_returns(fold_results)
    aggregate_metrics = _aggregate_metrics(fold_results, config)
    stability = compute_parameter_stability(fold_results)
    per_fold_summary: list[dict[str, object]] = []
    for fr in fold_results:
        per_fold_summary.append(
            {
                "fold": fr.fold.fold_index,
                "train_range": f"[{fr.fold.train_start_idx}, {fr.fold.train_end_idx})",
                "test_range": f"[{fr.fold.test_start_idx}, {fr.fold.test_end_idx})",
                "selected_params": fr.selected_parameters,
                "training_sharpe": fr.training_sharpe,
                "oos_sharpe": fr.oos_metrics["sharpe_ratio"],
                "oos_cagr": fr.oos_metrics["cagr"],
                "oos_max_drawdown": fr.oos_metrics["max_drawdown"],
                "oos_total_return": fr.oos_metrics["total_return"],
                "oos_turnover": fr.oos_metrics["avg_daily_turnover"],
                "oos_cost_drag": fr.oos_metrics["total_cost_drag"],
            }
        )
    return WalkForwardReport(
        folds=fold_results,
        stitched_oos_returns=stitched_returns,
        stitched_oos_equity=stitched_equity,
        aggregate_metrics=aggregate_metrics,
        parameter_stability=stability,
        per_fold_summary=per_fold_summary,
    )
