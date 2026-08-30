"""WFO deep audit: analysis, null test, cost sensitivity, regime analysis.

Run with: python -m tests.wfo_deep_audit
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trendbot.domain.backtest import run_backtest, compute_benchmark_returns
from trendbot.domain.metrics import compute_metrics
from trendbot.domain.models import ParameterGrid, WalkForwardConfig
from trendbot.domain.walk_forward import (
    _required_history,
    enumerate_parameter_combinations,
    generate_folds,
    run_oos_fold,
    run_walk_forward,
    select_parameters,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ANN_FACTOR = 365


def make_synthetic_close(n: int = 600, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    a_ret = np.random.normal(0.0004, 0.012, n)
    b_ret = np.random.normal(0.0002, 0.018, n)
    c_ret = np.random.normal(0.0003, 0.015, n)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + a_ret),
            "B": 50 * np.cumprod(1 + b_ret),
            "C": 75 * np.cumprod(1 + c_ret),
        },
        index=dates,
    )


def default_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_window=200,
        test_window=60,
        step=60,
        minimum_training_bars=100,
        minimum_training_observations=30,
        minimum_training_trades=1,
        ann_factor=ANN_FACTOR,
        target_portfolio_vol=0.10,
        max_gross_leverage=1.0,
        taker_fee_pct=0.001,
        slippage_pct=0.0005,
        allow_short=True,
        min_history=30,
        sharpe_tie_tolerance=0.02,
    )


def default_grid() -> ParameterGrid:
    return ParameterGrid(
        lookbacks=[[5, 10, 21], [10, 21, 42]],
        vol_window=[21, 40],
        covariance_window=[60],
        covariance_shrinkage=[0.1],
        rebalance_threshold=[0.01],
    )


def fmt_pct(v: float) -> str:
    return f"{v*100:+.2f}%"


def fmt_float(v: float) -> str:
    return f"{v:.4f}"


# ===========================================================================
# PHASE 6: PARAMETER OPTIMIZATION
# ===========================================================================


def audit_parameter_optimization(grid: ParameterGrid) -> None:
    combos = enumerate_parameter_combinations(grid)
    print(f"\n{'='*70}")
    print("PHASE 6: PARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"  lookbacks options:          {len(grid.lookbacks)}")
    print(f"  vol_window options:         {len(grid.vol_window)}")
    print(f"  covariance_window options:  {len(grid.covariance_window)}")
    print(f"  covariance_shrinkage opts:  {len(grid.covariance_shrinkage)}")
    print(f"  rebalance_threshold opts:   {len(grid.rebalance_threshold)}")
    print(f"  TOTAL CANDIDATES:           {len(combos)}")
    print(f"  Optimization objective:     Training Sharpe ratio")


# ===========================================================================
# PHASE 7: MULTIPLE-TESTING BIAS
# ===========================================================================


def audit_multiple_testing(
    close: pd.DataFrame, grid: ParameterGrid, config: WalkForwardConfig,
) -> None:
    print(f"\n{'='*70}")
    print("PHASE 7: MULTIPLE-TESTING BIAS")
    print(f"{'='*70}")

    folds = generate_folds(len(close), config)
    all_combos = enumerate_parameter_combinations(grid)
    total_candidates = len(all_combos)

    for fold in folds:
        train_close = close.iloc[fold.train_start_idx:fold.train_end_idx]
        results = []
        for params in all_combos:
            if _required_history(params, config) + 1 >= len(train_close):
                continue
            try:
                from trendbot.domain.walk_forward import (
                    _run_backtest_safely,
                    _candidate_is_eligible,
                    _training_metrics,
                )
                bt = _run_backtest_safely(train_close, params, config)
                if bt is None or not _candidate_is_eligible(bt, params, config):
                    continue
                metrics = _training_metrics(bt, params, config)
                sharpe = metrics["sharpe_ratio"]
                if pd.notna(sharpe):
                    results.append((float(sharpe), params, metrics))
            except Exception:
                continue

        if not results:
            print(f"  Fold {fold.fold_index}: no eligible candidates")
            continue

        sharpes = sorted([r[0] for r in results], reverse=True)
        best = sharpes[0]
        median = sharpes[len(sharpes) // 2]
        mean_s = np.mean(sharpes)
        std_s = np.std(sharpes) if len(sharpes) > 1 else 0.0

        winner_sharpe = best
        rank = next(i + 1 for i, s in enumerate(sharpes) if s == winner_sharpe)

        print(f"  Fold {fold.fold_index}:")
        print(f"    eligible candidates:  {len(results)}/{total_candidates}")
        print(f"    best train Sharpe:    {fmt_float(best)}")
        print(f"    median train Sharpe:  {fmt_float(median)}")
        print(f"    mean train Sharpe:    {fmt_float(mean_s)}")
        print(f"    std train Sharpe:     {fmt_float(std_s)}")
        print(f"    winner rank:          {rank}/{len(results)}")


# ===========================================================================
# PHASE 8: PARAMETER STABILITY
# ===========================================================================


def audit_parameter_stability(report) -> None:
    print(f"\n{'='*70}")
    print("PHASE 8: PARAMETER STABILITY")
    print(f"{'='*70}")
    for param_name, counts in report.parameter_stability.items():
        print(f"  {param_name}:")
        for val, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {val}: {count}/{len(report.folds)} folds")


# ===========================================================================
# PHASE 9: TRAIN vs OOS DEGRADATION
# ===========================================================================


def audit_degradation(report) -> None:
    print(f"\n{'='*70}")
    print("PHASE 9: TRAIN vs OOS DEGRADATION")
    print(f"{'='*70}")
    print(f"  {'Fold':>4} | {'Train Shrp':>10} | {'OOS Shrp':>9} | {'Degrad%':>8} | {'Train CAGR':>10} | {'OOS CAGR':>9}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}")

    for fr in report.folds:
        t_sharpe = fr.training_metrics.get("sharpe_ratio", 0.0)
        o_sharpe = fr.oos_metrics.get("sharpe_ratio", 0.0)
        t_cagr = fr.training_metrics.get("cagr", 0.0)
        o_cagr = fr.oos_metrics.get("cagr", 0.0)

        if abs(t_sharpe) > 1e-10:
            degradation = 1.0 - o_sharpe / t_sharpe
            deg_str = fmt_pct(degradation)
        else:
            deg_str = "N/A"

        print(
            f"  {fr.fold.fold_index:>4} | {fmt_float(t_sharpe):>10} | "
            f"{fmt_float(o_sharpe):>9} | {deg_str:>8} | "
            f"{fmt_pct(t_cagr):>10} | {fmt_pct(o_cagr):>9}"
        )


# ===========================================================================
# PHASE 13: BENCHMARKS
# ===========================================================================


def audit_benchmarks(close: pd.DataFrame, report, config: WalkForwardConfig) -> None:
    print(f"\n{'='*70}")
    print("PHASE 13: BENCHMARKS")
    print(f"{'='*70}")

    bench_returns = compute_benchmark_returns(close, "equal_weight", 60)
    if bench_returns is not None:
        active_bench = bench_returns.iloc[60:]
        if len(active_bench) > 0:
            bench_metrics = compute_metrics(
                returns=active_bench,
                positions=pd.DataFrame({"_": np.ones(len(active_bench))}),
                turnover=pd.Series(0.0, index=active_bench.index),
                costs=pd.Series(0.0, index=active_bench.index),
                ann_factor=config.ann_factor,
            )
            print(f"  Equal-weight benchmark:")
            print(f"    CAGR:        {fmt_pct(bench_metrics['cagr'])}")
            print(f"    Sharpe:      {fmt_float(bench_metrics['sharpe_ratio'])}")
            print(f"    Max DD:      {fmt_pct(bench_metrics['max_drawdown'])}")

    oos_metrics = report.aggregate_metrics
    print(f"  Strategy OOS:")
    print(f"    CAGR:        {fmt_pct(oos_metrics['cagr'])}")
    print(f"    Sharpe:      {fmt_float(oos_metrics['sharpe_ratio'])}")
    print(f"    Max DD:      {fmt_pct(oos_metrics['max_drawdown'])}")

    if bench_returns is not None:
        active_bench = bench_returns.iloc[60:]
        oos_idx = report.stitched_oos_returns.index
        common = oos_idx.intersection(active_bench.index)
        if len(common) > 10:
            strat_r = report.stitched_oos_returns.reindex(common)
            bench_r = active_bench.reindex(common)
            active = (strat_r != 0) | (bench_r != 0)
            excess = strat_r[active] - bench_r[active]
            if len(excess) > 1 and excess.std() > 0:
                info_ratio = excess.mean() / excess.std() * np.sqrt(ANN_FACTOR)
                print(f"    Info ratio:  {fmt_float(info_ratio)}")


# ===========================================================================
# PHASE 14: COST SENSITIVITY
# ===========================================================================


def audit_cost_sensitivity(close: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("PHASE 14: COST SENSITIVITY")
    print(f"{'='*70}")

    config_base = default_config()
    grid = ParameterGrid(
        lookbacks=[[5, 10, 21], [10, 21, 42]],
        vol_window=[21, 40],
        covariance_window=[60],
        covariance_shrinkage=[0.1],
        rebalance_threshold=[0.01],
    )

    folds = generate_folds(len(close), config_base)
    fold = folds[0]
    train_close = close.iloc[fold.train_start_idx:fold.train_end_idx]
    params, _, _ = select_parameters(train_close, grid, config_base)
    full_close = close.iloc[fold.train_start_idx:fold.test_end_idx]

    cost_multipliers = [0.0, 0.5, 1.0, 2.0, 3.0]
    print(f"  Using fold 0 with params: {params}")
    print(f"  {'Mult':>5} | {'OOS CAGR':>9} | {'OOS Shrp':>9} | {'OOS MaxDD':>9} | {'Turnover':>9} | {'CostDrag':>9}")
    print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

    for mult in cost_multipliers:
        cfg = WalkForwardConfig(
            train_window=config_base.train_window,
            test_window=config_base.test_window,
            step=config_base.step,
            minimum_training_bars=config_base.minimum_training_bars,
            minimum_training_observations=0,
            minimum_training_trades=0,
            ann_factor=config_base.ann_factor,
            target_portfolio_vol=config_base.target_portfolio_vol,
            max_gross_leverage=config_base.max_gross_leverage,
            taker_fee_pct=config_base.taker_fee_pct * mult,
            slippage_pct=config_base.slippage_pct * mult,
            allow_short=config_base.allow_short,
            min_history=config_base.min_history,
        )
        result = run_oos_fold(full_close, fold, params, cfg)
        m = result.oos_metrics
        print(
            f"  {mult:>5.1f}x | {fmt_pct(m['cagr']):>9} | "
            f"{fmt_float(m['sharpe_ratio']):>9} | {fmt_pct(m['max_drawdown']):>9} | "
            f"{fmt_float(m['avg_daily_turnover']):>9} | "
            f"{fmt_float(m['total_cost_drag']):>9}"
        )


# ===========================================================================
# PHASE 15: PARAMETER PERTURBATION
# ===========================================================================


def audit_parameter_perturbation(close: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("PHASE 15: PARAMETER PERTURBATION")
    print(f"{'='*70}")

    config = default_config()
    grid_base = ParameterGrid(
        lookbacks=[[10, 21, 42]],
        vol_window=[40],
        covariance_window=[60],
        covariance_shrinkage=[0.1],
        rebalance_threshold=[0.01],
    )

    folds = generate_folds(len(close), config)
    fold = folds[0]
    train_close = close.iloc[fold.train_start_idx:fold.train_end_idx]
    params, base_sharpe, _ = select_parameters(train_close, grid_base, config)
    full_close = close.iloc[fold.train_start_idx:fold.test_end_idx]
    base_result = run_oos_fold(full_close, fold, params, config)

    print(f"  Base params: {params}")
    print(f"  Base OOS Sharpe: {fmt_float(base_result.oos_metrics['sharpe_ratio'])}")

    perturbations = [
        ("vol_window=20", {"vol_window": 20}),
        ("vol_window=60", {"vol_window": 60}),
        ("cov_window=40", {"covariance_window": 40}),
        ("cov_window=120", {"covariance_window": 120}),
        ("shrinkage=0.0", {"covariance_shrinkage": 0.0}),
        ("shrinkage=0.5", {"covariance_shrinkage": 0.5}),
        ("threshold=0.0", {"rebalance_threshold": 0.0}),
        ("threshold=0.01", {"rebalance_threshold": 0.01}),
    ]

    print(f"  {'Perturbation':<22} | {'OOS Sharpe':>10} | {'OOS CAGR':>9} | {'OOS MaxDD':>9}")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}")

    for label, overrides in perturbations:
        perturbed = dict(params)
        perturbed.update(overrides)
        try:
            result = run_oos_fold(full_close, fold, perturbed, config)
            m = result.oos_metrics
            print(
                f"  {label:<22} | {fmt_float(m['sharpe_ratio']):>10} | "
                f"{fmt_pct(m['cagr']):>9} | {fmt_pct(m['max_drawdown']):>9}"
            )
        except Exception as e:
            print(f"  {label:<22} | FAILED: {e}")


# ===========================================================================
# PHASE 16: NULL / RANDOMIZATION TEST
# ===========================================================================


def audit_null_test(close: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("PHASE 16: NULL / RANDOMIZATION TEST")
    print(f"{'='*70}")
    print("  Method: Shuffle daily returns per-asset (destroy serial correlation)")
    print("  while preserving marginal return distribution.")

    config = default_config()
    grid = ParameterGrid(
        lookbacks=[[5, 10, 21], [10, 21, 42]],
        vol_window=[21, 40],
        covariance_window=[60],
        covariance_shrinkage=[0.1],
        rebalance_threshold=[0.01],
    )

    n_trials = 2
    null_sharpes = []

    for trial in range(n_trials):
        np.random.seed(1000 + trial)
        shuffled = close.copy()
        for col in shuffled.columns:
            vals = shuffled[col].values.copy()
            rets = vals[1:] / vals[:-1] - 1
            np.random.shuffle(rets)
            new_prices = np.zeros(len(vals))
            new_prices[0] = vals[0]
            new_prices[1:] = vals[0] * np.cumprod(1 + rets)
            shuffled[col] = new_prices

        try:
            report = run_walk_forward(shuffled, grid, config)
            oos_sharpe = report.aggregate_metrics["sharpe_ratio"]
            null_sharpes.append(oos_sharpe)
            print(f"  Trial {trial}: OOS Sharpe = {fmt_float(oos_sharpe)}")
        except Exception as e:
            print(f"  Trial {trial}: FAILED ({e})")

    if null_sharpes:
        null_mean = np.mean(null_sharpes)
        null_std = np.std(null_sharpes) if len(null_sharpes) > 1 else 0.0
        print(f"  Null mean Sharpe:  {fmt_float(null_mean)}")
        print(f"  Null std Sharpe:   {fmt_float(null_std)}")
        print(f"  Null max Sharpe:   {fmt_float(max(null_sharpes))}")

    real_report = run_walk_forward(close, grid, config)
    real_sharpe = real_report.aggregate_metrics["sharpe_ratio"]
    print(f"  Real OOS Sharpe:   {fmt_float(real_sharpe)}")

    if null_sharpes:
        excess = real_sharpe - null_mean
        print(f"  Excess over null:  {fmt_float(excess)}")
        if null_std > 0:
            t_stat = excess / null_std
            print(f"  t-statistic:       {fmt_float(t_stat)}")
            if t_stat > 2.0:
                print(f"  Verdict: REAL signal appears to exceed noise (t > 2)")
            elif t_stat > 1.0:
                print(f"  Verdict: MARGINAL - some excess over noise (1 < t < 2)")
            else:
                print(f"  Verdict: WEAK - signal does not clearly exceed noise (t < 1)")


# ===========================================================================
# PHASE 17: DEFENSIVE STATISTICS
# ===========================================================================


def audit_defensive_stats(report) -> None:
    print(f"\n{'='*70}")
    print("PHASE 17: DEFENSIVE STATISTICS")
    print(f"{'='*70}")

    m = report.aggregate_metrics
    n_oos = len(report.stitched_oos_returns)
    n_years = n_oos / ANN_FACTOR

    print(f"  OOS observations:      {n_oos}")
    print(f"  OOS years:             {n_years:.2f}")
    print(f"  WFO folds:             {len(report.folds)}")
    print(f"  OOS CAGR:              {fmt_pct(m['cagr'])}")
    print(f"  OOS Sharpe:            {fmt_float(m['sharpe_ratio'])}")
    print(f"  OOS Sortino:           {fmt_float(m['sortino_ratio'])}")
    print(f"  OOS volatility:        {fmt_pct(m['annual_volatility'])}")
    print(f"  OOS max drawdown:      {fmt_pct(m['max_drawdown'])}")
    if m['annual_volatility'] > 1e-10:
        calmar = m['cagr'] / abs(m['max_drawdown']) if abs(m['max_drawdown']) > 1e-10 else 0.0
        print(f"  Calmar ratio:          {fmt_float(calmar)}")
    print(f"  Win rate:              {fmt_pct(m['daily_win_rate'])}")
    print(f"  Avg gross exposure:    {fmt_float(m['avg_gross_exposure'])}")
    print(f"  Avg daily turnover:    {fmt_float(m['avg_daily_turnover'])}")
    print(f"  Total cost drag:       {fmt_float(m['total_cost_drag'])}")
    print(f"  Fee drag %:            {fmt_pct(m['fee_drag_pct'])}")


# ===========================================================================
# PHASE 18: REGIME ANALYSIS
# ===========================================================================


def audit_regime_analysis(report, close: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("PHASE 18: REGIME ANALYSIS")
    print(f"{'='*70}")

    oos_ret = report.stitched_oos_returns
    if len(oos_ret) < 60:
        print("  Insufficient OOS data for regime analysis")
        return

    bench_ret = compute_benchmark_returns(close, "equal_weight", 60)
    if bench_ret is None:
        return

    common = oos_ret.index.intersection(bench_ret.index)
    oos_r = oos_ret.reindex(common)
    bench_r = bench_ret.reindex(common)

    cum_bench = (1 + bench_r).cumprod()
    rolling_ret_30 = cum_bench.pct_change(30)
    rolling_ret_90 = cum_bench.pct_change(90)

    high_vol_mask = rolling_ret_30.abs() > rolling_ret_30.abs().median()
    bull_mask = rolling_ret_90 > 0
    bear_mask = rolling_ret_90 <= 0

    regimes = {
        "All OOS": slice(None),
        "High vol (absolute 30d ret > median)": high_vol_mask,
        "Low vol (absolute 30d ret <= median)": ~high_vol_mask,
        "Bull (90d benchmark > 0)": bull_mask,
        "Bear (90d benchmark <= 0)": bear_mask,
    }

    print(f"  {'Regime':<45} | {'N':>4} | {'Sharpe':>7} | {'CAGR':>8} | {'MaxDD':>8}")
    print(f"  {'-'*45}-+-{'-'*4}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    for label, mask in regimes.items():
        if isinstance(mask, slice):
            subset = oos_r
        else:
            subset = oos_r[mask]
        subset = subset.dropna()
        if len(subset) < 10:
            print(f"  {label:<45} | {'N/A':>4}")
            continue

        ann_ret = subset.mean() * ANN_FACTOR
        ann_vol = subset.std() * np.sqrt(ANN_FACTOR) if len(subset) > 1 else 0.0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        total_ret = (1 + subset).prod() - 1
        n_yrs = len(subset) / ANN_FACTOR
        cagr = (1 + total_ret) ** (1 / n_yrs) - 1 if n_yrs > 0 else 0.0
        cum = (1 + subset).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()

        print(
            f"  {label:<45} | {len(subset):>4} | "
            f"{fmt_float(sharpe):>7} | {fmt_pct(cagr):>8} | {fmt_pct(max_dd):>8}"
        )


# ===========================================================================
# PHASE 19: PORTFOLIO RISK AUDIT
# ===========================================================================


def audit_portfolio_risk(report) -> None:
    print(f"\n{'='*70}")
    print("PHASE 19: PORTFOLIO RISK AUDIT")
    print(f"{'='*70}")

    all_positions = pd.concat([fr.oos_positions for fr in report.folds])
    all_turnover = pd.concat([fr.oos_turnover for fr in report.folds])
    all_costs = pd.concat([fr.oos_costs for fr in report.folds])

    gross_exp = all_positions.abs().sum(axis=1)
    print(f"  Target volatility:     10.00%")

    oos_ret = report.stitched_oos_returns
    if len(oos_ret) > 1:
        realized_vol = oos_ret.std() * np.sqrt(ANN_FACTOR)
        print(f"  Realized volatility:   {fmt_pct(realized_vol)}")
    else:
        realized_vol = 0.0
        print(f"  Realized volatility:   N/A")

    print(f"  Avg gross exposure:    {fmt_float(gross_exp.mean())}")
    print(f"  Max gross exposure:    {fmt_float(gross_exp.max())}")

    cap_binds = (gross_exp >= 0.99).sum()
    pct_cap = cap_binds / len(gross_exp) * 100 if len(gross_exp) > 0 else 0
    print(f"  % days cap binds:      {pct_cap:.1f}%")

    n_active = (all_positions.abs() > 1e-6).sum(axis=1)
    print(f"  Avg active assets:     {fmt_float(n_active.mean())}")
    print(f"  Avg daily turnover:    {fmt_float(all_turnover.mean())}")
    print(f"  Total cost drag:       {fmt_float(all_costs.sum())}")


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    print("=" * 70)
    print("WFO DEEP AUDIT - FULL ANALYSIS")
    print("=" * 70)

    close = make_synthetic_close(n=800, seed=42)
    config = default_config()
    grid = default_grid()

    print(f"\nDataset: {len(close)} bars, {close.columns.tolist()}")
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    folds = generate_folds(len(close), config)
    print(f"WFO folds: {len(folds)}")
    for f in folds:
        print(
            f"  Fold {f.fold_index}: "
            f"train=[{f.train_start_idx},{f.train_end_idx}) "
            f"test=[{f.test_start_idx},{f.test_end_idx})"
        )

    report = run_walk_forward(close, grid, config)

    audit_parameter_optimization(grid)
    audit_multiple_testing(close, grid, config)
    audit_parameter_stability(report)
    audit_degradation(report)
    audit_benchmarks(close, report, config)
    audit_cost_sensitivity(close)
    audit_parameter_perturbation(close)
    audit_null_test(close)
    audit_defensive_stats(report)
    audit_regime_analysis(report, close)
    audit_portfolio_risk(report)

    print(f"\n{'='*70}")
    print("AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
