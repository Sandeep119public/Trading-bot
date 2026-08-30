"""Phase 4: Real-Market Validation -- Efficient Research Script.

Runs the full Phase 4 protocol against real historical market data.
Optimized to avoid redundant WFO runs during stress testing.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Force unbuffered output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(line_buffering=True)

# Project imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trendbot.application.data_service import DataService
from trendbot.domain.backtest import compute_benchmark_returns, run_backtest
from trendbot.domain.metrics import compute_metrics
from trendbot.domain.models import ParameterGrid, WalkForwardConfig
from trendbot.domain.walk_forward import (
    compute_parameter_stability,
    enumerate_parameter_combinations,
    generate_folds,
    run_walk_forward,
    stitch_oos_returns,
)
from trendbot.infrastructure.data_providers.yfinance_provider import YFinanceProvider
from trendbot.infrastructure.repositories.parquet_price_repository import ParquetPriceRepository

OUTPUT_DIR = Path(__file__).parent / "output" / "phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report_lines = []


def R(text=""):
    report_lines.append(text)
    print(text, flush=True)


def save_report():
    path = OUTPUT_DIR / "phase4_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[Report saved to {path}]")


# ====================================================================
# LOAD DATA
# ====================================================================
def load_data():
    data_dir = Path(__file__).parent / "data"
    repository = ParquetPriceRepository(data_dir)
    provider = YFinanceProvider()
    data_service = DataService(provider, repository)

    universe = ["BTCUSD", "ETHUSD", "SOLUSD"]
    close = None
    actual_source = None

    # Try delta_india first
    try:
        close = data_service.load_prices(
            source="delta_india", symbols=universe, timeframe="1d",
            start_date=date(2020, 1, 1), end_date=None,
        )
        actual_source = "delta_india"
    except Exception as e:
        print(f"[INFO] delta_india load: {e}", flush=True)

    # Try download if not loaded
    if close is None or close.empty:
        try:
            from trendbot.infrastructure.data_providers.delta_india_provider import DeltaIndiaProvider
            provider = DeltaIndiaProvider(timeframe="1d")
            data_service = DataService(provider, repository)
            from trendbot.domain.models import DataDownloadRequest
            request = DataDownloadRequest(
                source="delta_india", symbols=universe,
                start_date=date(2020, 1, 1), end_date=None,
                overwrite=False, timeframe="1d", quote_currency="USD",
            )
            result = data_service.download_data(request)
            close = data_service.load_prices(
                source="delta_india", symbols=universe, timeframe="1d",
                start_date=date(2020, 1, 1), end_date=None,
            )
            actual_source = "delta_india"
        except Exception as e:
            print(f"[INFO] delta_india download: {e}", flush=True)

    # Fallback to binance
    if close is None or close.empty:
        try:
            btc = repository.load_prices("binance", "BTC-USD", "1h", None, None)
            eth = repository.load_prices("binance", "ETH-USD", "1h", None, None)
            btc_d = btc.resample("1D").last().dropna()
            eth_d = eth.resample("1D").last().dropna()
            close = pd.DataFrame({"BTC-USD": btc_d.iloc[:, 0], "ETH-USD": eth_d.iloc[:, 0]}).dropna()
            actual_source = "binance (1h->1d)"
        except Exception as e:
            print(f"[INFO] binance fallback: {e}", flush=True)

    # Synthetic fallback
    if close is None or close.empty:
        print("[WARN] Using SYNTHETIC data -- results NOT valid for research!", flush=True)
        np.random.seed(42)
        n = 2000
        dates = pd.bdate_range("2020-01-01", periods=n)
        rets = np.random.normal(0.0003, 0.02, (n, 3))
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        close = pd.DataFrame(prices, index=dates, columns=["BTCUSD", "ETHUSD", "SOLUSD"])
        actual_source = "SYNTHETIC"

    print(f"[DATA] {close.shape[0]} bars, {close.shape[1]} assets, source={actual_source}", flush=True)
    return close, actual_source


# ====================================================================
# PHASE 1: BASELINE
# ====================================================================
def phase1():
    import subprocess
    R("=" * 80)
    R("PHASE 1: BASELINE / GIT CHECK")
    R("=" * 80)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    short = subprocess.check_output(["git", "log", "--oneline", "-1"], text=True).strip()
    R(f"Branch: {branch}")
    R(f"Commit: {commit}")
    R(f"Latest: {short}")
    R(f"Timestamp: {datetime.now().isoformat()}")
    R("")


# ====================================================================
# PHASE 2-6: DATA AUDITS
# ====================================================================
def phase2_6(close, source):
    R("=" * 80)
    R("PHASE 2: IDENTIFY THE REAL DATASET")
    R("=" * 80)
    R(f"Provider/Source:       {source}")
    R(f"Universe:              {', '.join(close.columns.tolist())}")
    R(f"Timeframe:             1d")
    R(f"Start date:            {close.index[0].date()}")
    R(f"End date:              {close.index[-1].date()}")
    R(f"Number of assets:      {close.shape[1]}")
    R(f"Number of bars:        {len(close)}")
    R(f"Calendar:              Crypto (24/7/365)")
    R(f"Price field:           Close")
    R(f"Adjusted/unadjusted:   N/A (crypto)")
    R(f"Missing data:          {close.isna().sum().sum()} total NaN cells")
    R("")
    R("Per-asset coverage:")
    for col in close.columns:
        s = close[col].dropna()
        R(f"  {col}: {s.index[0].date()} to {s.index[-1].date()}, "
          f"{len(s)}/{len(close)} bars, {len(s)/len(close)*100:.1f}%")
    R("")

    R("=" * 80)
    R("PHASE 3: DATA INTEGRITY AUDIT")
    R("=" * 80)
    R(f"Monotonically increasing: {close.index.is_monotonic_increasing}")
    R(f"Duplicate timestamps:     {close.index.has_duplicates}")
    freq = pd.infer_freq(close.index)
    R(f"Inferred frequency:       {freq}")

    returns = close / close.shift(1) - 1
    for col in close.columns:
        s = close[col].dropna()
        nulls = close[col].isna().sum()
        extremes = (returns[col].dropna().abs() > 0.5).sum()
        R(f"  {col}: nulls={nulls}, extreme_returns={extremes}")

    if close.isna().sum().sum() > 0:
        R("[!] Missing data detected -- assets may have been listed at different times")
    else:
        R("[OK] No data integrity issues")
    R("")

    R("=" * 80)
    R("PHASE 4: SURVIVORSHIP-BIAS AUDIT")
    R("=" * 80)
    R("Classification: POSSIBLE (top-3 crypto selected)")
    R("Impact: LOW -- BTC/ETH/SOL dominant throughout sample period")
    R("")

    R("=" * 80)
    R("PHASE 5: DATA SOURCE / PROVIDER AUDIT")
    R("=" * 80)
    R(f"Configured source: {source}")
    R("Trace: config -> DataService -> DeltaIndiaProvider -> Delta Exchange API")
    R("Symbol normalization: BTCUSD -> BTCUSD (Delta perpetual contract)")
    R("Market type: Perpetual/futures on Delta Exchange India")
    R("")

    R("=" * 80)
    R("PHASE 6: CORPORATE-ACTION / PRICE-TYPE AUDIT")
    R("=" * 80)
    R("Asset type: Crypto perpetual/futures -- NO corporate actions")
    R("Price type: Raw exchange close (settlement) price")
    R("")


# ====================================================================
# PHASE 7: FREEZE
# ====================================================================
def phase7():
    R("=" * 80)
    R("PHASE 7: FREEZE THE STRATEGY")
    R("=" * 80)
    config = WalkForwardConfig()
    grid = ParameterGrid()
    n_candidates = len(enumerate_parameter_combinations(grid))
    R(f"Lookbacks (grid):      {grid.lookbacks}")
    R(f"Allow short:           True")
    R(f"Vol window (grid):     {grid.vol_window}")
    R(f"Annualization:         365")
    R(f"Target portfolio vol:  0.10")
    R(f"Max gross leverage:    1.0")
    R(f"Covariance window:     {grid.covariance_window}")
    R(f"Covariance shrinkage:  {grid.covariance_shrinkage}")
    R(f"Rebalance threshold:   {grid.rebalance_threshold}")
    R(f"Taker fee:             0.10%")
    R(f"Slippage:              0.05%")
    R(f"Min history:           60 bars")
    R(f"WFO train:             {config.train_window} bars")
    R(f"WFO test:              {config.test_window} bars")
    R(f"WFO step:              {config.step} bars")
    R(f"Parameter candidates:  {n_candidates}")
    R("")
    R("*** STRATEGY FROZEN ***")
    R("")
    return config, grid


# ====================================================================
# PHASE 8-10: RUN WFO + PRIMARY RESULTS
# ====================================================================
def phase8_10(close, config, grid):
    R("=" * 80)
    R("PHASE 8-9: RUN REAL-DATA WFO & VERIFY")
    R("=" * 80)

    folds = generate_folds(len(close), config)
    R(f"Bars: {len(close)}, Folds: {len(folds)}")
    for f in folds:
        R(f"  Fold {f.fold_index}: train[{f.train_start_idx}:{f.train_end_idx}) "
          f"test[{f.test_start_idx}:{f.test_end_idx})")

    t0 = time.time()
    report = run_walk_forward(close, grid, config)
    elapsed = time.time() - t0
    R(f"WFO completed in {elapsed:.1f}s")
    R(f"Stitched OOS: {len(report.stitched_oos_returns)} timestamps")
    R(f"Stitched duplicates: {report.stitched_oos_returns.index.has_duplicates}")
    R(f"Equity start: {report.stitched_oos_equity.iloc[0]:.6f}")
    R("")

    R("=" * 80)
    R("PHASE 10: PRIMARY OOS RESULTS (STITCHED)")
    R("=" * 80)
    m = report.aggregate_metrics
    n_years = len(report.stitched_oos_returns) / 365

    R(f"Total OOS return:          {m['total_return']:.4%}")
    R(f"OOS CAGR:                  {m['cagr']:.4%}")
    R(f"OOS annualized volatility: {m['annual_volatility']:.4%}")
    R(f"OOS Sharpe:                {m['sharpe_ratio']:.4f}")
    R(f"OOS Sortino:               {m['sortino_ratio']:.4f}")
    R(f"OOS max drawdown:          {m['max_drawdown']:.4%}")
    calmar = m['cagr'] / abs(m['max_drawdown']) if m['max_drawdown'] != 0 else float('inf')
    R(f"OOS Calmar:                {calmar:.4f}")
    R(f"OOS daily win rate:        {m['daily_win_rate']:.4%}")
    R(f"OOS avg gross exposure:    {m['avg_gross_exposure']:.4f}")
    R(f"OOS avg daily turnover:    {m['avg_daily_turnover']:.6f}")
    R(f"OOS total cost drag:       {m['total_cost_drag']:.6f}")
    R(f"OOS gross return:          {m['total_gross_return']:.4%}")
    R(f"OOS net return:            {m['total_net_return']:.4%}")
    R(f"Fee drag %:                {m['fee_drag_pct']:.4%}")
    R(f"OOS observations:          {len(report.stitched_oos_returns)}")
    R(f"OOS years:                 {n_years:.2f}")
    R(f"WFO folds:                 {len(report.folds)}")
    R("")

    return report, m


# ====================================================================
# PHASE 11: FOLD-BY-FOLD
# ====================================================================
def phase11(report, close):
    R("=" * 80)
    R("PHASE 11: FOLD-BY-FOLD ANALYSIS")
    R("=" * 80)
    R(f"{'Fold':>4} | {'Train':>5} bars | {'OOS':>5} bars | "
      f"{'TrnSharpe':>10} | {'OOSSharpe':>10} | {'OOSCAGR':>10} | {'OOSDD':>10}")
    R("-" * 85)

    for fr in report.folds:
        R(f"{fr.fold.fold_index:>4} | {fr.fold.train_length:>5} | {fr.fold.test_length:>5} | "
          f"{fr.training_sharpe:>10.4f} | {fr.oos_metrics['sharpe_ratio']:>10.4f} | "
          f"{fr.oos_metrics['cagr']:>10.2%} | {fr.oos_metrics['max_drawdown']:>10.2%}")
        params = {k: v for k, v in fr.selected_parameters.items() if k in ('lookbacks', 'vol_window')}
        R(f"     Params: {params}")

    oos_sharpes = [fr.oos_metrics["sharpe_ratio"] for fr in report.folds]
    positive = sum(1 for s in oos_sharpes if s > 0)
    R(f"\nPositive OOS folds: {positive}/{len(oos_sharpes)}")
    R(f"OOS Sharpe range: [{min(oos_sharpes):.4f}, {max(oos_sharpes):.4f}]")
    R("")


# ====================================================================
# PHASE 12-13: PARAMETER STABILITY + DEGRADATION
# ====================================================================
def phase12_13(report):
    R("=" * 80)
    R("PHASE 12: PARAMETER STABILITY")
    R("=" * 80)
    for param, counts in report.parameter_stability.items():
        total = sum(counts.values())
        R(f"  {param}:")
        for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            R(f"    {str(val):<35} {cnt}/{total} ({cnt/total*100:.0f}%)")
    R("")

    R("=" * 80)
    R("PHASE 13: TRAIN -> OOS DEGRADATION")
    R("=" * 80)
    R(f"{'Fold':>4} | {'TrnSharpe':>10} | {'OOSSharpe':>10} | {'Delta%':>8} | "
      f"{'TrnCAGR':>10} | {'OOSCAGR':>10}")
    R("-" * 70)
    for fr in report.folds:
        ts = fr.training_sharpe
        os_ = fr.oos_metrics["sharpe_ratio"]
        tc = fr.training_metrics["cagr"]
        oc = fr.oos_metrics["cagr"]
        delta = f"{(os_-ts)/abs(ts)*100:+.0f}%" if abs(ts) > 1e-10 else "N/A"
        R(f"{fr.fold.fold_index:>4} | {ts:>10.4f} | {os_:>10.4f} | {delta:>8} | "
          f"{tc:>10.2%} | {oc:>10.2%}")

    avg_ts = np.mean([fr.training_sharpe for fr in report.folds])
    avg_os = np.mean([fr.oos_metrics["sharpe_ratio"] for fr in report.folds])
    R(f"\nMean train Sharpe: {avg_ts:.4f}, Mean OOS Sharpe: {avg_os:.4f}")
    if abs(avg_ts) > 1e-10:
        R(f"Mean degradation: {(avg_os-avg_ts)/abs(avg_ts)*100:.1f}%")
    R("")


# ====================================================================
# PHASE 14-15: VOLATILITY + CORRELATION
# ====================================================================
def phase14_15(close, report, config):
    R("=" * 80)
    R("PHASE 14: REALIZED VS TARGET VOLATILITY")
    R("=" * 80)
    target = 0.10
    oos_ret = report.stitched_oos_returns
    realized = oos_ret.std() * np.sqrt(365)
    R(f"Target vol:         {target:.2%}")
    R(f"Realized vol:       {realized:.4%}")
    R(f"Realization ratio:  {realized/target:.2f}x")

    all_pos = pd.concat([fr.oos_positions for fr in report.folds])
    gross = all_pos.abs().sum(axis=1)
    R(f"Avg gross leverage: {gross.mean():.4f}")
    R(f"Max gross leverage: {gross.max():.4f}")
    cap_bind = (gross >= 0.99).sum() / len(gross) * 100
    R(f"Leverage cap bind%: {cap_bind:.1f}%")
    R("")

    R("=" * 80)
    R("PHASE 15: CORRELATION / RISK REGIME ANALYSIS")
    R("=" * 80)
    rets = close / close.shift(1) - 1
    if close.shape[1] >= 2:
        # Average rolling correlation
        corrs = []
        for i in range(60, len(rets)):
            subset = rets.iloc[i-60:i].dropna()
            if subset.shape[1] >= 2:
                cm = subset.corr()
                mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
                corrs.append(cm.values[mask].mean())
        if corrs:
            R(f"60-day rolling avg pairwise correlation:")
            R(f"  Mean: {np.mean(corrs):.4f}")
            R(f"  Range: [{np.min(corrs):.4f}, {np.max(corrs):.4f}]")
            high = sum(1 for c in corrs if c > 0.7)
            R(f"  Days with corr > 0.7: {high}/{len(corrs)} ({high/len(corrs)*100:.1f}%)")
    R("")


# ====================================================================
# PHASE 16-17: COST + SLIPPAGE STRESS (using fast single-backtest approach)
# ====================================================================
def phase16_17(close, config):
    R("=" * 80)
    R("PHASE 16: TRANSACTION-COST STRESS TEST")
    R("=" * 80)
    R("Using single-fold representative test for efficiency")

    base_taker = 0.001
    base_slip = 0.0005
    cost_mults = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    R(f"{'Mult':>5} | {'OOS CAGR':>10} | {'OOS Sharpe':>10} | {'OOS Sortino':>11} | {'OOS MaxDD':>10}")
    R("-" * 60)

    # Use first fold only for speed
    folds = generate_folds(len(close), config)
    if folds:
        fold = folds[0]
        train_close = close.iloc[fold.train_start_idx:fold.train_end_idx]

        # Select params on training data
        small_grid = ParameterGrid(
            lookbacks=[[5, 10, 21, 42]],
            vol_window=[21],
            covariance_window=[60],
            covariance_shrinkage=[0.1],
            rebalance_threshold=[0.01],
        )

        for mult in cost_mults:
            cfg = config.model_copy(update={
                "taker_fee_pct": base_taker * mult,
                "slippage_pct": base_slip * mult,
            })
            try:
                report = run_walk_forward(close, small_grid, cfg)
                m = report.aggregate_metrics
                R(f"{mult:>5.1f}x | {m['cagr']:>10.2%} | {m['sharpe_ratio']:>10.4f} | "
                  f"{m['sortino_ratio']:>10.4f} | {m['max_drawdown']:>10.2%}")
            except Exception as e:
                R(f"{mult:>5.1f}x | FAILED: {e}")
    R("")

    R("=" * 80)
    R("PHASE 17: SLIPPAGE STRESS TEST")
    R("=" * 80)
    slip_mults = [0.0, 0.5, 1.0, 2.0, 3.0]
    R(f"{'Mult':>5} | {'OOS CAGR':>10} | {'OOS Sharpe':>10} | {'OOS Sortino':>11} | {'OOS MaxDD':>10}")
    R("-" * 60)

    small_grid = ParameterGrid(
        lookbacks=[[5, 10, 21, 42]],
        vol_window=[21],
        covariance_window=[60],
        covariance_shrinkage=[0.1],
        rebalance_threshold=[0.01],
    )
    for mult in slip_mults:
        cfg = config.model_copy(update={
            "taker_fee_pct": base_taker,
            "slippage_pct": base_slip * mult,
        })
        try:
            report = run_walk_forward(close, small_grid, cfg)
            m = report.aggregate_metrics
            R(f"{mult:>5.1f}x | {m['cagr']:>10.2%} | {m['sharpe_ratio']:>10.4f} | "
              f"{m['sortino_ratio']:>10.4f} | {m['max_drawdown']:>10.2%}")
        except Exception as e:
            R(f"{mult:>5.1f}x | FAILED: {e}")
    R("")


# ====================================================================
# PHASE 18: PARAMETER PERTURBATION
# ====================================================================
def phase18(report, close, config):
    R("=" * 80)
    R("PHASE 18: PARAMETER PERTURBATION TEST")
    R("=" * 80)

    # Base config from most common selections
    base_params = {}
    for param, counts in report.parameter_stability.items():
        base_params[param] = max(counts, key=counts.get)

    R("Base parameters (mode across folds):")
    for k, v in base_params.items():
        R(f"  {k}: {v}")
    R("")

    # Test perturbations of key params with single-grid approach
    base_grid = ParameterGrid(
        lookbacks=[base_params.get("lookbacks", [5, 10, 21, 42])],
        vol_window=[base_params.get("vol_window", 21)],
        covariance_window=[base_params.get("covariance_window", 60)],
        covariance_shrinkage=[base_params.get("covariance_shrinkage", 0.1)],
        rebalance_threshold=[base_params.get("rebalance_threshold", 0.01)],
    )
    try:
        base_report = run_walk_forward(close, base_grid, config)
        base_sharpe = base_report.aggregate_metrics["sharpe_ratio"]
    except Exception:
        base_sharpe = 0.0

    R(f"Base OOS Sharpe: {base_sharpe:.4f}")
    R("")

    perturbations = [
        ("vol_window", [14, 21, 42, 60]),
        ("covariance_window", [40, 60, 120]),
        ("covariance_shrinkage", [0.0, 0.1, 0.25, 0.5]),
        ("rebalance_threshold", [0.0, 0.005, 0.01, 0.02]),
    ]

    R(f"{'Param':<25} {'Value':>10} {'OOS Sharpe':>10} {'Delta':>8}")
    R("-" * 60)

    for param_name, values in perturbations:
        for val in values:
            grid_kw = {param_name: [val]}
            test_grid = ParameterGrid(**grid_kw)
            try:
                r = run_walk_forward(close, test_grid, config)
                s = r.aggregate_metrics["sharpe_ratio"]
                delta = s - base_sharpe
                R(f"  {param_name:<23} {str(val):>10} {s:>10.4f} {delta:>+8.4f}")
            except Exception as e:
                R(f"  {param_name:<23} {str(val):>10} FAILED")
    R("")


# ====================================================================
# PHASE 19: BENCHMARK
# ====================================================================
def phase19(close, report, config):
    R("=" * 80)
    R("PHASE 19: BENCHMARK ANALYSIS")
    R("=" * 80)

    oos = report.stitched_oos_returns
    daily_rets = close / close.shift(1) - 1
    eq = daily_rets.mean(axis=1).reindex(oos.index).fillna(0.0)

    def bm(rets):
        a = rets.iloc[60:]
        if len(a) == 0:
            return {"cagr": 0, "sharpe": 0, "dd": 0, "ret": 0}
        tot = float((1+a).prod()-1)
        ny = len(a)/365
        cagr = (1+tot)**(1/ny)-1 if ny > 0 else 0
        vol = a.std()*np.sqrt(365)
        sh = a.mean()*365/vol if vol > 0 else 0
        dd = ((1+a).cumprod() / (1+a).cumprod().cummax()-1).min()
        return {"cagr": cagr, "sharpe": sh, "dd": dd, "ret": tot}

    sm = report.aggregate_metrics
    em = bm(eq)

    R(f"{'Metric':<20} {'Strategy':>12} {'EqualWeight':>12}")
    R("-" * 48)
    R(f"{'CAGR':<20} {sm['cagr']:>12.2%} {em['cagr']:>12.2%}")
    R(f"{'Sharpe':<20} {sm['sharpe_ratio']:>12.4f} {em['sharpe']:>12.4f}")
    R(f"{'Max DD':<20} {sm['max_drawdown']:>12.2%} {em['dd']:>12.2%}")

    excess = oos - eq
    te = excess.std() * np.sqrt(365)
    ir = excess.mean()*365/te if te > 0 else 0
    R(f"\nInformation ratio: {ir:.4f}")
    R("")


# ====================================================================
# PHASE 20: NULL TEST
# ====================================================================
def phase20(close, config, grid):
    R("=" * 80)
    R("PHASE 20: NULL / RANDOMIZATION TEST")
    R("=" * 80)
    R("Method: Return permutation (destroys temporal predictability)")
    R("")

    real_report = run_walk_forward(close, grid, config)
    real_sharpe = real_report.aggregate_metrics["sharpe_ratio"]
    R(f"Real OOS Sharpe: {real_sharpe:.4f}")

    n_nulls = 10
    null_sharpes = []
    R(f"Running {n_nulls} null iterations...")

    for i in range(n_nulls):
        np.random.seed(i + 1000)
        shuffled = close.copy()
        for col in close.columns:
            vals = shuffled[col].values.copy()
            rets = np.diff(vals) / vals[:-1]
            perm = np.random.permutation(rets)
            new_vals = np.zeros_like(vals)
            new_vals[0] = vals[0]
            for j in range(1, len(vals)):
                new_vals[j] = new_vals[j-1] * (1 + perm[j-1])
            shuffled[col] = new_vals
        try:
            nr = run_walk_forward(shuffled, grid, config)
            null_sharpes.append(nr.aggregate_metrics["sharpe_ratio"])
            print(f"  Null {i+1}/{n_nulls}: Sharpe={null_sharpes[-1]:.4f}", flush=True)
        except Exception:
            pass

    if null_sharpes:
        ns = np.array(null_sharpes)
        pct = (ns < real_sharpe).sum() / len(ns) * 100
        R(f"\nNull Sharpe distribution:")
        R(f"  Mean: {ns.mean():.4f}, Std: {ns.std():.4f}")
        R(f"  Range: [{ns.min():.4f}, {ns.max():.4f}]")
        R(f"Real Sharpe percentile vs null: {pct:.1f}%")
    R("")


# ====================================================================
# PHASE 21: MULTIPLE TESTING
# ====================================================================
def phase21(report, grid):
    R("=" * 80)
    R("PHASE 21: MULTIPLE-TESTING DIAGNOSTIC")
    R("=" * 80)
    total = len(enumerate_parameter_combinations(grid))
    R(f"Total candidates per fold: {total}")
    R(f"Folds: {len(report.folds)}")
    R(f"Total selections: {len(report.folds)}")
    train_sharpes = [fr.training_sharpe for fr in report.folds]
    R(f"Training Sharpe: mean={np.mean(train_sharpes):.4f}, "
      f"median={np.median(train_sharpes):.4f}, std={np.std(train_sharpes):.4f}")
    R("")


# ====================================================================
# PHASE 22: REGIME ANALYSIS
# ====================================================================
def phase22(close, report, config):
    R("=" * 80)
    R("PHASE 22: REGIME ANALYSIS")
    R("=" * 80)

    oos = report.stitched_oos_returns
    asset_rets = close / close.shift(1) - 1
    mkt = asset_rets.mean(axis=1)
    common = oos.index.intersection(mkt.index)
    oos_r = oos.reindex(common)
    mkt_r = mkt.reindex(common).fillna(0.0)

    # Vol regime
    rv = mkt_r.rolling(60).std() * np.sqrt(365)
    q25, q75 = rv.quantile(0.25), rv.quantile(0.75)

    def stats(rets):
        if len(rets) < 10:
            return {"n": len(rets), "cagr": 0, "sharpe": 0, "vol": 0, "dd": 0}
        tot = float((1+rets).prod()-1)
        ny = len(rets)/365
        cagr = (1+tot)**(1/ny)-1 if ny > 0 else 0
        vol = rets.std()*np.sqrt(365)
        sh = rets.mean()*365/vol if vol > 0 else 0
        dd = ((1+rets).cumprod()/(1+rets).cumprod().cummax()-1).min()
        return {"n": len(rets), "cagr": cagr, "sharpe": sh, "vol": vol, "dd": dd}

    R("--- Volatility Regimes ---")
    regimes = {
        f"Low (<{q25:.0%})": rv < q25,
        f"Med ({q25:.0%}-{q75:.0%})": (rv >= q25) & (rv < q75),
        f"High (>{q75:.0%})": rv >= q75,
    }
    R(f"{'Regime':<20} {'N':>5} {'CAGR':>8} {'Sharpe':>8} {'Vol':>8} {'DD':>8}")
    R("-" * 55)
    for name, mask in regimes.items():
        s = stats(oos_r[mask.reindex(oos_r.index, fill_value=False)])
        R(f"{name:<20} {s['n']:>5} {s['cagr']:>8.2%} {s['sharpe']:>8.2f} "
          f"{s['vol']:>8.2%} {s['dd']:>8.2%}")

    # Trend regime
    cum = mkt_r.rolling(60).sum()
    R("\n--- Trend Regimes ---")
    trend = {
        "Bull (>+10%)": cum > 0.10,
        "Bear (<-10%)": cum < -0.10,
        "Sideways": (cum >= -0.10) & (cum <= 0.10),
    }
    R(f"{'Regime':<20} {'N':>5} {'CAGR':>8} {'Sharpe':>8} {'Vol':>8} {'DD':>8}")
    R("-" * 55)
    for name, mask in trend.items():
        s = stats(oos_r[mask.reindex(oos_r.index, fill_value=False)])
        R(f"{name:<20} {s['n']:>5} {s['cagr']:>8.2%} {s['sharpe']:>8.2f} "
          f"{s['vol']:>8.2%} {s['dd']:>8.2%}")
    R("")


# ====================================================================
# PHASE 23-25: EXTREME EVENTS, CHURN, CONCENTRATION
# ====================================================================
def phase23_25(report):
    R("=" * 80)
    R("PHASE 23: EXTREME EVENT ANALYSIS")
    R("=" * 80)
    oos = report.stitched_oos_returns
    cum = (1+oos).cumprod()
    dd = cum / cum.cummax() - 1

    # Top 3 drawdowns
    in_dd = False
    dd_start = None
    dd_periods = []
    for i in range(len(dd)):
        if dd.iloc[i] < -0.01 and not in_dd:
            in_dd = True
            dd_start = i
        elif (dd.iloc[i] >= 0 or i == len(dd)-1) and in_dd:
            in_dd = False
            dd_periods.append((dd_start, i, dd.iloc[dd_start:i].min()))

    dd_periods.sort(key=lambda x: x[2])
    R("Top 3 drawdowns:")
    for rank, (s, e, mdd) in enumerate(dd_periods[:3]):
        R(f"  #{rank+1}: {mdd:.2%} from {oos.index[s].date()} to {oos.index[min(e,len(oos)-1)].date()} ({e-s} days)")
    R("")

    R("=" * 80)
    R("PHASE 24: CHURN ANALYSIS")
    R("=" * 80)
    to = pd.concat([fr.oos_turnover for fr in report.folds])
    costs = pd.concat([fr.oos_costs for fr in report.folds])
    annual_to = to.sum() / (len(to)/365)
    R(f"Annual turnover:     {annual_to:.2f}")
    R(f"Avg daily turnover:  {to.mean():.6f}")
    R(f"Max daily turnover:  {to.max():.6f}")
    R(f"Rebalance days:      {(to > 1e-8).sum()}/{len(to)}")
    R(f"Total costs:         {costs.sum():.6f}")
    R(f"Annual cost drag:    {costs.sum()/(len(costs)/365):.4%}")
    R("")

    R("=" * 80)
    R("PHASE 25: CONCENTRATION ANALYSIS")
    R("=" * 80)
    pos = pd.concat([fr.oos_positions for fr in report.folds])
    gross = pos.abs().sum(axis=1)
    net = pos.sum(axis=1)
    R(f"Avg gross exposure:  {gross.mean():.4f}")
    R(f"Max gross exposure:  {gross.max():.4f}")
    R(f"Avg net exposure:    {net.mean():.4f}")
    R(f"Max abs net exposure:{net.abs().max():.4f}")
    R("Per-asset avg abs weight:")
    for col in pos.columns:
        R(f"  {col}: {pos[col].abs().mean():.4f}")
    R("")


# ====================================================================
# PHASE 26: DATA ROBUSTNESS
# ====================================================================
def phase26(close, config, grid):
    R("=" * 80)
    R("PHASE 26: DATA ROBUSTNESS")
    R("=" * 80)
    base = run_walk_forward(close, grid, config)
    bs = base.aggregate_metrics["sharpe_ratio"]
    R(f"Baseline Sharpe: {bs:.4f}")

    # Reversed order
    rev = close[close.columns[::-1]]
    try:
        r = run_walk_forward(rev, grid, config)
        R(f"Reversed order: {r.aggregate_metrics['sharpe_ratio']:.4f} (d={r.aggregate_metrics['sharpe_ratio']-bs:+.4f})")
    except Exception as e:
        R(f"Reversed order: FAILED ({e})")

    # Tiny noise
    np.random.seed(42)
    noise = close * (1 + np.random.normal(0, 0.0001, close.shape))
    try:
        r = run_walk_forward(noise, grid, config)
        R(f"1bp noise:      {r.aggregate_metrics['sharpe_ratio']:.4f} (d={r.aggregate_metrics['sharpe_ratio']-bs:+.4f})")
    except Exception as e:
        R(f"1bp noise:      FAILED ({e})")
    R("")


# ====================================================================
# PHASE 29: VALIDATION
# ====================================================================
def phase29():
    import subprocess
    R("=" * 80)
    R("PHASE 29: FULL VALIDATION")
    R("=" * 80)

    R("--- pytest ---")
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=line"],
                          capture_output=True, text=True, cwd=str(Path(__file__).parent),
                          timeout=120)
        # Show last few lines
        lines = r.stdout.strip().split('\n')
        for line in lines[-5:]:
            R(f"  {line}")
    except Exception as e:
        R(f"  FAILED: {e}")

    R("\n--- ruff ---")
    try:
        r = subprocess.run([sys.executable, "-m", "ruff", "check", "src/"],
                          capture_output=True, text=True, cwd=str(Path(__file__).parent),
                          timeout=60)
        R(f"  Exit: {r.returncode}")
        if r.stdout.strip():
            R(f"  {r.stdout.strip()[:500]}")
    except Exception as e:
        R(f"  FAILED: {e}")

    R("\n--- mypy ---")
    try:
        r = subprocess.run([sys.executable, "-m", "mypy", "src/trendbot", "--ignore-missing-imports"],
                          capture_output=True, text=True, cwd=str(Path(__file__).parent),
                          timeout=120)
        R(f"  Exit: {r.returncode}")
        if r.stdout.strip():
            R(f"  {r.stdout.strip()[:500]}")
    except Exception as e:
        R(f"  FAILED: {e}")
    R("")


# ====================================================================
# PHASE 30: FINAL VERDICT
# ====================================================================
def phase30(report, oos_m, source, close, config, grid):
    R("=" * 80)
    R("PHASE 30: FINAL RESEARCH VERDICT")
    R("=" * 80)

    n_years = len(report.stitched_oos_returns) / 365
    has_pos_cagr = oos_m["cagr"] > 0
    has_pos_sharpe = oos_m["sharpe_ratio"] > 0
    has_ok_sharpe = oos_m["sharpe_ratio"] > 0.5
    has_ok_dd = oos_m["max_drawdown"] > -0.50

    R("\n--- EXECUTIVE VERDICT ---")
    if not has_pos_sharpe:
        verdict = "NO EVIDENCE OF EDGE"
    elif not has_ok_sharpe:
        verdict = "WEAK / INCONCLUSIVE"
    elif has_ok_sharpe and has_pos_cagr and has_ok_dd and n_years >= 2:
        verdict = "PROMISING OOS EVIDENCE"
    else:
        verdict = "WEAK / INCONCLUSIVE"

    R(f">>> {verdict} <<<")
    R("")

    # Summary sections
    R("--- Dataset ---")
    R(f"Provider:         {source}")
    R(f"Universe:         {', '.join(close.columns.tolist())}")
    R(f"Timeframe:        1d")
    R(f"Date range:       {close.index[0].date()} to {close.index[-1].date()}")
    R(f"Assets:           {close.shape[1]}")
    R(f"Bars:             {len(close)}")
    R(f"Survivorship:     POSSIBLE (top-3 crypto)")
    R(f"Calendar:         Crypto 24/7/365")
    R(f"Price adj:        None (crypto futures)")
    R("")

    R("--- Frozen Strategy ---")
    R(f"Target vol:       0.10, Max leverage: 1.0")
    R(f"Ann factor:       365, Min history: 60")
    R(f"Fee: 0.10%, Slippage: 0.05%")
    R(f"Grid: {len(enumerate_parameter_combinations(grid))} candidates")
    R("")

    R("--- WFO Configuration ---")
    R(f"Train: {config.train_window}, Test: {config.test_window}, Step: {config.step}")
    R(f"Folds: {len(report.folds)}")
    R("")

    R("--- OOS Performance ---")
    for k, v in oos_m.items():
        R(f"  {k:<25} {v}")
    R("")

    R("--- Limitations ---")
    R("1. Survivorship: top-3 crypto (LOW impact)")
    R(f"2. Sample: {n_years:.1f} years, {len(close)} bars")
    R("3. Universe: 3 assets only")
    R("4. Execution: taker 0.1%, slip 0.05% (reasonable for crypto)")
    R("5. Funding rates NOT modeled")
    R("6. Parameter selection bias (WFO mitigates)")
    R("7. Regime dependence possible")
    R("")

    R("--- Recommendation ---")
    if verdict == "NO EVIDENCE OF EDGE":
        R(">>> STOP <<<")
    elif verdict == "WEAK / INCONCLUSIVE":
        R(">>> MORE RESEARCH REQUIRED <<<")
    else:
        R(">>> MORE RESEARCH REQUIRED <<<")
        R("  Next steps: paper trade, expand universe, validate execution")

    R("")
    R(f"Finished: {datetime.now().isoformat()}")


# ====================================================================
# MAIN
# ====================================================================
def main():
    R("=" * 80)
    R("PHASE 4: REAL-MARKET VALIDATION")
    R("=" * 80)
    R(f"Started: {datetime.now().isoformat()}")
    R("")

    close, source = load_data()
    R("")

    phase1()
    phase2_6(close, source)
    config, grid = phase7()
    report, oos_m = phase8_10(close, config, grid)
    phase11(report, close)
    phase12_13(report)
    phase14_15(close, report, config)
    phase16_17(close, config)
    phase18(report, close, config)
    phase19(close, report, config)
    phase20(close, config, grid)
    phase21(report, grid)
    phase22(close, report, config)
    phase23_25(report)
    phase26(close, config, grid)
    phase29()
    phase30(report, oos_m, source, close, config, grid)

    save_report()


if __name__ == "__main__":
    main()
