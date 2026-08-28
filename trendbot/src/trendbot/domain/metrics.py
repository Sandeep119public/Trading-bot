"""Performance metrics calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    positions: pd.DataFrame,
    turnover: pd.Series,
    costs: pd.Series,
    ann_factor: int = 365,
) -> dict[str, float]:
    """Compute comprehensive backtest performance metrics.

    Args:
        returns: Strategy daily returns.
        positions: Position weights over time.
        turnover: Daily turnover.
        costs: Daily trading costs.
        ann_factor: Annualization factor.

    Returns:
        Dictionary of performance metrics.
    """
    if len(returns) == 0:
        return _empty_metrics()

    active = returns.iloc[1:] if len(returns) > 1 else returns.iloc[:0]
    if len(active) == 0:
        return _empty_metrics()

    total_return = (1 + active).prod() - 1
    n_years = len(active) / ann_factor
    cagr = (1 + total_return) ** (1 / max(n_years, 1e-10)) - 1

    ann_vol = active.std() * np.sqrt(ann_factor)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    downside = active[active < 0]
    downside_vol = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else 0.0
    sortino = cagr / downside_vol if downside_vol > 0 else 0.0

    cumulative = (1 + active).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()

    win_rate = (active > 0).sum() / len(active) if len(active) > 0 else 0.0

    gross_exposure = positions.abs().sum(axis=1)
    avg_gross = gross_exposure.mean()

    avg_turnover = turnover.mean()
    total_cost_drag = costs.sum()

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "daily_win_rate": float(win_rate),
        "avg_gross_exposure": float(avg_gross),
        "avg_daily_turnover": float(avg_turnover),
        "total_cost_drag": float(total_cost_drag),
    }


def _empty_metrics() -> dict[str, float]:
    """Return empty metrics dictionary."""
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "annual_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "daily_win_rate": 0.0,
        "avg_gross_exposure": 0.0,
        "avg_daily_turnover": 0.0,
        "total_cost_drag": 0.0,
    }


def compute_monthly_returns(returns: pd.Series, ann_factor: int = 365) -> pd.DataFrame:
    """Compute monthly return heatmap data.

    Args:
        returns: Daily strategy returns.
        ann_factor: Annualization factor.

    Returns:
        DataFrame with years as rows, months as columns.
    """
    df = returns.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.month
    monthly = df.groupby(["year", "month"])["return"].apply(lambda x: (1 + x).prod() - 1)
    monthly = monthly.unstack(level="month")
    monthly.columns = [f"M{int(m):02d}" for m in monthly.columns]
    return monthly


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown time series.

    Args:
        returns: Daily strategy returns.

    Returns:
        Series of drawdown values.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return cumulative / running_max - 1
