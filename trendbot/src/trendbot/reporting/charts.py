"""Plotly chart generation for backtest results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from trendbot.domain.metrics import compute_drawdown_series


def equity_curve_chart(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "Equity Curve",
) -> go.Figure:
    """Create equity curve chart with optional benchmark overlay.

    Args:
        returns: Strategy daily returns.
        benchmark_returns: Optional benchmark daily returns.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    equity = (1 + returns).cumprod()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, name="Strategy", mode="lines")
    )

    if benchmark_returns is not None:
        bench_eq = (1 + benchmark_returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=bench_eq.index, y=bench_eq.values, name="Benchmark", mode="lines"
            )
        )

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Cumulative Return")
    return fig


def drawdown_chart(returns: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Create drawdown chart.

    Args:
        returns: Strategy daily returns.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    dd = compute_drawdown_series(returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def exposure_chart(positions: pd.DataFrame, title: str = "Exposure") -> go.Figure:
    """Create gross/long/short exposure chart.

    Args:
        positions: Position weights DataFrame.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    gross = positions.abs().sum(axis=1)
    long_exp = positions.clip(lower=0).sum(axis=1)
    short_exp = positions.clip(upper=0).abs().sum(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gross.index, y=gross.values, name="Gross", mode="lines"))
    fig.add_trace(go.Scatter(x=long_exp.index, y=long_exp.values, name="Long", mode="lines"))
    fig.add_trace(go.Scatter(x=short_exp.index, y=short_exp.values, name="Short", mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Exposure")
    return fig


def turnover_chart(turnover: pd.Series, title: str = "Daily Turnover") -> go.Figure:
    """Create turnover chart.

    Args:
        turnover: Daily turnover Series.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=turnover.index, y=turnover.values, mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Turnover")
    return fig


def positions_heatmap(
    positions: pd.DataFrame,
    title: str = "Position Weights",
) -> go.Figure:
    """Create position weights heatmap.

    Args:
        positions: Position weights DataFrame.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(data=go.Heatmap(
        z=positions.values.T,
        x=positions.index,
        y=positions.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Asset")
    return fig
