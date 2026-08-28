"""Results panel component for displaying backtest results."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from trendbot.reporting.charts import (
    drawdown_chart,
    equity_curve_chart,
    exposure_chart,
    positions_heatmap,
    turnover_chart,
)
from trendbot.reporting.exports import (
    export_executed_weights,
    export_positions,
    export_returns,
    export_stats,
    export_turnover,
)


def render_results_panel(result: object) -> None:
    """Render the complete results panel.

    Args:
        result: BacktestResult with all output data.
    """
    st.subheader("Performance Metrics")

    stats = result.stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{stats['total_return']:.2%}")
        st.metric("CAGR", f"{stats['cagr']:.2%}")
    with col2:
        st.metric("Annual Volatility", f"{stats['annual_volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
    with col3:
        st.metric("Sortino Ratio", f"{stats['sortino_ratio']:.2f}")
        st.metric("Max Drawdown", f"{stats['max_drawdown']:.2%}")
    with col4:
        st.metric("Win Rate", f"{stats['daily_win_rate']:.2%}")
        st.metric("Avg Gross Exposure", f"{stats['avg_gross_exposure']:.2f}")

    col5, col6 = st.columns(2)
    with col5:
        st.metric("Avg Daily Turnover", f"{stats['avg_daily_turnover']:.4f}")
    with col6:
        st.metric("Total Cost Drag", f"{stats['total_cost_drag']:.4f}")

    st.divider()

    st.subheader("Charts")
    returns = result.returns
    benchmark = result.benchmark_returns

    st.plotly_chart(equity_curve_chart(returns, benchmark), use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(drawdown_chart(returns), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(turnover_chart(result.turnover), use_container_width=True)

    st.plotly_chart(exposure_chart(result.positions), use_container_width=True)

    if result.positions.shape[1] <= 20:
        st.plotly_chart(positions_heatmap(result.positions), use_container_width=True)

    st.divider()

    st.subheader("Latest Positions")
    last_pos = result.positions.iloc[-1]
    last_pos = last_pos[last_pos.abs() > 0.001]
    if not last_pos.empty:
        st.dataframe(last_pos.sort_values(ascending=False).to_frame("weight"))
    else:
        st.info("No active positions at end of period")

    st.divider()

    st.subheader("Export Results")
    output_dir = Path("output")

    if st.button("Export All Results"):
        export_returns(result.returns, output_dir)
        export_positions(result.positions, output_dir)
        export_executed_weights(result.executed_weights, output_dir)
        export_turnover(result.turnover, output_dir)
        export_stats(result.stats, output_dir)
        st.success(f"Exported to {output_dir}/")

    csv_returns = result.returns.to_frame("return").to_csv()
    st.download_button("Download returns.csv", csv_returns, "returns.csv", "text/csv")

    csv_pos = result.positions.to_csv()
    st.download_button("Download positions.csv", csv_pos, "positions.csv", "text/csv")

    stats_json = json.dumps(result.stats, indent=2)
    st.download_button("Download stats.json", stats_json, "stats.json", "application/json")
