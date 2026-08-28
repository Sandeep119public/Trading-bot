"""Diagnostics page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trendbot.application.data_service import DataService


def render_diagnostics_page(data_service: DataService, selected_symbols: list[str]) -> None:
    """Render the Diagnostics page.

    Args:
        data_service: DataService instance.
        selected_symbols: Currently selected universe symbols.
    """
    st.header("Diagnostics")

    if not selected_symbols:
        st.info("Select assets to view diagnostics.")
        return

    st.subheader("Data Quality")

    try:
        close = data_service.load_prices(
            source=st.session_state.data_source,
            symbols=selected_symbols,
            timeframe="1d",
            start_date=None,
            end_date=None,
        )
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Bars", len(close))
    with col2:
        st.metric("Assets", close.shape[1])
    with col3:
        st.metric("Missing %", f"{close.isna().mean().mean():.2%}")

    st.subheader("Per-Asset Summary")
    asset_info = []
    for sym in close.columns:
        s = close[sym].dropna()
        asset_info.append({
            "symbol": sym,
            "start": s.index[0].date() if len(s) > 0 else None,
            "end": s.index[-1].date() if len(s) > 0 else None,
            "rows": len(s),
            "missing_pct": f"{close[sym].isna().mean():.2%}",
        })
    st.dataframe(pd.DataFrame(asset_info), use_container_width=True)

    result = st.session_state.get("backtest_result")
    if result is not None:
        st.divider()
        st.subheader("Signal Summary")
        returns = result.returns
        active_days = (returns != 0).sum()
        st.metric("Active Trading Days", int(active_days))
        st.metric("Total Days", len(returns))

        st.subheader("Position Summary")
        pos = result.positions
        st.metric("Avg Assets Held", f"{(pos.abs() > 0.001).sum(axis=1).mean():.1f}")

        st.subheader("Turnover Summary")
        turnover = result.turnover
        st.metric("Avg Daily Turnover", f"{turnover.mean():.4f}")
        st.metric("Max Daily Turnover", f"{turnover.max():.4f}")

        st.subheader("Cost Summary")
        costs = result.costs
        st.metric("Total Costs", f"{costs.sum():.6f}")
        st.metric("Avg Daily Cost", f"{costs.mean():.6f}")
    else:
        st.info("Run a backtest to see signal and position diagnostics.")
