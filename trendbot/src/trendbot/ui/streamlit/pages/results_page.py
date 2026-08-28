"""Results page."""

from __future__ import annotations

import streamlit as st

from trendbot.ui.streamlit.components.results_panel import render_results_panel


def render_results_page() -> None:
    """Render the Results page."""
    st.header("Results")

    result = st.session_state.get("backtest_result")

    if result is None:
        st.info("No backtest results yet. Run a backtest first.")
        return

    render_results_panel(result)
