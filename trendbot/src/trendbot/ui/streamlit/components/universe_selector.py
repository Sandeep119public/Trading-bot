"""Universe selector component."""

from __future__ import annotations

import streamlit as st

from trendbot.application.data_service import DataService


def render_universe_selector(data_service: DataService) -> list[str]:
    """Render universe selector with available datasets.

    Args:
        data_service: DataService instance.

    Returns:
        List of selected ticker symbols.
    """
    st.subheader("Select Backtest Universe")

    datasets = data_service.list_datasets()

    if not datasets:
        st.info("No datasets available. Download data first.")
        return []

    symbols = list({d.symbol for d in datasets})
    symbols.sort()

    selected = st.multiselect(
        "Available Assets",
        options=symbols,
        default=[s for s in st.session_state.universe if s in symbols],
        key="universe_select",
    )

    st.session_state.universe = selected

    if selected:
        st.caption(f"Selected {len(selected)} assets for backtesting")

    return selected
