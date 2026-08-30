"""Universe selector component."""

from __future__ import annotations

import streamlit as st

from trendbot.application.data_service import DataService


def render_universe_selector(data_service: DataService) -> list[str]:
    """Render universe selector using only datasets for the active source/timeframe."""
    st.subheader("Select Backtest Universe")

    source = st.session_state.data_source
    timeframe = st.session_state.get("timeframe", "1d")
    datasets = [
        d for d in data_service.list_datasets()
        if d.source == source and d.timeframe == timeframe
    ]

    if not datasets:
        st.info(
            f"No {source} {timeframe} datasets available. "
            "Download data from the Data Manager first."
        )
        return []

    symbols = sorted({d.symbol for d in datasets})
    current_universe = st.session_state.get("universe", [])
    default = [s for s in current_universe if s in symbols]
    selected = st.multiselect(
        "Available Assets",
        options=symbols,
        default=default,
        key="universe_select",
    )
    st.session_state.universe = selected
    if selected:
        st.caption(f"Selected {len(selected)} assets for backtesting")
    return selected
