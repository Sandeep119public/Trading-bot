"""Data Manager page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trendbot.application.data_service import DataService


def render_data_page(data_service: DataService) -> None:
    """Render the Data Manager page."""
    st.header("Data Manager")

    from trendbot.ui.streamlit.components.data_download_form import render_data_download_form
    render_data_download_form(data_service)

    st.divider()
    st.subheader("Downloaded Datasets")
    datasets = data_service.list_datasets()

    if not datasets:
        st.info("No datasets downloaded yet.")
        return

    df = pd.DataFrame([d.model_dump() for d in datasets])
    display_cols = [
        "source", "symbol", "timeframe", "start_date", "end_date", "rows", "last_updated",
    ]
    st.dataframe(df[display_cols], use_container_width=True)

    active_source = st.session_state.data_source
    active_timeframe = st.session_state.get("timeframe", "1d")
    active_datasets = [
        d for d in datasets
        if d.source == active_source and d.timeframe == active_timeframe
    ]
    if not active_datasets:
        return

    st.divider()
    st.subheader("Dataset Preview")
    symbols = sorted({d.symbol for d in active_datasets})
    selected_symbol = st.selectbox("Select asset to preview", options=symbols, key="preview_sym")
    try:
        preview = data_service.load_prices(
            source=active_source,
            symbols=[selected_symbol],
            timeframe=active_timeframe,
            start_date=None,
            end_date=None,
        )
        st.dataframe(preview.head(20))
        st.caption(f"Total rows: {len(preview)}")
    except Exception as e:
        st.error(f"Failed to load preview: {e}")

    st.divider()
    st.subheader("Delete Dataset")
    del_symbol = st.selectbox("Select asset to delete", options=symbols, key="del_sym")
    if st.button("Delete", key="del_btn"):
        data_service.delete_dataset(active_source, del_symbol, active_timeframe)
        st.success(f"Deleted {del_symbol} ({active_source}, {active_timeframe})")
        st.rerun()
