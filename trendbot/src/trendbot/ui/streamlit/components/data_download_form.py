"""Data download form component."""

from __future__ import annotations

from datetime import date

import streamlit as st

from trendbot.application.data_service import DataService
from trendbot.domain.models import DataDownloadRequest


def render_data_download_form(data_service: DataService) -> None:
    """Render the data download form.

    Args:
        data_service: DataService instance for data operations.
    """
    st.subheader("Download New Data")

    col1, col2 = st.columns(2)

    with col1:
        ticker_input = st.text_input(
            "Ticker Symbols (comma-separated)",
            value="BTC-USD, ETH-USD",
            key="ticker_input",
        )
        symbols = [s.strip().upper() for s in ticker_input.split(",") if s.strip()]

    with col2:
        start = st.date_input("Start Date", key="dl_start_date")
        end = st.date_input(
            "End Date (blank for latest)", value=None, key="dl_end_date"
        )

    overwrite = st.checkbox("Overwrite existing data", key="dl_overwrite")

    if st.button("Download Data", key="download_btn"):
        if not symbols:
            st.error("Please enter at least one ticker symbol.")
            return

        with st.spinner(f"Downloading {len(symbols)} symbols..."):
            end_date = (
                end
                if end is None or isinstance(end, date)
                else date.fromisoformat(end)
            )
            start_date = (
                start
                if isinstance(start, date)
                else date.fromisoformat(start)
            )
            request = DataDownloadRequest(
                source=st.session_state.data_source,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                overwrite=overwrite,
            )
            result = data_service.download_data(request)

        if result.success:
            st.success(result.message)
        else:
            st.warning(result.message)

        if result.symbols_failed:
            for fail in result.symbols_failed:
                st.error(fail)
