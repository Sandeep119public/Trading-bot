"""Backtest page."""

from __future__ import annotations

import streamlit as st

from trendbot.application.backtest_service import BacktestService
from trendbot.application.data_service import DataService


def render_backtest_page(
    data_service: DataService,
    backtest_service: BacktestService,
    selected_symbols: list[str],
) -> None:
    """Render the Backtest configuration and execution page.

    Args:
        data_service: DataService instance.
        backtest_service: BacktestService instance.
        selected_symbols: Currently selected universe symbols.
    """
    st.header("Backtest")

    if not selected_symbols:
        st.warning("Please select assets in the Universe Selector first.")
        return

    st.subheader("Configuration Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Universe:** {', '.join(selected_symbols)}")
        st.markdown(f"**Source:** {st.session_state.data_source}")
        date_range = (
            f"{st.session_state.start_date} to "
            f"{st.session_state.end_date or 'latest'}"
        )
        st.markdown(f"**Date Range:** {date_range}")
    with col2:
        st.markdown(f"**Lookbacks:** {st.session_state.lookbacks}")
        st.markdown(f"**Allow Short:** {st.session_state.allow_short}")
        st.markdown(f"**Target Vol:** {st.session_state.target_portfolio_vol}")
        st.markdown(f"**Max Leverage:** {st.session_state.max_gross_leverage}")

    st.divider()

    st.subheader("Backtest Parameters")
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.min_history = st.number_input(
            "Minimum History (bars)",
            min_value=0,
            max_value=500,
            value=st.session_state.min_history,
            key="bt_min_hist",
        )
    with col4:
        st.session_state.benchmark = st.selectbox(
            "Benchmark",
            options=["equal_weight", "none"],
            index=0 if st.session_state.benchmark == "equal_weight" else 1,
            key="bt_benchmark",
        )

    st.divider()

    if st.button("Run Backtest", type="primary", key="run_bt_btn"):
        from trendbot.ui.streamlit.state import get_backtest_request
        request = get_backtest_request(selected_symbols)

        with st.spinner("Running backtest..."):
            try:
                dto = backtest_service.run(request)
            except ValueError as e:
                st.error("Data integrity check failed:")
                st.warning(str(e))
                return
            except Exception as e:
                st.error(f"Backtest failed unexpectedly: {e}")
                return

        if dto.error:
            # Check if it's a data integrity error (multi-line with dash bullets)
            if "Data integrity check failed" in dto.error:
                st.error("Data Integrity Check Failed")
                st.warning(dto.error)
            else:
                st.error(f"Backtest failed: {dto.error}")
        else:
            st.session_state.backtest_result = dto.result
            st.success("Backtest completed!")
            st.rerun()
