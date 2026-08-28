"""Execution parameters form component."""

from __future__ import annotations

import streamlit as st


def render_execution_params() -> None:
    """Render execution parameters configuration."""
    st.subheader("Execution Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.fee_bps = st.number_input(
            "Fee (bps)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.fee_bps,
            step=0.5,
            key="exec_fee",
        )

    with col2:
        st.session_state.slippage_bps = st.number_input(
            "Slippage (bps)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.slippage_bps,
            step=0.5,
            key="exec_slip",
        )

    with col3:
        st.session_state.rebalance_threshold = st.number_input(
            "Rebalance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rebalance_threshold,
            step=0.005,
            format="%.3f",
            key="exec_thresh",
        )
