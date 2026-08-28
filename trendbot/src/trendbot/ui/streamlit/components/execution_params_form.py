"""Execution parameters form component."""

from __future__ import annotations

import streamlit as st


def render_execution_params() -> None:
    """Render execution parameters configuration."""
    st.subheader("Execution Parameters")

    if st.button("Binance Spot Default", key="binance_preset"):
        st.session_state.taker_fee_pct = 0.001
        st.session_state.maker_fee_pct = 0.0005
        st.session_state.slippage_pct = 0.0005

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.session_state.taker_fee_pct = st.number_input(
            "Taker Fee (%)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.taker_fee_pct * 100,
            step=0.01,
            format="%.2f",
            key="exec_taker_fee",
        ) / 100.0

    with col2:
        st.session_state.maker_fee_pct = st.number_input(
            "Maker Fee (%)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.maker_fee_pct * 100,
            step=0.01,
            format="%.2f",
            key="exec_maker_fee",
        ) / 100.0

    with col3:
        st.session_state.slippage_pct = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.slippage_pct * 100,
            step=0.01,
            format="%.2f",
            key="exec_slippage",
        ) / 100.0

    with col4:
        st.session_state.rebalance_threshold = st.number_input(
            "Rebalance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rebalance_threshold,
            step=0.005,
            format="%.3f",
            key="exec_thresh",
        )
