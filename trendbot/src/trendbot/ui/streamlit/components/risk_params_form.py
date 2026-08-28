"""Risk parameters form component."""

from __future__ import annotations

import streamlit as st


def render_risk_params() -> None:
    """Render risk parameters configuration."""
    st.subheader("Risk Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.target_portfolio_vol = st.number_input(
            "Target Portfolio Volatility",
            min_value=0.01,
            max_value=2.0,
            value=st.session_state.target_portfolio_vol,
            step=0.01,
            format="%.2f",
            key="risk_target_vol",
        )

    with col2:
        st.session_state.max_gross_leverage = st.number_input(
            "Max Gross Leverage",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.max_gross_leverage,
            step=0.1,
            format="%.1f",
            key="risk_max_lev",
        )
