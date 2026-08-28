"""Strategy parameters form component."""

from __future__ import annotations

import streamlit as st

PRESETS = {
    "Original-style trend": {
        "lookbacks": [5, 10, 21, 42],
        "allow_short": True,
        "vol_window": 21,
        "ann_factor": 365,
        "target_portfolio_vol": 0.10,
        "max_gross_leverage": 1.0,
    },
    "Slow trend": {
        "lookbacks": [21, 42, 63, 126],
        "allow_short": True,
        "vol_window": 42,
        "ann_factor": 365,
        "target_portfolio_vol": 0.08,
        "max_gross_leverage": 0.8,
    },
    "Fast trend": {
        "lookbacks": [5, 10, 15, 21],
        "allow_short": True,
        "vol_window": 14,
        "ann_factor": 365,
        "target_portfolio_vol": 0.15,
        "max_gross_leverage": 1.5,
    },
    "Long-only crypto": {
        "lookbacks": [10, 21, 42],
        "allow_short": False,
        "vol_window": 21,
        "ann_factor": 365,
        "target_portfolio_vol": 0.12,
        "max_gross_leverage": 1.0,
    },
    "Conservative": {
        "lookbacks": [21, 42, 63],
        "allow_short": False,
        "vol_window": 42,
        "ann_factor": 252,
        "target_portfolio_vol": 0.05,
        "max_gross_leverage": 0.5,
    },
    "Aggressive": {
        "lookbacks": [5, 10, 15, 21],
        "allow_short": True,
        "vol_window": 14,
        "ann_factor": 365,
        "target_portfolio_vol": 0.20,
        "max_gross_leverage": 2.0,
    },
}


def render_strategy_params() -> None:
    """Render strategy parameters configuration form."""
    st.subheader("Strategy Parameters")

    st.markdown("**Presets**")
    cols = st.columns(len(PRESETS))
    for i, (name, params) in enumerate(PRESETS.items()):
        if cols[i].button(name, key=f"preset_{i}"):
            for k, v in params.items():
                st.session_state[k] = v
            st.rerun()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Momentum**")
        lb_str = st.text_input(
            "Lookbacks (comma-separated)",
            value=", ".join(str(x) for x in st.session_state.lookbacks),
            key="lb_input",
        )
        try:
            st.session_state.lookbacks = [int(x.strip()) for x in lb_str.split(",") if x.strip()]
        except ValueError:
            st.error("Lookbacks must be comma-separated integers")

        st.session_state.allow_short = st.checkbox(
            "Allow Short Positions",
            value=st.session_state.allow_short,
            key="allow_short_cb",
        )

    with col2:
        st.markdown("**Volatility**")
        st.session_state.vol_window = st.number_input(
            "Volatility Window",
            min_value=5,
            max_value=252,
            value=st.session_state.vol_window,
            key="vol_window_input",
        )
        st.session_state.ann_factor = st.selectbox(
            "Annualization Factor",
            options=[252, 365],
            index=0 if st.session_state.ann_factor == 252 else 1,
            key="ann_factor_select",
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Risk**")
        st.session_state.target_portfolio_vol = st.number_input(
            "Target Portfolio Volatility",
            min_value=0.01,
            max_value=2.0,
            value=st.session_state.target_portfolio_vol,
            step=0.01,
            format="%.2f",
            key="target_vol_input",
        )
        st.session_state.max_gross_leverage = st.number_input(
            "Max Gross Leverage",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.max_gross_leverage,
            step=0.1,
            format="%.1f",
            key="max_lev_input",
        )

    with col4:
        st.markdown("**Execution**")
        st.session_state.fee_bps = st.number_input(
            "Fee (bps)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.fee_bps,
            step=0.5,
            key="fee_input",
        )
        st.session_state.slippage_bps = st.number_input(
            "Slippage (bps)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.slippage_bps,
            step=0.5,
            key="slip_input",
        )
        st.session_state.rebalance_threshold = st.number_input(
            "Rebalance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.rebalance_threshold,
            step=0.005,
            format="%.3f",
            key="thresh_input",
        )
