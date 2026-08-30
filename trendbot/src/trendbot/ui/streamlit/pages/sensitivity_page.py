"""Parameter Sensitivity Analysis page."""

from __future__ import annotations

import streamlit as st

from trendbot.application.backtest_service import BacktestService
from trendbot.application.data_service import DataService
from trendbot.domain.sensitivity import compute_plateau_metrics, run_sensitivity_analysis


def render_sensitivity_page(
    data_service: DataService,
    backtest_service: BacktestService,
    selected_symbols: list[str],
) -> None:
    """Render the Parameter Sensitivity Analysis page.

    Args:
        data_service: DataService instance.
        backtest_service: BacktestService instance.
        selected_symbols: Currently selected universe symbols.
    """
    st.header("Parameter Sensitivity Analysis")
    st.caption("Detect overfitting by testing parameter robustness across a grid of values.")

    if not selected_symbols:
        st.warning("Please select assets in the Universe Selector first.")
        return

    if st.session_state.backtest_result is None:
        st.info("Run a backtest first to establish baseline parameters.")
        return

    _render_configuration_section(selected_symbols)
    _render_heatmap_section(selected_symbols, data_service)


def _render_configuration_section(selected_symbols: list[str]) -> None:
    """Render the parameter range configuration UI."""
    st.subheader("Parameter Ranges")

    base_lookbacks = st.session_state.lookbacks
    st.markdown(f"**Base Lookbacks:** `{base_lookbacks}`")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Lookback Multipliers**")
        st.caption("Scales base lookbacks (e.g., 0.5x = half speed)")
        lb_min = st.number_input("Min multiplier", 0.1, 2.0, 0.5, 0.1, key="lb_min")
        lb_max = st.number_input("Max multiplier", 0.5, 4.0, 2.0, 0.1, key="lb_max")
        lb_steps = st.number_input("Steps", 2, 10, 5, 1, key="lb_steps")

    with col2:
        st.markdown("**Volatility Windows**")
        st.caption("Rolling window for asset vol estimation")
        vol_default = "14, 21, 30, 42, 63"
        vol_input = st.text_input(
            "Windows (comma-separated)",
            value=vol_default,
            key="vol_windows_input",
        )
        vol_windows = [int(v.strip()) for v in vol_input.split(",") if v.strip().isdigit()]

    with col3:
        st.markdown("**Target Portfolio Vol**")
        st.caption("Annualized target volatility")
        tv_default = "0.05, 0.10, 0.15"
        tv_input = st.text_input(
            "Target vols (comma-separated)",
            value=tv_default,
            key="target_vols_input",
        )
        target_vols = [float(v.strip()) for v in tv_input.split(",") if v.strip()]

    # Preview the grid
    lookback_multipliers = [
        round(lb_min + (lb_max - lb_min) * i / max(lb_steps - 1, 1), 2)
        for i in range(lb_steps)
    ]
    total_combos = len(lookback_multipliers) * len(vol_windows) * len(target_vols)

    st.info(
        f"**Grid size:** {len(lookback_multipliers)} lookback configs x "
        f"{len(vol_windows)} vol windows x {len(target_vols)} target vols "
        f"= **{total_combos} backtests**"
    )

    st.session_state["sensitivity_config"] = {
        "lookback_multipliers": lookback_multipliers,
        "vol_windows": vol_windows,
        "target_vols": target_vols,
    }


def _render_heatmap_section(
    selected_symbols: list[str],
    data_service: DataService,
) -> None:
    """Render the heatmap results section."""
    st.divider()

    if st.button("Run Sensitivity Analysis", type="primary", key="run_sensitivity_btn"):
        _run_analysis(selected_symbols, data_service)

    if (
        "sensitivity_results" in st.session_state
        and st.session_state.sensitivity_results is not None
    ):
        _render_results()


def _run_analysis(selected_symbols: list[str], data_service: DataService) -> None:
    """Execute the sensitivity analysis."""
    config = st.session_state.get("sensitivity_config", {})
    lookback_multipliers = config.get("lookback_multipliers", [0.5, 1.0, 2.0])
    vol_windows = config.get("vol_windows", [14, 21, 30, 42, 63])
    target_vols = config.get("target_vols", [0.05, 0.10, 0.15])

    from trendbot.ui.streamlit.state import get_backtest_request

    request = get_backtest_request(selected_symbols)

    try:
        close = data_service.load_prices(
            source=request.data_selection.source,
            symbols=selected_symbols,
            timeframe=request.data_selection.timeframe,
            start_date=request.data_selection.start_date,
            end_date=request.data_selection.end_date,
        )
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if close.empty:
        st.error("No price data available for the selected universe.")
        return

    n_backtests = (
        len(lookback_multipliers) * len(vol_windows) * len(target_vols)
    )
    with st.spinner(f"Running {n_backtests} backtests..."):
        results_df = run_sensitivity_analysis(
            close=close,
            base_lookbacks=request.momentum.lookbacks,
            base_vol_window=request.volatility.vol_window,
            base_target_vol=request.risk.target_portfolio_vol,
            base_ann_factor=request.volatility.ann_factor,
            base_max_leverage=request.risk.max_gross_leverage,
            base_taker_fee_pct=request.execution.taker_fee_pct,
            base_slippage_pct=request.execution.slippage_pct,
            base_rebalance_threshold=request.execution.rebalance_threshold,
            base_min_history=request.backtest.min_history,
            allow_short=request.momentum.allow_short,
            lookback_multipliers=lookback_multipliers,
            vol_windows=vol_windows,
            target_vols=target_vols,
        )

    st.session_state.sensitivity_results = results_df
    st.rerun()


def _render_results() -> None:
    """Render the sensitivity analysis results with heatmaps."""
    results_df = st.session_state.sensitivity_results

    if results_df.empty:
        st.warning("No results returned from sensitivity analysis.")
        return

    plateau = compute_plateau_metrics(results_df)

    # Plateau summary
    st.subheader("Robustness Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Sharpe", f"{plateau['max_sharpe']:.3f}")
    with col2:
        st.metric("Plateau Coverage", f"{plateau['plateau_pct']:.1f}%")
    with col3:
        status = "Robust" if plateau["plateau_found"] else "Overfitted"
        st.metric("Diagnosis", status)

    if plateau["plateau_found"]:
        st.success(
            f"**Robust parameter region found.** "
            f"{plateau['plateau_pct']:.1f}% of parameter combinations achieve "
            f"Sharpe >= {plateau['threshold_sharpe']:.3f} (80% of max). "
            f"Lookback configs in plateau: {', '.join(plateau['plateau_configs'][:5])}"
        )
    else:
        st.warning(
            f"**Possible overfitting detected.** Only {plateau['plateau_pct']:.1f}% "
            f"of parameter combinations achieve Sharpe >= {plateau['threshold_sharpe']:.3f}. "
            f"Consider simplifying the strategy."
        )

    st.divider()

    # Heatmaps
    st.subheader("Sharpe Ratio Heatmaps")

    _render_heatmaps(results_df)

    # Full results table
    st.divider()
    st.subheader("Full Results Table")
    st.dataframe(
        results_df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )


def _render_heatmaps(results_df) -> None:
    """Render Plotly heatmaps for each target volatility level."""
    import plotly.graph_objects as go

    target_vols = sorted(results_df["target_vol"].unique())

    for tgt_vol in target_vols:
        subset = results_df[results_df["target_vol"] == tgt_vol]

        pivot = subset.pivot_table(
            index="vol_window",
            columns="lookback_config",
            values="sharpe_ratio",
            aggfunc="first",
        )

        if pivot.empty:
            continue

        # Sort axes logically
        pivot = pivot.sort_index(ascending=True)
        try:
            pivot = pivot[sorted(pivot.columns, key=lambda x: eval(x))]
        except Exception:
            pivot = pivot[sorted(pivot.columns)]

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[f"LB {col}" for col in pivot.columns],
                y=[f"Vol {int(v)}" for v in pivot.index],
                colorscale="RdYlGn",
                text=[[f"{v:.2f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Sharpe"),
                hovertemplate=(
                    "Lookback: %{x}<br>"
                    "Vol Window: %{y}<br>"
                    "Sharpe: %{z:.3f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"Sharpe Ratio | Target Vol = {tgt_vol:.0%}",
            xaxis_title="Lookback Configuration",
            yaxis_title="Volatility Window",
            height=400,
            margin=dict(l=60, r=30, t=50, b=60),
        )

        st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{tgt_vol}")

    # CAGR heatmap
    st.subheader("CAGR Heatmaps")
    for tgt_vol in target_vols:
        subset = results_df[results_df["target_vol"] == tgt_vol]

        pivot = subset.pivot_table(
            index="vol_window",
            columns="lookback_config",
            values="cagr",
            aggfunc="first",
        )

        if pivot.empty:
            continue

        pivot = pivot.sort_index(ascending=True)
        try:
            pivot = pivot[sorted(pivot.columns, key=lambda x: eval(x))]
        except Exception:
            pivot = pivot[sorted(pivot.columns)]

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[f"LB {col}" for col in pivot.columns],
                y=[f"Vol {int(v)}" for v in pivot.index],
                colorscale="RdYlGn",
                text=[[f"{v:.1%}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="CAGR"),
                hovertemplate=(
                    "Lookback: %{x}<br>"
                    "Vol Window: %{y}<br>"
                    "CAGR: %{z:.2%}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"CAGR | Target Vol = {tgt_vol:.0%}",
            xaxis_title="Lookback Configuration",
            yaxis_title="Volatility Window",
            height=400,
            margin=dict(l=60, r=30, t=50, b=60),
        )

        st.plotly_chart(fig, use_container_width=True, key=f"cagr_heatmap_{tgt_vol}")
