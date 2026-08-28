"""Main Streamlit application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports (must be before trendbot imports)
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st

from trendbot.application.backtest_service import BacktestService
from trendbot.application.data_service import DataService
from trendbot.infrastructure.data_providers.yfinance_provider import YFinanceProvider
from trendbot.infrastructure.repositories.parquet_price_repository import (
    ParquetPriceRepository,
)
from trendbot.ui.streamlit.components.strategy_params_form import (
    render_strategy_params,
)
from trendbot.ui.streamlit.components.universe_selector import render_universe_selector
from trendbot.ui.streamlit.pages.backtest_page import render_backtest_page
from trendbot.ui.streamlit.pages.data_page import render_data_page
from trendbot.ui.streamlit.pages.diagnostics_page import render_diagnostics_page
from trendbot.ui.streamlit.pages.results_page import render_results_page
from trendbot.ui.streamlit.state import init_state, reset_defaults

st.set_page_config(
    page_title="TrendBot - Trend Following Backtester",
    page_icon="📈",
    layout="wide",
)

init_state()

# Initialize infrastructure
DATA_DIR = Path(__file__).parent.parent.parent / "data"
provider = YFinanceProvider()
repository = ParquetPriceRepository(DATA_DIR)
data_service = DataService(provider, repository)
backtest_service = BacktestService(repository)

# Sidebar
with st.sidebar:
    st.title("📈 TrendBot")
    st.caption("Multi-Horizon Trend Following Backtester")

    st.session_state.data_source = st.selectbox(
        "Data Source",
        options=["yfinance"],
        key="sidebar_source",
    )

    st.session_state.start_date = st.text_input(
        "Start Date",
        value=st.session_state.start_date,
        key="sidebar_start",
    )

    end_val = st.session_state.end_date if st.session_state.end_date else ""
    st.session_state.end_date = st.text_input(
        "End Date (blank for latest)",
        value=end_val,
        key="sidebar_end",
    ) or None

    st.divider()
    st.caption(f"Output dir: {Path('output').absolute()}")

    if st.button("Reset to Defaults", key="reset_btn"):
        reset_defaults()
        st.rerun()

# Main content - Tabs
tab_data, tab_strategy, tab_backtest, tab_results, tab_diag = st.tabs([
    "Data Manager",
    "Strategy Parameters",
    "Backtest",
    "Results",
    "Diagnostics",
])

with tab_data:
    render_data_page(data_service)

with tab_strategy:
    render_strategy_params()

selected_symbols = render_universe_selector(data_service)

with tab_backtest:
    render_backtest_page(data_service, backtest_service, selected_symbols)

with tab_results:
    render_results_page()

with tab_diag:
    render_diagnostics_page(data_service, selected_symbols)
