"""Data download form component."""

from __future__ import annotations

from datetime import date

import streamlit as st

from trendbot.application.data_service import DataService
from trendbot.domain.models import DataDownloadRequest

SOURCE_LABELS = {
    "binance": "Binance Spot",
    "binance_futures": "Binance USD-M Futures",
    "delta_india": "Delta Exchange India Futures",
    "yfinance": "Yahoo Finance",
}

PRESETS = {
    "binance_futures": "BTCUSDT, ETHUSDT, SOLUSDT",
    "delta_india": "BTCUSD, ETHUSD, SOLUSD",
    "binance": "BTC, ETH, SOL",
    "yfinance": "BTC-USD, ETH-USD",
}


def render_data_download_form(data_service: DataService) -> None:
    """Render the data download form."""
    st.subheader("Download New Data")

    col1, col2 = st.columns(2)
    with col1:
        source_keys = list(SOURCE_LABELS)
        current = st.session_state.get("data_source", "binance_futures")
        source_index = source_keys.index(current) if current in source_keys else 0
        exchange = st.selectbox(
            "Market / Data Source",
            options=source_keys,
            index=source_index,
            format_func=lambda x: SOURCE_LABELS[x],
            key="dl_exchange",
        )
        st.session_state.data_source = exchange

        if exchange in {"binance", "binance_futures"}:
            st.session_state.quote_currency = st.selectbox(
                "Quote Currency",
                options=["USDT", "USDC", "BTC", "ETH"],
                index=0,
                key="dl_quote_currency",
            )

        default_symbols = PRESETS[exchange]
        ticker_input = st.text_input(
            "Contract Symbols (comma-separated)",
            value=default_symbols,
            key="ticker_input",
            help=(
                "Binance Futures: BTCUSDT, ETHUSDT. "
                "Delta India: BTCUSD, ETHUSD. Use perpetual/futures contracts, not spot tickers."
            ),
        )
        symbols = [s.strip().upper() for s in ticker_input.split(",") if s.strip()]

        if exchange == "binance_futures":
            st.caption("USDⓈ-M futures/perpetuals. Example: BTCUSDT, ETHUSDT, SOLUSDT")
        elif exchange == "delta_india":
            st.caption("India futures/perpetuals. Example: BTCUSD, ETHUSD, SOLUSD")

    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            options=["1d", "1h", "4h"],
            index=0,
            key="dl_timeframe",
        )
        st.session_state.timeframe = timeframe

        start = st.date_input("Start Date", key="dl_start_date")
        end = st.date_input("End Date (blank for latest)", value=None, key="dl_end_date")

    overwrite = st.checkbox("Overwrite existing data", key="dl_overwrite")

    if st.button("Download Data", key="download_btn"):
        if not symbols:
            st.error("Please enter at least one contract symbol.")
            return

        if exchange == "delta_india":
            bad = [s for s in symbols if not s.endswith("USD")]
            if bad:
                st.error(
                    "Delta India futures symbols should end in USD, e.g. BTCUSD, ETHUSD, SOLUSD. "
                    f"Invalid: {', '.join(bad)}"
                )
                return

        with st.spinner(f"Downloading {len(symbols)} contracts..."):
            end_date = end if end is None or isinstance(end, date) else date.fromisoformat(end)
            start_date = start if isinstance(start, date) else date.fromisoformat(start)
            request = DataDownloadRequest(
                source=exchange,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                overwrite=overwrite,
                timeframe=timeframe,
                quote_currency=(
                    st.session_state.quote_currency
                    if exchange in {"binance", "binance_futures"}
                    else "USD"
                ),
            )
            result = data_service.download_data(request)

        if result.success:
            st.success(result.message)
        else:
            st.warning(result.message)
        for fail in result.symbols_failed:
            st.error(fail)
