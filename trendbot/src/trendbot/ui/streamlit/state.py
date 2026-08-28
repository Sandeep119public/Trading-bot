"""Streamlit session state management."""

from __future__ import annotations

import streamlit as st

from trendbot.domain.models import (
    BacktestParams,
    BacktestRequest,
    DataDownloadRequest,
    ExecutionParams,
    MomentumParams,
    RiskParams,
    VolatilityParams,
    load_defaults,
)


def init_state() -> None:
    """Initialize session state with default values."""
    if "initialized" not in st.session_state:
        defaults = load_defaults()

        st.session_state.data_source = defaults["data"]["source"]
        st.session_state.timeframe = defaults["data"].get("timeframe", "1d")
        st.session_state.quote_currency = defaults["data"].get("quote_currency", "USDT")
        st.session_state.start_date = defaults["data"]["start_date"]
        st.session_state.end_date = defaults["data"].get("end_date")
        st.session_state.overwrite = defaults["data"]["overwrite"]
        st.session_state.universe = defaults["universe"]

        st.session_state.lookbacks = defaults["strategy"]["lookbacks"]
        st.session_state.allow_short = defaults["strategy"]["allow_short"]

        st.session_state.vol_window = defaults["volatility"]["vol_window"]
        st.session_state.ann_factor = defaults["volatility"]["ann_factor"]

        st.session_state.target_portfolio_vol = defaults["risk"]["target_portfolio_vol"]
        st.session_state.max_gross_leverage = defaults["risk"]["max_gross_leverage"]

        st.session_state.taker_fee_pct = defaults["execution"]["taker_fee_pct"]
        st.session_state.maker_fee_pct = defaults["execution"]["maker_fee_pct"]
        st.session_state.slippage_pct = defaults["execution"]["slippage_pct"]
        st.session_state.rebalance_threshold = defaults["execution"]["rebalance_threshold"]

        st.session_state.min_history = defaults["backtest"]["min_history"]
        st.session_state.benchmark = defaults["backtest"]["benchmark"]

        st.session_state.backtest_result = None
        st.session_state.last_request = None
        st.session_state.initialized = True


def reset_defaults() -> None:
    """Reset all session state to defaults."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def get_download_request(symbols: list[str]) -> DataDownloadRequest:
    """Build a DataDownloadRequest from session state."""
    from datetime import date

    end = None
    if st.session_state.end_date:
        end = date.fromisoformat(st.session_state.end_date) if isinstance(
            st.session_state.end_date, str
        ) else st.session_state.end_date

    return DataDownloadRequest(
        source=st.session_state.data_source,
        symbols=symbols,
        start_date=date.fromisoformat(st.session_state.start_date)
        if isinstance(st.session_state.start_date, str)
        else st.session_state.start_date,
        end_date=end,
        overwrite=st.session_state.overwrite,
    )


def get_backtest_request(symbols: list[str]) -> BacktestRequest:
    """Build a BacktestRequest from session state."""
    from datetime import date

    end = None
    if st.session_state.end_date:
        end = date.fromisoformat(st.session_state.end_date) if isinstance(
            st.session_state.end_date, str
        ) else st.session_state.end_date

    return BacktestRequest(
        data_selection={
            "source": st.session_state.data_source,
            "symbols": symbols,
            "start_date": date.fromisoformat(st.session_state.start_date)
            if isinstance(st.session_state.start_date, str)
            else st.session_state.start_date,
            "end_date": end,
            "timeframe": st.session_state.timeframe,
        },
        momentum=MomentumParams(
            lookbacks=st.session_state.lookbacks,
            allow_short=st.session_state.allow_short,
        ),
        volatility=VolatilityParams(
            vol_window=st.session_state.vol_window,
            ann_factor=st.session_state.ann_factor,
        ),
        risk=RiskParams(
            target_portfolio_vol=st.session_state.target_portfolio_vol,
            max_gross_leverage=st.session_state.max_gross_leverage,
        ),
        execution=ExecutionParams(
            taker_fee_pct=st.session_state.taker_fee_pct,
            maker_fee_pct=st.session_state.maker_fee_pct,
            slippage_pct=st.session_state.slippage_pct,
            rebalance_threshold=st.session_state.rebalance_threshold,
        ),
        backtest=BacktestParams(
            min_history=st.session_state.min_history,
            benchmark=st.session_state.benchmark,
        ),
    )
