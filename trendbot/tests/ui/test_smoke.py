"""UI smoke tests."""

import sys
from pathlib import Path

# Add both src and project root to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
for p in [str(src_path), str(project_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def test_app_imports():
    """Test that the main app module can be imported."""
    from trendbot.ui.streamlit import app
    assert app is not None


def test_state_imports():
    """Test that state module imports correctly."""
    from trendbot.ui.streamlit.state import init_state, reset_defaults
    assert callable(init_state)
    assert callable(reset_defaults)


def test_component_imports():
    """Test that all component modules import correctly."""
    from trendbot.ui.streamlit.components.data_download_form import render_data_download_form
    from trendbot.ui.streamlit.components.execution_params_form import render_execution_params
    from trendbot.ui.streamlit.components.results_panel import render_results_panel
    from trendbot.ui.streamlit.components.risk_params_form import render_risk_params
    from trendbot.ui.streamlit.components.strategy_params_form import render_strategy_params
    from trendbot.ui.streamlit.components.universe_selector import render_universe_selector

    assert callable(render_data_download_form)
    assert callable(render_universe_selector)
    assert callable(render_strategy_params)
    assert callable(render_risk_params)
    assert callable(render_execution_params)
    assert callable(render_results_panel)


def test_page_imports():
    """Test that all page modules import correctly."""
    from trendbot.ui.streamlit.pages.backtest_page import render_backtest_page
    from trendbot.ui.streamlit.pages.data_page import render_data_page
    from trendbot.ui.streamlit.pages.diagnostics_page import render_diagnostics_page
    from trendbot.ui.streamlit.pages.results_page import render_results_page

    assert callable(render_data_page)
    assert callable(render_backtest_page)
    assert callable(render_results_page)
    assert callable(render_diagnostics_page)


def test_defaults_load():
    """Test that default config loads correctly."""
    from trendbot.domain.models import load_defaults
    defaults = load_defaults()
    assert "data" in defaults
    strategy = defaults["strategy"]
    assert "lookbacks" in strategy
    assert "allow_short" in strategy
