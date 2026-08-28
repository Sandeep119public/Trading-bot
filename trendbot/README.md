# TrendBot

Multi-horizon trend-following backtesting application.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```bash
streamlit run src/trendbot/ui/streamlit/app.py
```

## Architecture

```
trendbot/
├── src/trendbot/          # Core library
│   ├── domain/            # Strategy logic, models (no IO)
│   ├── application/       # Services, ports, DTOs
│   ├── infrastructure/    # Data providers, repositories
│   └── reporting/         # Charts, exports
├── ui/streamlit/          # Streamlit UI
├── config/                # Default configuration
├── data/                  # Local data storage
├── output/                # Backtest exports
└── tests/                 # Unit, integration, UI tests
```

**Dependency rule:** UI → Application → Domain

## Strategy

Multi-horizon momentum with volatility targeting:
- Signal: `sign(close_t - close_{t-lookback})` across multiple lookbacks
- Sizing: Risk-budgeted weights scaled by asset volatility
- Execution: Rebalance threshold, transaction costs, 1-day lag

## Testing

```bash
pytest                    # Run all tests
ruff check .             # Lint
mypy src                  # Type check
```

## Configuration

Default parameters in `config/defaults.yaml`. All parameters configurable via UI.
