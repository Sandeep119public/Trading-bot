# TrendBot

Multi-horizon trend-following backtesting application.

## Quick Start

**Mac / Linux:**
```bash
chmod +x start.sh
./start.sh
```

**Windows Command Prompt:**
```bat
start.bat
```

**Any OS (Python 3.11+ required):**
```bash
python start.py
```

**Or use Make (Mac/Linux):**
```bash
make setup   # First time only: creates venv, installs deps
make run     # Start the Streamlit app
```

These scripts automatically create a virtual environment, install all dependencies, set up required data directories, and launch the Streamlit UI at `http://localhost:8501`.

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

## Development

```bash
make test      # Run all tests
make lint      # Lint with ruff
make typecheck # Type check with mypy
make clean     # Remove venv, caches, and data directories
```

## Configuration

Default parameters in `config/defaults.yaml`. All parameters configurable via UI.
