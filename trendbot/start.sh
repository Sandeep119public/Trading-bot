#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
STREAMLIT_APP="src/trendbot/ui/streamlit/app.py"

echo "=== TrendBot Startup ==="

# 1. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment found."
fi

# 2. Activate the virtual environment
echo "[2/4] Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3. Create required directories
echo "[3/4] Ensuring data directories exist..."
mkdir -p data/raw data/metadata output

# 4. Install package in editable mode with dev dependencies
echo "[4/4] Installing dependencies (this may take a moment on first run)..."
pip install -e ".[dev]" -q

echo ""
echo "Starting TrendBot..."
echo "The app will open at http://localhost:8501"
echo "Press Ctrl+C to stop."
echo ""

streamlit run "$STREAMLIT_APP"
