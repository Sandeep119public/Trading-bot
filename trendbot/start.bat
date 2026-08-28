@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set VENV_DIR=.venv
set STREAMLIT_APP=src\trendbot\ui\streamlit\app.py

echo === TrendBot Startup ===

REM 1. Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo [1/4] Creating virtual environment...
    python -m venv "%VENV_DIR%"
) else (
    echo [1/4] Virtual environment found.
)

REM 2. Activate the virtual environment
echo [2/4] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM 3. Create required directories
echo [3/4] Ensuring data directories exist...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\metadata" mkdir "data\metadata"
if not exist "output" mkdir "output"

REM 4. Install package in editable mode with dev dependencies
echo [4/4] Installing dependencies (this may take a moment on first run)...
pip install -e ".[dev]" -q

echo.
echo Starting TrendBot...
echo The app will open at http://localhost:8501
echo Press Ctrl+C to stop.
echo.

streamlit run "%STREAMLIT_APP%"
