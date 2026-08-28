#!/usr/bin/env python3
"""Universal TrendBot startup script.

Works on any OS as long as Python 3.11+ is installed.
Usage: python start.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / ".venv"
STREAMLIT_APP = SCRIPT_DIR / "src" / "trendbot" / "ui" / "streamlit" / "app.py"
REQUIRED_DIRS = [
    SCRIPT_DIR / "data" / "raw",
    SCRIPT_DIR / "data" / "metadata",
    SCRIPT_DIR / "output",
]


def find_python() -> str:
    """Find the best available Python interpreter."""
    for candidate in [sys.executable, "python3", "python"]:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            version_line = result.stdout.strip()
            version_str = version_line.split()[-1]
            major, minor = map(int, version_str.split(".")[:2])
            if major >= 3 and minor >= 11:
                return candidate
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            continue
    print("Error: Python 3.11 or higher is required.", file=sys.stderr)
    sys.exit(1)


def create_venv(python: str) -> None:
    """Create the virtual environment if it doesn't exist."""
    if VENV_DIR.exists():
        print("[1/4] Virtual environment found.")
    else:
        print("[1/4] Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)


def get_pip_path() -> Path:
    """Get the pip executable path inside the venv."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def get_python_path() -> Path:
    """Get the Python executable path inside the venv."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_streamlit_path() -> Path:
    """Get the streamlit executable path inside the venv."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "streamlit.exe"
    return VENV_DIR / "bin" / "streamlit"


def ensure_directories() -> None:
    """Create required data and output directories."""
    print("[3/4] Ensuring data directories exist...")
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def install_dependencies() -> None:
    """Install the package in editable mode with dev dependencies."""
    print("[4/4] Installing dependencies (this may take a moment on first run)...")
    pip = get_pip_path()
    subprocess.run(
        [str(pip), "install", "-e", ".[dev]", "-q"],
        cwd=str(SCRIPT_DIR),
        check=True,
    )


def run_streamlit() -> None:
    """Launch the Streamlit application."""
    streamlit = get_streamlit_path()
    print()
    print("Starting TrendBot...")
    print("The app will open at http://localhost:8501")
    print("Press Ctrl+C to stop.")
    print()
    subprocess.run(
        [str(streamlit), "run", str(STREAMLIT_APP), "--server.headless", "true"],
        cwd=str(SCRIPT_DIR),
        check=True,
    )


def main() -> None:
    print("=== TrendBot Startup ===")
    python = find_python()
    create_venv(python)
    print("[2/4] Activating virtual environment...")
    ensure_directories()
    install_dependencies()
    run_streamlit()


if __name__ == "__main__":
    main()
