"""Export functionality for backtest results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def export_returns(returns: pd.Series, output_dir: str | Path) -> Path:
    """Export returns to CSV.

    Args:
        returns: Strategy daily returns.
        output_dir: Output directory path.

    Returns:
        Path to saved file.
    """
    path = Path(output_dir) / "returns.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_frame("strategy_return").to_csv(path)
    return path


def export_positions(positions: pd.DataFrame, output_dir: str | Path) -> Path:
    """Export positions to CSV.

    Args:
        positions: Position weights DataFrame.
        output_dir: Output directory path.

    Returns:
        Path to saved file.
    """
    path = Path(output_dir) / "positions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(path)
    return path


def export_executed_weights(weights: pd.DataFrame, output_dir: str | Path) -> Path:
    """Export executed weights to CSV.

    Args:
        weights: Executed weights DataFrame.
        output_dir: Output directory path.

    Returns:
        Path to saved file.
    """
    path = Path(output_dir) / "executed_weights.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(path)
    return path


def export_turnover(turnover: pd.Series, output_dir: str | Path) -> Path:
    """Export turnover to CSV.

    Args:
        turnover: Daily turnover Series.
        output_dir: Output directory path.

    Returns:
        Path to saved file.
    """
    path = Path(output_dir) / "turnover.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    turnover.to_frame("turnover").to_csv(path)
    return path


def export_stats(stats: dict[str, float], output_dir: str | Path) -> Path:
    """Export performance statistics to JSON.

    Args:
        stats: Performance metrics dictionary.
        output_dir: Output directory path.

    Returns:
        Path to saved file.
    """
    path = Path(output_dir) / "stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return path
