"""Signal generation for multi-horizon trend-following strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_signals(
    close: pd.DataFrame,
    lookbacks: list[int],
    allow_short: bool = True,
) -> pd.DataFrame:
    """Compute momentum signals for each asset across multiple lookbacks.

    Args:
        close: DataFrame of daily close prices (index=date, columns=assets).
        lookbacks: List of lookback periods in bars.
        allow_short: If False, clip signals to [0, 1].

    Returns:
        DataFrame of normalized momentum scores in [-1, 1] or [0, 1].
    """
    signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    for lb in lookbacks:
        shifted = close.shift(lb)
        signal = np.sign(close - shifted)
        signals += signal

    score = signals / len(lookbacks)

    if not allow_short:
        score = score.clip(lower=0.0)

    return score
