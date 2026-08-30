"""Historical dynamic top-N liquidity universe reconstruction.

This module eliminates survivorship bias by reconstructing the investable
universe at each historical date using only information available at that
time.  The universe is recalculated monthly based on trailing 21-day
dollar volume, with stablecoins excluded.

The design follows a causal information contract:

    information available at t
            ↓
    universe decision at t
            ↓
    signal / risk / execution at t
            ↓
    position t → t+1
            ↓
    return(t+1)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from trendbot.domain.models import UniverseConfig, UniverseMode, UniverseSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stablecoin filter — configurable data, not hard-coded logic
# ---------------------------------------------------------------------------

STABLECOINS: frozenset[str] = frozenset({
    # USD-pegged stablecoins (base asset names)
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS",
    "PYUSD", "USDP", "FRAX", "LUSD", "GUSD", "BUSD",
    # Additional known stablecoins
    "EURT", "AEUR", "USDD", "USDJ", "UST",
})

# Symbols that represent stablecoin pairs (quote side)
_STABLECOIN_QUOTES: frozenset[str] = frozenset({
    "USDT", "USDC", "DAI", "FDUSD", "BUSD", "USD",
})


# ---------------------------------------------------------------------------
# Symbol normalisation helpers
# ---------------------------------------------------------------------------


def extract_base_asset(symbol: str) -> str:
    """Extract the base asset from various symbol formats.

    Supported formats:
        BTCUSD, BTC/USDT, BTC-USDT, BTCUSDT, BTC:USDT

    Returns the base asset in uppercase (e.g. 'BTC').
    """
    value = symbol.upper().strip()

    # Handle BTC/USDT:USDT format (perp)
    if ":" in value:
        value = value.split(":")[0]

    if "/" in value:
        return value.split("/")[0]

    if "-" in value:
        parts = value.split("-")
        return parts[0]

    for suffix in ("USDT", "USDC", "BUSD", "FDUSD", "USD", "BTC", "ETH"):
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]

    return value


def is_stablecoin(symbol: str) -> bool:
    """Check if a symbol or its base asset is a stablecoin."""
    base = extract_base_asset(symbol)
    if base in STABLECOINS:
        return True
    # Also check the full symbol against stablecoin set
    return symbol.upper() in STABLECOINS


def filter_stablecoins(symbols: list[str]) -> list[str]:
    """Remove stablecoins from a list of symbols."""
    return [s for s in symbols if not is_stablecoin(s)]


# ---------------------------------------------------------------------------
# Dollar volume computation
# ---------------------------------------------------------------------------


def compute_dollar_volume(
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily dollar volume = close * base_volume.

    Both DataFrames must have aligned indices and columns.  Missing data
    is filled with NaN (not zero) so that it is excluded from ranking.

    Args:
        close: DataFrame of daily close prices (index=date, columns=assets).
        volume: DataFrame of daily base-unit volume (same shape as close).

    Returns:
        DataFrame of daily dollar volume with NaN for missing observations.
    """
    aligned_volume = volume.reindex(index=close.index, columns=close.columns)
    return close * aligned_volume


def compute_trailing_dollar_volume(
    dollar_volume: pd.DataFrame,
    window: int = 21,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute trailing sum of dollar volume over `window` days.

    Args:
        dollar_volume: Daily dollar volume DataFrame.
        window: Rolling window size in trading days.
        min_periods: Minimum valid observations required. Defaults to `window`.

    Returns:
        Rolling sum of dollar volume.  NaN where insufficient data.
    """
    if min_periods is None:
        min_periods = window
    return dollar_volume.rolling(window=window, min_periods=min_periods).sum()


# ---------------------------------------------------------------------------
# Top-N selection with deterministic tie-breaking
# ---------------------------------------------------------------------------


def select_top_n(
    trailing_volume: pd.Series,
    top_n: int,
    min_volume_days: int = 15,
    previous_rank: pd.Series | None = None,
) -> list[str]:
    """Select the top-N assets by trailing dollar volume.

    Tie-breaking rules (in order):
        1. Higher previous rank (if provided) wins.
        2. Lexicographically smaller symbol wins.

    Assets with NaN trailing volume are excluded.  Assets with fewer
    than `min_volume_days` observations in the trailing window are
    also excluded (their volume is NaN from insufficient history).

    Args:
        trailing_volume: Series of trailing dollar volume indexed by symbol.
        top_n: Number of assets to select.
        min_volume_days: Minimum valid volume observations required.
        previous_rank: Series mapping symbol → rank (1=best) from
            the previous period.  Used for deterministic tie-breaking.

    Returns:
        Sorted list of selected symbols (best first).
    """
    valid = trailing_volume.dropna()
    if valid.empty:
        return []

    if previous_rank is not None:
        rank_series = previous_rank.reindex(valid.index).fillna(len(valid) + 1)
    else:
        rank_series = pd.Series(0, index=valid.index, dtype=int)

    sort_df = pd.DataFrame({
        "volume": valid,
        "prev_rank": rank_series,
        "symbol": valid.index,
    })

    sort_df = sort_df.sort_values(
        by=["volume", "prev_rank", "symbol"],
        ascending=[False, True, True],
    )

    return sort_df["symbol"].tolist()[:top_n]


# ---------------------------------------------------------------------------
# Monthly rebalance date detection
# ---------------------------------------------------------------------------


def is_rebalance_date(
    timestamp: pd.Timestamp,
    rebalance_day: int = 1,
) -> bool:
    """Check if a timestamp falls on a monthly rebalance date.

    Convention: the universe is recomputed on the first available trading
    bar whose date >= rebalance_day of the month.  This function checks
    the calendar date of the timestamp.

    If the rebalance_day falls on a weekend or holiday, the first
    subsequent trading bar is used.  This function is called only on
    actual trading bars (from the data index), so the first bar on or
    after the 1st will naturally satisfy this check.
    """
    return timestamp.day == rebalance_day


def find_rebalance_dates(
    date_index: pd.DatetimeIndex,
    rebalance_day: int = 1,
) -> list[pd.Timestamp]:
    """Find all monthly rebalance dates in a date index.

    Returns the first trading bar in each month whose day >= rebalance_day.
    """
    rebalance_dates: list[pd.Timestamp] = []
    seen_months: set[tuple[int, int]] = set()

    for ts in date_index:
        month_key = (ts.year, ts.month)
        if month_key in seen_months:
            continue
        if ts.day >= rebalance_day:
            rebalance_dates.append(ts)
            seen_months.add(month_key)

    return rebalance_dates


# ---------------------------------------------------------------------------
# Universe schedule precomputation
# ---------------------------------------------------------------------------


def compute_universe_schedule(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    config: UniverseConfig,
) -> dict[pd.Timestamp, list[str]]:
    """Precompute the full historical universe schedule.

    This is the core survivorship-bias-free universe reconstruction.
    At each monthly rebalance date, the universe is determined using
    only information available at or before that date.

    Args:
        close: Full history of close prices (index=date, columns=assets).
        volume: Full history of base-unit volume (same shape).
        config: Universe configuration.

    Returns:
        Dictionary mapping each rebalance date to its list of selected assets.
    """
    if config.mode == UniverseMode.STATIC:
        return _compute_static_schedule(close, config)

    return _compute_dynamic_schedule(close, volume, config)


def _compute_static_schedule(
    close: pd.DataFrame,
    config: UniverseConfig,
) -> dict[pd.Timestamp, list[str]]:
    """Static universe: same assets for all dates."""
    if config.symbols:
        symbols = list(config.symbols)
    else:
        symbols = list(close.columns)
    if config.exclude_stablecoins:
        symbols = filter_stablecoins(symbols)
    return {ts: symbols for ts in close.index}


def _compute_dynamic_schedule(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    config: UniverseConfig,
) -> dict[pd.Timestamp, list[str]]:
    """Dynamic top-N universe with monthly reconstitution."""
    dollar_volume = compute_dollar_volume(close, volume)
    trailing_volume = compute_trailing_dollar_volume(
        dollar_volume,
        window=config.liquidity_window,
    )

    rebalance_dates = find_rebalance_dates(close.index, config.rebalance_day)

    # Phase 1: Compute universe at each rebalance date
    rebalance_universes: dict[pd.Timestamp, list[str]] = {}
    current_universe: list[str] = []
    previous_rank: pd.Series | None = None

    for reb_date in rebalance_dates:
        trailing_at_date = trailing_volume.loc[:reb_date].iloc[-1]

        eligible = trailing_at_date.dropna()
        if config.exclude_stablecoins:
            eligible_symbols = [s for s in eligible.index if not is_stablecoin(s)]
            eligible = eligible[eligible_symbols]

        min_obs = config.min_volume_days
        valid_mask = trailing_volume.loc[:reb_date].notna().sum() >= min_obs
        valid_symbols = valid_mask[valid_mask].index
        eligible = eligible.reindex(valid_symbols).dropna()

        new_universe = select_top_n(
            eligible,
            top_n=config.top_n,
            min_volume_days=min_obs,
            previous_rank=previous_rank,
        )

        current_universe = new_universe
        if new_universe:
            previous_rank = pd.Series(
                {s: i + 1 for i, s in enumerate(new_universe)},
            )

        rebalance_universes[reb_date] = list(current_universe)

    # Phase 2: Fill the schedule for all trading days
    schedule: dict[pd.Timestamp, list[str]] = {}
    current_univ = rebalance_universes.get(rebalance_dates[0], []) if rebalance_dates else []

    for ts in close.index:
        if ts in rebalance_universes:
            current_univ = rebalance_universes[ts]
        schedule[ts] = list(current_univ)

    return schedule


# ---------------------------------------------------------------------------
# Universe provider protocol and implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class UniverseProvider(Protocol):
    """Protocol for components that provide universe membership."""

    def get_universe(
        self,
        timestamp: pd.Timestamp,
    ) -> list[str]:
        """Return the list of eligible assets at the given timestamp."""
        ...

    def get_all_members(self) -> list[str]:
        """Return the union of all assets that were ever in the universe."""
        ...


class StaticUniverse:
    """Static universe: same assets for all dates. Backward compatible."""

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = list(symbols)

    def get_universe(self, timestamp: pd.Timestamp) -> list[str]:
        return list(self._symbols)

    def get_all_members(self) -> list[str]:
        return list(self._symbols)


class HistoricalLiquidityUniverse:
    """Historically reconstructed dynamic top-N liquidity universe.

    This is the primary entry point for survivorship-bias-free backtesting.
    It precomputes the full universe schedule from historical data, then
    serves universe membership queries at each timestamp.
    """

    def __init__(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        config: UniverseConfig,
    ) -> None:
        self._close = close
        self._volume = volume
        self._config = config
        self._schedule = compute_universe_schedule(close, volume, config)
        self._all_members = sorted({
            s for members in self._schedule.values() for s in members
        })

    def get_universe(self, timestamp: pd.Timestamp) -> list[str]:
        """Return the eligible assets at the given timestamp.

        If the exact timestamp is not in the schedule (e.g. between
        rebalance dates), returns the most recent rebalance's universe.
        """
        if timestamp in self._schedule:
            return list(self._schedule[timestamp])

        # Find the most recent rebalance date
        prior_dates = [d for d in self._schedule if d <= timestamp]
        if not prior_dates:
            return []
        latest = max(prior_dates)
        return list(self._schedule[latest])

    def get_all_members(self) -> list[str]:
        """Return the union of all assets that were ever in the universe."""
        return list(self._all_members)

    @property
    def config(self) -> UniverseConfig:
        return self._config

    @property
    def schedule(self) -> dict[pd.Timestamp, list[str]]:
        """Return the full universe schedule (read-only)."""
        return dict(self._schedule)

    def get_snapshots(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> list[UniverseSnapshot]:
        """Generate per-rebalance snapshots for auditing."""
        dollar_volume = compute_dollar_volume(close, volume)
        trailing_volume = compute_trailing_dollar_volume(
            dollar_volume,
            window=self._config.liquidity_window,
        )

        snapshots: list[UniverseSnapshot] = []
        rebalance_dates = find_rebalance_dates(
            close.index, self._config.rebalance_day,
        )

        prev_members: list[str] = []
        for reb_date in rebalance_dates:
            members = self.get_universe(reb_date)
            entries = [s for s in members if s not in prev_members]
            exits = [s for s in prev_members if s not in members]

            trailing_at = trailing_volume.loc[:reb_date].iloc[-1]
            rankings = {s: i + 1 for i, s in enumerate(members)}
            volumes = {s: float(trailing_at.get(s, 0.0)) for s in members}

            snapshots.append(UniverseSnapshot(
                date=reb_date,
                members=members,
                rankings=rankings,
                dollar_volumes_21d=volumes,
                entries=entries,
                exits=exits,
            ))

            prev_members = members

        return snapshots

    def export_audit_csv(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        output_path: Path | str,
    ) -> None:
        """Export universe audit trail to CSV for reproducibility.

        Columns:
            date, symbol, rank, dollar_volume_21d, selected,
            stablecoin_excluded, first_seen, last_seen, entry, exit
        """
        snapshots = self.get_snapshots(close, volume)

        # Track first/last seen across all snapshots
        first_seen: dict[str, pd.Timestamp] = {}
        last_seen: dict[str, pd.Timestamp] = {}
        for snap in snapshots:
            for s in snap.members:
                if s not in first_seen:
                    first_seen[s] = snap.date
                last_seen[s] = snap.date

        rows: list[dict] = []
        all_symbols = sorted(close.columns)

        for snap in snapshots:
            for sym in all_symbols:
                base = extract_base_asset(sym)
                is_sc = is_stablecoin(sym) or is_stablecoin(base)
                selected = sym in snap.members
                rank = snap.rankings.get(sym, 0)
                vol_21d = snap.dollar_volumes_21d.get(sym, 0.0)
                entry = sym in snap.entries
                exit_ = sym in snap.exits
                fs = first_seen.get(sym, pd.NaT)
                ls = last_seen.get(sym, pd.NaT)

                rows.append({
                    "date": snap.date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "rank": rank,
                    "dollar_volume_21d": round(vol_21d, 2),
                    "selected": selected,
                    "stablecoin_excluded": is_sc and not selected,
                    "first_seen": fs.strftime("%Y-%m-%d") if pd.notna(fs) else "",
                    "last_seen": ls.strftime("%Y-%m-%d") if pd.notna(ls) else "",
                    "entry": entry,
                    "exit": exit_,
                })

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Universe audit exported to %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Universe-aware backtest helper
# ---------------------------------------------------------------------------


def filter_close_for_universe(
    close: pd.DataFrame,
    universe: list[str],
) -> pd.DataFrame:
    """Filter a close-price DataFrame to only universe members.

    Assets not in the universe are dropped.  This is used at each
    backtest step to restrict the signal/risk engine to eligible assets.

    Args:
        close: Close-price DataFrame (may contain all historical assets).
        universe: List of currently eligible asset symbols.

    Returns:
        Filtered close DataFrame containing only universe members.
    """
    available = [c for c in universe if c in close.columns]
    return close[available]


def filter_volume_for_universe(
    volume: pd.DataFrame,
    universe: list[str],
) -> pd.DataFrame:
    """Filter a volume DataFrame to only universe members."""
    available = [c for c in universe if c in volume.columns]
    return volume[available]
