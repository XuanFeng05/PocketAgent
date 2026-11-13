"""Read-only multi-symbol daily inputs with explicit historical membership."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from backtest_layer.data import _flag, demo_bars, load_daily_bars, normalize_daily_bars

MAX_PORTFOLIO_SYMBOLS = 64
MAX_PORTFOLIO_ROWS = 250_000


def normalize_portfolio_bars(bars: Any, *, symbols: Iterable[str] | None = None,
                             require_membership: bool = False) -> pd.DataFrame:
    frame = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    if "datetime" not in frame and "date" in frame:
        frame = frame.rename(columns={"date": "datetime"})
    if "symbol" not in frame:
        raise ValueError("Portfolio input requires a symbol column.")
    if len(frame) > MAX_PORTFOLIO_ROWS:
        raise ValueError(f"Portfolio input exceeds {MAX_PORTFOLIO_ROWS} rows; narrow the date range.")
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    if symbols is not None:
        if isinstance(symbols, (str, bytes)):
            raise ValueError("symbols must be an array, not a string")
        selected = tuple(sorted({str(s).upper().strip() for s in symbols}))
        if not selected or any(not s for s in selected):
            raise ValueError("symbols must contain nonempty identifiers")
        missing = set(selected) - set(frame["symbol"])
        if missing:
            raise ValueError(f"Portfolio symbols have no data: {sorted(missing)}")
        frame = frame.loc[frame["symbol"].isin(selected)].copy()
    if frame.empty or frame["symbol"].nunique() > MAX_PORTFOLIO_SYMBOLS:
        raise ValueError(f"Portfolio input requires 1 to {MAX_PORTFOLIO_SYMBOLS} symbols.")
    has_membership = "is_member" in frame
    if require_membership and not has_membership:
        raise ValueError("Historical daily is_member facts are required for this portfolio.")
    if "membership_known_at" in frame and not has_membership:
        raise ValueError("membership_known_at requires is_member.")
    parts = []
    for symbol, part in frame.groupby("symbol", sort=True):
        normalized = normalize_daily_bars(part, symbol=str(symbol), min_rows=1)
        original_sources = set(normalized["membership_source"]) if "membership_source" in normalized else set()
        if original_sources - {"historical_daily", "user_declared_static_pool"}:
            raise ValueError("Unknown membership_source provenance")
        historical = has_membership and "user_declared_static_pool" not in original_sources
        if require_membership and not historical:
            raise ValueError("Historical daily membership is required; static user pools are not sufficient.")
        normalized["is_member"] = normalized["is_member"].map(lambda x: _flag(x, "is_member")) if has_membership else True
        if "membership_known_at" in normalized:
            known = pd.to_datetime(normalized["membership_known_at"], errors="raise", format="ISO8601")
            if known.isna().any() or known.dt.tz is not None:
                raise ValueError("membership_known_at requires finite local timestamps.")
            close_times = normalized["datetime"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(15, "h")
            if (known.to_numpy(dtype="datetime64[ns]") > close_times).any():
                raise ValueError("Membership cannot become known after the row's decision close.")
            normalized["membership_known_at"] = known
        else:
            # Explicit daily facts become known at this close, not retroactively
            # at the open. A declared static user pool is known at midnight.
            normalized["membership_known_at"] = normalized["datetime"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(15 if historical else 0, "h")
        normalized["membership_source"] = "historical_daily" if historical else "user_declared_static_pool"
        if "listed_from" in normalized:
            listed = pd.to_datetime(normalized["listed_from"], errors="raise").dt.normalize()
            if listed.isna().any() or (normalized["datetime"] < listed).any():
                raise ValueError("Observed bars cannot predate listed_from.")
        parts.append(normalized)
    result = pd.concat(parts, ignore_index=True).sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)
    if result["datetime"].nunique() < 2:
        raise ValueError("Portfolio needs at least two distinct daily dates.")
    return result


def load_portfolio_bars(*, csv_path: str | Path | None = None, db_path: str | Path | None = None,
                        symbols: Iterable[str] | None = None, start: str | None = None,
                        end: str | None = None, require_membership: bool = False) -> pd.DataFrame:
    if bool(csv_path) == bool(db_path):
        raise ValueError("Specify exactly one CSV path or market database path.")
    if csv_path:
        frame = pd.read_csv(Path(csv_path), dtype={"symbol": str})
    else:
        selected = tuple(sorted({str(s).strip().upper() for s in (symbols or ())}))
        if not selected or len(selected) > MAX_PORTFOLIO_SYMBOLS:
            raise ValueError("Market portfolio input requires an explicit bounded symbols list.")
        if require_membership:
            raise ValueError("Market storage has no daily membership contract; provide a CSV with historical is_member facts.")
        frame = pd.concat([load_daily_bars(db_path=db_path, symbol=s, start=start, end=end) for s in selected], ignore_index=True)
    normalized = normalize_portfolio_bars(frame, symbols=symbols, require_membership=require_membership)
    if start:
        normalized = normalized.loc[normalized["datetime"] >= pd.Timestamp(start)].copy()
    if end:
        normalized = normalized.loc[normalized["datetime"] <= pd.Timestamp(end)].copy()
    # Re-normalization preserves the explicit membership provenance rather
    # than claiming generated static flags are genuine historical membership.
    if normalized.empty or normalized["datetime"].nunique() < 2:
        raise ValueError("Portfolio date range has fewer than two observations.")
    return normalized.reset_index(drop=True)


def demo_portfolio_bars(*, symbols: Iterable[str] | None = None, days: int = 120) -> pd.DataFrame:
    names = tuple(symbols or ("DEMO_A", "DEMO_B", "DEMO_C"))
    if not names or len(set(names)) != len(names):
        raise ValueError("Demo symbols must be nonempty and unique.")
    parts = []
    for index, symbol in enumerate(names):
        part = demo_bars(symbol=str(symbol), days=days)
        t = np.arange(len(part), dtype=float)
        close = np.round(10 + index * 5 + t * (0.012 + index * .003) + .7 * np.sin(t / (7 + index) + index), 2)
        opening = np.concatenate(([close[0]], close[:-1]))
        part["open"] = opening
        part["close"] = close
        part["high"] = np.maximum(opening, close) + .05
        part["low"] = np.minimum(opening, close) - .05
        part["amount"] = close * part["volume"]
        part["is_member"] = True
        part["membership_known_at"] = part["datetime"]
        parts.append(part)
    return normalize_portfolio_bars(pd.concat(parts, ignore_index=True), require_membership=True)
