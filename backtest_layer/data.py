from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _flag(value: Any, name: str) -> bool:
    if value is True or value == 1 or str(value).strip().lower() in {"true", "1"}:
        return True
    if value is False or value == 0 or str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"{name} must contain explicit boolean / 0 / 1 values, got {value!r}.")


def normalize_daily_bars(bars: Any, *, symbol: str | None = None, min_rows: int = 2) -> pd.DataFrame:
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 1:
        raise ValueError("min_rows must be a positive integer.")
    frame = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    if "datetime" not in frame and "date" in frame:
        frame = frame.rename(columns={"date": "datetime"})
    required = {"datetime", "open", "high", "low", "close", "volume", "is_st"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Daily input missing columns: {', '.join(sorted(missing))}. Dated is_st is required.")
    if "symbol" not in frame:
        if not symbol:
            raise ValueError("Input needs a symbol column or an explicit symbol.")
        frame["symbol"] = str(symbol).upper()
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    if frame["symbol"].isin(["", "NONE", "NAN", "<NA>"]).any():
        raise ValueError("Symbols must be nonempty.")
    if symbol:
        frame = frame.loc[frame["symbol"].eq(str(symbol).upper())].copy()
    if len(frame) < min_rows:
        raise ValueError(f"At least {min_rows} daily bars for the selected symbol are required.")
    if frame["symbol"].nunique() != 1:
        raise ValueError("This engine uses one independent single-stock account; select one symbol.")
    if "freq" in frame and not frame["freq"].astype(str).eq("daily").all():
        raise ValueError("Only daily input is supported by the CPU backtest.")
    if "adjust" in frame and not frame["adjust"].astype(str).isin(["none", "3", "unadjusted"]).all():
        raise ValueError("Execution needs unadjusted daily prices (adjust=none).")
    dates = pd.to_datetime(frame["datetime"], errors="raise")
    if dates.isna().any():
        raise ValueError("Every bar requires a valid date; missing dates are not allowed.")
    if dates.dt.tz is not None:
        raise ValueError("Use local exchange dates without timezone offsets.")
    frame["datetime"] = dates.dt.normalize()
    if frame["datetime"].duplicated().any():
        raise ValueError("Duplicate symbol/date rows are not allowed.")
    for field in ["open", "high", "low", "close", "volume"]:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(frame[field].to_numpy(dtype=float)).all():
            raise ValueError(f"{field} must be finite.")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative.")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any() or (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("OHLC bounds are inconsistent.")
    if "amount" not in frame:
        frame["amount"] = frame["close"] * frame["volume"]
    else:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="raise")
        if not np.isfinite(frame["amount"].to_numpy(dtype=float)).all() or (frame["amount"] < 0).any():
            raise ValueError("amount must be finite and nonnegative; omit the column to estimate from close*volume.")
    for field in ["is_st", "is_tradeable", "market_can_buy", "market_can_sell"]:
        if field in frame:
            frame[field] = frame[field].map(lambda x: _flag(x, field))
    if "corporate_action" in frame and frame["corporate_action"].map(lambda x: _flag(x, "corporate_action")).any():
        raise ValueError("Corporate-action intervals are not supported yet; choose an unaffected interval.")
    for field in ["limit_pct", "limit_reference_close"]:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="raise")
            if not np.isfinite(frame[field].to_numpy(dtype=float)).all() or (frame[field] <= 0).any():
                raise ValueError(f"{field} must be finite and positive.")
    if "limit_pct" in frame and (frame["limit_pct"] >= 1).any():
        raise ValueError("limit_pct must be below 1.")
    return frame.sort_values("datetime", kind="stable").reset_index(drop=True)


def load_daily_bars(
    *, csv_path: str | Path | None = None, db_path: str | Path | None = None,
    symbol: str | None = None, start: str | None = None, end: str | None = None,
) -> pd.DataFrame:
    """Read existing data without building caches or modifying its catalog."""
    if bool(csv_path) == bool(db_path):
        raise ValueError("Specify exactly one of csv_path or db_path.")
    if csv_path:
        frame = pd.read_csv(Path(csv_path), dtype={"symbol": str})
    else:
        if not symbol:
            raise ValueError("A symbol is required for market storage input.")
        from data_layer.storage.duckdb_storage import load_kline_from_duckdb, load_stock_status_daily_from_duckdb

        frame = load_kline_from_duckdb(db_path, symbol=symbol, start=start, end=end, freq="daily", adjust="none")
        if frame.empty:
            raise ValueError("No unadjusted daily bars found for the selected symbol and interval.")
        status = load_stock_status_daily_from_duckdb(db_path, symbol=symbol, start=start, end=end)
        if status.empty:
            raise ValueError("No dated ST status coverage; provide historical stock_status_daily before backtesting.")
        frame["_date"] = pd.to_datetime(frame["datetime"]).dt.normalize()
        status["_date"] = pd.to_datetime(status["date"]).dt.normalize()
        if status.duplicated(["symbol", "_date"]).any():
            raise ValueError("Duplicate dated ST facts in market storage.")
        frame = frame.merge(status[["symbol", "_date", "is_st"]], on=["symbol", "_date"], how="left", validate="many_to_one")
        if frame["is_st"].isna().any():
            raise ValueError("Dated ST status is incomplete for this interval.")
        frame = frame.drop(columns="_date")
    time_column = "datetime" if "datetime" in frame else "date"
    if time_column in frame:
        dates = pd.to_datetime(frame[time_column], errors="raise").dt.normalize()
        # Validate before filtering: NaT comparisons would silently discard a
        # malformed row whenever start/end is supplied.
        if dates.isna().any():
            raise ValueError("Every bar requires a valid date; missing dates are not allowed.")
        if start:
            frame = frame.loc[dates >= pd.Timestamp(start)].copy()
            dates = dates.loc[frame.index]
        if end:
            frame = frame.loc[dates <= pd.Timestamp(end)].copy()
    return normalize_daily_bars(frame, symbol=symbol)


def demo_bars(*, symbol: str = "DEMO", days: int = 100) -> pd.DataFrame:
    """Synthetic, deterministic OHLC used only to exercise the research flow."""
    if days < 2:
        raise ValueError("Demo needs at least two days.")
    rows = []
    current_date = date(2025, 1, 2)
    previous = 10.0
    for index in range(days):
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        opening = round(previous * (1 + 0.002 * np.sin(index)), 2)
        close = round(10 + index * 0.015 + 0.8 * np.sin(index / 7), 2)
        rows.append({"symbol": symbol, "datetime": current_date.isoformat(), "open": opening,
                     "high": max(opening, close) + 0.05, "low": min(opening, close) - 0.05,
                     "close": close, "volume": 1_000_000, "amount": 10_000_000.0,
                     "is_st": False, "adjust": "none", "freq": "daily"})
        previous = close
        current_date += timedelta(days=1)
    return pd.DataFrame(rows)
