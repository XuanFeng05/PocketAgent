"""Point-in-time weekly MACD and daily Force Index for Triple Screen v1.

The calendar contract is explicit: one row per natural date, including closed
dates. Boundary weeks without a complete calendar are never published. No
missing open-session bar is fabricated or interpreted as a suspension.
"""
from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from research_layer.technical import normalize_technical_bars
from strategy_layer.triple_screen import resolve_triple_settings


def _calendar_flag(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError("Calendar is_trading_day must contain explicit boolean or 0/1 values.")


def _local_dates(values: pd.Series, label: str) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError(f"{label} requires valid local dates without timezone offsets.")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError(f"{label} requires dates at midnight, not intraday timestamps.")
    # DuckDB date columns may arrive at microsecond resolution while market
    # shards use nanoseconds. DatetimeIndex.equals treats those dtypes as
    # different even for identical dates, so normalize representation once.
    return dates.astype("datetime64[ns]")


def _normalize_calendar(calendar: Any, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = calendar.copy(deep=True) if isinstance(calendar, pd.DataFrame) else pd.DataFrame(calendar)
    if not {"date", "is_trading_day"}.issubset(frame.columns) or frame.empty:
        raise ValueError("Calendar requires nonempty date and is_trading_day columns.")
    frame["date"] = _local_dates(frame["date"], "Calendar")
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate calendar dates are not allowed; select one exchange calendar.")
    frame["is_trading_day"] = frame["is_trading_day"].map(_calendar_flag)
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame.iloc[0]["date"] > start or frame.iloc[-1]["date"] < end:
        raise ValueError("Calendar does not cover the complete daily input range.")
    expected = pd.date_range(frame.iloc[0]["date"], frame.iloc[-1]["date"], freq="D")
    missing = expected.difference(pd.DatetimeIndex(frame["date"]))
    if len(missing):
        example = ", ".join(value.date().isoformat() for value in missing[:5])
        raise ValueError(f"Calendar is missing natural dates ({example}); include closed-day rows.")
    return frame


def build_triple_features(bars: Any, calendar: Any, settings=None) -> pd.DataFrame:
    """Return close-time features, never the unfinished week's final MACD.

    The result has one row per input daily bar. ``weekly_available_at`` is the
    actual final exchange session's 15:00 close, not a nominal Friday label.
    A caller evaluating an opening order must therefore use prior-close rows.
    ``volume`` is the supplied share count; no inference or unit conversion is
    performed. The runner is responsible for recording the source's units and
    freezing one coherent adjustment version for the entire input snapshot.
    """
    resolved = resolve_triple_settings(settings)
    original = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    date_column = "datetime" if "datetime" in original else "date"
    if date_column in original:
        original[date_column] = _local_dates(original[date_column], "Daily bars")
    frame = normalize_technical_bars(original)
    # Derived columns are always recomputed, including when a prior output
    # snapshot is supplied. Never trust externally supplied weekly features.
    frame = frame.drop(columns=["force_index_2", "weekly_hist", "weekly_hist_previous",
                                "weekly_completed_count", "weekly_available_at", "weekly_up",
                                "indicators_ready", "entry_ready", "session_close"], errors="ignore")
    if "volume" not in frame:
        raise ValueError("Triple Screen requires explicit daily volume in shares for Force Index.")
    # normalize_technical_bars already rejects negative, missing, or infinite
    # numeric volume. Zero volume is retained as a non-tradeable observation.
    cal = _normalize_calendar(calendar, frame.iloc[0]["datetime"], frame.iloc[-1]["datetime"])
    if resolved.is_us:
        if "session_close" not in cal:
            raise ValueError("US research requires an explicit local session_close calendar including early closes.")
        opens = cal.loc[cal["is_trading_day"]].copy()
        closes = pd.to_datetime(opens["session_close"], errors="raise", format="mixed")
        if closes.isna().any() or closes.dt.tz is not None or not closes.dt.normalize().eq(opens["date"]).all():
            raise ValueError("US session_close must be a local timestamp on each trading date.")
        valid_close = (closes.dt.hour.isin([13, 16]) & closes.dt.minute.eq(0)
                       & closes.dt.second.eq(0) & closes.dt.microsecond.eq(0)
                       & closes.dt.nanosecond.eq(0))
        if not valid_close.all():
            raise ValueError("Unsupported US session close; expected 13:00 or 16:00 New York time.")
        close_lookup = dict(zip(opens["date"], closes))
    else:
        close_lookup = {d: pd.Timestamp.combine(d.date(), time(15)) for d in cal["date"]}
    session_dates = pd.DatetimeIndex(cal.loc[cal["is_trading_day"], "date"])
    daily_dates = pd.DatetimeIndex(frame["datetime"])
    unexpected = daily_dates.difference(session_dates)
    if len(unexpected):
        raise ValueError(f"Daily bars fall on a calendar-closed date: {unexpected[0].date().isoformat()}.")
    in_range = session_dates[(session_dates >= daily_dates[0]) & (session_dates <= daily_dates[-1])]
    missing_bars = in_range.difference(daily_dates)
    if len(missing_bars):
        example = ", ".join(value.date().isoformat() for value in missing_bars[:5])
        raise ValueError(f"Missing open-session daily bars ({example}); suspension must be explicit, not inferred.")

    force = frame["close"].diff().fillna(0.0) * frame["volume"]
    if not np.isfinite(force.to_numpy(dtype=float)).all():
        raise ValueError("Force Index overflow; verify price and share-volume units.")
    frame["force_index_2"] = force.ewm(span=resolved.force_span, adjust=False, min_periods=1).mean()
    periods = frame["datetime"].dt.to_period("W-SUN")
    calendar_periods = cal["date"].dt.to_period("W-SUN")
    calendar_start, calendar_end = cal.iloc[0]["date"], cal.iloc[-1]["date"]
    weekly = []
    for period, group in frame.groupby(periods, sort=True):
        first_date, last_date = period.start_time, period.end_time.normalize()
        if first_date < calendar_start or last_date > calendar_end:
            continue
        expected = cal.loc[calendar_periods.eq(period) & cal["is_trading_day"], "date"]
        if expected.empty:
            continue
        # Both a truncated first week and a currently developing last week
        # fail this equality. Neither may seed the weekly EMA history.
        observed = pd.DatetimeIndex(group["datetime"])
        if not observed.equals(pd.DatetimeIndex(expected)):
            continue
        weekly.append({"weekly_available_at": close_lookup[expected.iloc[-1]],
                       "weekly_close": float(group.iloc[-1]["close"])})

    frame["_decision_close"] = frame["datetime"].map(close_lookup)
    # The selected market calendar owns close times, including when input bars
    # carry stale or invalid session_close values from an earlier snapshot.
    frame["session_close"] = frame["_decision_close"]
    if weekly:
        complete = pd.DataFrame(weekly)
        close = complete["weekly_close"]
        dif = close.ewm(span=resolved.macd_fast, adjust=False).mean() - close.ewm(span=resolved.macd_slow, adjust=False).mean()
        dea = dif.ewm(span=resolved.macd_signal, adjust=False).mean()
        complete["weekly_hist"] = dif - dea
        complete["weekly_hist_previous"] = complete["weekly_hist"].shift(1)
        complete["weekly_completed_count"] = np.arange(1, len(complete) + 1)
        frame = pd.merge_asof(frame, complete.drop(columns="weekly_close"),
                              left_on="_decision_close", right_on="weekly_available_at", direction="backward")
        frame["weekly_completed_count"] = frame["weekly_completed_count"].fillna(0).astype(int)
    else:
        frame["weekly_available_at"] = pd.NaT
        frame["weekly_hist"] = np.nan
        frame["weekly_hist_previous"] = np.nan
        frame["weekly_completed_count"] = 0
    frame["weekly_up"] = frame["weekly_hist"].gt(frame["weekly_hist_previous"])
    frame["indicators_ready"] = (frame["weekly_completed_count"].ge(resolved.warmup_weeks)
                                  & frame["weekly_hist_previous"].notna()
                                  & frame["force_index_2"].notna())
    tradeable = frame["volume"].gt(0)
    if "is_tradeable" in frame:
        tradeable &= frame["is_tradeable"]
    frame["entry_ready"] = frame["indicators_ready"] & frame["weekly_up"] & frame["force_index_2"].lt(0) & tradeable
    return frame.drop(columns="_decision_close").reset_index(drop=True)
