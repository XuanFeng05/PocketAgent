"""Causal daily and completed-week indicators for the user's trend research.

This module makes no entry, exit, channel-center, or position-sizing decisions.
The caller must cap its source at the evaluation end before calling. Returned
daily features are available only at that row's 15:00 close; opening decisions
must use the preceding completed daily row.
"""
from __future__ import annotations

from datetime import time
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from feature_layer.triple_screen import _local_dates, _normalize_calendar
from research_layer.technical import normalize_technical_bars


DERIVED_COLUMNS = (
    "daily_ema13", "daily_ema26", "true_range", "atr", "efi_raw", "efi2", "efi13",
    "efi2_previous", "efi2_previous2", "efi2_slope", "efi2_slope_previous",
    "weekly_close", "weekly_ema13", "weekly_ema26", "weekly_ema13_previous",
    "weekly_ema26_previous", "weekly_ema13_slope", "weekly_ema26_slope",
    "weekly_completed_count", "weekly_available_at", "session_close",
    "daily_available_at", "daily_ready", "weekly_ready", "indicators_ready",
)


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Seed on the first period's arithmetic mean, then use alpha=1/period."""
    data = values.to_numpy(dtype=float)
    output = np.full(len(data), np.nan, dtype=float)
    if len(data) >= period:
        # Dividing before summing avoids overflow when all valid ranges are large.
        output[period - 1] = np.sum(data[:period] / period)
        for i in range(period, len(data)):
            output[i] = output[i - 1] * ((period - 1) / period) + data[i] / period
    return pd.Series(output, index=values.index, dtype=float)


def build_user_trend_features(bars: Any, calendar: Any, *, atr_period: int = 14) -> pd.DataFrame:
    """Build pre-adjusted CN close-time EMA13/26, EFI2/13 and Wilder ATR.

    EMA values seed with the first observed close and use ``adjust=False``.
    Force Index seeds its first raw price-change observation at zero and uses
    the supplied share-volume without adjustment. Weekly EMA values seed with
    the first complete week's close; their slopes compare adjacent complete
    weeks. Readiness requires 27 complete weeks and a seeded daily ATR.

    Every natural calendar date must be present, with closed days explicit.
    A truncated first/last week never seeds or publishes a weekly observation.
    Zero-volume rows are retained rather than removed or filled.
    """
    if isinstance(atr_period, (bool, np.bool_)) or not isinstance(atr_period, Integral) or atr_period < 1:
        raise ValueError("atr_period must be a positive integer")
    atr_period = int(atr_period)
    original = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    date_column = "datetime" if "datetime" in original else "date"
    if date_column in original:
        original[date_column] = _local_dates(original[date_column], "Daily bars")
    frame = normalize_technical_bars(original)
    frame = frame.drop(columns=list(DERIVED_COLUMNS), errors="ignore")
    if "volume" not in frame:
        raise ValueError("User trend indicators require explicit daily volume in shares")
    cal = _normalize_calendar(calendar, frame.iloc[0]["datetime"], frame.iloc[-1]["datetime"])
    sessions = pd.DatetimeIndex(cal.loc[cal["is_trading_day"], "date"])
    dates = pd.DatetimeIndex(frame["datetime"])
    unexpected = dates.difference(sessions)
    if len(unexpected):
        raise ValueError(f"Daily bars fall on a calendar-closed date: {unexpected[0].date().isoformat()}")
    expected = sessions[(sessions >= dates[0]) & (sessions <= dates[-1])]
    missing = expected.difference(dates)
    if len(missing):
        examples = ", ".join(x.date().isoformat() for x in missing[:5])
        raise ValueError(f"Missing open-session daily bars ({examples}); suspension must be explicit")

    close = frame["close"]
    previous_close = close.shift(1)
    true_range = pd.concat((frame["high"] - frame["low"],
                            (frame["high"] - previous_close).abs(),
                            (frame["low"] - previous_close).abs()), axis=1).max(axis=1)
    if not np.isfinite(true_range.to_numpy(dtype=float)).all():
        raise ValueError("True Range overflow; verify price units")
    frame["true_range"] = true_range
    frame["atr"] = _wilder_average(true_range, atr_period)
    frame["daily_ema13"] = close.ewm(span=13, adjust=False, min_periods=1).mean()
    frame["daily_ema26"] = close.ewm(span=26, adjust=False, min_periods=1).mean()
    with np.errstate(over="ignore", invalid="ignore"):
        force = close.diff().fillna(0.0) * frame["volume"]
    if not np.isfinite(force.to_numpy(dtype=float)).all():
        raise ValueError("Force Index overflow; verify price and share-volume units")
    frame["efi_raw"] = force
    frame["efi2"] = force.ewm(span=2, adjust=False, min_periods=1).mean()
    frame["efi13"] = force.ewm(span=13, adjust=False, min_periods=1).mean()
    frame["efi2_previous"] = frame["efi2"].shift(1)
    frame["efi2_previous2"] = frame["efi2"].shift(2)
    frame["efi2_slope"] = frame["efi2"].diff()
    frame["efi2_slope_previous"] = frame["efi2_slope"].shift(1)
    frame["daily_available_at"] = pd.to_datetime([
        pd.Timestamp.combine(day.date(), time(15)) for day in frame["datetime"]
    ]).astype("datetime64[ns]")
    frame["session_close"] = frame["daily_available_at"]

    periods = frame["datetime"].dt.to_period("W-SUN")
    cal_periods = cal["date"].dt.to_period("W-SUN")
    calendar_start, calendar_end = cal.iloc[0]["date"], cal.iloc[-1]["date"]
    weeks = []
    for period, group in frame.groupby(periods, sort=True):
        if period.start_time < calendar_start or period.end_time.normalize() > calendar_end:
            continue
        expected_dates = cal.loc[cal_periods.eq(period) & cal["is_trading_day"], "date"]
        if expected_dates.empty or not pd.DatetimeIndex(group["datetime"]).equals(pd.DatetimeIndex(expected_dates)):
            continue
        weeks.append({"weekly_available_at": pd.Timestamp.combine(expected_dates.iloc[-1].date(), time(15)),
                      "weekly_close": float(group.iloc[-1]["close"])})
    weekly_fields = ("weekly_close", "weekly_ema13", "weekly_ema26", "weekly_ema13_previous",
                     "weekly_ema26_previous", "weekly_ema13_slope", "weekly_ema26_slope")
    if weeks:
        complete = pd.DataFrame(weeks)
        complete["weekly_available_at"] = pd.to_datetime(complete["weekly_available_at"]).astype("datetime64[ns]")
        for period in (13, 26):
            field = f"weekly_ema{period}"
            complete[field] = complete["weekly_close"].ewm(span=period, adjust=False, min_periods=1).mean()
            complete[field + "_previous"] = complete[field].shift(1)
            complete[field + "_slope"] = complete[field].diff()
        complete["weekly_completed_count"] = np.arange(1, len(complete) + 1)
        frame = pd.merge_asof(frame, complete, left_on="daily_available_at", right_on="weekly_available_at", direction="backward")
        frame["weekly_completed_count"] = frame["weekly_completed_count"].fillna(0).astype(int)
    else:
        frame["weekly_available_at"] = pd.NaT
        frame["weekly_completed_count"] = 0
        for field in weekly_fields:
            frame[field] = np.nan
    frame["daily_ready"] = frame["atr"].notna()
    frame["weekly_ready"] = (frame["weekly_completed_count"].ge(27)
                              & frame["weekly_ema13_previous"].notna()
                              & frame["weekly_ema26_previous"].notna())
    frame["indicators_ready"] = frame["daily_ready"] & frame["weekly_ready"]
    # Keep the same schema even when a prefix contains no complete week yet.
    source_columns = [column for column in frame.columns if column not in DERIVED_COLUMNS]
    return frame.loc[:, source_columns + list(DERIVED_COLUMNS)].reset_index(drop=True)
