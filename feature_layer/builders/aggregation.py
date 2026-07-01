from __future__ import annotations

import pandas as pd


INTRADAY_MINUTES: dict[str, int] = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 60,
}

EXPECTED_SOURCE_ROWS_FROM_5MIN: dict[str, int] = {
    "5min": 1,
    "15min": 3,
    "30min": 6,
    "60min": 12,
    "daily": 48,
    "weekly": 240,
    "monthly": 1008,
}

EXPECTED_SOURCE_ROWS_FROM_DAILY: dict[str, int] = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
}

def aggregate_ohlcv_from_base(
    bars: pd.DataFrame,
    target_freq: str,
    *,
    base_freq: str = "5min",
    decision_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Aggregate base-frequency OHLCV bars into an as-of higher-frequency stream.

    The input datetime is treated as the completed base bar end. When
    decision_time is supplied, rows after that time are ignored, so the last
    higher-frequency row may be a partial bar built only from visible data.
    """
    target = normalize_frequency(target_freq)
    base = normalize_frequency(base_freq)
    working = _prepare_bars(bars, decision_time=decision_time)
    if working.empty:
        return _empty_aggregated_frame()

    if target == base:
        return _base_as_aggregated(working, target_freq=target)

    if base in INTRADAY_MINUTES and target in INTRADAY_MINUTES:
        return _aggregate_intraday(working, target_freq=target, base_freq=base)

    if base in INTRADAY_MINUTES and target == "daily":
        return _aggregate_by_period(
            working,
            target_freq=target,
            period_column=working["datetime"].dt.date,
            expected_source_rows=EXPECTED_SOURCE_ROWS_FROM_5MIN[target],
        )

    if target == "weekly":
        week_start = (working["datetime"] - pd.to_timedelta(working["datetime"].dt.weekday, unit="D")).dt.date
        expected = _expected_source_rows(base=base, target=target)
        return _aggregate_by_period(
            working,
            target_freq=target,
            period_column=week_start,
            expected_source_rows=expected,
        )

    if target == "monthly":
        month_start = working["datetime"].dt.to_period("M").dt.to_timestamp().dt.date
        expected = _expected_source_rows(base=base, target=target)
        return _aggregate_by_period(
            working,
            target_freq=target,
            period_column=month_start,
            expected_source_rows=expected,
        )

    raise ValueError(f"Unsupported target frequency: {target_freq}")


def normalize_frequency(freq: str) -> str:
    value = str(freq or "").strip().lower()
    aliases = {
        "5": "5min",
        "15": "15min",
        "30": "30min",
        "60": "60min",
        "d": "daily",
        "1d": "daily",
        "w": "weekly",
        "1w": "weekly",
        "m": "monthly",
        "1m": "monthly",
    }
    return aliases.get(value, value)


def _prepare_bars(
    bars: pd.DataFrame,
    *,
    decision_time: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()

    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    working = bars.copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"]).copy()

    if decision_time is not None:
        cutoff = pd.to_datetime(decision_time)
        working = working.loc[working["datetime"] <= cutoff].copy()

    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in working.columns:
            working[column] = 0.0 if column in {"volume", "amount"} else pd.NA
        working[column] = pd.to_numeric(working[column], errors="coerce")

    if "symbol" not in working.columns:
        working["symbol"] = ""
    if "adjust" not in working.columns:
        working["adjust"] = ""

    return working.sort_values(["symbol", "adjust", "datetime"]).reset_index(drop=True)


def _base_as_aggregated(working: pd.DataFrame, *, target_freq: str) -> pd.DataFrame:
    result = working.copy()
    result["bar_start"] = result["datetime"]
    result["bar_end"] = result["datetime"]
    result["available_at"] = result["datetime"]
    result["source_start"] = result["datetime"]
    result["source_end"] = result["datetime"]
    result["source_rows"] = 1
    result["expected_source_rows"] = 1
    result["is_partial"] = False
    result["progress"] = 1.0
    result["freq"] = target_freq
    result = _refresh_pct_chg(result)
    return result[_aggregated_columns()]


def _aggregate_intraday(
    working: pd.DataFrame,
    *,
    target_freq: str,
    base_freq: str,
) -> pd.DataFrame:
    target_minutes = INTRADAY_MINUTES[target_freq]
    base_minutes = INTRADAY_MINUTES[base_freq]
    if target_minutes < base_minutes or target_minutes % base_minutes != 0:
        raise ValueError(f"{target_freq} cannot be built from {base_freq}")

    source_rows = target_minutes // base_minutes
    result_parts: list[pd.DataFrame] = []
    group_cols = _meta_group_columns(working)
    for _, day_frame in working.groupby(group_cols + [working["datetime"].dt.date], sort=False, dropna=False):
        day_frame = day_frame.copy()
        day_frame["_bucket"] = range(len(day_frame))
        day_frame["_bucket"] = day_frame["_bucket"] // source_rows
        result_parts.append(
            _aggregate_groups(
                day_frame,
                group_cols=group_cols + ["_bucket"],
                target_freq=target_freq,
                expected_source_rows=source_rows,
            )
        )

    if not result_parts:
        return _empty_aggregated_frame()
    return pd.concat(result_parts, ignore_index=True).sort_values(_sort_columns()).reset_index(drop=True)


def _aggregate_by_period(
    working: pd.DataFrame,
    *,
    target_freq: str,
    period_column: pd.Series,
    expected_source_rows: int,
) -> pd.DataFrame:
    frame = working.copy()
    frame["_period"] = period_column.values
    return _aggregate_groups(
        frame,
        group_cols=_meta_group_columns(frame) + ["_period"],
        target_freq=target_freq,
        expected_source_rows=expected_source_rows,
    ).sort_values(_sort_columns()).reset_index(drop=True)


def _aggregate_groups(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    target_freq: str,
    expected_source_rows: int,
) -> pd.DataFrame:
    aggregated = (
        frame.groupby(group_cols, sort=False, dropna=False)
        .agg(
            symbol=("symbol", "first"),
            adjust=("adjust", "first"),
            bar_start=("datetime", "min"),
            bar_end=("datetime", "max"),
            source_start=("datetime", "min"),
            source_end=("datetime", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            source_rows=("datetime", "count"),
        )
        .reset_index(drop=True)
    )
    aggregated["freq"] = target_freq
    aggregated["available_at"] = aggregated["bar_end"]
    aggregated["expected_source_rows"] = int(expected_source_rows)
    aggregated["is_partial"] = aggregated["source_rows"] < int(expected_source_rows)
    aggregated["progress"] = (aggregated["source_rows"] / int(expected_source_rows)).clip(upper=1.0)
    aggregated = _refresh_pct_chg(aggregated)
    return aggregated[_aggregated_columns()]


def _expected_source_rows(*, base: str, target: str) -> int:
    if base in INTRADAY_MINUTES:
        return EXPECTED_SOURCE_ROWS_FROM_5MIN[target]
    if base == "daily":
        return EXPECTED_SOURCE_ROWS_FROM_DAILY[target]
    raise ValueError(f"{target} cannot be built from {base}")


def _refresh_pct_chg(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["pctChg"] = 0.0
    for _, index in result.groupby(_meta_group_columns(result), sort=False, dropna=False).groups.items():
        part = result.loc[index].sort_values("bar_end")
        prev_close = part["close"].shift(1)
        pct = part["close"] / prev_close - 1
        first_pct = part["close"] / part["open"] - 1
        pct = pct.where(prev_close.notna() & (prev_close != 0), first_pct)
        result.loc[part.index, "pctChg"] = pct
    return result


def _meta_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("symbol", "adjust") if column in frame.columns]


def _sort_columns() -> list[str]:
    return ["symbol", "adjust", "bar_end"]


def _aggregated_columns() -> list[str]:
    return [
        "symbol",
        "freq",
        "adjust",
        "bar_start",
        "bar_end",
        "available_at",
        "source_start",
        "source_end",
        "source_rows",
        "expected_source_rows",
        "is_partial",
        "progress",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pctChg",
    ]


def _empty_aggregated_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_aggregated_columns())
