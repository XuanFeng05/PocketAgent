from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from data_layer.schemas.kline_schema import (
    KLINE_SCHEMA,
    NUMERIC_KLINE_COLUMNS,
    OPTIONAL_KLINE_COLUMNS,
    REQUIRED_KLINE_COLUMNS,
    TEXT_KLINE_COLUMNS,
)
from data_layer.validators.kline_validator import assert_valid_kline_dataframe


TRADE_CALENDAR_COLUMNS = ["date", "is_trading_day", "exchange", "source"]
STOCK_LIQUIDITY_DAILY_COLUMNS = ["symbol", "date", "turn", "source"]
STOCK_STATUS_DAILY_COLUMNS = ["symbol", "date", "is_st", "source"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common provider column names to PocketAgent internal K-line schema.
    """
    rename_map = {
        "code": KLINE_SCHEMA.symbol,
        "date": KLINE_SCHEMA.datetime,
        "time": "time",
        "pctchg": KLINE_SCHEMA.pctChg,
        "pct_chg": KLINE_SCHEMA.pctChg,
    }

    normalized = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return normalized


def _build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the unified datetime column.

    BaoStock daily/weekly/monthly data usually has:
        date

    BaoStock minute data usually has:
        date, time

    Some minute time values may look like:
        20230103103000000
    """
    result = df.copy()

    if KLINE_SCHEMA.datetime not in result.columns:
        if "date" in result.columns:
            result[KLINE_SCHEMA.datetime] = result["date"]
        else:
            return result

    if "time" in result.columns:
        time_str = result["time"].astype(str).str.strip()

        # BaoStock minute time often includes full timestamp like 20230103103000000.
        # Use the first 12 chars: YYYYMMDDHHMM.
        full_ts_mask = time_str.str.len() >= 12
        if full_ts_mask.any():
            parsed = pd.to_datetime(
                time_str.str.slice(0, 12),
                format="%Y%m%d%H%M",
                errors="coerce",
            )
            result.loc[full_ts_mask, KLINE_SCHEMA.datetime] = parsed.loc[full_ts_mask].astype(str)

        # Fallback: combine date + HHMM/HH:MM if needed.
        remaining_mask = pd.to_datetime(result[KLINE_SCHEMA.datetime], format="mixed", errors="coerce").isna()
        if remaining_mask.any() and "date" in result.columns:
            combined = result["date"].astype(str).str.strip() + " " + time_str
            parsed = pd.to_datetime(combined, format="mixed", errors="coerce")
            result.loc[remaining_mask, KLINE_SCHEMA.datetime] = parsed.loc[remaining_mask].astype(str)

    result[KLINE_SCHEMA.datetime] = pd.to_datetime(
        result[KLINE_SCHEMA.datetime],
        errors="coerce",
    )

    return result


def normalize_kline_dataframe(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    source: str = "baostock",
    freq: str = "daily",
    adjust: str = "pre",
) -> pd.DataFrame:
    """
    Convert raw provider K-line data into PocketAgent internal schema.

    The output keeps only the core cross-frequency market fields. Missing
    optional fields are filled with None.

    Required across all frequencies:
        symbol, datetime, open, high, low, close, volume

    Optional or derived:
        amount, pctChg, source, freq, adjust
    """
    if df is None:
        raise ValueError("Input dataframe is None.")

    result = df.copy()
    result = _normalize_columns(result)
    result = _build_datetime(result)

    if symbol is not None:
        result[KLINE_SCHEMA.symbol] = symbol

    result[KLINE_SCHEMA.source] = source
    result[KLINE_SCHEMA.freq] = freq
    result[KLINE_SCHEMA.adjust] = adjust

    for column in OPTIONAL_KLINE_COLUMNS:
        if column not in result.columns:
            result[column] = None

    for column in REQUIRED_KLINE_COLUMNS:
        if column not in result.columns:
            result[column] = None

    # Numeric conversion. Missing optional columns are allowed.
    for column in NUMERIC_KLINE_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if KLINE_SCHEMA.volume in result.columns:
        result[KLINE_SCHEMA.volume] = result[KLINE_SCHEMA.volume].fillna(0.0)

    # Text normalization.
    for column in TEXT_KLINE_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("string")

    # Standard column order first, then preserve any provider extras after that.
    standard_columns = [
        KLINE_SCHEMA.symbol,
        KLINE_SCHEMA.datetime,
        KLINE_SCHEMA.open,
        KLINE_SCHEMA.high,
        KLINE_SCHEMA.low,
        KLINE_SCHEMA.close,
        KLINE_SCHEMA.volume,
        KLINE_SCHEMA.amount,
        KLINE_SCHEMA.pctChg,
        KLINE_SCHEMA.source,
        KLINE_SCHEMA.freq,
        KLINE_SCHEMA.adjust,
    ]

    extra_columns = [column for column in result.columns if column not in standard_columns]
    result = result[standard_columns + extra_columns]

    # Drop rows without datetime or OHLC close price.
    result = result.dropna(subset=[KLINE_SCHEMA.datetime, KLINE_SCHEMA.close])

    result = result.sort_values(
        [KLINE_SCHEMA.symbol, KLINE_SCHEMA.freq, KLINE_SCHEMA.adjust, KLINE_SCHEMA.datetime]
    ).reset_index(drop=True)

    assert_valid_kline_dataframe(result)

    return result


def normalize_trade_calendar_dataframe(
    df: pd.DataFrame,
    *,
    source: str = "baostock",
    exchange: str = "CN",
) -> pd.DataFrame:
    """Normalize provider trading calendar data for storage."""
    if df is None:
        raise ValueError("Input dataframe is None.")

    result = df.copy().rename(
        columns={
            "calendar_date": "date",
            "trade_date": "date",
            "is_open": "is_trading_day",
        }
    )

    for column in ["date", "is_trading_day"]:
        if column not in result.columns:
            result[column] = None

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    result["is_trading_day"] = result["is_trading_day"].map(_to_bool)
    result["exchange"] = str(exchange or "CN")
    result["source"] = source

    result = result.dropna(subset=["date"])
    return result[TRADE_CALENDAR_COLUMNS].drop_duplicates(subset=["exchange", "date"], keep="last").reset_index(drop=True)


def normalize_stock_liquidity_dataframe(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    source: str = "baostock",
) -> pd.DataFrame:
    """Normalize daily liquidity facts such as turnover."""
    if df is None:
        raise ValueError("Input dataframe is None.")

    result = df.copy().rename(columns={"code": "symbol", "trade_date": "date"})

    if symbol is not None:
        result["symbol"] = symbol

    for column in ["symbol", "date", "turn"]:
        if column not in result.columns:
            result[column] = None

    result["symbol"] = result["symbol"].astype(str).str.replace("\ufeff", "", regex=False).str.strip().str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    result["turn"] = pd.to_numeric(result["turn"], errors="coerce")
    result["source"] = source

    result = result.dropna(subset=["symbol", "date"])
    return result[STOCK_LIQUIDITY_DAILY_COLUMNS].drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)


def normalize_stock_status_dataframe(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    source: str = "baostock",
) -> pd.DataFrame:
    """Normalize dated historical ST status for market-rule constraints."""
    if df is None:
        raise ValueError("Input dataframe is None.")

    result = df.copy().rename(
        columns={"code": "symbol", "trade_date": "date", "isST": "is_st"}
    )
    if symbol is not None:
        result["symbol"] = symbol
    for column in ["symbol", "date", "is_st"]:
        if column not in result.columns:
            result[column] = None

    result["symbol"] = result["symbol"].astype(str).str.replace("\ufeff", "", regex=False).str.strip().str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    status = pd.to_numeric(result["is_st"], errors="coerce")
    result["is_st"] = status.map(lambda value: bool(int(value)) if pd.notna(value) else pd.NA).astype("boolean")
    result["source"] = source
    result = result.dropna(subset=["symbol", "date", "is_st"])
    return result[STOCK_STATUS_DAILY_COLUMNS].drop_duplicates(
        subset=["symbol", "date"], keep="last"
    ).reset_index(drop=True)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "open"}


def load_kline_csv(
    path: str | Path,
    *,
    symbol: Optional[str] = None,
    source: str = "csv",
    freq: str = "daily",
    adjust: str = "pre",
) -> pd.DataFrame:
    """
    Load a CSV file and normalize it into PocketAgent K-line schema.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"K-line CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    return normalize_kline_dataframe(
        df,
        symbol=symbol,
        source=source,
        freq=freq,
        adjust=adjust,
    )
