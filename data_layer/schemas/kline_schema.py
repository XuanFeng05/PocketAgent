from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KlineSchema:
    """
    PocketAgent internal core K-line schema.

    Required fields are the minimum OHLCV fields available across daily,
    weekly, monthly, and minute data.

    pctChg is a local derived market field calculated by data_layer from
    close / previous close. Non-universal facts such as turnover and ST status
    live in extension tables instead of this core table.
    """

    symbol: str = "symbol"
    datetime: str = "datetime"

    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"

    volume: str = "volume"
    amount: str = "amount"

    pctChg: str = "pctChg"

    source: str = "source"
    freq: str = "freq"
    adjust: str = "adjust"


KLINE_SCHEMA = KlineSchema()


REQUIRED_KLINE_COLUMNS: list[str] = [
    KLINE_SCHEMA.symbol,
    KLINE_SCHEMA.datetime,
    KLINE_SCHEMA.open,
    KLINE_SCHEMA.high,
    KLINE_SCHEMA.low,
    KLINE_SCHEMA.close,
    KLINE_SCHEMA.volume,
]


OPTIONAL_KLINE_COLUMNS: list[str] = [
    KLINE_SCHEMA.amount,
    KLINE_SCHEMA.pctChg,
    KLINE_SCHEMA.source,
    KLINE_SCHEMA.freq,
    KLINE_SCHEMA.adjust,
]


NUMERIC_KLINE_COLUMNS: list[str] = [
    KLINE_SCHEMA.open,
    KLINE_SCHEMA.high,
    KLINE_SCHEMA.low,
    KLINE_SCHEMA.close,
    KLINE_SCHEMA.volume,
    KLINE_SCHEMA.amount,
    KLINE_SCHEMA.pctChg,
]


TEXT_KLINE_COLUMNS: list[str] = [
    KLINE_SCHEMA.symbol,
    KLINE_SCHEMA.source,
    KLINE_SCHEMA.freq,
    KLINE_SCHEMA.adjust,
]


def get_all_kline_columns() -> list[str]:
    """
    Return all standard K-line columns used by PocketAgent.
    """
    return REQUIRED_KLINE_COLUMNS + OPTIONAL_KLINE_COLUMNS


def find_missing_kline_columns(columns: list[str]) -> list[str]:
    """
    Return required K-line columns that are missing from the given column list.
    """
    column_set = set(columns)
    return [column for column in REQUIRED_KLINE_COLUMNS if column not in column_set]


def is_valid_kline_columns(columns: list[str]) -> bool:
    """
    Check whether the given columns satisfy the minimum K-line requirement.
    """
    return len(find_missing_kline_columns(columns)) == 0
