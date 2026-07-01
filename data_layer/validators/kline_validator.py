from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from data_layer.schemas.kline_schema import (
    KLINE_SCHEMA,
    NUMERIC_KLINE_COLUMNS,
    REQUIRED_KLINE_COLUMNS,
    find_missing_kline_columns,
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            message = "K-line dataframe validation failed:\n" + "\n".join(
                f"- {error}" for error in self.errors
            )
            raise ValueError(message)


def _check_required_columns(df: pd.DataFrame, errors: list[str]) -> None:
    missing = find_missing_kline_columns(list(df.columns))
    if missing:
        errors.append(f"Missing required columns: {missing}")


def _check_empty_dataframe(df: pd.DataFrame, errors: list[str], warnings: list[str]) -> None:
    if df.empty:
        warnings.append("K-line dataframe is empty.")


def _check_datetime(df: pd.DataFrame, errors: list[str]) -> None:
    if KLINE_SCHEMA.datetime not in df.columns:
        return

    parsed = pd.to_datetime(df[KLINE_SCHEMA.datetime], format="mixed", errors="coerce")
    invalid_count = int(parsed.isna().sum())

    if invalid_count > 0:
        errors.append(f"Invalid datetime values: {invalid_count}")


def _check_required_numeric(df: pd.DataFrame, errors: list[str]) -> None:
    required_numeric = [
        KLINE_SCHEMA.open,
        KLINE_SCHEMA.high,
        KLINE_SCHEMA.low,
        KLINE_SCHEMA.close,
    ]

    for column in required_numeric:
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(numeric.isna().sum())

        if invalid_count > 0:
            errors.append(f"Invalid numeric values in {column}: {invalid_count}")


def _check_volume_numeric(df: pd.DataFrame, warnings: list[str]) -> None:
    if KLINE_SCHEMA.volume not in df.columns:
        return

    numeric = pd.to_numeric(df[KLINE_SCHEMA.volume], errors="coerce")
    invalid_count = int(numeric.isna().sum())

    if invalid_count > 0:
        warnings.append(f"Invalid volume values treated as zero: {invalid_count}")


def _check_optional_numeric(df: pd.DataFrame, warnings: list[str]) -> None:
    optional_numeric = [
        column
        for column in NUMERIC_KLINE_COLUMNS
        if column not in {
            KLINE_SCHEMA.open,
            KLINE_SCHEMA.high,
            KLINE_SCHEMA.low,
            KLINE_SCHEMA.close,
            KLINE_SCHEMA.volume,
        }
    ]

    for column in optional_numeric:
        if column not in df.columns:
            continue

        # Optional fields are allowed to be missing for some frequencies.
        non_empty = df[column].notna()
        if not bool(non_empty.any()):
            continue

        numeric = pd.to_numeric(df.loc[non_empty, column], errors="coerce")
        invalid_count = int(numeric.isna().sum())

        if invalid_count > 0:
            warnings.append(f"Invalid optional numeric values in {column}: {invalid_count}")


def _check_ohlc_relationship(df: pd.DataFrame, warnings: list[str]) -> None:
    needed = [
        KLINE_SCHEMA.open,
        KLINE_SCHEMA.high,
        KLINE_SCHEMA.low,
        KLINE_SCHEMA.close,
    ]

    if any(column not in df.columns for column in needed):
        return

    open_ = pd.to_numeric(df[KLINE_SCHEMA.open], errors="coerce")
    high = pd.to_numeric(df[KLINE_SCHEMA.high], errors="coerce")
    low = pd.to_numeric(df[KLINE_SCHEMA.low], errors="coerce")
    close = pd.to_numeric(df[KLINE_SCHEMA.close], errors="coerce")

    valid = open_.notna() & high.notna() & low.notna() & close.notna()

    if not bool(valid.any()):
        return

    bad_high = valid & ((high < open_) | (high < close) | (high < low))
    bad_low = valid & ((low > open_) | (low > close) | (low > high))

    bad_count = int((bad_high | bad_low).sum())
    if bad_count > 0:
        warnings.append(f"Suspicious OHLC relationship rows: {bad_count}")


def _check_negative_values(df: pd.DataFrame, warnings: list[str]) -> None:
    non_negative_columns = [
        KLINE_SCHEMA.open,
        KLINE_SCHEMA.high,
        KLINE_SCHEMA.low,
        KLINE_SCHEMA.close,
        KLINE_SCHEMA.volume,
        KLINE_SCHEMA.amount,
    ]

    for column in non_negative_columns:
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        negative_count = int((numeric < 0).sum())

        if negative_count > 0:
            warnings.append(f"Negative values in {column}: {negative_count}")


def _check_duplicate_keys(df: pd.DataFrame, warnings: list[str]) -> None:
    key_columns = [
        KLINE_SCHEMA.symbol,
        KLINE_SCHEMA.datetime,
        KLINE_SCHEMA.freq,
        KLINE_SCHEMA.adjust,
    ]

    if any(column not in df.columns for column in key_columns):
        return

    duplicate_count = int(df.duplicated(subset=key_columns).sum())
    if duplicate_count > 0:
        warnings.append(f"Duplicate K-line keys: {duplicate_count}")


def validate_kline_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Validate a PocketAgent K-line dataframe.

    Strict errors:
        - missing required columns
        - invalid datetime
        - invalid required numeric OHLC fields

    Warnings:
        - empty dataframe
        - invalid volume values, which storage normalizes to zero
        - missing/invalid optional fields
        - suspicious OHLC relationship
        - duplicate keys
        - negative numeric values

    pctChg is optional on input because data_layer refreshes it after storage.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df is None:
        return ValidationResult(ok=False, errors=["Input dataframe is None."])

    _check_required_columns(df, errors)
    _check_empty_dataframe(df, errors, warnings)

    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    _check_datetime(df, errors)
    _check_required_numeric(df, errors)
    _check_volume_numeric(df, warnings)

    _check_optional_numeric(df, warnings)
    _check_ohlc_relationship(df, warnings)
    _check_negative_values(df, warnings)
    _check_duplicate_keys(df, warnings)

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def assert_valid_kline_dataframe(df: pd.DataFrame) -> None:
    """
    Raise ValueError if K-line dataframe is invalid.
    """
    result = validate_kline_dataframe(df)
    result.raise_if_failed()


def validate_kline_columns(columns: Iterable[str]) -> ValidationResult:
    """
    Validate a column collection without requiring actual data rows.
    """
    column_list = list(columns)
    missing = [column for column in REQUIRED_KLINE_COLUMNS if column not in column_list]

    if missing:
        return ValidationResult(
            ok=False,
            errors=[f"Missing required columns: {missing}"],
        )

    return ValidationResult(ok=True)
