from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data_layer.storage.duckdb_storage import load_kline_from_duckdb


@dataclass
class SymbolConsistencyReport:
    """
    Consistency report for one symbol.
    """

    symbol: str
    ok: bool
    rows: int = 0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "ok": self.ok,
            "rows": self.rows,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def check_kline_dataframe_consistency(
    df: pd.DataFrame,
    *,
    symbol: str | None = None,
    max_calendar_gap_days: int = 14,
) -> SymbolConsistencyReport:
    """
    Check consistency for one symbol's K-line dataframe.

    This is not a strict trading calendar validator. It only catches obvious
    data problems and suspicious large calendar gaps.
    """
    if df is None or df.empty:
        return SymbolConsistencyReport(
            symbol=symbol or "",
            ok=False,
            rows=0,
            issues=["No data."],
        )

    working = df.copy()

    if symbol is None:
        if "symbol" in working.columns and not working["symbol"].dropna().empty:
            symbol = str(working["symbol"].dropna().iloc[0])
        else:
            symbol = ""

    issues: list[str] = []
    warnings: list[str] = []

    required_columns = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
    missing_columns = [column for column in required_columns if column not in working.columns]
    if missing_columns:
        return SymbolConsistencyReport(
            symbol=symbol,
            ok=False,
            rows=int(len(working)),
            issues=[f"Missing required columns: {missing_columns}"],
        )

    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    bad_datetime_count = int(working["datetime"].isna().sum())
    if bad_datetime_count > 0:
        issues.append(f"Invalid datetime rows: {bad_datetime_count}")

    working = working.dropna(subset=["datetime"])
    working = working.sort_values("datetime").reset_index(drop=True)

    duplicate_count = int(working.duplicated(subset=["symbol", "datetime"]).sum())
    if duplicate_count > 0:
        issues.append(f"Duplicated symbol-datetime rows: {duplicate_count}")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
        bad_count = int(working[column].isna().sum())
        if bad_count > 0:
            issues.append(f"Non-numeric values in {column}: {bad_count}")

    if issues:
        return SymbolConsistencyReport(
            symbol=symbol,
            ok=False,
            rows=int(len(working)),
            issues=issues,
            warnings=warnings,
        )

    high = working["high"]
    low = working["low"]
    open_ = working["open"]
    close = working["close"]
    volume = working["volume"]

    high_less_than_low = int((high < low).sum())
    if high_less_than_low > 0:
        issues.append(f"Rows with high < low: {high_less_than_low}")

    high_less_than_open = int((high < open_).sum())
    if high_less_than_open > 0:
        warnings.append(f"Rows with high < open: {high_less_than_open}")

    high_less_than_close = int((high < close).sum())
    if high_less_than_close > 0:
        warnings.append(f"Rows with high < close: {high_less_than_close}")

    low_greater_than_open = int((low > open_).sum())
    if low_greater_than_open > 0:
        warnings.append(f"Rows with low > open: {low_greater_than_open}")

    low_greater_than_close = int((low > close).sum())
    if low_greater_than_close > 0:
        warnings.append(f"Rows with low > close: {low_greater_than_close}")

    negative_price_count = int(((open_ < 0) | (high < 0) | (low < 0) | (close < 0)).sum())
    if negative_price_count > 0:
        issues.append(f"Rows with negative prices: {negative_price_count}")

    negative_volume_count = int((volume < 0).sum())
    if negative_volume_count > 0:
        issues.append(f"Rows with negative volume: {negative_volume_count}")

    if len(working) >= 2:
        gaps = working["datetime"].diff().dt.days.dropna()
        large_gap_count = int((gaps > max_calendar_gap_days).sum())
        if large_gap_count > 0:
            warnings.append(
                f"Large calendar gaps greater than {max_calendar_gap_days} days: {large_gap_count}"
            )

    return SymbolConsistencyReport(
        symbol=symbol,
        ok=len(issues) == 0,
        rows=int(len(working)),
        issues=issues,
        warnings=warnings,
    )


def build_consistency_report(
    db_path: str | Path,
    *,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    max_calendar_gap_days: int = 14,
) -> pd.DataFrame:
    """
    Build consistency report from DuckDB K-line data.
    """
    if symbols is None:
        df = load_kline_from_duckdb(db_path, start=start, end=end)
        if df.empty:
            return pd.DataFrame(columns=["symbol", "ok", "rows", "issues", "warnings"])
        symbols = sorted(df["symbol"].astype(str).unique().tolist())
    else:
        df = load_kline_from_duckdb(db_path, symbols=symbols, start=start, end=end)

    reports: list[dict] = []

    for symbol in symbols:
        symbol_df = df[df["symbol"].astype(str) == str(symbol)].copy()
        report = check_kline_dataframe_consistency(
            symbol_df,
            symbol=str(symbol),
            max_calendar_gap_days=max_calendar_gap_days,
        )
        reports.append(report.to_dict())

    return pd.DataFrame(reports)

