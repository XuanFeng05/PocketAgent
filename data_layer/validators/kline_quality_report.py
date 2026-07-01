from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data_layer.validators.kline_validator import ValidationResult, validate_kline_dataframe


@dataclass
class KlineQualityReport:
    """
    Quality report for one K-line dataframe.
    """

    symbol: str | None
    rows: int
    start_datetime: str | None
    end_datetime: str | None
    validation_ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "rows": self.rows,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "validation_ok": self.validation_ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "null_counts": self.null_counts,
            "duplicate_count": self.duplicate_count,
        }


def build_kline_quality_report(
    df: pd.DataFrame,
) -> KlineQualityReport:
    """
    Build a quality report for one K-line dataframe.
    """
    if df is None or df.empty:
        return KlineQualityReport(
            symbol=None,
            rows=0,
            start_datetime=None,
            end_datetime=None,
            validation_ok=False,
            errors=["K-line dataframe is empty."],
        )

    validation: ValidationResult = validate_kline_dataframe(df)

    symbol: str | None = None
    if "symbol" in df.columns and not df["symbol"].dropna().empty:
        unique_symbols = df["symbol"].dropna().astype(str).unique().tolist()
        symbol = unique_symbols[0] if len(unique_symbols) == 1 else ",".join(unique_symbols[:5])

    start_datetime: str | None = None
    end_datetime: str | None = None
    if "datetime" in df.columns:
        parsed_datetime = pd.to_datetime(df["datetime"], errors="coerce")
        valid_datetime = parsed_datetime.dropna()
        if not valid_datetime.empty:
            start_datetime = str(valid_datetime.min())
            end_datetime = str(valid_datetime.max())

    null_counts = {
        column: int(count)
        for column, count in df.isna().sum().items()
        if int(count) > 0
    }

    duplicate_count = 0
    if {"symbol", "datetime"}.issubset(set(df.columns)):
        duplicate_count = int(df.duplicated(subset=["symbol", "datetime"]).sum())

    return KlineQualityReport(
        symbol=symbol,
        rows=int(len(df)),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        validation_ok=validation.ok,
        errors=validation.errors,
        warnings=validation.warnings,
        null_counts=null_counts,
        duplicate_count=duplicate_count,
    )


def build_quality_report_dataframe(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build quality reports for multiple symbol dataframes.
    """
    rows: list[dict] = []

    for symbol, df in frames.items():
        report = build_kline_quality_report(df)
        item = report.to_dict()
        item["requested_symbol"] = symbol
        rows.append(item)

    return pd.DataFrame(rows)

