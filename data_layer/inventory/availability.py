from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data_layer.storage.duckdb_storage import get_kline_inventory


@dataclass
class SymbolAvailability:
    """
    Data availability status for one symbol.
    """

    symbol: str
    available: bool
    rows: int = 0
    start_datetime: str | None = None
    end_datetime: str | None = None
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "available": self.available,
            "rows": self.rows,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "issues": self.issues,
        }


def build_availability_report(
    db_path: str | Path,
    *,
    symbols: list[str] | None = None,
    min_rows: int = 200,
    required_start: str | None = None,
    required_end: str | None = None,
) -> pd.DataFrame:
    """
    Build data availability report from DuckDB inventory.

    Args:
        db_path:
            DuckDB database path.
        symbols:
            Symbols to check. If None, all symbols in the database are checked.
        min_rows:
            Minimum required rows for a symbol to be considered available.
        required_start:
            Optional required start date, YYYY-MM-DD.
        required_end:
            Optional required end date, YYYY-MM-DD.

    Returns:
        DataFrame with columns:
            symbol, available, rows, start_datetime, end_datetime, issues
    """
    inventory = get_kline_inventory(db_path)

    if inventory.empty:
        if symbols is None:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "available",
                    "rows",
                    "start_datetime",
                    "end_datetime",
                    "issues",
                ]
            )

        return pd.DataFrame(
            [
                SymbolAvailability(
                    symbol=symbol,
                    available=False,
                    rows=0,
                    issues=["No K-line data found in database."],
                ).to_dict()
                for symbol in symbols
            ]
        )

    inventory = inventory.copy()
    inventory["symbol"] = inventory["symbol"].astype(str)
    inventory["freq"] = inventory.get("freq", "").astype(str)
    inventory["adjust"] = inventory.get("adjust", "").astype(str)

    # Coverage / available-universe uses daily K-line as the project-level
    # availability baseline. The inventory table contains one row per
    # symbol/freq/adjust slice; if we simply keep the last row per symbol,
    # weekly rows can overwrite daily rows because of alphabetical ordering.
    # Prefer daily data for each symbol and only fall back to any available
    # slice when daily data is absent.
    daily_inventory = inventory.loc[inventory["freq"].str.lower().eq("daily")].copy()
    baseline_inventory = daily_inventory if not daily_inventory.empty else inventory

    if symbols is None:
        check_symbols = sorted(baseline_inventory["symbol"].unique().tolist())
    else:
        check_symbols = [str(symbol) for symbol in symbols]

    inventory_by_symbol = {}
    for symbol, group in inventory.groupby("symbol", sort=False):
        daily_group = group.loc[group["freq"].str.lower().eq("daily")]
        selected = daily_group.iloc[0] if not daily_group.empty else group.iloc[0]
        inventory_by_symbol[str(symbol)] = selected

    results: list[SymbolAvailability] = []

    required_start_ts = pd.to_datetime(required_start) if required_start else None
    required_end_ts = pd.to_datetime(required_end) if required_end else None

    for symbol in check_symbols:
        issues: list[str] = []

        row = inventory_by_symbol.get(symbol)
        if row is None:
            results.append(
                SymbolAvailability(
                    symbol=symbol,
                    available=False,
                    rows=0,
                    issues=["Symbol not found in database."],
                )
            )
            continue

        rows = int(row["rows"])
        start_datetime = pd.to_datetime(row["start_datetime"])
        end_datetime = pd.to_datetime(row["end_datetime"])

        if rows < min_rows:
            issues.append(f"Rows below minimum requirement: {rows} < {min_rows}")

        if required_start_ts is not None and start_datetime > required_start_ts:
            issues.append(
                f"Start date is later than required: {start_datetime.date()} > {required_start_ts.date()}"
            )

        if required_end_ts is not None and end_datetime < required_end_ts:
            issues.append(
                f"End date is earlier than required: {end_datetime.date()} < {required_end_ts.date()}"
            )

        results.append(
            SymbolAvailability(
                symbol=symbol,
                available=len(issues) == 0,
                rows=rows,
                start_datetime=str(start_datetime.date()),
                end_datetime=str(end_datetime.date()),
                issues=issues,
            )
        )

    return pd.DataFrame([item.to_dict() for item in results])


def get_available_symbols(
    db_path: str | Path,
    *,
    symbols: list[str] | None = None,
    min_rows: int = 200,
    required_start: str | None = None,
    required_end: str | None = None,
) -> list[str]:
    """
    Return symbols that pass the availability check.
    """
    report = build_availability_report(
        db_path,
        symbols=symbols,
        min_rows=min_rows,
        required_start=required_start,
        required_end=required_end,
    )

    if report.empty:
        return []

    return report.loc[report["available"] == True, "symbol"].astype(str).tolist()

