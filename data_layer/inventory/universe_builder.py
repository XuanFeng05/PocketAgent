from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_layer.inventory.availability import build_availability_report


def read_symbol_list(path: str | Path) -> list[str]:
    """
    Read symbols from a text file.

    Empty lines and lines starting with # are ignored.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Symbol list file not found: {file_path}")

    symbols: list[str] = []

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line)

    return symbols


def write_symbol_list(symbols: list[str], path: str | Path) -> None:
    """
    Write symbols to a text file, one symbol per line.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    unique_symbols = sorted(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))

    file_path.write_text(
        "\n".join(unique_symbols) + ("\n" if unique_symbols else ""),
        encoding="utf-8",
    )


def build_available_universe(
    *,
    db_path: str | Path,
    candidate_symbols: list[str],
    output_path: str | Path,
    report_path: str | Path | None = None,
    min_rows: int = 200,
    required_start: str | None = None,
    required_end: str | None = None,
) -> pd.DataFrame:
    """
    Build available universe from candidate symbols and DuckDB coverage.

    Args:
        db_path:
            DuckDB database path.
        candidate_symbols:
            Candidate symbols to check.
        output_path:
            Text file to save available symbols.
        report_path:
            Optional CSV file to save full availability report.
        min_rows:
            Minimum required K-line rows.
        required_start:
            Required start date, YYYY-MM-DD.
        required_end:
            Required end date, YYYY-MM-DD.

    Returns:
        Full availability report dataframe.
    """
    report = build_availability_report(
        db_path,
        symbols=candidate_symbols,
        min_rows=min_rows,
        required_start=required_start,
        required_end=required_end,
    )

    if report.empty:
        available_symbols: list[str] = []
    else:
        available_symbols = report.loc[report["available"] == True, "symbol"].astype(str).tolist()

    write_symbol_list(available_symbols, output_path)

    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_file, index=False, encoding="utf-8-sig")

    return report


def build_available_universe_from_file(
    *,
    db_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    min_rows: int = 200,
    required_start: str | None = None,
    required_end: str | None = None,
) -> pd.DataFrame:
    """
    Build available universe from a candidate symbol text file.
    """
    candidate_symbols = read_symbol_list(candidate_path)

    return build_available_universe(
        db_path=db_path,
        candidate_symbols=candidate_symbols,
        output_path=output_path,
        report_path=report_path,
        min_rows=min_rows,
        required_start=required_start,
        required_end=required_end,
    )
