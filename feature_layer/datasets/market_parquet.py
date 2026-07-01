from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import duckdb
import pandas as pd

from data_layer.storage.duckdb_storage import (
    KLINE_SELECT_COLUMNS,
    KLINE_TABLE_NAME,
    STOCK_LIQUIDITY_DAILY_TABLE_NAME,
    STOCK_STATUS_DAILY_TABLE_NAME,
    connect_duckdb_read_only,
)
from feature_layer.builders.aggregation import normalize_frequency


@dataclass(frozen=True)
class MarketParquetCacheInfo:
    cache_dir: Path
    fingerprint: str
    symbols: tuple[str, ...]
    adjust: str
    trade_freq: str
    exported_files: int
    reused_files: int
    seconds: float


def market_parquet_cache_root(default_parent: str | Path) -> Path:
    return Path(default_parent)


def resolve_market_parquet_cache_dir(
    db_path: str | Path,
    *,
    cache_root: str | Path,
    adjust: str,
    trade_freq: str,
    end: str | None,
) -> Path:
    db_file = Path(db_path).resolve()
    try:
        stat = db_file.stat()
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
    except FileNotFoundError:
        size = 0
        mtime_ns = 0
    payload = {
        "version": 1,
        "db_path": str(db_file),
        "db_size": int(size),
        "db_mtime_ns": int(mtime_ns),
        "adjust": str(adjust),
        "trade_freq": normalize_frequency(trade_freq),
        "end": end or "",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return Path(cache_root) / fingerprint


def ensure_market_parquet_cache(
    db_path: str | Path,
    *,
    cache_root: str | Path,
    symbols: Iterable[str],
    adjust: str,
    trade_freq: str,
    end: str | None,
    force: bool = False,
) -> MarketParquetCacheInfo:
    """Export source market tables to symbol/frequency parquet shards.

    This is the source-side cache used by the distributed-ready feature builder:
    workers read independent parquet shards instead of all concurrently scanning
    the same market.duckdb file.
    """

    started = perf_counter()
    symbol_values = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols if str(symbol).strip()))
    normalized_trade_freq = normalize_frequency(trade_freq)
    cache_dir = resolve_market_parquet_cache_dir(
        db_path,
        cache_root=cache_root,
        adjust=adjust,
        trade_freq=normalized_trade_freq,
        end=end,
    )
    if force and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    reused = 0
    with connect_duckdb_read_only(db_path) as conn:
        conn.execute("SET threads = 1")
        for symbol in symbol_values:
            for freq in (normalized_trade_freq, "daily"):
                out = _kline_file(cache_dir, symbol=symbol, freq=freq, adjust=adjust)
                if out.exists():
                    reused += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                query = _kline_export_query(symbol=symbol, freq=freq, adjust=adjust, end=end)
                _copy_query_to_parquet(conn, query, out)
                exported += 1

            for table_name, file_func, query_func in (
                (STOCK_LIQUIDITY_DAILY_TABLE_NAME, _liquidity_file, _liquidity_export_query),
                (STOCK_STATUS_DAILY_TABLE_NAME, _status_file, _status_export_query),
            ):
                out = file_func(cache_dir, symbol=symbol)
                if out.exists():
                    reused += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                if not _table_exists(conn, table_name):
                    _write_empty_table_parquet(conn, out, _empty_columns_for_table(table_name))
                    exported += 1
                    continue
                query = query_func(symbol=symbol, end=end)
                _copy_query_to_parquet(conn, query, out)
                exported += 1

    manifest = {
        "version": 1,
        "db_path": str(Path(db_path).resolve()),
        "symbols": list(symbol_values),
        "adjust": str(adjust),
        "trade_freq": normalized_trade_freq,
        "end": end,
        "exported_files": exported,
        "reused_files": reused,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return MarketParquetCacheInfo(
        cache_dir=cache_dir,
        fingerprint=cache_dir.name,
        symbols=symbol_values,
        adjust=str(adjust),
        trade_freq=normalized_trade_freq,
        exported_files=int(exported),
        reused_files=int(reused),
        seconds=float(perf_counter() - started),
    )


def load_kline_from_market_parquet_cache(
    cache_dir: str | Path,
    *,
    symbols: Iterable[str],
    freq: str,
    adjust: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    files = [
        _kline_file(Path(cache_dir), symbol=str(symbol).upper(), freq=normalize_frequency(freq), adjust=adjust)
        for symbol in symbols
    ]
    columns = list(KLINE_SELECT_COLUMNS)
    frame = _read_parquet_files(files, columns=columns)
    if frame.empty:
        return frame
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if start is not None:
        frame = frame.loc[frame["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["datetime"] <= pd.Timestamp(_end_bound(end))]
    return frame.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_stock_liquidity_from_market_parquet_cache(
    cache_dir: str | Path,
    *,
    symbols: Iterable[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    files = [_liquidity_file(Path(cache_dir), symbol=str(symbol).upper()) for symbol in symbols]
    frame = _read_parquet_files(files, columns=["symbol", "date", "turn", "source"])
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    if start is not None:
        frame = frame.loc[pd.to_datetime(frame["date"]) >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[pd.to_datetime(frame["date"]) <= pd.Timestamp(end)]
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_stock_status_from_market_parquet_cache(
    cache_dir: str | Path,
    *,
    symbols: Iterable[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    files = [_status_file(Path(cache_dir), symbol=str(symbol).upper()) for symbol in symbols]
    frame = _read_parquet_files(files, columns=["symbol", "date", "is_st", "source"])
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    if start is not None:
        frame = frame.loc[pd.to_datetime(frame["date"]) >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[pd.to_datetime(frame["date"]) <= pd.Timestamp(end)]
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def _kline_file(cache_dir: Path, *, symbol: str, freq: str, adjust: str) -> Path:
    return cache_dir / "kline" / f"freq={_safe(freq)}" / f"adjust={_safe(adjust)}" / f"{_safe(symbol)}.parquet"


def _liquidity_file(cache_dir: Path, *, symbol: str) -> Path:
    return cache_dir / "stock_liquidity_daily" / f"{_safe(symbol)}.parquet"


def _status_file(cache_dir: Path, *, symbol: str) -> Path:
    return cache_dir / "stock_status_daily" / f"{_safe(symbol)}.parquet"


def _safe(value: object) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_").strip()


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        rows = conn.execute("SHOW TABLES").fetchall()
    except Exception:
        return False
    return str(table_name) in {str(row[0]) for row in rows}


def _kline_export_query(*, symbol: str, freq: str, adjust: str, end: str | None) -> str:
    columns = ", ".join(KLINE_SELECT_COLUMNS)
    query = (
        f"SELECT {columns} FROM {KLINE_TABLE_NAME} "
        f"WHERE symbol = {_sql_string(symbol)} "
        f"AND freq = {_sql_string(freq)} "
        f"AND adjust = {_sql_string(adjust)}"
    )
    if end is not None:
        query += f" AND datetime <= {_sql_string(_end_bound(end))}"
    query += " ORDER BY symbol, datetime"
    return query


def _liquidity_export_query(*, symbol: str, end: str | None) -> str:
    query = (
        f"SELECT symbol, date, turn, source FROM {STOCK_LIQUIDITY_DAILY_TABLE_NAME} "
        f"WHERE symbol = {_sql_string(symbol)}"
    )
    if end is not None:
        query += f" AND date <= {_sql_string(end)}"
    query += " ORDER BY symbol, date"
    return query


def _status_export_query(*, symbol: str, end: str | None) -> str:
    query = (
        f"SELECT symbol, date, is_st, source FROM {STOCK_STATUS_DAILY_TABLE_NAME} "
        f"WHERE symbol = {_sql_string(symbol)}"
    )
    if end is not None:
        query += f" AND date <= {_sql_string(end)}"
    query += " ORDER BY symbol, date"
    return query


def _copy_query_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    conn.execute(f"COPY ({query}) TO {_sql_string(temp)} (FORMAT PARQUET)")
    temp.replace(output_path)


def _write_empty_table_parquet(
    conn: duckdb.DuckDBPyConnection,
    output_path: Path,
    columns: dict[str, str],
) -> None:
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    select_sql = ", ".join(f"CAST(NULL AS {typ}) AS {name}" for name, typ in columns.items())
    conn.execute(
        f"COPY (SELECT {select_sql} WHERE FALSE) TO {_sql_string(temp)} (FORMAT PARQUET)"
    )
    temp.replace(output_path)


def _empty_columns_for_table(table_name: str) -> dict[str, str]:
    if table_name == STOCK_LIQUIDITY_DAILY_TABLE_NAME:
        return {"symbol": "VARCHAR", "date": "DATE", "turn": "DOUBLE", "source": "VARCHAR"}
    return {"symbol": "VARCHAR", "date": "DATE", "is_st": "BOOLEAN", "source": "VARCHAR"}


def _read_parquet_files(files: list[Path], *, columns: list[str]) -> pd.DataFrame:
    existing = [path for path in files if path.exists()]
    if not existing:
        return pd.DataFrame(columns=columns)
    with duckdb.connect(":memory:") as conn:
        conn.execute("SET threads = 1")
        frame = conn.execute(f"SELECT * FROM read_parquet({_sql_list(existing)})").fetchdf()
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_list(values: Iterable[str | Path]) -> str:
    return "[" + ", ".join(_sql_string(value) for value in values) + "]"


def _end_bound(end: str | object) -> object:
    if isinstance(end, str) and len(end.strip()) == 10 and end.strip()[4] == "-" and end.strip()[7] == "-":
        return f"{end.strip()} 23:59:59.999999"
    return end
