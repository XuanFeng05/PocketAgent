from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from data_layer.validators.kline_validator import assert_valid_kline_dataframe


KLINE_TABLE_NAME = "kline_bars"
DOWNLOAD_MANIFEST_TABLE_NAME = "download_manifest"
TRADE_CALENDAR_TABLE_NAME = "trade_calendar"
STOCK_LIQUIDITY_DAILY_TABLE_NAME = "stock_liquidity_daily"
STOCK_STATUS_DAILY_TABLE_NAME = "stock_status_daily"
DERIVED_BAR_MANIFEST_TABLE_NAME = "derived_bar_manifest"

KLINE_SELECT_COLUMNS: list[str] = [
    "symbol",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pctChg",
    "source",
    "freq",
    "adjust",
]

KLINE_HASH_COLUMNS: list[str] = list(KLINE_SELECT_COLUMNS)

KLINE_COLUMN_TYPES: dict[str, str] = {
    "symbol": "VARCHAR",
    "datetime": "TIMESTAMP",
    "open": "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE",
    "close": "DOUBLE",
    "volume": "DOUBLE",
    "amount": "DOUBLE",
    "pctChg": "DOUBLE",
    "source": "VARCHAR",
    "freq": "VARCHAR",
    "adjust": "VARCHAR",
}



def _partitioned_market_root_if_available(path: str | Path) -> Path | None:
    """Return shard root when catalog + market_parts are available.

    This keeps legacy function names working while the project moves from a
    single market.duckdb file to partitioned parquet shards.
    """
    raw = Path(path)
    candidates: list[Path] = []
    if raw.suffix.lower() == ".duckdb" and raw.name.lower() == "market.duckdb":
        candidates.append(raw.parent)
    elif raw.suffix.lower() != ".duckdb":
        candidates.append(raw)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "market_catalog.duckdb").exists() and (candidate / "market_parts").exists():
            return candidate
    return None


def _legacy_duckdb_file_if_available(path: str | Path) -> Path | None:
    """Return a legacy DuckDB file path, never a shard-storage directory.

    Several dashboard inputs now point at ``runtime_layer/data`` because that is
    the market shard root.  During a fresh setup that directory may already
    exist before ``market_catalog.duckdb`` is created.  Legacy DuckDB readers
    must treat that as "no legacy DB" instead of asking DuckDB to open the
    directory as a database file.
    """
    db_file = Path(path)
    if not db_file.exists() or db_file.is_dir():
        return None
    return db_file


# Known index names used by earlier versions. They can block none/pre coexistence
# if they were created without `adjust`, so normal initialization removes them.
KNOWN_KLINE_INDEXES: tuple[str, ...] = (
    "idx_kline_daily_symbol_datetime",
    "idx_kline_daily_symbol_datetime_freq",
    "idx_kline_daily_symbol_datetime_freq_adjust",
    "idx_kline_bars_symbol_datetime",
    "idx_kline_bars_symbol_datetime_freq",
    "idx_kline_bars_symbol_datetime_freq_adjust",
)


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure the parent directory of a file path exists."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def connect_duckdb(db_path: str | Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection.

    Use read_only=True for concurrent feature-build readers. DuckDB permits
    multiple read-only processes, while the default read-write connection takes
    a file lock on Windows and blocks worker processes from opening the same DB.
    """
    db_file = Path(db_path) if read_only else ensure_parent_dir(db_path)
    return duckdb.connect(str(db_file), read_only=read_only)


def connect_duckdb_read_only(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection for concurrent readers."""
    return connect_duckdb(db_path, read_only=True)


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    try:
        rows = conn.execute("SHOW TABLES").fetchall()
    except Exception:
        return False
    return table_name in {str(row[0]) for row in rows}


def _get_existing_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    except duckdb.CatalogException:
        return set()
    return {str(row[1]) for row in rows}


def _ensure_kline_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Add missing extended K-line columns to an existing table."""
    if not _table_exists(conn, table_name):
        return

    existing = _get_existing_columns(conn, table_name)
    for column in KLINE_SELECT_COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {KLINE_COLUMN_TYPES[column]}")
            existing.add(column)

    # Metadata columns used by current storage. They are optional for old data.
    metadata_columns = {
        "created_at": "TIMESTAMP",
        "updated_at": "TIMESTAMP",
        "row_hash": "VARCHAR",
    }
    for column, column_type in metadata_columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {column_type}")
            existing.add(column)


def _drop_known_kline_indexes(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Drop known legacy unique indexes that can prevent multi-adjust storage."""
    for index_name in KNOWN_KLINE_INDEXES:
        try:
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        except Exception:
            # Index cleanup must never block normal download/save.
            pass

    # Best-effort cleanup for user/local indexes containing the table name.
    # This is intentionally conservative: only names starting with idx_<table>.
    try:
        index_rows = conn.execute("SELECT index_name FROM duckdb_indexes() WHERE table_name = ?", [table_name]).fetchall()
    except Exception:
        return

    prefix = f"idx_{table_name}_"
    for (index_name,) in index_rows:
        name = str(index_name)
        if name.startswith(prefix):
            try:
                conn.execute(f"DROP INDEX IF EXISTS {name}")
            except Exception:
                pass


def _kline_select_exprs(conn: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    """Return SELECT expressions that tolerate old tables missing new columns."""
    existing = _get_existing_columns(conn, table_name)
    return [column if column in existing else f"NULL AS {column}" for column in KLINE_SELECT_COLUMNS]


def _normalize_datetime_bound(value: str | object | None, *, is_end: bool = False) -> object | None:
    """Treat YYYY-MM-DD end filters as inclusive through the whole day."""
    if value is None:
        return None
    if is_end and isinstance(value, str):
        text = value.strip()
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            return f"{text} 23:59:59.999999"
    return value


def init_duckdb(db_path: str | Path, *, table_name: str = KLINE_TABLE_NAME) -> None:
    """
    Initialize DuckDB storage.

    Current write strategy is delete-then-insert using:
        symbol + datetime + freq + adjust

    We intentionally do not depend on PRIMARY KEY / INSERT OR REPLACE because
    existing local DuckDB files may contain legacy unique constraints. Dropping
    known legacy indexes and using explicit deletes keeps none/pre data separate.
    """
    db_file = ensure_parent_dir(db_path)
    conn = duckdb.connect(str(db_file))
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                symbol VARCHAR NOT NULL,
                datetime TIMESTAMP NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                pctChg DOUBLE,
                source VARCHAR,
                freq VARCHAR NOT NULL,
                adjust VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                row_hash VARCHAR
            )
            """
        )
        _ensure_kline_columns(conn, table_name)
        _drop_known_kline_indexes(conn, table_name)

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DOWNLOAD_MANIFEST_TABLE_NAME} (
                id BIGINT,
                job_id VARCHAR,
                symbol VARCHAR NOT NULL,
                freq VARCHAR,
                adjust VARCHAR,
                source VARCHAR,
                requested_start TIMESTAMP NOT NULL,
                requested_end TIMESTAMP NOT NULL,
                actual_start TIMESTAMP,
                actual_end TIMESTAMP,
                rows BIGINT,
                data_hash VARCHAR,
                hash_algorithm VARCHAR,
                status VARCHAR NOT NULL,
                downloaded_at TIMESTAMP NOT NULL,
                report_path VARCHAR,
                error VARCHAR
            )
            """
        )
        try:
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{DOWNLOAD_MANIFEST_TABLE_NAME}_lookup
                ON {DOWNLOAD_MANIFEST_TABLE_NAME}(symbol, freq, adjust, requested_start, requested_end, status)
                """
            )
        except Exception:
            # Manifest index is an optimization only.
            pass

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TRADE_CALENDAR_TABLE_NAME} (
                date DATE NOT NULL,
                is_trading_day BOOLEAN NOT NULL,
                exchange VARCHAR NOT NULL,
                source VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {STOCK_LIQUIDITY_DAILY_TABLE_NAME} (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                turn DOUBLE,
                source VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {STOCK_STATUS_DAILY_TABLE_NAME} (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                is_st BOOLEAN NOT NULL,
                source VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DERIVED_BAR_MANIFEST_TABLE_NAME} (
                symbol VARCHAR NOT NULL,
                base_freq VARCHAR NOT NULL,
                target_freq VARCHAR NOT NULL,
                adjust VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                base_start TIMESTAMP,
                base_end TIMESTAMP,
                base_rows BIGINT,
                base_hash VARCHAR,
                target_rows BIGINT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    finally:
        conn.close()


def init_kline_table(
    db_path: str | Path,
    *,
    table_name: str = KLINE_TABLE_NAME,
) -> None:
    """Compatibility wrapper for callers that initialize only the K-line table."""
    init_duckdb(db_path, table_name=table_name)


def save_trade_calendar_to_duckdb(
    df: pd.DataFrame,
    db_path: str | Path,
    *,
    table_name: str = TRADE_CALENDAR_TABLE_NAME,
) -> int:
    """Save trading-calendar rows into DuckDB."""
    if df is None:
        raise ValueError("Input dataframe is None.")

    normalized = df.copy()
    for column in ["date", "is_trading_day", "exchange", "source"]:
        if column not in normalized.columns:
            normalized[column] = None

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
    normalized["is_trading_day"] = normalized["is_trading_day"].map(_truthy_calendar_value)
    normalized["exchange"] = normalized["exchange"].fillna("CN").astype(str)
    normalized["source"] = normalized["source"].fillna("unknown").astype(str)
    normalized = normalized.dropna(subset=["date", "exchange"])
    normalized = normalized[["date", "is_trading_day", "exchange", "source"]].drop_duplicates(
        subset=["exchange", "date"],
        keep="last",
    )
    if normalized.empty:
        return 0

    init_duckdb(db_path)
    with connect_duckdb(db_path) as conn:
        conn.register("_trade_calendar_insert_df", normalized)
        try:
            conn.execute(
                f"""
                DELETE FROM {table_name}
                USING _trade_calendar_insert_df AS incoming
                WHERE {table_name}.exchange = incoming.exchange
                  AND {table_name}.date = incoming.date
                """
            )
            conn.execute(
                f"""
                INSERT INTO {table_name} (
                    date,
                    is_trading_day,
                    exchange,
                    source,
                    created_at,
                    updated_at
                )
                SELECT
                    date,
                    is_trading_day,
                    exchange,
                    source,
                    CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP
                FROM _trade_calendar_insert_df
                """
            )
        finally:
            try:
                conn.unregister("_trade_calendar_insert_df")
            except Exception:
                pass

    return int(len(normalized))


def _truthy_calendar_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "open"}


def load_trade_calendar_from_duckdb(
    db_path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    exchange: str | None = None,
    table_name: str = TRADE_CALENDAR_TABLE_NAME,
) -> pd.DataFrame:
    """Load trading-calendar rows from DuckDB."""
    columns = ["date", "is_trading_day", "exchange", "source"]
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import load_trade_calendar_from_market_shards

        return load_trade_calendar_from_market_shards(shard_root, start=start, end=end, exchange=exchange)

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return pd.DataFrame(columns=columns)

    with connect_duckdb_read_only(db_file) as conn:
        if not _table_exists(conn, table_name):
            return pd.DataFrame(columns=columns)

        query = f"SELECT {', '.join(columns)} FROM {table_name} WHERE 1 = 1"
        params: list[object] = []
        if start is not None:
            query += " AND date >= ?"
            params.append(start)
        if end is not None:
            query += " AND date <= ?"
            params.append(end)
        if exchange is not None:
            query += " AND exchange = ?"
            params.append(exchange)
        query += " ORDER BY exchange, date"
        return conn.execute(query, params).fetchdf()


def save_stock_liquidity_daily_to_duckdb(
    df: pd.DataFrame,
    db_path: str | Path,
    *,
    table_name: str = STOCK_LIQUIDITY_DAILY_TABLE_NAME,
) -> int:
    """Save daily stock liquidity rows such as turnover."""
    if df is None:
        raise ValueError("Input dataframe is None.")

    normalized = df.copy()
    for column in ["symbol", "date", "turn", "source"]:
        if column not in normalized.columns:
            normalized[column] = None

    normalized["symbol"] = normalized["symbol"].astype(str).str.replace("\ufeff", "", regex=False).str.strip().str.upper()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
    normalized["turn"] = pd.to_numeric(normalized["turn"], errors="coerce")
    normalized["source"] = normalized["source"].fillna("unknown").astype(str)
    normalized = normalized.dropna(subset=["symbol", "date"])
    normalized = normalized[["symbol", "date", "turn", "source"]].drop_duplicates(
        subset=["symbol", "date"],
        keep="last",
    )
    if normalized.empty:
        return 0

    init_duckdb(db_path)
    with connect_duckdb(db_path) as conn:
        conn.register("_stock_liquidity_insert_df", normalized)
        try:
            conn.execute(
                f"""
                DELETE FROM {table_name}
                USING _stock_liquidity_insert_df AS incoming
                WHERE {table_name}.symbol = incoming.symbol
                  AND {table_name}.date = incoming.date
                """
            )
            conn.execute(
                f"""
                INSERT INTO {table_name} (
                    symbol,
                    date,
                    turn,
                    source,
                    created_at,
                    updated_at
                )
                SELECT
                    symbol,
                    date,
                    turn,
                    source,
                    CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP
                FROM _stock_liquidity_insert_df
                """
            )
        finally:
            try:
                conn.unregister("_stock_liquidity_insert_df")
            except Exception:
                pass

    return int(len(normalized))


def load_stock_liquidity_daily_from_duckdb(
    db_path: str | Path,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    table_name: str = STOCK_LIQUIDITY_DAILY_TABLE_NAME,
) -> pd.DataFrame:
    """Load daily stock liquidity rows from DuckDB."""
    columns = ["symbol", "date", "turn", "source"]
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import load_daily_liquidity_from_market_shards

        return load_daily_liquidity_from_market_shards(shard_root, symbol=symbol, symbols=symbols, start=start, end=end)

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return pd.DataFrame(columns=columns)

    with connect_duckdb_read_only(db_file) as conn:
        if not _table_exists(conn, table_name):
            return pd.DataFrame(columns=columns)

        query = f"SELECT {', '.join(columns)} FROM {table_name} WHERE 1 = 1"
        params: list[object] = []
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(str(symbol).upper())
        if symbols is not None:
            symbol_list = [str(item).upper() for item in symbols]
            if symbol_list:
                placeholders = ", ".join(["?"] * len(symbol_list))
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbol_list)
        if start is not None:
            query += " AND date >= ?"
            params.append(start)
        if end is not None:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY symbol, date"
        return conn.execute(query, params).fetchdf()


def save_stock_status_daily_to_duckdb(
    df: pd.DataFrame,
    db_path: str | Path,
    *,
    table_name: str = STOCK_STATUS_DAILY_TABLE_NAME,
) -> int:
    """Save dated ST facts used by market-rule constraints."""
    if df is None:
        raise ValueError("Input dataframe is None.")

    normalized = df.copy()
    for column in ["symbol", "date", "is_st", "source"]:
        if column not in normalized.columns:
            normalized[column] = None
    normalized["symbol"] = normalized["symbol"].astype(str).str.replace("\ufeff", "", regex=False).str.strip().str.upper()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
    normalized["is_st"] = normalized["is_st"].astype("boolean")
    normalized["source"] = normalized["source"].fillna("unknown").astype(str)
    normalized = normalized.dropna(subset=["symbol", "date", "is_st"])
    normalized = normalized[["symbol", "date", "is_st", "source"]].drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    if normalized.empty:
        return 0

    init_duckdb(db_path)
    with connect_duckdb(db_path) as conn:
        conn.register("_stock_status_insert_df", normalized)
        try:
            conn.execute(
                f"""
                DELETE FROM {table_name}
                USING _stock_status_insert_df AS incoming
                WHERE {table_name}.symbol = incoming.symbol
                  AND {table_name}.date = incoming.date
                """
            )
            conn.execute(
                f"""
                INSERT INTO {table_name} (
                    symbol, date, is_st, source, created_at, updated_at
                )
                SELECT symbol, date, is_st, source, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                FROM _stock_status_insert_df
                """
            )
        finally:
            try:
                conn.unregister("_stock_status_insert_df")
            except Exception:
                pass
    return int(len(normalized))


def load_stock_status_daily_from_duckdb(
    db_path: str | Path,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    table_name: str = STOCK_STATUS_DAILY_TABLE_NAME,
) -> pd.DataFrame:
    """Load historical ST flags without exposing them as market features."""
    columns = ["symbol", "date", "is_st", "source"]
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import load_daily_status_from_market_shards

        return load_daily_status_from_market_shards(shard_root, symbol=symbol, symbols=symbols, start=start, end=end)

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return pd.DataFrame(columns=columns)
    with connect_duckdb_read_only(db_file) as conn:
        if not _table_exists(conn, table_name):
            return pd.DataFrame(columns=columns)
        query = f"SELECT {', '.join(columns)} FROM {table_name} WHERE 1 = 1"
        params: list[object] = []
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(str(symbol).upper())
        if symbols is not None:
            symbol_list = [str(item).upper() for item in symbols]
            if symbol_list:
                placeholders = ", ".join(["?"] * len(symbol_list))
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbol_list)
        if start is not None:
            query += " AND date >= ?"
            params.append(start)
        if end is not None:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY symbol, date"
        return conn.execute(query, params).fetchdf()


def _normalize_coverage_date(value: str | object | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).normalize()


def _date_series_bounds(values: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    dates = pd.to_datetime(values, errors="coerce").dt.normalize().dropna().drop_duplicates()
    if dates.empty:
        return None, None, 0
    dates = dates.sort_values()
    return pd.Timestamp(dates.iloc[0]), pd.Timestamp(dates.iloc[-1]), int(len(dates))


def _calendar_day_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end < start:
        return 0
    return int((end - start).days) + 1


def _weekday_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end < start:
        return 0
    days = pd.date_range(start=start, end=end, freq="D")
    return int(sum(day.weekday() < 5 for day in days))


def is_trade_calendar_range_covered(
    db_path: str | Path,
    *,
    start: str,
    end: str,
    exchange: str = "CN",
    min_coverage_ratio: float = 0.75,
) -> bool:
    """Return True when local trade-calendar rows plausibly cover the range."""
    start_date = _normalize_coverage_date(start)
    end_date = _normalize_coverage_date(end)
    if start_date is None or end_date is None or end_date < start_date:
        return False

    df = load_trade_calendar_from_duckdb(db_path, start=str(start), end=str(end), exchange=exchange)
    if df.empty and exchange:
        df = load_trade_calendar_from_duckdb(db_path, start=str(start), end=str(end), exchange=None)
    if df.empty or "date" not in df.columns:
        return False

    first_date, last_date, row_count = _date_series_bounds(df["date"])
    if first_date is None or last_date is None:
        return False
    if first_date > start_date or last_date < end_date:
        return False

    expected = max(1, int(_calendar_day_count(start_date, end_date) * min_coverage_ratio))
    return row_count >= expected


def is_daily_liquidity_range_covered(
    db_path: str | Path,
    *,
    symbol: str,
    start: str,
    end: str,
    min_coverage_ratio: float = 0.6,
    boundary_tolerance_days: int = 4,
) -> bool:
    """Return True when turnover and ST-status extension rows cover the range."""
    start_date = _normalize_coverage_date(start)
    end_date = _normalize_coverage_date(end)
    if start_date is None or end_date is None or end_date < start_date:
        return False

    df = load_stock_liquidity_daily_from_duckdb(db_path, symbol=symbol, start=str(start), end=str(end))
    if df.empty or "date" not in df.columns:
        return False

    if "turn" in df.columns:
        df = df[pd.to_numeric(df["turn"], errors="coerce").notna()]
        if df.empty:
            return False

    first_date, last_date, row_count = _date_series_bounds(df["date"])
    if first_date is None or last_date is None:
        return False

    latest_allowed_start = start_date + pd.Timedelta(days=boundary_tolerance_days)
    earliest_allowed_end = end_date - pd.Timedelta(days=boundary_tolerance_days)
    if first_date > latest_allowed_start:
        return False
    if last_date < earliest_allowed_end:
        return False

    expected = max(1, int(_weekday_count(start_date, end_date) * min_coverage_ratio))
    if row_count < expected:
        return False

    status = load_stock_status_daily_from_duckdb(
        db_path, symbol=symbol, start=str(start), end=str(end)
    )
    if status.empty or "date" not in status.columns:
        return False
    status_first, status_last, status_rows = _date_series_bounds(status["date"])
    if status_first is None or status_last is None:
        return False
    if status_first > latest_allowed_start or status_last < earliest_allowed_end:
        return False
    return status_rows >= expected


def _normalize_kline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a K-line dataframe into the standard storage format."""
    assert_valid_kline_dataframe(df)

    normalized = df.copy()
    for column in KLINE_SELECT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    normalized["symbol"] = normalized["symbol"].astype(str).str.replace("\ufeff", "", regex=False).str.strip().str.upper()
    normalized["datetime"] = pd.to_datetime(normalized["datetime"], format="mixed", errors="coerce")
    normalized = normalized.dropna(subset=["symbol", "datetime", "close"])

    required_numeric_columns = ["open", "high", "low", "close"]
    optional_numeric_columns = ["amount", "pctChg"]

    for column in required_numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce").fillna(0.0)

    for column in optional_numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column in ["source", "freq", "adjust"]:
        normalized[column] = normalized[column].astype("string")

    normalized["source"] = normalized["source"].fillna("unknown")
    normalized["freq"] = normalized["freq"].fillna("daily")
    normalized["adjust"] = normalized["adjust"].fillna("none")

    return normalized[KLINE_SELECT_COLUMNS].reset_index(drop=True)


def _refresh_pct_chg_for_affected_slices(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    """
    Recalculate pctChg for slices touched by the current insert.

    The first row in a slice uses close / open - 1. If older data is inserted
    later, the whole symbol/freq/adjust slice is recalculated so that former
    first rows use the newly available previous close.
    """
    conn.execute(
        f"""
        WITH affected AS (
            SELECT DISTINCT symbol, COALESCE(freq, '') AS freq, COALESCE(adjust, '') AS adjust
            FROM _kline_insert_df
        ),
        ordered AS (
            SELECT
                target.symbol,
                target.datetime,
                COALESCE(target.freq, '') AS freq,
                COALESCE(target.adjust, '') AS adjust,
                target.open,
                target.close,
                LAG(target.close) OVER (
                    PARTITION BY target.symbol, COALESCE(target.freq, ''), COALESCE(target.adjust, '')
                    ORDER BY target.datetime
                ) AS previous_close
            FROM {table_name} AS target
            JOIN affected
              ON target.symbol = affected.symbol
             AND COALESCE(target.freq, '') = affected.freq
             AND COALESCE(target.adjust, '') = affected.adjust
        ),
        calculated AS (
            SELECT
                symbol,
                datetime,
                freq,
                adjust,
                CASE
                    WHEN previous_close IS NOT NULL AND previous_close != 0
                        THEN close / previous_close - 1
                    WHEN open IS NOT NULL AND open != 0
                        THEN close / open - 1
                    ELSE NULL
                END AS pct_chg
            FROM ordered
        )
        UPDATE {table_name} AS target
        SET
            pctChg = calculated.pct_chg,
            updated_at = CURRENT_TIMESTAMP
        FROM calculated
        WHERE target.symbol = calculated.symbol
          AND target.datetime = calculated.datetime
          AND COALESCE(target.freq, '') = calculated.freq
          AND COALESCE(target.adjust, '') = calculated.adjust
        """
    )


def save_kline_to_duckdb(
    df: pd.DataFrame,
    db_path: str | Path,
    *,
    table_name: str = KLINE_TABLE_NAME,
    replace_symbol: bool = False,
) -> int:
    """
    Save K-line rows into DuckDB.

    Stable write strategy:
        1. normalize rows
        2. de-duplicate incoming rows by symbol + datetime + freq + adjust
        3. delete old rows matching the same key
        4. insert new rows

    This avoids ambiguous DuckDB upserts and lets none/pre coexist.
    """
    normalized = _normalize_kline_dataframe(df)
    if normalized.empty:
        return 0

    key_columns = ["symbol", "datetime", "freq", "adjust"]
    normalized = normalized.drop_duplicates(subset=key_columns, keep="last").copy()

    init_duckdb(db_path, table_name=table_name)

    db_file = ensure_parent_dir(db_path)
    conn = duckdb.connect(str(db_file))
    try:
        _ensure_kline_columns(conn, table_name)
        _drop_known_kline_indexes(conn, table_name)
        conn.register("_kline_insert_df", normalized)

        if replace_symbol:
            conn.execute(
                f"""
                DELETE FROM {table_name}
                WHERE symbol IN (SELECT DISTINCT symbol FROM _kline_insert_df)
                """
            )
        else:
            conn.execute(
                f"""
                DELETE FROM {table_name}
                USING _kline_insert_df AS incoming
                WHERE {table_name}.symbol = incoming.symbol
                  AND {table_name}.datetime = incoming.datetime
                  AND COALESCE({table_name}.freq, '') = COALESCE(incoming.freq, '')
                  AND COALESCE({table_name}.adjust, '') = COALESCE(incoming.adjust, '')
                """
            )

        insert_columns = ", ".join(KLINE_SELECT_COLUMNS)
        select_columns = ", ".join(KLINE_SELECT_COLUMNS)
        conn.execute(
            f"""
            INSERT INTO {table_name} (
                {insert_columns},
                created_at,
                updated_at,
                row_hash
            )
            SELECT
                {select_columns},
                CURRENT_TIMESTAMP AS created_at,
                CURRENT_TIMESTAMP AS updated_at,
                NULL AS row_hash
            FROM _kline_insert_df
            """
        )
        _refresh_pct_chg_for_affected_slices(conn, table_name)
        return int(len(normalized))
    finally:
        try:
            conn.unregister("_kline_insert_df")
        except Exception:
            pass
        conn.close()


def load_kline_from_duckdb(
    db_path: str | Path,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    freq: str | None = None,
    adjust: str | None = None,
    table_name: str = KLINE_TABLE_NAME,
) -> pd.DataFrame:
    """Load K-line rows from DuckDB."""
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import load_kline_from_market_shards

        return load_kline_from_market_shards(
            shard_root,
            symbol=symbol,
            symbols=symbols,
            start=start,
            end=end,
            freq=freq,
            adjust=adjust,
        )

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS)

    with connect_duckdb_read_only(db_file) as conn:
        if not _table_exists(conn, table_name):
            return pd.DataFrame(columns=KLINE_SELECT_COLUMNS)

        select_exprs = _kline_select_exprs(conn, table_name)
        query = f"""
            SELECT
                {", ".join(select_exprs)}
            FROM {table_name}
            WHERE 1 = 1
        """
        params: list[object] = []

        if symbol is not None:
            query += " AND symbol = ?"
            params.append(str(symbol).upper())

        if symbols is not None:
            symbol_list = [str(item).upper() for item in symbols]
            if symbol_list:
                placeholders = ", ".join(["?"] * len(symbol_list))
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbol_list)

        if start is not None:
            query += " AND datetime >= ?"
            params.append(_normalize_datetime_bound(start))

        if end is not None:
            query += " AND datetime <= ?"
            params.append(_normalize_datetime_bound(end, is_end=True))

        if freq is not None:
            query += " AND COALESCE(freq, '') = COALESCE(?, '')"
            params.append(freq)

        if adjust is not None:
            query += " AND COALESCE(adjust, '') = COALESCE(?, '')"
            params.append(adjust)

        query += " ORDER BY symbol, datetime, freq, adjust"
        return conn.execute(query, params).fetchdf()


def get_kline_inventory(
    db_path: str | Path,
    *,
    table_name: str = KLINE_TABLE_NAME,
) -> pd.DataFrame:
    """Return data inventory by symbol/frequency/adjustment slice."""
    empty = pd.DataFrame(columns=["symbol", "freq", "adjust", "source", "rows", "start_datetime", "end_datetime"])
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import get_market_shard_inventory

        return get_market_shard_inventory(shard_root)

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return empty

    with connect_duckdb_read_only(db_file) as conn:
        if not _table_exists(conn, table_name):
            return empty
        existing = _get_existing_columns(conn, table_name)
        freq_expr = "COALESCE(freq, '')" if "freq" in existing else "''"
        adjust_expr = "COALESCE(adjust, '')" if "adjust" in existing else "''"
        source_expr = "COALESCE(source, '')" if "source" in existing else "''"
        query = f"""
            SELECT
                symbol,
                {freq_expr} AS freq,
                {adjust_expr} AS adjust,
                {source_expr} AS source,
                COUNT(*) AS rows,
                MIN(datetime) AS start_datetime,
                MAX(datetime) AS end_datetime
            FROM {table_name}
            GROUP BY symbol, freq, adjust, source
            ORDER BY symbol, freq, adjust
        """
        return conn.execute(query).fetchdf()


def get_kline_coverage(
    db_path: str | Path,
    *,
    symbol: str,
    freq: str | None = None,
    adjust: str | None = None,
    table_name: str = KLINE_TABLE_NAME,
) -> dict[str, object] | None:
    """Return row count and date coverage for one symbol/frequency/adjust slice."""
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import load_kline_from_market_shards

        df = load_kline_from_market_shards(shard_root, symbol=symbol, freq=freq, adjust=adjust)
        if df.empty:
            return None
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        return {
            "symbol": str(symbol).upper(),
            "freq": freq,
            "adjust": adjust,
            "rows": int(len(df)),
            "start_datetime": dt.min(),
            "end_datetime": dt.max(),
        }

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return None

    with connect_duckdb_read_only(db_file) as conn:
        if not _table_exists(conn, table_name):
            return None
        query = f"""
            SELECT
                COUNT(*) AS rows,
                MIN(datetime) AS start_datetime,
                MAX(datetime) AS end_datetime
            FROM {table_name}
            WHERE symbol = ?
        """
        params: list[object] = [str(symbol).upper()]

        if freq is not None:
            query += " AND COALESCE(freq, '') = COALESCE(?, '')"
            params.append(freq)

        if adjust is not None:
            query += " AND COALESCE(adjust, '') = COALESCE(?, '')"
            params.append(adjust)

        row = conn.execute(query, params).fetchone()

    if row is None or int(row[0] or 0) == 0:
        return None

    return {
        "symbol": str(symbol).upper(),
        "freq": freq,
        "adjust": adjust,
        "rows": int(row[0] or 0),
        "start_datetime": row[1],
        "end_datetime": row[2],
    }


def expected_min_rows_for_range(
    start: str,
    end: str,
    *,
    freq: str | None = None,
    min_coverage_ratio: float = 0.75,
) -> int:
    """Return a conservative minimum expected row count for one requested date range."""
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts < start_ts:
        return 1

    business_days = len(pd.bdate_range(start_ts.normalize(), end_ts.normalize()))
    if business_days <= 0:
        business_days = 1

    freq_value = (freq or "daily").lower()
    bars_per_day = {
        "daily": 1,
        "d": 1,
        "weekly": 1 / 5,
        "w": 1 / 5,
        "monthly": 1 / 21,
        "m": 1 / 21,
        "60min": 4,
        "60": 4,
        "1h": 4,
        "30min": 8,
        "30": 8,
        "15min": 16,
        "15": 16,
        "5min": 48,
        "5": 48,
    }.get(freq_value, 1)

    expected = business_days * bars_per_day
    return max(1, int(expected * min_coverage_ratio))


def _kline_boundary_tolerance_days(freq: str | None) -> int:
    freq_value = (freq or "daily").lower()
    return {
        "daily": 4,
        "d": 4,
        "weekly": 10,
        "w": 10,
        "monthly": 31,
        "m": 31,
        "60min": 4,
        "60": 4,
        "1h": 4,
        "30min": 4,
        "30": 4,
    }.get(freq_value, 4)


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _canonicalize_kline_for_hash(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic dataframe representation for hashing."""
    if df.empty:
        return pd.DataFrame(columns=KLINE_HASH_COLUMNS)

    canonical = df.copy()
    for column in KLINE_HASH_COLUMNS:
        if column not in canonical.columns:
            canonical[column] = ""

    canonical = canonical[KLINE_HASH_COLUMNS]
    canonical["symbol"] = canonical["symbol"].astype(str).str.upper()
    canonical["datetime"] = pd.to_datetime(canonical["datetime"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    for column in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce").fillna(0.0)

    for column in ["source", "freq", "adjust"]:
        canonical[column] = canonical[column].fillna("").astype(str)

    canonical = canonical.sort_values(["symbol", "freq", "adjust", "datetime"]).reset_index(drop=True)
    return canonical


def calculate_kline_data_hash(df: pd.DataFrame) -> str:
    """Calculate a stable sha256 hash for normalized K-line rows."""
    canonical = _canonicalize_kline_for_hash(df)
    payload = canonical.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calculate_kline_data_hash_from_duckdb(
    db_path: str | Path,
    *,
    symbol: str,
    freq: str | None = None,
    adjust: str | None = None,
    start: str | None = None,
    end: str | None = None,
    table_name: str = KLINE_TABLE_NAME,
) -> tuple[str, dict[str, object]]:
    """Calculate hash and summary for a local K-line slice."""
    df = load_kline_from_duckdb(db_path, symbol=symbol, start=start, end=end, freq=freq, adjust=adjust, table_name=table_name)
    if df.empty:
        return calculate_kline_data_hash(df), {"rows": 0, "actual_start": None, "actual_end": None}

    return calculate_kline_data_hash(df), {
        "rows": int(len(df)),
        "actual_start": str(pd.to_datetime(df["datetime"], format="mixed", errors="coerce").min())[:19],
        "actual_end": str(pd.to_datetime(df["datetime"], format="mixed", errors="coerce").max())[:19],
    }


def record_download_manifest(
    db_path: str | Path,
    *,
    symbol: str,
    freq: str | None,
    adjust: str | None,
    requested_start: str,
    requested_end: str,
    status: str,
    source: str | None = None,
    job_id: str | None = None,
    report_path: str | None = None,
    error: str | None = None,
    table_name: str = KLINE_TABLE_NAME,
) -> dict[str, object]:
    """Record the completion status and local data hash for one download request."""
    init_duckdb(db_path, table_name=table_name)

    data_hash = None
    summary: dict[str, object] = {"rows": 0, "actual_start": None, "actual_end": None}
    if status == "completed":
        data_hash, summary = calculate_kline_data_hash_from_duckdb(
            db_path,
            symbol=symbol,
            freq=freq,
            adjust=adjust,
            start=requested_start,
            end=requested_end,
            table_name=table_name,
        )
        if int(summary.get("rows") or 0) <= 0:
            status = "failed"
            error = error or "No local rows found after successful download."
            data_hash = None

    with connect_duckdb(db_path) as conn:
        next_id = int(conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {DOWNLOAD_MANIFEST_TABLE_NAME}").fetchone()[0] or 1)
        conn.execute(
            f"""
            INSERT INTO {DOWNLOAD_MANIFEST_TABLE_NAME} (
                id, job_id, symbol, freq, adjust, source,
                requested_start, requested_end, actual_start, actual_end,
                rows, data_hash, hash_algorithm, status, downloaded_at,
                report_path, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                next_id,
                job_id,
                str(symbol).upper(),
                freq,
                adjust,
                source,
                requested_start,
                requested_end,
                summary.get("actual_start"),
                summary.get("actual_end"),
                int(summary.get("rows") or 0),
                data_hash,
                "sha256:kline_v2" if data_hash else None,
                status,
                _utc_now_naive(),
                report_path,
                error,
            ],
        )

    return {
        "symbol": str(symbol).upper(),
        "freq": freq,
        "adjust": adjust,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "actual_start": summary.get("actual_start"),
        "actual_end": summary.get("actual_end"),
        "rows": int(summary.get("rows") or 0),
        "data_hash": data_hash,
        "status": status,
        "error": error,
    }


def find_completed_download_manifest(
    db_path: str | Path,
    *,
    symbol: str,
    freq: str | None,
    adjust: str | None,
    requested_start: str,
    requested_end: str,
) -> dict[str, object] | None:
    """Find the latest completed manifest whose requested range covers this request."""
    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return None

    init_duckdb(db_file)
    with connect_duckdb_read_only(db_file) as conn:
        try:
            row = conn.execute(
                f"""
                SELECT
                    id, job_id, symbol, freq, adjust, source,
                    requested_start, requested_end, actual_start, actual_end,
                    rows, data_hash, hash_algorithm, status, downloaded_at,
                    report_path, error
                FROM {DOWNLOAD_MANIFEST_TABLE_NAME}
                WHERE symbol = ?
                  AND COALESCE(freq, '') = COALESCE(?, '')
                  AND COALESCE(adjust, '') = COALESCE(?, '')
                  AND status = 'completed'
                  AND requested_start <= ?
                  AND requested_end >= ?
                ORDER BY downloaded_at DESC, id DESC
                LIMIT 1
                """,
                [str(symbol).upper(), freq, adjust, requested_start, requested_end],
            ).fetchone()
        except duckdb.CatalogException:
            return None

    if row is None:
        return None

    keys = [
        "id", "job_id", "symbol", "freq", "adjust", "source",
        "requested_start", "requested_end", "actual_start", "actual_end",
        "rows", "data_hash", "hash_algorithm", "status", "downloaded_at",
        "report_path", "error",
    ]
    return dict(zip(keys, row))


def verify_completed_download_manifest(
    db_path: str | Path,
    *,
    symbol: str,
    freq: str | None,
    adjust: str | None,
    requested_start: str,
    requested_end: str,
) -> tuple[bool, str, dict[str, object] | None]:
    """Return whether a completed manifest plus current local hash permits skipping."""
    manifest = find_completed_download_manifest(
        db_path,
        symbol=symbol,
        freq=freq,
        adjust=adjust,
        requested_start=requested_start,
        requested_end=requested_end,
    )
    if manifest is None:
        return False, "no completed manifest found", None

    expected_hash = str(manifest.get("data_hash") or "")
    if not expected_hash:
        return False, "completed manifest has no data hash", manifest

    actual_start = manifest.get("actual_start") or requested_start
    actual_end = manifest.get("actual_end") or requested_end
    current_hash, summary = calculate_kline_data_hash_from_duckdb(
        db_path,
        symbol=symbol,
        freq=freq,
        adjust=adjust,
        start=str(actual_start)[:19],
        end=str(actual_end)[:19],
    )

    manifest_rows = int(manifest.get("rows") or 0)
    current_rows = int(summary.get("rows") or 0)
    if current_rows != manifest_rows:
        return False, f"manifest rows={manifest_rows}, local rows={current_rows}", manifest

    if str(summary.get("actual_start") or "")[:19] != str(manifest.get("actual_start") or "")[:19]:
        return False, f"manifest start={manifest.get('actual_start')}, local start={summary.get('actual_start')}", manifest

    if str(summary.get("actual_end") or "")[:19] != str(manifest.get("actual_end") or "")[:19]:
        return False, f"manifest end={manifest.get('actual_end')}, local end={summary.get('actual_end')}", manifest

    if current_hash != expected_hash:
        return False, "manifest hash mismatch", manifest

    return True, f"manifest hash verified; rows={current_rows}, actual={summary.get('actual_start')} -> {summary.get('actual_end')}", manifest


def delete_download_manifest(
    db_path: str | Path,
    *,
    symbol: str | None = None,
    freq: str | None = None,
    adjust: str | None = None,
) -> int:
    """Delete manifest rows matching a local data deletion."""
    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return 0

    init_duckdb(db_file)
    where = "WHERE 1 = 1"
    params: list[object] = []
    if symbol is not None:
        where += " AND symbol = ?"
        params.append(str(symbol).upper())
    if freq is not None:
        where += " AND COALESCE(freq, '') = COALESCE(?, '')"
        params.append(freq)
    if adjust is not None:
        where += " AND COALESCE(adjust, '') = COALESCE(?, '')"
        params.append(adjust)

    with connect_duckdb(db_file) as conn:
        before = int(conn.execute(f"SELECT COUNT(*) FROM {DOWNLOAD_MANIFEST_TABLE_NAME} {where}", params).fetchone()[0] or 0)
        conn.execute(f"DELETE FROM {DOWNLOAD_MANIFEST_TABLE_NAME} {where}", params)
        return before


def is_kline_range_covered(
    db_path: str | Path,
    *,
    symbol: str,
    start: str,
    end: str,
    freq: str | None = None,
    adjust: str | None = None,
    table_name: str = KLINE_TABLE_NAME,
    min_coverage_ratio: float = 0.75,
) -> bool:
    """Return True only if the stored slice appears complete enough to skip."""
    coverage = get_kline_coverage(db_path, symbol=symbol, freq=freq, adjust=adjust, table_name=table_name)
    if not coverage:
        return False

    stored_start = pd.to_datetime(coverage["start_datetime"])
    stored_end = pd.to_datetime(coverage["end_datetime"])
    required_start = pd.to_datetime(start)
    required_end = pd.to_datetime(end)

    boundary_tolerance_days = _kline_boundary_tolerance_days(freq)

    if not bool(stored_start <= required_start + pd.Timedelta(days=boundary_tolerance_days)):
        return False
    if not bool(stored_end >= required_end - pd.Timedelta(days=boundary_tolerance_days)):
        return False

    expected_min_rows = expected_min_rows_for_range(start, end, freq=freq, min_coverage_ratio=min_coverage_ratio)
    return int(coverage.get("rows") or 0) >= expected_min_rows


def plan_kline_download_ranges(
    db_path: str | Path,
    *,
    symbol: str,
    requested_start: str,
    requested_end: str,
    freq: str | None = None,
    adjust: str | None = None,
) -> list[tuple[str, str]]:
    """Return the date ranges that still need downloading for one K-line slice."""
    if is_kline_range_covered(
        db_path,
        symbol=symbol,
        start=requested_start,
        end=requested_end,
        freq=freq,
        adjust=adjust,
    ):
        return []

    coverage = get_kline_coverage(db_path, symbol=symbol, freq=freq, adjust=adjust)
    if not coverage:
        return [(requested_start, requested_end)]

    stored_start = pd.to_datetime(coverage.get("start_datetime"), errors="coerce")
    stored_end = pd.to_datetime(coverage.get("end_datetime"), errors="coerce")
    request_start = pd.to_datetime(requested_start, errors="coerce")
    request_end = pd.to_datetime(requested_end, errors="coerce")
    if any(pd.isna(value) for value in [stored_start, stored_end, request_start, request_end]):
        return [(requested_start, requested_end)]

    stored_start_date = pd.Timestamp(stored_start).normalize()
    stored_end_date = pd.Timestamp(stored_end).normalize()
    request_start_date = pd.Timestamp(request_start).normalize()
    request_end_date = pd.Timestamp(request_end).normalize()

    if not is_kline_range_covered(
        db_path,
        symbol=symbol,
        start=str(stored_start_date.date()),
        end=str(stored_end_date.date()),
        freq=freq,
        adjust=adjust,
    ):
        return [(requested_start, requested_end)]

    ranges: list[tuple[str, str]] = []
    tolerance = pd.Timedelta(days=_kline_boundary_tolerance_days(freq))
    if request_start_date + tolerance < stored_start_date:
        prefix_end = stored_start_date - pd.Timedelta(days=1)
        if prefix_end >= request_start_date:
            ranges.append((str(request_start_date.date()), str(prefix_end.date())))

    if request_end_date - tolerance > stored_end_date:
        suffix_start = stored_end_date + pd.Timedelta(days=1)
        if suffix_start <= request_end_date:
            ranges.append((str(suffix_start.date()), str(request_end_date.date())))

    return ranges or [(requested_start, requested_end)]


def delete_symbol_from_duckdb(
    db_path: str | Path,
    symbol: str,
    *,
    freq: str | None = None,
    adjust: str | None = None,
    table_name: str = KLINE_TABLE_NAME,
) -> int:
    """Delete rows for one symbol, optionally limited to freq/adjust."""
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import delete_symbol_market_slice

        return delete_symbol_market_slice(shard_root, symbol, freq=freq, adjust=adjust)

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return 0

    init_duckdb(db_file, table_name=table_name)

    query = f"DELETE FROM {table_name} WHERE symbol = ?"
    params: list[object] = [str(symbol).upper()]
    count_query = f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?"
    count_params: list[object] = [str(symbol).upper()]

    if freq is not None:
        query += " AND COALESCE(freq, '') = COALESCE(?, '')"
        params.append(freq)
        count_query += " AND COALESCE(freq, '') = COALESCE(?, '')"
        count_params.append(freq)
    if adjust is not None:
        query += " AND COALESCE(adjust, '') = COALESCE(?, '')"
        params.append(adjust)
        count_query += " AND COALESCE(adjust, '') = COALESCE(?, '')"
        count_params.append(adjust)

    with connect_duckdb(db_file) as conn:
        before = int(conn.execute(count_query, count_params).fetchone()[0] or 0)
        conn.execute(query, params)
        if _table_exists(conn, DERIVED_BAR_MANIFEST_TABLE_NAME):
            manifest_query = f"DELETE FROM {DERIVED_BAR_MANIFEST_TABLE_NAME} WHERE symbol = ?"
            manifest_params: list[object] = [str(symbol).upper()]
            if freq is not None:
                manifest_query += " AND (base_freq = ? OR target_freq = ?)"
                manifest_params.extend([freq, freq])
            if adjust is not None:
                manifest_query += " AND adjust = ?"
                manifest_params.append(adjust)
            conn.execute(manifest_query, manifest_params)

    if before > 0:
        delete_download_manifest(db_file, symbol=symbol, freq=freq, adjust=adjust)
    return before


def delete_symbol_extensions_from_duckdb(db_path: str | Path, symbol: str) -> int:
    """Delete daily extension facts when the whole symbol is removed."""
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import delete_symbol_market_shards

        return delete_symbol_market_shards(shard_root, symbol)

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return 0
    init_duckdb(db_file)
    clean_symbol = str(symbol).upper()
    deleted = 0
    with connect_duckdb(db_file) as conn:
        for extension_table in (STOCK_LIQUIDITY_DAILY_TABLE_NAME, STOCK_STATUS_DAILY_TABLE_NAME):
            if not _table_exists(conn, extension_table):
                continue
            count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {extension_table} WHERE symbol = ?", [clean_symbol]
                ).fetchone()[0]
                or 0
            )
            conn.execute(f"DELETE FROM {extension_table} WHERE symbol = ?", [clean_symbol])
            deleted += count
    return deleted


def delete_many_symbols_from_duckdb(
    db_path: str | Path,
    slices: Iterable[dict[str, object]],
    *,
    table_name: str = KLINE_TABLE_NAME,
) -> int:
    """Delete multiple symbol/frequency/adjust slices.

    For partitioned shard storage this uses a real batch delete path: one
    catalog scan, batch file unlink, and one catalog transaction.
    """
    slices = list(slices)
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import delete_many_symbol_market_slices

        result = delete_many_symbol_market_slices(
            shard_root,
            slices,
            include_full_symbol_extensions=False,
        )
        return result.deleted_rows

    total_deleted = 0
    for item in slices:
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        freq = item.get("freq")
        adjust = item.get("adjust")
        total_deleted += delete_symbol_from_duckdb(
            db_path,
            symbol,
            freq=str(freq) if freq not in (None, "", "-") else None,
            adjust=str(adjust) if adjust not in (None, "", "-") else None,
            table_name=table_name,
        )
    return total_deleted


def load_kline_page_from_duckdb(
    db_path: str | Path,
    *,
    symbol: str,
    freq: str | None = None,
    adjust: str | None = None,
    offset: int | None = 0,
    limit: int = 100,
    max_limit: int = 1200,
    table_name: str = KLINE_TABLE_NAME,
) -> tuple[pd.DataFrame, int]:
    """Load one paginated K-line slice and return (rows, total_rows)."""
    shard_root = _partitioned_market_root_if_available(db_path)
    if shard_root is not None:
        from data_layer.storage.partitioned_storage import load_kline_page_from_market_shards

        return load_kline_page_from_market_shards(
            shard_root,
            symbol=symbol,
            freq=freq,
            adjust=adjust,
            offset=offset,
            limit=limit,
            max_limit=max_limit,
        )

    db_file = _legacy_duckdb_file_if_available(db_path)
    if db_file is None:
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS), 0

    limit = min(max(1, int(max_limit or 1200)), max(1, int(limit or 100)))

    with connect_duckdb_read_only(db_file) as conn:
        if not _table_exists(conn, table_name):
            return pd.DataFrame(columns=KLINE_SELECT_COLUMNS), 0
        select_exprs = _kline_select_exprs(conn, table_name)
        where = "WHERE symbol = ?"
        params: list[object] = [str(symbol).upper()]
        if freq is not None:
            where += " AND COALESCE(freq, '') = COALESCE(?, '')"
            params.append(freq)
        if adjust is not None:
            where += " AND COALESCE(adjust, '') = COALESCE(?, '')"
            params.append(adjust)

        count_query = f"SELECT COUNT(*) FROM {table_name} {where}"
        data_query = f"""
            SELECT {", ".join(select_exprs)}
            FROM {table_name}
            {where}
            ORDER BY datetime
            LIMIT ? OFFSET ?
        """
        total = int(conn.execute(count_query, params).fetchone()[0] or 0)
        resolved_offset = max(0, total - limit) if offset is None else max(0, int(offset or 0))
        df = conn.execute(data_query, [*params, limit, resolved_offset]).fetchdf()
        return df, total
