from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import shutil
from pathlib import Path
from threading import Lock
from typing import Iterable
from uuid import uuid4

import duckdb
import pandas as pd

from data_layer.storage.duckdb_storage import KLINE_SELECT_COLUMNS, _normalize_kline_dataframe
from data_layer.storage.data_loader import (
    normalize_stock_liquidity_dataframe,
    normalize_stock_status_dataframe,
    normalize_trade_calendar_dataframe,
)

DEFAULT_DATA_ROOT = Path("runtime_layer/data")
MARKET_PARTS_DIR = "market_parts"
CATALOG_DB_NAME = "market_catalog.duckdb"
KLINE_DATASET = "kline"
TRADE_CALENDAR_DATASET = "trade_calendar"
DAILY_LIQUIDITY_DATASET = "daily_liquidity"
DAILY_STATUS_DATASET = "daily_status"

CATALOG_TABLE = "market_shards"
_CATALOG_LOCK = Lock()


@dataclass(frozen=True)
class MarketShardRecord:
    dataset: str
    symbol: str
    freq: str | None
    adjust: str | None
    shard_path: str
    bucket: str | None
    rows: int
    start_datetime: str | None
    end_datetime: str | None
    data_hash: str | None
    storage_format: str
    status: str
    updated_at: str
    error: str | None = None


@dataclass(frozen=True)
class MarketShardWriteResult:
    symbol: str
    ok: bool
    rows: int = 0
    freq: str | None = None
    adjust: str | None = None
    shard_path: str | None = None
    bucket: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    data_hash: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MarketShardDeleteResult:
    deleted_rows: int = 0
    deleted_extension_rows: int = 0
    deleted_files: int = 0
    matched_catalog_rows: int = 0


def ensure_data_root(root: str | Path | None = None) -> Path:
    data_root = Path(root or DEFAULT_DATA_ROOT)
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def market_parts_root(root: str | Path | None = None) -> Path:
    return ensure_data_root(root) / MARKET_PARTS_DIR


def catalog_path(root: str | Path | None = None) -> Path:
    return ensure_data_root(root) / CATALOG_DB_NAME


def now_utc_text() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def safe_symbol(symbol: str | None) -> str:
    text = str(symbol or "GLOBAL").replace("\ufeff", "").strip().upper()
    return text.replace(".", "_").replace("/", "_").replace("\\", "_")


def stable_bucket(symbol: str | None, *, buckets: int = 1000) -> str:
    text = str(symbol or "GLOBAL").upper().encode("utf-8")
    value = int(hashlib.sha1(text).hexdigest()[:8], 16) % int(buckets)
    width = max(3, len(str(buckets - 1)))
    return f"{value:0{width}d}"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def kline_shard_path(
    root: str | Path | None,
    *,
    symbol: str,
    freq: str,
    adjust: str,
) -> Path:
    data_root = ensure_data_root(root)
    bucket = stable_bucket(symbol)
    filename = f"{str(freq).lower()}_{str(adjust).lower()}.parquet"
    return data_root / MARKET_PARTS_DIR / f"bucket={bucket}" / safe_symbol(symbol) / filename


def trade_calendar_shard_path(root: str | Path | None) -> Path:
    return ensure_data_root(root) / MARKET_PARTS_DIR / "_global" / "trade_calendar.parquet"


def daily_liquidity_shard_path(root: str | Path | None, *, symbol: str) -> Path:
    data_root = ensure_data_root(root)
    bucket = stable_bucket(symbol)
    return data_root / MARKET_PARTS_DIR / f"bucket={bucket}" / safe_symbol(symbol) / "daily_liquidity.parquet"


def daily_status_shard_path(root: str | Path | None, *, symbol: str) -> Path:
    data_root = ensure_data_root(root)
    bucket = stable_bucket(symbol)
    return data_root / MARKET_PARTS_DIR / f"bucket={bucket}" / safe_symbol(symbol) / "daily_status.parquet"


def init_market_catalog(root: str | Path | None = None) -> Path:
    db_file = catalog_path(root)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with _CATALOG_LOCK:
        with duckdb.connect(str(db_file)) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {CATALOG_TABLE} (
                    dataset VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    freq VARCHAR,
                    adjust VARCHAR,
                    shard_path VARCHAR NOT NULL,
                    bucket VARCHAR,
                    rows BIGINT,
                    start_datetime TIMESTAMP,
                    end_datetime TIMESTAMP,
                    data_hash VARCHAR,
                    storage_format VARCHAR,
                    status VARCHAR NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    error VARCHAR
                )
                """
            )
            try:
                conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{CATALOG_TABLE}_lookup
                    ON {CATALOG_TABLE}(dataset, symbol, freq, adjust, status)
                    """
                )
            except Exception:
                pass
    return db_file

def _data_hash(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    safe = df.copy()
    for column in columns:
        if column not in safe.columns:
            safe[column] = None
    safe = safe[columns]
    # Stable across platforms and independent of pandas' object repr details.
    csv_text = safe.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S")
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp.parquet")
    conn = duckdb.connect(database=":memory:")
    try:
        conn.register("_out_df", df)
        target = str(tmp_path).replace("'", "''")
        conn.execute(
            f"COPY (SELECT * FROM _out_df) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        try:
            conn.unregister("_out_df")
        except Exception:
            pass
        conn.close()
    tmp_path.replace(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with duckdb.connect(database=":memory:") as conn:
        return conn.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()


def _refresh_pct_chg_in_memory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.sort_values(["symbol", "freq", "adjust", "datetime"]).copy()
    grouped = result.groupby(["symbol", "freq", "adjust"], dropna=False)["close"]
    previous_close = grouped.shift(1)
    close = pd.to_numeric(result["close"], errors="coerce")
    open_ = pd.to_numeric(result["open"], errors="coerce")
    previous_close = pd.to_numeric(previous_close, errors="coerce")
    pct = close / previous_close - 1
    fallback = close / open_ - 1
    result["pctChg"] = pct.where(previous_close.notna() & (previous_close != 0), fallback.where(open_.notna() & (open_ != 0)))
    return result


def upsert_market_catalog(root: str | Path | None, record: MarketShardRecord) -> None:
    init_market_catalog(root)
    db_file = catalog_path(root)
    payload = asdict(record)
    with _CATALOG_LOCK:
        with duckdb.connect(str(db_file)) as conn:
            conn.execute(
                f"""
                DELETE FROM {CATALOG_TABLE}
                WHERE dataset = ?
                  AND symbol = ?
                  AND COALESCE(freq, '') = COALESCE(?, '')
                  AND COALESCE(adjust, '') = COALESCE(?, '')
                """,
                [record.dataset, record.symbol, record.freq, record.adjust],
            )
            conn.execute(
                f"""
                INSERT INTO {CATALOG_TABLE} (
                    dataset, symbol, freq, adjust, shard_path, bucket,
                    rows, start_datetime, end_datetime, data_hash,
                    storage_format, status, updated_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    payload["dataset"], payload["symbol"], payload["freq"], payload["adjust"],
                    payload["shard_path"], payload["bucket"], int(payload["rows"]),
                    payload["start_datetime"], payload["end_datetime"], payload["data_hash"],
                    payload["storage_format"], payload["status"], payload["updated_at"], payload["error"],
                ],
            )


def get_market_catalog_record(
    root: str | Path | None,
    *,
    dataset: str,
    symbol: str,
    freq: str | None = None,
    adjust: str | None = None,
) -> dict[str, object] | None:
    db_file = catalog_path(root)
    if not db_file.exists():
        return None
    with duckdb.connect(str(db_file), read_only=True) as conn:
        try:
            row = conn.execute(
                f"""
                SELECT dataset, symbol, freq, adjust, shard_path, bucket, rows,
                       start_datetime, end_datetime, data_hash, storage_format,
                       status, updated_at, error
                FROM {CATALOG_TABLE}
                WHERE dataset = ?
                  AND symbol = ?
                  AND COALESCE(freq, '') = COALESCE(?, '')
                  AND COALESCE(adjust, '') = COALESCE(?, '')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                [dataset, str(symbol).upper(), freq, adjust],
            ).fetchone()
        except duckdb.CatalogException:
            return None
    if row is None:
        return None
    keys = [
        "dataset", "symbol", "freq", "adjust", "shard_path", "bucket", "rows",
        "start_datetime", "end_datetime", "data_hash", "storage_format",
        "status", "updated_at", "error",
    ]
    return dict(zip(keys, row))


def _calendar_flag_is_trading(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class _CalendarCoverageWindow:
    start: str
    end: str
    has_trading_days: bool | None
    calendar_available: bool


@dataclass(frozen=True)
class MarketCoverageSnapshot:
    """In-memory snapshot used to plan a whole download job.

    Download planning can touch the same trade calendar and catalog thousands
    of times when most slices are already covered.  This snapshot lets the App
    Layer read both once, then perform all coverage checks in memory.
    """

    root: Path
    records: dict[tuple[str, str, str, str], dict[str, object]]
    trading_dates: tuple[pd.Timestamp, ...]
    calendar_available: bool


def _today_date() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def _trading_dates_from_calendar(calendar: pd.DataFrame) -> pd.Series:
    if calendar.empty or "date" not in calendar.columns or "is_trading_day" not in calendar.columns:
        return pd.Series(dtype="datetime64[ns]")
    working = calendar.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["date"])
    if working.empty:
        return pd.Series(dtype="datetime64[ns]")
    trading = working[working["is_trading_day"].map(_calendar_flag_is_trading)]
    if trading.empty:
        return pd.Series(dtype="datetime64[ns]")
    return trading["date"].sort_values().drop_duplicates().reset_index(drop=True)


def _catalog_key(
    *,
    dataset: str,
    symbol: str,
    freq: str | None,
    adjust: str | None,
) -> tuple[str, str, str, str]:
    return (
        str(dataset or ""),
        str(symbol or "").upper(),
        str(freq or ""),
        str(adjust or ""),
    )


def _calendar_window_from_trading_dates(
    trading_dates: tuple[pd.Timestamp, ...] | list[pd.Timestamp] | pd.Series,
    *,
    start: str,
    end: str,
    calendar_available: bool,
) -> _CalendarCoverageWindow:
    request_start = pd.to_datetime(start, errors="coerce")
    request_end = pd.to_datetime(end, errors="coerce")
    if pd.isna(request_start) or pd.isna(request_end):
        return _CalendarCoverageWindow(str(start), str(end), None, calendar_available)

    request_start = request_start.normalize()
    request_end = request_end.normalize()
    if request_start > request_end:
        return _CalendarCoverageWindow(str(start), str(end), None, calendar_available)

    actionable_end = min(request_end, _today_date())
    if request_start > actionable_end:
        return _CalendarCoverageWindow(str(start), str(actionable_end.date()), False, True)

    if not calendar_available:
        return _CalendarCoverageWindow(str(start), str(end), None, False)

    if isinstance(trading_dates, pd.Series):
        dates = trading_dates
    else:
        dates = pd.Series(list(trading_dates), dtype="datetime64[ns]")
    if dates.empty:
        return _CalendarCoverageWindow(str(start), str(actionable_end.date()), False, True)

    dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize()
    dates = dates[(dates >= request_start) & (dates <= actionable_end)]
    if dates.empty:
        return _CalendarCoverageWindow(str(start), str(actionable_end.date()), False, True)

    return _CalendarCoverageWindow(
        start=str(dates.min().date()),
        end=str(dates.max().date()),
        has_trading_days=True,
        calendar_available=True,
    )


def build_market_coverage_snapshot(
    root: str | Path | None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> MarketCoverageSnapshot:
    """Read catalog and trade calendar once for a download planning pass."""
    data_root = resolve_market_data_root(root)
    catalog = _completed_catalog_df(data_root)
    records: dict[tuple[str, str, str, str], dict[str, object]] = {}
    if not catalog.empty:
        for row in catalog.to_dict(orient="records"):
            key = _catalog_key(
                dataset=str(row.get("dataset") or ""),
                symbol=str(row.get("symbol") or ""),
                freq=row.get("freq"),
                adjust=row.get("adjust"),
            )
            if key in records:
                continue
            row = dict(row)
            shard_path = str(row.get("shard_path") or "")
            row["_shard_exists"] = bool(shard_path and (data_root / shard_path).exists())
            records[key] = row

    calendar_available = False
    trading_dates = pd.Series(dtype="datetime64[ns]")
    path = trade_calendar_shard_path(data_root)
    if path.exists():
        try:
            calendar = _read_parquet(path)
            if start is not None:
                start_ts = pd.to_datetime(start, errors="coerce")
                if not pd.isna(start_ts) and "date" in calendar.columns:
                    calendar = calendar[pd.to_datetime(calendar["date"], errors="coerce") >= start_ts.normalize()]
            if end is not None:
                end_ts = pd.to_datetime(end, errors="coerce")
                if not pd.isna(end_ts) and "date" in calendar.columns:
                    actionable_end = min(end_ts.normalize(), _today_date())
                    calendar = calendar[pd.to_datetime(calendar["date"], errors="coerce") <= actionable_end]
            trading_dates = _trading_dates_from_calendar(calendar)
            calendar_available = True
        except Exception:
            calendar_available = False
            trading_dates = pd.Series(dtype="datetime64[ns]")

    return MarketCoverageSnapshot(
        root=data_root,
        records=records,
        trading_dates=tuple(pd.to_datetime(trading_dates, errors="coerce").dropna().dt.normalize()),
        calendar_available=calendar_available,
    )


def _snapshot_record(
    snapshot: MarketCoverageSnapshot,
    *,
    dataset: str,
    symbol: str,
    freq: str | None = None,
    adjust: str | None = None,
) -> dict[str, object] | None:
    return snapshot.records.get(_catalog_key(dataset=dataset, symbol=symbol, freq=freq, adjust=adjust))


def _record_covers_window(record: dict[str, object] | None, *, start: str, end: str) -> bool:
    if not record or record.get("status") != "completed":
        return False
    if int(record.get("rows") or 0) <= 0:
        return False
    if record.get("_shard_exists") is False:
        return False
    stored_start = pd.to_datetime(record.get("start_datetime"), errors="coerce")
    stored_end = pd.to_datetime(record.get("end_datetime"), errors="coerce")
    request_start = pd.to_datetime(start, errors="coerce")
    request_end = pd.to_datetime(end, errors="coerce")
    if any(pd.isna(value) for value in [stored_start, stored_end, request_start, request_end]):
        return False
    return stored_start.normalize() <= request_start.normalize() and stored_end.normalize() >= request_end.normalize()


def _daily_extension_hash_columns(dataset: str) -> list[str] | None:
    dataset_value = str(dataset or "")
    if dataset_value == DAILY_LIQUIDITY_DATASET:
        return ["symbol", "date", "turn", "source"]
    if dataset_value == DAILY_STATUS_DATASET:
        return ["symbol", "date", "is_st", "source"]
    return None


def _normalize_daily_extension_for_hash(df: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    columns = _daily_extension_hash_columns(dataset)
    if columns is None:
        return pd.DataFrame()
    safe = df.copy()
    for column in columns:
        if column not in safe.columns:
            safe[column] = None
    safe = safe[columns].copy()
    safe["symbol"] = safe["symbol"].astype(str).str.upper()
    safe["date"] = pd.to_datetime(safe["date"], errors="coerce").dt.date
    safe = safe.dropna(subset=["symbol", "date"])
    safe = safe.drop_duplicates(subset=["symbol", "date"], keep="last")
    safe = safe.sort_values(["symbol", "date"]).reset_index(drop=True)
    return safe[columns]


def _daily_extension_record_hash_matches(
    root: str | Path | None,
    *,
    dataset: str,
    record: dict[str, object] | None,
) -> bool:
    """Verify a small daily extension shard against its catalog hash.

    This is intentionally limited to daily_liquidity/daily_status.  Rehashing
    large intraday K-line parquet files on every skip would make repeat-run
    planning slow again, while these daily extension shards are small enough to
    validate strictly.
    """
    columns = _daily_extension_hash_columns(dataset)
    if columns is None:
        return False
    if not record or not str(record.get("data_hash") or "").strip():
        return False
    shard_path = str(record.get("shard_path") or "")
    if not shard_path:
        return False
    data_root = ensure_data_root(root)
    path = data_root / shard_path
    if not path.exists():
        return False
    try:
        df = _read_parquet(path)
        safe = _normalize_daily_extension_for_hash(df, dataset=dataset)
        if safe.empty:
            return False
        expected_rows = int(record.get("rows") or 0)
        if expected_rows > 0 and int(len(safe)) != expected_rows:
            return False
        actual_hash = _data_hash(safe, columns)
    except Exception:
        return False
    return actual_hash == str(record.get("data_hash") or "")


def is_daily_extension_range_covered(
    root: str | Path | None,
    *,
    dataset: str,
    symbol: str,
    start: str,
    end: str,
    verify_hash: bool = True,
) -> bool:
    if dataset not in {DAILY_LIQUIDITY_DATASET, DAILY_STATUS_DATASET}:
        return False
    if not is_catalog_range_covered(
        root,
        dataset=dataset,
        symbol=symbol,
        freq=dataset,
        adjust="-",
        start=start,
        end=end,
    ):
        return False
    if not verify_hash:
        return True
    record = get_market_catalog_record(root, dataset=dataset, symbol=symbol, freq=dataset, adjust="-")
    return _daily_extension_record_hash_matches(root, dataset=dataset, record=record)


def is_daily_extension_range_covered_in_snapshot(
    snapshot: MarketCoverageSnapshot,
    *,
    dataset: str,
    symbol: str,
    start: str,
    end: str,
    verify_hash: bool = True,
) -> bool:
    if dataset not in {DAILY_LIQUIDITY_DATASET, DAILY_STATUS_DATASET}:
        return False
    if not is_catalog_range_covered_in_snapshot(
        snapshot,
        dataset=dataset,
        symbol=symbol,
        freq=dataset,
        adjust="-",
        start=start,
        end=end,
    ):
        return False
    if not verify_hash:
        return True
    record = _snapshot_record(snapshot, dataset=dataset, symbol=symbol, freq=dataset, adjust="-")
    return _daily_extension_record_hash_matches(snapshot.root, dataset=dataset, record=record)


def _normalize_range_to_trading_dates_from_snapshot(
    snapshot: MarketCoverageSnapshot,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if start > end:
        return None
    if not snapshot.calendar_available:
        return start.normalize(), end.normalize()
    dates = pd.Series(list(snapshot.trading_dates), dtype="datetime64[ns]")
    if dates.empty:
        return None
    dates = dates[(dates >= start.normalize()) & (dates <= end.normalize())]
    if dates.empty:
        return None
    return dates.min(), dates.max()


def is_catalog_range_covered_in_snapshot(
    snapshot: MarketCoverageSnapshot,
    *,
    dataset: str,
    symbol: str,
    start: str,
    end: str,
    freq: str | None = None,
    adjust: str | None = None,
) -> bool:
    effective_start = str(start)
    effective_end = str(end)
    if dataset != TRADE_CALENDAR_DATASET:
        window = _calendar_window_from_trading_dates(
            snapshot.trading_dates,
            start=str(start),
            end=str(end),
            calendar_available=snapshot.calendar_available,
        )
        if window.has_trading_days is False:
            return True
        if window.has_trading_days is True:
            effective_start = window.start
            effective_end = window.end

    record = _snapshot_record(snapshot, dataset=dataset, symbol=symbol, freq=freq, adjust=adjust)
    return _record_covers_window(record, start=effective_start, end=effective_end)


def plan_catalog_download_ranges_in_snapshot(
    snapshot: MarketCoverageSnapshot,
    *,
    dataset: str,
    symbol: str,
    requested_start: str,
    requested_end: str,
    freq: str | None = None,
    adjust: str | None = None,
) -> list[tuple[str, str]]:
    effective_start = str(requested_start)
    effective_end = str(requested_end)
    if dataset != TRADE_CALENDAR_DATASET:
        window = _calendar_window_from_trading_dates(
            snapshot.trading_dates,
            start=str(requested_start),
            end=str(requested_end),
            calendar_available=snapshot.calendar_available,
        )
        if window.has_trading_days is False:
            return []
        if window.has_trading_days is True:
            effective_start = window.start
            effective_end = window.end

    record = _snapshot_record(snapshot, dataset=dataset, symbol=symbol, freq=freq, adjust=adjust)
    if _record_covers_window(record, start=effective_start, end=effective_end):
        return []
    if not record or record.get("status") != "completed":
        return [(effective_start, effective_end)]

    stored_start = pd.to_datetime(record.get("start_datetime"), errors="coerce")
    stored_end = pd.to_datetime(record.get("end_datetime"), errors="coerce")
    request_start = pd.to_datetime(effective_start, errors="coerce")
    request_end = pd.to_datetime(effective_end, errors="coerce")
    if any(pd.isna(value) for value in [stored_start, stored_end, request_start, request_end]):
        return [(effective_start, effective_end)]

    ranges: list[tuple[str, str]] = []
    stored_start = stored_start.normalize()
    stored_end = stored_end.normalize()
    request_start = request_start.normalize()
    request_end = request_end.normalize()
    if request_start < stored_start:
        prefix_end = stored_start - pd.Timedelta(days=1)
        normalized = _normalize_range_to_trading_dates_from_snapshot(snapshot, start=request_start, end=prefix_end)
        if normalized is not None:
            prefix_start, prefix_end = normalized
            ranges.append((str(prefix_start.date()), str(prefix_end.date())))
    if request_end > stored_end:
        suffix_start = stored_end + pd.Timedelta(days=1)
        normalized = _normalize_range_to_trading_dates_from_snapshot(snapshot, start=suffix_start, end=request_end)
        if normalized is not None:
            suffix_start, suffix_end = normalized
            ranges.append((str(suffix_start.date()), str(suffix_end.date())))
    return ranges or [(effective_start, effective_end)]


def _calendar_coverage_window(
    root: str | Path | None,
    *,
    start: str,
    end: str,
) -> _CalendarCoverageWindow:
    """Return the actionable trading-day window for a requested date range.

    The requested range and the data-bearing trading range are not the same.
    If a user asks for 2020-01-01 -> 2020-01-31, the first actual A-share
    trading day is 2020-01-02.  Coverage checks should use 2020-01-02 as the
    left edge, otherwise every rerun asks BaoStock for a known holiday and gets
    0 rows.

    The right edge is safe to normalize only for historical dates.  For today
    or future dates, the planner only treats dates up to today as actionable,
    so it does not request future data and does not require a shard to cover a
    future right edge before it can skip the already-current local data.

    Returns has_trading_days=None when no usable local calendar exists, so the
    caller can keep legacy range behavior instead of adding extra remote calls.
    """
    request_start = pd.to_datetime(start, errors="coerce")
    request_end = pd.to_datetime(end, errors="coerce")
    if pd.isna(request_start) or pd.isna(request_end):
        return _CalendarCoverageWindow(str(start), str(end), None, False)

    request_start = request_start.normalize()
    request_end = request_end.normalize()
    if request_start > request_end:
        return _CalendarCoverageWindow(str(start), str(end), None, False)

    # Future dates are not actionable for historical downloads.  We cap the
    # planning window to today while leaving the persisted catalog range based
    # on actual rows.  On later runs, today's date moves forward and the same
    # planner will naturally request newly actionable trading days.
    actionable_end = min(request_end, _today_date())
    if request_start > actionable_end:
        return _CalendarCoverageWindow(str(start), str(actionable_end.date()), False, True)

    try:
        calendar = load_trade_calendar_from_market_shards(
            root,
            start=str(request_start.date()),
            end=str(actionable_end.date()),
        )
    except Exception:
        return _CalendarCoverageWindow(str(start), str(end), None, False)

    if calendar.empty:
        return _CalendarCoverageWindow(str(start), str(end), None, False)

    trading_dates = _trading_dates_from_calendar(calendar)
    if trading_dates.empty:
        return _CalendarCoverageWindow(str(start), str(actionable_end.date()), False, True)

    trading_dates = trading_dates[(trading_dates >= request_start) & (trading_dates <= actionable_end)]
    if trading_dates.empty:
        return _CalendarCoverageWindow(str(start), str(actionable_end.date()), False, True)

    return _CalendarCoverageWindow(
        start=str(trading_dates.min().date()),
        end=str(trading_dates.max().date()),
        has_trading_days=True,
        calendar_available=True,
    )


def _normalize_range_to_trading_days(
    root: str | Path | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if start > end:
        return None
    try:
        calendar = load_trade_calendar_from_market_shards(
            root,
            start=str(start.date()),
            end=str(end.date()),
        )
    except Exception:
        return start, end
    if calendar.empty:
        return start, end
    trading_dates = _trading_dates_from_calendar(calendar)
    if trading_dates.empty:
        return None
    trading_dates = trading_dates[(trading_dates >= start.normalize()) & (trading_dates <= end.normalize())]
    if trading_dates.empty:
        return None
    return trading_dates.min(), trading_dates.max()


def is_catalog_range_covered(
    root: str | Path | None,
    *,
    dataset: str,
    symbol: str,
    start: str,
    end: str,
    freq: str | None = None,
    adjust: str | None = None,
) -> bool:
    effective_start = str(start)
    effective_end = str(end)
    if dataset != TRADE_CALENDAR_DATASET:
        window = _calendar_coverage_window(root, start=str(start), end=str(end))
        if window.has_trading_days is False:
            return True
        if window.has_trading_days is True:
            effective_start = window.start
            effective_end = window.end

    record = get_market_catalog_record(root, dataset=dataset, symbol=symbol, freq=freq, adjust=adjust)
    if not record or record.get("status") != "completed":
        return False
    if int(record.get("rows") or 0) <= 0:
        return False
    shard = ensure_data_root(root) / str(record.get("shard_path") or "")
    if not shard.exists():
        return False
    stored_start = pd.to_datetime(record.get("start_datetime"), errors="coerce")
    stored_end = pd.to_datetime(record.get("end_datetime"), errors="coerce")
    request_start = pd.to_datetime(effective_start, errors="coerce")
    request_end = pd.to_datetime(effective_end, errors="coerce")
    if any(pd.isna(value) for value in [stored_start, stored_end, request_start, request_end]):
        return False
    return stored_start.normalize() <= request_start.normalize() and stored_end.normalize() >= request_end.normalize()


def plan_catalog_download_ranges(
    root: str | Path | None,
    *,
    dataset: str,
    symbol: str,
    requested_start: str,
    requested_end: str,
    freq: str | None = None,
    adjust: str | None = None,
) -> list[tuple[str, str]]:
    effective_start = str(requested_start)
    effective_end = str(requested_end)
    if dataset != TRADE_CALENDAR_DATASET:
        window = _calendar_coverage_window(
            root,
            start=str(requested_start),
            end=str(requested_end),
        )
        if window.has_trading_days is False:
            return []
        if window.has_trading_days is True:
            effective_start = window.start
            effective_end = window.end

    if is_catalog_range_covered(root, dataset=dataset, symbol=symbol, start=effective_start, end=effective_end, freq=freq, adjust=adjust):
        return []
    record = get_market_catalog_record(root, dataset=dataset, symbol=symbol, freq=freq, adjust=adjust)
    if not record or record.get("status") != "completed":
        return [(effective_start, effective_end)]
    stored_start = pd.to_datetime(record.get("start_datetime"), errors="coerce")
    stored_end = pd.to_datetime(record.get("end_datetime"), errors="coerce")
    request_start = pd.to_datetime(effective_start, errors="coerce")
    request_end = pd.to_datetime(effective_end, errors="coerce")
    if any(pd.isna(value) for value in [stored_start, stored_end, request_start, request_end]):
        return [(effective_start, effective_end)]
    ranges: list[tuple[str, str]] = []
    stored_start = stored_start.normalize()
    stored_end = stored_end.normalize()
    request_start = request_start.normalize()
    request_end = request_end.normalize()
    if request_start < stored_start:
        prefix_end = stored_start - pd.Timedelta(days=1)
        normalized = _normalize_range_to_trading_days(root, start=request_start, end=prefix_end)
        if normalized is not None:
            prefix_start, prefix_end = normalized
            ranges.append((str(prefix_start.date()), str(prefix_end.date())))
    if request_end > stored_end:
        suffix_start = stored_end + pd.Timedelta(days=1)
        normalized = _normalize_range_to_trading_days(root, start=suffix_start, end=request_end)
        if normalized is not None:
            suffix_start, suffix_end = normalized
            ranges.append((str(suffix_start.date()), str(suffix_end.date())))
    return ranges or [(effective_start, effective_end)]

def delete_symbol_market_shards(root: str | Path | None, symbol: str) -> int:
    data_root = ensure_data_root(root)
    bucket = stable_bucket(symbol)
    symbol_dir = data_root / MARKET_PARTS_DIR / f"bucket={bucket}" / safe_symbol(symbol)
    deleted = 0
    if symbol_dir.exists():
        for child in symbol_dir.glob("*.parquet"):
            child.unlink(missing_ok=True)
            deleted += 1
        try:
            shutil.rmtree(symbol_dir)
        except OSError:
            pass
    db_file = catalog_path(root)
    if db_file.exists():
        with _CATALOG_LOCK:
            with duckdb.connect(str(db_file)) as conn:
                try:
                    conn.execute(f"DELETE FROM {CATALOG_TABLE} WHERE symbol = ?", [str(symbol).upper()])
                except duckdb.CatalogException:
                    pass
    return deleted


def save_kline_to_market_shard(
    df: pd.DataFrame,
    root: str | Path | None,
    *,
    replace_symbol: bool = False,
    update_catalog: bool = True,
) -> MarketShardWriteResult:
    normalized = _normalize_kline_dataframe(df)
    if normalized.empty:
        return MarketShardWriteResult(symbol="", ok=True, rows=0)

    symbols = normalized["symbol"].dropna().astype(str).str.upper().unique().tolist()
    freqs = normalized["freq"].dropna().astype(str).str.lower().unique().tolist()
    adjusts = normalized["adjust"].dropna().astype(str).str.lower().unique().tolist()
    if len(symbols) != 1 or len(freqs) != 1 or len(adjusts) != 1:
        raise ValueError("save_kline_to_market_shard expects exactly one symbol × freq × adjust slice")

    symbol = symbols[0]
    freq = freqs[0]
    adjust = adjusts[0]
    data_root = ensure_data_root(root)
    bucket = stable_bucket(symbol)
    path = kline_shard_path(data_root, symbol=symbol, freq=freq, adjust=adjust)

    if replace_symbol:
        delete_symbol_market_shards(data_root, symbol)

    existing = pd.DataFrame(columns=KLINE_SELECT_COLUMNS)
    if path.exists() and not replace_symbol:
        existing = _read_parquet(path)
        if not existing.empty:
            existing = _normalize_kline_dataframe(existing)

    # Avoid pandas FutureWarning from concatenating an empty schema-only frame
    # with the first real shard batch.  The first write should simply use the
    # normalized payload; later incremental writes merge real existing rows.
    if existing.empty:
        combined = normalized.copy()
    else:
        combined = pd.concat([existing, normalized], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "datetime", "freq", "adjust"], keep="last")
    combined = _refresh_pct_chg_in_memory(combined)
    combined = combined.sort_values(["symbol", "freq", "adjust", "datetime"]).reset_index(drop=True)
    combined = combined[KLINE_SELECT_COLUMNS]

    _write_parquet_atomic(combined, path)

    start_dt = str(pd.to_datetime(combined["datetime"], errors="coerce").min())[:19]
    end_dt = str(pd.to_datetime(combined["datetime"], errors="coerce").max())[:19]
    digest = _data_hash(combined, KLINE_SELECT_COLUMNS)
    rel_path = _relative(path, data_root)
    record = MarketShardRecord(
        dataset=KLINE_DATASET,
        symbol=symbol,
        freq=freq,
        adjust=adjust,
        shard_path=rel_path,
        bucket=bucket,
        rows=int(len(combined)),
        start_datetime=start_dt,
        end_datetime=end_dt,
        data_hash=digest,
        storage_format="parquet",
        status="completed",
        updated_at=now_utc_text(),
        error=None,
    )
    if update_catalog:
        upsert_market_catalog(data_root, record)

    return MarketShardWriteResult(
        symbol=symbol,
        ok=True,
        rows=int(len(normalized)),
        freq=freq,
        adjust=adjust,
        shard_path=rel_path,
        bucket=bucket,
        start_datetime=start_dt,
        end_datetime=end_dt,
        data_hash=digest,
    )


def save_trade_calendar_to_market_shard(
    df: pd.DataFrame,
    root: str | Path | None,
    *,
    update_catalog: bool = True,
) -> MarketShardWriteResult:
    normalized = normalize_trade_calendar_dataframe(df)
    path = trade_calendar_shard_path(root)
    data_root = ensure_data_root(root)
    existing = _read_parquet(path) if path.exists() else pd.DataFrame(columns=normalized.columns)
    combined = pd.concat([existing, normalized], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
    combined = combined.dropna(subset=["date"]).drop_duplicates(subset=["exchange", "date"], keep="last")
    combined = combined.sort_values(["exchange", "date"]).reset_index(drop=True)
    _write_parquet_atomic(combined, path)
    start_dt = str(pd.to_datetime(combined["date"], errors="coerce").min().date()) if not combined.empty else None
    end_dt = str(pd.to_datetime(combined["date"], errors="coerce").max().date()) if not combined.empty else None
    digest = _data_hash(combined, ["date", "is_trading_day", "exchange", "source"])
    rel_path = _relative(path, data_root)
    record = MarketShardRecord(
        dataset=TRADE_CALENDAR_DATASET,
        symbol="GLOBAL",
        freq="calendar",
        adjust="-",
        shard_path=rel_path,
        bucket=None,
        rows=int(len(combined)),
        start_datetime=start_dt,
        end_datetime=end_dt,
        data_hash=digest,
        storage_format="parquet",
        status="completed",
        updated_at=now_utc_text(),
    )
    if update_catalog:
        upsert_market_catalog(data_root, record)
    return MarketShardWriteResult(symbol="TRADE_CALENDAR", ok=True, rows=int(len(normalized)), freq="calendar", adjust="-", shard_path=rel_path, data_hash=digest, start_datetime=start_dt, end_datetime=end_dt)


def save_daily_liquidity_to_market_shard(
    df: pd.DataFrame,
    root: str | Path | None,
    *,
    update_catalog: bool = True,
) -> MarketShardWriteResult:
    normalized = normalize_stock_liquidity_dataframe(df)
    if normalized.empty:
        return MarketShardWriteResult(symbol="", ok=True, rows=0, freq=DAILY_LIQUIDITY_DATASET, adjust="-")
    symbols = normalized["symbol"].dropna().astype(str).str.upper().unique().tolist()
    if len(symbols) != 1:
        raise ValueError("save_daily_liquidity_to_market_shard expects exactly one symbol")
    symbol = symbols[0]
    path = daily_liquidity_shard_path(root, symbol=symbol)
    return _save_daily_extension(normalized, root, path, dataset=DAILY_LIQUIDITY_DATASET, symbol=symbol, columns=["symbol", "date", "turn", "source"], update_catalog=update_catalog)


def save_daily_status_to_market_shard(
    df: pd.DataFrame,
    root: str | Path | None,
    *,
    update_catalog: bool = True,
) -> MarketShardWriteResult:
    normalized = normalize_stock_status_dataframe(df)
    if normalized.empty:
        return MarketShardWriteResult(symbol="", ok=True, rows=0, freq=DAILY_STATUS_DATASET, adjust="-")
    symbols = normalized["symbol"].dropna().astype(str).str.upper().unique().tolist()
    if len(symbols) != 1:
        raise ValueError("save_daily_status_to_market_shard expects exactly one symbol")
    symbol = symbols[0]
    path = daily_status_shard_path(root, symbol=symbol)
    return _save_daily_extension(normalized, root, path, dataset=DAILY_STATUS_DATASET, symbol=symbol, columns=["symbol", "date", "is_st", "source"], update_catalog=update_catalog)


def refresh_kline_catalog_from_market_shard(
    root: str | Path | None,
    *,
    symbol: str,
    freq: str,
    adjust: str,
) -> MarketShardWriteResult:
    """Refresh one K-line catalog row from an existing parquet shard.

    Multiprocess BaoStock downloads write parquet shards in child processes but
    keep DuckDB catalog writes in the parent process.  This helper reads the
    finished shard and upserts the corresponding lightweight catalog record.
    """
    data_root = ensure_data_root(root)
    symbol_value = str(symbol).upper()
    freq_value = str(freq).lower()
    adjust_value = str(adjust).lower()
    path = kline_shard_path(data_root, symbol=symbol_value, freq=freq_value, adjust=adjust_value)
    if not path.exists():
        return MarketShardWriteResult(
            symbol=symbol_value,
            ok=False,
            rows=0,
            freq=freq_value,
            adjust=adjust_value,
            error=f"Shard file does not exist: {_relative(path, data_root)}",
        )

    df = _read_parquet(path)
    if df.empty:
        return MarketShardWriteResult(
            symbol=symbol_value,
            ok=False,
            rows=0,
            freq=freq_value,
            adjust=adjust_value,
            shard_path=_relative(path, data_root),
            error="Shard file is empty",
        )

    normalized = _normalize_kline_dataframe(df)
    if normalized.empty:
        return MarketShardWriteResult(
            symbol=symbol_value,
            ok=False,
            rows=0,
            freq=freq_value,
            adjust=adjust_value,
            shard_path=_relative(path, data_root),
            error="Shard file has no valid K-line rows after normalization",
        )
    normalized = normalized[KLINE_SELECT_COLUMNS]
    start_dt = str(pd.to_datetime(normalized["datetime"], errors="coerce").min())[:19]
    end_dt = str(pd.to_datetime(normalized["datetime"], errors="coerce").max())[:19]
    digest = _data_hash(normalized, KLINE_SELECT_COLUMNS)
    rel_path = _relative(path, data_root)
    bucket = stable_bucket(symbol_value)
    record = MarketShardRecord(
        dataset=KLINE_DATASET,
        symbol=symbol_value,
        freq=freq_value,
        adjust=adjust_value,
        shard_path=rel_path,
        bucket=bucket,
        rows=int(len(normalized)),
        start_datetime=start_dt,
        end_datetime=end_dt,
        data_hash=digest,
        storage_format="parquet",
        status="completed",
        updated_at=now_utc_text(),
        error=None,
    )
    upsert_market_catalog(data_root, record)
    return MarketShardWriteResult(
        symbol=symbol_value,
        ok=True,
        rows=int(len(normalized)),
        freq=freq_value,
        adjust=adjust_value,
        shard_path=rel_path,
        bucket=bucket,
        start_datetime=start_dt,
        end_datetime=end_dt,
        data_hash=digest,
    )


def refresh_daily_extension_catalog_from_market_shard(
    root: str | Path | None,
    *,
    symbol: str,
    dataset: str,
) -> MarketShardWriteResult:
    """Refresh one daily extension catalog row from an existing shard."""
    data_root = ensure_data_root(root)
    symbol_value = str(symbol).upper()
    dataset_value = str(dataset)
    if dataset_value == DAILY_LIQUIDITY_DATASET:
        path = daily_liquidity_shard_path(data_root, symbol=symbol_value)
        columns = ["symbol", "date", "turn", "source"]
    elif dataset_value == DAILY_STATUS_DATASET:
        path = daily_status_shard_path(data_root, symbol=symbol_value)
        columns = ["symbol", "date", "is_st", "source"]
    else:
        return MarketShardWriteResult(
            symbol=symbol_value,
            ok=False,
            rows=0,
            freq=dataset_value,
            adjust="-",
            error=f"Unsupported daily extension dataset: {dataset_value}",
        )

    if not path.exists():
        return MarketShardWriteResult(
            symbol=symbol_value,
            ok=False,
            rows=0,
            freq=dataset_value,
            adjust="-",
            error=f"Shard file does not exist: {_relative(path, data_root)}",
        )

    df = _read_parquet(path)
    if df.empty:
        return MarketShardWriteResult(
            symbol=symbol_value,
            ok=False,
            rows=0,
            freq=dataset_value,
            adjust="-",
            shard_path=_relative(path, data_root),
            error="Shard file is empty",
        )
    for column in columns:
        if column not in df.columns:
            df[column] = None
    safe = df[columns].copy()
    safe["symbol"] = safe["symbol"].astype(str).str.upper()
    safe["date"] = pd.to_datetime(safe["date"], errors="coerce").dt.date
    safe = safe.dropna(subset=["symbol", "date"])
    start_dt = str(pd.to_datetime(safe["date"], errors="coerce").min().date()) if not safe.empty else None
    end_dt = str(pd.to_datetime(safe["date"], errors="coerce").max().date()) if not safe.empty else None
    digest = _data_hash(safe, columns)
    rel_path = _relative(path, data_root)
    bucket = stable_bucket(symbol_value)
    record = MarketShardRecord(
        dataset=dataset_value,
        symbol=symbol_value,
        freq=dataset_value,
        adjust="-",
        shard_path=rel_path,
        bucket=bucket,
        rows=int(len(safe)),
        start_datetime=start_dt,
        end_datetime=end_dt,
        data_hash=digest,
        storage_format="parquet",
        status="completed",
        updated_at=now_utc_text(),
        error=None,
    )
    upsert_market_catalog(data_root, record)
    return MarketShardWriteResult(
        symbol=symbol_value,
        ok=True,
        rows=int(len(safe)),
        freq=dataset_value,
        adjust="-",
        shard_path=rel_path,
        bucket=bucket,
        data_hash=digest,
        start_datetime=start_dt,
        end_datetime=end_dt,
    )


def _save_daily_extension(
    normalized: pd.DataFrame,
    root: str | Path | None,
    path: Path,
    *,
    dataset: str,
    symbol: str,
    columns: list[str],
    update_catalog: bool,
) -> MarketShardWriteResult:
    data_root = ensure_data_root(root)
    existing = _read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
    combined = pd.concat([existing, normalized], ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
    combined = combined.dropna(subset=["symbol", "date"]).drop_duplicates(subset=["symbol", "date"], keep="last")
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    combined = combined[columns]
    _write_parquet_atomic(combined, path)
    start_dt = str(pd.to_datetime(combined["date"], errors="coerce").min().date()) if not combined.empty else None
    end_dt = str(pd.to_datetime(combined["date"], errors="coerce").max().date()) if not combined.empty else None
    digest = _data_hash(combined, columns)
    rel_path = _relative(path, data_root)
    bucket = stable_bucket(symbol)
    record = MarketShardRecord(
        dataset=dataset,
        symbol=symbol,
        freq=dataset,
        adjust="-",
        shard_path=rel_path,
        bucket=bucket,
        rows=int(len(combined)),
        start_datetime=start_dt,
        end_datetime=end_dt,
        data_hash=digest,
        storage_format="parquet",
        status="completed",
        updated_at=now_utc_text(),
    )
    if update_catalog:
        upsert_market_catalog(data_root, record)
    return MarketShardWriteResult(symbol=symbol, ok=True, rows=int(len(normalized)), freq=dataset, adjust="-", shard_path=rel_path, bucket=bucket, data_hash=digest, start_datetime=start_dt, end_datetime=end_dt)


def load_kline_from_market_shard(
    root: str | Path | None,
    *,
    symbol: str,
    freq: str,
    adjust: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    path = kline_shard_path(root, symbol=symbol, freq=freq, adjust=adjust)
    if not path.exists():
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS)
    query = "SELECT * FROM read_parquet(?) WHERE 1 = 1"
    params: list[object] = [str(path)]
    if start is not None:
        query += " AND datetime >= ?"
        params.append(start)
    if end is not None:
        query += " AND datetime <= ?"
        params.append(f"{end} 23:59:59.999999" if len(str(end)) == 10 else end)
    query += " ORDER BY datetime"
    with duckdb.connect(database=":memory:") as conn:
        return conn.execute(query, params).fetchdf()


def has_market_shard_storage(root: str | Path | None = None) -> bool:
    data_root = Path(root or DEFAULT_DATA_ROOT)
    return (data_root / CATALOG_DB_NAME).exists() and (data_root / MARKET_PARTS_DIR).exists()


def resolve_market_data_root(path: str | Path | None = None) -> Path:
    """Resolve either a storage root or a legacy market.duckdb path to the shard root."""
    raw = Path(path or DEFAULT_DATA_ROOT)
    candidates: list[Path] = []
    if raw.suffix.lower() == ".duckdb" and raw.name.lower() == "market.duckdb":
        candidates.append(raw.parent)
    elif raw.suffix.lower() != ".duckdb":
        candidates.append(raw)
    if not candidates:
        return raw

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / CATALOG_DB_NAME).exists() and (candidate / MARKET_PARTS_DIR).exists():
            return candidate
    return candidates[0]


def _completed_catalog_df(root: str | Path | None, *, dataset: str | None = None) -> pd.DataFrame:
    db_file = catalog_path(root)
    columns = [
        "dataset", "symbol", "freq", "adjust", "shard_path", "bucket", "rows",
        "start_datetime", "end_datetime", "data_hash", "storage_format", "status",
        "updated_at", "error",
    ]
    if not db_file.exists():
        return pd.DataFrame(columns=columns)
    query = f"SELECT {', '.join(columns)} FROM {CATALOG_TABLE} WHERE status = 'completed'"
    params: list[object] = []
    if dataset is not None:
        query += " AND dataset = ?"
        params.append(dataset)
    query += " ORDER BY symbol, freq, adjust, updated_at DESC"
    with duckdb.connect(str(db_file), read_only=True) as conn:
        try:
            return conn.execute(query, params).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=columns)


def get_market_shard_inventory(root: str | Path | None = None) -> pd.DataFrame:
    """Return inventory from the lightweight catalog without scanning shard files."""
    df = _completed_catalog_df(root, dataset=KLINE_DATASET)
    columns = ["symbol", "freq", "adjust", "source", "rows", "start_datetime", "end_datetime", "shard_path", "bucket", "storage_format", "data_hash"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    result = df.copy()
    result["source"] = "baostock"
    result = result[columns].sort_values(["symbol", "freq", "adjust"]).reset_index(drop=True)
    return result


def _catalog_records_for_kline(
    root: str | Path | None,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    freq: str | None = None,
    adjust: str | None = None,
) -> pd.DataFrame:
    catalog = _completed_catalog_df(root, dataset=KLINE_DATASET)
    if catalog.empty:
        return catalog
    result = catalog.copy()
    if symbol is not None:
        result = result[result["symbol"].astype(str).str.upper().eq(str(symbol).upper())]
    if symbols is not None:
        symbol_set = {str(item).upper() for item in symbols}
        if symbol_set:
            result = result[result["symbol"].astype(str).str.upper().isin(symbol_set)]
    if freq is not None:
        result = result[result["freq"].fillna("").astype(str).eq(str(freq))]
    if adjust is not None:
        result = result[result["adjust"].fillna("").astype(str).eq(str(adjust))]
    return result.reset_index(drop=True)


def _filter_datetime_range(df: pd.DataFrame, *, column: str, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    result = df.copy()
    values = pd.to_datetime(result[column], errors="coerce")
    if start is not None:
        result = result.loc[values >= pd.to_datetime(start, errors="coerce")]
        values = pd.to_datetime(result[column], errors="coerce")
    if end is not None:
        end_value = f"{end} 23:59:59.999999" if isinstance(end, str) and len(end) == 10 else end
        result = result.loc[values <= pd.to_datetime(end_value, errors="coerce")]
    return result


def load_kline_from_market_shards(
    root: str | Path | None,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    freq: str | None = None,
    adjust: str | None = None,
) -> pd.DataFrame:
    """Load K-line rows from partitioned parquet shards selected by catalog."""
    data_root = resolve_market_data_root(root)
    catalog = _catalog_records_for_kline(data_root, symbol=symbol, symbols=symbols, freq=freq, adjust=adjust)
    if catalog.empty:
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS)

    parts: list[pd.DataFrame] = []
    for record in catalog.to_dict(orient="records"):
        path = data_root / str(record.get("shard_path") or "")
        if not path.exists():
            continue
        df = _read_parquet(path)
        if df.empty:
            continue
        if start is not None or end is not None:
            df = _filter_datetime_range(df, column="datetime", start=start, end=end)
        if symbol is not None:
            df = df[df["symbol"].astype(str).str.upper().eq(str(symbol).upper())]
        if symbols is not None:
            symbol_set = {str(item).upper() for item in symbols}
            if symbol_set:
                df = df[df["symbol"].astype(str).str.upper().isin(symbol_set)]
        if freq is not None and "freq" in df.columns:
            df = df[df["freq"].fillna("").astype(str).eq(str(freq))]
        if adjust is not None and "adjust" in df.columns:
            df = df[df["adjust"].fillna("").astype(str).eq(str(adjust))]
        if not df.empty:
            for column in KLINE_SELECT_COLUMNS:
                if column not in df.columns:
                    df[column] = None
            parts.append(df[KLINE_SELECT_COLUMNS])
    if not parts:
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["symbol", "datetime", "freq", "adjust"]).reset_index(drop=True)
    return combined


def load_kline_page_from_market_shards(
    root: str | Path | None,
    *,
    symbol: str,
    freq: str | None = None,
    adjust: str | None = None,
    offset: int | None = 0,
    limit: int = 100,
    max_limit: int = 1200,
) -> tuple[pd.DataFrame, int]:
    limit = min(max(1, int(max_limit or 1200)), max(1, int(limit or 100)))
    df = load_kline_from_market_shards(root, symbol=symbol, freq=freq, adjust=adjust)
    if df.empty:
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS), 0
    df = df.sort_values("datetime").reset_index(drop=True)
    total = int(len(df))
    resolved_offset = max(0, total - limit) if offset is None else max(0, int(offset or 0))
    return df.iloc[resolved_offset: resolved_offset + limit].reset_index(drop=True), total


def load_trade_calendar_from_market_shards(
    root: str | Path | None,
    *,
    start: str | None = None,
    end: str | None = None,
    exchange: str | None = None,
) -> pd.DataFrame:
    columns = ["date", "is_trading_day", "exchange", "source"]
    data_root = resolve_market_data_root(root)
    record = get_market_catalog_record(data_root, dataset=TRADE_CALENDAR_DATASET, symbol="GLOBAL", freq="calendar", adjust="-")
    if not record:
        return pd.DataFrame(columns=columns)
    path = data_root / str(record.get("shard_path") or "")
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = _read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = _filter_datetime_range(df, column="date", start=start, end=end)
    if exchange is not None and "exchange" in df.columns:
        df = df[df["exchange"].astype(str).eq(str(exchange))]
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df[columns].sort_values(["exchange", "date"]).reset_index(drop=True)


def _load_daily_extension_from_market_shards(
    root: str | Path | None,
    *,
    dataset: str,
    columns: list[str],
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    data_root = resolve_market_data_root(root)
    catalog = _completed_catalog_df(data_root, dataset=dataset)
    if catalog.empty:
        return pd.DataFrame(columns=columns)
    if symbol is not None:
        catalog = catalog[catalog["symbol"].astype(str).str.upper().eq(str(symbol).upper())]
    if symbols is not None:
        symbol_set = {str(item).upper() for item in symbols}
        if symbol_set:
            catalog = catalog[catalog["symbol"].astype(str).str.upper().isin(symbol_set)]
    parts: list[pd.DataFrame] = []
    for record in catalog.to_dict(orient="records"):
        path = data_root / str(record.get("shard_path") or "")
        if not path.exists():
            continue
        df = _read_parquet(path)
        if df.empty:
            continue
        df = _filter_datetime_range(df, column="date", start=start, end=end)
        for column in columns:
            if column not in df.columns:
                df[column] = None
        parts.append(df[columns])
    if not parts:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(parts, ignore_index=True)
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_daily_liquidity_from_market_shards(
    root: str | Path | None,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    return _load_daily_extension_from_market_shards(
        root,
        dataset=DAILY_LIQUIDITY_DATASET,
        columns=["symbol", "date", "turn", "source"],
        symbol=symbol,
        symbols=symbols,
        start=start,
        end=end,
    )


def load_daily_status_from_market_shards(
    root: str | Path | None,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    return _load_daily_extension_from_market_shards(
        root,
        dataset=DAILY_STATUS_DATASET,
        columns=["symbol", "date", "is_st", "source"],
        symbol=symbol,
        symbols=symbols,
        start=start,
        end=end,
    )


def _normalize_delete_filter(item: object) -> dict[str, str | None]:
    if isinstance(item, dict):
        raw_symbol = item.get("symbol")
        raw_freq = item.get("freq")
        raw_adjust = item.get("adjust")
    else:
        raw_symbol = getattr(item, "symbol", None)
        raw_freq = getattr(item, "freq", None)
        raw_adjust = getattr(item, "adjust", None)

    symbol = str(raw_symbol or "").strip().upper()

    def _optional(value: object) -> str | None:
        if value in (None, "", "-"):
            return None
        text = str(value).strip()
        return text or None

    return {
        "symbol": symbol,
        "freq": _optional(raw_freq),
        "adjust": _optional(raw_adjust),
    }


def _cleanup_empty_market_dirs(path: Path, stop_at: Path) -> None:
    current = path
    stop = stop_at.resolve()
    while current.exists():
        try:
            if current.resolve() == stop:
                break
        except OSError:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def delete_many_symbol_market_slices(
    root: str | Path | None,
    items: Iterable[object],
    *,
    include_full_symbol_extensions: bool = True,
) -> MarketShardDeleteResult:
    """Delete many partitioned market shards with one catalog scan and one catalog update.

    This is the fast path for the Data Layer inventory delete action.  The old
    compatibility functions deleted one row at a time, reopening DuckDB and
    rescanning the full catalog for every selected slice.  Here we read the
    catalog once, unlink all matched shard files, then delete all matched catalog
    rows in a single transaction.
    """
    data_root = resolve_market_data_root(root)
    filters = [_normalize_delete_filter(item) for item in items]
    filters = [item for item in filters if item["symbol"]]
    if not filters:
        return MarketShardDeleteResult()

    catalog = _completed_catalog_df(data_root)
    if catalog.empty:
        return MarketShardDeleteResult()

    matched_frames: list[pd.DataFrame] = []
    for item in filters:
        symbol = str(item["symbol"])
        freq = item["freq"]
        adjust = item["adjust"]
        is_full_symbol_delete = freq is None and adjust is None

        datasets = [KLINE_DATASET]
        if include_full_symbol_extensions and is_full_symbol_delete:
            datasets.extend([DAILY_LIQUIDITY_DATASET, DAILY_STATUS_DATASET])

        subset = catalog[
            catalog["dataset"].astype(str).isin(datasets)
            & catalog["symbol"].astype(str).str.upper().eq(symbol)
        ]
        if not is_full_symbol_delete:
            if freq is not None:
                subset = subset[subset["freq"].fillna("").astype(str).eq(str(freq))]
            if adjust is not None:
                subset = subset[subset["adjust"].fillna("").astype(str).eq(str(adjust))]
        if not subset.empty:
            matched_frames.append(subset)

    if not matched_frames:
        return MarketShardDeleteResult()

    matched = pd.concat(matched_frames, ignore_index=True).drop_duplicates(
        subset=["dataset", "symbol", "freq", "adjust", "shard_path"],
        keep="last",
    )
    if matched.empty:
        return MarketShardDeleteResult()

    deleted_rows = int(matched.loc[matched["dataset"].eq(KLINE_DATASET), "rows"].fillna(0).astype("int64").sum())
    deleted_extension_rows = int(
        matched.loc[matched["dataset"].isin([DAILY_LIQUIDITY_DATASET, DAILY_STATUS_DATASET]), "rows"]
        .fillna(0)
        .astype("int64")
        .sum()
    )

    market_parts = data_root / MARKET_PARTS_DIR
    deleted_files = 0
    touched_dirs: set[Path] = set()
    for shard_path in matched["shard_path"].dropna().astype(str).unique().tolist():
        path = data_root / shard_path
        touched_dirs.add(path.parent)
        if path.exists():
            path.unlink(missing_ok=True)
            deleted_files += 1

    for directory in sorted(touched_dirs, key=lambda value: len(value.parts), reverse=True):
        _cleanup_empty_market_dirs(directory, market_parts)

    delete_keys = matched[["dataset", "symbol", "freq", "adjust"]].drop_duplicates().copy()
    db_file = catalog_path(data_root)
    if db_file.exists() and not delete_keys.empty:
        with _CATALOG_LOCK:
            with duckdb.connect(str(db_file)) as conn:
                try:
                    conn.register("_delete_market_shard_keys", delete_keys)
                    conn.execute("BEGIN TRANSACTION")
                    conn.execute(
                        f"""
                        DELETE FROM {CATALOG_TABLE}
                        USING _delete_market_shard_keys AS keys
                        WHERE {CATALOG_TABLE}.dataset = keys.dataset
                          AND {CATALOG_TABLE}.symbol = keys.symbol
                          AND COALESCE({CATALOG_TABLE}.freq, '') = COALESCE(keys.freq, '')
                          AND COALESCE({CATALOG_TABLE}.adjust, '') = COALESCE(keys.adjust, '')
                        """
                    )
                    conn.execute("COMMIT")
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                finally:
                    try:
                        conn.unregister("_delete_market_shard_keys")
                    except Exception:
                        pass

    return MarketShardDeleteResult(
        deleted_rows=deleted_rows,
        deleted_extension_rows=deleted_extension_rows,
        deleted_files=deleted_files,
        matched_catalog_rows=int(len(matched)),
    )


def delete_symbol_market_slice(root: str | Path | None, symbol: str, *, freq: str | None = None, adjust: str | None = None) -> int:
    data_root = resolve_market_data_root(root)
    catalog = _catalog_records_for_kline(data_root, symbol=symbol, freq=freq, adjust=adjust)
    deleted_rows = 0
    for record in catalog.to_dict(orient="records"):
        path = data_root / str(record.get("shard_path") or "")
        deleted_rows += int(record.get("rows") or 0)
        path.unlink(missing_ok=True)
    db_file = catalog_path(data_root)
    if db_file.exists():
        with _CATALOG_LOCK:
            with duckdb.connect(str(db_file)) as conn:
                query = f"DELETE FROM {CATALOG_TABLE} WHERE dataset = ? AND symbol = ?"
                params: list[object] = [KLINE_DATASET, str(symbol).upper()]
                if freq is not None:
                    query += " AND COALESCE(freq, '') = COALESCE(?, '')"
                    params.append(freq)
                if adjust is not None:
                    query += " AND COALESCE(adjust, '') = COALESCE(?, '')"
                    params.append(adjust)
                conn.execute(query, params)
    return deleted_rows
