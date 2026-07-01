from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd

from download_layer.config.data_catalog import catalog_payload
from download_layer.services.data_collector import DataCollector, SymbolDownloadResult, read_symbols_from_file
from data_layer.storage.duckdb_storage import (
    is_daily_liquidity_range_covered,
    is_trade_calendar_range_covered,
    plan_kline_download_ranges,
    record_download_manifest,
    verify_completed_download_manifest,
)
from data_layer.storage.partitioned_storage import (
    DAILY_LIQUIDITY_DATASET,
    DAILY_STATUS_DATASET,
    KLINE_DATASET,
    TRADE_CALENDAR_DATASET,
    delete_symbol_market_shards,
    build_market_coverage_snapshot,
    get_market_catalog_record,
    is_catalog_range_covered,
    is_catalog_range_covered_in_snapshot,
    is_daily_extension_range_covered,
    is_daily_extension_range_covered_in_snapshot,
    plan_catalog_download_ranges,
    plan_catalog_download_ranges_in_snapshot,
    refresh_daily_extension_catalog_from_market_shard,
    refresh_kline_catalog_from_market_shard,
)

from app_layer.backend.jobs import JobManager
from app_layer.backend.json_utils import dataframe_to_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = "runtime_layer/data"
DEFAULT_STORAGE_ROOT = "runtime_layer/data"
DEFAULT_DOWNLOAD_REPORT_DIR = "runtime_layer/runs"
ADJUST_FLAG_TO_NAME = {"1": "post", "2": "pre", "3": "none"}
FREQ_TO_STORAGE = {
    "d": "daily",
    "daily": "daily",
    "w": "weekly",
    "weekly": "weekly",
    "15": "15min",
    "15min": "15min",
    "15m": "15min",
    "5": "5min",
    "5min": "5min",
    "5m": "5min",
    "60": "60min",
    "60min": "60min",
    "1h": "60min",
    "30": "30min",
    "30min": "30min",
}
STORAGE_TO_BAOSTOCK = {
    "daily": "d",
    "weekly": "w",
    "15min": "15",
    "5min": "5",
    "60min": "60",
    "30min": "30",
}
DOWNLOAD_PROGRESS_WEIGHTS = {
    "weekly": 1.0,
    "daily": 5.0,
    "5min": 240.0,
    "15min": 80.0,
    "60min": 20.0,
    "30min": 40.0,
}
CALENDAR_PROGRESS_WEIGHT = 1.0
DAILY_LIQUIDITY_PROGRESS_WEIGHT = 3.0


def add_job_log(jobs: JobManager, job_id: str, message: str, *, level: str = "info") -> None:
    add_log = getattr(jobs, "add_log", None)
    if callable(add_log):
        add_log(job_id, message, level=level)


def storage_adjust_for_freq(freq: str, adjustflag: str) -> str:
    # Store the requested adjustment mode for every frequency.  If a local slice
    # exists with a different adjustment value, manifest verification will not
    # match and the selected adjustment will be downloaded/saved separately.
    return ADJUST_FLAG_TO_NAME.get(str(adjustflag), str(adjustflag))


def download_progress_weight(freq: str) -> float:
    return DOWNLOAD_PROGRESS_WEIGHTS.get(str(freq or "").lower(), 5.0)


def total_download_progress_weight(symbols: list[str], freqs: list[str], adjustflags: list[str]) -> float:
    kline_weight = sum(download_progress_weight(freq) for freq in freqs) * len(symbols) * len(adjustflags)
    liquidity_weight = DAILY_LIQUIDITY_PROGRESS_WEIGHT * len(symbols)
    return max(1.0, CALENDAR_PROGRESS_WEIGHT + kline_weight + liquidity_weight)


def resolve_project_path(path: str | Path | None, default: str | Path | None = None) -> Path:
    raw = Path(path or default or "")
    if raw.is_absolute():
        return raw
    return PROJECT_ROOT / raw


def normalize_symbol_list(symbols: list[str] | str | None) -> list[str]:
    if symbols is None:
        return []
    if isinstance(symbols, str):
        parts = symbols.replace(",", "\n").splitlines()
    else:
        parts = [str(item) for item in symbols]
    result: list[str] = []
    for item in parts:
        symbol = item.strip().upper()
        if symbol and symbol not in result:
            result.append(symbol)
    return result


def normalize_freqs(raw: Any) -> list[str]:
    if raw is None:
        raw = ["d"]
    if isinstance(raw, str):
        values = raw.replace(",", "\n").splitlines()
    else:
        values = [str(item) for item in raw]

    result: list[str] = []
    for value in values:
        key = str(value).strip().lower()
        storage = FREQ_TO_STORAGE.get(key)
        if storage is None:
            raise ValueError(f"Unsupported frequency: {value}")
        if storage not in result:
            result.append(storage)
    return result or ["daily"]



def download_catalog() -> dict[str, Any]:
    """Return current and planned downloadable dataset catalog for the app layer."""
    return catalog_payload()

def normalize_adjustflags(raw: Any) -> list[str]:
    """
    Normalize requested BaoStock adjustment flags.

    Supported:
        1 = post-adjusted
        2 = pre-adjusted
        3 = none / raw price

    PocketAgent default:
        ["3", "2"] = raw execution price + pre-adjusted technical-analysis price
    """
    if raw is None:
        raw = ["3", "2"]

    if isinstance(raw, str):
        values = raw.replace(",", "\n").splitlines()
    else:
        values = [str(item) for item in raw]

    result: list[str] = []
    for value in values:
        key = str(value).strip()
        if key not in ADJUST_FLAG_TO_NAME:
            raise ValueError(f"Unsupported adjustflag: {value}. Expected one of 1, 2, 3.")
        if key not in result:
            result.append(key)

    return result or ["3", "2"]


def _resolve_symbols(payload: dict[str, Any]) -> list[str]:
    symbols = normalize_symbol_list(payload.get("symbols") or payload.get("manual_symbols"))
    symbols_file = payload.get("symbols_file")
    if symbols_file:
        file_symbols = read_symbols_from_file(resolve_project_path(symbols_file))
        for symbol in normalize_symbol_list(file_symbols):
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols


def start_download_job(payload: dict[str, Any], jobs: JobManager) -> dict[str, Any]:
    active = jobs.active_job("download")
    if active is not None:
        raise RuntimeError(
            f"A download job is already {active.get('status')}: {active.get('job_id')}. "
            "Stop it or wait for it to finish before starting another download."
        )

    symbols = _resolve_symbols(payload)
    if not symbols:
        raise ValueError("No symbols provided. Use manual symbols or a symbol file.")

    freqs = normalize_freqs(payload.get("freqs") or payload.get("freq") or ["d"])
    adjustflags = normalize_adjustflags(
        payload.get("adjustflags")
        if payload.get("adjustflags") is not None
        else payload.get("adjustflag")
    )
    storage_mode = str(payload.get("storage_mode") or payload.get("storage") or "shard").lower()
    if storage_mode not in {"shard", "duckdb"}:
        storage_mode = "shard"
    storage_root = resolve_project_path(payload.get("storage_root"), DEFAULT_STORAGE_ROOT)
    workers = max(1, int(payload.get("workers") or 1))

    extension_tasks = 1 + len(symbols)
    total_tasks = len(symbols) * len(freqs) * len(adjustflags) + extension_tasks
    job_id = jobs.create_job("download", title="Download market data")

    jobs.update_job(
        job_id,
        total=total_tasks,
        message=(
            f"Queued {len(symbols)} symbols × {len(freqs)} frequencies "
            f"× {len(adjustflags)} adjustment modes"
        ),
    )

    adjust_names = [ADJUST_FLAG_TO_NAME.get(flag, flag) for flag in adjustflags]
    add_job_log(
        jobs,
        job_id,
        f"Queued {len(symbols)} symbols × {len(freqs)} frequencies × {len(adjustflags)} adjustments; "
        f"adjustflags={adjustflags} ({adjust_names}); "
        f"requested_skip_existing={payload.get('skip_existing', True)}, "
        f"replace={payload.get('replace_symbol') or payload.get('replace_existing') or False}",
    )

    jobs.submit(job_id, _run_download_job, job_id, payload, symbols, freqs, adjustflags, jobs)

    return {
        "job_id": job_id,
        "status": "queued",
        "total": total_tasks,
        "symbols": len(symbols),
        "freqs": freqs,
        "adjustflags": adjustflags,
        "storage_mode": storage_mode,
        "storage_root": str(
            storage_root.relative_to(PROJECT_ROOT)
            if storage_root.is_relative_to(PROJECT_ROOT)
            else storage_root
        ),
        "workers": workers,
    }





def _app_symbol_result_weight(result: SymbolDownloadResult) -> float:
    if str(result.freq or "") == "daily_liquidity":
        return DAILY_LIQUIDITY_PROGRESS_WEIGHT
    return download_progress_weight(str(result.freq or "daily"))


def _result_is_skipped(result: SymbolDownloadResult) -> bool:
    return str(result.error or "").startswith("Skipped:")


def _result_is_warning(result: SymbolDownloadResult) -> bool:
    return str(result.error or "").startswith("Warning:")


def _download_report_status(result: SymbolDownloadResult) -> str:
    if _result_is_skipped(result):
        return "skipped"
    if _result_is_warning(result):
        return "warning"
    return "success" if result.ok else "failed"


def _download_report_error(result: SymbolDownloadResult) -> str | None:
    return None if _result_is_skipped(result) else result.error


def _is_daily_extension_bundle_covered(
    *,
    storage_mode: str,
    db_path: Path,
    storage_root: Path,
    coverage_snapshot: Any | None,
    symbol: str,
    start: str,
    end: str,
) -> bool:
    """Return True when both turnover and historical ST-status are covered."""
    if storage_mode == "shard":
        def covered(dataset: str) -> bool:
            if coverage_snapshot is not None:
                return is_daily_extension_range_covered_in_snapshot(
                    coverage_snapshot,
                    dataset=dataset,
                    symbol=symbol,
                    start=start,
                    end=end,
                    verify_hash=True,
                )
            return is_daily_extension_range_covered(
                storage_root,
                dataset=dataset,
                symbol=symbol,
                start=start,
                end=end,
                verify_hash=True,
            )

        return covered(DAILY_LIQUIDITY_DATASET) and covered(DAILY_STATUS_DATASET)

    return is_daily_liquidity_range_covered(
        db_path,
        symbol=symbol,
        start=start,
        end=end,
    )


def _download_symbol_bundle_shard_for_app(
    *,
    symbol: str,
    db_path: Path,
    storage_root: Path,
    start: str,
    end: str,
    kline_tasks: list[dict[str, Any]],
    download_liquidity: bool,
    replace_symbol: bool,
    request_sleep: float,
) -> tuple[list[SymbolDownloadResult], list[tuple[str, str]]]:
    """Download one symbol bundle into shard storage in a child process.

    Child processes must not read or write ``market_catalog.duckdb``.  The
    parent process plans skip/range work before the pool starts and refreshes
    the catalog only after the pool has fully stopped.  This avoids Windows
    DuckDB file locks between worker readers and parent writers.
    """
    collector = DataCollector(
        db_path=db_path,
        storage_root=storage_root,
        storage_mode="shard",
        request_sleep_seconds=request_sleep,
        update_catalog=False,
    )
    results: list[SymbolDownloadResult] = []
    logs: list[tuple[str, str]] = []
    replaced = False
    logged_in = False

    try:
        for task in kline_tasks:
            freq = str(task["freq"])
            adjustflag = str(task["adjustflag"])
            storage_adjust = str(task["adjust"])
            planned_ranges = [(str(a), str(b)) for a, b in task.get("ranges", [(start, end)])]
            baostock_freq = STORAGE_TO_BAOSTOCK[freq]
            label = f"{symbol} / {freq} / {storage_adjust}"

            if not logged_in:
                collector.client.login()
                logged_in = True
            logs.append(("info", f"REQUEST {label}: ranges={planned_ranges}"))

            chunk_results: list[SymbolDownloadResult] = []
            replace_this = bool(replace_symbol and not replaced)
            for range_index, (chunk_start, chunk_end) in enumerate(planned_ranges):
                result = collector.download_kline(
                    symbol,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    frequency=baostock_freq,
                    adjustflag=adjustflag,
                    replace_symbol=replace_this and range_index == 0,
                )
                result.freq = result.freq or freq
                result.adjust = result.adjust or storage_adjust
                chunk_results.append(result)
                if request_sleep > 0 and range_index < len(planned_ranges) - 1:
                    sleep(request_sleep)
            if replace_this:
                replaced = True

            if len(chunk_results) == 1:
                result = chunk_results[0]
            else:
                failure_errors = [str(item.error) for item in chunk_results if not item.ok and item.error]
                warning_errors = [str(item.error) for item in chunk_results if item.ok and _result_is_warning(item)]
                messages = failure_errors or warning_errors
                result = SymbolDownloadResult(
                    symbol=symbol,
                    ok=all(item.ok for item in chunk_results),
                    rows=sum(int(item.rows) for item in chunk_results),
                    error="; ".join(messages) if messages else None,
                    freq=freq,
                    adjust=storage_adjust,
                )
            results.append(result)
            if _result_is_warning(result):
                logs.append(("warning", f"WARN {label}: {result.error}"))
            elif result.ok:
                logs.append(("info", f"OK {label}: saved {result.rows} rows"))
            else:
                logs.append(("error", f"FAILED {label}: {result.error}"))
            if request_sleep > 0:
                sleep(request_sleep)

        if download_liquidity:
            label = f"{symbol} / daily_liquidity"
            if not logged_in:
                collector.client.login()
                logged_in = True
            result = collector.download_daily_liquidity(symbol, start_date=start, end_date=end)
            results.append(result)
            if _result_is_warning(result):
                logs.append(("warning", f"WARN {label}: {result.error}"))
            elif result.ok:
                logs.append(("info", f"OK {label}: saved {result.rows} rows"))
            else:
                logs.append(("error", f"FAILED {label}: {result.error}"))
    finally:
        if logged_in:
            collector.client.logout()

    return results, logs


def _shutdown_process_pool_now(executor: ProcessPoolExecutor) -> None:
    """Best-effort hard stop for BaoStock child processes after user cancel."""
    for process in list(getattr(executor, "_processes", {}) .values()):
        try:
            process.terminate()
        except Exception:
            pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)


def _refresh_catalog_after_process_result(
    storage_root: Path,
    result: SymbolDownloadResult,
) -> list[tuple[str, str]]:
    """Refresh shard catalog entries in the parent process after a worker finishes.

    BaoStock is not thread-safe, so app-level parallel download uses separate
    processes.  Child processes only write parquet shards; keeping catalog writes
    here avoids concurrent DuckDB writers against market_catalog.duckdb.
    """
    messages: list[tuple[str, str]] = []
    if not result.ok or int(result.rows or 0) <= 0:
        return messages
    if str(result.error or "").startswith("Skipped:"):
        return messages

    symbol = str(result.symbol or "").upper()
    freq = str(result.freq or "")
    adjust = str(result.adjust or "-")
    try:
        if freq == "daily_liquidity":
            liquidity = refresh_daily_extension_catalog_from_market_shard(
                storage_root,
                symbol=symbol,
                dataset=DAILY_LIQUIDITY_DATASET,
            )
            status = refresh_daily_extension_catalog_from_market_shard(
                storage_root,
                symbol=symbol,
                dataset=DAILY_STATUS_DATASET,
            )
            if liquidity.ok:
                messages.append(("info", f"CATALOG {symbol} / daily_liquidity: rows={liquidity.rows}"))
            else:
                messages.append(("warning", f"CATALOG {symbol} / daily_liquidity skipped: {liquidity.error}"))
            if status.ok:
                messages.append(("info", f"CATALOG {symbol} / daily_status: rows={status.rows}"))
            else:
                messages.append(("warning", f"CATALOG {symbol} / daily_status skipped: {status.error}"))
            return messages

        refreshed = refresh_kline_catalog_from_market_shard(
            storage_root,
            symbol=symbol,
            freq=freq,
            adjust=adjust,
        )
        if refreshed.ok:
            messages.append((
                "info",
                f"CATALOG {symbol} / {freq} / {adjust}: rows={refreshed.rows}, hash={str(refreshed.data_hash or '')[:12]}",
            ))
        else:
            messages.append(("error", f"CATALOG {symbol} / {freq} / {adjust} failed: {refreshed.error}"))
    except Exception as exc:
        messages.append(("error", f"CATALOG {symbol} / {freq} / {adjust} failed: {exc}"))
    return messages


def _run_download_job(
    job_id: str,
    payload: dict[str, Any],
    symbols: list[str],
    freqs: list[str],
    adjustflags: list[str],
    jobs: JobManager,
) -> dict[str, Any]:
    storage_mode = str(payload.get("storage_mode") or payload.get("storage") or "shard").lower()
    if storage_mode not in {"shard", "duckdb"}:
        storage_mode = "shard"
    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    storage_root = resolve_project_path(payload.get("storage_root"), DEFAULT_STORAGE_ROOT)
    start = payload.get("start") or payload.get("start_date")
    end = payload.get("end") or payload.get("end_date")
    if not start or not end:
        raise ValueError("Both start and end dates are required.")

    replace_symbol = bool(payload.get("replace_symbol") or payload.get("replace_existing") or False)
    requested_skip_existing = bool(payload.get("skip_existing", True))
    skip_existing = requested_skip_existing and not replace_symbol
    request_sleep = float(payload.get("sleep") or payload.get("request_sleep_seconds") or 0.2)
    workers = max(1, int(payload.get("workers") or 1))
    if storage_mode != "shard":
        workers = 1

    report_path_raw = payload.get("report_path") or payload.get("report")
    if report_path_raw:
        report_path = resolve_project_path(report_path_raw)
    else:
        report_path = resolve_project_path(f"{DEFAULT_DOWNLOAD_REPORT_DIR}/download_{job_id}_report.csv")

    collector = DataCollector(
        db_path=db_path,
        storage_root=storage_root,
        storage_mode=storage_mode,
        request_sleep_seconds=request_sleep,
    )

    results: list[SymbolDownloadResult] = []
    saved_rows = 0
    succeeded = 0
    failed = 0
    skipped = 0
    completed = 0
    total = len(symbols) * len(freqs) * len(adjustflags) + 1 + len(symbols)
    completed_weight = 0.0
    total_weight = total_download_progress_weight(symbols, freqs, adjustflags)
    replaced_symbols: set[str] = set()

    adjust_names = [ADJUST_FLAG_TO_NAME.get(flag, flag) for flag in adjustflags]

    jobs.update_job(job_id, status="running", progress=0.0, message="Checking local manifest and inventory")
    add_job_log(
        jobs,
        job_id,
        f"CONFIG storage={storage_mode}, db={db_path}, storage_root={storage_root}, start={start}, end={end}, "
        f"adjustflags={adjustflags} ({adjust_names}), replace={replace_symbol}, "
        f"requested_skip_existing={requested_skip_existing}, effective_skip_existing={skip_existing}, "
        f"sleep={request_sleep}, workers={workers}",
    )

    if replace_symbol and requested_skip_existing:
        add_job_log(jobs, job_id, "replace=True, so skip_existing is disabled for this run")

    client_logged_in = False
    calendar_ok = False
    calendar_error: str | None = None

    def progress_value() -> float:
        return min(1.0, max(0.0, completed_weight / total_weight if total_weight else 1.0))

    try:
        jobs.update_job(
            job_id,
            current="trade_calendar",
            completed=completed,
            total=total,
            progress=progress_value(),
            message="Downloading required trading calendar",
        )
        add_job_log(jobs, job_id, "REQUEST trade_calendar: required for future progress features")

        calendar_covered = (
            is_catalog_range_covered(
                storage_root,
                dataset=TRADE_CALENDAR_DATASET,
                symbol="GLOBAL",
                freq="calendar",
                adjust="-",
                start=str(start),
                end=str(end),
            )
            if storage_mode == "shard"
            else is_trade_calendar_range_covered(db_path, start=str(start), end=str(end))
        )
        if skip_existing and calendar_covered:
            results.append(
                SymbolDownloadResult(
                    symbol="TRADE_CALENDAR",
                    ok=True,
                    rows=0,
                    error="Skipped: local trade calendar coverage verified",
                    freq="calendar",
                    adjust="-",
                )
            )
            skipped += 1
            succeeded += 1
            completed += 1
            completed_weight += CALENDAR_PROGRESS_WEIGHT
            calendar_ok = True
            add_job_log(jobs, job_id, "SKIP trade_calendar: local coverage verified")
        else:
            if not client_logged_in:
                collector.client.login()
                client_logged_in = True

            calendar_result = collector.download_trade_calendar(start_date=str(start), end_date=str(end))
            results.append(
                SymbolDownloadResult(
                    symbol="TRADE_CALENDAR",
                    ok=calendar_result.ok,
                    rows=calendar_result.rows,
                    error=calendar_result.error,
                    freq="calendar",
                    adjust="-",
                )
            )
            saved_rows += int(calendar_result.rows)
            completed += 1
            completed_weight += CALENDAR_PROGRESS_WEIGHT
            calendar_ok = bool(calendar_result.ok)
            calendar_error = calendar_result.error
            if calendar_result.ok:
                succeeded += 1
                add_job_log(jobs, job_id, f"OK trade_calendar: saved {calendar_result.rows} rows")
            else:
                failed += 1
                freqs = []
                jobs.update_job(job_id, status="failed", error=calendar_result.error)
                add_job_log(jobs, job_id, f"FAILED trade_calendar: {calendar_result.error}", level="error")

        jobs.update_job(
            job_id,
            current="trade_calendar",
            completed=completed,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            saved_rows=saved_rows,
            progress=progress_value(),
            message="Trading calendar ready" if calendar_ok else f"Trading calendar failed: {calendar_error}",
        )

        coverage_snapshot = None
        if storage_mode == "shard" and calendar_ok and not jobs.is_cancel_requested(job_id):
            try:
                coverage_snapshot = build_market_coverage_snapshot(storage_root, start=str(start), end=str(end))
                add_job_log(
                    jobs,
                    job_id,
                    f"Planning snapshot loaded: catalog_records={len(coverage_snapshot.records)}, "
                    f"trading_days={len(coverage_snapshot.trading_dates)}",
                )
            except Exception as exc:
                add_job_log(jobs, job_id, f"Planning snapshot unavailable; falling back to per-slice checks: {exc}", level="warn")
                coverage_snapshot = None

        if storage_mode == "shard" and workers > 1 and calendar_ok and not jobs.is_cancel_requested(job_id):
            add_job_log(
                jobs,
                job_id,
                f"Multiprocess shard download enabled: workers={workers}; each worker uses its own BaoStock session; "
                "child processes do not touch market_catalog.duckdb; catalog refresh runs after the pool stops",
            )
            if replace_symbol:
                add_job_log(jobs, job_id, "REPLACE: deleting selected symbol shards in parent before process pool starts")
                for symbol in symbols:
                    delete_symbol_market_shards(storage_root, symbol)

            def consume_result(result: SymbolDownloadResult) -> None:
                nonlocal saved_rows, succeeded, failed, skipped, completed, completed_weight
                results.append(result)
                saved_rows += int(result.rows)
                completed += 1
                completed_weight += _app_symbol_result_weight(result)
                if result.ok:
                    succeeded += 1
                    if str(result.error or "").startswith("Skipped:"):
                        skipped += 1
                else:
                    failed += 1

            planning_skipped_before = skipped
            planned_symbol_tasks: list[dict[str, Any]] = []
            for symbol in symbols:
                kline_tasks: list[dict[str, Any]] = []
                for freq in freqs:
                    for adjustflag in adjustflags:
                        storage_adjust = storage_adjust_for_freq(freq, adjustflag)
                        label = f"{symbol} / {freq} / {storage_adjust}"
                        planned_ranges = [(str(start), str(end))]
                        if skip_existing and not replace_symbol:
                            if coverage_snapshot is not None:
                                planned_ranges = plan_catalog_download_ranges_in_snapshot(
                                    coverage_snapshot,
                                    dataset=KLINE_DATASET,
                                    symbol=symbol,
                                    requested_start=str(start),
                                    requested_end=str(end),
                                    freq=freq,
                                    adjust=storage_adjust,
                                )
                            else:
                                planned_ranges = plan_catalog_download_ranges(
                                    storage_root,
                                    dataset=KLINE_DATASET,
                                    symbol=symbol,
                                    requested_start=str(start),
                                    requested_end=str(end),
                                    freq=freq,
                                    adjust=storage_adjust,
                                )
                            if not planned_ranges:
                                result = SymbolDownloadResult(
                                    symbol=symbol,
                                    ok=True,
                                    rows=0,
                                    error="Skipped: catalog range coverage verified",
                                    freq=freq,
                                    adjust=storage_adjust,
                                )
                                consume_result(result)
                                continue
                        kline_tasks.append(
                            {
                                "freq": freq,
                                "adjustflag": adjustflag,
                                "adjust": storage_adjust,
                                "ranges": planned_ranges,
                            }
                        )

                liquidity_covered = _is_daily_extension_bundle_covered(
                    storage_mode=storage_mode,
                    db_path=db_path,
                    storage_root=storage_root,
                    coverage_snapshot=coverage_snapshot,
                    symbol=symbol,
                    start=str(start),
                    end=str(end),
                )
                download_liquidity = True
                if skip_existing and liquidity_covered:
                    result = SymbolDownloadResult(
                        symbol=symbol,
                        ok=True,
                        rows=0,
                        error="Skipped: local daily turnover and ST-status coverage verified",
                        freq="daily_liquidity",
                        adjust="-",
                    )
                    consume_result(result)
                    download_liquidity = False

                if kline_tasks or download_liquidity:
                    planned_symbol_tasks.append(
                        {
                            "symbol": symbol,
                            "kline_tasks": kline_tasks,
                            "download_liquidity": download_liquidity,
                        }
                    )

            add_job_log(
                jobs,
                job_id,
                f"Planning finished: skipped={skipped - planning_skipped_before}, "
                f"download_bundles={len(planned_symbol_tasks)}",
            )

            jobs.update_job(
                job_id,
                current="parallel_download",
                completed=completed,
                total=total,
                succeeded=succeeded,
                failed=failed,
                skipped=skipped,
                saved_rows=saved_rows,
                progress=progress_value(),
                message=f"Downloading {len(planned_symbol_tasks)} symbol bundles with {workers} workers",
            )

            catalog_refresh_queue: list[SymbolDownloadResult] = []
            cancelled = False
            executor = ProcessPoolExecutor(max_workers=workers)
            pending: dict[Any, str] = {}
            task_iter = iter(planned_symbol_tasks)

            def submit_next() -> bool:
                if jobs.is_cancel_requested(job_id):
                    return False
                try:
                    task = next(task_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _download_symbol_bundle_shard_for_app,
                    symbol=str(task["symbol"]),
                    db_path=db_path,
                    storage_root=storage_root,
                    start=str(start),
                    end=str(end),
                    kline_tasks=list(task["kline_tasks"]),
                    download_liquidity=bool(task["download_liquidity"]),
                    replace_symbol=False,
                    request_sleep=request_sleep,
                )
                pending[future] = str(task["symbol"])
                return True

            try:
                for _ in range(min(workers, len(planned_symbol_tasks))):
                    submit_next()
                while pending:
                    if jobs.is_cancel_requested(job_id):
                        cancelled = True
                        jobs.update_job(job_id, status="cancelled", message="Cancelling download workers")
                        for future in pending:
                            future.cancel()
                        break
                    done, _ = wait(set(pending), timeout=0.5, return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    for future in done:
                        symbol = pending.pop(future)
                        try:
                            bundle_results, bundle_logs = future.result()
                        except Exception as exc:
                            bundle_results = [
                                SymbolDownloadResult(
                                    symbol=symbol,
                                    ok=False,
                                    rows=0,
                                    error=str(exc),
                                    freq="bundle",
                                    adjust="-",
                                )
                            ]
                            bundle_logs = [("error", f"FAILED {symbol} bundle: {exc}")]
                        for level, message in bundle_logs:
                            add_job_log(jobs, job_id, message, level=level)
                        for result in bundle_results:
                            consume_result(result)
                            catalog_refresh_queue.append(result)
                        jobs.update_job(
                            job_id,
                            current=symbol,
                            completed=completed,
                            succeeded=succeeded,
                            failed=failed,
                            skipped=skipped,
                            saved_rows=saved_rows,
                            progress=progress_value(),
                            message=f"Finished symbol bundle {symbol}",
                        )
                        submit_next()
            finally:
                if cancelled:
                    _shutdown_process_pool_now(executor)
                else:
                    executor.shutdown(wait=True, cancel_futures=False)

            if cancelled:
                symbols = []
                jobs.update_job(job_id, status="cancelled", progress=progress_value(), message="Cancelled by user")
            else:
                for result in catalog_refresh_queue:
                    for catalog_level, catalog_message in _refresh_catalog_after_process_result(storage_root, result):
                        add_job_log(jobs, job_id, catalog_message, level=catalog_level)
                # The existing serial loop below is retained for duckdb and single-worker mode.
                symbols = []


        for freq in freqs:
            baostock_freq = STORAGE_TO_BAOSTOCK[freq]

            for adjustflag in adjustflags:
                storage_adjust = storage_adjust_for_freq(freq, adjustflag)

                for symbol in symbols:
                    if jobs.is_cancel_requested(job_id):
                        jobs.update_job(
                            job_id,
                            status="cancelled",
                            progress=progress_value(),
                            message="Cancelled by user",
                        )
                        break

                    label = f"{symbol} / {freq} / {storage_adjust}"
                    jobs.update_job(
                        job_id,
                        current=label,
                        completed=completed,
                        total=total,
                        progress=progress_value(),
                        message=f"Downloading {label}",
                    )

                    did_request = False
                    skip_ok = False
                    skip_reason = "skip disabled"
                    skip_manifest = None
                    planned_ranges = [(str(start), str(end))]

                    if skip_existing:
                        if storage_mode == "shard":
                            if coverage_snapshot is not None:
                                planned_ranges = plan_catalog_download_ranges_in_snapshot(
                                    coverage_snapshot,
                                    dataset=KLINE_DATASET,
                                    symbol=symbol,
                                    requested_start=str(start),
                                    requested_end=str(end),
                                    freq=freq,
                                    adjust=storage_adjust,
                                )
                            else:
                                planned_ranges = plan_catalog_download_ranges(
                                    storage_root,
                                    dataset=KLINE_DATASET,
                                    symbol=symbol,
                                    requested_start=str(start),
                                    requested_end=str(end),
                                    freq=freq,
                                    adjust=storage_adjust,
                                )
                            if not planned_ranges:
                                skip_ok = True
                                skip_reason = "catalog range coverage verified"
                        else:
                            skip_ok, skip_reason, skip_manifest = verify_completed_download_manifest(
                                db_path,
                                symbol=symbol,
                                requested_start=str(start),
                                requested_end=str(end),
                                freq=freq,
                                adjust=storage_adjust,
                            )
                            if not skip_ok:
                                planned_ranges = plan_kline_download_ranges(
                                    db_path,
                                    symbol=symbol,
                                    requested_start=str(start),
                                    requested_end=str(end),
                                    freq=freq,
                                    adjust=storage_adjust,
                                )
                                if not planned_ranges:
                                    skip_ok = True
                                    skip_reason = "local range coverage verified"

                    if skip_existing and skip_ok:
                        result = SymbolDownloadResult(
                            symbol=symbol,
                            ok=True,
                            rows=0,
                            error=f"Skipped: {skip_reason}",
                            freq=freq,
                            adjust=storage_adjust,
                        )
                        results.append(result)
                        add_job_log(jobs, job_id, f"SKIP {label}: {skip_reason}")

                        skipped += 1
                        succeeded += 1
                        completed += 1
                        completed_weight += download_progress_weight(freq)

                        jobs.update_job(
                            job_id,
                            current=label,
                            completed=completed,
                            succeeded=succeeded,
                            failed=failed,
                            skipped=skipped,
                            saved_rows=saved_rows,
                            progress=progress_value(),
                            message=f"Skipped {label}: manifest hash verified",
                        )
                    else:
                        did_request = True
                        request_reason = skip_reason if skip_existing else "skip disabled"
                        range_note = (
                            f"ranges={planned_ranges}"
                            if planned_ranges != [(str(start), str(end))]
                            else f"range={start}->{end}"
                        )
                        add_job_log(jobs, job_id, f"REQUEST {label}: {request_reason}; {range_note}; downloading from BaoStock")

                        if not client_logged_in:
                            jobs.update_job(job_id, message="Logging in to BaoStock")
                            add_job_log(jobs, job_id, "Login to BaoStock because at least one task requires remote download")
                            collector.client.login()
                            client_logged_in = True

                        # Important:
                        # When downloading symbol × freq × adjust, replace must only happen
                        # once per symbol. Otherwise later slices may delete earlier slices.
                        replace_this_symbol = bool(replace_symbol and symbol not in replaced_symbols)

                        if replace_this_symbol:
                            add_job_log(
                                jobs,
                                job_id,
                                f"REPLACE {symbol}: deleting existing rows only once before first downloaded slice",
                            )

                        chunk_results: list[SymbolDownloadResult] = []
                        for range_index, (chunk_start, chunk_end) in enumerate(planned_ranges):
                            chunk_result = collector.download_kline(
                                symbol,
                                start_date=chunk_start,
                                end_date=chunk_end,
                                frequency=baostock_freq,
                                adjustflag=adjustflag,
                                replace_symbol=replace_this_symbol and range_index == 0,
                            )
                            chunk_result.freq = chunk_result.freq or freq
                            chunk_result.adjust = chunk_result.adjust or storage_adjust
                            chunk_results.append(chunk_result)
                            if len(planned_ranges) > 1:
                                if chunk_result.ok:
                                    add_job_log(
                                        jobs,
                                        job_id,
                                        f"OK {label} chunk {chunk_start}->{chunk_end}: saved {chunk_result.rows} rows",
                                    )
                                else:
                                    add_job_log(
                                        jobs,
                                        job_id,
                                        f"FAILED {label} chunk {chunk_start}->{chunk_end}: {chunk_result.error}",
                                        level="error",
                                    )
                            if request_sleep > 0 and range_index < len(planned_ranges) - 1:
                                sleep(request_sleep)

                        if replace_this_symbol:
                            replaced_symbols.add(symbol)

                        if len(chunk_results) == 1:
                            result = chunk_results[0]
                        else:
                            failure_errors = [str(item.error) for item in chunk_results if not item.ok and item.error]
                            warning_errors = [str(item.error) for item in chunk_results if item.ok and _result_is_warning(item)]
                            messages = failure_errors or warning_errors
                            result = SymbolDownloadResult(
                                symbol=symbol,
                                ok=all(item.ok for item in chunk_results),
                                rows=sum(int(item.rows) for item in chunk_results),
                                error="; ".join(messages) if messages else None,
                                freq=freq,
                                adjust=storage_adjust,
                            )

                        results.append(result)
                        saved_rows += int(result.rows)

                        if storage_mode == "shard":
                            catalog_entry = get_market_catalog_record(
                                storage_root,
                                dataset=KLINE_DATASET,
                                symbol=symbol,
                                freq=freq,
                                adjust=storage_adjust,
                            )
                            if _result_is_warning(result):
                                succeeded += 1
                                add_job_log(jobs, job_id, f"WARN {label}: {result.error}", level="warning")
                            elif result.ok:
                                succeeded += 1
                                add_job_log(
                                    jobs,
                                    job_id,
                                    f"OK {label}: saved {result.rows} rows; "
                                    f"catalog rows={catalog_entry.get('rows') if catalog_entry else '-'}, "
                                    f"hash={str((catalog_entry or {}).get('data_hash') or '')[:12]}",
                                )
                            else:
                                failed += 1
                                add_job_log(jobs, job_id, f"FAILED {label}: {result.error}", level="error")
                        else:
                            manifest_status = "completed" if result.ok else "failed"
                            manifest = record_download_manifest(
                                db_path,
                                symbol=symbol,
                                freq=freq,
                                adjust=storage_adjust,
                                requested_start=str(start),
                                requested_end=str(end),
                                status=manifest_status,
                                source="baostock",
                                job_id=job_id,
                                report_path=str(
                                    report_path.relative_to(PROJECT_ROOT)
                                    if report_path.is_relative_to(PROJECT_ROOT)
                                    else report_path
                                ),
                                error=result.error if not result.ok else None,
                            )

                            if _result_is_warning(result):
                                succeeded += 1
                                add_job_log(jobs, job_id, f"WARN {label}: {result.error}", level="warning")
                            elif result.ok and manifest.get("status") == "completed":
                                succeeded += 1
                                add_job_log(
                                    jobs,
                                    job_id,
                                    f"OK {label}: saved {result.rows} rows; "
                                    f"manifest rows={manifest.get('rows')}, "
                                    f"hash={str(manifest.get('data_hash') or '')[:12]}",
                                )
                            elif result.ok:
                                failed += 1
                                result.ok = False
                                result.error = str(manifest.get("error") or "Manifest verification failed after save")
                                add_job_log(jobs, job_id, f"FAILED {label}: {result.error}", level="error")
                            else:
                                failed += 1
                                add_job_log(jobs, job_id, f"FAILED {label}: {result.error}", level="error")

                        completed += 1
                        completed_weight += download_progress_weight(freq)

                        jobs.update_job(
                            job_id,
                            current=label,
                            completed=completed,
                            succeeded=succeeded,
                            failed=failed,
                            skipped=skipped,
                            saved_rows=saved_rows,
                            progress=progress_value(),
                            message=f"Finished {label}: {result.rows} rows" if result.ok else f"Failed {label}: {result.error}",
                        )

                    if did_request and request_sleep > 0 and completed < total:
                        sleep(request_sleep)

                if jobs.is_cancel_requested(job_id):
                    break

            if jobs.is_cancel_requested(job_id):
                break

        if calendar_ok and not jobs.is_cancel_requested(job_id):
            for symbol in symbols:
                if jobs.is_cancel_requested(job_id):
                    jobs.update_job(
                        job_id,
                        status="cancelled",
                        progress=progress_value(),
                        message="Cancelled by user",
                    )
                    break

                label = f"{symbol} / daily_liquidity"
                jobs.update_job(
                    job_id,
                    current=label,
                    completed=completed,
                    total=total,
                    progress=progress_value(),
                    message=f"Checking {label}",
                )

                liquidity_covered = _is_daily_extension_bundle_covered(
                    storage_mode=storage_mode,
                    db_path=db_path,
                    storage_root=storage_root,
                    coverage_snapshot=coverage_snapshot,
                    symbol=symbol,
                    start=str(start),
                    end=str(end),
                )
                if skip_existing and liquidity_covered:
                    result = SymbolDownloadResult(
                        symbol=symbol,
                        ok=True,
                        rows=0,
                        error="Skipped: local daily turnover and ST-status coverage verified",
                        freq="daily_liquidity",
                        adjust="-",
                    )
                    results.append(result)
                    skipped += 1
                    succeeded += 1
                    completed += 1
                    completed_weight += DAILY_LIQUIDITY_PROGRESS_WEIGHT
                    add_job_log(jobs, job_id, f"SKIP {label}: local coverage verified")
                    jobs.update_job(
                        job_id,
                        current=label,
                        completed=completed,
                        succeeded=succeeded,
                        failed=failed,
                        skipped=skipped,
                        saved_rows=saved_rows,
                        progress=progress_value(),
                        message=f"Skipped {label}: local coverage verified",
                    )
                    continue

                add_job_log(jobs, job_id, f"REQUEST {label}: turnover + ST-status extension tables")
                if not client_logged_in:
                    jobs.update_job(job_id, message="Logging in to BaoStock")
                    add_job_log(jobs, job_id, "Login to BaoStock for daily turnover and ST-status extensions")
                    collector.client.login()
                    client_logged_in = True

                result = collector.download_daily_liquidity(
                    symbol,
                    start_date=str(start),
                    end_date=str(end),
                )
                results.append(result)
                saved_rows += int(result.rows)
                completed += 1
                completed_weight += DAILY_LIQUIDITY_PROGRESS_WEIGHT

                if _result_is_warning(result):
                    succeeded += 1
                    add_job_log(jobs, job_id, f"WARN {label}: {result.error}", level="warning")
                elif result.ok:
                    succeeded += 1
                    add_job_log(jobs, job_id, f"OK {label}: saved {result.rows} rows")
                else:
                    failed += 1
                    add_job_log(jobs, job_id, f"FAILED {label}: {result.error}", level="error")

                jobs.update_job(
                    job_id,
                    current=label,
                    completed=completed,
                    succeeded=succeeded,
                    failed=failed,
                    skipped=skipped,
                    saved_rows=saved_rows,
                    progress=progress_value(),
                    message=f"Finished {label}: {result.rows} rows" if result.ok else f"Failed {label}: {result.error}",
                )

                if request_sleep > 0 and completed < total:
                    sleep(request_sleep)

    finally:
        if client_logged_in:
            collector.client.logout()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "freq": r.freq,
                "adjust": r.adjust,
                "ok": r.ok,
                "rows": r.rows,
                "status": _download_report_status(r),
                "error": _download_report_error(r),
                "message": r.error if _result_is_skipped(r) else "",
            }
            for r in results
        ]
    )
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    result_payload = {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "saved_rows": saved_rows,
        "report_path": str(
            report_path.relative_to(PROJECT_ROOT)
            if report_path.is_relative_to(PROJECT_ROOT)
            else report_path
        ),
        "adjustflags": adjustflags,
        "storage_mode": storage_mode,
        "storage_root": str(
            storage_root.relative_to(PROJECT_ROOT)
            if storage_root.is_relative_to(PROJECT_ROOT)
            else storage_root
        ),
    }

    current_job = jobs.get_job(job_id)
    if current_job and current_job.get("status") == "cancelled":
        jobs.update_job(job_id, result=result_payload)

    return result_payload


def read_csv_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Read a generated CSV report and return a compact frontend-friendly payload."""
    raw_path = payload.get("path") or payload.get("report_path")
    if not raw_path:
        raise ValueError("Report path is required.")

    report_path = resolve_project_path(raw_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {raw_path}")

    df = pd.read_csv(report_path)
    records = dataframe_to_records(df)

    failed_rows: list[dict[str, Any]] = []
    if records:
        for row in records:
            ok_value = row.get("ok", row.get("success", row.get("available", True)))
            rows_value = row.get("rows", row.get("saved_rows"))
            error_value = str(row.get("error") or row.get("issues") or "").strip()
            status_value = str(row.get("status") or "").strip().lower()
            if error_value.startswith("Warning:") and status_value not in {"failed", "warning"}:
                status_value = "warning"
                row["status"] = "warning"

            ok_bool = True
            if isinstance(ok_value, bool):
                ok_bool = ok_value
            elif isinstance(ok_value, str):
                ok_bool = ok_value.strip().lower() not in {"false", "0", "failed", "fail", "no"}

            zero_rows = False
            try:
                zero_rows = rows_value is not None and int(float(rows_value)) == 0
            except Exception:
                zero_rows = False

            if status_value == "skipped":
                continue
            if status_value in {"failed", "warning"} or (not ok_bool) or bool(error_value) or zero_rows:
                failed_rows.append(row)

    failed_count = sum(1 for row in failed_rows if str(row.get("status") or "").strip().lower() == "failed")
    warning_count = sum(1 for row in failed_rows if str(row.get("status") or "").strip().lower() == "warning")

    return {
        "path": str(report_path.relative_to(PROJECT_ROOT) if report_path.is_relative_to(PROJECT_ROOT) else report_path),
        "total": len(records),
        "failed": len(failed_rows),
        "failed_count": failed_count,
        "warnings": warning_count,
        "issue_rows": len(failed_rows),
        "rows": records[:500],
        "failed_rows": failed_rows[:200],
    }
