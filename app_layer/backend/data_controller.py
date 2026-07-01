from __future__ import annotations

from pathlib import Path
from typing import Any

from data_layer.inventory.availability import build_availability_report
from data_layer.inventory.universe_builder import build_available_universe_from_file
from data_layer.storage.duckdb_storage import (
    delete_many_symbols_from_duckdb,
    delete_symbol_extensions_from_duckdb,
    delete_symbol_from_duckdb,
    get_kline_inventory,
    load_kline_from_duckdb,
    load_kline_page_from_duckdb,
    load_stock_liquidity_daily_from_duckdb,
    load_stock_status_daily_from_duckdb,
)
from data_layer.storage.partitioned_storage import (
    catalog_path as market_catalog_path,
    delete_many_symbol_market_slices,
    has_market_shard_storage,
    resolve_market_data_root,
)
from data_layer.storage.portable_bundle import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_FEATURE_OUTPUT_DIR,
    export_portable_bundle,
    import_portable_bundle,
    inspect_portable_bundle,
)
from feature_layer.builders.materialize import (
    DERIVED_TARGETS_BY_BASE,
    DEFAULT_DERIVED_TARGETS,
    materialize_derived_bars,
)

from app_layer.backend.json_utils import dataframe_to_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = "runtime_layer/data"
DEFAULT_REPORT_DIR = "runtime_layer/reports"
DEFAULT_AVAILABLE_UNIVERSE_PATH = "config/universe/available_universe.txt"


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
        symbol = item.replace("\ufeff", "").strip().upper()
        if symbol and not symbol.startswith("#") and symbol not in result:
            result.append(symbol)
    return result


def _read_symbols_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return normalize_symbol_list(path.read_text(encoding="utf-8-sig"))


def _resolve_symbols(payload: dict[str, Any]) -> list[str]:
    symbols = normalize_symbol_list(payload.get("symbols") or payload.get("manual_symbols"))
    symbols_file = payload.get("symbols_file")
    if symbols_file:
        for symbol in _read_symbols_from_file(resolve_project_path(symbols_file)):
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _relative_or_absolute(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


def inventory_payload(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    db_file = resolve_project_path(db_path)
    inventory = get_kline_inventory(db_file)
    records = dataframe_to_records(inventory)

    if not inventory.empty and "rows" in inventory and "symbol" in inventory:
        symbol_total_rows = inventory.groupby("symbol")["rows"].sum().to_dict()
        daily_rows: dict[str, int] = {}
        if "freq" in inventory:
            daily_rows = inventory[inventory["freq"].astype(str) == "daily"].groupby("symbol")["rows"].sum().to_dict()
        for record in records:
            symbol = record.get("symbol")
            record["symbol_total_rows"] = int(symbol_total_rows.get(symbol, 0))
            record["symbol_daily_rows"] = int(daily_rows.get(symbol, 0))

    total_rows = int(inventory["rows"].sum()) if not inventory.empty and "rows" in inventory else 0
    start_date = None
    end_date = None
    if not inventory.empty:
        start_date = str(inventory["start_datetime"].min())[:10]
        end_date = str(inventory["end_datetime"].max())[:10]

    freq_summary: dict[str, dict[str, int]] = {}
    if not inventory.empty:
        for freq, group in inventory.groupby("freq"):
            freq_summary[str(freq or "-")] = {
                "symbols": int(group["symbol"].nunique()),
                "rows": int(group["rows"].sum()),
            }

    shard_root = resolve_market_data_root(db_file)
    shard_exists = has_market_shard_storage(shard_root)
    catalog_file = market_catalog_path(shard_root)

    return {
        "db_path": str(Path(db_path)),
        "db_exists": db_file.exists() or shard_exists,
        "storage_mode": "shard" if shard_exists else "duckdb",
        "storage_root": str(shard_root if shard_exists else db_file.parent),
        "catalog_path": str(catalog_file) if shard_exists else None,
        "catalog_exists": catalog_file.exists() if shard_exists else False,
        "table": "market_shards" if shard_exists else "kline_bars",
        "total_symbols": int(inventory["symbol"].nunique()) if not inventory.empty and "symbol" in inventory else 0,
        "total_rows": total_rows,
        "start_date": start_date,
        "end_date": end_date,
        "freq_summary": freq_summary,
        "symbols": records,
    }


def check_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    symbols = _resolve_symbols(payload)
    output_raw = payload.get("output") or payload.get("report_path")
    output_path = resolve_project_path(output_raw) if output_raw else None
    min_rows = int(payload.get("min_rows") or 200)
    required_start = payload.get("required_start") or None
    required_end = payload.get("required_end") or None

    report = build_availability_report(
        db_path,
        symbols=symbols or None,
        min_rows=min_rows,
        required_start=required_start,
        required_end=required_end,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False, encoding="utf-8-sig")

    available = int(report["available"].sum()) if not report.empty else 0
    total = int(len(report))
    return {
        "total": total,
        "available": available,
        "unavailable": total - available,
        "report_path": _relative_or_absolute(output_path),
        "rows": dataframe_to_records(report.head(200)),
    }


def build_universe(payload: dict[str, Any]) -> dict[str, Any]:
    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    candidates = payload.get("candidates") or payload.get("candidate_path") or payload.get("symbols_file")
    output = payload.get("output") or DEFAULT_AVAILABLE_UNIVERSE_PATH
    report_path = payload.get("report") or payload.get("report_path")
    min_rows = int(payload.get("min_rows") or 200)
    required_start = payload.get("required_start") or None
    required_end = payload.get("required_end") or None

    if not candidates:
        raise ValueError("Candidate symbol file is required.")

    resolved_report_path = resolve_project_path(report_path) if report_path else None
    report = build_available_universe_from_file(
        db_path=db_path,
        candidate_path=resolve_project_path(candidates),
        output_path=resolve_project_path(output),
        report_path=resolved_report_path,
        min_rows=min_rows,
        required_start=required_start,
        required_end=required_end,
    )

    available = int(report["available"].sum()) if not report.empty else 0
    total = int(len(report))
    return {
        "total": total,
        "available": available,
        "unavailable": total - available,
        "output_path": output,
        "report_path": _relative_or_absolute(resolved_report_path),
        "rows": dataframe_to_records(report.head(200)),
    }


def materialize_bars(
    payload: dict[str, Any],
    *,
    progress_callback: Any = None,
    cancel_check: Any = None,
) -> dict[str, Any]:
    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    output_raw = payload.get("output") or payload.get("report_path")
    output_path = resolve_project_path(output_raw) if output_raw else None
    targets = _normalize_csv_values(payload.get("targets")) or None
    symbols = _resolve_symbols(payload)
    base_freq = str(payload.get("base_freq") or "5min")
    adjust = str(payload.get("adjust") or "none")
    chunk_size = int(payload.get("chunk_size") or 5)
    start = payload.get("start") or None
    end = payload.get("end") or None

    summary = materialize_derived_bars(
        db_path,
        symbols=symbols or None,
        base_freq=base_freq,
        adjust=adjust,
        targets=targets,
        start=start,
        end=end,
        chunk_size=chunk_size,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    run_stats = dict(summary.attrs)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    detail_rows = summary.loc[summary["symbol"].ne("__TOTAL__")] if not summary.empty else summary
    ok_rows = detail_rows.loc[detail_rows["status"].eq("ok")] if not detail_rows.empty else detail_rows
    skipped_rows = detail_rows.loc[detail_rows["status"].eq("skipped")] if not detail_rows.empty else detail_rows
    failed_rows = detail_rows.loc[detail_rows["status"].eq("failed")] if not detail_rows.empty else detail_rows
    total_saved = int(run_stats.get("saved_rows") or 0)
    return {
        "total": int(len(summary)),
        "ok": int(len(ok_rows)),
        "skipped": int(len(skipped_rows)),
        "failed": int(len(failed_rows)),
        "saved_rows": total_saved,
        "cancelled": bool(run_stats.get("cancelled")),
        "completed_units": int(run_stats.get("completed_units") or 0),
        "total_units": int(run_stats.get("total_units") or 0),
        "succeeded_units": int(run_stats.get("succeeded_units") or 0),
        "failed_units": int(run_stats.get("failed_units") or 0),
        "report_path": _relative_or_absolute(output_path),
        "rows": dataframe_to_records(summary.head(300)),
    }


def start_materialize_bars_job(payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    active = [
        job for job in jobs.list_jobs()
        if job.get("type") == "data_materialize"
        and job.get("status") in {"queued", "running"}
    ]
    if active:
        raise ValueError(f"A derived-bar job is already active: {active[-1]['job_id']}")
    job_id = jobs.create_job("data_materialize", title="Build derived bars")
    jobs.update_job(job_id, total=0, message="Queued derived bar build")
    jobs.submit(job_id, _run_materialize_bars_job, job_id, payload, jobs)
    return {"job_id": job_id, "status": "queued"}


def _run_materialize_bars_job(job_id: str, payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    output_raw = payload.get("output") or payload.get("report_path")
    if not output_raw:
        payload = {
            **payload,
            "output": f"{DEFAULT_REPORT_DIR}/materialize_{job_id}.csv",
        }

    if jobs.is_cancel_requested(job_id):
        jobs.update_job(job_id, status="cancelled", message="Cancelled before start")
        return {"cancelled": True, "saved_rows": 0}

    jobs.update_job(
        job_id,
        status="running",
        current="materialize",
        completed=0,
        total=0,
        progress=0.0,
        message="Preparing derived-bar aggregation",
    )
    add_log = getattr(jobs, "add_log", None)
    if callable(add_log):
        add_log(
            job_id,
            (
                f"CONFIG db={payload.get('db_path') or DEFAULT_DB_PATH}, "
                f"base={payload.get('base_freq') or '5min'}, adjust={payload.get('adjust') or 'none'}, "
                f"targets={payload.get('targets') or ','.join(DERIVED_TARGETS_BY_BASE.get(str(payload.get('base_freq') or '5min'), DEFAULT_DERIVED_TARGETS))}"
            ),
        )

    last_logged_completed = -1

    def on_progress(update: dict[str, Any]) -> None:
        nonlocal last_logged_completed
        jobs.update_job(job_id, **update)
        completed = int(update.get("completed") or 0)
        if callable(add_log) and completed > last_logged_completed:
            add_log(job_id, str(update.get("message") or update.get("current") or "Progress updated"))
            last_logged_completed = completed

    result = materialize_bars(
        payload,
        progress_callback=on_progress,
        cancel_check=lambda: jobs.is_cancel_requested(job_id),
    )
    if result.get("cancelled") or jobs.is_cancel_requested(job_id):
        jobs.update_job(
            job_id,
            status="cancelled",
            completed=int(result.get("completed_units") or 0),
            total=int(result.get("total_units") or 0),
            succeeded=int(result.get("succeeded_units") or 0),
            failed=int(result.get("failed_units") or 0),
            saved_rows=int(result.get("saved_rows") or 0),
            message="Cancelled safely at a target transaction boundary",
        )
        if callable(add_log):
            add_log(job_id, f"CANCELLED materialize: saved {result.get('saved_rows', 0)} rows")
        return result
    jobs.update_job(
        job_id,
        completed=int(result.get("total_units") or 0),
        total=int(result.get("total_units") or 0),
        succeeded=int(result.get("succeeded_units") or 0),
        failed=int(result.get("failed_units") or 0),
        saved_rows=int(result.get("saved_rows") or 0),
        progress=1.0,
        message="Derived bars built",
    )
    if callable(add_log):
        add_log(job_id, f"OK materialize: saved {result.get('saved_rows', 0)} rows")
    return result


def read_symbol_file(payload: dict[str, Any]) -> dict[str, Any]:
    raw_path = payload.get("path") or payload.get("symbols_file")
    if not raw_path:
        raise ValueError("Symbol file path is required.")
    file_path = resolve_project_path(raw_path)
    symbols = _read_symbols_from_file(file_path)
    return {
        "path": _relative_or_absolute(file_path),
        "exists": file_path.exists(),
        "count": len(symbols),
        "symbols": symbols,
        "text": "\n".join(symbols),
    }


def write_symbol_file(payload: dict[str, Any]) -> dict[str, Any]:
    raw_path = payload.get("path") or payload.get("symbols_file")
    if not raw_path:
        raise ValueError("Symbol file path is required.")
    symbols = normalize_symbol_list(payload.get("symbols") or payload.get("text"))
    file_path = resolve_project_path(raw_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
    return {
        "path": _relative_or_absolute(file_path),
        "count": len(symbols),
        "symbols": symbols,
    }


def symbol_detail(payload: dict[str, Any]) -> dict[str, Any]:
    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Symbol is required.")
    freq = payload.get("freq") or None
    adjust = payload.get("adjust") or None
    offset = int(payload.get("offset") or 0)
    limit = int(payload.get("limit") or 100)

    df, total = load_kline_page_from_duckdb(
        db_path,
        symbol=symbol,
        freq=freq,
        adjust=adjust,
        offset=offset,
        limit=limit,
    )
    coverage_df = load_kline_from_duckdb(db_path, symbol=symbol, freq=freq, adjust=adjust)
    if coverage_df.empty:
        return {
            "symbol": symbol,
            "freq": freq,
            "adjust": adjust,
            "rows": 0,
            "offset": offset,
            "limit": limit,
            "items": [],
            "has_more": False,
        }

    display_df = df.drop(columns=["source"], errors="ignore")
    if str(freq or "").lower() == "daily" and not display_df.empty:
        display_df = display_df.copy()
        display_df["turn"] = None
        display_df["isST"] = None
        liquidity_df = load_stock_liquidity_daily_from_duckdb(
            db_path,
            symbol=symbol,
            start=str(coverage_df["datetime"].min())[:10],
            end=str(coverage_df["datetime"].max())[:10],
        )
        if not liquidity_df.empty:
            turn_by_date = dict(
                zip(
                    liquidity_df["date"].astype(str).str.slice(0, 10),
                    liquidity_df["turn"],
                )
            )
            display_df["turn"] = display_df["datetime"].astype(str).str.slice(0, 10).map(turn_by_date)
        status_df = load_stock_status_daily_from_duckdb(
            db_path,
            symbol=symbol,
            start=str(coverage_df["datetime"].min())[:10],
            end=str(coverage_df["datetime"].max())[:10],
        )
        if not status_df.empty:
            status_by_date = dict(
                zip(status_df["date"].astype(str).str.slice(0, 10), status_df["is_st"])
            )
            display_df["isST"] = display_df["datetime"].astype(str).str.slice(0, 10).map(status_by_date)
    elif "turn" in display_df.columns:
        display_df = display_df.drop(columns=["turn"], errors="ignore")

    return {
        "symbol": symbol,
        "freq": freq,
        "adjust": adjust,
        "rows": int(total),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(df) < total,
        "start_datetime": str(coverage_df["datetime"].min())[:19],
        "end_datetime": str(coverage_df["datetime"].max())[:19],
        "items": dataframe_to_records(display_df),
    }


def delete_symbol_data(payload: dict[str, Any]) -> dict[str, Any]:
    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Symbol is required.")
    freq = payload.get("freq") or None
    adjust = payload.get("adjust") or None

    shard_root = resolve_market_data_root(db_path)
    if has_market_shard_storage(shard_root):
        result = delete_many_symbol_market_slices(
            shard_root,
            [{"symbol": symbol, "freq": freq, "adjust": adjust}],
            include_full_symbol_extensions=True,
        )
        return {
            "symbol": symbol,
            "freq": freq,
            "adjust": adjust,
            "deleted_rows": result.deleted_rows,
            "deleted_extension_rows": result.deleted_extension_rows,
            "deleted_files": result.deleted_files,
        }

    deleted_rows = delete_symbol_from_duckdb(db_path, symbol, freq=freq, adjust=adjust)
    deleted_extensions = 0
    if freq is None and adjust is None:
        deleted_extensions = delete_symbol_extensions_from_duckdb(db_path, symbol)
    return {
        "symbol": symbol,
        "freq": freq,
        "adjust": adjust,
        "deleted_rows": deleted_rows,
        "deleted_extension_rows": deleted_extensions,
    }


def delete_symbol_data_batch(payload: dict[str, Any]) -> dict[str, Any]:
    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    items = payload.get("items") or payload.get("symbols") or []
    if not isinstance(items, list) or not items:
        raise ValueError("No inventory rows selected for deletion.")

    shard_root = resolve_market_data_root(db_path)
    if has_market_shard_storage(shard_root):
        result = delete_many_symbol_market_slices(
            shard_root,
            items,
            include_full_symbol_extensions=True,
        )
        return {
            "items": items,
            "count": len(items),
            "deleted_rows": result.deleted_rows,
            "deleted_extension_rows": result.deleted_extension_rows,
            "deleted_files": result.deleted_files,
            "matched_catalog_rows": result.matched_catalog_rows,
        }

    deleted_rows = delete_many_symbols_from_duckdb(db_path, items)
    deleted_extensions = 0
    for item in items:
        if item.get("freq") in (None, "", "-") and item.get("adjust") in (None, "", "-"):
            deleted_extensions += delete_symbol_extensions_from_duckdb(
                db_path, str(item.get("symbol") or "")
            )
    return {
        "items": items,
        "count": len(items),
        "deleted_rows": deleted_rows,
        "deleted_extension_rows": deleted_extensions,
    }




def inspect_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    bundle_path = payload.get("bundle_path") or DEFAULT_BUNDLE_PATH
    return inspect_portable_bundle(project_root=PROJECT_ROOT, bundle_path=bundle_path)


def start_export_bundle_job(payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    active = [
        job for job in jobs.list_jobs()
        if job.get("type") == "portable_bundle_export"
        and job.get("status") in {"queued", "running"}
    ]
    if active:
        raise ValueError(f"A bundle export job is already active: {active[-1]['job_id']}")
    job_id = jobs.create_job("portable_bundle_export", title="Export portable bundle")
    jobs.update_job(job_id, total=0, message="Queued portable bundle export")
    jobs.submit(job_id, _run_export_bundle_job, job_id, payload, jobs)
    return {"job_id": job_id, "status": "queued"}


def start_import_bundle_job(payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    active = [
        job for job in jobs.list_jobs()
        if job.get("type") == "portable_bundle_import"
        and job.get("status") in {"queued", "running"}
    ]
    if active:
        raise ValueError(f"A bundle import job is already active: {active[-1]['job_id']}")
    job_id = jobs.create_job("portable_bundle_import", title="Import portable bundle")
    jobs.update_job(job_id, total=0, message="Queued portable bundle import")
    jobs.submit(job_id, _run_import_bundle_job, job_id, payload, jobs)
    return {"job_id": job_id, "status": "queued"}


def _bundle_payload_options(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_path": payload.get("bundle_path") or DEFAULT_BUNDLE_PATH,
        "db_path": payload.get("db_path") or DEFAULT_DB_PATH,
        "feature_output_dir": payload.get("feature_output_dir") or DEFAULT_FEATURE_OUTPUT_DIR,
        "include_data": bool(payload.get("include_data", True)),
        "include_feature": bool(payload.get("include_feature", True)),
    }


def _run_export_bundle_job(job_id: str, payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    options = _bundle_payload_options(payload)
    if jobs.is_cancel_requested(job_id):
        jobs.update_job(job_id, status="cancelled", message="Cancelled before start")
        return {"cancelled": True}

    jobs.update_job(
        job_id,
        status="running",
        current="bundle export",
        completed=0,
        total=0,
        progress=0.0,
        message="Preparing portable bundle export",
    )
    add_log = getattr(jobs, "add_log", None)
    if callable(add_log):
        add_log(
            job_id,
            (
                f"CONFIG bundle={options['bundle_path']}, db={options['db_path']}, "
                f"feature_output={options['feature_output_dir']}, "
                f"include_data={options['include_data']}, include_feature={options['include_feature']}"
            ),
        )

    last_completed = -1

    def on_progress(update: dict[str, Any]) -> None:
        nonlocal last_completed
        jobs.update_job(job_id, **update)
        completed = int(update.get("completed") or 0)
        if callable(add_log) and completed > last_completed:
            add_log(job_id, str(update.get("message") or update.get("current") or "Progress updated"))
            last_completed = completed

    result = export_portable_bundle(
        project_root=PROJECT_ROOT,
        bundle_path=options["bundle_path"],
        db_path=options["db_path"],
        feature_output_dir=options["feature_output_dir"],
        include_data=options["include_data"],
        include_feature=options["include_feature"],
        overwrite=bool(payload.get("overwrite", True)),
        progress_callback=on_progress,
        cancel_check=lambda: jobs.is_cancel_requested(job_id),
    )
    if jobs.is_cancel_requested(job_id):
        jobs.update_job(job_id, status="cancelled", message="Cancelled")
        return {**result, "cancelled": True}
    jobs.update_job(
        job_id,
        completed=int(result.get("file_count") or 0),
        total=int(result.get("file_count") or 0),
        progress=1.0,
        message="Portable bundle exported",
    )
    if callable(add_log):
        add_log(job_id, f"OK export: {result.get('file_count', 0)} files -> {result.get('bundle_path')}")
    return result


def _run_import_bundle_job(job_id: str, payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    bundle_path = payload.get("bundle_path") or DEFAULT_BUNDLE_PATH
    include_data = bool(payload.get("include_data", True))
    include_feature = bool(payload.get("include_feature", True))
    replace_existing = bool(payload.get("replace_existing", True))

    if jobs.is_cancel_requested(job_id):
        jobs.update_job(job_id, status="cancelled", message="Cancelled before start")
        return {"cancelled": True}

    jobs.update_job(
        job_id,
        status="running",
        current="bundle import",
        completed=0,
        total=0,
        progress=0.0,
        message="Preparing portable bundle import",
    )
    add_log = getattr(jobs, "add_log", None)
    if callable(add_log):
        add_log(
            job_id,
            (
                f"CONFIG bundle={bundle_path}, include_data={include_data}, "
                f"include_feature={include_feature}, replace_existing={replace_existing}"
            ),
        )

    last_completed = -1

    def on_progress(update: dict[str, Any]) -> None:
        nonlocal last_completed
        jobs.update_job(job_id, **update)
        completed = int(update.get("completed") or 0)
        if callable(add_log) and completed > last_completed:
            add_log(job_id, str(update.get("message") or update.get("current") or "Progress updated"))
            last_completed = completed

    result = import_portable_bundle(
        project_root=PROJECT_ROOT,
        bundle_path=bundle_path,
        include_data=include_data,
        include_feature=include_feature,
        replace_existing=replace_existing,
        progress_callback=on_progress,
        cancel_check=lambda: jobs.is_cancel_requested(job_id),
    )
    if jobs.is_cancel_requested(job_id):
        jobs.update_job(job_id, status="cancelled", message="Cancelled")
        return {**result, "cancelled": True}
    jobs.update_job(
        job_id,
        completed=int(result.get("restored_files") or 0),
        total=int(result.get("restored_files") or 0),
        progress=1.0,
        message="Portable bundle imported",
    )
    if callable(add_log):
        add_log(job_id, f"OK import: restored {result.get('restored_files', 0)} files from {result.get('bundle_path')}")
    return result

def _normalize_csv_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").replace("\n", ",").split(",")
    else:
        parts = [str(item) for item in value]
    result: list[str] = []
    for item in parts:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result
