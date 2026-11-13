"""Fixed-universe US daily downloads, separate from BaoStock market shards.

The common CLI and download page share this service. Existing research parsing,
calendar checks and immutable successful-cache hashes remain authoritative.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Callable
from uuid import uuid4

import pandas as pd

from research_layer import elder_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADJUSTMENT = "vendor_OHLC * (adjclose / vendor_close)"
ProgressCallback = Callable[[dict], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _number(value, name: str, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its permitted range.")
    return float(value)


def _validated_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Yahoo configuration must be a JSON object.")
    result = dict(config)
    expected = {"market": "US", "source": elder_data.SOURCE, "interval": "1d",
                "volume_unit": "shares"}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Yahoo {key} must be {value!r}; BaoStock conventions are not interchangeable.")
    adjustment = str(result.get("price_adjustment", ""))
    if "".join(adjustment.split()).replace("(", "").replace(")", "") != "vendor_OHLC*adjclose/vendor_close":
        raise ValueError(f"Yahoo price_adjustment must be {ADJUSTMENT!r}; BaoStock conventions are not interchangeable.")
    result["price_adjustment"] = ADJUSTMENT
    if result.get("calendar", "XNYS") != "XNYS":
        raise ValueError("The US daily adapter requires the XNYS calendar.")
    result["calendar"] = "XNYS"
    for key in ("start", "end"):
        try:
            if date.fromisoformat(result[key]).isoformat() != result[key]:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{key} must be a date in YYYY-MM-DD format.") from None
    if result["start"] > result["end"]:
        raise ValueError("start must not be after end.")
    for key in ("universe_file", "data_root", "report_root", "preserve_universe_file", "reuse_data_root"):
        if key not in result:
            if key in ("universe_file", "data_root", "report_root"):
                raise ValueError(f"Missing Yahoo configuration field: {key}")
            continue
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} must be a nonempty path.")
        result[key] = str(_path(result[key]))
    count = result.get("fixed_universe_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("fixed_universe_count must be a positive integer.")
    symbols = elder_data.load_symbols(Path(result["universe_file"]), count)
    if "preserve_universe_file" in result:
        preserved = elder_data.load_symbols(Path(result["preserve_universe_file"]), 100)
        if symbols[:100] != preserved:
            raise ValueError("The frozen universe must retain the original 100 symbols in order.")
    root = Path(result["data_root"])
    ashare_root = (PROJECT_ROOT / "runtime_layer/data").resolve()
    if root == PROJECT_ROOT or root == ashare_root or ashare_root in root.parents or root in ashare_root.parents:
        raise ValueError("Yahoo data_root must be independent of the BaoStock runtime_layer/data tree.")
    if any((root / item).exists() for item in ("market_catalog.duckdb", "market_parts", "market_data.duckdb")):
        raise ValueError("Refusing to write Yahoo files into an existing A-share market store.")
    workers = result.get("workers", 1)
    attempts = result.get("maximum_attempts", 1)
    for value, key, maximum in ((workers, "workers", 16), (attempts, "maximum_attempts", 5)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{key} must be an integer between 1 and {maximum}.")
    result["workers"], result["maximum_attempts"] = workers, attempts
    result["network_timeout_seconds"] = _number(result.get("network_timeout_seconds", 20), "network_timeout_seconds", minimum=0.001, maximum=120)
    result["outer_timeout_seconds"] = _number(result.get("outer_timeout_seconds", 2400), "outer_timeout_seconds", minimum=0.001)
    result["sleep_seconds_per_worker"] = _number(result.get("sleep_seconds_per_worker", result.get("sleep_seconds", 0.5)), "sleep_seconds", minimum=0)
    return result


def load_yahoo_config(config_path: str | Path) -> dict:
    """Load without network access; relative configuration paths use repo root."""
    return _validated_config(json.loads(_path(config_path).read_text(encoding="utf-8")))


def _verify_provider_identity(symbol: str, vendor: dict) -> None:
    meta = vendor["provider_meta"]
    if meta.get("symbol") != symbol:
        raise ValueError(f"Provider returned {meta.get('symbol')!r} instead of {symbol!r}.")
    if meta.get("dataGranularity", "1d") != "1d":
        raise ValueError("Provider returned a non-daily interval.")


def _restore_checked_symbol(symbol: str, config: dict, calendar: pd.DataFrame, timeout: float) -> dict:
    """Keep research cache/archive conventions, validating identity before success.

    Parsing and calendar numerics are the existing research functions. We keep
    this persistence boundary here so stricter new entry-point guarantees do not
    alter the source identities of already-frozen research experiments.
    """
    root = Path(config["data_root"])
    raw, daily, metadata = (root / directory / f"{symbol}{suffix}"
                            for directory, suffix in (("raw", ".json"), ("daily", ".csv"), ("metadata", ".json")))
    if metadata.exists():
        prior = json.loads(metadata.read_text(encoding="utf-8"))
        if prior.get("status") == "success":
            # The caller re-reads and audits this cache before counting success.
            # Never repair an existing successful version by overwriting bytes.
            return {**prior, "reused_validated_cache": True}
    for path in (raw, daily, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
    body, request = elder_data.fetch_chart(symbol, config["start"], config["end"], timeout)
    if raw.exists():
        old_hash = elder_data.sha256(raw)
        archive = root / "failed_attempts" / symbol
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f"{old_hash}.json").write_bytes(raw.read_bytes())
        if metadata.exists():
            (archive / f"{old_hash}.metadata.json").write_bytes(metadata.read_bytes())
    raw.write_bytes(body)
    try:
        if any(request.get(key) != value for key, value in {"source": elder_data.SOURCE, "symbol": symbol,
                                                           "start": config["start"], "end": config["end"]}.items()):
            raise ValueError("Yahoo request identity mismatch.")
        frame, vendor = elder_data.parse_chart(body, symbol, config["start"], config["end"])
        _verify_provider_identity(symbol, vendor)
        quality = elder_data.audit_frame(frame, calendar, config["start"], config["end"])
        temporary = daily.with_name(daily.name + ".partial")
        frame.to_csv(temporary, index=False)
        temporary.replace(daily)
        record = {"status": "success", **request, **vendor, "quality": quality,
                  "raw_file": str(raw), "daily_file": str(daily),
                  "raw_sha256": elder_data.sha256(raw), "daily_sha256": elder_data.sha256(daily)}
    except Exception as exc:
        elder_data.save_json(metadata, {"status": "failed", **request, "raw_file": str(raw),
                                       "raw_sha256": elder_data.sha256(raw), "error_type": type(exc).__name__, "error": str(exc)})
        raise
    elder_data.save_json(metadata, record)
    return record


def _verify_symbol(symbol: str, config: dict, calendar: pd.DataFrame, record: dict) -> dict:
    root = Path(config["data_root"])
    raw = root / "raw" / f"{symbol}.json"
    daily = root / "daily" / f"{symbol}.csv"
    if record.get("status") != "success" or record.get("source") != elder_data.SOURCE:
        raise ValueError("Missing successful Yahoo source metadata.")
    if record.get("symbol") != symbol or record.get("start") != config["start"] or record.get("end") != config["end"]:
        raise ValueError("Snapshot symbol/date identity mismatch.")
    if elder_data.sha256(raw) != record["raw_sha256"] or elder_data.sha256(daily) != record["daily_sha256"]:
        raise ValueError("Raw/CSV hash mismatch; refusing to overwrite a successful snapshot.")
    rebuilt, vendor = elder_data.parse_chart(raw.read_bytes(), symbol, config["start"], config["end"])
    _verify_provider_identity(symbol, vendor)
    frame = pd.read_csv(daily)
    pd.testing.assert_frame_equal(frame, rebuilt, check_dtype=False, rtol=1e-12, atol=1e-12)
    return elder_data.audit_frame(frame, calendar, config["start"], config["end"])


def audit_yahoo(config: dict, config_path: str | Path) -> dict:
    """Audit the frozen US snapshot without network access or rewriting its data."""
    config = _validated_config(config)
    config_path = _path(config_path)
    config_hash = elder_data.sha256(config_path)
    symbols = elder_data.load_symbols(Path(config["universe_file"]), config["fixed_universe_count"])
    root = Path(config["data_root"])
    rows, global_error = [], None
    hashes = {}
    try:
        calendar = elder_data.make_calendar(config["start"], config["end"])
        pd.testing.assert_frame_equal(pd.read_csv(root / "calendar.csv", keep_default_na=False), calendar, check_dtype=False)
        hashes["calendar.csv"] = elder_data.sha256(root / "calendar.csv")
    except Exception as exc:
        global_error = exc
    for symbol in symbols:
        try:
            if global_error is not None:
                raise global_error
            metadata = root / "metadata" / f"{symbol}.json"
            record = json.loads(metadata.read_text(encoding="utf-8"))
            quality = _verify_symbol(symbol, config, calendar, record)
            rows.append({"symbol": symbol, "status": "success", "error": None, **quality})
            hashes[f"raw/{symbol}.json"] = record["raw_sha256"]
            hashes[f"daily/{symbol}.csv"] = record["daily_sha256"]
            hashes[f"metadata/{symbol}.json"] = elder_data.sha256(metadata)
        except Exception as exc:
            rows.append({"symbol": symbol, "status": "failed", "rows": 0, "error_type": type(exc).__name__, "error": str(exc)})
    succeeded = sum(row["status"] == "success" for row in rows)
    report_path = Path(config["report_root"]) / ("yahoo_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8] + ".json")
    report = {"status": "completed" if succeeded == len(symbols) else "partial" if succeeded else "failed",
              "source": elder_data.SOURCE, "market": "US", "calendar": "XNYS", "price_adjustment": ADJUSTMENT,
              "volume_unit": "shares", "requested_start": config["start"], "requested_end": config["end"],
              "audited_at_utc": _now(), "successful_stocks": succeeded, "failed_stocks": len(symbols) - succeeded,
              "fixed_universe_count": len(symbols), "rows": sum(row["rows"] for row in rows), "outcomes": rows,
              "file_hashes": hashes, "config_sha256": config_hash,
              "universe_sha256": elder_data.sha256(Path(config["universe_file"])),
              "all_hashes_and_source_reconstruction_verified": succeeded == len(symbols), "report_path": str(report_path)}
    elder_data.save_json(report_path, report)
    return report


def _download_symbol(symbol: str, config: dict, calendar: pd.DataFrame, deadline: float) -> dict:
    root = Path(config["data_root"])
    attempts = []
    for attempt in range(1, config["maximum_attempts"] + 1):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Outer download deadline reached before the next request.")
            metadata = root / "metadata" / f"{symbol}.json"
            if metadata.exists():
                prior = json.loads(metadata.read_text(encoding="utf-8"))
                if prior.get("status") == "success" and (prior.get("start"), prior.get("end")) != (config["start"], config["end"]):
                    raise ValueError("Existing successful snapshot has different dates; choose a new independent data_root.")
            record = _restore_checked_symbol(symbol, config, calendar, min(config["network_timeout_seconds"], remaining))
            quality = _verify_symbol(symbol, config, calendar, record)
            return {"symbol": symbol, "status": "success", "rows": quality["rows"], "error": None,
                    "actual_first_date": quality["first_date"], "actual_last_date": quality["last_date"],
                    "raw_sha256": record["raw_sha256"], "daily_sha256": record["daily_sha256"],
                    "metadata_sha256": elder_data.sha256(metadata), "quality": quality,
                    "download_started_utc": record.get("download_started_utc"),
                    "download_finished_utc": record.get("download_finished_utc"),
                    "reused_validated_cache": record.get("reused_validated_cache", False), "prior_failed_attempts": attempts}
        except Exception as exc:
            attempts.append({"attempt": attempt, "at_utc": _now(), "error_type": type(exc).__name__, "error": str(exc)})
            # Retry transport failures only; never repeatedly fetch around data faults.
            if not isinstance(exc, OSError) or attempt == config["maximum_attempts"] or time.monotonic() >= deadline:
                break
            time.sleep(min(2 ** attempt, 8, max(0, deadline - time.monotonic())))
        finally:
            time.sleep(min(config["sleep_seconds_per_worker"], max(0, deadline - time.monotonic())))
    return {"symbol": symbol, "status": "failed", "rows": 0, "error": attempts[-1]["error"],
            "error_type": attempts[-1]["error_type"], "attempts": attempts}


def download_yahoo(config: dict, config_path: str | Path, *, only: list[str] | None = None,
                   progress_callback: ProgressCallback | None = None,
                   cancel_check: Callable[[], bool] | None = None) -> dict:
    """Download/audit fixed members and account for every requested symbol.

    Cancel/deadline stops new requests. Already-running requests finish at a
    symbol boundary, bounded by the configured socket timeout; this is not a
    process-killing hard timeout. An external process timeout may enforce that.
    """
    config = _validated_config(config)
    config_path = _path(config_path)
    # Resolve the original file before any network or destination mutation.
    config_hash = elder_data.sha256(config_path)
    universe = elder_data.load_symbols(Path(config["universe_file"]), config["fixed_universe_count"])
    symbols = list(universe if only is None else only)
    if not symbols or len(symbols) != len(set(symbols)) or not set(symbols).issubset(universe):
        raise ValueError("--only must contain distinct members of the unchanged fixed universe.")
    root, reports = Path(config["data_root"]), Path(config["report_root"])
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    report_path = reports / f"yahoo_download_{identifier}.json"
    log_path = reports / f"yahoo_download_{identifier}.jsonl"
    identity = {"market": "US", "source": elder_data.SOURCE, "interval": "1d", "calendar": "XNYS",
                "price_adjustment": ADJUSTMENT, "volume_unit": "shares", "start": config["start"], "end": config["end"],
                "universe_sha256": elder_data.sha256(Path(config["universe_file"]))}
    report = {**identity, "status": "running", "requested_symbols": symbols, "total": len(symbols),
              "completed": 0, "success": 0, "failed": 0, "cancelled": 0, "outcomes": [],
              "started_at_utc": _now(), "data_root": str(root), "report_path": str(report_path),
              "events_path": str(log_path), "config_path": str(config_path), "config_sha256": config_hash,
              "effective_config": config,
              "fixed_universe_count": len(universe), "workers": config["workers"],
              "network_timeout_seconds": config["network_timeout_seconds"], "outer_timeout_seconds": config["outer_timeout_seconds"],
              "timeout_policy": "Stop scheduling at deadline; active symbols finish with bounded socket timeouts. Use an outer process timeout for a hard wall-clock bound.",
              "calendar_provider": "exchange_calendars XNYS", "source_reconstruction_verified": False}

    def emit(event: str, row: dict | None = None):
        elder_data.save_json(report_path, report)
        if progress_callback is not None:
            progress_callback({"event": event, **{k: report[k] for k in ("status", "completed", "total", "success", "failed", "cancelled", "report_path")},
                               "current_symbol": row["symbol"] if row else None, "result": row})

    def record(row: dict):
        report["outcomes"].append(row)
        key = row["status"]
        report[key] += 1
        report["completed"] = report["success"] + report["failed"]
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"recorded_at_utc": _now(), **row}, ensure_ascii=False, allow_nan=False) + "\n")
        emit("symbol_finished", row)

    deadline = time.monotonic() + config["outer_timeout_seconds"]
    emit("started")
    stop_reason = None
    lock_path = root / ".yahoo_download.lock"
    owns_lock = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        try:
            handle = lock_path.open("x", encoding="utf-8")
        except FileExistsError:
            raise ValueError("Another Yahoo download owns this data_root lock. If an earlier process was interrupted, verify it has stopped before manually removing " + str(lock_path)) from None
        owns_lock = True
        with handle:
            json.dump({"pid": os.getpid(), "started_at_utc": report["started_at_utc"], "report_path": str(report_path)}, handle)
        identity_path = root / "market_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing US market identity differs; choose a new independent data_root.")
        calendar = elder_data.make_calendar(config["start"], config["end"])
        calendar_path = root / "calendar.csv"
        if calendar_path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(calendar_path, keep_default_na=False), calendar, check_dtype=False)
        else:
            calendar.to_csv(calendar_path, index=False)
        if not identity_path.exists():
            elder_data.save_json(identity_path, identity)
        report["calendar_sha256"] = elder_data.sha256(calendar_path)
        import exchange_calendars
        report["calendar_package_version"] = exchange_calendars.__version__
        pending = iter(symbols)
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            active = {}
            exhausted = False
            while active or not exhausted:
                if cancel_check is not None and cancel_check():
                    stop_reason = "cancelled"
                elif time.monotonic() >= deadline and stop_reason is None:
                    stop_reason = "timeout"
                while not stop_reason and not exhausted and len(active) < config["workers"]:
                    symbol = next(pending, None)
                    if symbol is None:
                        exhausted = True
                        break
                    active[pool.submit(_download_symbol, symbol, config, calendar, deadline)] = symbol
                if not active:
                    break
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    symbol = active.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {"symbol": symbol, "status": "failed", "rows": 0, "error_type": type(exc).__name__, "error": str(exc)}
                    record(row)
        completed_symbols = {row["symbol"] for row in report["outcomes"]}
        for symbol in symbols:
            if symbol not in completed_symbols:
                record({"symbol": symbol, "status": "cancelled" if stop_reason == "cancelled" else "failed", "rows": 0,
                        "error_type": "CancelledBeforeStart" if stop_reason == "cancelled" else "OuterTimeout",
                        "error": "Not started: user cancellation." if stop_reason == "cancelled" else "Not started: outer download deadline reached."})
    except Exception as exc:
        completed_symbols = {row["symbol"] for row in report["outcomes"]}
        for symbol in symbols:
            if symbol not in completed_symbols:
                record({"symbol": symbol, "status": "failed", "rows": 0, "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if owns_lock:
            lock_path.unlink()
    by_symbol = {row["symbol"]: row for row in report["outcomes"]}
    report["outcomes"] = [by_symbol[symbol] for symbol in symbols]
    report["status"] = ("cancelled" if stop_reason == "cancelled" else "partial" if report["failed"] and report["success"]
                        else "failed" if report["failed"] else "completed")
    report["finished_at_utc"] = _now()
    report["source_reconstruction_verified"] = report["success"] == len(symbols)
    emit("finished")
    return report
