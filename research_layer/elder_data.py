"""Restore the fixed US research universe from a versioned Yahoo daily snapshot.

This is an independent US research input path; no A-share market configuration
or existing downloader is changed. Raw responses and generated data stay under
runtime_layer and are never source artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NY = ZoneInfo("America/New_York")
SOURCE = "yahoo_finance_chart"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_symbols(path: Path, expected: int = 100) -> list[str]:
    symbols = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(symbols) != expected or len(set(symbols)) != expected:
        raise ValueError(f"Expected exactly {expected} distinct symbols; got {len(symbols)} rows.")
    if not all(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,11}", s) for s in symbols):
        raise ValueError("Invalid symbol in fixed universe.")
    return symbols


def make_calendar(start: str, end: str) -> pd.DataFrame:
    import exchange_calendars as xcals

    dates = pd.date_range(start, end, freq="D")
    calendar = xcals.get_calendar("XNYS", start=start, end=end)
    schedule = calendar.schedule.loc[start:end]
    sessions = {pd.Timestamp(i).date().isoformat(): row for i, row in schedule.iterrows()}
    if start <= "2025-01-09" <= end and "2025-01-09" in sessions:
        raise ValueError("XNYS calendar incorrectly treats 2025-01-09 Carter closure as open.")
    rows = []
    for day in dates:
        key = day.date().isoformat()
        item = sessions.get(key)
        def local_stamp(name):
            if item is None:
                return ""
            stamp = pd.Timestamp(item[name])
            if stamp.tz is None:
                stamp = stamp.tz_localize("UTC")
            return stamp.tz_convert("America/New_York").tz_localize(None).isoformat()
        rows.append({"date": key, "is_trading_day": item is not None,
                     "session_open": local_stamp("open"), "session_close": local_stamp("close"),
                     "exchange": "XNYS", "timezone": "America/New_York"})
    return pd.DataFrame(rows)


def fetch_chart(symbol: str, start: str, end: str, timeout: float = 20):
    first = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    exclusive = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(days=1)
    params = {"period1": int(first.timestamp()), "period2": int(exclusive.timestamp()),
              "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe="") + "?" + urlencode(params)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as response:
        body = response.read()
        status = response.status
    return body, {"source": SOURCE, "url": url, "symbol": symbol, "start": start, "end": end,
                  "params": params, "download_started_utc": started,
                  "download_finished_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": round(time.monotonic() - clock, 6), "http_status": status,
                  "network_timeout_seconds": timeout}


def parse_chart(body: bytes, symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(body)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo chart error: {chart['error']}")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("Expected one chart result.")
    result = results[0]
    meta = result.get("meta", {})
    if meta.get("currency") != "USD" or meta.get("instrumentType") != "EQUITY":
        raise ValueError("Expected a USD equity; universe replacement is not allowed.")
    timestamps = result.get("timestamp", [])
    if not timestamps:
        raise ValueError("No daily rows returned.")
    quote_data = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    if any(len(quote_data.get(k, [])) != len(timestamps) for k in ["open", "high", "low", "close", "volume"]) or len(adjusted) != len(timestamps):
        raise ValueError("Mismatched OHLCV/adjusted-close array lengths.")
    rows = []
    for index, timestamp in enumerate(timestamps):
        date = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(NY).date().isoformat()
        if not start <= date <= end:
            raise ValueError(f"Provider returned a bar outside the requested dates: {date}")
        values = {k: quote_data[k][index] for k in ["open", "high", "low", "close", "volume"]}
        if any(v is None or not isinstance(v, (int, float)) or not math.isfinite(v) for v in [*values.values(), adjusted[index]]):
            raise ValueError(f"Missing/nonfinite OHLCV or adjusted close on {date}; no rows silently discarded.")
        if min(values[k] for k in ["open", "high", "low", "close"]) <= 0 or adjusted[index] <= 0 or values["volume"] < 0:
            raise ValueError(f"Nonpositive price or negative volume on {date}.")
        tolerance = values["high"] * 1e-9
        if values["high"] + tolerance < max(values["open"], values["close"], values["low"]) or values["low"] - tolerance > min(values["open"], values["close"], values["high"]):
            raise ValueError(f"Illegal OHLC range on {date}.")
        factor = adjusted[index] / values["close"]
        row = {"symbol": symbol, "datetime": date,
               **{k: values[k] * factor for k in ["open", "high", "low", "close"]},
               "volume": values["volume"],
               **{"vendor_" + k: values[k] for k in ["open", "high", "low", "close"]},
               "adjust_factor": factor, "source": SOURCE, "freq": "daily", "adjust": "pre",
               "is_tradeable": values["volume"] > 0}
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame["datetime"].duplicated().any() or not frame["datetime"].is_monotonic_increasing:
        raise ValueError("Duplicate or unordered dates; refusing to deduplicate silently.")
    # These are the provider's split-adjusted quote prices, not a guarantee of
    # historical nominal execution prices or historical exchange tick size.
    return frame, {"provider_meta": meta, "events": result.get("events", {}),
                   "adjustment_definition": "vendor_OHLC * adjclose/vendor_close",
                   "volume_unit": "shares", "vendor_quote_caveat":
                   "Vendor OHLC may already reflect splits; adjust_factor alone does not recover historical nominal exchange ticks."}


def audit_frame(frame: pd.DataFrame, calendar: pd.DataFrame, start: str, end: str) -> dict:
    sessions = set(calendar.loc[calendar["is_trading_day"], "date"])
    observed = set(frame["datetime"])
    unexpected = sorted(observed - sessions)
    in_range = {d for d in sessions if max(start, frame.iloc[0]["datetime"]) <= d <= end}
    missing = sorted(in_range - observed)
    if unexpected or missing:
        raise ValueError(f"Calendar mismatch: closed-day bars={unexpected[:8]}, missing sessions={missing[:8]}; no fill or deletion permitted.")
    zero = frame.loc[frame["volume"].eq(0), "datetime"].tolist()
    natural = calendar.assign(_period=pd.to_datetime(calendar["date"]).dt.to_period("W-SUN"))
    weeks = []
    for period, group in natural.groupby("_period", sort=True):
        if period.start_time.date().isoformat() < start or period.end_time.date().isoformat() > end:
            continue
        expected = set(group.loc[group["is_trading_day"], "date"])
        if expected and expected.issubset(observed):
            weeks.append(max(expected))
    eligible = None
    if len(weeks) >= 100:
        later = sorted(d for d in sessions if d > weeks[99])
        eligible = later[0] if later else None
    return {"rows": len(frame), "first_date": frame.iloc[0]["datetime"], "last_date": frame.iloc[-1]["datetime"],
            "duplicate_dates": 0, "illegal_OHLC": 0, "missing_volume": 0, "zero_volume_dates": zero,
            "zero_volume_policy": "Retained as nontradeable observations; not removed or replaced.",
            "missing_open_sessions": missing, "closed_day_rows": unexpected, "complete_weeks": len(weeks),
            "warmup_100_completed_weeks_first_eligible_session": eligible,
            "rows_2026_quality_only": int(frame["datetime"].str.startswith("2026").sum()),
            "adjust": "pre", "volume_unit": "shares", "data_version_matches_previous_cloud_snapshot": False}


def restore_symbol(symbol: str, start: str, end: str, root: Path, calendar: pd.DataFrame, timeout: float) -> dict:
    raw = root / "raw" / f"{symbol}.json"
    daily = root / "daily" / f"{symbol}.csv"
    metadata = root / "metadata" / f"{symbol}.json"
    for path in [raw, daily, metadata]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if metadata.exists():
        prior = json.loads(metadata.read_text(encoding="utf-8"))
        if prior.get("status") == "success" and prior.get("start") == start and prior.get("end") == end:
            if raw.exists() and daily.exists() and sha256(raw) == prior["raw_sha256"] and sha256(daily) == prior["daily_sha256"]:
                return {**prior, "reused_validated_cache": True}
            raise ValueError(f"Existing successful snapshot hash mismatch for {symbol}; refusing overwrite.")
    body, request = fetch_chart(symbol, start, end, timeout)
    # A failed response can be useful for diagnosis. Retrying a failed symbol
    # preserves its earlier bytes/metadata before writing the new attempt.
    if raw.exists():
        old_hash = sha256(raw)
        archive = root / "failed_attempts" / symbol
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f"{old_hash}.json").write_bytes(raw.read_bytes())
        if metadata.exists():
            (archive / f"{old_hash}.metadata.json").write_bytes(metadata.read_bytes())
    raw.write_bytes(body)
    try:
        frame, vendor = parse_chart(body, symbol, start, end)
        quality = audit_frame(frame, calendar, start, end)
        temporary = daily.with_name(daily.name + ".partial")
        frame.to_csv(temporary, index=False)
        temporary.replace(daily)
        record = {"status": "success", **request, **vendor, "quality": quality,
                  "raw_file": str(raw), "daily_file": str(daily), "raw_sha256": sha256(raw), "daily_sha256": sha256(daily)}
    except Exception as exc:
        record = {"status": "failed", **request, "raw_file": str(raw), "raw_sha256": sha256(raw),
                  "error_type": type(exc).__name__, "error": str(exc)}
        save_json(metadata, record)
        raise
    save_json(metadata, record)
    return record


def audit_snapshot(config: dict, config_path: Path) -> dict:
    """Re-read every file and reconcile hashes, dates and the source transformation."""
    root = Path(config["data_root"])
    symbols = load_symbols(Path(config["universe_file"]), config["fixed_universe_count"])
    calendar = make_calendar(config["start"], config["end"])
    stored_calendar = pd.read_csv(root / "calendar.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(stored_calendar, calendar, check_dtype=False)
    rows = []
    hashes = {"calendar.csv": sha256(root / "calendar.csv")}
    for symbol in symbols:
        try:
            metadata = json.loads((root / "metadata" / f"{symbol}.json").read_text(encoding="utf-8"))
            if metadata["status"] != "success" or metadata["start"] != config["start"] or metadata["end"] != config["end"]:
                raise ValueError("Missing successful matching download metadata.")
            raw = root / "raw" / f"{symbol}.json"
            daily = root / "daily" / f"{symbol}.csv"
            if sha256(raw) != metadata["raw_sha256"] or sha256(daily) != metadata["daily_sha256"]:
                raise ValueError("Raw/CSV hash mismatch.")
            rebuilt, _ = parse_chart(raw.read_bytes(), symbol, config["start"], config["end"])
            frame = pd.read_csv(daily)
            pd.testing.assert_frame_equal(frame, rebuilt, check_dtype=False, rtol=1e-12, atol=1e-12)
            quality = audit_frame(frame, calendar, config["start"], config["end"])
            last60 = frame.tail(60)
            diagnostic = {"last_requested_session_vendor_close": float(frame.iloc[-1]["vendor_close"]),
                          "last60_median_vendor_close_times_volume": float((last60.vendor_close * last60.volume).median()),
                          "no_stock_reselection": True}
            rows.append({"symbol": symbol, "status": "success", **quality, **diagnostic})
            hashes[str(raw.relative_to(root))] = metadata["raw_sha256"]
            hashes[str(daily.relative_to(root))] = metadata["daily_sha256"]
            hashes[f"metadata/{symbol}.json"] = sha256(root / "metadata" / f"{symbol}.json")
        except Exception as exc:
            rows.append({"symbol": symbol, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    good = [r for r in rows if r["status"] == "success"]
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": SOURCE, "market": "US", "fixed_universe_count": len(symbols),
              "successful_stocks": len(good), "failed_stocks": len(rows) - len(good),
              "requested_start": config["start"], "requested_end": config["end"],
              "rows": sum(r["rows"] for r in good), "calendar_natural_dates": len(calendar),
              "calendar_open_sessions": int(calendar.is_trading_day.sum()),
              "all_hashes_and_source_reconstruction_verified": len(good) == len(rows),
              "actual_first_dates": sorted({r["first_date"] for r in good}),
              "actual_last_dates": sorted({r["last_date"] for r in good}),
              "zero_volume_rows": sum(len(r["zero_volume_dates"]) for r in good),
              "warmup_100_week_eligible_dates": sorted({r["warmup_100_completed_weeks_first_eligible_session"] for r in good if r["warmup_100_completed_weeks_first_eligible_session"]}),
              "rows_2026_quality_only": sum(r["rows_2026_quality_only"] for r in good),
              "minimum_last_requested_session_vendor_close": min((r["last_requested_session_vendor_close"] for r in good), default=None),
              "minimum_last60_median_vendor_dollar_volume": min((r["last60_median_vendor_close_times_volume"] for r in good), default=None),
              "price_below_10_USD_at_last_requested_session": [r["symbol"] for r in good if r["last_requested_session_vendor_close"] < 10],
              "approximate_last60_median_dollar_volume_below_25million": [r["symbol"] for r in good if r["last60_median_vendor_close_times_volume"] < 25_000_000],
              "liquidity_diagnostic_caveat": "Vendor close times share volume is an approximate dollar-turnover diagnostic, not exact traded value, spread, executable liquidity or capacity. Threshold flags never replace symbols.",
              "config_sha256": sha256(config_path), "universe_sha256": sha256(Path(config["universe_file"])),
              "downloader_source_sha256": sha256(Path(__file__)), "file_hashes": hashes, "outcomes": rows,
              "limitations": ["New download vintage; the vanished prior cloud snapshot is unavailable for byte equality.",
                              "Yahoo split/dividend adjustment is not the BaoStock adjustment convention.",
                              "Vendor quote prices may already reflect splits; nominal historical exchange ticks are not certified.",
                              "Current-survivor fixed sample, not a point-in-time whole-market universe.",
                              "2026 rows are quality-only; profitability research remains on previously inspected 2022–2025."]}
    save_json(Path(config["report_root"]) / "quality_summary.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/research/us100_download.json")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Reconcile all fixed-universe source files without downloading.")
    parser.add_argument("--only", nargs="+", help="Retry only named members of the fixed universe.")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    universe = load_symbols(Path(config["universe_file"]), config["fixed_universe_count"])
    if args.audit_only:
        if args.probe or args.only:
            parser.error("--audit-only cannot be combined with --probe or --only.")
        report = audit_snapshot(config, Path(args.config))
        print(json.dumps({k: v for k, v in report.items() if k not in {"file_hashes", "outcomes"}}, ensure_ascii=False), flush=True)
        return 0 if report["failed_stocks"] == 0 else 2
    if args.probe:
        scope = config["probe"]
        symbols = [scope["symbol"]]
        start, end, root = scope["start"], scope["end"], Path(scope["data_root"])
    else:
        start, end, root = config["start"], config["end"], Path(config["data_root"])
        symbols = args.only or universe
        if len(set(symbols)) != len(symbols) or not set(symbols).issubset(universe):
            parser.error("--only must contain distinct members of the unchanged fixed universe.")
    root.mkdir(parents=True, exist_ok=True)
    calendar = make_calendar(start, end)
    calendar.to_csv(root / "calendar.csv", index=False)
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    log = root / "download_events.jsonl"
    for index, symbol in enumerate(symbols):
        try:
            record = restore_symbol(symbol, start, end, root, calendar, config["network_timeout_seconds"])
            brief = {"symbol": symbol, "status": "success", "rows": record["quality"]["rows"],
                     "first_date": record["quality"]["first_date"], "last_date": record["quality"]["last_date"],
                     "raw_sha256": record["raw_sha256"], "daily_sha256": record["daily_sha256"],
                     "reused_validated_cache": record.get("reused_validated_cache", False)}
        except Exception as exc:
            brief = {"symbol": symbol, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        rows.append(brief)
        brief["progress"] = f"{index + 1}/{len(symbols)}"
        brief["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(brief, ensure_ascii=False) + "\n")
        print(json.dumps(brief, ensure_ascii=False), flush=True)
        if index + 1 < len(symbols):
            time.sleep(config["sleep_seconds"])
    import exchange_calendars as xcals
    final = {"source": SOURCE, "scope": "fixed_requested_us100_only", "probe": args.probe,
             "requested_symbols": symbols, "requested_start": start, "requested_end": end,
             "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
             "success": sum(r["status"] == "success" for r in rows), "failed": sum(r["status"] == "failed" for r in rows),
             "calendar_sha256": sha256(root / "calendar.csv"), "calendar_provider": "exchange_calendars XNYS",
             "calendar_package_version": xcals.__version__, "natural_dates": len(calendar),
             "open_sessions": int(calendar["is_trading_day"].sum()), "outcomes": rows,
             "config_sha256": sha256(Path(args.config)), "universe_sha256": sha256(Path(config["universe_file"])),
             "previous_cloud_snapshot_restored": False,
             "price_adjustment": "vendor_OHLC * adjclose/vendor_close", "volume_unit": "shares"}
    # --only creates a separate attempt record and never erases the full report.
    name = "probe_report.json" if args.probe else ("retry_report_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".json" if args.only else "download_report.json")
    save_json(root / name, final)
    report_root = Path(config["report_root"])
    save_json(report_root / name, final)
    print(json.dumps({"complete": True, "success": final["success"], "failed": final["failed"], "report": str(root / name)}), flush=True)
    return 0 if final["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
