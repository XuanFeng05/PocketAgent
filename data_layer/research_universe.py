"""Read-only market coverage and explicit historical A-share universe records.

Provider records are an inventory, not proof of complete historical exchange
coverage. In particular, an empty BJ response never certifies BSE coverage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pandas as pd


MASTER_COLUMNS = ["symbol", "name", "exchange", "ipo_date", "delist_date", "status", "type"]
PROVIDER_REFERENCE = "https://www.baostock.com/mainContent?file=stockBasic.md"


def _write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _unique_run(output_root, prefix):
    now = datetime.now(timezone.utc)
    path = Path(output_root).resolve() / f"{prefix}_{now:%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _date(value, *, required=False):
    if value is None or pd.isna(value) or str(value).strip() == "":
        if required:
            raise ValueError("Missing listing date.")
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError("Use a valid local calendar date.")
    return stamp.date().isoformat()


def normalize_symbol_master(frame) -> pd.DataFrame:
    """Normalize provider stock metadata; keep delisted equities and exclusions.

    Requires the provider's security-type classification as well as an explicit
    exchange/code. Indexes and B shares are excluded with reasons in attrs.
    Malformed stock lifetimes or contradictory duplicate identities are errors.
    """
    data = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    data = data.rename(columns={"code": "symbol", "code_name": "name", "ipoDate": "ipo_date", "outDate": "delist_date"})
    required = {"symbol", "ipo_date", "delist_date", "status", "type"}
    if required - set(data):
        raise ValueError(f"Symbol master missing fields: {sorted(required - set(data))}")
    rows, exclusions = [], []
    for item in data.to_dict("records"):
        raw = str(item["symbol"]).strip().upper()
        match = re.fullmatch(r"(SH|SZ|BJ)\.(\d{6})", raw)
        if match:
            exchange, code = match.groups()
        else:
            match = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", raw)
            if not match:
                exclusions.append({"symbol": raw, "reason": "unsupported_or_ambiguous_exchange_code"})
                continue
            code, exchange = match.groups()
        symbol = f"{code}.{exchange}"
        kind = str(item["type"]).strip()
        if kind not in {"1", "1.0"}:
            exclusions.append({"symbol": symbol, "reason": "not_stock", "provider_type": kind})
            continue
        supported = (exchange == "SH" and code.startswith("6")) or (exchange == "SZ" and code.startswith(("0", "3"))) or (exchange == "BJ" and code.startswith(("4", "8", "920")))
        if not supported:
            exclusions.append({"symbol": symbol, "reason": "not_supported_A_share_code"})
            continue
        status = str(item["status"]).strip()
        if status not in {"0", "1", "0.0", "1.0"}:
            raise ValueError(f"Invalid listing status for {symbol}.")
        ipo = _date(item["ipo_date"], required=True)
        out = _date(item["delist_date"])
        if out and out < ipo:
            raise ValueError(f"Delisting precedes listing: {symbol}.")
        rows.append({"symbol": symbol, "name": str(item.get("name") or ""), "exchange": exchange,
                     "ipo_date": ipo, "delist_date": out, "status": "listed" if status.startswith("1") else "delisted", "type": "A_share"})
    result = pd.DataFrame(rows, columns=MASTER_COLUMNS).drop_duplicates()
    if result["symbol"].duplicated().any():
        raise ValueError("Conflicting symbol-master records; resolve identities before research.")
    result = result.sort_values("symbol").reset_index(drop=True)
    result.attrs["excluded_records"] = exclusions
    result.attrs["scope_note"] = "Observed provider metadata, not an independently reconciled complete historical A-share list. BJ code migrations require an explicit identity map."
    return result


def classify_historical_membership(master: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Retain every security and label overlap; unknown lifetimes stay unknown.

    IPO is inclusive; outDate is treated as the first non-listed day. Provider
    metadata is retrospective identity information, not a historical name/status
    feature available to trading decisions.
    """
    start, end = _date(start, required=True), _date(end, required=True)
    if start > end:
        raise ValueError("start must not exceed end.")
    result = master.copy(deep=True)
    states, reasons = [], []
    for row in result.to_dict("records"):
        if row["ipo_date"] > end:
            state, reason = "excluded", "not_yet_listed"
        elif row["delist_date"] and row["delist_date"] <= start:
            state, reason = "excluded", "delisted_before_interval"
        elif row["status"] == "delisted" and not row["delist_date"]:
            state, reason = "unknown", "missing_delisting_date"
        else:
            state, reason = "eligible", "lifetime_overlaps_interval"
        states.append(state)
        reasons.append(reason)
    result["membership_status"], result["membership_reason"] = states, reasons
    return result


def _calendar_report(path: Path):
    if not path.is_file():
        return {"status": "missing", "path": str(path)}, set(), set()
    frame = pd.read_parquet(path)
    if frame.empty:
        return {"status": "invalid", "reason": "empty_calendar", "path": str(path)}, set(), set()
    if not {"date", "is_trading_day"}.issubset(frame):
        return {"status": "invalid", "reason": "missing_calendar_fields", "path": str(path)}, set(), set()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    flags = frame["is_trading_day"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if dates.isna().any() or flags.isna().any() or dates.dt.tz is not None or dates.duplicated().any():
        return {"status": "invalid", "reason": "bad_or_duplicate_calendar_rows", "path": str(path)}, set(), set()
    days = set(dates.dt.strftime("%Y-%m-%d"))
    opens = set(dates.loc[flags].dt.strftime("%Y-%m-%d"))
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = sorted(set(expected.strftime("%Y-%m-%d")) - days)
    return {"status": "complete" if not missing else "incomplete", "path": str(path), "rows": len(frame),
            "start": min(days) if days else None, "end": max(days) if days else None,
            "open_days": len(opens), "missing_calendar_days": missing,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, days, opens


def audit_local_pre(data_root, output_root=None, *, target_symbols=None) -> dict:
    """Audit saved shards read-only. Never calculate or rank strategy returns."""
    root = Path(data_root).resolve()
    shards = root / "market_parts" if (root / "market_parts").exists() else root
    calendar, calendar_days, open_days = _calendar_report(shards / "_global" / "trade_calendar.parquet")
    all_files = sorted(shards.glob("bucket=*/*/daily_pre.parquet"))
    targets = sorted({str(value).strip().upper() for value in target_symbols}) if target_symbols is not None else []
    if any(not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", value) for value in targets):
        raise ValueError("Target symbols require explicit six-digit code and SH/SZ/BJ exchange.")
    files = [path for path in all_files if target_symbols is None or path.parent.name.replace("_", ".") in targets]
    found, rows = set(), []
    for path in files:
        symbol = path.parent.name.replace("_", ".")
        # A corrupt existing shard is invalid, not both invalid and missing.
        if symbol in found:
            existing = next(item for item in rows if item["symbol"] == symbol)
            existing["status"] = "needs_review"
            existing["issues"].append("multiple_daily_pre_shards")
            existing.setdefault("duplicate_paths", []).append(str(path))
            continue
        found.add(symbol)
        row = {"symbol": symbol, "path": str(path), "status": "invalid", "issues": []}
        try:
            content = path.read_bytes()
            frame = pd.read_parquet(path)
            row.update(rows=len(frame), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
                       modified_ns=path.stat().st_mtime_ns)
            required = {"symbol", "datetime", "open", "high", "low", "close", "volume", "adjust"}
            if required - set(frame):
                raise ValueError(f"Missing fields: {sorted(required - set(frame))}")
            observed_symbols = frame["symbol"].astype(str).str.upper().unique().tolist()
            if observed_symbols != [symbol]:
                row["issues"].append("symbol_path_mismatch")
            dates = pd.to_datetime(frame["datetime"], errors="coerce")
            if dates.dt.tz is not None:
                raise ValueError("Timezone-bearing dates are unsupported.")
            valid_dates = dates.dropna().dt.normalize()
            days = set(valid_dates.dt.strftime("%Y-%m-%d"))
            row.update(start=min(days) if days else None, end=max(days) if days else None,
                       duplicate_dates=int(valid_dates.duplicated().sum()), missing_dates=int(dates.isna().sum()),
                       adjust_values=sorted(frame["adjust"].astype(str).unique().tolist()),
                       sources=sorted(frame["source"].dropna().astype(str).unique().tolist()) if "source" in frame else [])
            if row["duplicate_dates"]: row["issues"].append("duplicate_dates")
            if row["missing_dates"] or not days: row["issues"].append("invalid_or_empty_dates")
            if (dates.dropna() != valid_dates).any(): row["issues"].append("non_daily_timestamps")
            if row["adjust_values"] != ["pre"]: row["issues"].append("not_uniform_pre_adjusted")
            numeric = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
            prices = numeric[["open", "high", "low", "close"]]
            row.update(invalid_price_rows=int((~np.isfinite(prices).all(axis=1) | (prices <= 0).any(axis=1)).sum()),
                       invalid_volume_rows=int((~np.isfinite(numeric["volume"]) | (numeric["volume"] < 0)).sum()),
                       zero_volume_rows=int(numeric["volume"].eq(0).sum()))
            row["invalid_ohlc_rows"] = int(((prices["high"] < prices[["open", "close", "low"]].max(axis=1)) | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))).sum())
            if row["invalid_price_rows"]: row["issues"].append("invalid_prices")
            if row["invalid_volume_rows"]: row["issues"].append("invalid_volume")
            if row["invalid_ohlc_rows"]: row["issues"].append("inconsistent_ohlc")
            if "freq" in frame and not frame["freq"].eq("daily").all(): row["issues"].append("not_daily")
            row["missing_trading_dates"], row["unknown_calendar_dates"], row["nontrading_bars"] = [], [], []
            if days and calendar_days:
                expected = {day for day in open_days if row["start"] <= day <= row["end"]}
                row["missing_trading_dates"] = sorted(expected - days)
                row["unknown_calendar_dates"] = sorted(days - calendar_days)
                row["nontrading_bars"] = sorted((days & calendar_days) - open_days)
                if row["missing_trading_dates"]: row["issues"].append("missing_trading_bars_unexplained")
                if row["unknown_calendar_dates"]: row["issues"].append("calendar_not_covering_bars")
                if row["nontrading_bars"]: row["issues"].append("bars_on_closed_dates")
            else:
                row["issues"].append("calendar_unavailable")
            if calendar["status"] != "complete": row["issues"].append("calendar_incomplete_or_invalid")
            row["adjustment_version_status"] = "unverified_no_factor_snapshot_id"
            row["status"] = "ready_for_feature_validation" if not row["issues"] else "needs_review"
        except Exception as exc:
            row["error"] = str(exc)
            row["issues"].append("read_or_schema_error")
        rows.append(row)
    for symbol in targets:
        if symbol not in found:
            rows.append({"symbol": symbol, "status": "missing", "path": None, "rows": 0, "issues": ["no_daily_pre_shard"]})
    by_state = {state: sum(r["status"] == state for r in rows) for state in sorted({r["status"] for r in rows})}
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
              "calendar": calendar, "summary": {"local_shards": len(files), "local_shards_total": len(all_files), "observed_symbols": len(found),
              "target_count": len(targets) if target_symbols is not None else None, "rows": sum(r.get("rows", 0) for r in rows),
              "statuses": by_state, "all_A_share_coverage_verified": False,
              "adjustment_versions_verified": False}, "symbols": rows,
              "limitations": ["Audit does not calculate strategy returns or choose stocks by performance.",
                              "Local shards do not establish a historical full-market universe.",
                              "Hashes freeze observed bytes but do not certify consistent provider adjustment vintages; verify factors before cross-vintage joins.",
                              "Missing open-day rows are unexplained gaps, not automatically assumed suspensions."]}
    if output_root is not None:
        run = _unique_run(output_root, "coverage")
        report["path"] = str(run)
        _write_json(run / "coverage.json", report)
    return report


def _fetch_worker(output: Path) -> int:
    import socket
    socket.setdefaulttimeout(12)
    import baostock as bs
    payload = {"provider": "baostock", "method": "query_stock_basic", "retrieved_at": datetime.now(timezone.utc).isoformat(),
               "reference": PROVIDER_REFERENCE, "fields": [], "rows": [], "status": "failed"}
    try:
        login = bs.login()
        payload["login_error_code"] = login.error_code
        if login.error_code != "0":
            raise RuntimeError(f"Login error {login.error_code}: {login.error_msg}")
        response = bs.query_stock_basic()
        payload["fields"] = list(response.fields)
        while response.error_code == "0" and response.next():
            payload["rows"].append(response.get_row_data())
        payload.update(query_error_code=response.error_code, query_error_msg=response.error_msg)
        if response.error_code != "0":
            raise RuntimeError(f"Query error {response.error_code}: {response.error_msg}")
        if not payload["rows"]:
            raise RuntimeError("Empty stock-basic response is not coverage evidence.")
        payload["status"] = "completed"
    except Exception as exc:
        payload["error"] = str(exc)
    finally:
        _write_json(output / "provider_raw.json", payload)
        # Closing our own socket avoids an additional potentially blocking logout.
        from baostock.common import context
        sock = getattr(context, "default_socket", None)
        if sock is not None:
            sock.close()
    return 0 if payload["status"] == "completed" else 1


def fetch_baostock_master(output_root, *, timeout_seconds=45) -> dict:
    """Bounded subprocess fetch, preserving raw response and explicit limitations."""
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= 120:
        raise ValueError("timeout_seconds must be between 1 and 120.")
    run = _unique_run(output_root, "symbol_master")
    status = {"path": str(run), "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
              "provider": "baostock", "method": "query_stock_basic", "requested_exchanges": ["SH", "SZ", "BJ"],
              "coverage_verified": False, "reference": PROVIDER_REFERENCE}
    _write_json(run / "status.json", status)
    command = [sys.executable, "-B", "-m", "data_layer.research_universe", "_fetch_worker", "--output", str(run)]
    try:
        process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=timeout_seconds)
        (run / "provider.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        status["returncode"] = process.returncode
        raw_path = run / "provider_raw.json"
        if not raw_path.is_file():
            raise RuntimeError("Provider worker did not return an auditable response.")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if raw["status"] != "completed":
            raise RuntimeError(raw.get("error", "Provider query failed."))
        master = normalize_symbol_master(pd.DataFrame(raw["rows"], columns=raw["fields"]))
        master.to_csv(run / "symbol_master.csv", index=False, encoding="utf-8")
        _write_json(run / "excluded_records.json", master.attrs["excluded_records"])
        observed = {exchange: int(master["exchange"].eq(exchange).sum()) for exchange in ("SH", "SZ", "BJ")}
        status.update(status="completed", observed_equities=len(master), exchange_counts=observed,
                      delisted_records=int(master["status"].eq("delisted").sum()),
                      missing_delisting_dates=int((master["status"].eq("delisted") & master["delist_date"].isna()).sum()),
                      raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                      limitations=["Provider stock-basic records are not independently reconciled to all exchange listing/delisting histories.",
                                   "No BSE records means BSE coverage is missing, never complete.",
                                   "BSE old/new codes need a historical security identity map before merging.",
                                   "Metadata existence does not prove historical adjusted OHLC availability."])
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
        (run / "provider.log").write_text(text(exc.stdout) + text(exc.stderr), encoding="utf-8")
        status.update(status="failed", error="Provider subprocess exceeded timeout and was terminated; no complete response is assumed.")
    except Exception as exc:
        status.update(status="failed", error=str(exc))
    _write_json(run / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description="Historical A-share metadata and saved pre-adjusted coverage")
    parser.add_argument("action", choices=["audit", "fetch-master", "_fetch_worker"])
    parser.add_argument("--data", default="runtime_layer/data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "_fetch_worker":
        raise SystemExit(_fetch_worker(Path(args.output)))
    result = audit_local_pre(args.data, args.output) if args.action == "audit" else fetch_baostock_master(args.output)
    print(json.dumps({key: result[key] for key in ("path", "summary", "status", "error", "exchange_counts", "observed_equities", "delisted_records") if key in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
