"""Bounded Yahoo daily-data downloader for an explicit US research universe.

Keeps provider responses, corporate actions, adjustment factors and hashes.
No substitute symbols, missing-bar filling, volume filling or price repair.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data_layer.storage.partitioned_storage import save_kline_to_market_shard

TIMEZONE = ZoneInfo("America/New_York")
SOURCE = "yahoo_finance_chart"
PRICE_BASIS = "yahoo_split_dividend_adjusted_ohlc"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calendar_frame(start, end):
    import exchange_calendars as xcals
    calendar = xcals.get_calendar("XNYS", start=start, end=end)
    schedule = calendar.schedule.loc[start:end]
    sessions = {pd.Timestamp(d).date(): pd.Timestamp(row["close"]).tz_convert(TIMEZONE).tz_localize(None).isoformat()
                for d, row in schedule.iterrows()}
    days = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": days, "is_trading_day": [d.date() in sessions for d in days],
                         "session_close": [sessions.get(d.date()) for d in days],
                         "exchange": "XNYS", "source": "exchange_calendars_XNYS"})


def request_chart(symbol, start, end, timeout):
    first = datetime.fromisoformat(start).replace(tzinfo=TIMEZONE)
    exclusive_end = (datetime.fromisoformat(end) + timedelta(days=1)).replace(tzinfo=TIMEZONE)
    parameters = {"period1": int(first.timestamp()), "period2": int(exclusive_end.timestamp()),
                  "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{urlencode(parameters)}"
    request = Request(url, headers={"User-Agent": "PocketAgent educational research"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(20_000_001)
    if len(raw) > 20_000_000:
        raise ValueError("Unexpected oversized provider response")
    return raw, url


def normalize_chart(raw, symbol, calendar, *, min_price=10., min_dollar_volume=25_000_000.):
    envelope = json.loads(raw)["chart"]
    if envelope.get("error") or not envelope.get("result"):
        raise ValueError(f"Provider error: {envelope.get('error')}")
    result = envelope["result"][0]; meta = result["meta"]
    if meta.get("symbol") != symbol or meta.get("currency") != "USD" or meta.get("instrumentType") != "EQUITY":
        raise ValueError("Unexpected symbol/currency/instrument identity")
    if meta.get("exchangeName") not in {"NMS", "NYQ", "NGM", "NCM"}:
        raise ValueError(f"Not an allowed US primary exchange: {meta.get('exchangeName')}")
    if meta.get("exchangeTimezoneName") != "America/New_York":
        raise ValueError("Unexpected exchange timezone")
    screening_price = float(meta["regularMarketPrice"])
    if not np.isfinite(screening_price) or screening_price < min_price:
        raise ValueError(f"Price screen failed: {screening_price} < {min_price} USD")
    quote_data = result["indicators"]["quote"][0]
    timestamps = result["timestamp"]
    frame = pd.DataFrame({name: quote_data[name] for name in ("open", "high", "low", "close", "volume")})
    frame["datetime"] = [pd.Timestamp(datetime.fromtimestamp(t, TIMEZONE).date()) for t in timestamps]
    frame["adjusted_close"] = result["indicators"]["adjclose"][0]["adjclose"]
    if frame.empty or frame["datetime"].duplicated().any() or not frame["datetime"].is_monotonic_increasing:
        raise ValueError("Empty, duplicate or unordered dates")
    numeric = frame[["open", "high", "low", "close", "volume", "adjusted_close"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Missing/nonfinite provider OHLC, adjusted close or volume; values are not filled")
    prices = numeric[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or (numeric.adjusted_close <= 0).any() or (numeric.volume < 0).any():
        raise ValueError("Invalid prices or negative volume")
    if (prices.high < prices[["open", "close", "low"]].max(axis=1)).any() or (prices.low > prices[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC ordering")
    sessions = set(pd.to_datetime(calendar.loc[calendar.is_trading_day, "date"]))
    observed = set(frame.datetime)
    # All explicitly selected companies predate this research range. A later IPO
    # must be handled explicitly, never by silently shortening the required range.
    missing, unexpected = sorted(sessions-observed), sorted(observed-sessions)
    if missing or unexpected:
        raise ValueError(f"Calendar mismatch: missing={[str(d.date()) for d in missing[:8]]}, unexpected={[str(d.date()) for d in unexpected[:8]]}")
    dollar_volume = float((numeric.close * numeric.volume).tail(60).median())
    if dollar_volume < min_dollar_volume:
        raise ValueError(f"Liquidity screen failed: median daily dollar volume {dollar_volume}")
    factor = numeric.adjusted_close / numeric.close
    output = frame[["datetime"]].copy()
    for column in ("open", "high", "low", "close"):
        output[column] = numeric[column] * factor
    output["volume"] = numeric.volume  # provider-reported shares; no invented scaling
    output["symbol"] = symbol; output["freq"] = "daily"; output["adjust"] = "pre"; output["source"] = SOURCE
    quality = {"rows": len(frame), "start": str(frame.datetime.min().date()), "end": str(frame.datetime.max().date()),
               "screening_price_usd": screening_price, "screening_price_time": meta.get("regularMarketTime"),
               "median_last60_dollar_volume": dollar_volume, "zero_volume_rows": int((numeric.volume == 0).sum()),
               "missing_sessions": 0, "duplicate_dates": 0, "invalid_ohlc_rows": 0,
               "exchange": meta["exchangeName"], "price_basis": PRICE_BASIS,
               "volume_basis": "provider_reported_shares_not_rescaled_by_dividend_factor",
               "minimum_vendor_close_in_range": float(numeric.close.min()),
               "split_events": len(result.get("events", {}).get("splits", {})),
               "dividend_events": len(result.get("events", {}).get("dividends", {}))}
    return output, frame.assign(adjustment_factor=factor), quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-12")
    parser.add_argument("--storage-root", type=Path, default=Path("runtime_layer/data_us"))
    parser.add_argument("--timeout", type=float, default=20.)
    parser.add_argument("--sleep", type=float, default=.5)
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.read_text(encoding="utf-8-sig").splitlines() if s.strip() and not s.startswith("#")]
    if not symbols or len(symbols) != len(set(symbols)) or any(not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", s) for s in symbols):
        parser.error("Expected an explicit unique US ticker list")
    root=args.storage_root; root.mkdir(parents=True, exist_ok=True)
    manifest_path=root/"download_manifest.json"
    if manifest_path.exists():
        parser.error("Use a new output root for a new snapshot; preserve existing evidence")
    calendar=calendar_frame(args.start,args.end);calendar.to_csv(root/"calendar.csv",index=False)
    manifest={"source":SOURCE,"price_basis":PRICE_BASIS,"downloaded_at_utc":datetime.now(timezone.utc).isoformat(),
              "requested_start":args.start,"requested_end":args.end,"symbols":symbols,"symbols_sha256":sha(args.symbols),
              "calendar_sha256":sha(root/"calendar.csv"),"calendar_package_version":importlib.metadata.version("exchange-calendars"),
              "parameters":{"interval":"1d","includeAdjustedClose":True,"events":"div,splits","workers":1,"sleep":args.sleep,"timeout":args.timeout},
              "selection":{"min_current_price_usd":10,"min_median_last60_dollar_volume":25000000,
                           "policy":"explicit_large_company_candidates_not_return_ranked; exploratory_survivor_sample"},
              "status":"running","outcomes":[]}
    write_json(manifest_path,manifest)
    for index,symbol in enumerate(symbols):
        row={"symbol":symbol,"status":"failed","requested_at_utc":datetime.now(timezone.utc).isoformat()}
        try:
            raw,url=request_chart(symbol,args.start,args.end,args.timeout)
            raw_path=root/"provider_raw"/f"{symbol}.json";raw_path.parent.mkdir(parents=True,exist_ok=True);raw_path.write_bytes(raw)
            row.update(url=url,raw_sha256=sha(raw_path),raw_path=str(raw_path.relative_to(root)))
            bars,vendor,quality=normalize_chart(raw,symbol,calendar)
            vendor_path=root/"provider_normalized"/f"{symbol}.csv";vendor_path.parent.mkdir(parents=True,exist_ok=True);vendor.to_csv(vendor_path,index=False)
            saved=save_kline_to_market_shard(bars,root)
            if not saved.ok: raise ValueError(str(saved))
            row.update(status="completed",**quality,shard_path=saved.shard_path,shard_sha256=sha(root/saved.shard_path),vendor_csv_sha256=sha(vendor_path))
        except Exception as exc:
            row.update(error_type=type(exc).__name__,error=str(exc))
        manifest["outcomes"].append(row);write_json(manifest_path,manifest)
        pd.DataFrame(manifest["outcomes"]).to_csv(root/"download_report.csv",index=False)
        print(json.dumps({"completed":index+1,"total":len(symbols),**row},ensure_ascii=False),flush=True)
        if index+1<len(symbols):time.sleep(args.sleep)
    successful=sum(r["status"]=="completed" for r in manifest["outcomes"])
    manifest.update(status="completed" if successful==len(symbols) else "partial" if successful else "failed",successful_symbols=successful,failed_symbols=len(symbols)-successful,finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path,manifest)
    print(json.dumps({"status":manifest["status"],"successful":successful,"failed":len(symbols)-successful,"root":str(root)}),flush=True)
    if successful != len(symbols):raise SystemExit(2)


if __name__ == "__main__":
    main()
