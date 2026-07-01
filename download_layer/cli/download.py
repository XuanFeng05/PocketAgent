from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import sleep

import pandas as pd

from download_layer.services.data_collector import DataCollector, SymbolDownloadResult, read_symbols_from_file
from data_layer.storage.partitioned_storage import (
    DAILY_LIQUIDITY_DATASET,
    DAILY_STATUS_DATASET,
    delete_symbol_market_shards,
    ensure_data_root,
    refresh_daily_extension_catalog_from_market_shard,
    refresh_kline_catalog_from_market_shard,
)


DEFAULT_DB_PATH = "runtime_layer/data"
DEFAULT_STORAGE_ROOT = "runtime_layer/data"
DEFAULT_REPORT_PATH = "runtime_layer/runs/download_report.csv"


FREQ_ALIASES = {
    "d": "d",
    "day": "d",
    "daily": "d",
    "w": "w",
    "week": "w",
    "weekly": "w",
    "m": "m",
    "month": "m",
    "monthly": "m",
    "5": "5",
    "5m": "5",
    "5min": "5",
    "15": "15",
    "15m": "15",
    "15min": "15",
    "30": "30",
    "30m": "30",
    "30min": "30",
    "60": "60",
    "60m": "60",
    "60min": "60",
}


ADJUST_ALIASES = {
    "1": "1",
    "post": "1",
    "post-adjusted": "1",
    "hou": "1",
    "后复权": "1",
    "2": "2",
    "pre": "2",
    "pre-adjusted": "2",
    "qian": "2",
    "前复权": "2",
    "3": "3",
    "none": "3",
    "raw": "3",
    "unadjusted": "3",
    "不复权": "3",
}


ADJUST_NAMES = {
    "1": "post",
    "2": "pre",
    "3": "none",
}


def _result_is_skipped(result: SymbolDownloadResult) -> bool:
    return str(result.error or "").startswith("Skipped:")


def _result_is_warning(result: SymbolDownloadResult) -> bool:
    return str(result.error or "").startswith("Warning:")


def _result_status(result: SymbolDownloadResult) -> str:
    if _result_is_skipped(result):
        return "skipped"
    if _result_is_warning(result):
        return "warning"
    return "success" if result.ok else "failed"


def _result_error(result: SymbolDownloadResult) -> str | None:
    return None if _result_is_skipped(result) else result.error


def parse_csv_values(raw: str | None) -> list[str]:
    if raw is None:
        return []

    values: list[str] = []
    for part in str(raw).replace(";", ",").replace("\n", ",").split(","):
        value = part.strip()
        if value:
            values.append(value)

    return values


def normalize_freqs(raw: str | None) -> list[str]:
    values = parse_csv_values(raw)

    if not values:
        values = ["d"]

    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key not in FREQ_ALIASES:
            raise ValueError(
                f"Unsupported frequency: {value}. "
                "Supported examples: d,w,m,5,15,30,60,daily,weekly,monthly,5min,30min."
            )

        freq = FREQ_ALIASES[key]
        if freq not in result:
            result.append(freq)

    return result


def normalize_adjustflags(
    *,
    adjustflags: str | None,
    legacy_adjustflag: str | None,
) -> list[str]:
    """
    Normalize adjustment flags.

    New default:
        3,2 = raw execution price + pre-adjusted technical-analysis price

    Backward compatibility:
        If --adjustflag is provided, use only that one.
    """
    if legacy_adjustflag:
        values = [legacy_adjustflag]
    else:
        values = parse_csv_values(adjustflags)
        if not values:
            values = ["3", "2"]

    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key not in ADJUST_ALIASES:
            raise ValueError(
                f"Unsupported adjust flag: {value}. "
                "Supported examples: 1,2,3,post,pre,none."
            )

        flag = ADJUST_ALIASES[key]
        if flag not in result:
            result.append(flag)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download A-share K-line data into partitioned market shards."
    )

    parser.add_argument(
        "--storage",
        choices=["shard", "duckdb"],
        default="shard",
        help="Storage backend. shard is canonical; duckdb is a legacy compatibility mode.",
    )
    parser.add_argument(
        "--storage-root",
        default=DEFAULT_STORAGE_ROOT,
        help="Root directory for partitioned shard storage. Default: runtime_layer/data.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel symbol download workers. Only shard storage can use workers > 1.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Market data root by default. Only pass a .duckdb file when --storage duckdb is used.",
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Symbol txt file. One symbol per line.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--freqs",
        default="d",
        help=(
            "Comma-separated BaoStock frequencies. "
            "Examples: d,5,30 or daily,5min,30min. Default: d."
        ),
    )
    parser.add_argument(
        "--freq",
        default=None,
        help="Legacy single frequency option. If provided, overrides --freqs.",
    )
    parser.add_argument(
        "--adjustflags",
        default="3,2",
        help=(
            "Comma-separated adjustment flags. "
            "1=post-adjusted, 2=pre-adjusted, 3=none. Default: 3,2."
        ),
    )
    parser.add_argument(
        "--adjustflag",
        choices=["1", "2", "3"],
        default=None,
        help=(
            "Legacy single adjustment flag. "
            "If provided, overrides --adjustflags. "
            "1=post-adjusted, 2=pre-adjusted, 3=none."
        ),
    )
    parser.add_argument(
        "--replace-symbol",
        action="store_true",
        help="Delete existing rows/shards for downloaded symbols before inserting.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between provider requests inside each worker.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help="CSV path to save final download result report. No progress state is written to this file during the run.",
    )

    return parser


def _build_collector(args: argparse.Namespace, *, update_catalog: bool = True) -> DataCollector:
    return DataCollector(
        db_path=Path(args.db),
        storage_root=Path(args.storage_root),
        storage_mode=args.storage,
        request_sleep_seconds=args.sleep,
        update_catalog=update_catalog,
    )


def _download_symbol_bundle(
    *,
    symbol: str,
    args: argparse.Namespace,
    freqs: list[str],
    adjustflags: list[str],
    update_catalog: bool = True,
) -> list[SymbolDownloadResult]:
    collector = _build_collector(args, update_catalog=update_catalog)
    results: list[SymbolDownloadResult] = []
    completed_for_symbol = 0
    total_for_symbol = len(freqs) * len(adjustflags) + 1
    replaced = False

    collector.client.login()
    try:
        for freq in freqs:
            for adjustflag in adjustflags:
                # In shard mode, the parent process pre-deletes symbol shards
                # before fan-out.  Child processes should not delete or update
                # catalog rows while other workers are active.
                replace_this = bool(args.replace_symbol and args.storage != "shard" and not replaced)
                result = collector.download_kline(
                    symbol,
                    start_date=args.start,
                    end_date=args.end,
                    frequency=freq,
                    adjustflag=adjustflag,
                    replace_symbol=replace_this,
                )
                if replace_this:
                    replaced = True
                results.append(result)
                completed_for_symbol += 1
                if args.sleep > 0 and completed_for_symbol < total_for_symbol:
                    sleep(args.sleep)

        liquidity_result = collector.download_daily_liquidity(
            symbol,
            start_date=args.start,
            end_date=args.end,
        )
        results.append(liquidity_result)
    finally:
        collector.client.logout()

    return results


def _result_rows(results: list[SymbolDownloadResult]) -> list[dict[str, object]]:
    return [
        {
            "symbol": item.symbol,
            "freq": item.freq,
            "adjust": item.adjust,
            "ok": item.ok,
            "rows": item.rows,
            "status": _result_status(item),
            "error": _result_error(item),
            "message": item.error if _result_is_skipped(item) else "",
        }
        for item in results
    ]


def _refresh_catalog_after_process_result(storage_root: str | Path, result: SymbolDownloadResult) -> list[str]:
    """Refresh catalog rows in the parent process after multiprocess shard downloads."""
    messages: list[str] = []
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
                messages.append(f"  CATALOG daily_liquidity rows={liquidity.rows}")
            else:
                messages.append(f"  CATALOG daily_liquidity skipped: {liquidity.error}")
            if status.ok:
                messages.append(f"  CATALOG daily_status rows={status.rows}")
            else:
                messages.append(f"  CATALOG daily_status skipped: {status.error}")
            return messages

        refreshed = refresh_kline_catalog_from_market_shard(
            storage_root,
            symbol=symbol,
            freq=freq,
            adjust=adjust,
        )
        if refreshed.ok:
            messages.append(f"  CATALOG rows={refreshed.rows} hash={str(refreshed.data_hash or '')[:12]}")
        else:
            messages.append(f"  CATALOG failed: {refreshed.error}")
    except Exception as exc:
        messages.append(f"  CATALOG failed: {exc}")
    return messages


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    symbols = read_symbols_from_file(args.symbols)
    if not symbols:
        raise SystemExit(f"No symbols found in symbol file: {args.symbols}")

    freqs = normalize_freqs(args.freq or args.freqs)
    adjustflags = normalize_adjustflags(
        adjustflags=args.adjustflags,
        legacy_adjustflag=args.adjustflag,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers or 1))
    if args.storage == "duckdb" and workers > 1:
        print("--storage duckdb uses a single writer; forcing --workers 1.")
        workers = 1

    if args.storage == "shard":
        ensure_data_root(args.storage_root)
        if args.replace_symbol:
            for symbol in symbols:
                delete_symbol_market_shards(args.storage_root, symbol)

    total = len(symbols) * len(freqs) * len(adjustflags) + 1 + len(symbols)
    completed = 0
    results: list[SymbolDownloadResult] = []

    print(
        f"Downloading {len(symbols)} symbols × {len(freqs)} freqs × "
        f"{len(adjustflags)} adjustments = {total} tasks"
    )
    print(f"Storage: {args.storage}")
    if args.storage == "shard":
        print(f"Storage root: {Path(args.storage_root)}")
        print(f"Catalog: {Path(args.storage_root) / 'market_catalog.duckdb'}")
        print(f"Workers: {workers}")
    else:
        print(f"DB: {Path(args.db)}")
    print(f"Freqs: {freqs}")
    print(f"Adjustflags: {adjustflags} ({[ADJUST_NAMES.get(x, x) for x in adjustflags]})")

    # Calendar is global and tiny: download it once, then fan out symbol workers.
    calendar_collector = _build_collector(args, update_catalog=True)
    calendar_collector.client.login()
    try:
        completed += 1
        print(f"[{completed}/{total}] trade_calendar")
        calendar_result = calendar_collector.download_trade_calendar(start_date=args.start, end_date=args.end)
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
    finally:
        calendar_collector.client.logout()

    if calendar_result.ok:
        print(f"  OK rows={calendar_result.rows}")
    else:
        print(f"  FAILED {calendar_result.error}")
        pd.DataFrame(_result_rows(results)).to_csv(report_path, index=False, encoding="utf-8-sig")
        raise SystemExit(1)

    if workers == 1:
        for symbol in symbols:
            bundle_results = _download_symbol_bundle(symbol=symbol, args=args, freqs=freqs, adjustflags=adjustflags, update_catalog=True)
            for result in bundle_results:
                completed += 1
                results.append(result)
                label = f"{result.symbol} freq={result.freq} adjust={result.adjust}"
                print(f"[{completed}/{total}] {label}")
                if _result_is_warning(result):
                    print(f"  WARNING {result.error}")
                elif result.ok:
                    print(f"  OK rows={result.rows}")
                else:
                    print(f"  FAILED {result.error}")
    else:
        print("Multiprocess shard download enabled: each worker uses an independent BaoStock session; catalog refresh runs after the pool stops.")
        catalog_refresh_queue: list[SymbolDownloadResult] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_symbol = {
                executor.submit(_download_symbol_bundle, symbol=symbol, args=args, freqs=freqs, adjustflags=adjustflags, update_catalog=False): symbol
                for symbol in symbols
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    bundle_results = future.result()
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
                for result in bundle_results:
                    completed += 1
                    results.append(result)
                    catalog_refresh_queue.append(result)
                    label = f"{result.symbol} freq={result.freq} adjust={result.adjust}"
                    print(f"[{completed}/{total}] {label}")
                    if _result_is_warning(result):
                        print(f"  WARNING {result.error}")
                    elif result.ok:
                        print(f"  OK rows={result.rows}")
                    else:
                        print(f"  FAILED {result.error}")
        if args.storage == "shard" and workers > 1:
            print("Refreshing shard catalog after all workers stopped...")
            for result in catalog_refresh_queue:
                for catalog_message in _refresh_catalog_after_process_result(args.storage_root, result):
                    print(catalog_message)

    report_df = pd.DataFrame(_result_rows(results))
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    failed = [item for item in results if not item.ok]
    saved_rows = sum(int(item.rows) for item in results)

    print("")
    print("Download finished.")
    print(f"Total tasks: {len(results)}")
    print(f"Succeeded: {len(results) - len(failed)}")
    print(f"Failed: {len(failed)}")
    print(f"Saved rows: {saved_rows}")
    print(f"Report: {report_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
