from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Iterable

import pandas as pd

from download_layer.clients.baostock_client import (
    BAOSTOCK_ADJUST_TO_NAME,
    BAOSTOCK_FREQUENCY_TO_STORAGE,
    BaoStockClient,
    to_baostock_symbol,
)
from data_layer.storage.duckdb_storage import (
    save_kline_to_duckdb,
    save_stock_liquidity_daily_to_duckdb,
    save_stock_status_daily_to_duckdb,
    save_trade_calendar_to_duckdb,
)
from data_layer.storage.partitioned_storage import (
    save_daily_liquidity_to_market_shard,
    save_daily_status_to_market_shard,
    save_kline_to_market_shard,
    save_trade_calendar_to_market_shard,
)


@dataclass
class SymbolDownloadResult:
    """Download result for one symbol/frequency/adjustment slice."""

    symbol: str
    ok: bool
    rows: int = 0
    error: str | None = None
    freq: str | None = None
    adjust: str | None = None


@dataclass
class BatchDownloadResult:
    """Download result for a batch of symbols."""

    results: list[SymbolDownloadResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for item in self.results if item.ok)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.results if not item.ok)

    @property
    def saved_rows(self) -> int:
        return sum(item.rows for item in self.results)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": item.symbol,
                    "freq": item.freq,
                    "adjust": item.adjust,
                    "ok": item.ok,
                    "rows": item.rows,
                    "error": item.error,
                }
                for item in self.results
            ]
        )


@dataclass
class DatasetDownloadResult:
    """Download result for one non-symbol or extension dataset."""

    dataset: str
    ok: bool
    rows: int = 0
    error: str | None = None


class DataCollector:
    """
    High-level service for downloading market data and saving it into the configured local storage.

    Download Layer responsibility:
        - call provider client
        - orchestrate request parameters
        - return download result

    Data Layer responsibility:
        - normalize schema
        - validate dataframe
        - save to DuckDB or partitioned market shards
    """

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        storage_root: str | Path | None = None,
        storage_mode: str = "shard",
        client: BaoStockClient | None = None,
        request_sleep_seconds: float = 0.2,
        update_catalog: bool = True,
    ) -> None:
        self.storage_mode = str(storage_mode or "shard").lower()
        self.db_path = Path(db_path) if db_path is not None else Path("runtime_layer/data")
        default_storage_root = self.db_path.parent if self.db_path.suffix.lower() == ".duckdb" else self.db_path
        self.storage_root = Path(storage_root) if storage_root is not None else default_storage_root
        self.client = client or BaoStockClient()
        self.request_sleep_seconds = float(request_sleep_seconds or 0)
        self.update_catalog = bool(update_catalog)

    def download_trade_calendar(
        self,
        *,
        start_date: str,
        end_date: str,
        exchange: str = "CN",
    ) -> DatasetDownloadResult:
        """Download the trading calendar required by later feature progress calculations."""
        try:
            df = self.client.query_trade_calendar(
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
            )
            if self.storage_mode in {"shard", "partitioned", "parts"}:
                saved = save_trade_calendar_to_market_shard(df, self.storage_root, update_catalog=self.update_catalog)
                rows = saved.rows
            else:
                rows = save_trade_calendar_to_duckdb(df, self.db_path)
            return DatasetDownloadResult(dataset="trade_calendar", ok=True, rows=rows)
        except Exception as exc:
            return DatasetDownloadResult(dataset="trade_calendar", ok=False, error=str(exc))

    def download_daily_liquidity(
        self,
        symbol: str,
        *,
        start_date: str,
        end_date: str,
    ) -> SymbolDownloadResult:
        """Download daily turnover and ST status into extension tables."""
        try:
            baostock_symbol = to_baostock_symbol(symbol)
            df = self.client.query_daily_liquidity(
                baostock_symbol,
                start_date=start_date,
                end_date=end_date,
            )
            if df.empty:
                return SymbolDownloadResult(
                    symbol=symbol.upper(),
                    ok=True,
                    rows=0,
                    error=(
                        f"Warning: BaoStock returned 0 rows for {symbol.upper()} / daily_liquidity "
                        f"from {start_date} to {end_date}."
                    ),
                    freq="daily_liquidity",
                    adjust="-",
                )
            if self.storage_mode in {"shard", "partitioned", "parts"}:
                liquidity = save_daily_liquidity_to_market_shard(df, self.storage_root, update_catalog=self.update_catalog)
                save_daily_status_to_market_shard(df, self.storage_root, update_catalog=self.update_catalog)
                rows = liquidity.rows
            else:
                rows = save_stock_liquidity_daily_to_duckdb(df, self.db_path)
                save_stock_status_daily_to_duckdb(df, self.db_path)
            return SymbolDownloadResult(
                symbol=symbol.upper(),
                ok=True,
                rows=rows,
                freq="daily_liquidity",
                adjust="-",
            )
        except Exception as exc:
            return SymbolDownloadResult(
                symbol=symbol.upper(),
                ok=False,
                rows=0,
                error=str(exc),
                freq="daily_liquidity",
                adjust="-",
            )

    def download_kline(
        self,
        symbol: str,
        *,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
        replace_symbol: bool = False,
    ) -> SymbolDownloadResult:
        """
        Download one K-line slice for one symbol and save it to DuckDB.

        One slice means:

            symbol × frequency × adjustflag

        Examples:
            000001.SZ × daily × none
            000001.SZ × daily × pre
            000001.SZ × 60min × none
            000001.SZ × 60min × pre
        """
        frequency = str(frequency or "d").lower()
        adjustflag = str(adjustflag or "3")

        freq_name = _storage_freq_from_baostock(frequency)
        adjust_name = _adjust_name(adjustflag)

        try:
            baostock_symbol = to_baostock_symbol(symbol)

            df = self.client.query_kline(
                baostock_symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                adjustflag=adjustflag,
            )

            if df.empty:
                period_hint = (
                    " Weekly/monthly bars may require a completed period inside the requested date range."
                    if freq_name in {"weekly", "monthly"}
                    else ""
                )
                # BaoStock can return error_code=0 with an empty dataframe for
                # valid no-data windows, such as pre-listing dates, post-delisting
                # dates, or a weekly/monthly window with no completed bar. Keep it
                # visible in reports as a warning, but do not count it as a failed
                # provider/storage request.
                return SymbolDownloadResult(
                    symbol=symbol.upper(),
                    ok=True,
                    rows=0,
                    error=(
                        f"Warning: BaoStock returned 0 rows for {symbol.upper()} / {freq_name} / {adjust_name} "
                        f"from {start_date} to {end_date}.{period_hint}"
                    ),
                    freq=freq_name,
                    adjust=adjust_name,
                )

            # Keep user-facing symbol format in storage.
            df["symbol"] = symbol.upper()

            if self.storage_mode in {"shard", "partitioned", "parts"}:
                saved = save_kline_to_market_shard(
                    df,
                    self.storage_root,
                    replace_symbol=replace_symbol,
                    update_catalog=self.update_catalog,
                )
                saved_rows = saved.rows
            else:
                saved_rows = save_kline_to_duckdb(
                    df,
                    self.db_path,
                    replace_symbol=replace_symbol,
                )

            freq_value = (
                str(df["freq"].iloc[0])
                if "freq" in df.columns and not df.empty
                else freq_name
            )
            adjust_value = (
                str(df["adjust"].iloc[0])
                if "adjust" in df.columns and not df.empty
                else adjust_name
            )

            return SymbolDownloadResult(
                symbol=symbol.upper(),
                ok=True,
                rows=saved_rows,
                error=None,
                freq=freq_value,
                adjust=adjust_value,
            )

        except Exception as exc:
            return SymbolDownloadResult(
                symbol=symbol.upper(),
                ok=False,
                rows=0,
                error=str(exc),
                freq=freq_name,
                adjust=adjust_name,
            )

    def download_daily_kline(
        self,
        symbol: str,
        *,
        start_date: str,
        end_date: str,
        adjustflag: str = "3",
        replace_symbol: bool = False,
    ) -> SymbolDownloadResult:
        """Download daily K-line data for one symbol and save it to DuckDB."""
        return self.download_kline(
            symbol,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
            replace_symbol=replace_symbol,
        )

    def download_many_kline(
        self,
        symbols: Iterable[str],
        *,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
        replace_symbol: bool = False,
    ) -> BatchDownloadResult:
        """
        Download one frequency and one adjustment mode for many symbols.

        This method is kept for compatibility. The App Layer now usually loops over
        multiple frequencies and multiple adjustment flags before calling
        download_kline().
        """
        results: list[SymbolDownloadResult] = []
        symbol_list = list(symbols)

        with self.client:
            for index, symbol in enumerate(symbol_list):
                result = self.download_kline(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjustflag=adjustflag,
                    replace_symbol=replace_symbol,
                )
                results.append(result)

                if self.request_sleep_seconds > 0 and index < len(symbol_list) - 1:
                    sleep(self.request_sleep_seconds)

        return BatchDownloadResult(results=results)

    def download_many_adjusted_kline(
        self,
        symbols: Iterable[str],
        *,
        start_date: str,
        end_date: str,
        frequencies: Iterable[str],
        adjustflags: Iterable[str],
        replace_symbol: bool = False,
    ) -> BatchDownloadResult:
        """
        Download many symbols × many frequencies × many adjustment flags.

        This is useful for CLI or future service usage. The current App Layer has
        its own job-progress loop, so it may call download_kline() directly.
        """
        results: list[SymbolDownloadResult] = []
        symbol_list = list(symbols)
        frequency_list = [str(freq).lower() for freq in frequencies]
        adjustflag_list = [str(flag) for flag in adjustflags]

        with self.client:
            total_tasks = len(symbol_list) * len(frequency_list) * len(adjustflag_list)
            completed = 0

            for frequency in frequency_list:
                for adjustflag in adjustflag_list:
                    for symbol in symbol_list:
                        result = self.download_kline(
                            symbol,
                            start_date=start_date,
                            end_date=end_date,
                            frequency=frequency,
                            adjustflag=adjustflag,
                            replace_symbol=replace_symbol,
                        )
                        results.append(result)
                        completed += 1

                        if self.request_sleep_seconds > 0 and completed < total_tasks:
                            sleep(self.request_sleep_seconds)

        return BatchDownloadResult(results=results)

    def download_many_daily_kline(
        self,
        symbols: Iterable[str],
        *,
        start_date: str,
        end_date: str,
        adjustflag: str = "3",
        replace_symbol: bool = False,
    ) -> BatchDownloadResult:
        """Download daily K-line data for many symbols."""
        return self.download_many_kline(
            symbols,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
            replace_symbol=replace_symbol,
        )


def _storage_freq_from_baostock(frequency: str) -> str:
    return BAOSTOCK_FREQUENCY_TO_STORAGE.get(str(frequency).lower(), str(frequency))


def _adjust_name(adjustflag: str) -> str:
    return BAOSTOCK_ADJUST_TO_NAME.get(str(adjustflag), str(adjustflag))


def read_symbols_from_file(path: str | Path) -> list[str]:
    """Read symbols from a plain text file, one symbol per line."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    symbols: list[str] = []
    for line in file_path.read_text(encoding="utf-8-sig").splitlines():
        symbol = line.replace("\ufeff", "").strip().upper()
        if not symbol or symbol.startswith("#"):
            continue
        if symbol not in symbols:
            symbols.append(symbol)

    return symbols
