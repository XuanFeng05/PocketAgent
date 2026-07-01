from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Callable, Iterable, TypeVar

import pandas as pd

from data_layer.storage.data_loader import (
    normalize_kline_dataframe,
    normalize_stock_liquidity_dataframe,
    normalize_stock_status_dataframe,
    normalize_trade_calendar_dataframe,
)


BAOSTOCK_FREQUENCY_TO_STORAGE = {
    "d": "daily",
    "w": "weekly",
    "m": "monthly",
    "5": "5min",
    "15": "15min",
    "30": "30min",
    "60": "60min",
}


BAOSTOCK_ADJUST_TO_NAME = {
    "1": "post",
    "2": "pre",
    "3": "none",
}


T = TypeVar("T")


def get_baostock_kline_fields(frequency: str) -> str:
    """
    Return BaoStock query fields by frequency.

    Field rule:
        Core K-line requests only ask for cross-frequency market fields.
        Derived fields such as pctChg are calculated locally. Daily facts such
        as turnover are downloaded through separate extension assets.
    """
    frequency = str(frequency or "d").lower()

    if frequency in {"d", "w", "m"}:
        return "date,code,open,high,low,close,volume,amount,adjustflag"

    if frequency in {"5", "15", "30", "60"}:
        return (
            "date,time,code,open,high,low,close,"
            "volume,amount,adjustflag"
        )

    raise ValueError(f"Unsupported BaoStock frequency: {frequency}")


def empty_kline_dataframe(
    *,
    symbol: str,
    storage_freq: str,
    adjust_name: str,
    source: str = "baostock",
) -> pd.DataFrame:
    """
    Return an empty dataframe using PocketAgent's core K-line schema.
    """
    return pd.DataFrame(
        columns=[
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
    ).assign(
        symbol=pd.Series(dtype="string"),
        source=source,
        freq=storage_freq,
        adjust=adjust_name,
    )


@dataclass
class BaoStockClient:
    """
    Minimal BaoStock client wrapper.

    BaoStock is imported lazily inside login(), so the project can still import
    this module even if baostock is not installed yet.
    """

    logged_in: bool = False
    max_retries: int = 3
    retry_sleep_seconds: float = 1.5

    def login(self) -> None:
        """Login to BaoStock."""
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError(
                "baostock is not installed. Install it with: pip install baostock"
            ) from exc

        result = bs.login()
        if getattr(result, "error_code", "0") != "0":
            raise RuntimeError(
                f"BaoStock login failed: {result.error_code}, {result.error_msg}"
            )

        self.logged_in = True

    def logout(self) -> None:
        """Logout from BaoStock."""
        try:
            import baostock as bs
        except ImportError:
            self.logged_in = False
            return

        if self.logged_in:
            bs.logout()
        self.logged_in = False

    def _reset_session_after_error(self) -> None:
        """Best-effort logout before retrying a BaoStock query."""
        try:
            self.logout()
        except Exception:
            self.logged_in = False

    def _with_retries(self, description: str, query: Callable[[], T]) -> T:
        """Run a BaoStock query with reconnect retries for transient socket errors."""
        attempts = max(1, int(self.max_retries or 1))
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                if not self.logged_in:
                    self.login()
                return query()
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                self._reset_session_after_error()
                if self.retry_sleep_seconds > 0:
                    sleep(float(self.retry_sleep_seconds) * attempt)

        raise RuntimeError(f"{description} failed after {attempts} attempts: {last_error}") from last_error

    def __enter__(self) -> "BaoStockClient":
        self.login()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.logout()

    def query_kline(
        self,
        symbol: str,
        *,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
    ) -> pd.DataFrame:
        """
        Query K-line data from BaoStock.

        Args:
            symbol:
                BaoStock code, for example sh.600000 or sz.000001.
            start_date:
                YYYY-MM-DD.
            end_date:
                YYYY-MM-DD.
            frequency:
                BaoStock frequency: d, w, m, 5, 15, 30, or 60.
            adjustflag:
                BaoStock adjustment flag: 1=post, 2=pre, 3=none.

        Notes:
            PocketAgent downloads only core K-line fields here. pctChg is
            calculated by data_layer after storage, and non-universal facts such
            as turnover are downloaded through extension methods.
        """
        frequency = str(frequency or "d").lower()
        if frequency not in BAOSTOCK_FREQUENCY_TO_STORAGE:
            raise ValueError(f"Unsupported BaoStock frequency: {frequency}")

        fields = get_baostock_kline_fields(frequency)

        effective_adjustflag = str(adjustflag or "3")
        if effective_adjustflag not in BAOSTOCK_ADJUST_TO_NAME:
            raise ValueError(
                f"Unsupported BaoStock adjustflag: {effective_adjustflag}. "
                "Expected one of: 1=post, 2=pre, 3=none."
            )

        def run_query() -> pd.DataFrame:
            import baostock as bs

            result = bs.query_history_k_data_plus(
                symbol,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                adjustflag=effective_adjustflag,
            )

            if result.error_code != "0":
                raise RuntimeError(
                    f"BaoStock query failed for {symbol} frequency={frequency} "
                    f"adjustflag={effective_adjustflag}: "
                    f"{result.error_code}, {result.error_msg}"
                )

            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())

            storage_freq = BAOSTOCK_FREQUENCY_TO_STORAGE[frequency]
            adjust_name = BAOSTOCK_ADJUST_TO_NAME[effective_adjustflag]

            if not rows:
                return empty_kline_dataframe(
                    symbol=symbol,
                    storage_freq=storage_freq,
                    adjust_name=adjust_name,
                )

            raw = pd.DataFrame(rows, columns=result.fields)

            return normalize_kline_dataframe(
                raw,
                symbol=from_baostock_symbol(symbol),
                source="baostock",
                freq=storage_freq,
                adjust=adjust_name,
            )

        return self._with_retries(
            f"BaoStock K-line query for {symbol} frequency={frequency} adjustflag={effective_adjustflag}",
            run_query,
        )

    def query_daily_kline(
        self,
        symbol: str,
        *,
        start_date: str,
        end_date: str,
        adjustflag: str = "3",
    ) -> pd.DataFrame:
        """Query daily K-line data from BaoStock."""
        return self.query_kline(
            symbol,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
        )

    def query_many_daily_kline(
        self,
        symbols: Iterable[str],
        *,
        start_date: str,
        end_date: str,
        adjustflag: str = "3",
    ) -> dict[str, pd.DataFrame]:
        """Query daily K-line data for multiple symbols."""
        data: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            data[symbol] = self.query_daily_kline(
                symbol,
                start_date=start_date,
                end_date=end_date,
                adjustflag=adjustflag,
            )
        return data

    def query_trade_calendar(
        self,
        *,
        start_date: str,
        end_date: str,
        exchange: str = "CN",
    ) -> pd.DataFrame:
        """Query trading dates from BaoStock."""
        def run_query() -> pd.DataFrame:
            import baostock as bs

            result = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            if result.error_code != "0":
                raise RuntimeError(
                    f"BaoStock trade calendar query failed: {result.error_code}, {result.error_msg}"
                )

            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())

            if rows:
                raw = pd.DataFrame(rows, columns=result.fields)
            else:
                raw = pd.DataFrame(columns=["calendar_date", "is_trading_day"])

            return normalize_trade_calendar_dataframe(raw, source="baostock", exchange=exchange)

        return self._with_retries("BaoStock trade calendar query", run_query)

    def query_daily_liquidity(
        self,
        symbol: str,
        *,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Query daily turnover and ST status extension facts."""
        def run_query() -> pd.DataFrame:
            import baostock as bs

            result = bs.query_history_k_data_plus(
                symbol,
                "date,code,turn,isST",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )

            if result.error_code != "0":
                raise RuntimeError(
                    f"BaoStock liquidity query failed for {symbol}: {result.error_code}, {result.error_msg}"
                )

            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())

            if rows:
                raw = pd.DataFrame(rows, columns=result.fields)
            else:
                raw = pd.DataFrame(columns=["date", "code", "turn", "isST"])

            normalized_symbol = from_baostock_symbol(symbol)
            liquidity = normalize_stock_liquidity_dataframe(
                raw, symbol=normalized_symbol, source="baostock"
            )
            status = normalize_stock_status_dataframe(
                raw, symbol=normalized_symbol, source="baostock"
            )
            return liquidity.merge(
                status[["symbol", "date", "is_st"]],
                on=["symbol", "date"],
                how="left",
            )

        return self._with_retries(f"BaoStock liquidity query for {symbol}", run_query)

def _remove_bom(value: object) -> str:
    """Remove UTF-8 BOM and surrounding whitespace from symbol-like text."""
    return str(value).replace("\ufeff", "").replace("\uFEFF", "").strip()


def _infer_baostock_exchange(code: str) -> str:
    code = _remove_bom(code)
    if code.startswith(("6", "9")):
        return "sh"
    return "sz"


def to_baostock_symbol(symbol: str) -> str:
    """
    Convert internal symbol format to BaoStock format.

    Supported:
        000001.SZ -> sz.000001
        600000.SH -> sh.600000
        sz.000001 -> sz.000001
        sh.600000 -> sh.600000
        000001    -> sz.000001
        600000    -> sh.600000
    """
    value = _remove_bom(symbol)
    if not value:
        raise ValueError("Empty symbol.")

    lower = _remove_bom(value.lower())

    if lower.startswith(("sh.", "sz.")):
        exchange, code = lower.split(".", 1)
        code = _remove_bom(code)
        if len(code) != 6 or not code.isdigit():
            raise ValueError(f"Invalid BaoStock symbol code: {symbol!r}")
        return f"{exchange}.{code}"

    if "." in value:
        code, exchange = value.split(".", 1)
        code = _remove_bom(code).upper()
        exchange = _remove_bom(exchange).upper()
        if len(code) != 6 or not code.isdigit():
            raise ValueError(f"Invalid stock symbol code: {symbol!r}")
        if exchange in {"SH", "SSE"}:
            return f"sh.{code}"
        if exchange in {"SZ", "SZSE"}:
            return f"sz.{code}"
        raise ValueError(f"Unsupported stock exchange in symbol: {symbol!r}")

    code = _remove_bom(value)
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"Invalid stock symbol: {symbol!r}")
    return f"{_infer_baostock_exchange(code)}.{code}"


def from_baostock_symbol(symbol: str) -> str:
    """
    Convert BaoStock format into common A-share format.

    Examples:
        sz.000001 -> 000001.SZ
        sh.600000 -> 600000.SH
    """
    value = _remove_bom(symbol).lower()
    if value.startswith("sz."):
        return _remove_bom(value.replace("sz.", "")).upper() + ".SZ"
    if value.startswith("sh."):
        return _remove_bom(value.replace("sh.", "")).upper() + ".SH"
    return _remove_bom(symbol).upper()
