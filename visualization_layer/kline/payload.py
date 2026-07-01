from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_layer.storage.duckdb_storage import (
    load_kline_page_from_duckdb,
    load_stock_liquidity_daily_from_duckdb,
    load_stock_status_daily_from_duckdb,
)


DEFAULT_VISIBLE_WINDOW = 240
MAX_VISUALIZATION_WINDOW = 9600


def build_kline_chart_payload(
    db_path: str | Path,
    *,
    symbol: str,
    freq: str,
    adjust: str,
    limit: int = DEFAULT_VISIBLE_WINDOW,
    offset: int | None = None,
) -> dict[str, Any]:
    """Build one chart-ready window on the symbol's absolute K-line timeline."""
    clean_symbol = str(symbol or "").strip().upper()
    clean_freq = str(freq or "daily").strip()
    clean_adjust = str(adjust or "none").strip()
    visible_window = min(MAX_VISUALIZATION_WINDOW, max(1, int(limit or DEFAULT_VISIBLE_WINDOW)))

    df, total_rows = load_kline_page_from_duckdb(
        db_path,
        symbol=clean_symbol,
        freq=clean_freq,
        adjust=clean_adjust,
        offset=offset,
        limit=visible_window,
        max_limit=MAX_VISUALIZATION_WINDOW,
    )
    if df.empty:
        return _empty_payload(
            clean_symbol,
            clean_freq,
            clean_adjust,
            visible_window,
            total_rows=total_rows,
            offset=max(0, int(offset or 0)),
        )

    df = df.sort_values("datetime").reset_index(drop=True)
    resolved_offset = max(0, total_rows - len(df)) if offset is None else max(0, int(offset))
    df = _merge_daily_extensions(db_path, df, symbol=clean_symbol, freq=clean_freq)
    bars = [_bar_record(row) for row in df.to_dict(orient="records")]

    latest = bars[-1] if bars else None
    previous = bars[-2] if len(bars) >= 2 else None
    summary = _build_summary(clean_symbol, clean_freq, clean_adjust, bars, latest, previous)

    return {
        "symbol": clean_symbol,
        "freq": clean_freq,
        "adjust": clean_adjust,
        "limit": visible_window,
        "offset": resolved_offset,
        "rows": len(bars),
        "total_rows": total_rows,
        "has_more_before": resolved_offset > 0,
        "has_more_after": resolved_offset + len(bars) < total_rows,
        "start_datetime": bars[0]["datetime"] if bars else None,
        "end_datetime": bars[-1]["datetime"] if bars else None,
        "summary": summary,
        "bars": bars,
        "features": feature_placeholder_contract(),
    }


def feature_placeholder_contract() -> dict[str, Any]:
    """Stable frontend contract for future feature-layer overlays."""
    return {
        "status": "pending_feature_layer",
        "endpoint": "/api/feature/visualization-overlays",
        "price_overlays": [
            {"id": "ema", "label": "EMA", "status": "pending_feature_layer", "series": []},
            {"id": "ma", "label": "MA", "status": "pending_feature_layer", "series": []},
        ],
        "indicator_panels": [
            {"id": "macd", "label": "MACD", "status": "pending_feature_layer", "series": []},
            {"id": "kdj", "label": "KDJ", "status": "pending_feature_layer", "series": []},
            {"id": "rsi", "label": "RSI", "status": "pending_feature_layer", "series": []},
        ],
    }


def _merge_daily_extensions(db_path: str | Path, df: pd.DataFrame, *, symbol: str, freq: str) -> pd.DataFrame:
    if str(freq).lower() != "daily" or df.empty:
        return df

    result = df.copy()
    result["turn"] = None
    result["is_st"] = None
    liquidity = load_stock_liquidity_daily_from_duckdb(
        db_path,
        symbol=symbol,
        start=str(result["datetime"].min())[:10],
        end=str(result["datetime"].max())[:10],
    )
    if not liquidity.empty:
        turn_by_date = dict(
            zip(
                liquidity["date"].astype(str).str.slice(0, 10),
                liquidity["turn"],
            )
        )
        result["turn"] = result["datetime"].astype(str).str.slice(0, 10).map(turn_by_date)
    status = load_stock_status_daily_from_duckdb(
        db_path,
        symbol=symbol,
        start=str(result["datetime"].min())[:10],
        end=str(result["datetime"].max())[:10],
    )
    if not status.empty:
        status_by_date = dict(
            zip(status["date"].astype(str).str.slice(0, 10), status["is_st"])
        )
        result["is_st"] = result["datetime"].astype(str).str.slice(0, 10).map(status_by_date)
    return result


def _bar_record(row: dict[str, Any]) -> dict[str, Any]:
    dt = pd.to_datetime(row.get("datetime"), errors="coerce")
    return {
        "datetime": None if pd.isna(dt) else dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": _number_or_none(row.get("open")),
        "high": _number_or_none(row.get("high")),
        "low": _number_or_none(row.get("low")),
        "close": _number_or_none(row.get("close")),
        "volume": _number_or_none(row.get("volume")),
        "amount": _number_or_none(row.get("amount")),
        "pctChg": _number_or_none(row.get("pctChg")),
        "turn": _number_or_none(row.get("turn")),
        "is_st": None if pd.isna(row.get("is_st")) else bool(row.get("is_st")),
    }


def _build_summary(
    symbol: str,
    freq: str,
    adjust: str,
    bars: list[dict[str, Any]],
    latest: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    latest_close = latest.get("close") if latest else None
    previous_close = previous.get("close") if previous else (latest.get("open") if latest else None)
    change = None
    if latest_close is not None and previous_close not in (None, 0):
        change = latest_close - previous_close

    pct_chg = latest.get("pctChg") if latest else None
    if pct_chg is None and change is not None and previous_close:
        pct_chg = change / previous_close

    highs = [item["high"] for item in bars if item.get("high") is not None]
    lows = [item["low"] for item in bars if item.get("low") is not None]

    return {
        "symbol": symbol,
        "freq": freq,
        "adjust": adjust,
        "latest_datetime": latest.get("datetime") if latest else None,
        "latest_close": latest_close,
        "previous_close": previous_close,
        "change": change,
        "pctChg": pct_chg,
        "window_high": max(highs) if highs else None,
        "window_low": min(lows) if lows else None,
    }


def _empty_payload(
    symbol: str,
    freq: str,
    adjust: str,
    limit: int,
    *,
    total_rows: int = 0,
    offset: int = 0,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "freq": freq,
        "adjust": adjust,
        "limit": limit,
        "offset": offset,
        "rows": 0,
        "total_rows": total_rows,
        "has_more_before": offset > 0,
        "has_more_after": offset < total_rows,
        "start_datetime": None,
        "end_datetime": None,
        "summary": _build_summary(symbol, freq, adjust, [], None, None),
        "bars": [],
        "features": feature_placeholder_contract(),
    }


def _number_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
