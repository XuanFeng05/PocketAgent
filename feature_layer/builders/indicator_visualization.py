from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_layer.storage.duckdb_storage import load_kline_page_from_duckdb
from feature_layer.builders.indicator_calculations import (
    calculate_indicator,
    clean_numeric_series,
)
from feature_layer.indicator_registry import (
    fields_for_indicator,
    indicator_lookback,
    load_indicator_specs,
)


MAX_VISUALIZATION_ROWS = 9600
WARMUP_MULTIPLIER = 4


def build_indicator_visualization_payload(
    db_path: str | Path,
    *,
    symbol: str,
    freq: str,
    adjust: str,
    offset: int | None,
    limit: int,
) -> dict[str, Any]:
    """Build chart series aligned one-to-one with a K-line visualization window."""
    clean_symbol = str(symbol or "").strip().upper()
    clean_freq = str(freq or "daily").strip()
    clean_adjust = str(adjust or "none").strip()
    visible_limit = min(MAX_VISUALIZATION_ROWS, max(1, int(limit or 1)))
    indicators = tuple(
        item
        for item in load_indicator_specs()
        if item.enabled and clean_freq in item.frequencies
    )
    longest_lookback = max((indicator_lookback(item) for item in indicators), default=1)
    desired_warmup = longest_lookback * WARMUP_MULTIPLIER

    if offset is None:
        loaded_offset = None
        fetch_limit = visible_limit + desired_warmup
    else:
        visible_offset = max(0, int(offset))
        loaded_offset = max(0, visible_offset - desired_warmup)
        fetch_limit = visible_limit + (visible_offset - loaded_offset)

    bars, total_rows = load_kline_page_from_duckdb(
        db_path,
        symbol=clean_symbol,
        freq=clean_freq,
        adjust=clean_adjust,
        offset=loaded_offset,
        limit=fetch_limit,
        max_limit=MAX_VISUALIZATION_ROWS + desired_warmup,
    )
    if bars.empty:
        return _empty_payload(clean_symbol, clean_freq, clean_adjust, offset, total_rows)

    bars = bars.sort_values("datetime").reset_index(drop=True)
    actual_loaded_offset = max(0, total_rows - len(bars)) if loaded_offset is None else loaded_offset
    visible_offset = max(0, total_rows - visible_limit) if offset is None else max(0, int(offset))
    skip = max(0, visible_offset - actual_loaded_offset)
    visible_rows = min(visible_limit, max(0, len(bars) - skip))
    stop = skip + visible_rows
    if not indicators:
        return {
            "status": "unsupported_frequency",
            "symbol": clean_symbol,
            "freq": clean_freq,
            "adjust": clean_adjust,
            "offset": visible_offset,
            "rows": visible_rows,
            "warmup_rows": skip,
            "indicators": [],
        }

    high = clean_numeric_series(bars["high"])
    low = clean_numeric_series(bars["low"])
    close = clean_numeric_series(bars["close"])
    volume = clean_numeric_series(bars["volume"]).fillna(0.0)
    payload_indicators: list[dict[str, Any]] = []
    for index, indicator in enumerate(indicators):
        calculation = calculate_indicator(
            indicator,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        clip_by_name = {
            field.name: field.clip for field in fields_for_indicator(indicator)
        }
        model_series = []
        for name, values in calculation.model_fields.items():
            clean = clean_numeric_series(values)
            clip = clip_by_name.get(name)
            if clip is not None:
                clean = clean.clip(clip[0], clip[1])
            clean = clean.fillna(0.0)
            model_series.append(
                {
                    "id": name,
                    "label": _model_label(indicator.kind, name),
                    "clip": list(clip) if clip is not None else None,
                    "values": _values(clean.iloc[skip:stop]),
                }
            )

        display_series = _display_series(
            indicator.kind,
            indicator.id,
            indicator.params,
            calculation.display_fields,
            skip=skip,
            stop=stop,
            palette_index=index,
        )
        lookback = indicator_lookback(indicator)
        coverage = [
            min(1.0, (visible_offset + row_index + 1) / float(max(1, lookback)))
            for row_index in range(visible_rows)
        ]
        payload_indicators.append(
            {
                "id": indicator.id,
                "label": _indicator_label(indicator.kind, indicator.id),
                "kind": indicator.kind,
                "render_target": indicator.render_target,
                "default_visible": indicator.default_visible,
                "params": dict(indicator.params),
                "lookback": lookback,
                "axis": _axis_contract(indicator.kind),
                "display_series": display_series,
                "model_series": model_series,
                "coverage": coverage,
            }
        )

    return {
        "status": "ready",
        "symbol": clean_symbol,
        "freq": clean_freq,
        "adjust": clean_adjust,
        "offset": visible_offset,
        "rows": visible_rows,
        "warmup_rows": skip,
        "indicators": payload_indicators,
    }


def _display_series(
    kind: str,
    indicator_id: str,
    params: dict[str, int],
    fields: dict[str, pd.Series],
    *,
    skip: int,
    stop: int,
    palette_index: int,
) -> list[dict[str, Any]]:
    if kind == "ema_channel":
        colors = _ema_colors(params, palette_index)
        return [
            {
                "id": f"{indicator_id}_fast",
                "label": f"EMA{params['fast']}",
                "style": "line",
                "color": colors[0],
                "values": _values(clean_numeric_series(fields[f"{indicator_id}_fast"]).iloc[skip:stop]),
            },
            {
                "id": f"{indicator_id}_slow",
                "label": f"EMA{params['slow']}",
                "style": "line",
                "color": colors[1],
                "values": _values(clean_numeric_series(fields[f"{indicator_id}_slow"]).iloc[skip:stop]),
            },
        ]

    if kind == "efi":
        return [
            {
                "id": f"{indicator_id}2",
                "label": f"EFI{params['fast']}",
                "style": "line",
                "color": "#d85cc6",
                "values": _values(clean_numeric_series(fields[f"{indicator_id}2"]).iloc[skip:stop]),
            },
            {
                "id": f"{indicator_id}13",
                "label": f"EFI{params['slow']}",
                "style": "line",
                "color": "#2bd7a7",
                "values": _values(clean_numeric_series(fields[f"{indicator_id}13"]).iloc[skip:stop]),
            },
        ]

    model_names = list(fields)
    styles = {
        "macd": ["line", "line", "histogram"],
        "kd": ["line", "line", "histogram"],
    }[kind]
    colors = {
        "macd": ["#ff8a00", "#00b9f2", "#ff335f"],
        "kd": ["#ff8a00", "#00b9f2", "#d85cc6"],
    }[kind]
    return [
        {
            "id": name,
            "label": _model_label(kind, name),
            "style": styles[index],
            "color": colors[index],
            "model_field": name,
        }
        for index, name in enumerate(model_names)
    ]


def _axis_contract(kind: str) -> dict[str, Any]:
    if kind == "ema_channel":
        return {"mode": "price", "reference_lines": []}
    if kind == "kd":
        return {"mode": "bounded", "min": -1.0, "max": 1.0, "reference_lines": [0.0, 0.2, 0.8]}
    if kind == "efi":
        return {"mode": "zero_centered", "reference_lines": []}
    return {"mode": "zero_centered", "reference_lines": [0.0]}


def _ema_colors(params: dict[str, int], index: int) -> tuple[str, str]:
    known = {
        (13, 21): ("#00c2ff", "#e45acb"),
        (144, 169): ("#1687ff", "#2bd7a7"),
    }
    palette = [
        ("#00c2ff", "#e45acb"),
        ("#1687ff", "#2bd7a7"),
        ("#ff8a00", "#f7d154"),
    ]
    return known.get((params["fast"], params["slow"]), palette[index % len(palette)])


def _indicator_label(kind: str, indicator_id: str) -> str:
    if kind == "ema_channel":
        return f"EMA {indicator_id.replace('_', '/')}"
    return kind.upper()


def _model_label(kind: str, name: str) -> str:
    suffixes = {
        "macd": {"dif_pct": "DIF", "dea_pct": "DEA", "hist_pct": "HIST"},
        "kd": {"k_norm": "K", "d_norm": "D", "diff_norm": "K-D"},
        "efi": {"13_norm": "EFI13", "2_norm": "EFI2"},
        "ema_channel": {
            "position": "Position",
            "gap": "Gap",
            "width": "Width",
            "slope": "Slope",
        },
    }[kind]
    for suffix, label in sorted(suffixes.items(), key=lambda item: -len(item[0])):
        if name.endswith(suffix):
            return label
    return name


def _values(series: pd.Series) -> list[float | None]:
    result: list[float | None] = []
    for value in series.tolist():
        result.append(None if value is None or pd.isna(value) else float(value))
    return result


def _empty_payload(
    symbol: str,
    freq: str,
    adjust: str,
    offset: int | None,
    total_rows: int,
) -> dict[str, Any]:
    return {
        "status": "no_data",
        "symbol": symbol,
        "freq": freq,
        "adjust": adjust,
        "offset": max(0, int(offset or 0)),
        "rows": 0,
        "total_rows": total_rows,
        "warmup_rows": 0,
        "indicators": [],
    }
