from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _is_missing(value: Any) -> bool:
    """Return True for pandas/numpy missing values and non-finite floats."""
    try:
        if value is pd.NA or pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    return False


def to_jsonable(value: Any) -> Any:
    """Convert common Python/pandas objects into strict JSON-safe values.

    Important: NaN/inf must be handled before the generic float branch. Python's
    json.dumps can otherwise emit NaN, which browsers reject as invalid JSON.
    """
    if value is None or _is_missing(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return dataframe_to_records(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return str(value)


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to strict JSON records without losing intraday time."""
    if df.empty:
        return []
    safe = df.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = safe.to_dict(orient="records")
    return [to_jsonable(row) for row in records]


def dumps_json(payload: Any) -> bytes:
    return json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
