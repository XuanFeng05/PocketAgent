from __future__ import annotations

from pathlib import Path
from typing import Any

from visualization_layer.kline.payload import build_kline_chart_payload
from app_layer.backend.feature_controller import feature_visualization_payload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = "runtime_layer/data"


def resolve_project_path(path: str | Path | None, default: str | Path | None = None) -> Path:
    raw = Path(path or default or "")
    if raw.is_absolute():
        return raw
    return PROJECT_ROOT / raw


def kline_chart_payload(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Symbol is required.")

    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    chart = build_kline_chart_payload(
        db_path,
        symbol=symbol,
        freq=str(payload.get("freq") or "daily"),
        adjust=str(payload.get("adjust") or "none"),
        limit=int(payload.get("limit") or 240),
        offset=(
            None
            if payload.get("offset") in (None, "")
            else int(payload["offset"])
        ),
    )
    chart["features"] = feature_visualization_payload(
        {
            "db_path": db_path,
            "symbol": symbol,
            "freq": chart["freq"],
            "adjust": chart["adjust"],
            "offset": chart["offset"],
            "limit": max(1, chart["rows"]),
        }
    )
    return chart
