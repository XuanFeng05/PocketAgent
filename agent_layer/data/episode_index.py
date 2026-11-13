from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from agent_layer.data.cache_schema import EPISODE_INDEX_NAME, safe_symbol_name
from agent_layer.data.tensor_store import write_parquet_frame


def build_episode_index_frame(symbol_summaries: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in symbol_summaries:
        decision_count = int(item.get("decision_count") or 0)
        if decision_count <= 0:
            continue
        rows.append(
            {
                "symbol": str(item.get("symbol") or ""),
                "safe_symbol": str(item.get("safe_symbol") or safe_symbol_name(str(item.get("symbol") or ""))),
                "first_decision_time": item.get("first_decision_time"),
                "last_decision_time": item.get("last_decision_time"),
                "decision_count": decision_count,
                "start_offset": 0,
                "end_offset": decision_count - 1,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "symbol",
            "safe_symbol",
            "first_decision_time",
            "last_decision_time",
            "decision_count",
            "start_offset",
            "end_offset",
        ),
    )


def write_episode_index(cache_dir: str | Path, frame: pd.DataFrame) -> Path:
    output = Path(cache_dir) / EPISODE_INDEX_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    return write_parquet_frame(output, frame)


def read_episode_index(cache_dir: str | Path) -> pd.DataFrame:
    path = Path(cache_dir) / EPISODE_INDEX_NAME
    with duckdb.connect(":memory:") as conn:
        return conn.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
