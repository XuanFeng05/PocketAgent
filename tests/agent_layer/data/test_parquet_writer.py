from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from agent_layer.data.tensor_store import write_parquet_frame


def test_duckdb_parquet_writer_does_not_require_optional_pandas_engine(tmp_path: Path) -> None:
    output = tmp_path / "cache.parquet"
    write_parquet_frame(
        output,
        pd.DataFrame({"decision": [1, 2], "symbol": ["000001.SZ", "002241.SZ"]}),
    )

    with duckdb.connect(":memory:") as connection:
        rows = connection.execute(
            "SELECT decision, symbol FROM read_parquet(?) ORDER BY decision",
            [str(output)],
        ).fetchall()

    assert rows == [(1, "000001.SZ"), (2, "002241.SZ")]
