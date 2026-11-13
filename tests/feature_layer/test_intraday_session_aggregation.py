from pathlib import Path

import duckdb
import pandas as pd
import pytest

from data_layer.storage.duckdb_storage import save_kline_to_duckdb
from feature_layer.builders.aggregation import aggregate_ohlcv_from_base
from feature_layer.builders.materialize import (
    DERIVED_SHARD_RULE_VERSION,
    _aggregate_derived_frame,
    materialize_derived_bars,
)


def bars(times: list[str], *, freq: str = "5min") -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame({
        "symbol": ["000001.SZ"] * n,
        "datetime": pd.to_datetime(["2026-10-30 " + t for t in times]),
        "open": [10.0 + i for i in range(n)],
        "high": [11.0 + i for i in range(n)],
        "low": [9.0 + i for i in range(n)],
        "close": [10.5 + i for i in range(n)],
        "volume": [100.0] * n, "amount": [1000.0] * n, "pctChg": [0.0] * n,
        "source": ["fixture"] * n, "freq": [freq] * n, "adjust": ["none"] * n,
    })


def test_missing_bar_does_not_shift_later_windows_or_cross_lunch() -> None:
    frame = bars(["09:35", "09:45", "09:50", "09:55", "10:00", "11:25", "11:30", "13:05", "13:10", "13:15"])
    result = aggregate_ohlcv_from_base(frame, "15min")
    assert result["bar_end"].dt.strftime("%H:%M").tolist() == ["09:45", "10:00", "11:30", "13:15"]
    assert result["source_rows"].tolist() == [2, 3, 2, 3]
    assert result["is_partial"].tolist() == [True, False, True, False]
    assert result["open"].tolist() == [10.0, 12.0, 15.0, 17.0]
    assert result["close"].tolist() == [11.5, 14.5, 16.5, 19.5]


def test_partial_asof_bar_uses_only_completed_base_bars() -> None:
    frame = bars(["09:35", "09:45", "09:50", "09:55", "10:00"])
    result = aggregate_ohlcv_from_base(frame, "15min", decision_time="2026-10-30 09:50")
    assert result["bar_end"].dt.strftime("%H:%M").tolist() == ["09:45", "09:50"]
    assert result["source_rows"].tolist() == [2, 1]
    assert result.iloc[-1]["close"] == 12.5


@pytest.mark.parametrize("base,target,times", [
    ("5min", "15min", ["09:35", "09:45", "09:50", "09:55", "10:00", "11:25", "11:30", "13:05", "13:10", "13:15"]),
    ("30min", "60min", ["10:00", "11:00", "11:30", "13:30", "14:00", "15:00"]),
])
def test_sql_and_shard_python_match_clock_aggregation(tmp_path: Path, base, target, times) -> None:
    frame = bars(times, freq=base)
    db = tmp_path / "market.duckdb"
    save_kline_to_duckdb(frame, db)
    summary = materialize_derived_bars(db, symbols=["000001.SZ"], base_freq=base, targets=[target])
    assert not summary["status"].eq("failed").any(), summary.to_dict("records")
    with duckdb.connect(str(db), read_only=True) as conn:
        sql = conn.execute("SELECT datetime, open, high, low, close, volume, amount FROM kline_bars WHERE freq = ? ORDER BY datetime", [target]).fetchdf()
    python = _aggregate_derived_frame(frame, base_freq=base, target_freq=target, source="derived_" + base)
    generic = aggregate_ohlcv_from_base(frame, target, base_freq=base).rename(columns={"bar_end": "datetime"})
    columns = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    pd.testing.assert_frame_equal(sql, python[columns].reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(sql, generic[columns].reset_index(drop=True), check_dtype=False)
    second = materialize_derived_bars(db, symbols=["000001.SZ"], base_freq=base, targets=[target])
    assert second.loc[second["symbol"].eq("000001.SZ"), "status"].iloc[0] == "skipped"


def test_old_manifest_cannot_skip_new_aggregation_rule(tmp_path: Path) -> None:
    frame = bars(["09:35", "09:45", "09:50", "09:55", "10:00"])
    db = tmp_path / "market.duckdb"
    save_kline_to_duckdb(frame, db)
    materialize_derived_bars(db, symbols=["000001.SZ"], targets=["15min"])
    with duckdb.connect(str(db)) as conn:
        conn.execute("UPDATE derived_bar_manifest SET rule_version = 'old_row_count_rule'")
        conn.execute("UPDATE kline_bars SET close=999 WHERE freq='15min'")
    result = materialize_derived_bars(db, symbols=["000001.SZ"], targets=["15min"])
    assert result.loc[result["symbol"].eq("000001.SZ"), "status"].iloc[0] == "ok"
    with duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute("SELECT MAX(close) FROM kline_bars WHERE freq='15min'").fetchone()[0] == 14.5
        assert conn.execute("SELECT rule_version FROM derived_bar_manifest").fetchone()[0] == DERIVED_SHARD_RULE_VERSION


@pytest.mark.parametrize("time", ["09:30", "09:36", "12:00", "13:00", "15:05"])
def test_off_grid_input_is_not_silently_folded_into_a_session(time) -> None:
    frame = bars([time])
    with pytest.raises(ValueError, match="trading-session grid"):
        aggregate_ohlcv_from_base(frame, "15min")
    with pytest.raises(ValueError, match="trading-session grid"):
        _aggregate_derived_frame(frame, base_freq="5min", target_freq="15min", source="fixture")
