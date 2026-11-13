import pandas as pd
import pytest

from feature_layer.datasets import compact


@pytest.mark.parametrize("decision_time", ["2026-10-15 09:35", "2026-10-30 09:35", "2026-10-30 15:00"])
def test_fast_monthly_index_excludes_current_month_final_and_partial_bars(decision_time: str) -> None:
    decisions = pd.DataFrame([{
        "decision_id": "test", "symbol": "000001.SZ", "adjust": "none",
        "decision_time": pd.Timestamp(decision_time), "visible_bar_end": pd.Timestamp(decision_time),
        "stage": "continuous",
    }])
    stable = pd.DataFrame({
        "symbol": ["000001.SZ"] * 3,
        "datetime": pd.to_datetime(["2026-09-30", "2026-10-15", "2026-10-30"]),
    })
    index, _ = compact._build_decision_index_fast(
        decisions, stable_by_freq={"monthly": stable}, frequencies=("monthly",),
        sequence_windows={"monthly": 12}, trade_freq="5min",
    )
    assert index.iloc[0]["stable_end_datetime"] == pd.Timestamp("2026-09-30")
    assert index.iloc[0]["valid_rows"] == 2  # One past month plus current as-of snapshot.
    assert bool(index.iloc[0]["snapshot_required"])


def test_slow_monthly_index_uses_identical_previous_month_boundary(monkeypatch) -> None:
    decision_time = pd.Timestamp("2026-10-30 09:35")
    decisions = pd.DataFrame([{
        "decision_id": "test", "symbol": "000001.SZ", "adjust": "none",
        "decision_time": decision_time, "visible_bar_end": decision_time, "stage": "continuous",
    }])
    stable = pd.DataFrame({"symbol": ["000001.SZ"] * 2, "datetime": pd.to_datetime(["2026-09-30", "2026-10-30"])})
    # Snapshot generation is separate from selecting stable history.
    monkeypatch.setattr(compact, "_decision_snapshot", lambda *args, **kwargs: None)
    index, _ = compact._build_decision_index(
        decisions, base=pd.DataFrame(), daily=pd.DataFrame(), stable_by_freq={"monthly": stable},
        frequencies=("monthly",), sequence_windows={"monthly": 12}, trade_freq="5min",
    )
    assert index.iloc[0]["stable_end_datetime"] == pd.Timestamp("2026-09-30")
    assert index.iloc[0]["valid_rows"] == 1
