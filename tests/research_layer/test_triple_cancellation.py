"""Cancellation is observable, retains evidence, and never masquerades as failure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_layer.triple_runner import run_triple_experiment
from research_layer.triple_study import run_triple_study
from research_layer.triple_validation import run_triple_validation


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def inputs(tmp_path):
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    x = np.arange(len(dates))
    close = 10 + .025*x + .4*np.sin(x/3)
    opening = np.r_[close[0], close[:-1]]
    parts = []
    for symbol in ["000001.SZ", "600036.SH"]:
        parts.append(pd.DataFrame({"symbol": symbol, "datetime": dates,
                                   "open": opening, "high": np.maximum(opening, close)+.2,
                                   "low": np.minimum(opening, close)-.2, "close": close,
                                   "volume": 1000000., "freq": "daily", "adjust": "pre"}))
    source = tmp_path / "prices.csv"
    pd.concat(parts).to_csv(source, index=False)
    calendar = pd.DataFrame({"date": pd.date_range("2019-12-30", "2020-07-05")})
    calendar["is_trading_day"] = calendar["date"].dt.weekday < 5
    cal_path = tmp_path / "calendar.csv"
    calendar.to_csv(cal_path, index=False)
    return {"source_type": "csv", "source_path": str(source), "calendar_path": str(cal_path),
            "symbols": ["000001.SZ", "600036.SH"], "start": "2020-02-03", "end": "2020-06-30",
            "settings": {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "warmup_weeks": 2},
            "cost_scenarios_bps": [0, 10]}


def invoke(kind, config, output, **kwargs):
    if kind == "batch":
        return run_triple_experiment(config, output, **kwargs)
    if kind == "annual":
        config = {k: v for k, v in config.items() if k not in {"start", "end"}}
        config["research_scope"] = "exploratory_validation"
        windows = [{"label": "first", "start": "2020-02-03", "end": "2020-04-30"},
                   {"label": "second", "start": "2020-05-01", "end": "2020-06-30"}]
        return run_triple_validation(config, windows, output, **kwargs)
    config = {**config, "research_scope": "exploratory_study"}
    cases = [{"id": "primary", "label": "完整", "settings": {}, "variant": "full"},
             {"id": "comparison", "label": "邻域", "settings": {"force_span": 3}, "variant": "full"}]
    return run_triple_study(config, cases, output, **kwargs)


@pytest.mark.parametrize("kind,total", [("batch", 4), ("annual", 8), ("study", 8)])
def test_pre_cancel_has_persisted_terminal_status_and_parent_start_event(inputs, tmp_path, kind, total):
    events = []
    with pytest.raises(InterruptedError, match="cancelled"):
        invoke(kind, inputs, tmp_path / "runs", cancel_check=lambda: True, progress_callback=events.append)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "run_started"
    assert event["completed"] == 0 and event["total"] == total
    root = Path(event["path"])
    assert root.name == event["run_id"]
    assert read(root / "status.json")["status"] == "cancelled"
    partial = read(root / "partial_result.json")
    assert partial["status"] == "cancelled"
    assert partial["recorded_tasks"] == 0
    assert partial["completed_tasks"] == partial["failed_tasks"] == 0
    assert partial["unfinished_tasks"] == total
    assert not list(root.rglob("result.json"))


@pytest.mark.parametrize("kind,total", [("batch", 4), ("annual", 8), ("study", 8)])
def test_cancel_after_first_task_preserves_it_and_does_not_run_next(inputs, tmp_path, kind, total):
    flag = [False]
    events = []
    source_before = Path(inputs["source_path"]).read_bytes()

    def progress(event):
        events.append(event)
        if event["event"] == "task_completed" and event["completed"] == 1:
            flag[0] = True

    with pytest.raises(InterruptedError):
        invoke(kind, inputs, tmp_path / "runs", cancel_check=lambda: flag[0], progress_callback=progress)
    parent = Path(events[0]["path"])
    # A child start event cannot replace the page's parent result path.
    assert all(e["path"] == str(parent) and e["run_id"] == parent.name for e in events)
    assert all(e["total"] == total for e in events)
    assert [e["completed"] for e in events] == sorted(e["completed"] for e in events)
    assert len([e for e in events if e["event"] == "task_completed"]) == 1
    partial = read(parent / "partial_result.json")
    assert partial["recorded_tasks"] == partial["completed_tasks"] == 1
    assert partial["failed_tasks"] == 0
    assert partial["unfinished_tasks"] == total-1
    assert read(parent / "status.json")["status"] == "cancelled"
    results = list(parent.rglob("outcomes/*/result.json"))
    assert len(results) == 1
    assert not list(parent.rglob("error.json"))
    result = read(results[0])
    assert result["strategy"]["symbol"] == "000001.SZ"
    if kind != "batch":
        child = Path(partial["active_child"]["path"])
        assert child.is_relative_to(parent)
        assert read(child / "status.json")["status"] == "cancelled"
        assert read(child / "partial_result.json")["completed_tasks"] == 1
        assert len(partial["active_outcomes"]) == 1
    assert Path(inputs["source_path"]).read_bytes() == source_before
    for manifest_path in parent.rglob("artifacts.json"):
        for relative, digest in read(manifest_path)["sha256"].items():
            assert hashlib.sha256((manifest_path.parent / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("kind,unit_event", [("annual", "window_completed"), ("study", "case_completed")])
def test_cancel_between_parent_units_retains_complete_first_unit(inputs, tmp_path, kind, unit_event):
    flag, events = [False], []

    def progress(event):
        events.append(event)
        if event["event"] == unit_event:
            flag[0] = True

    with pytest.raises(InterruptedError):
        invoke(kind, inputs, tmp_path / "runs", cancel_check=lambda: flag[0], progress_callback=progress)
    parent = Path(events[0]["path"])
    partial = read(parent / "partial_result.json")
    assert partial["completed_tasks"] == 4 and partial["unfinished_tasks"] == 4
    assert partial["failed_tasks"] == 0 and partial["active_child"] == {}
    assert partial["active_outcomes"] == []
    assert len(list(parent.rglob("outcomes/*/result.json"))) == 4
    assert len(list((parent / ("windows" if kind == "annual" else "cases")).iterdir())) == 1


def test_cancelled_batch_can_resume_without_mutating_completed_evidence(inputs, tmp_path):
    events, flag = [], [False]

    def progress(event):
        events.append(event)
        if event["event"] == "task_completed":
            flag[0] = True

    with pytest.raises(InterruptedError):
        run_triple_experiment(inputs, tmp_path / "runs", progress_callback=progress, cancel_check=lambda: flag[0])
    original = Path(events[0]["path"])
    before = {str(p.relative_to(original)): p.read_bytes() for p in original.rglob("*") if p.is_file()}
    resumed = run_triple_experiment(inputs, tmp_path / "runs", resume_from=original)
    assert resumed["status"] == "completed"
    assert resumed["summary"]["outcomes"][0]["reused"] is True
    assert all(not r["reused"] for r in resumed["summary"]["outcomes"][1:])
    assert all((original / p).read_bytes() == data for p, data in before.items())


def test_interrupted_error_inside_task_is_not_recorded_as_strategy_failure(inputs, tmp_path, monkeypatch):
    import research_layer.triple_runner as runner
    def interrupted(*args, **kwargs):
        raise InterruptedError("stop requested")
    monkeypatch.setattr(runner, "run_triple_screen", interrupted)
    events = []
    with pytest.raises(InterruptedError):
        runner.run_triple_experiment(inputs, tmp_path / "runs", progress_callback=events.append)
    root = Path(events[0]["path"])
    partial = read(root / "partial_result.json")
    assert partial["failed_tasks"] == 0
    assert partial["outcomes"] == []
    assert not list(root.rglob("error.json"))


@pytest.mark.parametrize("kind,total", [("batch", 4), ("annual", 8), ("study", 8)])
def test_successful_callbacks_count_tasks_without_changing_results(inputs, tmp_path, kind, total):
    control = invoke(kind, inputs, tmp_path / "control")
    events = []
    result = invoke(kind, inputs, tmp_path / "with_callbacks", progress_callback=events.append, cancel_check=lambda: False)
    assert result["status"] == control["status"] == "completed"
    assert events[-1]["completed"] == total
    assert all(event["total"] == total and event["path"] == result["path"] for event in events)
    assert len([e for e in events if e["event"] == "task_completed"]) == total
    if kind == "batch":
        assert result["summary"]["by_cost"] == control["summary"]["by_cost"]
    elif kind == "annual":
        assert result["summary"]["by_cost_window"] == control["summary"]["by_cost_window"]
    else:
        assert [r["by_cost"] for r in result["summary"]["cases"]] == [r["by_cost"] for r in control["summary"]["cases"]]
