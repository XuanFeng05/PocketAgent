from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research_layer.triple_validation as validation


def local_config(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("local fixture", encoding="utf-8")
    calendar = tmp_path / "calendar.csv"
    calendar.write_text("local fixture", encoding="utf-8")
    return {"source_type": "csv", "source_path": str(source), "calendar_path": str(calendar),
            "symbols": ["000001.SZ", "600036.SH"], "research_scope": "exploratory_validation",
            "settings": {"warmup_weeks": 2, "macd_fast": 2, "macd_slow": 3, "macd_signal": 2},
            "cost_scenarios_bps": [0, 10]}


def windows():
    return [{"label": "2022-H1", "start": "2022-01-01", "end": "2022-06-30"},
            {"label": "2022-H2", "start": "2022-07-01", "end": "2022-12-31"}]


def output(config, destination, *, shortened=False, failed_symbol=None):
    rows = []
    for symbol in config["symbols"]:
        for cost in config["cost_scenarios_bps"]:
            if symbol == failed_symbol:
                rows.append({"symbol": symbol, "cost_bps": cost, "status": "failed", "error": "missing input"})
                continue
            start = config["start"]
            if shortened and symbol == config["symbols"][0]:
                start = (date.fromisoformat(start) + timedelta(days=1)).isoformat()
            rows.append({"symbol": symbol, "cost_bps": cost, "status": "completed", "start": start, "end": config["end"],
                         "metrics": {"total_return": .1 - cost/10000, "maximum_drawdown": .05},
                         "buy_hold_return": .06, "excess_return_vs_buy_hold": .04 - cost/10000})
    return {"run_id": "child", "path": str(destination), "status": "completed", "summary": {"outcomes": rows}}


def test_protocol_is_registered_and_hashed_before_any_child_runs(tmp_path, monkeypatch):
    config, registered_windows, calls = local_config(tmp_path), windows(), []
    original_config, original_windows = deepcopy(config), deepcopy(registered_windows)

    def child(fixed, destination):
        run = Path(destination).parents[1]
        protocol = json.loads((run / "protocol.json").read_text())
        assert protocol["research_scope"] == "exploratory_validation"
        assert (run / "protocol.sha256").read_text().strip() == hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest()
        assert protocol["windows"] == registered_windows
        assert fixed["research_scope"] == "exploratory"
        assert fixed["symbols"] == config["symbols"]
        calls.append(deepcopy(fixed))
        return output(fixed, destination, shortened=True)

    monkeypatch.setattr(validation, "run_triple_experiment", child)
    report = validation.run_triple_validation(config, registered_windows, tmp_path / "reports")
    assert report["status"] == "completed"
    assert len(calls) == 2
    assert calls[0]["settings"] == calls[1]["settings"]
    assert report["summary"]["total_tasks"] == 8
    assert len(report["summary"]["by_cost_window"]) == 4
    first = report["summary"]["windows"][0]["outcomes"][0]
    assert first["requested_start"] == "2022-01-01"
    assert first["effective_start"] == "2022-01-02"
    assert first["effective_range_differs"] is True
    assert report["summary"]["by_cost_window"][0]["effective_ranges_aligned"] is False
    assert "equity_curve" not in report["summary"]
    assert config == original_config and registered_windows == original_windows


@pytest.mark.parametrize("damage", ["unknown_window", "overlap", "same_boundary", "duplicate_label", "invalid_date", "reverse", "missing_label"])
def test_invalid_windows_rejected_before_execution(tmp_path, monkeypatch, damage):
    config, values = local_config(tmp_path), windows()
    if damage == "unknown_window": values[0]["parameters"] = {}
    if damage == "overlap": values[1]["start"] = "2022-06-15"
    if damage == "same_boundary": values[1]["start"] = "2022-06-30"
    if damage == "duplicate_label": values[1]["label"] = values[0]["label"]
    if damage == "invalid_date": values[0]["start"] = "2022-01-01T00:00:00"
    if damage == "reverse": values[0]["end"] = "2021-12-31"
    if damage == "missing_label": del values[0]["label"]
    monkeypatch.setattr(validation, "run_triple_experiment", lambda *_: pytest.fail("child must not run"))
    with pytest.raises(ValueError): validation.run_triple_validation(config, values, tmp_path / "reports")
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("override", [{"start": "2022-01-01"}, {"parameter_grid": {}}, {"research_scope": "holdout"},
                                      {"research_scope": "exploratory"}, {"source_type": "remote"},
                                      {"source_path": "https://example.com/data.csv"}, {"symbols": []},
                                      {"settings": {"unknown": 1}}])
def test_unknown_configuration_and_network_sources_rejected(tmp_path, override):
    with pytest.raises(ValueError): validation.normalize_validation_protocol({**local_config(tmp_path), **override}, windows())


def test_windows_sorted_chronologically_before_registration(tmp_path):
    protocol = validation.normalize_validation_protocol(local_config(tmp_path), list(reversed(windows())))
    assert protocol["windows"] == windows()


def test_failed_window_keeps_every_symbol_cost_task_and_continues(tmp_path, monkeypatch):
    config = local_config(tmp_path)
    calls = []

    def child(fixed, destination):
        calls.append(fixed["start"])
        if len(calls) == 1:
            raise RuntimeError("local data unavailable")
        return output(fixed, destination, failed_symbol="600036.SH")

    monkeypatch.setattr(validation, "run_triple_experiment", child)
    report = validation.run_triple_validation(config, windows(), tmp_path / "reports")
    assert report["status"] == "partial"
    assert report["summary"]["total_tasks"] == 8
    assert report["summary"]["completed_tasks"] == 2
    assert report["summary"]["failed_tasks"] == 6
    assert [w["status"] for w in report["summary"]["windows"]] == ["failed", "partial"]
    assert (Path(report["path"]) / "window_results/window_000.json").is_file()


@pytest.mark.parametrize("damage", ["missing_task", "duplicate_task", "outside_window"])
def test_child_accounting_and_effective_dates_are_checked(tmp_path, monkeypatch, damage):
    config = local_config(tmp_path)

    def child(fixed, destination):
        result = output(fixed, destination)
        rows = result["summary"]["outcomes"]
        if damage == "missing_task": rows.pop()
        if damage == "duplicate_task": rows.append(rows[0])
        if damage == "outside_window": rows[0]["start"] = "2000-01-01"
        return result

    monkeypatch.setattr(validation, "run_triple_experiment", child)
    report = validation.run_triple_validation(config, windows(), tmp_path / "reports")
    assert report["status"] == "failed"
    assert report["summary"]["failed_tasks"] == 8


def test_new_run_never_overwrites_previous_protocol_or_results(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "run_triple_experiment", output)
    config = local_config(tmp_path)
    first = validation.run_triple_validation(config, windows(), tmp_path / "reports")
    saved = {p.relative_to(first["path"]): p.read_bytes() for p in Path(first["path"]).rglob("*") if p.is_file()}
    second = validation.run_triple_validation(config, windows(), tmp_path / "reports")
    assert first["run_id"] != second["run_id"]
    assert all((Path(first["path"]) / path).read_bytes() == data for path, data in saved.items())
    manifest = json.loads((Path(second["path"]) / "artifacts.json").read_text())
    for relative, checksum in manifest["sha256"].items():
        assert hashlib.sha256((Path(second["path"]) / relative).read_bytes()).hexdigest() == checksum


def test_short_real_integration_uses_independent_accounts_and_future_perturbation(tmp_path):
    config = local_config(tmp_path)
    config.update(symbols=["000001.SZ"], cost_scenarios_bps=[0])
    days = pd.bdate_range("2020-01-01", "2020-06-19")
    x = np.arange(len(days))
    close = 10 + .02 * x + .6 * np.sin(x/3)
    opening = np.r_[close[0], close[:-1]]
    source = pd.DataFrame({"symbol": "000001.SZ", "datetime": days, "open": opening,
                           "high": np.maximum(opening, close)+.15, "low": np.minimum(opening, close)-.15,
                           "close": close, "volume": 1000000., "freq": "daily", "adjust": "pre"})
    cal = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-06-30", freq="D")})
    cal["is_trading_day"] = cal["date"].dt.dayofweek < 5
    source.to_csv(config["source_path"], index=False)
    cal.to_csv(config["calendar_path"], index=False)
    registered = [{"label": "early", "start": "2020-02-03", "end": "2020-03-31"},
                  {"label": "later", "start": "2020-04-01", "end": "2020-06-19"}]
    first = validation.run_triple_validation(config, registered, tmp_path / "reports")
    assert first["status"] == "completed"
    saved = {p.relative_to(first["path"]): p.read_bytes() for p in Path(first["path"]).rglob("*") if p.is_file()}
    reports = []
    for window in first["summary"]["windows"]:
        row = window["outcomes"][0]
        report = json.loads((Path(window["path"]) / row["artifact_path"]).read_text())
        assert report["strategy"]["metrics"]["initial_nav"] == 1.0
        assert report["baselines"]["cash"]["metrics"]["final_nav"] == 1.0
        reports.append(report)
    assert reports[0]["strategy"]["settings"] == reports[1]["strategy"]["settings"]
    source.loc[source["datetime"].ge("2020-04-01"), ["open", "high", "low", "close"]] *= 1.4
    source.to_csv(config["source_path"], index=False)
    second = validation.run_triple_validation(config, registered, tmp_path / "reports")
    assert second["status"] == "completed"
    early = second["summary"]["windows"][0]
    again = json.loads((Path(early["path"]) / early["outcomes"][0]["artifact_path"]).read_text())
    assert reports[0] == again
    assert all((Path(first["path"]) / relative).read_bytes() == value for relative, value in saved.items())


def test_source_code_changes_during_a_window_do_not_count_as_success(tmp_path, monkeypatch):
    identity = ["original"]
    monkeypatch.setattr(validation, "_implementation_identity", lambda: identity[0])

    def child(config, destination):
        identity[0] = "changed"
        return output(config, destination)

    monkeypatch.setattr(validation, "run_triple_experiment", child)
    result = validation.run_triple_validation(local_config(tmp_path), windows(), tmp_path / "reports")
    assert result["status"] == "failed"
    assert result["summary"]["failed_tasks"] == 8


def test_protocol_tampering_cannot_publish_a_valid_final_summary(tmp_path, monkeypatch):
    def child(config, destination):
        protocol = Path(destination).parents[1] / "protocol.json"
        protocol.write_text("{}", encoding="utf-8")
        return output(config, destination)

    monkeypatch.setattr(validation, "run_triple_experiment", child)
    with pytest.raises(ValueError, match="protocol integrity"):
        validation.run_triple_validation(local_config(tmp_path), windows(), tmp_path / "reports")
    run = next((tmp_path / "reports").iterdir())
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    assert not (run / "summary.json").exists()
