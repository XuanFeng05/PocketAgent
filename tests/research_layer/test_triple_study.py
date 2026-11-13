import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research_layer.triple_study as study


PRIMARY = {"id": "primary", "label": "固定首版", "settings": {}, "variant": "full"}
FAST = {"id": "fast10", "label": "快线邻域", "settings": {"macd_fast": 10}, "variant": "full"}


def config(tmp_path, *, symbols=None):
    path = tmp_path / "local.csv"
    calendar = tmp_path / "calendar.csv"
    path.write_text("local-data-fixture", encoding="utf-8")
    calendar.write_text("local-calendar-fixture", encoding="utf-8")
    return {"source_type": "csv", "source_path": str(path), "calendar_path": str(calendar),
            "symbols": symbols or ["000001.SZ", "600036.SH"], "start": "2022-01-04", "end": "2024-12-31",
            "settings": {}, "research_scope": "exploratory_study", "cost_scenarios_bps": [10], "max_tasks": 2000}


def mock_child(monkeypatch, *, transform=None, failure=None, check=None):
    calls = []
    def run(request, output_root):
        case_id = Path(output_root).name
        calls.append(request)
        if check:
            check(request, Path(output_root))
        if failure and case_id == failure:
            raise RuntimeError("explicit child failure")
        root = Path(output_root) / "mock_run"
        root.mkdir(parents=True)
        rows = []
        for symbol in request["symbols"]:
            for cost in request["cost_scenarios_bps"]:
                value = 0.1 if case_id == "primary" else 0.2
                row = {"symbol": symbol, "cost_bps": cost, "status": "completed", "start": request["start"], "end": request["end"],
                       "metrics": {"total_return": value, "maximum_drawdown": .05}, "data_sha256": "a"*64,
                       "calendar_sha256": "b"*64, "buy_hold_return": 0.05, "excess_return_vs_buy_hold": value-.05,
                       "artifact_path": "not-expanded-by-study.json"}
                if transform:
                    row = transform(case_id, row)
                rows.append(row)
        summary = {"outcomes": rows, "price_basis": "pre", "research_scope": "exploratory"}
        saved_config = {"configuration": study.normalize_triple_config(request), "implementation_sha256": study._implementation_identity()}
        for name, obj in [("config.json", saved_config), ("summary.json", summary)]:
            (root/name).write_text(json.dumps(obj), encoding="utf-8")
        manifest = {"sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ["config.json", "summary.json"]}}
        (root/"artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"run_id": "mock_run", "status": "completed", "path": str(root), "summary": summary}
    monkeypatch.setattr(study, "run_triple_experiment", run)
    return calls


def test_protocol_precedes_child_and_pairs_stock_by_same_period_and_inputs(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    def check(request, output):
        root = output.parent.parent
        assert (root/"protocol.json").exists()
        assert (root/"protocol.sha256").read_text(encoding="utf-8").strip() == hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest()
    calls = mock_child(monkeypatch, check=check)
    result = study.run_triple_study(cfg, [FAST, PRIMARY], tmp_path/"reports")
    assert result["status"] == "completed"
    assert result["summary"]["total_tasks"] == 4
    assert calls[0]["settings"]["macd_fast"] == 12
    assert calls[1]["settings"]["macd_fast"] == 10
    assert all(r["settings"]["warmup_weeks"] == 100 and r["cost_scenarios_bps"] == [10] for r in calls)
    assert all(r["research_scope"] == "exploratory" for r in calls)
    pair = result["summary"]["cases"][1]["paired_vs_primary"][0]
    assert pair["paired_symbols"] == 2 and pair["excluded_symbols"] == 0
    assert pair["mean_return_delta"] == pytest.approx(.1)
    assert pair["positive_delta_ratio"] == 1
    assert "outcomes" not in result["summary"]["cases"][0]
    assert "best" not in result["summary"]
    manifest = json.loads((Path(result["path"])/"artifacts.json").read_text(encoding="utf-8"))
    assert len(manifest["child_manifests"]) == 2
    for relative, digest in manifest["sha256"].items():
        assert hashlib.sha256((Path(result["path"])/relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("replacement,reason", [
    ({"start": "2023-01-03"}, "effective_period_mismatch"),
    ({"data_sha256": "c"*64}, "input_identity_mismatch"),
    ({"calendar_sha256": "c"*64}, "calendar_identity_mismatch"),
    ({"status": "failed", "error": "one source missing"}, "case_failed"),
])
def test_unmatched_pairs_are_explicitly_excluded(tmp_path, monkeypatch, replacement, reason):
    def transform(case_id, row):
        if case_id != "primary" and row["symbol"] == "000001.SZ": row.update(replacement)
        return row
    mock_child(monkeypatch, transform=transform)
    result = study.run_triple_study(config(tmp_path), [PRIMARY, FAST], tmp_path/"reports")
    pair = result["summary"]["cases"][1]["paired_vs_primary"][0]
    assert pair["paired_symbols"] == 1 and pair["excluded_symbols"] == 1
    assert pair["exclusion_counts"] == {reason: 1}
    assert pair["pairs"][0]["symbol"] == "600036.SH"


def test_different_stock_horizons_not_pooled(tmp_path, monkeypatch):
    def transform(case_id, row):
        if row["symbol"] == "000001.SZ": row["start"] = "2023-01-03"
        return row
    mock_child(monkeypatch, transform=transform)
    result = study.run_triple_study(config(tmp_path), [PRIMARY, FAST], tmp_path/"reports")
    pair = result["summary"]["cases"][1]["paired_vs_primary"][0]
    assert pair["paired_symbols"] == 2 and pair["periods_aligned"] is False
    assert pair["mean_return_delta"] is None and pair["median_return_delta"] is None
    assert len(pair["by_effective_period"]) == 2


def test_case_failure_keeps_every_task_and_continues(tmp_path, monkeypatch):
    calls = mock_child(monkeypatch, failure="fast10")
    extra = {"id": "slow24", "label": "慢线邻域", "settings": {"macd_slow": 24}}
    result = study.run_triple_study(config(tmp_path), [PRIMARY, FAST, extra], tmp_path/"reports")
    assert len(calls) == 3
    assert result["status"] == "partial"
    assert result["summary"]["completed_tasks"] == 4 and result["summary"]["failed_tasks"] == 2
    saved = json.loads((Path(result["path"])/"case_results/fast10.json").read_text(encoding="utf-8"))
    assert len(saved["outcomes"]) == 2 and all(r["status"] == "failed" for r in saved["outcomes"])
    assert saved["paired_vs_primary"][0]["excluded_symbols"] == 2


def test_primary_failure_does_not_create_fake_pair_results(tmp_path, monkeypatch):
    mock_child(monkeypatch, failure="primary")
    result = study.run_triple_study(config(tmp_path), [PRIMARY, FAST], tmp_path/"reports")
    pair = result["summary"]["cases"][1]["paired_vs_primary"][0]
    assert pair["paired_symbols"] == 0
    assert pair["mean_return_delta"] is None
    assert pair["exclusion_counts"] == {"primary_failed_or_missing": 2}


@pytest.mark.parametrize("change", ["missing_primary", "primary_override", "primary_variant", "case_id", "duplicate", "variant", "warmup", "cost", "unknown", "label", "too_many"])
def test_invalid_cases_rejected_before_running(tmp_path, change):
    cfg = config(tmp_path)
    cases = [dict(PRIMARY), dict(FAST)]
    if change == "missing_primary": cases = [FAST]
    if change == "primary_override": cases[0]["settings"] = {"force_span": 3}
    if change == "primary_variant": cases[0]["variant"] = "drop_weekly"
    if change == "case_id": cases[1]["id"] = "../outside"
    if change == "duplicate": cases[1]["id"] = "primary"
    if change == "variant": cases[1]["variant"] = "unknown"
    if change == "warmup": cases[1]["settings"] = {"warmup_weeks": 2}
    if change == "cost": cases[1]["settings"] = {"cost_bps": 0}
    if change == "unknown": cases[1]["extra"] = True
    if change == "label": cases[1]["label"] = "bad\nlabel"
    if change == "too_many": cases = [PRIMARY] + [{"id": f"x{i}", "label": "x"} for i in range(16)]
    with pytest.raises(ValueError): study.normalize_study_protocol(cfg, cases)


@pytest.mark.parametrize("override", [{"research_scope": "holdout"}, {"symbols": []}, {"start": None},
    {"start": "2025-01-01"}, {"max_tasks": 3}, {"max_tasks": True}, {"max_tasks": 100001},
    {"source_path": "https://example.org/data.csv"}, {"source_path": "\\\\host\\data.csv"}, {"unknown": True}])
def test_invalid_config_rejected(tmp_path, override):
    with pytest.raises(ValueError): study.normalize_study_protocol({**config(tmp_path), **override}, [PRIMARY, FAST])


def test_protocol_mutation_invalidates_study(tmp_path, monkeypatch):
    def change(request, root):
        protocol = root.parent.parent/"protocol.json"
        protocol.write_text(protocol.read_text(encoding="utf-8") + " ")
    mock_child(monkeypatch, check=change)
    with pytest.raises(ValueError, match="changed"):
        study.run_triple_study(config(tmp_path), [PRIMARY], tmp_path/"reports")
    status = json.loads(next((tmp_path/"reports").glob("study_*/status.json")).read_text(encoding="utf-8"))
    assert status["status"] == "failed"


def test_implementation_mutation_invalidates_study(tmp_path, monkeypatch):
    marker = {"value": "a"*64}
    monkeypatch.setattr(study, "_implementation_identity", lambda: marker["value"])
    mock_child(monkeypatch, check=lambda *_: marker.update(value="b"*64))
    with pytest.raises(ValueError, match="changed"):
        study.run_triple_study(config(tmp_path), [PRIMARY], tmp_path/"reports")


def test_real_local_primary_integration_and_preserved_inputs(tmp_path):
    dates = pd.bdate_range("2020-01-01", "2020-06-19")
    x = np.arange(len(dates)); close = 10 + .02*x + .6*np.sin(x/3)
    opening = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame({"symbol": "000001.SZ", "datetime": dates, "open": opening,
                         "high": np.maximum(opening, close)+.15, "low": np.minimum(opening, close)-.15,
                         "close": close, "volume": 1000000., "adjust": "pre", "freq": "daily"})
    cal = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-06-30")})
    cal["is_trading_day"] = cal.date.dt.weekday < 5
    cfg = config(tmp_path, symbols=["000001.SZ"])
    bars.to_csv(cfg["source_path"], index=False);cal.to_csv(cfg["calendar_path"], index=False)
    before = Path(cfg["source_path"]).read_bytes()
    cfg.update(start="2020-02-03", end="2020-06-19", settings={"macd_fast":2,"macd_slow":3,"macd_signal":2,"warmup_weeks":2})
    result = study.run_triple_study(cfg, [PRIMARY], tmp_path/"reports")
    assert result["status"] == "completed"
    pair = result["summary"]["cases"][0]["paired_vs_primary"][0]
    assert pair["paired_symbols"] == 1 and pair["mean_return_delta"] == 0
    assert Path(cfg["source_path"]).read_bytes() == before
