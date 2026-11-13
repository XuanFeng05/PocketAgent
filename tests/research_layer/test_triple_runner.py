import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from research_layer.triple_runner import normalize_triple_config, run_triple_experiment
from strategy_layer.triple_screen import TripleScreenSettings


def fixture_config(tmp_path):
    calendar = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-06-30", freq="D")})
    calendar["is_trading_day"] = calendar["date"].dt.weekday < 5
    dates = pd.bdate_range("2020-01-01", "2020-06-19")
    x = np.arange(len(dates))
    close = 10 + .02 * x + .6 * np.sin(x / 3)
    opening = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame({"symbol": "000001.SZ", "datetime": dates, "open": opening,
                         "high": np.maximum(opening, close) + .15, "low": np.minimum(opening, close) - .15,
                         "close": close, "volume": 1000000., "amount": 10000000., "freq": "daily", "adjust": "pre"})
    path, cal = tmp_path / "input.csv", tmp_path / "calendar.csv"
    bars.to_csv(path, index=False)
    calendar.to_csv(cal, index=False)
    return {"source_type": "csv", "source_path": str(path), "calendar_path": str(cal),
            "symbol": "000001.SZ", "start": "2020-02-03", "end": "2020-06-19",
            "settings": {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "warmup_weeks": 2},
            "cost_scenarios_bps": [0, 10], "research_scope": "exploratory"}


def test_versioned_settings_validate_parameters_and_default_warmup():
    assert TripleScreenSettings().warmup_weeks == 100
    for settings in ({"macd_fast": 30}, {"warmup_weeks": 1}, {"cost_bps": float("nan")}, {"force_span": True}):
        with pytest.raises(ValueError):
            TripleScreenSettings(**settings)


def test_explicit_price_basis_and_research_scope(tmp_path):
    config = fixture_config(tmp_path)
    for override in ({"adjust": "none"}, {"research_scope": "untouched_holdout"},
                     {"variant": "unknown"},
                     {"symbols": ["000001.SZ", "000001.SZ"]}, {"cost_scenarios_bps": [0, -1]}):
        with pytest.raises(ValueError):
            normalize_triple_config({**config, **override})


def test_all_symbols_accounted_for_and_cost_baselines_aligned(tmp_path):
    config = fixture_config(tmp_path)
    config.update(symbols=["000001.SZ", "600036.SH"])
    source_bytes = Path(config["source_path"]).read_bytes()
    output = run_triple_experiment(config, tmp_path / "reports")
    assert output["status"] == "partial"
    summary = output["summary"]
    assert summary["total_tasks"] == 4
    assert summary["completed_tasks"] == 2
    assert summary["failed_tasks"] == 2
    assert {r["symbol"] for r in summary["outcomes"]} == {"000001.SZ", "600036.SH"}
    assert Path(config["source_path"]).read_bytes() == source_bytes
    rows = [r for r in summary["outcomes"] if r["status"] == "completed"]
    assert rows[1]["metrics"]["total_return"] <= rows[0]["metrics"]["total_return"]
    for row in rows:
        report = json.loads((Path(output["path"]) / row["artifact_path"]).read_text(encoding="utf-8"))
        selected = report["strategy"]
        for baseline in report["baselines"].values():
            assert (baseline["start"], baseline["end"]) == (selected["start"], selected["end"])
            assert baseline["cost_bps"] == row["cost_bps"]
        assert report["baselines"]["cash"]["metrics"]["total_return"] == 0
    index = json.loads((Path(output["path"]) / "artifacts.json").read_text())
    for relative, digest in index["sha256"].items():
        assert hashlib.sha256((Path(output["path"]) / relative).read_bytes()).hexdigest() == digest


def test_resume_reuses_verified_results_in_new_run_without_overwriting_source(tmp_path):
    config = fixture_config(tmp_path)
    first = run_triple_experiment(config, tmp_path / "reports")
    saved = {p.relative_to(first["path"]).as_posix(): p.read_bytes() for p in Path(first["path"]).rglob("*") if p.is_file()}
    second = run_triple_experiment(config, tmp_path / "reports", resume_from=first["path"])
    assert first["path"] != second["path"]
    assert second["status"] == "completed"
    assert all(r["reused"] for r in second["summary"]["outcomes"])
    assert first["summary"]["by_cost"] == second["summary"]["by_cost"]
    assert all((Path(first["path"]) / p).read_bytes() == data for p, data in saved.items())
    with pytest.raises(ValueError, match="configuration"):
        run_triple_experiment({**config, "cost_scenarios_bps": [20]}, tmp_path / "reports", resume_from=first["path"])


def test_resume_rejects_corrupted_input(tmp_path):
    config = fixture_config(tmp_path)
    first = run_triple_experiment(config, tmp_path / "reports")
    path = Path(first["path"]) / "inputs" / "000001_SZ.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="integrity"):
        run_triple_experiment(config, tmp_path / "reports", resume_from=first["path"])


def test_nearby_costs_have_distinct_artifacts(tmp_path):
    config = fixture_config(tmp_path)
    config["cost_scenarios_bps"] = [0.12345671, 0.12345672]
    output = run_triple_experiment(config, tmp_path / "reports")
    rows = output["summary"]["outcomes"]
    assert len({r["task_id"] for r in rows}) == 2
    assert len({r["artifact_path"] for r in rows}) == 2
    for row in rows:
        report = json.loads((Path(output["path"]) / row["artifact_path"]).read_text())
        assert report["strategy"]["settings"]["cost_bps"] == row["cost_bps"]


def test_resume_uses_hash_verified_checkpoint_not_mutable_progress(tmp_path):
    config = fixture_config(tmp_path)
    first = run_triple_experiment(config, tmp_path / "reports")
    root = Path(first["path"])
    progress = json.loads((root / "progress.json").read_text())
    progress["outcomes"][1]["artifact_path"] = progress["outcomes"][0]["artifact_path"]
    (root / "progress.json").write_text(json.dumps(progress))
    second = run_triple_experiment(config, tmp_path / "reports", resume_from=root)
    assert second["summary"]["by_cost"] == first["summary"]["by_cost"]
    manifest = json.loads((root / "artifacts.json").read_text())
    checkpoint = root / manifest["outcome_files"][1]
    checkpoint.write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="integrity"):
        run_triple_experiment(config, tmp_path / "reports", resume_from=root)


def test_optional_halt_amount_is_preserved_not_fabricated(tmp_path):
    config = fixture_config(tmp_path)
    frame = pd.read_csv(config["source_path"])
    frame.loc[12, "volume"] = 0
    frame.loc[12, "amount"] = np.nan
    frame.to_csv(config["source_path"], index=False)
    output = run_triple_experiment(config, tmp_path / "reports")
    assert output["status"] == "completed"
    saved = pd.read_csv(Path(output["path"]) / "inputs" / "000001_SZ.csv")
    assert "amount" not in saved
    assert pd.isna(saved.loc[12, "source_amount_unused"])
    assert saved.loc[12, "volume"] == 0
    assert saved.loc[12, "close"] == frame.loc[12, "close"]


def test_checkpoint_storage_is_one_event_per_task(tmp_path):
    config = fixture_config(tmp_path)
    config["cost_scenarios_bps"] = [0, 5, 10, 20]
    output = run_triple_experiment(config, tmp_path / "reports")
    events = list((Path(output["path"]) / "checkpoints").glob("*.json"))
    assert len(events) == 4
    assert all("outcomes" not in json.loads(p.read_text()) for p in events)


def test_ablation_variant_is_explicit_and_cannot_reuse_another_variant(tmp_path):
    config = {**fixture_config(tmp_path), "variant": "drop_weekly"}
    output = run_triple_experiment(config, tmp_path / "reports")
    assert output["status"] == "completed"
    assert output["summary"]["research_variant"] == "drop_weekly"
    for row in output["summary"]["outcomes"]:
        report = json.loads((Path(output["path"]) / row["artifact_path"]).read_text())
        assert report["strategy"]["research_variant"] == "drop_weekly"
    with pytest.raises(ValueError, match="configuration"):
        run_triple_experiment({**config, "variant": "full"}, tmp_path / "reports", resume_from=output["path"])


def test_cli_real_configuration_path(tmp_path):
    config = fixture_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run([sys.executable, "-B", "-m", "research_layer.cli.triple_screen",
                             "--config", str(path), "--output", str(tmp_path / "reports")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    final = json.loads(result.stdout.splitlines()[-1])
    assert final["status"] == "completed"
    snapshot = pd.read_csv(Path(final["path"]) / "inputs" / "000001_SZ.csv")
    assert set(snapshot["adjust"]) == {"pre"}
