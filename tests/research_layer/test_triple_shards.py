import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research_layer.triple_shards as shards


def config(tmp_path, symbols=None):
    source = tmp_path / "local.csv"
    calendar = tmp_path / "calendar.csv"
    source.write_text("local mock source")
    calendar.write_text("local mock calendar")
    return {"source_type": "csv", "source_path": str(source), "calendar_path": str(calendar),
            "symbols": symbols or ["000001.SZ", "600036.SH"], "settings": {}, "cost_scenarios_bps": [0, 10],
            "start": "2022-01-04", "end": "2024-12-31", "research_scope": "exploratory"}


def mock_child(monkeypatch, *, exception_shard=None, fail_symbol=None, unknown_after=None, outside=None):
    calls = []
    def run(request, output_root, *, resume_from=None):
        calls.append({"request": request, "output": str(output_root), "resume_from": str(resume_from) if resume_from else None})
        parent = Path(output_root).parent.parent
        assert (parent / "protocol.json").exists()
        assert hashlib.sha256((parent / "protocol.json").read_bytes()).hexdigest() == (parent / "protocol.sha256").read_text().strip()
        shard_id = Path(output_root).name
        if shard_id == exception_shard:
            raise RuntimeError("one shard failed before child creation")
        child = Path(outside) if outside else Path(output_root) / "mock_run"
        child.mkdir(parents=True, exist_ok=True)
        saved = {"configuration": shards.normalize_triple_config(request), "implementation_sha256": shards._implementation_identity()}
        (child / "config.json").write_text(json.dumps(saved))
        (child / "result.json").write_text("{}")
        files = ["config.json", "result.json"]
        events, rows = [], []
        for symbol in request["symbols"]:
            for cost in request["cost_scenarios_bps"]:
                if unknown_after is not None and len(rows) >= unknown_after:
                    continue
                row = {"symbol": symbol, "cost_bps": cost, "status": "failed" if symbol == fail_symbol else "completed",
                       "task_id": f"task_{len(rows)}", "artifact_path": "result.json", "reused": bool(resume_from)}
                if symbol == fail_symbol: row["error"] = "missing local data"
                event = f"event_{len(rows)}.json"
                (child / event).write_text(json.dumps(row))
                rows.append(row); files.append(event); events.append(event)
        if unknown_after is None:
            summary = {"outcomes": rows}
            (child / "summary.json").write_text(json.dumps(summary))
            files.append("summary.json")
        else:
            summary = {}
        manifest = {"sha256": {name: hashlib.sha256((child / name).read_bytes()).hexdigest() for name in files}, "outcome_files": events}
        (child / "artifacts.json").write_text(json.dumps(manifest))
        if unknown_after is not None:
            raise RuntimeError("interrupted child after committed outcomes")
        return {"run_id": "mock_run", "path": str(child), "status": "completed", "summary": summary}
    monkeypatch.setattr(shards, "run_triple_experiment", run)
    return calls


def test_more_than_100_symbols_are_sequentially_split_in_fixed_order(tmp_path, monkeypatch):
    symbols = [f"{n:06d}.SZ" for n in range(1, 206)]
    calls = mock_child(monkeypatch)
    result = shards.run_triple_shards(config(tmp_path, symbols), tmp_path / "reports")
    assert result["status"] == "completed"
    assert [len(c["request"]["symbols"]) for c in calls] == [100, 100, 5]
    assert [s for c in calls for s in c["request"]["symbols"]] == symbols
    assert result["summary"]["counts"] == {"completed": 410, "failed": 0, "unknown": 0}
    protocol = json.loads((Path(result["path"]) / "protocol.json").read_text())
    assert [(s["start_index"], s["end_index_exclusive"]) for s in protocol["shards"]] == [(0, 100), (100, 200), (200, 205)]
    assert protocol["bounded_memory_verified"] is False


@pytest.mark.parametrize("size", [0, 501, True, 1.5])
def test_invalid_shard_sizes_are_rejected(tmp_path, size):
    with pytest.raises(ValueError, match="shard_size"):
        shards.make_shard_protocol(config(tmp_path), size)


def test_duplicate_symbols_and_network_sources_are_rejected(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        shards.make_shard_protocol({**cfg, "symbols": ["000001.SZ", "000001.SZ"]})
    for path in ["https://example.com/data.csv", r"\\server\data.csv"]:
        with pytest.raises(ValueError, match="Network"):
            shards.make_shard_protocol({**cfg, "source_path": path})


def test_oversized_monolithic_csv_is_not_misrepresented_as_memory_bounded(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(shards, "MAX_CSV_BYTES", 1)
    with pytest.raises(ValueError, match="per-symbol market storage"):
        shards.make_shard_protocol(cfg)


def test_failed_and_unknown_tasks_are_not_dropped(tmp_path, monkeypatch):
    symbols = ["000001.SZ", "000002.SZ", "600036.SH"]
    calls = mock_child(monkeypatch, fail_symbol="000001.SZ", exception_shard="shard_00001")
    result = shards.run_triple_shards(config(tmp_path, symbols), tmp_path / "reports", shard_size=1)
    assert len(calls) == 3
    assert result["status"] == "incomplete"
    assert result["summary"]["counts"] == {"completed": 2, "failed": 2, "unknown": 2}
    assert sum(result["summary"]["counts"].values()) == 6
    assert {r["symbol"] for s in result["summary"]["shards"] for r in s["outcomes"]} == set(symbols)


def test_partial_child_events_are_verified_and_unfinished_tasks_stay_unknown(tmp_path, monkeypatch):
    mock_child(monkeypatch, unknown_after=1)
    result = shards.run_triple_shards(config(tmp_path), tmp_path / "reports")
    assert result["status"] == "incomplete"
    assert result["summary"]["counts"] == {"completed": 1, "failed": 0, "unknown": 3}
    assert result["summary"]["shards"][0]["child_path"]


def test_resume_passes_verified_children_and_never_overwrites_original(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    calls = mock_child(monkeypatch)
    first = shards.run_triple_shards(cfg, tmp_path / "reports", shard_size=1)
    original = {p.relative_to(first["path"]).as_posix(): p.read_bytes() for p in Path(first["path"]).rglob("*") if p.is_file()}
    second = shards.run_triple_shards(cfg, tmp_path / "reports", shard_size=1, resume_from=first["path"])
    assert first["path"] != second["path"]
    assert all(c["resume_from"] is not None for c in calls[2:])
    assert all(row["reused"] for record in second["summary"]["shards"] for row in record["outcomes"])
    assert all((Path(first["path"]) / name).read_bytes() == contents for name, contents in original.items())


def test_resume_rejects_changed_protocol_and_recursive_child_corruption(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    mock_child(monkeypatch)
    first = shards.run_triple_shards(cfg, tmp_path / "reports", shard_size=1)
    with pytest.raises(ValueError, match="identity"):
        shards.run_triple_shards({**cfg, "variant": "drop_weekly"}, tmp_path / "reports", shard_size=1, resume_from=first["path"])
    child = Path(first["summary"]["shards"][0]["child_path"])
    (child / "result.json").write_text("corrupted")
    with pytest.raises(ValueError, match="integrity"):
        shards.run_triple_shards(cfg, tmp_path / "reports", shard_size=1, resume_from=first["path"])


def test_child_output_escape_is_not_accepted_as_completed(tmp_path, monkeypatch):
    mock_child(monkeypatch, outside=tmp_path / "outside")
    result = shards.run_triple_shards(config(tmp_path), tmp_path / "reports")
    assert result["summary"]["counts"] == {"completed": 0, "failed": 0, "unknown": 4}
    assert "escapes" in result["summary"]["shards"][0]["error"]


def test_running_source_with_live_owner_cannot_be_resumed(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    mock_child(monkeypatch)
    first = shards.run_triple_shards(cfg, tmp_path / "reports")
    (Path(first["path"]) / "status.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="still running"):
        shards.run_triple_shards(cfg, tmp_path / "reports", resume_from=first["path"])


def real_config(tmp_path):
    days = pd.bdate_range("2024-01-01", periods=70)
    x = np.arange(len(days))
    pieces = []
    for symbol, scale in [("000001.SZ", 1), ("600036.SH", 2)]:
        close = scale * (10 + x*.02 + .5*np.sin(x/3))
        opening = np.r_[close[0], close[:-1]]
        pieces.append(pd.DataFrame({"symbol": symbol, "datetime": days, "open": opening,
                                    "high": np.maximum(opening, close)+.2, "low": np.minimum(opening, close)-.1,
                                    "close": close, "volume": 1000000., "freq": "daily", "adjust": "pre"}))
    cfg = config(tmp_path)
    pd.concat(pieces).to_csv(cfg["source_path"], index=False)
    calendar_days = pd.date_range("2024-01-01", "2024-04-30")
    pd.DataFrame({"date": calendar_days, "is_trading_day": calendar_days.dayofweek < 5}).to_csv(cfg["calendar_path"], index=False)
    cfg.update(start="2024-02-01", end="2024-04-05", settings={"macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "warmup_weeks": 2})
    return cfg


def test_two_symbol_real_child_integration_and_resume_reuse(tmp_path, monkeypatch):
    cfg = real_config(tmp_path)
    first = shards.run_triple_shards(cfg, tmp_path / "reports", shard_size=1)
    assert first["status"] == "completed"
    assert first["summary"]["counts"] == {"completed": 4, "failed": 0, "unknown": 0}
    # No already-completed strategy/baseline may be scored again during recovery.
    import research_layer.triple_runner as runner
    def no_rescore(*args, **kwargs):
        raise AssertionError("Completed child task was rescored")
    monkeypatch.setattr(runner, "run_triple_screen", no_rescore)
    monkeypatch.setattr(runner, "run_technical_backtest", no_rescore)
    second = shards.run_triple_shards(cfg, tmp_path / "reports", shard_size=1, resume_from=first["path"])
    assert second["status"] == "completed"
    assert all(row["reused"] for s in second["summary"]["shards"] for row in s["outcomes"])


def test_interrupted_child_resumes_committed_task_without_losing_unknowns(tmp_path, monkeypatch):
    cfg = real_config(tmp_path)
    original_run = shards.run_triple_experiment
    def interrupt_after_first(config, output, *, resume_from=None):
        def stop(event):
            if event["completed"] == 1:
                raise KeyboardInterrupt("stop after persisted outcome")
        return original_run(config, output, resume_from=resume_from, progress_callback=stop)
    monkeypatch.setattr(shards, "run_triple_experiment", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        shards.run_triple_shards(cfg, tmp_path / "reports", shard_size=1)
    original = next((tmp_path / "reports").glob("shards_*"))
    status = json.loads((original / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert (status["completed"], status["failed"], status["unknown"]) == (1, 0, 3)
    monkeypatch.setattr(shards, "run_triple_experiment", original_run)
    resumed = shards.run_triple_shards(cfg, tmp_path / "reports", shard_size=1, resume_from=original)
    assert resumed["status"] == "completed"
    rows = [row for record in resumed["summary"]["shards"] for row in record["outcomes"]]
    assert sum(row["reused"] for row in rows) == 1
    assert len(rows) == 4
