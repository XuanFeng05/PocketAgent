from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from app_layer.backend.server import DashboardServer, RequestHandler, ThreadingHTTPServer


@pytest.fixture
def dashboard(tmp_path):
    state = DashboardServer(research_root=tmp_path / "research")
    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    server.app_state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    state.jobs.shutdown()


def call(base, path, payload=None):
    req = Request(base+path, data=None if payload is None else json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=20) as response:
        return json.load(response)["data"]


def wait(base, job_id):
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        record = call(base, f"/api/research/runs/{job_id}")
        if record["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return record
        time.sleep(.03)
    raise AssertionError("Research run did not reach a terminal state")


def test_batch_records_failures_and_loads_candidate_fifo_on_demand(dashboard):
    base, _ = dashboard
    config = {"mode":"batch", "source_type":"demo", "strategy":"sma_trend",
              "parameter_grid":{"fast_window":[3,8],"slow_window":[5]}}
    preview = call(base, "/api/research/preview", config)
    assert preview["candidate_count"] == 2
    response = call(base, "/api/research/start", config)
    record = wait(base, response["job_id"])
    assert record["status"] == "completed", record
    batch = record["result"]["batch"]
    assert batch["failed_count"] == 1 and batch["completed_count"] == 1
    successful = next(t for t in batch["trials"] if t["status"] == "completed")
    assert "result" not in successful  # Large reports are stored, not repeated in the table payload.
    report = call(base, f"/api/research/runs/{record['job_id']}/trials/{successful['candidate_id']}")
    assert set(report["baselines"]) == {"cash", "buy_hold"}
    fifo = report["strategy"]["fifo_report"]["summary"]
    metrics = report["strategy"]["metrics"]
    assert fifo["fee_reconciliation_difference"] == pytest.approx(0,abs=1e-8)
    assert fifo["realized_net_pnl"] + fifo["unrealized_net_pnl"] == pytest.approx(metrics["final_nav"]-metrics["initial_nav"])
    path = Path(record["result_path"])
    assert len((path/"trials.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (path/successful["artifact_path"]/"fifo_closed.csv").is_file()
    assert (path/"trials.csv").is_file()


def test_walkforward_api_keeps_holdout_unseen_and_saves_fold_fifo(dashboard):
    base, _ = dashboard
    config = {"mode":"walk_forward", "source_type":"demo", "strategy":"sma_trend",
              "parameter_grid":{"fast_window":[3,5],"slow_window":[15]},
              "walk_forward":{"train_bars":40,"validation_bars":15,"gap_bars":1,"holdout_bars":15}}
    response = call(base, "/api/research/start", config)
    record = wait(base,response["job_id"])
    assert record["status"] == "completed",record
    workflow = record["result"]["walk_forward"]
    assert workflow["holdout"]["status"] == "reserved"
    assert workflow["holdout"]["result"] is None
    assert workflow["summary"]["completed_folds"] == 2
    for fold in workflow["folds"]:
        assert fold["periods"]["train"]["end"] < fold["periods"]["validation"]["start"]
        assert len(fold["train_trials"]) == 2
        assert "result" not in fold["train_trials"][0]
        assert fold["validation"]["strategy"]["fifo_report"]
        assert (Path(record["result_path"])/"folds"/fold["fold_id"]/"validation"/"fifo_closed.csv").is_file()


def test_preflight_rejects_oversized_grid_before_creating_run(dashboard):
    base,state = dashboard
    with pytest.raises(HTTPError) as error:
        call(base,"/api/research/start",{"mode":"batch","parameter_grid":{"fast_window":list(range(1,10)),"slow_window":list(range(10,20))}})
    assert error.value.code == 400
    assert state.research.list_runs() == []


def test_cancel_retains_completed_trials_and_is_terminal(dashboard,monkeypatch):
    base,state = dashboard
    import research_layer.batch as batch_module
    original = batch_module.run_backtest
    entered = threading.Event()
    resume = threading.Event()
    counts = {"n":0}
    def controlled(*args,**kwargs):
        counts["n"] += 1
        if counts["n"] == 4:  # two baselines and one saved candidate precede this step
            entered.set()
            assert resume.wait(10)
        return original(*args,**kwargs)
    monkeypatch.setattr(batch_module,"run_backtest",controlled)
    response = call(base,"/api/research/start",{"mode":"batch","parameter_grid":{"fast_window":[2,3,4],"slow_window":[10]}})
    try:
        assert entered.wait(10)
        with pytest.raises(HTTPError) as busy:
            call(base,"/api/research/start",{"source_type":"demo"})
        assert busy.value.code == 400
        cancelled = call(base,f"/api/research/runs/{response['job_id']}/cancel",{})
        assert cancelled["status"] == "cancelling"
    finally:
        resume.set()
    record = wait(base,response["job_id"])
    assert record["status"] == "cancelled",record
    assert len(record["result"]["trials"]) >= 1
    assert (Path(record["result_path"])/"partial_result.json").is_file()


def test_atomic_record_retries_transient_windows_reader_lock(tmp_path,monkeypatch):
    from app_layer.backend.research_controller import _write_json,_read_json
    destination=tmp_path/"request.json"
    destination.write_text('{"status":"running"}',encoding="utf-8")
    replace=Path.replace
    calls={"n":0}
    def locked_once(self,target):
        if Path(target)==destination and calls["n"]==0:
            calls["n"]+=1
            raise PermissionError("simulated Windows reader handle")
        return replace(self,target)
    monkeypatch.setattr(Path,"replace",locked_once)
    _write_json(destination,{"status":"completed"})
    assert _read_json(destination)=={"status":"completed"}
    assert calls["n"]==1
    assert not list(tmp_path.glob('*.tmp'))
