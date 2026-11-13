from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
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


def request(base, path, payload=None):
    req = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=20) as response:
        return json.load(response)


def wait_run(base, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        record = request(base, f"/api/research/runs/{job_id}")["data"]
        if record["status"] in {"completed", "failed"}:
            return record
        time.sleep(0.02)
    raise AssertionError("Research job did not finish")


def test_http_demo_comparison_persists_and_survives_service_restart(dashboard):
    base, state = dashboard
    with urlopen(base + "/") as response:
        page = response.read().decode()
    assert 'id="researchForm"' in page
    assert request(base, "/api/health")["mode"] == "cpu_research"
    spec = request(base, "/api/research/spec")["data"]
    assert spec["execution_defaults"]["max_position_ratio"] == 1.0
    response = request(base, "/api/research/start", {"source_type": "demo", "strategy": "sma_trend"})
    record = wait_run(base, response["data"]["job_id"])
    assert record["status"] == "completed", record
    result = record["result"]
    assert result["synthetic"] is True
    assert set(result["baselines"]) == {"cash", "buy_hold"}
    assert result["strategy"]["equity_curve"]
    assert result["baselines"]["cash"]["metrics"]["total_return"] == 0
    run_path = Path(record["result_path"])
    for name in ["config.json", "result.json", "equity.csv", "fills.csv", "input.csv"]:
        assert (run_path / name).is_file()
    second_state = DashboardServer(research_root=state.research.output_root)
    try:
        assert second_state.research.detail(record["job_id"])["result"] == result
    finally:
        second_state.jobs.shutdown()


def test_invalid_data_and_failed_strategy_are_visible(dashboard):
    base, state = dashboard
    with pytest.raises(HTTPError) as error:
        request(base, "/api/research/start", {"source_type": "csv", "source_path": "missing_file_for_test.csv"})
    assert error.value.code == 400
    response = request(base, "/api/research/start", {
        "source_type": "demo", "strategy": "sma_trend", "parameters": {"fast_window": 20, "slow_window": 5},
    })
    record = wait_run(base, response["data"]["job_id"])
    assert record["status"] == "failed"
    assert "fast_window" in record["error"]
    assert state.research.list_runs()[0]["status"] == "failed"


def test_legacy_training_reports_optional_dependency(dashboard):
    base, _ = dashboard
    if __import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") is not None:
        pytest.skip("This assertion specifically tests an installation without torch")
    with pytest.raises(HTTPError) as error:
        request(base, "/api/agent/spec")
    assert error.value.code == 503
    assert "PyTorch" in json.load(error.value)["error"]


def test_complete_cpu_flow_with_torch_import_explicitly_forbidden(tmp_path):
    code = r'''
import importlib.abc, sys, time
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('CPU path attempted torch import: ' + fullname)
sys.meta_path.insert(0, NoTorch())
from app_layer.backend.server import DashboardServer
s=DashboardServer(research_root=sys.argv[1])
try:
    run=s.research.start({'source_type':'demo','strategy':'sma_trend'})
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        item=s.research.detail(run['job_id'])
        if item['status'] in {'completed','failed'}: break
        time.sleep(.01)
    assert item['status']=='completed', item
    assert 'torch' not in sys.modules
finally:s.jobs.shutdown()
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path / "no_torch")],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr


def test_orphaned_request_becomes_interrupted_without_touching_live_owner(dashboard):
    import os
    import psutil
    from app_layer.backend.research_controller import _write_json

    _, state = dashboard
    service = state.research
    orphan = {"job_id": "orphan", "status": "running", "created_at": "2026-11-07", "config": {},
              "owner": {"pid": os.getpid(), "process_started": psutil.Process().create_time(), "session": "expired"}}
    _write_json(service._request_path("orphan"), orphan)
    assert service.detail("orphan")["status"] == "interrupted"
    assert json.loads(service._request_path("orphan").read_text(encoding="utf-8"))["status"] == "interrupted"
    job_id = state.jobs.create_job("strategy_backtest")
    live = {**orphan, "job_id": job_id, "owner": service.owner}
    _write_json(service._request_path(job_id), live)
    second = DashboardServer(research_root=service.output_root)
    try:
        assert second.research.detail(job_id)["status"] == "running"
    finally:
        second.jobs.shutdown()
