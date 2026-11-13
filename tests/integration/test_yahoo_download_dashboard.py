from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from app_layer.backend import download_controller as dispatch
from app_layer.backend import yahoo_download_controller as yahoo
from app_layer.backend.server import DashboardServer, RequestHandler, ThreadingHTTPServer


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    # The download engine has its own real-normalizer/provider tests. Here the
    # deterministic service boundary exercises real HTTP routing and job state.
    monkeypatch.setattr(yahoo, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dispatch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(yahoo, "load_yahoo_config", lambda path: {
        "fixed_universe_count": 2, "start": "2020-01-01", "end": "2026-07-12",
        "data_root": str(tmp_path / "us_data"),
    })
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
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def wait_job(base, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = request(base, f"/api/jobs/{job_id}")["job"]
        if job["status"] in {"completed", "partial", "failed", "cancelled"}:
            return job
        time.sleep(.01)
    raise AssertionError("Download job did not settle")


def test_profiles_match_registered_market_configs():
    profiles = yahoo.yahoo_profiles()
    assert [p["id"] for p in profiles] == ["us100", "us500"]
    assert [p["fixed_universe_count"] for p in profiles] == [100, 500]
    assert all(p["market"] == "US" and p["calendar"] == "XNYS" for p in profiles)
    assert all(p["data_root"] != "runtime_layer/data" for p in profiles)
    assert {p["id"] for p in dispatch.download_catalog()["providers"]} == {"baostock", "yahoo"}


@pytest.mark.parametrize("status,second_status", [("completed", "success"), ("partial", "failed"), ("failed", "failed")])
def test_download_http_preserves_terminal_states_and_readable_report(dashboard, monkeypatch, status, second_status):
    base, state = dashboard
    first_status = "failed" if status == "failed" else "success"

    def service(config, path, *, progress_callback, cancel_check):
        assert not cancel_check()
        assert path.name == "us500_download.json"
        outcomes = []
        for i, (symbol, row_status) in enumerate([("AAPL", first_status), ("MSFT", second_status)], 1):
            row = {"symbol": symbol, "status": row_status, "rows": 20 if row_status == "success" else 0,
                   "error": None if row_status == "success" else "Network timeout"}
            outcomes.append(row)
            progress_callback({"event": "symbol_finished", "completed": i, "total": 2,
                "success": sum(r["status"] == "success" for r in outcomes),
                "failed": sum(r["status"] == "failed" for r in outcomes), "current_symbol": symbol, "result": row})
        return {"status": status, "report_path": "runtime_layer/reports/provider.json", "outcomes": outcomes}

    monkeypatch.setattr(yahoo, "download_yahoo", service)
    created = request(base, "/api/download/start", {"source": "yahoo"})["data"]
    job = wait_job(base, created["job_id"])
    assert job["status"] == status
    assert job["source"] == "yahoo" and job["market"] == "US"
    assert job["completed"] == 2 and job["total"] == 2
    assert job["result"]["status"] == status
    assert job["result"]["service_report_path"].endswith("provider.json")
    report = request(base, "/api/download/report?path=" + job["result"]["report_path"])["data"]
    assert report["total"] == 2
    assert report["failed_count"] == sum(s == "failed" for s in [first_status, second_status])
    assert all(r["adjust"] == "split_dividend_adjusted" for r in report["rows"])


def test_download_cancel_is_visible_and_completed_rows_survive(dashboard, monkeypatch):
    base, state = dashboard
    started = threading.Event()

    def service(config, path, *, progress_callback, cancel_check):
        done = {"symbol": "AAPL", "status": "success", "rows": 20, "error": None}
        progress_callback({"event": "symbol_finished", "completed": 1, "total": 2,
                           "success": 1, "failed": 0, "current_symbol": "AAPL", "result": done})
        started.set()
        deadline = time.monotonic() + 5
        while not cancel_check() and time.monotonic() < deadline:
            time.sleep(.01)
        assert cancel_check()
        return {"status": "cancelled", "report_path": "provider.json", "outcomes": [done,
            {"symbol": "MSFT", "status": "cancelled", "rows": 0, "error": "Cancelled before request"}]}

    monkeypatch.setattr(yahoo, "download_yahoo", service)
    job_id = request(base, "/api/download/start", {"source": "yahoo", "profile": "us100"})["data"]["job_id"]
    assert started.wait(5)
    running = request(base, f"/api/jobs/{job_id}")["job"]
    assert running["completed"] == 1 and running["progress"] == .5
    with pytest.raises(HTTPError):
        request(base, "/api/download/start", {"source": "baostock", "symbols": ["000001.SZ"]})
    request(base, f"/api/jobs/{job_id}/cancel", {})
    job = wait_job(base, job_id)
    assert job["status"] == "cancelled" and job["completed"] == 1 and job["saved_rows"] == 20
    assert job["result"]["outcomes"][0]["status"] == "success"
    assert Path(yahoo.PROJECT_ROOT / job["result"]["report_path"]).exists()


@pytest.mark.parametrize("payload", [
    {"source": "other"}, {"source": "yahoo", "profile": "../outside"},
    {"source": "yahoo", "storage_root": "runtime_layer/data"},
    {"source": "yahoo", "adjustflags": ["2"]}, {"source": "yahoo", "symbols": ["PENNY"]},
])
def test_invalid_sources_profiles_and_mixed_market_payloads_fail_before_job(dashboard, payload):
    base, state = dashboard
    with pytest.raises(HTTPError) as error:
        request(base, "/api/download/start", payload)
    assert error.value.code == 400
    assert state.jobs.list_jobs() == []
