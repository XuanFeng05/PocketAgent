from __future__ import annotations

import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import pytest

from app_layer.backend.server import DashboardServer, RequestHandler, ThreadingHTTPServer


@pytest.fixture
def dashboard(tmp_path):
    state = DashboardServer(research_root=tmp_path / "research")
    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    server.app_state = state
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown(); server.server_close(); thread.join(timeout=5); state.jobs.shutdown()


def request(base, path, payload=None):
    req = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def wait_run(base, job_id):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        value = request(base, f"/api/triple/runs/{job_id}")["data"]
        if value["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return value
        time.sleep(.05)
    raise AssertionError("Triple Screen dashboard task did not finish")


def test_triple_page_spec_and_local_single_flow(dashboard):
    base, state = dashboard
    with urlopen(base + "/triple.html") as response:
        page = response.read().decode()
    assert "三重滤网研究" in page
    spec = request(base, "/api/triple/spec")["data"]
    assert spec["price_basis"] == "pre"
    assert spec["symbol_count"] == 100
    assert len(spec["symbols"]) == 100
    job = request(base, "/api/triple/start", {
        "mode": "single", "symbols": ["000001.SZ"],
        "start": "2022-01-04", "end": "2022-12-30", "costs_bps": [10],
    })["data"]
    record = wait_run(base, job["job_id"])
    assert record["status"] == "completed", record
    assert record["config"]["symbols"] == ["000001.SZ"]
    assert record["config"]["adjust"] == "pre"
    assert Path(record["result_path"]).exists()
    report_url = record.get("report_url")
    assert report_url and "report" in report_url
    with urlopen(base + report_url) as response:
        assert "三重滤网" in response.read().decode("utf-8")
    assert request(base, "/api/triple/runs")["data"]


def test_triple_dashboard_rejects_unknown_symbols_and_invalid_modes(dashboard):
    base, _ = dashboard
    for payload in [
        {"mode": "single", "symbols": ["999999.SZ"]},
        {"mode": "unknown", "symbols": ["000001.SZ"]},
        {"mode": "single", "symbols": ["000001.SZ", "000002.SZ"]},
    ]:
        try:
            request(base, "/api/triple/start", payload)
        except Exception as exc:
            assert "HTTP Error 400" in str(exc)
        else:
            raise AssertionError("invalid Triple Screen payload was accepted")


@pytest.mark.parametrize("mode,payload_extra", [
    ("annual", {"windows": [{"label": "smoke", "start": "2022-01-01", "end": "2022-12-31"}]}),
    ("study", {"cases": [{"id": "primary", "label": "完整", "settings": {}, "variant": "full"}]}),
])
def test_triple_dashboard_supports_annual_and_study_modes(dashboard, mode, payload_extra):
    base, _ = dashboard
    payload = {"mode": mode, "symbols": ["000001.SZ"], "start": "2022-01-04", "end": "2022-12-30",
               "costs_bps": [10], **payload_extra}
    job = request(base, "/api/triple/start", payload)["data"]
    record = wait_run(base, job["job_id"])
    assert record["status"] == "completed", record
    assert Path(record["result_path"]).exists()
