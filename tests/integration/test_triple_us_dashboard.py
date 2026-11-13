"""Exercise the repaired dashboard contract through the separate US routes."""
import json
import threading
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from app_layer.backend.server import DashboardServer, RequestHandler, ThreadingHTTPServer
from data_layer.storage.partitioned_storage import save_kline_to_market_shard


@pytest.mark.parametrize("mode", ["single", "batch", "annual", "study"])
def test_us_routes_preserve_progress_children_and_market_isolation(tmp_path, mode):
    market = tmp_path / "us"
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    for symbol in ["AAPL", "MSFT"]:
        close = 20 + np.arange(len(dates)) * .02 + np.sin(np.arange(len(dates)))
        opening = np.r_[close[0], close[:-1]]
        saved = save_kline_to_market_shard(pd.DataFrame({
            "symbol": symbol, "datetime": dates, "open": opening, "close": close,
            "high": np.maximum(opening, close) + .2, "low": np.minimum(opening, close) - .2,
            "volume": 1e6, "freq": "daily", "adjust": "pre", "source": "synthetic_us_fixture",
        }), market)
        assert saved.ok
    days = pd.date_range("2019-12-30", "2020-04-05")
    pd.DataFrame({"date": days, "is_trading_day": days.weekday < 5,
                  "session_close": [f"{d.date()}T16:00:00" if d.weekday() < 5 else None for d in days],
                  "source": "synthetic_weekday_calendar"}).to_csv(market / "calendar.csv", index=False)
    state = DashboardServer(research_root=tmp_path / "out", triple_data_root=tmp_path / "cn", triple_us_data_root=market)
    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    server.app_state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def api(path, payload=None):
        request = Request(base + path, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.load(response)["data"]
    try:
        spec = api("/api/triple-us/spec")
        assert spec["market"] == "US" and spec["symbols"] == ["AAPL", "MSFT"]
        assert api("/api/triple/spec")["symbols"] == []
        body = {"mode": mode, "symbols": ["AAPL", "MSFT"] if mode == "batch" else ["AAPL"],
                "start": "2020-02-03", "end": "2020-03-31", "costs_bps": [10],
                "settings": {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "warmup_weeks": 2}}
        if mode == "study":
            body["cases"] = [spec["default_cases"][0], spec["default_cases"][-1]]
        plan = api("/api/triple-us/preview", body)
        assert plan["config"]["settings"]["version"] == "triple_screen_us_pre_v1"
        assert plan["config"]["calendar_path"] == str(market / "calendar.csv")
        job = api("/api/triple-us/start", body)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            record = api("/api/triple-us/runs/" + job["job_id"])
            if record["status"] not in {"queued", "running", "cancelling"}:
                break
            time.sleep(.02)
        assert record["status"] == "completed", record
        assert record["progress"]["completed"] == plan["total_tasks"]
        assert record["summary"]["failed_tasks"] == 0
        assert record["report_links"]
        for link in record["report_links"]:
            assert link["url"].startswith("/api/triple-us/runs/")
            with urlopen(base + link["url"], timeout=10) as response:
                assert "美股" in response.read().decode()
        assert api("/api/triple/runs") == []
        assert api("/api/triple-us/runs")[0]["job_id"] == job["job_id"]
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5); state.jobs.shutdown()
