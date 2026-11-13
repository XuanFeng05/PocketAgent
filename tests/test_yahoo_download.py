"""Contract tests for independent Yahoo dispatch, source audits and cancellation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pytest

from download_layer.cli import download
from download_layer.services import yahoo_downloader as yahoo
from research_layer import elder_data


def make_config(tmp_path, symbols=("AAPL", "MSFT"), **overrides):
    universe = tmp_path / "symbols.txt"
    universe.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    config = {"market": "US", "source": "yahoo_finance_chart", "interval": "1d", "volume_unit": "shares",
              "price_adjustment": yahoo.ADJUSTMENT, "calendar": "XNYS", "universe_file": str(universe),
              "fixed_universe_count": len(symbols), "start": "2024-01-01", "end": "2024-01-03",
              "data_root": str(tmp_path / "us_data"), "report_root": str(tmp_path / "reports"),
              "workers": 1, "network_timeout_seconds": 2, "outer_timeout_seconds": 60,
              "sleep_seconds": 0, "maximum_attempts": 1, **overrides}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return config, path


def provider(symbol, start, end, timeout):
    stamps = [int(datetime(2024, 1, day, 15, tzinfo=timezone.utc).timestamp()) for day in (2, 3)]
    body = json.dumps({"chart": {"error": None, "result": [{
        "meta": {"symbol": symbol, "currency": "USD", "instrumentType": "EQUITY", "longName": "Test Équity"},
        "timestamp": stamps, "indicators": {
            "quote": [{"open": [100, 102], "high": [105, 106], "low": [99, 101], "close": [104, 105], "volume": [0, 2000]}],
            "adjclose": [{"adjclose": [52, 52.5]}]}}]}}).encode()
    return body, {"source": elder_data.SOURCE, "symbol": symbol, "start": start, "end": end,
                  "download_started_utc": "2026-11-13T00:00:00+00:00", "download_finished_utc": "2026-11-13T00:00:01+00:00"}


def forbid(*args, **kwargs):
    raise AssertionError("Network must not be called")


def test_baostock_defaults_and_required_arguments_unchanged():
    args = download.build_parser().parse_args(["--symbols", "fixed.txt", "--start", "2020-01-01", "--end", "2026-07-12"])
    assert args.source == "baostock"
    assert (args.storage, args.workers, args.adjustflags) == ("shard", 1, "3,2")
    with pytest.raises(SystemExit) as exc:
        download.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("arguments", [
    ["--source", "yahoo"],
    ["--source", "yahoo", "--config", "x.json", "--adjustflags", "2"],
    ["--source", "yahoo", "--config", "x.json", "--replace-symbol"],
    ["--source", "yahoo", "--config", "x.json", "--workers=1"],
    ["--source", "baostock", "--config", "x.json"],
    ["--source", "yahoo", "--config", "x.json", "--audit-only", "--only", "AAPL"],
])
def test_cli_rejects_provider_mixing_before_any_network(monkeypatch, arguments):
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    with pytest.raises(SystemExit) as exc:
        download.main(arguments)
    assert exc.value.code == 2


def test_cli_dispatches_real_service_and_returns_failure_exit(tmp_path, monkeypatch, capsys):
    config, path = make_config(tmp_path, symbols=("AAPL",))
    monkeypatch.setattr(elder_data, "fetch_chart", provider)
    with pytest.raises(SystemExit) as exc:
        download.main(["--source", "yahoo", "--config", str(path)])
    assert exc.value.code == 0
    assert '"event": "finished"' in capsys.readouterr().out
    daily = Path(config["data_root"]) / "daily/AAPL.csv"
    daily.write_text(daily.read_text(encoding="utf-8").replace("52.0", "53.0"), encoding="utf-8")
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    with pytest.raises(SystemExit) as exc:
        download.main(["--source", "yahoo", "--config", str(path)])
    assert exc.value.code == 2


def test_download_audits_real_source_zero_volume_and_reuses_cache(tmp_path, monkeypatch):
    config, path = make_config(tmp_path)
    calls = []
    def fetch(*args):
        calls.append(args)
        return provider(*args)
    monkeypatch.setattr(elder_data, "fetch_chart", fetch)
    events = []
    result = yahoo.download_yahoo(config, path, progress_callback=events.append)
    assert result["status"] == "completed" and result["success"] == 2
    assert result["source_reconstruction_verified"]
    assert result["market"] == "US" and result["calendar"] == "XNYS"
    assert not (Path(config["data_root"]) / ".yahoo_download.lock").exists()
    assert all(0 < args[-1] <= 2 for args in calls)
    assert [row["symbol"] for row in result["outcomes"]] == ["AAPL", "MSFT"]
    assert result["outcomes"][0]["quality"]["zero_volume_dates"] == ["2024-01-02"]
    assert events[0]["event"] == "started" and events[-1]["event"] == "finished"
    assert json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))["status"] == "completed"
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    second = yahoo.download_yahoo(config, path)
    assert second["status"] == "completed"
    assert all(row["reused_validated_cache"] for row in second["outcomes"])
    assert second["report_path"] != result["report_path"]


def test_success_and_failure_are_all_accounted_for(tmp_path, monkeypatch):
    config, path = make_config(tmp_path, workers=2)
    def fetch(symbol, *args):
        if symbol == "MSFT":
            raise TimeoutError("simulated transport timeout")
        return provider(symbol, *args)
    monkeypatch.setattr(elder_data, "fetch_chart", fetch)
    result = yahoo.download_yahoo(config, path)
    assert result["status"] == "partial"
    assert (result["success"], result["failed"], result["total"]) == (1, 1, 2)
    assert result["outcomes"][1]["error"] == "simulated transport timeout"
    assert len(Path(result["events_path"]).read_text(encoding="utf-8").splitlines()) == 2


def test_cancel_preserves_completed_files_and_accounts_unstarted_symbols(tmp_path, monkeypatch):
    config, path = make_config(tmp_path)
    calls, events = [], []
    def fetch(*args):
        calls.append(args[0])
        return provider(*args)
    monkeypatch.setattr(elder_data, "fetch_chart", fetch)
    result = yahoo.download_yahoo(config, path, progress_callback=events.append,
                                  cancel_check=lambda: any(e["completed"] == 1 for e in events))
    assert calls == ["AAPL"]
    assert result["status"] == "cancelled"
    assert (result["success"], result["cancelled"], result["failed"]) == (1, 1, 0)
    assert [r["status"] for r in result["outcomes"]] == ["success", "cancelled"]
    assert (Path(config["data_root"]) / "daily/AAPL.csv").is_file()


def test_cancel_before_start_never_calls_provider(tmp_path, monkeypatch):
    config, path = make_config(tmp_path)
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    result = yahoo.download_yahoo(config, path, cancel_check=lambda: True)
    assert result["status"] == "cancelled" and result["cancelled"] == 2


def test_outer_deadline_accounts_all_unstarted_symbols(tmp_path, monkeypatch):
    config, path = make_config(tmp_path, outer_timeout_seconds=0.001)
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    original = elder_data.make_calendar
    def slow_calendar(*args):
        time.sleep(0.01)
        return original(*args)
    monkeypatch.setattr(elder_data, "make_calendar", slow_calendar)
    result = yahoo.download_yahoo(config, path)
    assert result["status"] == "failed" and result["failed"] == 2
    assert all(r["error_type"] == "OuterTimeout" for r in result["outcomes"])


def test_rejects_scope_changes_and_ashare_root_before_writing(tmp_path, monkeypatch):
    config, path = make_config(tmp_path)
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    with pytest.raises(ValueError, match="distinct members"):
        yahoo.download_yahoo(config, path, only=["TSLA"])
    assert not Path(config["data_root"]).exists()
    config["data_root"] = str(yahoo.PROJECT_ROOT / "runtime_layer/data")
    with pytest.raises(ValueError, match="independent"):
        yahoo.download_yahoo(config, path)


def test_existing_successful_date_range_cannot_be_overwritten(tmp_path, monkeypatch):
    config, path = make_config(tmp_path)
    monkeypatch.setattr(elder_data, "fetch_chart", provider)
    first = yahoo.download_yahoo(config, path)
    raw = Path(config["data_root"]) / "raw/AAPL.json"
    original = raw.read_bytes()
    config["start"] = "2023-12-01"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    second = yahoo.download_yahoo(config, path)
    assert second["status"] == "failed"
    assert "market identity differs" in second["outcomes"][0]["error"]
    assert raw.read_bytes() == original
    assert Path(first["report_path"]).exists()
    assert not (Path(config["data_root"]) / ".yahoo_download.lock").exists()


def test_conflicting_task_cannot_write_or_remove_another_download_lock(tmp_path, monkeypatch):
    config, path = make_config(tmp_path)
    root = Path(config["data_root"])
    root.mkdir()
    lock = root / ".yahoo_download.lock"
    lock.write_text('private-owner-lock', encoding="utf-8")
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    report = yahoo.download_yahoo(config, path)
    assert report["status"] == "failed" and report["failed"] == 2
    assert all("data_root lock" in row["error"] for row in report["outcomes"])
    assert lock.read_text(encoding="utf-8") == 'private-owner-lock'
    assert not (root / "calendar.csv").exists()
    assert not (root / "raw").exists()


def test_relative_config_paths_do_not_depend_on_process_cwd(tmp_path, monkeypatch):
    config, path = make_config(tmp_path)
    config["universe_file"] = "symbols.txt"
    config["data_root"] = "data_us"
    config["report_root"] = "reports"
    config["price_adjustment"] = "vendor_OHLC * adjclose/vendor_close"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(yahoo, "PROJECT_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = yahoo.load_yahoo_config("config.json")
    assert loaded["universe_file"] == str(tmp_path / "symbols.txt")
    assert loaded["data_root"] == str(tmp_path / "data_us")
    assert loaded["price_adjustment"] == yahoo.ADJUSTMENT


def test_wrong_provider_symbol_is_failed_before_success_metadata_and_audit(tmp_path, monkeypatch):
    config, path = make_config(tmp_path, symbols=("AAPL",))
    def wrong_provider(*args):
        body, request = provider(*args)
        response = json.loads(body)
        response["chart"]["result"][0]["meta"]["symbol"] = "MSFT"
        return json.dumps(response).encode(), request
    monkeypatch.setattr(elder_data, "fetch_chart", wrong_provider)
    report = yahoo.download_yahoo(config, path)
    assert report["status"] == "failed" and "MSFT" in report["outcomes"][0]["error"]
    root = Path(config["data_root"])
    assert json.loads((root / "metadata/AAPL.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert (root / "raw/AAPL.json").exists()
    assert not (root / "daily/AAPL.csv").exists()
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    with pytest.raises(SystemExit) as exc:
        download.main(["--source", "yahoo", "--config", str(path), "--audit-only"])
    assert exc.value.code == 2


@pytest.mark.parametrize("bad_identity", ["provider_symbol", "source"])
def test_audit_rejects_preexisting_bad_identity_without_rewriting_it(tmp_path, monkeypatch, bad_identity):
    config, path = make_config(tmp_path, symbols=("AAPL",))
    monkeypatch.setattr(elder_data, "fetch_chart", provider)
    assert yahoo.download_yahoo(config, path)["status"] == "completed"
    root = Path(config["data_root"])
    raw, metadata = root / "raw/AAPL.json", root / "metadata/AAPL.json"
    record = json.loads(metadata.read_text(encoding="utf-8"))
    if bad_identity == "provider_symbol":
        response = json.loads(raw.read_bytes())
        response["chart"]["result"][0]["meta"]["symbol"] = "MSFT"
        raw.write_bytes(json.dumps(response).encode())
        record["raw_sha256"] = elder_data.sha256(raw)
    else:
        record["source"] = "baostock"
    metadata.write_text(json.dumps(record), encoding="utf-8")
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(elder_data, "fetch_chart", forbid)
    audit = yahoo.audit_yahoo(config, path)
    assert audit["failed_stocks"] == 1 and not audit["all_hashes_and_source_reconstruction_verified"]
    assert yahoo.download_yahoo(config, path)["status"] == "failed"
    assert all(p.read_bytes() == data for p, data in original.items())
