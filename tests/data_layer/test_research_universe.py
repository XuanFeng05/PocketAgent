from pathlib import Path
import json
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from data_layer.research_universe import audit_local_pre, classify_historical_membership, normalize_symbol_master, fetch_baostock_master


def master_rows():
    return pd.DataFrame([
        {"code": "sh.600001", "code_name": "Old", "ipoDate": "1999-01-01", "outDate": "2020-02-01", "status": "0", "type": "1"},
        {"code": "sz.000001", "code_name": "Bank", "ipoDate": "1991-01-01", "outDate": "", "status": "1", "type": "1"},
        {"code": "sh.000001", "code_name": "Index", "ipoDate": "1990-01-01", "outDate": "", "status": "1", "type": "2"},
        {"code": "sz.200001", "code_name": "B", "ipoDate": "1991-01-01", "outDate": "", "status": "1", "type": "1"},
        {"code": "bj.920001", "code_name": "BJ", "ipoDate": "2025-01-01", "outDate": "", "status": "1", "type": "1"},
    ])


def test_master_keeps_delisted_and_bj_and_records_non_a_exclusions():
    result = normalize_symbol_master(master_rows())
    assert result["symbol"].tolist() == ["000001.SZ", "600001.SH", "920001.BJ"]
    assert result.loc[result["symbol"].eq("600001.SH"), "status"].item() == "delisted"
    assert {item["reason"] for item in result.attrs["excluded_records"]} == {"not_stock", "not_supported_A_share_code"}


def test_historical_membership_does_not_filter_by_current_survival():
    result = classify_historical_membership(normalize_symbol_master(master_rows()), "2020-01-01", "2020-01-31")
    by_symbol = result.set_index("symbol")
    assert by_symbol.loc["600001.SH", "membership_status"] == "eligible"
    assert by_symbol.loc["920001.BJ", "membership_reason"] == "not_yet_listed"
    later = classify_historical_membership(normalize_symbol_master(master_rows()), "2020-02-01", "2020-02-28")
    assert later.set_index("symbol").loc["600001.SH", "membership_reason"] == "delisted_before_interval"


def test_unknown_lifetime_is_not_treated_as_survivor():
    data = master_rows()
    data.loc[0, "outDate"] = ""
    result = classify_historical_membership(normalize_symbol_master(data), "2020-01-01", "2020-12-31")
    assert result.set_index("symbol").loc["600001.SH", "membership_status"] == "unknown"


def test_conflicting_metadata_and_invalid_dates_fail():
    data = master_rows()
    other = data.iloc[[0]].copy()
    other["ipoDate"] = "1998-01-01"
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_symbol_master(pd.concat([data, other]))
    data.loc[0, "outDate"] = "1990-01-01"
    with pytest.raises(ValueError, match="precedes"):
        normalize_symbol_master(data)


def write_market(tmp_path, *, missing=False):
    root = tmp_path / "market_parts"
    calendar = root / "_global" / "trade_calendar.parquet"
    calendar.parent.mkdir(parents=True)
    pd.DataFrame({"date": pd.date_range("2024-01-01", "2024-01-07"),
                  "is_trading_day": [False, True, True, True, True, False, False]}).to_parquet(calendar, index=False)
    path = root / "bucket=001" / "000001_SZ" / "daily_pre.parquet"
    path.parent.mkdir(parents=True)
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    if missing: dates.remove("2024-01-03")
    pd.DataFrame({"symbol": "000001.SZ", "datetime": pd.to_datetime(dates), "open": 10., "high": 11., "low": 9.,
                  "close": 10., "volume": 1000, "adjust": "pre", "freq": "daily"}).to_parquet(path, index=False)
    return path


def test_audit_reports_missing_targets_and_keeps_inputs_unchanged(tmp_path):
    path = write_market(tmp_path)
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    result = audit_local_pre(tmp_path, tmp_path / "reports", target_symbols=["000001.SZ", "600000.SH"])
    rows = {row["symbol"]: row for row in result["symbols"]}
    assert rows["000001.SZ"]["status"] == "ready_for_feature_validation"
    assert rows["600000.SH"]["status"] == "missing"
    assert result["summary"]["all_A_share_coverage_verified"] is False
    assert result["summary"]["adjustment_versions_verified"] is False
    assert (Path(result["path"]) / "coverage.json").is_file()
    assert path.read_bytes() == before and path.stat().st_mtime_ns == modified


def test_calendar_gap_is_not_assumed_to_be_suspension(tmp_path):
    write_market(tmp_path, missing=True)
    result = audit_local_pre(tmp_path)
    row = result["symbols"][0]
    assert row["missing_trading_dates"] == ["2024-01-03"]
    assert row["status"] == "needs_review"
    assert "missing_trading_bars_unexplained" in row["issues"]


def test_bad_pre_adjustment_flag_and_duplicate_dates_are_reported(tmp_path):
    path = write_market(tmp_path)
    data = pd.read_parquet(path)
    data.loc[0, "adjust"] = "none"
    data.loc[1, "datetime"] = data.loc[0, "datetime"]
    data.to_parquet(path, index=False)
    row = audit_local_pre(tmp_path)["symbols"][0]
    assert row["duplicate_dates"] == 1
    assert "not_uniform_pre_adjusted" in row["issues"]
    assert "duplicate_dates" in row["issues"]


def test_corrupt_existing_shard_is_not_duplicated_as_missing(tmp_path):
    path = write_market(tmp_path)
    path.write_bytes(b"not a parquet file")
    result = audit_local_pre(tmp_path, target_symbols=["000001.SZ", "600000.SH"])
    assert len(result["symbols"]) == 2
    assert result["summary"]["statuses"] == {"invalid": 1, "missing": 1}


def test_target_audit_reconciles_only_requested_symbols(tmp_path):
    write_market(tmp_path)
    result = audit_local_pre(tmp_path, target_symbols=["600000.SH"])
    assert result["summary"]["target_count"] == 1
    assert result["summary"]["local_shards_total"] == 1
    assert result["symbols"] == [{"symbol": "600000.SH", "status": "missing", "path": None, "rows": 0, "issues": ["no_daily_pre_shard"]}]


def test_empty_calendar_is_reported_without_certifying_coverage(tmp_path):
    write_market(tmp_path)
    calendar = tmp_path / "market_parts/_global/trade_calendar.parquet"
    pd.DataFrame({"date": [], "is_trading_day": []}).to_parquet(calendar, index=False)
    result = audit_local_pre(tmp_path)
    assert result["calendar"]["status"] == "invalid"
    assert result["symbols"][0]["status"] == "needs_review"


def test_provider_snapshot_retains_raw_and_does_not_certify_full_market(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        destination = Path(command[-1])
        source = master_rows()
        payload = {"status": "completed", "fields": source.columns.tolist(), "rows": source.values.tolist()}
        (destination / "provider_raw.json").write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="test snapshot", stderr="")
    monkeypatch.setattr("data_layer.research_universe.subprocess.run", fake_run)
    result = fetch_baostock_master(tmp_path)
    assert result["status"] == "completed"
    assert result["coverage_verified"] is False
    assert result["exchange_counts"] == {"SH": 1, "SZ": 1, "BJ": 1}
    assert result["delisted_records"] == 1
    assert (Path(result["path"]) / "provider_raw.json").is_file()
    assert (Path(result["path"]) / "excluded_records.json").is_file()


def test_provider_timeout_is_failed_and_keeps_received_diagnostics(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial diagnostic")
    monkeypatch.setattr("data_layer.research_universe.subprocess.run", timeout)
    result = fetch_baostock_master(tmp_path, timeout_seconds=2)
    assert result["status"] == "failed"
    assert result["coverage_verified"] is False
    assert "observed_equities" not in result
    assert (Path(result["path"]) / "provider.log").read_text() == "partial diagnostic"
