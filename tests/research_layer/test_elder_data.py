"""Small synthetic parser fixtures; never used as research market data."""
from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from research_layer.elder_data import audit_frame, load_symbols, make_calendar, parse_chart, restore_symbol, sha256


def payload(*, volume=1200, low=98.0, missing_close=False):
    day = int(datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc).timestamp())
    return json.dumps({"chart": {"error": None, "result": [{
        "meta": {"currency": "USD", "instrumentType": "EQUITY"},
        "timestamp": [day], "indicators": {
            "quote": [{"open": [100.0], "high": [105.0], "low": [low],
                       "close": [None if missing_close else 104.0], "volume": [volume]}],
            "adjclose": [{"adjclose": [52.0]}]}}]}}).encode()


def test_adjustment_retains_vendor_prices_and_share_volume():
    frame, metadata = parse_chart(payload(), "AAPL", "2024-01-01", "2024-01-31")
    bar = frame.iloc[0]
    assert (bar.open, bar.high, bar.low, bar.close) == (50.0, 52.5, 49.0, 52.0)
    assert (bar.vendor_open, bar.vendor_close, bar.adjust_factor, bar.volume) == (100.0, 104.0, 0.5, 1200)
    assert bar.datetime == "2024-01-02" and bar.adjust == "pre"
    assert metadata["volume_unit"] == "shares"


def test_zero_volume_is_retained_but_missing_price_is_rejected():
    frame, _ = parse_chart(payload(volume=0), "AAPL", "2024-01-01", "2024-01-31")
    assert len(frame) == 1 and frame.iloc[0].volume == 0 and not frame.iloc[0].is_tradeable
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        parse_chart(payload(missing_close=True), "AAPL", "2024-01-01", "2024-01-31")


def test_invalid_OHLC_and_outside_requested_period_fail_closed():
    with pytest.raises(ValueError, match="Illegal OHLC"):
        parse_chart(payload(low=102), "AAPL", "2024-01-01", "2024-01-31")
    with pytest.raises(ValueError, match="outside"):
        parse_chart(payload(), "AAPL", "2024-02-01", "2024-02-29")


def test_missing_session_is_not_silently_treated_as_suspension():
    frame, _ = parse_chart(payload(volume=0), "AAPL", "2024-01-01", "2024-01-31")
    calendar = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                             "is_trading_day": [False, True, True]})
    with pytest.raises(ValueError, match="missing sessions=.*2024-01-03"):
        audit_frame(frame, calendar, "2024-01-01", "2024-01-03")


def test_calendar_includes_carter_closure_and_early_close():
    january = make_calendar("2025-01-01", "2025-01-15").set_index("date")
    assert not january.loc["2025-01-09", "is_trading_day"]
    assert january.loc["2025-01-09", "session_close"] == ""
    november = make_calendar("2025-11-24", "2025-11-30").set_index("date")
    assert november.loc["2025-11-28", "session_close"] == "2025-11-28T13:00:00"


def test_fixed_universe_count_and_duplicates_are_not_relaxed(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("AAPL\nAAPL\n")
    with pytest.raises(ValueError, match="distinct"):
        load_symbols(path, expected=2)
    path.write_text("AAPL\nMSFT\n")
    assert load_symbols(path, expected=2) == ["AAPL", "MSFT"]


def test_retry_preserves_failed_response_and_metadata(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "AAPL.json"
    meta = tmp_path / "metadata" / "AAPL.json"
    raw.parent.mkdir()
    meta.parent.mkdir()
    old = b'{"error":"previous provider failure"}'
    raw.write_bytes(old)
    meta.write_text('{"status":"failed"}')
    old_hash = sha256(raw)
    monkeypatch.setattr("research_layer.elder_data.fetch_chart", lambda *a: (
        payload(), {"symbol": "AAPL", "start": "2024-01-01", "end": "2024-01-02"}))
    calendar = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "is_trading_day": [False, True]})
    result = restore_symbol("AAPL", "2024-01-01", "2024-01-02", tmp_path, calendar, 20)
    assert result["status"] == "success"
    archive = tmp_path / "failed_attempts" / "AAPL"
    assert (archive / f"{old_hash}.json").read_bytes() == old
    assert json.loads((archive / f"{old_hash}.metadata.json").read_text())["status"] == "failed"
