"""Contracts for global EFI confirmation, with and without the ATR extension."""
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

import research_layer.user_trend as engine


POLICIES = ("OR", "AND_EXPANDED", "AND_BAND")


def bars(values):
    dates = pd.bdate_range("2025-01-06", periods=len(values))
    return pd.DataFrame([
        {"symbol": "000001.SZ", "datetime": d, "open": o, "high": h,
         "low": low, "close": c, "volume": 1000., "freq": "daily", "adjust": "pre"}
        for d, (o, h, low, c) in zip(dates, values)
    ])


def calendar(data):
    days = pd.date_range(data.datetime.iloc[0], data.datetime.iloc[-1])
    return pd.DataFrame({"date": days, "is_trading_day": days.dayofweek < 5})


def observations(monkeypatch, *, efi=(-1., -4., -8.), slope=(1., 1.)):
    def build(frame, cal, *, atr_period):
        out = frame.copy()
        out["daily_ema13"] = [99.] + [49.]*(len(out)-1)
        out["daily_ema26"] = [101.] + [50.]*(len(out)-1)
        out["atr"] = 4.
        for key, value in zip(("efi2", "efi2_previous", "efi2_previous2"), efi):
            out[key] = value
        out["efi13"] = -999.
        out["weekly_ema13"] = 90.
        out["weekly_ema26"] = 100.
        out["weekly_ema13_slope"] = slope[0]
        out["weekly_ema26_slope"] = slope[1]
        out["weekly_completed_count"] = 27
        out["indicators_ready"] = True
        out["daily_available_at"] = [datetime.combine(d.date(), time(15)) for d in out.datetime]
        out["weekly_available_at"] = [datetime.combine(d.date()-timedelta(days=7), time(15)) for d in out.datetime]
        return out
    monkeypatch.setattr(engine, "build_user_trend_features", build)


def run(data, policy="OR", **kwargs):
    return engine.run_user_trend(data, calendar(data), start=data.datetime.iloc[0],
                                 end=data.datetime.iloc[-1], entry_combination=policy, **kwargs)


@pytest.mark.parametrize("low,efi,expected,raw_flags", [
    (99.5, (1., 1., 1.), (True, False, False), (True, False)),
    (99.5, (-1., -4., -8.), (True, True, True), (True, True)),
    (97.5, (-1., -4., -8.), (True, True, False), (False, True)),
    (95., (-1., -4., -8.), (True, True, False), (False, True)),
    (94.9, (-1., -4., -8.), (False, False, False), (False, False)),
    (99., (-1., -4., -8.), (True, True, True), (True, True)),
    (101., (-1., -4., -8.), (True, True, True), (True, True)),
    (101.1, (-1., -4., -8.), (False, False, False), (False, False)),
])
def test_entry_truth_table_preserves_real_branch_observations(monkeypatch, low, efi, expected, raw_flags):
    observations(monkeypatch, efi=efi)
    data = bars([(102, 103, low, 102), (100, 101, 99.5, 100), (100, 101, 99.5, 100)])
    for policy, allowed in zip(POLICIES, expected):
        result = run(data, policy)
        assert bool(result["trades"]) is allowed
        observed = result["decisions"][0]
        assert (observed["branch_a_band_touch"], observed["branch_b_atr_efi"]) == raw_flags
        assert observed["entry_ready"] is allowed
        assert result["settings"]["entry_combination"] == policy
        if allowed:
            assert result["trades"][0]["entry_date"] == "2025-01-07"
            reasons = (["ema_band_pullback"] if raw_flags[0] else []) + (["one_atr_efi_recovery"] if raw_flags[1] else [])
            assert result["trades"][0]["entry_reasons"] == reasons


@pytest.mark.parametrize("efi", [(-2., -4., -8.), (0., -4., -8.), (-1., -4., -2.),
                                  (-1., 0., -8.), (-1., -1., -8.)])
@pytest.mark.parametrize("policy", ["AND_EXPANDED", "AND_BAND"])
def test_and_requires_negative_twice_rising_efi_with_strict_half_ratio(monkeypatch, efi, policy):
    observations(monkeypatch, efi=efi)
    data = bars([(100, 101, 99.5, 100)]*3)
    result = run(data, policy)
    assert result["decisions"][0]["branch_a_band_touch"] is True
    assert result["decisions"][0]["efi_recovery"] is False
    assert result["fills"] == []


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("slope", [(0., 1.), (1., -1.)])
def test_weekly_filter_remains_required_for_every_entry_policy(monkeypatch, policy, slope):
    observations(monkeypatch, slope=slope)
    result = run(bars([(100, 101, 99.5, 100)]*3), policy)
    assert result["decisions"][0]["branch_a_band_touch"] is True
    assert result["decisions"][0]["branch_b_atr_efi"] is True
    assert result["decisions"][0]["weekly_up"] is False
    assert result["fills"] == []


@pytest.mark.parametrize("path", ["ohlc", "olhc"])
def test_matching_entries_retain_exact_stop_trail_cost_and_timing(monkeypatch, path):
    observations(monkeypatch)
    data = bars([(100, 101, 99.5, 100), (100, 103, 99, 103), (103, 106, 102, 105.5)])
    original = run(data, "OR", intraday_path=path)
    for policy in ("AND_EXPANDED", "AND_BAND"):
        result = run(data, policy, intraday_path=path)
        for key in ("fills", "trades", "equity_curve", "decisions", "signals", "metrics"):
            assert result[key] == original[key]
        assert result["trades"][0]["hard_stop"] == 98
        assert result["trades"][0]["active_stop"] == 105
        assert policy in result["assumptions"][2]
        assert result["assumptions"][:2] == original["assumptions"][:2]
        assert result["assumptions"][3:] == original["assumptions"][3:]
    expected = 1.05*.999**2 if path == "ohlc" else 1.055*.999
    assert original["metrics"]["final_nav"] == pytest.approx(expected)


@pytest.mark.parametrize("policy", [None, True, "and", "AND", "choose_best", [], {}])
def test_invalid_entry_policy_is_rejected_before_data_processing(policy):
    with pytest.raises(ValueError, match="entry_combination"):
        engine.run_user_trend(None, None, start="2025-01-01", end="2025-12-31", entry_combination=policy)


def real_sample():
    size = 900
    rng = np.random.default_rng(112)
    close = 50 + .03*np.arange(size) + rng.normal(0, .35, size)
    opening = np.r_[close[0], close[:-1]]
    dates = pd.bdate_range("2020-01-06", periods=size)
    data = pd.DataFrame({"symbol": "000001.SZ", "datetime": dates, "open": opening,
                         "high": np.maximum(opening, close)+.05, "low": np.minimum(opening, close)-.05,
                         "close": close, "volume": 1000., "adjust": "pre", "freq": "daily"})
    days = pd.date_range(dates[0], dates[-1].date()+timedelta(days=7))
    return data, pd.DataFrame({"date": days, "is_trading_day": days.dayofweek < 5})


@pytest.mark.parametrize("mode", ["next_open", "signal_close"])
@pytest.mark.parametrize("path", ["ohlc", "olhc"])
def test_default_and_explicit_or_outputs_are_identical(mode, path):
    data, cal = real_sample()
    args = {"start": data.datetime.iloc[180], "end": data.datetime.iloc[-1],
            "entry_mode": mode, "intraday_path": path}
    default = engine.run_user_trend(data, cal, **args)
    explicit = engine.run_user_trend(data, cal, entry_combination="OR", **args)
    assert len(default["fills"]) >= 2
    assert default == explicit


@pytest.mark.parametrize("policy", ["AND_EXPANDED", "AND_BAND"])
@pytest.mark.parametrize("mode", ["next_open", "signal_close"])
def test_and_real_features_ignore_future_price_changes(policy, mode):
    data, cal = real_sample()
    args = {"start": data.datetime.iloc[180], "end": data.datetime.iloc[-1],
            "entry_mode": mode, "entry_combination": policy}
    original = engine.run_user_trend(data, cal, **args)
    altered = data.copy()
    altered.loc[851:, ["open", "high", "low", "close"]] *= 1.7
    future_changed = engine.run_user_trend(altered, cal, **args)
    cutoff = data.datetime.iloc[850].date().isoformat()
    assert len([f for f in original["fills"] if f["date"] <= cutoff]) >= 2
    for key in ("fills", "equity_curve", "decisions", "signals"):
        assert [r for r in original[key] if r["date"] <= cutoff] == [r for r in future_changed[key] if r["date"] <= cutoff]
