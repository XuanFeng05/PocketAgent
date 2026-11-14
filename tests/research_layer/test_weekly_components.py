"""Separate weekly entry selection from weekly liquidation with priced examples."""
import json
from datetime import datetime, time

import numpy as np
import pandas as pd
import pytest

import research_layer.triple_screen as engine


COMPONENT_VARIANTS = ("drop_weekly_entry", "drop_weekly_exit")


def bars(values):
    dates = pd.bdate_range("2025-01-06", periods=len(values))
    return pd.DataFrame([
        {"symbol": "000001.SZ", "datetime": day, "open": opening, "high": high,
         "low": low, "close": close, "volume": 1000., "freq": "daily", "adjust": "pre"}
        for day, (opening, high, low, close) in zip(dates, values)
    ])


def observed_features(monkeypatch, *, setup=(1,), down=(), unready=()):
    """Provide consistent observed facts, including the full-rule entry_ready flag."""
    def build(frame, calendar, settings):
        out = frame.copy()
        indices = range(len(out))
        out["weekly_hist"] = [0. if i in down else 1. for i in indices]
        out["weekly_hist_previous"] = .5
        out["weekly_up"] = [i not in down for i in indices]
        out["indicators_ready"] = [i not in unready for i in indices]
        out["entry_ready"] = [i in setup and i not in down and i not in unready for i in indices]
        out["force_index_2"] = [-1. if i in setup else 1. for i in indices]
        out["weekly_completed_count"] = 100
        out["session_close"] = [datetime.combine(day.date(), time(16 if settings.is_us else 15))
                                for day in out["datetime"]]
        out["weekly_available_at"] = out["session_close"]
        return out
    monkeypatch.setattr(engine, "build_triple_features", build)


def test_removing_entry_gate_still_exits_when_weekly_direction_is_down(monkeypatch):
    observed_features(monkeypatch, down=(1, 2))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13), (14, 16, 13, 15)])
    results = {variant: engine.run_triple_screen(data, None, {"cost_bps": 10}, variant=variant)
               for variant in ("full", "drop_weekly", *COMPONENT_VARIANTS)}

    for variant in ("full", "drop_weekly_exit"):
        assert results[variant]["fills"] == []
        assert results[variant]["metrics"]["final_nav"] == 1
    entry_only = results["drop_weekly_entry"]
    assert [fill["adjusted_price"] for fill in entry_only["fills"]] == [12, 14]
    assert entry_only["fills"][-1]["reason"] == "weekly_filter_invalidated"
    assert entry_only["metrics"]["final_nav"] == pytest.approx(14 / 12 * .999 ** 2)
    assert results["drop_weekly"]["metrics"]["final_nav"] == pytest.approx(15 / 12 * .999)
    signal = entry_only["signals"][0]
    assert signal["weekly_up"] is False
    assert signal["entry_ready"] is False
    assert signal["entry_rule_passed"] is True
    assert signal["disabled_filters"] == ["weekly_direction"]
    assert entry_only["disabled_filters"] == ["weekly_direction"]
    assert entry_only["research_variant"] == "drop_weekly_entry"


def test_removing_weekly_exit_preserves_entry_and_fixed_initial_stop(monkeypatch):
    observed_features(monkeypatch, down=(2, 3))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13),
                 (14, 16, 13, 15), (12, 14, 7, 13)])
    settings = {"cost_bps": 10}
    full = engine.run_triple_screen(data, None, settings)
    entry_only = engine.run_triple_screen(data, None, settings, variant="drop_weekly_entry")
    exit_only = engine.run_triple_screen(data, None, settings, variant="drop_weekly_exit")
    for result in (full, entry_only):
        assert result["fills"][-1]["reason"] == "weekly_filter_invalidated"
        assert result["metrics"]["final_nav"] == pytest.approx(14 / 12 * .999 ** 2)
    assert exit_only["equity_curve"][1]["exposure"] == 1
    assert exit_only["equity_curve"][1]["nav"] == pytest.approx(15 / 12 * .999)
    assert exit_only["equity_curve"][1]["pending_exit_reason"] is None
    assert exit_only["fills"][-1]["reason"] == "initial_stop"
    assert exit_only["fills"][-1]["adjusted_price"] == 8
    assert exit_only["metrics"]["final_nav"] == pytest.approx(8 / 12 * .999 ** 2)
    assert exit_only["decisions"][1]["weekly_up"] is False
    assert exit_only["decisions"][1]["entry_ready"] is False
    assert exit_only["disabled_filters"] == ["weekly_exit"]
    assert exit_only["research_variant"] == "drop_weekly_exit"


@pytest.mark.parametrize("variant", COMPONENT_VARIANTS)
@pytest.mark.parametrize("version", ["triple_screen_pre_v1", "triple_screen_us_pre_v1"])
def test_component_variants_preserve_market_entry_day_stop_rules(monkeypatch, variant, version):
    observed_features(monkeypatch, down=(1,) if variant == "drop_weekly_entry" else ())
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 7, 13)])
    result = engine.run_triple_screen(data, None, {"version": version, "cost_bps": 10}, variant=variant)
    assert result["fills"][0]["adjusted_price"] == 12
    assert result["metrics"]["ambiguous_entry_count"] == 1
    if version == "triple_screen_us_pre_v1":
        assert [fill["adjusted_price"] for fill in result["fills"]] == [12, 8]
        assert result["fills"][-1]["reason"] == "entry_day_stop_conservative"
        assert result["metrics"]["final_nav"] == pytest.approx(8 / 12 * .999 ** 2)
        assert result["metrics"]["final_exposure"] == 0
        assert result["equity_curve"][0]["valuation_time"] == "2025-01-08T16:00:00"
    else:
        assert len(result["fills"]) == 1
        assert result["metrics"]["final_nav"] == pytest.approx(13 / 12 * .999)
        assert result["metrics"]["final_exposure"] == 1
        assert result["equity_curve"][0]["pending_exit_reason"] == "entry_day_stop_t_plus_one"


@pytest.mark.parametrize("variant", COMPONENT_VARIANTS)
def test_component_variants_preserve_pending_stop_through_suspension(monkeypatch, variant):
    observed_features(monkeypatch, down=(1,) if variant == "drop_weekly_entry" else (3,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 7, 13),
                 (13, 14, 12, 13), (10, 12, 9, 11)])
    data.loc[3, "volume"] = 0
    result = engine.run_triple_screen(data, None, variant=variant)
    assert result["fills"][1]["status"] == "blocked"
    assert result["equity_curve"][1]["pending_exit_reason"] == "entry_day_stop_t_plus_one"
    assert result["fills"][-1]["reason"] == "entry_day_stop_t_plus_one"
    assert result["fills"][-1]["adjusted_price"] == 10
    assert result["metrics"]["final_nav"] == pytest.approx(10 / 12)


@pytest.mark.parametrize("variant", COMPONENT_VARIANTS)
@pytest.mark.parametrize("blocked_by", ["force", "breakout", "previous_volume", "current_volume"])
def test_component_variants_keep_other_entry_constraints(monkeypatch, variant, blocked_by):
    observed_features(monkeypatch, setup=() if blocked_by == "force" else (1,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    if blocked_by == "breakout":
        data.loc[2, ["open", "high", "low", "close"]] = [11, 12, 10, 11]
    elif blocked_by == "previous_volume":
        data.loc[1, "volume"] = 0
    elif blocked_by == "current_volume":
        data.loc[2, "volume"] = 0
    result = engine.run_triple_screen(data, None, variant=variant)
    assert result["fills"] == []
    assert result["metrics"]["final_nav"] == 1


def test_all_variants_preserve_identical_warmup_and_full_baseline(monkeypatch):
    observed_features(monkeypatch, setup=(3,), down=(4,), unready=(0, 1, 2))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13),
                 (13, 14, 12, 13), (13, 16, 12, 15), (15, 17, 14, 16)])
    results = {variant: engine.run_triple_screen(data, None, variant=variant)
               for variant in engine.RESEARCH_VARIANTS}
    default = engine.run_triple_screen(data, None)
    assert default == results["full"]
    assert [fill["adjusted_price"] for fill in default["fills"]] == [14, 15]
    assert default["metrics"]["final_nav"] == pytest.approx(15 / 14)
    for result in results.values():
        assert (result["start"], result["end"], result["warmup"]) == (
            default["start"], default["end"], default["warmup"])
        assert result["warmup"]["bars"] == 4


def real_feature_sample():
    dates = pd.bdate_range("2024-01-01", periods=120)
    index = np.arange(len(dates))
    close = 20 + index * .05 + np.sin(index / 3)
    opening = np.r_[close[0], close[:-1]]
    data = pd.DataFrame({
        "symbol": "000001.SZ", "datetime": dates, "open": opening,
        "high": np.maximum(opening, close) + .1 + 2 * (index % 3 == 0),
        "low": np.minimum(opening, close) - .3, "close": close,
        "volume": 1000., "freq": "daily", "adjust": "pre",
    })
    calendar_dates = pd.date_range("2024-01-01", "2024-07-07")
    calendar = pd.DataFrame({"date": calendar_dates, "is_trading_day": calendar_dates.dayofweek < 5})
    settings = {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "warmup_weeks": 3}
    return data, calendar, settings


@pytest.mark.parametrize("variant", COMPONENT_VARIANTS)
def test_real_features_future_perturbation_keeps_prior_traded_results(variant):
    data, calendar, settings = real_feature_sample()
    original = engine.run_triple_screen(data, calendar, settings, variant=variant)
    changed = data.copy()
    changed.loc[90:, ["open", "high", "low", "close"]] *= 2
    perturbed = engine.run_triple_screen(changed, calendar, settings, variant=variant)
    cutoff = data.loc[89, "datetime"].date().isoformat()
    earlier_fills = [row for row in original["fills"] if row["date"] <= cutoff]
    assert len(earlier_fills) >= 2
    for field in ("equity_curve", "fills", "decisions"):
        assert [row for row in original[field] if row["date"] <= cutoff] == [
            row for row in perturbed[field] if row["date"] <= cutoff]
    json.dumps(original, allow_nan=False)


@pytest.mark.parametrize("version", ["triple_screen_pre_v1", "triple_screen_us_pre_v1"])
@pytest.mark.parametrize("input_close", ["13:00:00", "invalid-close"])
def test_session_close_is_recomputed_from_selected_market_calendar(version, input_close):
    from feature_layer.triple_screen import build_triple_features

    data, calendar, settings = real_feature_sample()
    settings["version"] = version
    is_us = version == "triple_screen_us_pre_v1"
    early_close_date = data.loc[49, "datetime"]
    if is_us:
        calendar["session_close"] = [
            datetime.combine(day.date(), time(13 if day == early_close_date else 16))
            for day in calendar["date"]
        ]
    expected = engine.run_triple_screen(data, calendar, settings)
    contaminated = data.copy()
    contaminated["session_close"] = [
        f"{day.date().isoformat()}T{input_close}" if input_close == "13:00:00" else input_close
        for day in data["datetime"]
    ]
    actual = engine.run_triple_screen(contaminated, calendar, settings)
    assert actual == expected
    assert actual["metrics"]["position_changes"] > 0
    features = build_triple_features(contaminated, calendar, settings)
    expected_closes = [
        datetime.combine(day.date(), time(13 if is_us and day == early_close_date else 16 if is_us else 15))
        for day in data["datetime"]
    ]
    assert features["session_close"].tolist() == expected_closes
    expected_hour = 13 if is_us else 15
    observed_day = early_close_date.date().isoformat()
    early_day_equity = next(row for row in actual["equity_curve"] if row["date"] == observed_day)
    assert early_day_equity["valuation_time"] == f"{observed_day}T{expected_hour}:00:00"
