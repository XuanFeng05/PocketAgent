"""Behavioral tests of causality, T+1 and gap prices, not implementation mirrors."""
import json
from datetime import datetime, time

import numpy as np
import pandas as pd
import pytest

import research_layer.triple_screen as engine


def bars(values):
    """OHLC tuples; include two prior rows to define a candidate stop."""
    dates = pd.bdate_range("2025-01-06", periods=len(values))
    return pd.DataFrame([{"symbol": "000001.SZ", "datetime": day, "open": o, "high": h,
                          "low": l, "close": c, "volume": 1000., "amount": 10000.,
                          "freq": "daily", "adjust": "pre"}
                         for day, (o, h, l, c) in zip(dates, values)])


def fake_features(monkeypatch, *, setup_indices=(1,), down_indices=(), unready_indices=(), invalid=None):
    """Isolate execution from indicators with stated prior-close facts."""
    def build(frame, calendar, settings):
        out = frame.copy()
        out["weekly_hist"] = [0.0 if i in down_indices else 1.0 for i in range(len(out))]
        out["weekly_hist_previous"] = 0.5
        out["weekly_up"] = [i not in down_indices for i in range(len(out))]
        out["indicators_ready"] = [i not in unready_indices for i in range(len(out))]
        out["entry_ready"] = [i in setup_indices and i not in down_indices and i not in unready_indices for i in range(len(out))]
        out["force_index_2"] = [-1. if i in setup_indices else 1. for i in range(len(out))]
        out["weekly_completed_count"] = 100
        out["weekly_available_at"] = [datetime.combine(d.date(), time(15)) for d in out["datetime"]]
        if invalid:
            out.loc[invalid[0], invalid[1]] = invalid[2]
        return out
    monkeypatch.setattr(engine, "build_triple_features", build)


def test_intraday_entry_does_not_earn_pre_entry_or_overnight_move(monkeypatch):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    result = engine.run_triple_screen(data, None)
    assert result["metrics"]["final_nav"] == pytest.approx(13/12)
    assert result["fills"][0]["adjusted_price"] == 12
    assert result["fills"][0]["phase"] == "intraday"
    assert "execution_time" not in result["fills"][0]
    assert result["trades"][0]["initial_stop"] == 8


def test_gap_entry_uses_open_not_favorable_trigger(monkeypatch):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (15, 17, 14, 16)])
    result = engine.run_triple_screen(data, None)
    assert result["metrics"]["final_nav"] == pytest.approx(16/15)
    assert result["fills"][0]["adjusted_price"] == 15
    assert result["fills"][0]["phase"] == "open"


def test_equal_high_is_not_breakout_and_candidate_expires(monkeypatch):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (10, 12, 9, 11), (11, 15, 10, 14)])
    result = engine.run_triple_screen(data, None)
    assert result["metrics"]["final_nav"] == 1
    assert result["fills"] == []
    assert result["signals"][0]["status"] == "expired_no_breakout"


def test_daily_candidate_updates_to_latest_high(monkeypatch):
    fake_features(monkeypatch, setup_indices=(1, 2))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (10, 11, 9, 10), (10, 11.5, 9.5, 11.2)])
    result = engine.run_triple_screen(data, None)
    assert [s["entry_line"] for s in result["signals"]] == [12, 11]
    assert result["fills"][0]["adjusted_price"] == 11
    assert result["metrics"]["final_nav"] == pytest.approx(11.2/11)


def test_flat_candle_candidate_is_rejected_but_later_valid_setup_can_trade(monkeypatch):
    fake_features(monkeypatch, setup_indices=(1, 2))
    data = bars([(10, 10, 10, 10), (10, 10, 10, 10), (10.5, 12, 10, 11), (11, 13, 11, 12.5)])
    result = engine.run_triple_screen(data, None)
    assert result["signals"][0]["status"] == "rejected_nonpositive_stop_distance"
    assert result["decisions"][0]["candidate_rejection_reason"] == "rejected_nonpositive_stop_distance"
    assert result["equity_curve"][0]["nav"] == 1
    assert result["equity_curve"][0]["exposure"] == 0
    assert result["signals"][1]["status"] == "entered"
    assert result["metrics"]["position_changes"] == 1
    assert result["metrics"]["final_nav"] == pytest.approx(12.5/12)


def test_weekly_exit_latches_across_halt_and_does_not_reenter(monkeypatch):
    fake_features(monkeypatch, setup_indices=(1, 3, 4), down_indices=(2,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13),
                 (13, 14, 12, 13), (15, 18, 14, 17), (16, 18, 15, 17)])
    data.loc[3, "volume"] = 0
    result = engine.run_triple_screen(data, None)
    assert result["fills"][1]["status"] == "blocked"
    assert result["fills"][2]["reason"] == "weekly_filter_invalidated"
    assert result["fills"][2]["adjusted_price"] == 15
    assert result["equity_curve"][2]["nav"] == pytest.approx(15/12)
    assert result["equity_curve"][2]["exposure"] == 0
    assert sum(e["date"] == data.iloc[4]["datetime"].date().isoformat() and e["status"] == "applied" for e in result["fills"]) == 1


@pytest.mark.parametrize("opening, low, expected, phase", [(7, 6, 7, "open"), (11, 7, 8, "intraday")])
def test_stop_exits_at_open_on_gap_otherwise_stop(monkeypatch, opening, low, expected, phase):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13),
                 (opening, max(opening, 14), low, 14)])
    result = engine.run_triple_screen(data, None)
    assert result["fills"][-1]["adjusted_price"] == expected
    assert result["fills"][-1]["phase"] == phase
    assert result["metrics"]["final_nav"] == pytest.approx(expected/12)
    assert result["trades"][0]["net_return"] == pytest.approx(expected/12-1)


def test_entry_day_ambiguous_stop_defers_until_next_tradable_open(monkeypatch):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 7, 13),
                 (13, 14, 12, 13), (16, 18, 15, 17)])
    data.loc[3, "volume"] = 0
    result = engine.run_triple_screen(data, None)
    assert result["metrics"]["final_nav"] == pytest.approx(16/12)
    assert result["equity_curve"][0]["exposure"] == 1
    assert result["equity_curve"][0]["ambiguous_intraday_order"] is True
    assert result["equity_curve"][1]["pending_exit_reason"] == "entry_day_stop_t_plus_one"
    assert result["fills"][-1]["reason"] == "entry_day_stop_t_plus_one"
    assert result["fills"][-1]["adjusted_price"] == 16
    assert result["trades"][0]["ambiguous_intraday_order"] is True


def test_cost_is_charged_once_per_applied_entry_and_exit(monkeypatch):
    fake_features(monkeypatch, down_indices=(2,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (12, 13, 11, 12), (12, 13, 11, 12)])
    result = engine.run_triple_screen(data, None, {"cost_bps": 100})
    assert result["metrics"]["final_nav"] == pytest.approx(.99*.99)
    assert result["metrics"]["total_cost_nav_units"] == pytest.approx(.01+.0099)
    assert result["metrics"]["position_changes"] == 2


def test_start_does_not_simulate_old_positions_and_end_limits_features(monkeypatch):
    captured = []
    fake_features(monkeypatch, setup_indices=(1,))
    feature_builder = engine.build_triple_features
    def record(frame, calendar, settings):
        captured.append(len(frame))
        return feature_builder(frame, calendar, settings)
    monkeypatch.setattr(engine, "build_triple_features", record)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13), (15, 17, 14, 16), (16, 18, 15, 17)])
    result = engine.run_triple_screen(data, None, start="2025-01-09", end="2025-01-09")
    assert captured == [4]
    assert result["metrics"]["final_nav"] == 1
    assert result["fills"] == []
    assert result["bars"] == 1


def test_no_ready_period_is_reported_as_error(monkeypatch):
    fake_features(monkeypatch, setup_indices=(), unready_indices=(0, 1, 2))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    with pytest.raises(ValueError, match="No evaluable"):
        engine.run_triple_screen(data, None)


def test_missing_ready_indicator_cannot_silently_authorize_entry(monkeypatch):
    fake_features(monkeypatch, invalid=(1, "force_index_2", float("nan")))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    with pytest.raises(ValueError, match="nonfinite"):
        engine.run_triple_screen(data, None)


def test_future_weekly_timestamp_is_rejected(monkeypatch):
    fake_features(monkeypatch, invalid=(1, "weekly_available_at", pd.Timestamp("2025-01-10 15:00")))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    with pytest.raises(ValueError, match="unavailable"):
        engine.run_triple_screen(data, None)


def test_untradable_entry_expires_without_cost(monkeypatch):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    data["is_tradeable"] = [True, True, False]
    result = engine.run_triple_screen(data, None, {"cost_bps": 20})
    assert result["metrics"]["final_nav"] == 1
    assert result["signals"][0]["status"] == "expired_non_tradeable"


@pytest.mark.parametrize("source_unit", ["ns", "us"])
@pytest.mark.parametrize("feature_unit", ["ns", "us"])
def test_identical_dates_accept_different_timestamp_resolutions(monkeypatch, source_unit, feature_unit):
    fake_features(monkeypatch)
    original_builder = engine.build_triple_features
    def mixed_resolution(frame, calendar, settings):
        features = original_builder(frame, calendar, settings)
        features["datetime"] = features["datetime"].astype(f"datetime64[{feature_unit}]")
        return features
    monkeypatch.setattr(engine, "build_triple_features", mixed_resolution)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    data["datetime"] = data["datetime"].astype(f"datetime64[{source_unit}]")
    result = engine.run_triple_screen(data, None)
    assert result["metrics"]["final_nav"] == pytest.approx(13/12)
    assert result["start"] == "2025-01-08"


@pytest.mark.parametrize("variant", ["full", "drop_weekly", "drop_pullback", "drop_trigger"])
def test_real_features_future_price_changes_do_not_change_prior_results(variant):
    dates = pd.bdate_range("2024-01-01", periods=120)
    t = np.arange(len(dates))
    close = 20 + t*.05 + np.sin(t/3)
    opening = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({"symbol": "000001.SZ", "datetime": dates, "open": opening,
                          "high": np.maximum(opening, close)+.1+2*(t%3 == 0), "low": np.minimum(opening, close)-.3,
                          "close": close, "volume": 1000., "freq": "daily", "adjust": "pre"})
    calendar_dates = pd.date_range("2024-01-01", "2024-07-07")
    calendar = pd.DataFrame({"date": calendar_dates, "is_trading_day": calendar_dates.dayofweek < 5})
    settings = {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "warmup_weeks": 3}
    original = engine.run_triple_screen(frame, calendar, settings, variant=variant)
    changed = frame.copy()
    changed.loc[90:, ["open", "high", "low", "close"]] *= 2
    perturbed = engine.run_triple_screen(changed, calendar, settings, variant=variant)
    cutoff = dates[89].date().isoformat()
    for field in ("equity_curve", "fills", "decisions"):
        assert [r for r in original[field] if r["date"] <= cutoff] == [r for r in perturbed[field] if r["date"] <= cutoff]
    assert original["metrics"]["position_changes"] > 0
    json.dumps(original, allow_nan=False)


@pytest.mark.parametrize("variant", ["drop_everything", "", None, 1, {}, ["full"]])
def test_unknown_research_variant_is_rejected(variant):
    with pytest.raises(ValueError, match="variant"):
        engine.run_triple_screen([], None, variant=variant)


def test_default_and_explicit_full_keep_reference_execution(monkeypatch):
    fake_features(monkeypatch, down_indices=(2,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (12, 13, 11, 12), (15, 16, 14, 15)])
    default = engine.run_triple_screen(data, None, {"cost_bps": 100})
    explicit = engine.run_triple_screen(data, None, {"cost_bps": 100}, variant="full")
    assert default == explicit
    assert default["metrics"]["final_nav"] == pytest.approx(.99 * (15/12) * .99)
    assert [event["adjusted_price"] for event in default["fills"]] == [12, 15]
    assert [event["reason"] for event in default["fills"]] == ["prior_high_breakout", "weekly_filter_invalidated"]
    assert default["research_variant"] == "full"
    assert default["disabled_filters"] == []


def test_drop_weekly_enters_against_observed_weekly_direction_without_faking_features(monkeypatch):
    fake_features(monkeypatch, down_indices=(1, 2, 3))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13), (15, 17, 14, 16)])
    full = engine.run_triple_screen(data, None)
    removed = engine.run_triple_screen(data, None, variant="drop_weekly")
    assert full["metrics"]["final_nav"] == 1
    assert removed["metrics"]["final_nav"] == pytest.approx(16/12)
    assert removed["metrics"]["final_exposure"] == 1
    assert removed["signals"][0]["weekly_up"] is False
    assert removed["signals"][0]["entry_ready"] is False
    assert removed["signals"][0]["entry_rule_passed"] is True
    assert removed["signals"][0]["disabled_filters"] == ["weekly_direction_and_exit"]
    assert (full["start"], full["end"], full["warmup"]) == (removed["start"], removed["end"], removed["warmup"])


def test_drop_weekly_also_removes_weekly_exit_but_keeps_initial_stop(monkeypatch):
    fake_features(monkeypatch, down_indices=(2, 3))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13),
                 (15, 17, 14, 16), (10, 11, 7, 9)])
    full = engine.run_triple_screen(data, None)
    removed = engine.run_triple_screen(data, None, variant="drop_weekly")
    assert full["fills"][-1]["reason"] == "weekly_filter_invalidated"
    assert full["metrics"]["final_nav"] == pytest.approx(15/12)
    assert removed["fills"][-1]["reason"] == "initial_stop"
    assert removed["fills"][-1]["adjusted_price"] == 8
    assert removed["metrics"]["final_nav"] == pytest.approx(8/12)


def test_drop_pullback_accepts_positive_force_and_retains_weekly_exit(monkeypatch):
    fake_features(monkeypatch, setup_indices=(), down_indices=(2,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13), (15, 17, 14, 16)])
    full = engine.run_triple_screen(data, None)
    removed = engine.run_triple_screen(data, None, variant="drop_pullback")
    assert full["metrics"]["final_nav"] == 1
    assert removed["signals"][0]["force_index_2"] == 1
    assert removed["signals"][0]["entry_ready"] is False
    assert removed["signals"][0]["entry_rule_passed"] is True
    assert removed["fills"][-1]["reason"] == "weekly_filter_invalidated"
    assert removed["metrics"]["final_nav"] == pytest.approx(15/12)


def test_drop_pullback_still_requires_weekly_direction_and_breakout(monkeypatch):
    fake_features(monkeypatch, setup_indices=(), down_indices=(1,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    assert engine.run_triple_screen(data, None, variant="drop_pullback")["fills"] == []
    fake_features(monkeypatch, setup_indices=())
    data.loc[2, ["open", "high", "low", "close"]] = [10, 12, 9, 11]
    result = engine.run_triple_screen(data, None, variant="drop_pullback")
    assert result["fills"] == []
    assert result["signals"][0]["status"] == "expired_no_breakout"


def test_drop_trigger_enters_next_open_without_high_break_and_preserves_costs(monkeypatch):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (10, 11, 9, 10.5)])
    full = engine.run_triple_screen(data, None)
    removed = engine.run_triple_screen(data, None, {"cost_bps": 100}, variant="drop_trigger")
    assert full["fills"] == []
    assert removed["metrics"]["final_nav"] == pytest.approx(.99 * 10.5/10)
    assert removed["fills"][0]["phase"] == "open"
    assert removed["fills"][0]["adjusted_price"] == 10
    assert removed["fills"][0]["reason"] == "next_open_without_breakout_filter"
    assert removed["signals"][0]["observed_breakout"] is False
    assert removed["signals"][0]["trigger_required"] is False


@pytest.mark.parametrize("opening", [7, 8])
def test_drop_trigger_rejects_nonpositive_open_to_stop_distance(monkeypatch, opening):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (opening, 14, 6, 13)])
    result = engine.run_triple_screen(data, None, variant="drop_trigger")
    assert result["fills"] == []
    assert result["signals"][0]["status"] == "rejected_nonpositive_stop_distance"
    assert result["metrics"]["final_nav"] == 1


def test_drop_trigger_stop_on_entry_day_is_locked_until_next_tradable_open(monkeypatch):
    fake_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (10, 11, 7, 10.5),
                 (10.5, 11, 10, 10.5), (12, 13, 11, 12.5)])
    data.loc[3, "volume"] = 0
    result = engine.run_triple_screen(data, None, variant="drop_trigger")
    assert result["equity_curve"][0]["pending_exit_reason"] == "entry_day_stop_t_plus_one"
    assert result["equity_curve"][1]["exposure"] == 1
    assert result["fills"][-1]["adjusted_price"] == 12
    assert result["metrics"]["final_nav"] == pytest.approx(12/10)


@pytest.mark.parametrize("variant", ["full", "drop_weekly", "drop_pullback", "drop_trigger"])
def test_all_variants_keep_readiness_and_previous_tradability(monkeypatch, variant):
    fake_features(monkeypatch, unready_indices=(0, 1, 2))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13)])
    with pytest.raises(ValueError, match="No evaluable"):
        engine.run_triple_screen(data, None, variant=variant)
    fake_features(monkeypatch)
    data.loc[1, "volume"] = 0
    result = engine.run_triple_screen(data, None, variant=variant)
    assert result["fills"] == []
    assert result["decisions"][0]["entry_rule_passed"] is False
