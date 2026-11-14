"""Priced contracts for a post-hoc, prior-close conditional weekly exit."""
from datetime import datetime, time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research_layer.triple_screen as engine


VARIANT = "weekly_exit_loss_only"


def bars(values):
    dates = pd.bdate_range("2025-01-06", periods=len(values))
    return pd.DataFrame([
        {"symbol": "000001.SZ", "datetime": day, "open": opening, "high": high,
         "low": low, "close": close, "volume": 1000., "freq": "daily", "adjust": "pre"}
        for day, (opening, high, low, close) in zip(dates, values)
    ])


def observed_features(monkeypatch, *, setup=(1,), down=(2, 3, 4)):
    def build(frame, calendar, settings):
        out = frame.copy()
        out["weekly_hist"] = [0. if i in down else 1. for i in range(len(out))]
        out["weekly_hist_previous"] = .5
        out["weekly_up"] = [i not in down for i in range(len(out))]
        out["indicators_ready"] = True
        out["entry_ready"] = [i in setup and i not in down for i in range(len(out))]
        out["force_index_2"] = [-1. if i in setup else 1. for i in range(len(out))]
        out["weekly_completed_count"] = 100
        out["session_close"] = [datetime.combine(day.date(), time(16 if settings.is_us else 15))
                                for day in out["datetime"]]
        # A repeated weekly timestamp ensures daily reconsideration, not new-week gating.
        out["weekly_available_at"] = out["session_close"].iloc[0]
        return out
    monkeypatch.setattr(engine, "build_triple_features", build)


def test_profit_is_retained_then_loss_latches_exit_on_following_open(monkeypatch):
    observed_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13),
                 (14, 16, 11, 11.5), (11, 12, 10, 11.2)])
    result = engine.run_triple_screen(data, None, {"cost_bps": 10}, variant=VARIANT)
    assert [fill["adjusted_price"] for fill in result["fills"]] == [12, 11]
    assert result["fills"][-1]["reason"] == "weekly_filter_invalidated"
    assert result["equity_curve"][1]["exposure"] == 1
    assert result["equity_curve"][1]["pending_exit_reason"] is None
    assert result["metrics"]["final_nav"] == pytest.approx(11 / 12 * .999 ** 2)
    assert result["decisions"][1]["weekly_available_at"] == result["decisions"][2]["weekly_available_at"]
    assert result["disabled_filters"] == ["weekly_exit_when_price_profitable"]
    assert any("post-hoc" in text for text in result["assumptions"])


@pytest.mark.parametrize("prior_close,opening,should_exit", [
    (11., 15., True), (12., 15., True), (12.001, 11., False), (13., 11., False),
])
def test_price_boundary_uses_prior_close_ignores_costs_and_current_gap(monkeypatch, prior_close, opening, should_exit):
    observed_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 9, prior_close),
                 (opening, opening + 1, opening - 1, opening)])
    result = engine.run_triple_screen(data, None, {"cost_bps": 10}, variant=VARIANT)
    assert len(result["fills"]) == (2 if should_exit else 1)
    assert result["metrics"]["final_exposure"] == (0 if should_exit else 1)
    assert result["metrics"]["final_nav"] == pytest.approx(opening / 12 * .999 ** (2 if should_exit else 1))
    if should_exit:
        assert result["fills"][-1]["adjusted_price"] == opening


def test_pending_weekly_exit_survives_suspension_and_price_rebound(monkeypatch):
    observed_features(monkeypatch, down=(2,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 9, 11),
                 (11, 15, 10, 14), (15, 16, 14, 15)])
    data.loc[3, "volume"] = 0
    result = engine.run_triple_screen(data, None, variant=VARIANT)
    assert result["fills"][1]["status"] == "blocked"
    assert result["equity_curve"][1]["pending_exit_reason"] == "weekly_filter_invalidated"
    assert result["fills"][-1]["reason"] == "weekly_filter_invalidated"
    assert result["fills"][-1]["adjusted_price"] == 15
    assert result["metrics"]["final_nav"] == pytest.approx(15 / 12)
    assert result["metrics"]["final_exposure"] == 0


def test_entry_day_stop_pending_survives_profit_and_suspension(monkeypatch):
    observed_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 7, 13),
                 (13, 15, 12, 14), (15, 16, 14, 15)])
    data.loc[3, "volume"] = 0
    result = engine.run_triple_screen(data, None, variant=VARIANT)
    assert result["equity_curve"][0]["pending_exit_reason"] == "entry_day_stop_t_plus_one"
    assert result["equity_curve"][1]["pending_exit_reason"] == "entry_day_stop_t_plus_one"
    assert result["fills"][-1]["reason"] == "entry_day_stop_t_plus_one"
    assert result["fills"][-1]["adjusted_price"] == 15


def test_pending_exit_does_not_reenter_on_an_otherwise_valid_candidate(monkeypatch):
    observed_features(monkeypatch, setup=(1, 2), down=())
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 7, 13),
                 (15, 16, 12, 15)])
    result = engine.run_triple_screen(data, None, variant=VARIANT)
    assert result["decisions"][-1]["entry_rule_passed"] is True
    assert [fill["adjusted_price"] for fill in result["fills"]] == [12, 15]
    assert result["fills"][-1]["reason"] == "entry_day_stop_t_plus_one"
    assert result["metrics"]["final_exposure"] == 0


def test_nontradeable_stop_pending_survives_profit_and_weekly_recovery(monkeypatch):
    observed_features(monkeypatch, down=(2,))
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 9, 13),
                 (13, 15, 7, 14), (15, 16, 14, 15)])
    data.loc[3, "volume"] = 0
    result = engine.run_triple_screen(data, None, variant=VARIANT)
    assert result["equity_curve"][1]["pending_exit_reason"] == "stop_on_non_tradeable_bar"
    assert result["fills"][-1]["reason"] == "stop_on_non_tradeable_bar"


def test_profitable_prior_close_does_not_disable_initial_stop(monkeypatch):
    observed_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 9, 13),
                 (13, 15, 7, 14)])
    result = engine.run_triple_screen(data, None, variant=VARIANT)
    assert result["fills"][-1]["reason"] == "initial_stop"
    assert result["fills"][-1]["adjusted_price"] == 8
    assert result["metrics"]["final_nav"] == pytest.approx(8 / 12)


@pytest.mark.parametrize("variant,entry,exit_price", [
    ("full", 12, 14), ("drop_weekly_entry", 12, 14), ("drop_pullback", 12, 14),
    ("drop_trigger", 11, 14), ("drop_weekly", 12, None), ("drop_weekly_exit", 12, None),
])
def test_existing_variant_priced_behaviour_is_unchanged(monkeypatch, variant, entry, exit_price):
    observed_features(monkeypatch)
    data = bars([(10, 11, 8, 10), (10, 12, 9, 10), (11, 14, 10, 13),
                 (14, 16, 11, 11.5), (11, 12, 10, 11.2)])
    result = engine.run_triple_screen(data, None, {"cost_bps": 10}, variant=variant)
    assert result["fills"][0]["adjusted_price"] == entry
    assert len(result["fills"]) == (2 if exit_price is not None else 1)
    price = exit_price if exit_price is not None else 11.2
    assert result["metrics"]["final_nav"] == pytest.approx(price / entry * .999 ** (2 if exit_price is not None else 1))


def real_feature_sample():
    dates = pd.bdate_range("2024-01-01", periods=120)
    index = np.arange(len(dates))
    close = 20 + index * .05 + np.sin(index / 3)
    opening = np.r_[close[0], close[:-1]]
    data = pd.DataFrame({"symbol": "000001.SZ", "datetime": dates, "open": opening,
                         "high": np.maximum(opening, close) + .1 + 2 * (index % 3 == 0),
                         "low": np.minimum(opening, close) - .3, "close": close,
                         "volume": 1000., "freq": "daily", "adjust": "pre"})
    calendar_dates = pd.date_range("2024-01-01", "2024-07-07")
    calendar = pd.DataFrame({"date": calendar_dates, "is_trading_day": calendar_dates.dayofweek < 5})
    settings = {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "warmup_weeks": 3}
    return data, calendar, settings


def test_real_features_future_perturbation_does_not_change_prior_decisions():
    data, calendar, settings = real_feature_sample()
    original = engine.run_triple_screen(data, calendar, settings, variant=VARIANT)
    changed = data.copy()
    changed.loc[90:, ["open", "high", "low", "close"]] *= 2
    perturbed = engine.run_triple_screen(changed, calendar, settings, variant=VARIANT)
    cutoff = data.loc[89, "datetime"].date().isoformat()
    assert len([r for r in original["fills"] if r["date"] <= cutoff]) >= 2
    for field in ("equity_curve", "fills", "decisions"):
        assert [r for r in original[field] if r["date"] <= cutoff] == [r for r in perturbed[field] if r["date"] <= cutoff]


def test_new_rule_propagates_through_local_study_runner_and_verified_report(tmp_path):
    from research_layer.triple_report import render_triple_report
    from research_layer.triple_study import run_triple_study

    data, calendar, settings = real_feature_sample()
    source, calendar_path = tmp_path / "bars.csv", tmp_path / "calendar.csv"
    data.to_csv(source, index=False)
    calendar.to_csv(calendar_path, index=False)
    config = {"source_type": "csv", "source_path": str(source), "calendar_path": str(calendar_path),
              "symbols": ["000001.SZ"], "adjust": "pre", "start": "2024-02-01", "end": "2024-06-14",
              "settings": settings, "cost_scenarios_bps": [10], "max_tasks": 2}
    cases = [{"id": "primary", "label": "Original", "variant": "full"},
             {"id": "conditional", "label": "Post-hoc conditional exit", "variant": VARIANT}]
    output = run_triple_study(config, cases, tmp_path / "research")
    assert output["status"] == "completed"
    assert output["summary"]["completed_tasks"] == 2
    record = json.loads((Path(output["path"]) / "case_results/conditional.json").read_text(encoding="utf-8"))
    assert record["variant"] == VARIANT
    child = Path(record["child_path"])
    artifact = json.loads((child / record["outcomes"][0]["artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["strategy"]["research_variant"] == VARIANT
    assert artifact["strategy"]["disabled_filters"] == ["weekly_exit_when_price_profitable"]
    rendered = render_triple_report(child).read_text(encoding="utf-8")
    assert "后验探索" in rendered
    assert "周转弱且前收盘不高于入场价才退出" in rendered
