from __future__ import annotations

import copy
import pytest

from backtest_layer.data import demo_bars
from research_layer.walk_forward import build_walk_forward_plan, run_walk_forward


def candidates():
    return [{"candidate_id": "fast", "parameters": {"fast_window": 2, "slow_window": 5}},
            {"candidate_id": "slow", "parameters": {"fast_window": 5, "slow_window": 15}}]


def test_plan_has_chronological_disjoint_validation_and_reserved_holdout():
    plan = build_walk_forward_plan(100)
    assert plan["folds"] == [
        {"fold_id": "fold_001", "train_start": 0, "train_end": 40, "gap_start": 40, "gap_end": 41, "validation_start": 41, "validation_end": 56},
        {"fold_id": "fold_002", "train_start": 15, "train_end": 55, "gap_start": 55, "gap_end": 56, "validation_start": 56, "validation_end": 71},
    ]
    assert plan["holdout"] == {"start": 85, "end": 100, "evaluate": False}
    assert plan["calibration"] is None
    assert plan["unused_development_bars"] == 14
    expanding = build_walk_forward_plan(100, {"expanding": True, "evaluate_holdout": True})
    assert [fold["train_start"] for fold in expanding["folds"]] == [0, 0]
    assert expanding["calibration"] == {"train_start": 0, "train_end": 84, "gap_start": 84, "gap_end": 85}


@pytest.mark.parametrize("settings", [
    {"train_bars": True}, {"validation_bars": 1}, {"gap_bars": -1}, {"holdout_bars": 0},
    {"expanding": "false"}, {"evaluate_holdout": 1}, {"selection_metric": "future_return"}, {"typo": 5},
])
def test_invalid_window_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        build_walk_forward_plan(100, settings)


def test_short_data_and_unbounded_fold_count_are_rejected():
    with pytest.raises(ValueError, match="at least"):
        build_walk_forward_plan(50)
    with pytest.raises(ValueError, match="At most"):
        build_walk_forward_plan(1000, {"train_bars": 2, "validation_bars": 2, "holdout_bars": 2})


def test_future_changes_do_not_change_prior_training_selection():
    bars = demo_bars(days=100)
    events = []
    original = run_walk_forward(bars, "sma_trend", candidates(), progress_callback=events.append)
    changed = bars.copy()
    changed.loc[41:, ["open", "high", "low", "close"]] *= 2
    later = run_walk_forward(changed, "sma_trend", candidates())
    assert original["folds"][0]["selected_candidate_id"] == later["folds"][0]["selected_candidate_id"]
    assert original["folds"][0]["train_trials"] == later["folds"][0]["train_trials"]
    assert original["holdout"]["status"] == "reserved"
    assert original["holdout"]["result"] is None
    assert original["summary"]["completed_folds"] == 2
    assert "total_return" not in original["summary"]
    assert len([event for event in events if event["event"] == "trial_completed"]) == 4
    assert len([event for event in events if event["event"] == "fold_completed"]) == 2
    assert "result" in events[0]["trial"]
    assert "result" not in original["folds"][0]["train_trials"][0]
    for fold in original["folds"]:
        validation = fold["validation"]["strategy"]
        assert validation["start"] == fold["periods"]["validation"]["start"]
        assert len(validation["equity_curve"]) == 15
        assert validation["warmup"]["end"] < validation["start"]
        assert validation["metrics"]["initial_nav"] == 1_000_000


def test_reserved_holdout_prices_never_affect_results_and_explicit_test_is_separate():
    bars = demo_bars(days=100)
    original = run_walk_forward(bars, "sma_trend", candidates())
    changed = bars.copy()
    changed.loc[85:, ["open", "high", "low", "close"]] *= 3
    reserved = run_walk_forward(changed, "sma_trend", candidates())
    assert original == reserved
    evaluated = run_walk_forward(bars, "sma_trend", candidates(), settings={"evaluate_holdout": True})
    changed_evaluated = run_walk_forward(changed, "sma_trend", candidates(), settings={"evaluate_holdout": True})
    assert evaluated["holdout"]["status"] == "completed"
    assert evaluated["holdout"]["selected_candidate_id"] == changed_evaluated["holdout"]["selected_candidate_id"]
    assert evaluated["holdout"]["train_trials"] == changed_evaluated["holdout"]["train_trials"]
    assert evaluated["holdout"]["calibration_period"]["end"] < evaluated["holdout"]["period"]["start"]
    assert evaluated["folds"] == original["folds"]


def test_deterministic_ties_and_failed_trials_are_recorded():
    tied = run_walk_forward(demo_bars(), "cash", [
        {"candidate_id": "z", "parameters": {}}, {"candidate_id": "a", "parameters": {}},
    ])
    assert all(fold["selected_candidate_id"] == "a" for fold in tied["folds"])
    mixed = run_walk_forward(demo_bars(), "sma_trend", [
        {"candidate_id": "bad", "parameters": {"fast_window": 20, "slow_window": 5}}, candidates()[0],
    ])
    assert mixed["status"] == "completed"
    assert all(fold["train_trials"][0]["status"] == "failed" for fold in mixed["folds"])
    assert all(fold["selected_candidate_id"] == "fast" for fold in mixed["folds"])
    failed = run_walk_forward(demo_bars(), "sma_trend", [
        {"candidate_id": "bad", "parameters": {"fast_window": 20, "slow_window": 5}},
    ])
    assert failed["status"] == "failed"
    assert failed["summary"]["mean_validation_return"] is None
    assert all(fold["validation"] is None for fold in failed["folds"])


def test_cancellation_retains_completed_trial_callback():
    events = []
    def cancel():
        return bool(events)
    with pytest.raises(InterruptedError):
        run_walk_forward(demo_bars(), "sma_trend", candidates(), progress_callback=events.append, cancel_check=cancel)
    assert len(events) == 1
    assert events[0]["trial"]["status"] == "completed"
    assert events[0]["trial"]["result"]["fifo_report"]


def test_candidate_identity_is_unique_and_parameters_remain_unmodified():
    rows = candidates()
    before = copy.deepcopy(rows)
    run_walk_forward(demo_bars(), "sma_trend", rows)
    assert rows == before
    with pytest.raises(ValueError, match="unique"):
        run_walk_forward(demo_bars(), "sma_trend", [rows[0], rows[0]])
