from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_layer.batch import expand_parameter_grid, preflight_experiment
from research_layer.runner import run_experiment


def config(**updates):
    return {"mode": "batch", "source_type": "demo", "strategy": "sma_trend",
            "parameter_grid": {"fast_window": [2, 8], "slow_window": [5, 10]}, **updates}


def test_grid_is_bounded_deterministic_and_deduplicated():
    first = expand_parameter_grid({"b": [2, 2, 3], "a": [1, 2]}, {"c": 7})
    second = expand_parameter_grid({"a": [2, 1], "b": [3, 2]}, {"c": 7})
    assert first == second
    assert len(first) == 4
    assert all(item["parameters"]["c"] == 7 for item in first)
    with pytest.raises(ValueError, match="more than"):
        expand_parameter_grid({"a": list(range(100000))}, max_candidates=3)
    with pytest.raises(ValueError, match="nonempty"):
        expand_parameter_grid({"a": []})


def test_budget_is_checked_before_backtesting():
    assert preflight_experiment(config())["estimated_backtests"] == 6
    with pytest.raises(ValueError, match="budget"):
        preflight_experiment(config(max_backtests=5))
    with pytest.raises(ValueError, match="integer"):
        preflight_experiment(config(max_candidates=True))
    with pytest.raises(ValueError, match="bar evaluations"):
        preflight_experiment(config(max_bar_evaluations=100), 100)


def test_walk_forward_budget_uses_actual_shared_plan():
    settings = {"train_bars": 40, "validation_bars": 15, "gap_bars": 1,
                "holdout_bars": 15, "evaluate_holdout": True}
    preflight = preflight_experiment(config(mode="walk_forward", walk_forward=settings), 150)
    folds = len(preflight["walk_forward_plan"]["folds"])
    assert preflight["estimated_backtests"] == (folds + 1) * (4 + 3)
    with pytest.raises(ValueError, match="budget"):
        preflight_experiment(config(mode="walk_forward", walk_forward=settings, max_backtests=4), 150)


@pytest.mark.parametrize("settings", [
    {"unknown": 1}, {"train_bars": 1}, {"validation_bars": 2.5},
    {"gap_bars": -1}, {"holdout_bars": True}, {"expanding": "false"},
    {"evaluate_holdout": 1}, {"selection_metric": "future_profit"}, [], "wrong",
])
def test_preview_rejects_bad_walk_forward_settings_before_loading_data(settings):
    with pytest.raises((ValueError, TypeError)):
        preflight_experiment(config(mode="walk_forward", walk_forward=settings))


def test_valid_preview_does_not_invent_data_dependent_budget():
    preview = preflight_experiment(config(mode="walk_forward", walk_forward={"train_bars": 20}))
    assert preview["estimated_backtests"] is None
    assert preview["estimated_bar_evaluations"] is None
    assert preview["walk_forward_plan"] is None


def test_failed_candidates_are_retained_with_successful_reports(tmp_path):
    output = run_experiment(config(), tmp_path)
    result = output["result"]
    batch = result["batch"]
    assert result["mode"] == "batch"
    assert batch["completed_count"] == 3
    assert batch["failed_count"] == 1
    assert batch["ranking_scope"] == "in_sample"
    assert len(batch["trials"]) == 4
    assert all("result" not in trial and "artifact_path" in trial for trial in batch["trials"])
    failed = [trial for trial in batch["trials"] if trial["status"] == "failed"]
    assert "fast_window" in failed[0]["error"]
    assert len(batch["ranking"]) == 3
    root = Path(output["path"])
    journal = [json.loads(line) for line in (root / "trials.jsonl").read_text().splitlines()]
    assert len(journal) == 4
    for record in journal:
        folder = root / record["artifact_path"]
        assert (folder / "config.json").is_file()
        assert (folder / "status.json").is_file()
        if record["status"] == "completed":
            assert (folder / "result.json").is_file()
            assert (folder / "fifo_report.json").is_file()
            assert not (folder / "baselines").exists()
    assert (root / "input.csv").is_file()
    assert (root / "trials.csv").is_file()
    assert (root / "fifo_closed.csv").is_file()
    resolved = json.loads((root / "config.json").read_text())
    assert resolved["max_bar_evaluations"] == 2_000_000


def test_data_is_loaded_once_and_original_frame_not_modified(tmp_path, monkeypatch):
    from backtest_layer.data import demo_bars
    import research_layer.runner as runner
    bars = demo_bars()
    original = bars.copy(deep=True)
    calls = []
    def load(_):
        calls.append(1)
        from backtest_layer.data import normalize_daily_bars
        return normalize_daily_bars(bars)
    monkeypatch.setattr(runner, "_load_input", load)
    run_experiment(config(), tmp_path)
    assert calls == [1]
    assert bars.equals(original)


def test_cancellation_preserves_completed_trials_and_initial_progress(tmp_path):
    flag = False
    updates = []
    def progress(update):
        nonlocal flag
        updates.append(update)
        if update.get("event") == "trial_completed":
            flag = True
    with pytest.raises(InterruptedError):
        run_experiment(config(), tmp_path, progress_callback=progress, cancel_check=lambda: flag)
    assert updates[0]["path"] and updates[0]["run_id"]
    root = Path(updates[0]["path"])
    partial = json.loads((root / "partial_result.json").read_text())
    assert partial["status"] == "cancelled"
    assert len(partial["trials"]) == 1
    assert (root / partial["trials"][0]["artifact_path"] / "status.json").is_file()
    assert json.loads((root / "status.json").read_text())["status"] == "cancelled"


def test_input_failure_is_durable(tmp_path):
    updates = []
    with pytest.raises(FileNotFoundError):
        run_experiment(config(source_type="csv", source_path=str(tmp_path / "missing.csv")),
                       tmp_path / "runs", progress_callback=updates.append)
    root = Path(updates[0]["path"])
    assert json.loads((root / "status.json").read_text())["status"] == "failed"
    assert (root / "config.json").is_file()
    assert (root / "partial_result.json").is_file()


def test_all_failed_candidates_do_not_fabricate_a_winner(tmp_path):
    output = run_experiment(config(parameter_grid={"fast_window": [20], "slow_window": [5]}), tmp_path)
    assert output["result"]["batch"]["best_candidate_id"] is None
    assert output["result"]["batch"]["ranking"] == []
    assert output["result"]["strategy"] is None


def test_batch_identity_and_ranking_repeat(tmp_path):
    first = run_experiment(config(), tmp_path)["result"]
    second = run_experiment(config(), tmp_path)["result"]
    assert first["config_hash"] == second["config_hash"]
    assert first["batch"]["ranking"] == second["batch"]["ranking"]


def test_walk_forward_persistence_reserves_holdout_and_keeps_training_attempts(tmp_path):
    output = run_experiment(config(mode="walk_forward"), tmp_path)
    result = output["result"]
    workflow = result["walk_forward"]
    assert workflow["holdout"]["status"] == "reserved"
    assert result["selection_scope"] == "last_validation_fold"
    assert len(workflow["folds"]) >= 2
    root = Path(output["path"])
    trials = [json.loads(line) for line in (root / "trials.jsonl").read_text().splitlines()]
    assert len(trials) == len(workflow["folds"]) * 4
    assert len(list((root / "folds").glob("*.json"))) == len(workflow["folds"])
    assert all(trial["phase"] == "training" for trial in trials)
    assert all("result" not in trial for fold in workflow["folds"] for trial in fold["train_trials"])
    assert all("artifact_path" in trial for fold in workflow["folds"] for trial in fold["train_trials"])
    assert (root / "folds" / workflow["folds"][0]["fold_id"] / "validation" / "fifo_report.json").is_file()


def test_opt_in_holdout_and_cancelled_walk_forward_are_preserved(tmp_path):
    output = run_experiment(config(mode="walk_forward", walk_forward={"evaluate_holdout": True}), tmp_path / "complete")
    assert output["result"]["walk_forward"]["holdout"]["status"] == "completed"
    root = Path(output["path"])
    assert (root / "folds" / "holdout" / "validation" / "fifo_report.json").is_file()
    updates = []
    flag = False
    def on_progress(update):
        nonlocal flag
        updates.append(update)
        if update.get("event") == "fold_completed":
            flag = True
    with pytest.raises(InterruptedError):
        run_experiment(config(mode="walk_forward"), tmp_path / "cancelled", progress_callback=on_progress, cancel_check=lambda: flag)
    partial = json.loads((Path(updates[0]["path"]) / "partial_result.json").read_text())
    assert len(partial["folds"]) == 1
    assert len(partial["trials"]) == 4
    assert partial["status"] == "cancelled"
