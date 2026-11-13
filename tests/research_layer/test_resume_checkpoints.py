from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_layer.resume import resume_experiment
from research_layer.runner import run_experiment


def interrupted_run(tmp_path, mode="batch", event="trial_completed", count=1):
    config = {"mode": mode, "source_type": "demo", "strategy": "sma_trend",
              "parameter_grid": {"fast_window": [3, 5], "slow_window": [15, 20]}}
    state = {"n": 0, "stop": False}
    def progress(update):
        if update.get("event") == event:
            state["n"] += 1
            state["stop"] = state["n"] >= count
    with pytest.raises(InterruptedError):
        run_experiment(config, tmp_path, progress_callback=progress, cancel_check=lambda: state["stop"])
    return next(tmp_path.iterdir())


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def test_resume_skips_completed_candidates_and_preserves_original(tmp_path, monkeypatch):
    source = interrupted_run(tmp_path / "original", count=2)
    before = snapshot(source)
    import research_layer.batch as batch
    original = batch.run_backtest
    calls = []
    def measured(*args, **kwargs):
        calls.append(args[1] if len(args) > 1 else kwargs.get("strategy"))
        return original(*args, **kwargs)
    monkeypatch.setattr(batch, "run_backtest", measured)
    result = resume_experiment(source, tmp_path / "resumed")
    assert result["resume"]["reused_trials"] == 2
    assert result["resume"]["computed_trials"] == 2
    assert calls.count("sma_trend") == 2
    assert result["result"]["batch"]["candidate_count"] == 4
    assert snapshot(source) == before
    for trial in result["result"]["batch"]["trials"]:
        assert (Path(result["path"]) / trial["artifact_path"] / "result.json").is_file()


def test_resume_reuses_completed_fold_including_its_validation(tmp_path, monkeypatch):
    source = interrupted_run(tmp_path / "original", mode="walk_forward", event="fold_completed", count=1)
    import research_layer.walk_forward as wf
    original = wf.run_comparison
    calls = []
    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(wf, "run_comparison", measured)
    result = resume_experiment(source, tmp_path / "resumed")
    assert result["resume"]["reused_folds"] == 1
    assert len(calls) == 1
    assert result["result"]["walk_forward"]["summary"]["completed_folds"] == 2
    assert result["result"]["walk_forward"]["holdout"]["status"] == "reserved"


def test_modified_checkpoint_rejected_without_changing_source(tmp_path):
    source = interrupted_run(tmp_path / "original")
    report = next(source.glob("trials/*/result.json"))
    report.write_bytes(report.read_bytes() + b" ")
    before = snapshot(source)
    with pytest.raises(ValueError, match="checksum"):
        resume_experiment(source, tmp_path / "resumed")
    assert snapshot(source) == before


def test_complete_run_returns_verified_original_without_new_execution(tmp_path, monkeypatch):
    result = run_experiment({"source_type": "demo", "strategy": "cash"}, tmp_path / "original")
    import research_layer.runner as runner
    monkeypatch.setattr(runner, "run_experiment", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rerun")))
    recovered = resume_experiment(result["path"], tmp_path / "resumed")
    assert recovered["reused_run"] is True
    assert recovered["path"] == result["path"]
    assert not (tmp_path / "resumed").exists()


def test_legacy_unsigned_trials_are_recomputed_not_silently_reused(tmp_path):
    source = interrupted_run(tmp_path / "original")
    (source / "artifacts.json").unlink()
    result = resume_experiment(source, tmp_path / "resumed")
    assert result["resume"]["legacy_cache_recomputed"] is True
    assert result["resume"]["reused_trials"] == 0
    assert result["resume"]["computed_trials"] == 4


def test_source_version_change_rejects_resume(tmp_path, monkeypatch):
    source = interrupted_run(tmp_path / "original")
    import research_layer.runner as runner
    monkeypatch.setattr(runner, "_implementation_hash", lambda: "different-source")
    with pytest.raises(ValueError, match="code changed"):
        resume_experiment(source, tmp_path / "resumed")
