"""Verified task-boundary recovery into a new run, preserving the source run."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Callable
from uuid import uuid4

from backtest_layer.execution import ExecutionConfig
from research_layer.batch import preflight_experiment, run_parameter_batch
from research_layer.offline import _digest_file, _json, _read_json, _safe_relative, _sha, _write_json


class VerifiedRun:
    def __init__(self, directory: str | Path):
        self.path = Path(directory).resolve(strict=True)
        self.config = _read_json(self.path / "config.json")
        index_path = self.path / "artifacts.json"
        index = _read_json(index_path) if index_path.exists() else {"files": {}}
        self.files = index.get("files", {})
        if not isinstance(self.files, dict):
            raise ValueError("Invalid artifact integrity index")
        self.verified_cache = bool(self.files)
        for relative, digest in self.files.items():
            _safe_relative(relative)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Invalid artifact checksum")

    def checked_path(self, relative: str) -> Path | None:
        _safe_relative(relative)
        if relative not in self.files:
            return None
        path = (self.path / relative).resolve()
        if not path.is_relative_to(self.path) or not path.is_file():
            raise ValueError(f"Indexed artifact missing or linked outside run: {relative}")
        if _digest_file(path) != self.files[relative]:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
        return path

    def checked_json(self, relative: str) -> dict | None:
        path = self.checked_path(relative)
        return _read_json(path) if path is not None else None


def resume_experiment(
    source_run_dir: str | Path, output_root: str | Path | None = None, *, retry_failed: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Reuse verified completed candidates/folds; unfinished accounts restart.

    No source artifact is overwritten. A missing integrity index means legacy
    checkpoints cannot be trusted for reuse and are explicitly recomputed.
    A checksum mismatch is rejected rather than hidden by recomputation.
    """
    from research_layer.runner import (_implementation_hash, _load_input, _save_backtest,
                                       _save_comparison, _write_csv, run_experiment)
    source = VerifiedRun(source_run_dir)
    config = source.config
    if config.get("implementation_sha256") != _implementation_hash():
        raise ValueError("Calculation code changed since this run; create a new experiment instead of reusing checkpoints")
    if source.verified_cache:
        if source.checked_path("config.json") is None or source.checked_path("input.csv") is None:
            raise ValueError("Recovery requires checksummed configuration and input")
    elif not (source.path / "input.csv").is_file():
        raise ValueError("Run has no saved input snapshot to restart from")
    completed = source.checked_json("result.json")
    status = _read_json(source.path / "status.json") if (source.path / "status.json").exists() else {}
    if completed is not None and status.get("status") == "completed" and not retry_failed:
        return {"run_id": completed.get("run_id", source.path.name), "status": "completed", "path": str(source.path),
                "result": completed, "reused_run": True}
    if status.get("status") == "running":
        raise ValueError("Run is still marked running; stop/reconcile its owner before recovery")
    snapshot = {**config, "source_type": "csv", "source_path": str(source.path / "input.csv"),
                "snapshot_ready": True, "context": None}
    for key in ("context_sources", "start", "end"):
        snapshot.pop(key, None)
    destination = Path(output_root or source.path.parent).resolve()
    if destination == source.path or destination.is_relative_to(source.path):
        raise ValueError("Recovery output must be outside the original run directory")
    mode = config.get("mode", "single")
    if mode not in {"batch", "walk_forward"}:
        # One account simulation is the atomic recovery unit; there is no
        # fabricated mid-account state reconstruction.
        result = run_experiment(snapshot, destination, progress_callback=progress_callback, cancel_check=cancel_check)
        result["resume"] = {"source_run": str(source.path), "reused_trials": 0, "reused_folds": 0,
                            "boundary": "whole_account_task", "legacy_cache_recomputed": not source.verified_cache}
        _write_json(Path(result["path"]) / "resume.json", result["resume"])
        return result

    bars = _load_input(snapshot)
    plan = preflight_experiment(config, len(bars))
    execution = ExecutionConfig(**config.get("execution", {}))
    strategy = config["strategy"]
    run_id = f"research_resume_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run = destination / run_id
    run.mkdir(parents=True, exist_ok=False)
    counters = {"reused_trials": 0, "reused_folds": 0, "reused_holdouts": 0, "computed_trials": 0}
    resolved = {**config, "resumed_from": str(source.path), "original_config_hash": config.get("config_hash"),
                "retry_failed": retry_failed, "cache_verification": "sha256" if source.verified_cache else "none_recompute"}
    resolved.pop("config_hash", None)
    resolved["config_hash"] = _sha(_json(resolved))
    _write_json(run / "config.json", resolved)
    shutil.copyfile(source.path / "input.csv", run / "input.csv")
    _write_json(run / "plan.json", plan)
    trials = []
    folds = []

    def update_index() -> None:
        index = {p.relative_to(run).as_posix(): _digest_file(p) for p in run.rglob("*")
                 if p.is_file() and p not in {run / "artifacts.json", run / "status.json"} and not p.name.endswith(".tmp")}
        _write_json(run / "artifacts.json", {"schema_version": 1, "files": index})

    def notify(message: str) -> None:
        update = {"run_id": run_id, "path": str(run), "message": message, **counters}
        _write_json(run / "status.json", {**update, "status": "running"})
        if progress_callback:
            progress_callback(update)

    def check_cancelled() -> None:
        if cancel_check and cancel_check():
            raise InterruptedError("Recovery cancelled; completed task checkpoints retained in the new run")

    def record(event: dict) -> None:
        check_cancelled()
        if event.get("event") == "trial_completed":
            trial = event["trial"]
            phase, fold_id = event.get("phase", "batch"), event.get("fold_id", "batch")
            key = f"{phase}_{fold_id}_{trial['candidate_id']}"
            relative = f"trials/{key}"
            _safe_relative(relative)
            path = run / relative
            _write_json(path / "config.json", {"strategy": strategy, "parameters": trial["parameters"], "execution": config.get("execution", {}), "phase": phase, "fold_id": fold_id})
            _write_json(path / "status.json", {k: v for k, v in trial.items() if k != "result"})
            if trial.get("result"):
                result = trial["result"]
                _save_backtest(path, result["strategy"] if isinstance(result.get("strategy"), dict) else result)
            summary = {k: v for k, v in trial.items() if k != "result"}
            summary.update(artifact_path=relative, phase=phase, fold_id=fold_id)
            trial["artifact_path"] = relative
            trials.append(summary)
            with (run / "trials.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(_json(summary).decode())
            counters["reused_trials" if trial.get("reused") else "computed_trials"] += 1
        elif event.get("event") in {"fold_completed", "holdout_completed"}:
            fold_id = event.get("fold_id", "holdout")
            fold = event.get("fold", event.get("holdout"))
            folds.append(fold)
            _write_json(run / "folds" / f"{fold_id}.json", fold)
            comparison = fold.get("validation") or fold.get("result")
            if comparison:
                _save_comparison(run / "folds" / fold_id / "validation", comparison)
            if fold.get("reused"):
                counters["reused_holdouts" if fold_id == "holdout" else "reused_folds"] += 1
        update_index()
        notify("Recovery reused verified checkpoints and saved the completed task")

    def cached_trial(phase: str, fold_id: str, candidate: dict) -> dict | None:
        key = f"trials/{phase}_{fold_id}_{candidate['candidate_id']}"
        saved = source.checked_json(key + "/status.json")
        if saved is None or saved.get("parameters") != candidate["parameters"]:
            return None
        if saved.get("status") == "failed":
            return None if retry_failed else saved
        if saved.get("status") != "completed":
            return None
        report = source.checked_json(key + "/result.json")
        if report is None:
            return None
        if report.get("parameters") != candidate["parameters"] or report.get("strategy") != strategy:
            raise ValueError("Cached report parameters/strategy do not match the experiment")
        return {**saved, "candidate_id": candidate["candidate_id"], "parameters": candidate["parameters"], "result": report}

    def cached_fold(fold_id: str) -> dict | None:
        saved = source.checked_json(f"folds/{fold_id}.json")
        if saved is None or saved.get("status") not in {"completed", "failed"}:
            return None
        if saved["status"] == "failed" and retry_failed:
            return None
        phase = "holdout_calibration" if fold_id == "holdout" else "training"
        copied = []
        for summary in saved.get("train_trials", []):
            trial = cached_trial(phase, fold_id, summary)
            if trial is None:
                return None
            copied.append(trial)
        for trial in copied:
            trial["reused"] = True
            record({"event": "trial_completed", "phase": phase, "fold_id": fold_id, "trial": trial})
        return {**saved, "train_trials": [{k: v for k, v in trial.items() if k != "result"} for trial in copied]}

    update_index()
    try:
        notify("Verifying saved input and completed trial/fold checkpoints")
        check_cancelled()
        if mode == "batch":
            batch = run_parameter_batch(bars, strategy, plan["candidates"], execution,
                selection_metric=plan["selection_metric"], include_baselines=bool(config.get("include_baselines", True)),
                progress_callback=record, cancel_check=cancel_check, retain_results=False,
                cache_provider=lambda candidate: cached_trial("batch", "batch", candidate))
            selected = batch.pop("best_result") or {"strategy": None, "baselines": batch["baselines"]}
            batch.pop("baselines", None)
            result = {**selected, "mode": mode, "batch": batch, "selection_scope": "in_sample"}
        else:
            from research_layer.walk_forward import run_walk_forward
            workflow = run_walk_forward(bars, strategy, plan["candidates"], execution, settings=config.get("walk_forward"),
                progress_callback=record, cancel_check=cancel_check, trial_cache_provider=cached_trial,
                fold_cache_provider=cached_fold, holdout_cache_provider=lambda: cached_fold("holdout"))
            selected = [fold["validation"] for fold in workflow["folds"] if fold.get("validation")]
            result = {**(selected[-1] if selected else {"strategy": None, "baselines": {}}),
                      "mode": mode, "walk_forward": workflow, "selection_scope": "last_validation_fold"}
        result.update(run_id=run_id, config_hash=resolved["config_hash"], data_sha256=config.get("data_sha256"),
                      implementation_sha256=config["implementation_sha256"], synthetic=bool(config.get("synthetic")),
                      resume={"source_run": str(source.path), **counters, "legacy_cache_recomputed": not source.verified_cache})
        _save_comparison(run / "selected_report", result)
        if result.get("strategy"):
            _write_csv(run / "equity.csv", result["strategy"]["equity_curve"])
            _write_csv(run / "fills.csv", result["strategy"]["fills"])
        _write_json(run / "result.json", result)
        _write_json(run / "resume.json", result["resume"])
        _write_csv(run / "trials.csv", [{"candidate_id": t["candidate_id"], "status": t["status"], "reused": t.get("reused", False), "parameters": json.dumps(t["parameters"], sort_keys=True)} for t in trials])
        update_index()
        _write_json(run / "status.json", {"run_id": run_id, "status": "completed", **counters})
        return {"run_id": run_id, "status": "completed", "path": str(run), "result": result, "resume": result["resume"]}
    except BaseException as exc:
        state = "cancelled" if isinstance(exc, InterruptedError) else "failed"
        _write_json(run / "partial_result.json", {"run_id": run_id, "mode": mode, "status": state, "trials": trials, "folds": folds, "error": str(exc), "resume": counters})
        update_index()
        _write_json(run / "status.json", {"run_id": run_id, "status": state, "error": str(exc), **counters})
        raise
