"""Bounded, chronological parameter selection and out-of-sample validation.

Every validation fold is a fresh independent account. Fold returns are
summarized as a distribution, never concatenated into a fictitious portfolio.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import math
import re
from numbers import Integral
from statistics import mean, median
from typing import Any

from backtest_layer.data import normalize_daily_bars
from backtest_layer.engine import run_backtest, run_comparison
from backtest_layer.execution import ExecutionConfig


DEFAULT_SETTINGS = {
    "train_bars": 40, "validation_bars": 15, "gap_bars": 1,
    "holdout_bars": 15, "expanding": False, "evaluate_holdout": False,
    "selection_metric": "total_return",
}
MAX_FOLDS = 64
MAX_CANDIDATES = 128
_METRICS = {"total_return", "sharpe", "maximum_drawdown"}


def _integer(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}.")
    return int(value)


def resolve_walk_forward_settings(settings: dict | None = None) -> dict:
    """Validate the same settings before and after loading a dataset."""
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("Walk-forward settings must be a JSON object.")
    raw = dict(settings) if settings is not None else {}
    unknown = set(raw) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown walk-forward settings: {sorted(unknown)}")
    resolved = {**DEFAULT_SETTINGS, **raw}
    for field, minimum in [("train_bars", 2), ("validation_bars", 2), ("gap_bars", 0), ("holdout_bars", 2)]:
        resolved[field] = _integer(resolved[field], field, minimum)
    for field in ("expanding", "evaluate_holdout"):
        if not isinstance(resolved[field], bool):
            raise ValueError(f"{field} must be a boolean.")
    if not isinstance(resolved["selection_metric"], str) or resolved["selection_metric"] not in _METRICS:
        raise ValueError(f"selection_metric must be one of {sorted(_METRICS)}")
    return resolved


def build_walk_forward_plan(bar_count: int, settings: dict | None = None) -> dict:
    count = _integer(bar_count, "bar_count", 2)
    resolved = resolve_walk_forward_settings(settings)
    train, validation, gap, holdout = (resolved[key] for key in ("train_bars", "validation_bars", "gap_bars", "holdout_bars"))
    required = train + gap + validation + holdout
    if count < required:
        raise ValueError(f"Walk-forward needs at least {required} bars; received {count}. Reduce windows or select more data.")
    development_end = count - holdout
    folds = []
    for start in range(train + gap, development_end - validation + 1, validation):
        train_end = start - gap
        folds.append({"fold_id": f"fold_{len(folds) + 1:03d}",
                      "train_start": 0 if resolved["expanding"] else train_end - train,
                      "train_end": train_end, "gap_start": train_end, "gap_end": start,
                      "validation_start": start, "validation_end": start + validation})
        if len(folds) > MAX_FOLDS:
            raise ValueError(f"At most {MAX_FOLDS} validation folds are allowed; increase validation_bars or narrow the date range.")
    calibration_end = development_end - gap
    return {"settings": resolved, "bar_count": count, "folds": folds,
            "holdout": {"start": development_end, "end": count, "evaluate": resolved["evaluate_holdout"]},
            "calibration": {"train_start": 0 if resolved["expanding"] else calibration_end - train,
                            "train_end": calibration_end, "gap_start": calibration_end, "gap_end": development_end}
                           if resolved["evaluate_holdout"] else None,
            "unused_development_bars": development_end - folds[-1]["validation_end"],
            "index_convention": "zero-based half-open [start,end)"}


def run_walk_forward(
    bars: Any, strategy: str, candidates: list[dict],
    execution_config: ExecutionConfig | dict | None = None,
    settings: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    *, retain_trial_results: bool = False,
    trial_cache_provider: Callable[[str, str, dict], dict | None] | None = None,
    fold_cache_provider: Callable[[str], dict | None] | None = None,
    holdout_cache_provider: Callable[[], dict | None] | None = None,
) -> dict:
    frame = normalize_daily_bars(bars)
    plan = build_walk_forward_plan(len(frame), settings)
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"Provide 1 to {MAX_CANDIDATES} candidates.")
    candidate_rows = []
    identifiers = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("parameters"), dict):
            raise ValueError("Each candidate must have candidate_id and parameters.")
        identifier = candidate.get("candidate_id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", identifier) or identifier in identifiers:
            raise ValueError("Candidate IDs must be unique safe alphanumeric identifiers.")
        identifiers.add(identifier)
        # Reject unserializable/nonfinite parameter values; semantic strategy
        # errors remain individual failed trials below.
        json.dumps(candidate["parameters"], allow_nan=False)
        candidate_rows.append({"candidate_id": identifier, "parameters": dict(candidate["parameters"])})
    candidate_rows.sort(key=lambda item: item["candidate_id"])
    execution = execution_config if isinstance(execution_config, ExecutionConfig) else ExecutionConfig(**dict(execution_config or {}))
    metric = plan["settings"]["selection_metric"]
    maximize = metric != "maximum_drawdown"

    def check_cancelled() -> None:
        if cancel_check and cancel_check():
            raise InterruptedError("Walk-forward cancelled")

    def emit(event: dict) -> None:
        if progress_callback:
            progress_callback(event)

    def period(start: int, end: int) -> dict:
        return {"start_index": start, "end_index_exclusive": end, "bars": end - start,
                "start": frame.iloc[start]["datetime"].date().isoformat() if start < end else None,
                "end": frame.iloc[end - 1]["datetime"].date().isoformat() if start < end else None}

    def fit(scope: str, start: int, end: int, *, phase: str) -> tuple[list[dict], dict | None]:
        trials = []
        for candidate in candidate_rows:
            check_cancelled()
            trial = {**candidate, "fold_id": scope, "phase": phase, "status": "running"}
            cached = trial_cache_provider(phase, scope, candidate) if trial_cache_provider else None
            if cached is not None:
                if cached.get("candidate_id") != candidate["candidate_id"] or cached.get("parameters") != candidate["parameters"] or cached.get("status") not in {"completed", "failed"}:
                    raise ValueError("Cached training trial identity/status mismatch")
                trial.update(cached, reused=True)
            else:
                try:
                    result = run_backtest(frame.iloc[start:end], strategy, candidate["parameters"], execution,
                                          warmup_bars=frame.iloc[:start], cancel_check=cancel_check)
                    score = float(result["metrics"][metric])
                    if not math.isfinite(score):
                        raise ValueError("Selection score is not finite")
                    trial.update(status="completed", score=score, metrics=result["metrics"], result=result, reused=False)
                except InterruptedError:
                    raise
                except Exception as exc:
                    trial.update(status="failed", score=None, error=str(exc), reused=False)
            emit({"event": "trial_completed", "phase": phase, "fold_id": scope, "trial": trial})
            compact_trial = dict(trial) if retain_trial_results else {key: value for key, value in trial.items() if key != "result"}
            if "result" in trial and not retain_trial_results:
                compact_trial["artifact_key"] = {"fold_id": scope, "candidate_id": trial["candidate_id"]}
            trials.append(compact_trial)
        valid = [trial for trial in trials if trial["status"] == "completed"]
        valid.sort(key=lambda trial: ((-trial["score"] if maximize else trial["score"]), trial["candidate_id"]))
        return trials, valid[0] if valid else None

    def validation_result(start: int, end: int, candidate: dict) -> dict:
        check_cancelled()
        return run_comparison(frame.iloc[start:end], strategy, candidate["parameters"], execution,
                              warmup_bars=frame.iloc[:start], cancel_check=cancel_check)

    folds = []
    for boundaries in plan["folds"]:
        check_cancelled()
        fold_id = boundaries["fold_id"]
        periods = {"train": period(boundaries["train_start"], boundaries["train_end"]),
                   "gap": period(boundaries["gap_start"], boundaries["gap_end"]),
                   "validation": period(boundaries["validation_start"], boundaries["validation_end"])}
        cached_fold = fold_cache_provider(fold_id) if fold_cache_provider else None
        if cached_fold is not None:
            if cached_fold.get("fold_id") != fold_id or cached_fold.get("periods") != periods or cached_fold.get("status") not in {"completed", "failed"}:
                raise ValueError("Cached fold boundaries/status mismatch")
            fold = {**cached_fold, "reused": True}
            folds.append(fold)
            emit({"event": "fold_completed", "fold_id": fold_id, "fold": fold})
            continue
        trials, winner = fit(fold_id, boundaries["train_start"], boundaries["train_end"], phase="training")
        fold = {"fold_id": fold_id, "periods": periods, "train_trials": trials,
                "selected_candidate_id": winner["candidate_id"] if winner else None,
                "selected_parameters": dict(winner["parameters"]) if winner else None,
                "selection_metric": metric, "validation": None, "status": "failed"}
        if winner is not None:
            try:
                fold["validation"] = validation_result(boundaries["validation_start"], boundaries["validation_end"], winner)
                fold["status"] = "completed"
            except InterruptedError:
                raise
            except Exception as exc:
                fold["error"] = str(exc)
        else:
            fold["error"] = "No valid training candidate; validation was not evaluated."
        folds.append(fold)
        emit({"event": "fold_completed", "fold_id": fold_id, "fold": fold})

    holdout_range = plan["holdout"]
    holdout = {"status": "reserved", "period": period(holdout_range["start"], holdout_range["end"]),
               "evaluated": False, "result": None}
    if holdout_range["evaluate"]:
        cached_holdout = holdout_cache_provider() if holdout_cache_provider else None
        calibration = plan["calibration"]
        if cached_holdout is not None:
            if cached_holdout.get("period") != holdout["period"] or cached_holdout.get("status") not in {"completed", "failed"}:
                raise ValueError("Cached holdout boundaries/status mismatch")
            holdout = {**cached_holdout, "reused": True}
            emit({"event": "holdout_completed", "holdout": holdout})
            trials, winner = [], None
        else:
            trials, winner = fit("holdout", calibration["train_start"], calibration["train_end"], phase="holdout_calibration")
        if cached_holdout is None:
            holdout.update(train_trials=trials,
                           calibration_period=period(calibration["train_start"], calibration["train_end"]),
                           gap_period=period(calibration["gap_start"], calibration["gap_end"]),
                           selected_candidate_id=winner["candidate_id"] if winner else None,
                           selected_parameters=dict(winner["parameters"]) if winner else None)
            if winner is not None:
                try:
                    holdout["result"] = validation_result(holdout_range["start"], holdout_range["end"], winner)
                    holdout.update(status="completed", evaluated=True)
                except InterruptedError:
                    raise
                except Exception as exc:
                    holdout.update(status="failed", error=str(exc))
            else:
                holdout.update(status="failed", error="No valid calibration candidate; holdout was not evaluated.")
            emit({"event": "holdout_completed", "holdout": holdout})

    completed = [fold for fold in folds if fold["status"] == "completed"]
    returns = [fold["validation"]["strategy"]["metrics"]["total_return"] for fold in completed]
    excess = [fold["validation"]["excess_return_vs_buy_hold"] for fold in completed]
    drawdowns = [fold["validation"]["strategy"]["metrics"]["maximum_drawdown"] for fold in completed]
    status = "completed" if len(completed) == len(folds) and holdout["status"] != "failed" else "partial" if completed else "failed"
    return {"mode": "walk_forward", "status": status, "strategy_name": strategy,
            "plan": plan, "folds": folds, "holdout": holdout,
            "summary": {"total_folds": len(folds), "completed_folds": len(completed), "failed_folds": len(folds) - len(completed),
                        "mean_validation_return": mean(returns) if returns else None,
                        "median_validation_return": median(returns) if returns else None,
                        "positive_return_fold_ratio": mean(value > 0 for value in returns) if returns else None,
                        "mean_excess_return": mean(excess) if excess else None,
                        "win_rate_vs_buy_hold": mean(value > 0 for value in excess) if excess else None,
                        "maximum_fold_drawdown": max(drawdowns) if drawdowns else None},
            "assumptions": [
                "Selection uses only each fold's past training interval; validation and holdout are never ranking inputs.",
                "Every fold starts a fresh cash account. Summary statistics are independent-window diagnostics, not a continuous portfolio return.",
                "Earlier visible closes, including the gap, warm up the selected strategy; warmup bars never trade or contribute performance.",
                "The last warmup close may submit an order for the first validation open. All candidates and baselines use the same execution rules.",
                "Holdout is reserved by default; explicit evaluation uses a winner selected only on the final pre-holdout calibration interval.",
                "Repeatedly inspecting holdout results and changing the search invalidates an untouched-test claim; every usage must remain in the experiment log.",
                "Incomplete trailing development windows are not scored; their count is recorded in the plan.",
            ]}
