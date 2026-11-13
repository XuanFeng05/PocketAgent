"""Bounded, sequential parameter experiments with explicit in-sample ranking."""
from __future__ import annotations

import hashlib
from itertools import product
import json
import math
from typing import Any, Callable

from backtest_layer.engine import run_backtest
from backtest_layer.execution import ExecutionConfig
from strategy_layer.registry import strategy_catalog

DEFAULT_MAX_CANDIDATES = 64
DEFAULT_MAX_BACKTESTS = 512
HARD_MAX_CANDIDATES = 256
HARD_MAX_BACKTESTS = 2048
DEFAULT_MAX_BAR_EVALUATIONS = 2_000_000
HARD_MAX_BAR_EVALUATIONS = 20_000_000
SELECTION_METRICS = {"total_return", "sharpe", "maximum_drawdown"}


def positive_integer(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(float(value)) or int(value) != value or not 1 <= int(value) <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return int(value)


def expand_parameter_grid(
    parameter_grid: dict[str, list[Any]] | None,
    parameters: dict[str, Any] | None = None,
    *, max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Reject oversized grids before building the Cartesian product."""
    limit = positive_integer(max_candidates, "max_candidates", maximum=HARD_MAX_CANDIDATES)
    grid = {} if parameter_grid is None else parameter_grid
    base = {} if parameters is None else parameters
    if not isinstance(grid, dict) or not isinstance(base, dict):
        raise ValueError("parameter_grid and parameters must be JSON objects")
    if any(not isinstance(key, str) or not key for key in [*grid, *base]):
        raise ValueError("Parameter names must be nonempty strings")
    values = []
    count = 1
    for name in sorted(grid):
        entries = grid[name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"parameter_grid.{name} must be a nonempty array")
        unique = []
        seen = set()
        for value in entries:
            key = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if key not in seen:
                unique.append(value)
                seen.add(key)
                if len(unique) > limit:
                    raise ValueError(f"Parameter grid has more than {limit} candidates; reduce the ranges")
        values.append(unique)
        count *= len(unique)
        if count > limit:
            raise ValueError(f"Parameter grid has more than {limit} candidates; reduce the ranges")
    result = []
    for selected in product(*values):
        params = {**base, **dict(zip(sorted(grid), selected))}
        encoded = json.dumps(params, sort_keys=True, ensure_ascii=False, allow_nan=False)
        candidate_id = "candidate_" + hashlib.sha256(encoded.encode()).hexdigest()[:16]
        result.append({"candidate_id": candidate_id, "parameters": params})
    return sorted(result, key=lambda item: item["candidate_id"])


def preflight_experiment(config: dict[str, Any], bar_count: int | None = None) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    mode = config.get("mode") or "single"
    if mode not in {"single", "batch", "walk_forward"}:
        raise ValueError("mode must be single, batch or walk_forward")
    strategy = str(config.get("strategy") or "sma_trend")
    if strategy not in {item["name"] for item in strategy_catalog()}:
        raise ValueError(f"Unknown strategy {strategy!r}")
    budget = config.get("budget") or {}
    if not isinstance(budget, dict):
        raise ValueError("budget must be an object")
    limit = positive_integer(config.get("max_candidates", budget.get("max_candidates", DEFAULT_MAX_CANDIDATES)),
                             "max_candidates", maximum=HARD_MAX_CANDIDATES)
    backtests = positive_integer(config.get("max_backtests", budget.get("max_backtests", DEFAULT_MAX_BACKTESTS)),
                                 "max_backtests", maximum=HARD_MAX_BACKTESTS)
    bar_budget = positive_integer(config.get("max_bar_evaluations", DEFAULT_MAX_BAR_EVALUATIONS),
                                  "max_bar_evaluations", maximum=HARD_MAX_BAR_EVALUATIONS)
    candidates = expand_parameter_grid(config.get("parameter_grid") if mode != "single" else {},
                                       config.get("parameters"), max_candidates=limit)
    metric = str(config.get("selection_metric") or "total_return")
    if metric not in SELECTION_METRICS:
        raise ValueError("selection_metric must be total_return, sharpe or maximum_drawdown")
    estimated = (1 if mode == "single" else len(candidates)) + (2 if config.get("include_baselines", True) else 0)
    plan = None
    bar_work = None if bar_count is None else estimated * bar_count
    if mode == "walk_forward":
        from research_layer.walk_forward import build_walk_forward_plan, resolve_walk_forward_settings, MAX_CANDIDATES
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError(f"Walk-forward supports at most {MAX_CANDIDATES} candidates")
        settings = resolve_walk_forward_settings(config.get("walk_forward"))
        # Use the exact same validation as the data-aware plan, without
        # pretending to know the number of folds before input is loaded.
        if bar_count is None:
            estimated = None
        else:
            plan = build_walk_forward_plan(bar_count, settings)
            estimated = len(plan["folds"]) * (len(candidates) + 3)
            bar_work = sum(fold["train_end"] * len(candidates) + fold["validation_end"] * 3 for fold in plan["folds"])
            if plan["holdout"]["evaluate"]:
                estimated += len(candidates) + 3
                bar_work += plan["calibration"]["train_end"] * len(candidates) + plan["holdout"]["end"] * 3
    if estimated is not None and estimated > backtests:
        raise ValueError(f"Experiment needs {estimated} backtests, over the max_backtests budget {backtests}")
    if bar_work is not None and bar_work > bar_budget:
        raise ValueError(f"Experiment estimates {bar_work} bar evaluations, over max_bar_evaluations {bar_budget}; reduce dates or candidates")
    return {"mode": mode, "strategy": strategy, "candidates": candidates,
            "candidate_count": len(candidates), "max_candidates": limit, "max_backtests": backtests,
            "estimated_backtests": estimated, "workers": 1, "selection_metric": metric,
            "estimated_bar_evaluations": bar_work, "max_bar_evaluations": bar_budget,
            "walk_forward_plan": plan}


def rank_trials(trials: list[dict], metric: str) -> list[dict]:
    ranking = []
    for trial in trials:
        if trial["status"] != "completed":
            continue
        value = trial["metrics"].get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        score = -float(value) if metric == "maximum_drawdown" else float(value)
        ranking.append({"candidate_id": trial["candidate_id"], "parameters": trial["parameters"],
                        "metric": metric, "metric_value": float(value), "score": score,
                        "metrics": trial["metrics"], "ranking_scope": "in_sample"})
    ranking.sort(key=lambda item: (-item["score"], item["candidate_id"]))
    for index, item in enumerate(ranking, start=1):
        item["rank"] = index
    return ranking


def run_parameter_batch(
    bars, strategy: str, candidates: list[dict], execution_config: ExecutionConfig,
    *, selection_metric: str = "total_return", include_baselines: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    retain_results: bool = True,
    cache_provider: Callable[[dict], dict | None] | None = None,
) -> dict:
    if selection_metric not in SELECTION_METRICS:
        raise ValueError("Unsupported selection metric")

    def check_cancelled():
        if cancel_check and cancel_check():
            raise InterruptedError("Parameter experiment cancelled; completed trials are retained")

    baselines = {}
    for name in ("cash", "buy_hold") if include_baselines else ():
        check_cancelled()
        baselines[name] = run_backtest(bars, name, execution_config=execution_config, cancel_check=cancel_check)
        if progress_callback:
            progress_callback({"event": "baseline_completed", "phase": "batch", "baseline": name})
    trials = []
    best_result = None
    best_key = None
    for index, candidate in enumerate(candidates, start=1):
        check_cancelled()
        trial = {**candidate, "trial_id": candidate["candidate_id"], "phase": "batch", "ranking_scope": "in_sample"}
        cached = cache_provider(candidate) if cache_provider else None
        if cached is not None:
            if cached.get("candidate_id") != candidate["candidate_id"] or cached.get("parameters") != candidate["parameters"] or cached.get("status") not in {"completed", "failed"}:
                raise ValueError("Cached candidate identity/status mismatch")
            trial.update(cached, reused=True)
            if trial["status"] == "completed":
                report = trial["result"].get("strategy", trial["result"])
                trial["result"] = {"strategy": report, "baselines": baselines}
        else:
            try:
                report = run_backtest(bars, strategy, candidate["parameters"], execution_config, cancel_check=cancel_check)
                trial.update(status="completed", metrics=report["metrics"],
                             result={"strategy": report, "baselines": baselines}, reused=False)
            except InterruptedError:
                raise
            except Exception as exc:
                trial.update(status="failed", error=str(exc), metrics={}, reused=False)
        if progress_callback:
            progress_callback({"event": "trial_completed", "phase": "batch", "trial": trial,
                               "completed": index, "total": len(candidates), "progress": index / len(candidates)})
        ranking = rank_trials([trial], selection_metric)
        if ranking:
            key = (-ranking[0]["score"], trial["candidate_id"])
            if best_key is None or key < best_key:
                best_key, best_result = key, trial["result"]
        trials.append(trial if retain_results else {key: value for key, value in trial.items() if key != "result"})
    check_cancelled()
    ranking = rank_trials(trials, selection_metric)
    return {"mode": "batch", "trials": trials, "ranking": ranking,
            "candidate_count": len(candidates), "completed_count": sum(t["status"] == "completed" for t in trials),
            "failed_count": sum(t["status"] == "failed" for t in trials),
            "best_candidate_id": ranking[0]["candidate_id"] if ranking else None,
            "selection_metric": selection_metric, "ranking_scope": "in_sample",
            "best_result": best_result,
            "baselines": baselines,
            "assumptions": ["Ranking is in-sample research only; it is not out-of-sample evidence.",
                            "All attempted candidates, including failures, are retained. One CPU worker runs sequentially."]}
