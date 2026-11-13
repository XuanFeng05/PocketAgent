from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

from backtest_layer.data import demo_bars, load_daily_bars, normalize_daily_bars
from backtest_layer.engine import run_backtest, run_comparison
from backtest_layer.execution import ExecutionConfig
from research_layer.batch import preflight_experiment as _batch_preflight, run_parameter_batch, positive_integer


def preflight_experiment(config: dict, bar_count: int | None = None) -> dict:
    mode = config.get("mode") or "single"
    if mode in {"portfolio", "universe"}:
        from strategy_layer.portfolio import resolve_portfolio_settings
        symbols = config.get("symbols") or []
        if not isinstance(symbols, list) or any(not isinstance(s, str) or not s.strip() for s in symbols):
            raise ValueError("symbols must be an explicit list of identifiers")
        if mode == "portfolio":
            settings = resolve_portfolio_settings(config.get("portfolio"))
            strategy = settings.strategy
            count = 3
        else:
            strategy = str(config.get("strategy") or "sma_trend")
            _batch_preflight({**config, "mode": "single"})
            count = 3 * (len(symbols) or 3)
            positive_integer(config.get("max_symbols", 20), "max_symbols", maximum=100)
        max_runs = positive_integer(config.get("max_backtests", 512), "max_backtests", maximum=2048)
        max_bars = positive_integer(config.get("max_bar_evaluations", 2_000_000), "max_bar_evaluations", maximum=20_000_000)
        base = {"mode": mode, "strategy": strategy, "candidates": [], "candidate_count": 1,
                "max_candidates": 1, "max_backtests": max_runs, "max_bar_evaluations": max_bars,
                "estimated_backtests": count, "estimated_bar_evaluations": None if bar_count is None else 3 * bar_count,
                "workers": 1, "selection_metric": "total_return", "walk_forward_plan": None}
    elif mode == "screen":
        base = _batch_preflight({**config, "mode": "batch"}, bar_count)
        base = {**base, "mode": "screen"}
    else:
        base = _batch_preflight(config, bar_count)
    raw = config.get("robustness") or {}
    if not isinstance(raw, dict):
        raise ValueError("robustness must be an object")
    if raw.get("enabled", False) not in (True, False):
        raise ValueError("robustness.enabled must be boolean")
    multipliers = raw.get("cost_multipliers", [1, 1.5, 2]) if raw.get("enabled") else []
    if not isinstance(multipliers, list) or len(multipliers) > 6 or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or not .25 <= v <= 4 for v in multipliers):
        raise ValueError("Use at most six finite cost multipliers between 0.25 and 4")
    partitions = raw.get("partitions", 4)
    if isinstance(partitions,bool) or not isinstance(partitions,int) or partitions not in {2,4,6,8,10,12}:
        raise ValueError("robustness.partitions must be an even integer from 2 to 12")
    seed = config.get("random_seed")
    if seed is not None and (isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed < 2**32):
        raise ValueError("random_seed must be an integer in [0, 2**32)")
    additional = len({float(x) for x in multipliers if x != 1}) if mode != "screen" else 0
    random_runs = 0
    if seed is not None and mode in {"single","batch"}:
        random_runs = 1
    elif seed is not None and mode == "walk_forward" and base["walk_forward_plan"]:
        plan = base["walk_forward_plan"]
        random_runs = len(plan["folds"]) + int(plan["holdout"]["evaluate"])
    if base["estimated_backtests"] is not None:
        base["estimated_backtests"] += additional + random_runs
    if base["estimated_bar_evaluations"] is not None:
        base["estimated_bar_evaluations"] += (additional + random_runs) * bar_count
    if base["estimated_backtests"] is not None and base["estimated_backtests"] > base["max_backtests"]:
        raise ValueError("Combined experiment and diagnostic work exceeds max_backtests")
    if base["estimated_bar_evaluations"] is not None and base["estimated_bar_evaluations"] > base["max_bar_evaluations"]:
        raise ValueError("Combined experiment and diagnostic work exceeds max_bar_evaluations")
    return base


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_input(config: dict):
    source_type = str(config.get("source_type") or "demo").lower()
    if source_type not in {"demo", "csv", "market"}:
        raise ValueError("source_type must be demo, csv or market.")
    mode = config.get("mode") or "single"
    multi = mode in {"portfolio", "universe"}
    if multi:
        from backtest_layer.portfolio_data import demo_portfolio_bars, load_portfolio_bars
        if source_type == "demo":
            bars = demo_portfolio_bars(symbols=config.get("symbols") or None)
        elif mode == "portfolio":
            bars = load_portfolio_bars(csv_path=config.get("source_path") if source_type=="csv" else None,
                                       db_path=config.get("source_path") if source_type=="market" else None,
                                       symbols=config.get("symbols") or None,start=config.get("start"),end=config.get("end"),
                                       require_membership=bool((config.get("portfolio") or {}).get("require_membership",False)))
        elif source_type == "csv":
            bars = pd.read_csv(config["source_path"],dtype={"symbol":str})
        else:
            symbols = config.get("symbols")
            if not symbols:
                raise ValueError("Market universe research requires an explicit symbols list")
            pieces = []
            for symbol in symbols:
                try:
                    pieces.append(load_daily_bars(db_path=config["source_path"],symbol=symbol,start=config.get("start"),end=config.get("end")))
                except Exception as exc:
                    pieces.append(pd.DataFrame([{"symbol":symbol,"source_error":str(exc)}]))
            bars = pd.concat(pieces,ignore_index=True)
        if "datetime" not in bars and "date" in bars:
            bars = bars.rename(columns={"date":"datetime"})
        if config.get("symbols") and "symbol" in bars:
            bars = bars.loc[bars["symbol"].isin(config["symbols"])].copy()
        if "datetime" in bars:
            dates = pd.to_datetime(bars["datetime"],errors="coerce")
            if config.get("start"):
                bars = bars.loc[dates.isna() | dates.ge(config["start"])].copy()
                dates = dates.loc[bars.index]
            if config.get("end"):
                bars = bars.loc[dates.isna() | dates.le(config["end"])].copy()
    elif source_type == "demo":
        bars = normalize_daily_bars(demo_bars(symbol=str(config.get("symbol") or "DEMO")))
        if config.get("start"):
            bars = bars.loc[bars["datetime"] >= config["start"]].copy()
        if config.get("end"):
            bars = bars.loc[bars["datetime"] <= config["end"]].copy()
        bars = normalize_daily_bars(bars)
    else:
        source_path = config.get("source_path")
        if not source_path:
            raise ValueError("source_path is required for CSV or market data.")
        bars = load_daily_bars(csv_path=source_path if source_type == "csv" else None,
                           db_path=source_path if source_type == "market" else None,
                           symbol=config.get("symbol") or None, start=config.get("start") or None,
                           end=config.get("end") or None)
    context = config.get("context")
    if context and not config.get("snapshot_ready"):
        from feature_layer.context import enrich_daily_bars,demo_context_sources
        if not isinstance(context,dict):
            raise ValueError("context must be an object")
        sources = demo_context_sources(bars) if context.get("demo") else {
            "benchmark":context.get("benchmark_path"),"industry":context.get("industry_path"),"membership":context.get("membership_path")}
        bars = enrich_daily_bars(bars,**sources,benchmark_symbol=context.get("benchmark_symbol"),
                                  require_complete=context.get("require_complete",False))
    return bars


def _implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = {p for module in ["backtest_layer", "strategy_layer", "research_layer", "evaluation_layer/reports"]
             for p in (root / module).rglob("*.py")}
    paths.update((root / "evaluation_layer/metrics").glob("*.py"))
    paths.update([root / "feature_layer/rules.py", root / "feature_layer/context.py"])
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _save_backtest(path: Path, report: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "result.json", report)
    _write_csv(path / "equity.csv", report.get("equity_curve", []))
    _write_csv(path / "fills.csv", report.get("fills", []))
    if report.get("positions") is not None:
        _write_csv(path / "positions.csv",report["positions"])
    fifo = report.get("fifo_report") or report.get("trade_report")
    if fifo:
        _write_json(path / "fifo_report.json", fifo)
        _write_csv(path / "fifo_closed.csv", fifo.get("closed_trades", []))
        _write_csv(path / "fifo_open.csv", fifo.get("open_lots", []))


def _save_comparison(path: Path, comparison: dict) -> None:
    report = comparison.get("strategy")
    if report:
        _save_backtest(path, report)
    for name, report in comparison.get("baselines", {}).items():
        _save_backtest(path / "baselines" / name, report)


def _safe_key(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))[:120] or "item"


def run_experiment(
    config: dict[str, Any], output_root: str | Path,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run bounded CPU research and retain all attempts, including failures."""
    json.dumps(config, ensure_ascii=False, allow_nan=False)
    preflight = preflight_experiment(config)
    mode, strategy = preflight["mode"], preflight["strategy"]
    execution = ExecutionConfig(**dict(config.get("execution") or {}))
    run_id = f"research_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = {**config, "mode": mode, "strategy": strategy, "execution": asdict(execution),
                "workers": 1, "max_candidates": preflight["max_candidates"], "max_backtests": preflight["max_backtests"],
                "max_bar_evaluations": preflight["max_bar_evaluations"]}
    _write_json(run_dir / "config.json", resolved)
    _write_json(run_dir / "status.json", {"run_id": run_id, "status": "running", "phase": "loading"})
    trial_summaries: list[dict] = []
    completed_folds: list[dict] = []
    completed = 0
    total = preflight["estimated_backtests"] or 0
    artifact_hashes: dict[str, str] = {}

    def record_artifacts(*paths: Path) -> None:
        for path in paths:
            files = path.rglob("*") if path.is_dir() else [path]
            for item in files:
                if item.is_file() and not item.name.endswith(".tmp"):
                    artifact_hashes[item.relative_to(run_dir).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
        _write_json(run_dir / "artifacts.json", {"schema_version": 1, "files": artifact_hashes})

    def check_cancelled():
        if cancel_check and cancel_check():
            raise InterruptedError("Experiment cancelled; completed trial artifacts were retained")

    def notify(message: str, **extra):
        update = {"run_id": run_id, "path": str(run_dir), "mode": mode,
                  "completed": completed, "total": total,
                  "progress": min(1.0, completed / total) if total else 0.0, "message": message, **extra}
        _write_json(run_dir / "status.json", {**update, "status": "running"})
        if progress_callback:
            progress_callback(update)

    def record_event(event: dict):
        nonlocal completed
        if event.get("event") == "baseline_completed":
            completed += 1
        elif event.get("event") == "trial_completed":
            trial = event["trial"]
            phase = str(event.get("phase") or "batch")
            fold_id = event.get("fold_id", "batch")
            trial_key = "_".join(_safe_key(x) for x in [phase, fold_id, trial["candidate_id"]])
            trial_dir = run_dir / "trials" / trial_key
            _write_json(trial_dir / "config.json", {"strategy": strategy, "parameters": trial["parameters"],
                                                       "execution": asdict(execution), "phase": phase, "fold_id": fold_id})
            _write_json(trial_dir / "status.json", {key: value for key, value in trial.items() if key != "result"})
            if trial.get("result"):
                if "strategy" in trial["result"] and isinstance(trial["result"]["strategy"], dict):
                    # Baselines are shared and stored once with the selected
                    # report, not duplicated for every parameter candidate.
                    _save_backtest(trial_dir, trial["result"]["strategy"])
                else:
                    _save_backtest(trial_dir, trial["result"])
            summary = {key: value for key, value in trial.items() if key != "result"}
            summary.update(phase=phase, fold_id=fold_id, artifact_path=str(trial_dir.relative_to(run_dir)))
            trial["artifact_path"] = summary["artifact_path"]
            trial_summaries.append(summary)
            with (run_dir / "trials.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n")
            record_artifacts(trial_dir, run_dir / "trials.jsonl")
            completed += 1
        elif event.get("event") in {"validation_completed", "fold_completed", "holdout_completed"}:
            completed += 3
            fold = event.get("fold") or event.get("holdout") or {key: value for key, value in event.items() if key != "event"}
            completed_folds.append(fold)
            _write_json(run_dir / "folds" / f"{_safe_key(event.get('fold_id', 'holdout'))}.json", fold)
            comparison = fold.get("validation") or fold.get("result")
            if comparison:
                _save_comparison(run_dir / "folds" / _safe_key(event.get("fold_id", "holdout")) / "validation", comparison)
            record_artifacts(run_dir / "folds" / f"{_safe_key(event.get('fold_id', 'holdout'))}.json",
                             run_dir / "folds" / _safe_key(event.get("fold_id", "holdout")))
        notify(str(event.get("message") or f"Completed {completed}/{total} backtests"), event=event.get("event"))

    try:
        notify("Loading read-only input")
        check_cancelled()
        bars = _load_input(config)
        preflight = preflight_experiment(config, len(bars))
        total = preflight["estimated_backtests"] or 1
        data_hash = hashlib.sha256(bars.to_json(orient="records", date_format="iso", double_precision=15).encode()).hexdigest()
        resolved.update(source_type=str(config.get("source_type") or "demo").lower(),
                        source_path=str(config.get("source_path") or ""),
                        synthetic=bool(config.get("synthetic", str(config.get("source_type") or "demo").lower() == "demo")),
                        symbol=str(bars.iloc[0]["symbol"]), start=bars.iloc[0]["datetime"].date().isoformat(),
                        end=bars.iloc[-1]["datetime"].date().isoformat(), data_sha256=data_hash,
                        implementation_sha256=_implementation_hash(),
                        candidate_count=preflight["candidate_count"], estimated_backtests=total,
                        estimated_bar_evaluations=preflight["estimated_bar_evaluations"])
        identity = hashlib.sha256(json.dumps(resolved, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        resolved["config_hash"] = identity
        _write_json(run_dir / "config.json", resolved)
        bars.to_csv(run_dir / "input.csv", index=False)
        _write_json(run_dir / "plan.json", preflight)
        record_artifacts(run_dir / "config.json", run_dir / "input.csv", run_dir / "plan.json")
        notify("Input snapshot saved; running one CPU worker")
        check_cancelled()
        random_seed = config.get("random_seed")
        if mode == "single":
            if config.get("include_baselines", True):
                result = run_comparison(bars, strategy, config.get("parameters"), execution, cancel_check=cancel_check, random_seed=random_seed)
            else:
                result = {"strategy": run_backtest(bars, strategy, config.get("parameters"), execution, cancel_check=cancel_check), "baselines": {}}
            check_cancelled()
            result.update(mode="single", selection_scope="single_configuration")
        elif mode == "batch":
            batch = run_parameter_batch(bars, strategy, preflight["candidates"], execution,
                                        selection_metric=preflight["selection_metric"],
                                        include_baselines=bool(config.get("include_baselines", True)),
                                        progress_callback=record_event, cancel_check=cancel_check, retain_results=False)
            best = batch.pop("best_result") or {"strategy": None, "baselines": batch["baselines"]}
            batch.pop("baselines", None)
            result = {**best, "mode": "batch", "batch": batch, "selection_scope": "in_sample"}
        elif mode == "walk_forward":
            from research_layer.walk_forward import run_walk_forward
            workflow = run_walk_forward(bars, strategy, preflight["candidates"], execution,
                                        settings=config.get("walk_forward"),
                                        progress_callback=record_event, cancel_check=cancel_check, retain_trial_results=False)
            validations = [fold.get("validation") for fold in workflow["folds"] if fold.get("validation")]
            latest = validations[-1] if validations else {"strategy": None, "baselines": {}}
            result = {**latest, "mode": "walk_forward", "walk_forward": workflow,
                      "selection_scope": "last_validation_fold"}
        elif mode == "screen":
            from research_layer.screening import screen_candidates
            result = screen_candidates(bars, strategy, preflight["candidates"], settings=config.get("screen"), cancel_check=cancel_check)
            result["selection_scope"] = "training_only_approximate"
        elif mode == "portfolio":
            from backtest_layer.portfolio import run_portfolio_comparison
            result = run_portfolio_comparison(bars, settings=config.get("portfolio"), execution_config=execution, cancel_check=cancel_check)
            result["selection_scope"] = "shared_cash_portfolio"
        elif mode == "universe":
            from research_layer.universe import run_universe
            result = run_universe(bars, strategy=strategy, parameters=config.get("parameters"), execution_config=execution,
                                  max_symbols=int(config.get("max_symbols", 20)), cancel_check=cancel_check)
            result["selection_scope"] = "selected_symbol_not_aggregate"
        else:
            raise ValueError(f"Unsupported research mode: {mode}")
        result.update(run_id=run_id, synthetic=resolved["synthetic"], config_hash=identity,
                      data_sha256=data_hash, implementation_sha256=resolved["implementation_sha256"])
        if config.get("robustness", {}).get("enabled") and result.get("strategy"):
            from research_layer.robustness import analyze_robustness
            def _stress(execution_override):
                return run_backtest(bars, strategy, config.get("parameters"), ExecutionConfig(**execution_override))
            diagnostics = analyze_robustness(result, run_dir, rerun=_stress,
                                              cost_multipliers=config.get("robustness", {}).get("cost_multipliers", [1, 1.5, 2]),
                                              partitions=config.get("robustness", {}).get("partitions", 4), cancel_check=cancel_check)
            result["robustness"] = diagnostics
            _write_json(run_dir / "robustness.json", diagnostics)
        _save_comparison(run_dir / "selected_report", result)
        if result.get("strategy"):
            _write_csv(run_dir / "equity.csv", result["strategy"]["equity_curve"])
            _write_csv(run_dir / "fills.csv", result["strategy"]["fills"])
            if result["strategy"].get("positions") is not None:
                _write_csv(run_dir / "positions.csv", result["strategy"]["positions"])
            fifo = result["strategy"].get("fifo_report") or result["strategy"].get("trade_report")
            if fifo:
                _write_json(run_dir / "fifo_report.json", fifo)
                _write_csv(run_dir / "fifo_closed.csv", fifo.get("closed_trades", []))
                _write_csv(run_dir / "fifo_open.csv", fifo.get("open_lots", []))
        _write_csv(run_dir / "trials.csv", [{**{k: v for k, v in trial.items() if k not in {"parameters", "metrics"}},
                                               "parameters": json.dumps(trial["parameters"], sort_keys=True),
                                               **{f"metric_{k}": v for k, v in trial.get("metrics", {}).items()}}
                                              for trial in trial_summaries])
        _write_json(run_dir / "result.json", result)
        record_artifacts(run_dir / "selected_report", run_dir / "result.json", run_dir / "trials.csv",
                         *[p for p in [run_dir / "equity.csv", run_dir / "fills.csv", run_dir / "positions.csv", run_dir / "fifo_report.json", run_dir / "fifo_closed.csv", run_dir / "fifo_open.csv", run_dir / "robustness.json"] if p.exists()])
        completed = total
        notify("Experiment completed")
        _write_json(run_dir / "status.json", {"run_id": run_id, "status": "completed", "mode": mode,
                                                "completed": completed, "total": total, "progress": 1.0})
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, InterruptedError) else "failed"
        _write_json(run_dir / "partial_result.json", {"run_id": run_id, "mode": mode, "status": status,
                                                         "trials": trial_summaries, "folds": completed_folds,
                                                         "error": str(exc)})
        _write_json(run_dir / "status.json", {"run_id": run_id, "mode": mode, "status": status,
                                                "error": str(exc), "completed": completed, "total": total})
        record_artifacts(run_dir / "partial_result.json")
        raise
    return {"run_id": run_id, "status": "completed", "path": str(run_dir), "result": result}
