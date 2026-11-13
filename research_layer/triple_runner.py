"""Persist and batch one frozen Triple Screen rule without changing market data."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from statistics import mean, median
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from research_layer.technical import run_technical_backtest, normalize_technical_bars
from research_layer.triple_screen import run_triple_screen
from strategy_layer.triple_screen import resolve_triple_settings


def _json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = ["research_layer/triple_runner.py", "research_layer/triple_screen.py",
             "research_layer/technical_runner.py", "research_layer/technical.py",
             "feature_layer/triple_screen.py", "strategy_layer/triple_screen.py",
             "strategy_layer/base.py", "strategy_layer/registry.py", "strategy_layer/lightweight.py",
             "evaluation_layer/metrics/portfolio.py", "data_layer/storage/duckdb_storage.py",
             "data_layer/storage/partitioned_storage.py"]
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def normalize_triple_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Triple Screen configuration must be an object.")
    settings = resolve_triple_settings(config.get("settings"))
    variant = config.get("variant", "full")
    if variant not in {"full", "drop_weekly", "drop_pullback", "drop_trigger"}:
        raise ValueError("Unknown Triple Screen research variant.")
    if config.get("adjust", "pre") != "pre":
        raise ValueError("Triple Screen technical research requires adjust=pre.")
    if config.get("mode", "triple_screen") != "triple_screen":
        raise ValueError("This entry point only supports mode=triple_screen.")
    if config.get("research_scope", "exploratory") != "exploratory":
        raise ValueError("This entry point is exploratory; an untouched test requires a separately registered protocol.")
    source = config.get("source_type", "market")
    if source not in {"market", "csv"} or not config.get("source_path"):
        raise ValueError("Provide an existing market or CSV source_path.")
    raw_symbols = config.get("symbols")
    if raw_symbols is None:
        raw_symbols = [config.get("symbol")]
    if not isinstance(raw_symbols, list) or not 1 <= len(raw_symbols) <= 10000:
        raise ValueError("Provide an explicit symbols list of 1 to 10000 stocks.")
    symbols = [str(symbol).strip().upper() for symbol in raw_symbols]
    symbol_pattern = r"[A-Z]{1,5}(?:-[A-Z])?" if settings.is_us else r"\d{6}\.(SH|SZ|BJ)"
    if len(set(symbols)) != len(symbols) or any(not re.fullmatch(symbol_pattern, symbol) for symbol in symbols):
        raise ValueError("Symbols must be unique identifiers for the explicitly selected market/version.")
    costs = config.get("cost_scenarios_bps", [settings.cost_bps])
    if not isinstance(costs, list) or not 1 <= len(costs) <= 8:
        raise ValueError("cost_scenarios_bps needs 1 to 8 values.")
    for value in costs:
        replace(settings, cost_bps=value)
    if len(set(costs)) != len(costs):
        raise ValueError("Duplicate cost scenarios are not allowed.")
    dates = {}
    for field in ("start", "end"):
        if config.get(field):
            value = pd.Timestamp(config[field])
            if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
                raise ValueError("Use local dates without time or timezone.")
            dates[field] = value.date().isoformat()
    if dates.get("start", "") > dates.get("end", "9999-12-31"):
        raise ValueError("start must not exceed end.")
    if source == "csv" and not config.get("calendar_path"):
        raise ValueError("CSV input requires calendar_path with explicit daily open/closed facts.")
    if settings.is_us and not config.get("calendar_path"):
        raise ValueError("US input requires calendar_path; the default CN calendar must not be reused.")
    result = {**config, "schema_version": 1, "mode": "triple_screen", "adjust": "pre",
              "source_type": source, "source_path": str(Path(config["source_path"]).resolve()),
              "symbols": symbols, "settings": settings.to_dict(), "cost_scenarios_bps": [float(v) for v in costs],
              "research_scope": "exploratory", "variant": variant, **dates}
    result.pop("symbol", None)
    if config.get("calendar_path"):
        result["calendar_path"] = str(Path(config["calendar_path"]).resolve())
    return result


def _calendar(config: dict) -> pd.DataFrame:
    if config.get("calendar_path"):
        frame = pd.read_csv(config["calendar_path"])
    else:
        from data_layer.storage.duckdb_storage import load_trade_calendar_from_duckdb
        frame = load_trade_calendar_from_duckdb(config["source_path"], exchange="CN")
    if frame.empty or not {"date", "is_trading_day"}.issubset(frame.columns):
        raise ValueError("Explicit trade calendar is missing.")
    # Calendar includes closed days and the full final week; future prices are never loaded.
    return frame


def _load_bars(config: dict, symbol: str) -> pd.DataFrame:
    if config["source_type"] == "market":
        from data_layer.storage.duckdb_storage import load_kline_from_duckdb
        frame = load_kline_from_duckdb(config["source_path"], symbol=symbol,
                                      end=config.get("end"), freq="daily", adjust="pre")
    else:
        frame = pd.read_csv(config["source_path"], dtype={"symbol": str})
    # Force Index uses share volume, not monetary turnover. Keep the provider's
    # optional amount (including missing halt-day facts) as unused provenance;
    # do not fill it or combine raw money with adjusted prices.
    if "amount" in frame:
        if "source_amount_unused" in frame:
            raise ValueError("Ambiguous source amount fields.")
        frame = frame.rename(columns={"amount": "source_amount_unused"})
    frame = normalize_technical_bars(frame, symbol=symbol)
    if config.get("end"):
        frame = frame.loc[frame["datetime"] <= pd.Timestamp(config["end"])].copy()
    return normalize_technical_bars(frame, symbol=symbol)


def _path_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its run.")
    return path


def _task_id(symbol: str, cost: float, index: int) -> str:
    identity = hashlib.sha256(float(cost).hex().encode("ascii")).hexdigest()[:12]
    return f"{symbol.replace('.', '_')}_scenario_{index:02d}_{identity}"


def _summary(rows: list[dict], costs: list[float]) -> list[dict]:
    summaries = []
    for cost in costs:
        outcomes = [row for row in rows if row["cost_bps"] == cost]
        valid = [row for row in outcomes if row["status"] == "completed"]
        returns = [row["metrics"]["total_return"] for row in valid]
        excess = [row["excess_return_vs_buy_hold"] for row in valid]
        summaries.append({"cost_bps": cost, "requested_symbols": len(outcomes),
                          "completed_symbols": len(valid), "failed_symbols": len(outcomes) - len(valid),
                          "mean_return": mean(returns) if valid else None,
                          "median_return": median(returns) if valid else None,
                          "mean_excess_return": mean(excess) if valid else None,
                          "beat_buy_hold_ratio": mean(v > 0 for v in excess) if valid else None,
                          "beat_cash_ratio": mean(v > 0 for v in returns) if valid else None,
                          "worst_maximum_drawdown": max(row["metrics"]["maximum_drawdown"] for row in valid) if valid else None,
                          "periods_aligned": len({(r["start"], r["end"]) for r in valid}) <= 1 if valid else None,
                          "aggregation": "independent_stock_distribution_not_shared_portfolio"})
    return summaries


def run_triple_experiment(config: dict, output_root: str | Path, *,
                          progress_callback: Callable[[dict], None] | None = None,
                          resume_from: str | Path | None = None,
                          cancel_check: Callable[[], bool] | None = None) -> dict:
    """Run complete stock/cost units; cancellation preserves completed units.

    A cancellation request does not interrupt a numerical backtest mid-bar. It
    is observed before input loading and between whole stock/cost steps.
    ``run_started`` publishes the durable result path before any market input.
    """
    resolved = normalize_triple_config(config)
    implementation = _implementation_identity()
    source_run = Path(resume_from).resolve() if resume_from else None
    prior_index = {}
    prior_rows = {}
    expected_tasks = {_task_id(symbol, cost, index): (symbol, cost)
                      for symbol in resolved["symbols"] for index, cost in enumerate(resolved["cost_scenarios_bps"])}
    if source_run:
        prior_config = _read(source_run / "config.json")
        if prior_config["configuration"] != resolved or prior_config["implementation_sha256"] != implementation:
            raise ValueError("Resume configuration or implementation identity differs; start a new experiment.")
        prior_manifest = _read(source_run / "artifacts.json")
        prior_index = prior_manifest["sha256"]
        for relative, digest in prior_index.items():
            path = _path_inside(source_run, relative)
            if not path.is_file() or _digest(path) != digest:
                raise ValueError(f"Resume artifact integrity failure: {relative}")
        outcome_files = prior_manifest.get("outcome_files")
        if not isinstance(outcome_files, list) or len(set(outcome_files)) != len(outcome_files):
            raise ValueError("Resume requires verified immutable outcome events.")
        for relative in outcome_files:
            if relative not in prior_index:
                raise ValueError("Resume outcome event is not verified.")
            row = _read(_path_inside(source_run, relative))
            task_id = row.get("task_id")
            if task_id not in expected_tasks or task_id in prior_rows:
                raise ValueError("Resume contains duplicate or unknown task identities.")
            if (row.get("symbol"), row.get("cost_bps")) != expected_tasks[task_id]:
                raise ValueError("Resume task metadata does not match the frozen plan.")
            prior_rows[task_id] = row
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"triple_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run = Path(output_root).resolve() / run_id
    run.mkdir(parents=True, exist_ok=False)
    source_config = {"configuration": resolved, "implementation_sha256": implementation,
                     "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
                     "created_at": created_at, "resumed_from": str(source_run) if source_run else None}
    _json(run / "config.json", source_config)
    total = len(resolved["symbols"]) * len(resolved["cost_scenarios_bps"])
    _json(run / "plan.json", {"symbols": resolved["symbols"], "cost_scenarios_bps": resolved["cost_scenarios_bps"],
                             "total_tasks": total, "scope": "exploratory", "rule_version": resolved["settings"]["version"],
                             "variant": resolved["variant"]})
    _json(run / "status.json", {"run_id": run_id, "status": "running", "completed_tasks": 0, "total_tasks": total})
    _json(run / "progress.json", {"outcomes": []})
    artifacts = {"config.json": _digest(run / "config.json"), "plan.json": _digest(run / "plan.json")}
    outcomes = []
    outcome_files = []

    def register(path: Path):
        artifacts[path.relative_to(run).as_posix()] = _digest(path)

    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Triple Screen cancelled between stock/cost steps; completed results are preserved.")

    def checkpoint():
        # Retain one immutable event per task instead of N growing snapshots.
        # Only the final atomic manifest publishes a new event for recovery.
        if len(outcome_files) < len(outcomes):
            row = outcomes[-1]
            relative = f"checkpoints/{row['task_id']}.json"
            _json(run / relative, row)
            register(run / relative)
            outcome_files.append(relative)
        _json(run / "artifacts.json", {"schema_version": 2, "sha256": artifacts, "outcome_files": outcome_files})
        _json(run / "progress.json", {"outcomes": outcomes})
        _json(run / "status.json", {"run_id": run_id, "status": "running", "completed_tasks": len(outcomes), "total_tasks": total})

    checkpoint()
    try:
        if progress_callback:
            progress_callback({"event": "run_started", "run_id": run_id, "path": str(run),
                               "completed": 0, "total": total, "message": "Preparing local input"})
        check_cancelled()
        if source_run and "calendar.csv" in prior_index:
            (run / "calendar.csv").write_bytes((source_run / "calendar.csv").read_bytes())
            calendar = pd.read_csv(run / "calendar.csv")
        else:
            calendar = _calendar(resolved)
            calendar.to_csv(run / "calendar.csv", index=False)
        register(run / "calendar.csv")
        checkpoint()
        for symbol in resolved["symbols"]:
            check_cancelled()
            key = symbol.replace(".", "_")
            relative_input = f"inputs/{key}.csv"
            load_error = None
            try:
                input_path = run / relative_input
                input_path.parent.mkdir(parents=True, exist_ok=True)
                if source_run and relative_input in prior_index:
                    input_path.write_bytes((source_run / relative_input).read_bytes())
                    bars = pd.read_csv(input_path, dtype={"symbol": str})
                    bars["datetime"] = pd.to_datetime(bars["datetime"])
                else:
                    bars = _load_bars(resolved, symbol)
                    bars.to_csv(input_path, index=False)
                register(input_path)
            except InterruptedError:
                raise
            except Exception as exc:
                load_error = str(exc)
            for scenario_index, cost in enumerate(resolved["cost_scenarios_bps"]):
                check_cancelled()
                task_id = _task_id(symbol, cost, scenario_index)
                destination = run / "outcomes" / task_id
                row = {"task_id": task_id, "symbol": symbol, "cost_bps": cost, "status": "running", "reused": False}
                try:
                    if load_error:
                        raise ValueError(load_error)
                    destination.mkdir(parents=True, exist_ok=True)
                    old = prior_rows.get(task_id)
                    if old and old["status"] == "completed":
                        relative_report = old["artifact_path"]
                        if relative_report not in prior_index or old.get("data_sha256") != artifacts[relative_input] or old.get("calendar_sha256") != artifacts["calendar.csv"]:
                            raise ValueError("Cached outcome input/calendar identity mismatch.")
                        report = _read(_path_inside(source_run, relative_report))
                        selected = report.get("strategy", {})
                        expected_settings = {**resolved["settings"], "cost_bps": cost}
                        if (selected.get("symbol") != symbol or selected.get("strategy") != "triple_screen"
                                or selected.get("research_variant") != resolved["variant"]
                                or selected.get("settings") != expected_settings or selected.get("price_basis") != "pre"
                                or selected.get("requested_start") != resolved.get("start")
                                or selected.get("requested_end") != resolved.get("end")
                                or selected.get("start") != old.get("start") or selected.get("end") != old.get("end")):
                            raise ValueError("Cached report does not match its task settings and dates.")
                        for name in ("cash", "buy_hold"):
                            baseline = report.get("baselines", {}).get(name, {})
                            if (baseline.get("symbol") != symbol or baseline.get("cost_bps") != cost
                                    or baseline.get("price_basis") != "pre"
                                    or baseline.get("start") != selected["start"] or baseline.get("end") != selected["end"]):
                                raise ValueError("Cached baseline does not match its task.")
                        row["reused"] = True
                    else:
                        settings = {**resolved["settings"], "cost_bps": cost}
                        selected = run_triple_screen(bars, calendar, settings, start=resolved.get("start"),
                                                    end=resolved.get("end"), variant=resolved["variant"])
                        dates = pd.to_datetime(bars["datetime"])
                        evaluation = bars.loc[dates.between(selected["start"], selected["end"])].copy()
                        warmup = bars.loc[dates.lt(selected["start"])].copy()
                        session_closes = ({pd.Timestamp(r["date"]).date().isoformat(): r["session_close"]
                                           for r in calendar.to_dict("records") if r["is_trading_day"]}
                                          if resolved["settings"]["version"] == "triple_screen_us_pre_v1" else None)
                        baselines = {name: run_technical_backtest(evaluation, name, cost_bps=cost, warmup_bars=warmup,
                                                                session_closes=session_closes)
                                     for name in ("cash", "buy_hold")}
                        report = {"strategy": selected, "baselines": baselines,
                                  "excess_return_vs_buy_hold": selected["metrics"]["total_return"] - baselines["buy_hold"]["metrics"]["total_return"]}
                    _json(destination / "result.json", report)
                    pd.DataFrame(report["strategy"]["equity_curve"]).to_csv(destination / "equity.csv", index=False)
                    pd.DataFrame(report["strategy"]["fills"]).to_csv(destination / "position_changes.csv", index=False)
                    for path in destination.iterdir():
                        register(path)
                    row.update(status="completed", metrics=report["strategy"]["metrics"],
                               start=report["strategy"]["start"], end=report["strategy"]["end"],
                               buy_hold_return=report["baselines"]["buy_hold"]["metrics"]["total_return"],
                               excess_return_vs_buy_hold=report["excess_return_vs_buy_hold"],
                               data_sha256=artifacts[relative_input], calendar_sha256=artifacts["calendar.csv"],
                               artifact_path=(destination / "result.json").relative_to(run).as_posix())
                except InterruptedError:
                    raise
                except Exception as exc:
                    row.update(status="failed", error=str(exc))
                    _json(destination / "error.json", row)
                    register(destination / "error.json")
                outcomes.append(row)
                checkpoint()
                if progress_callback:
                    progress_callback({"event": "task_completed", "run_id": run_id, "path": str(run),
                                       "completed": len(outcomes), "total": total, "outcome": row,
                                       "message": f"{symbol} / {cost:g} bps: {row['status']}"})
                if len(outcomes) < total:
                    check_cancelled()
        successes = sum(row["status"] == "completed" for row in outcomes)
        status = "completed" if successes == total else "partial" if successes else "failed"
        summary = {"run_id": run_id, "status": status, "price_basis": "pre", "research_scope": "exploratory",
                   "research_variant": resolved["variant"],
                   "target_symbols": len(resolved["symbols"]), "total_tasks": total,
                   "completed_tasks": successes, "failed_tasks": total-successes,
                   "by_cost": _summary(outcomes, resolved["cost_scenarios_bps"]), "outcomes": outcomes,
                   "coverage_claim": "requested_list_only_not_certified_full_market"}
        _json(run / "summary.json", summary)
        register(run / "summary.json")
        _json(run / "artifacts.json", {"schema_version": 2, "sha256": artifacts, "outcome_files": outcome_files})
        _json(run / "status.json", {"run_id": run_id, "status": status, "completed_tasks": total,
                                   "successful_tasks": successes, "failed_tasks": total-successes, "total_tasks": total})
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, InterruptedError) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        successes = sum(row["status"] == "completed" for row in outcomes)
        partial = {"run_id": run_id, "status": status, "price_basis": "pre", "research_scope": "exploratory",
                   "research_variant": resolved["variant"], "target_symbols": len(resolved["symbols"]),
                   "total_tasks": total, "recorded_tasks": len(outcomes), "completed_tasks": successes,
                   "failed_tasks": len(outcomes) - successes, "unfinished_tasks": total-len(outcomes),
                   "outcomes": outcomes, "by_cost": _summary(outcomes, resolved["cost_scenarios_bps"])}
        _json(run / "partial_result.json", partial)
        register(run / "partial_result.json")
        _json(run / "artifacts.json", {"schema_version": 2, "sha256": artifacts, "outcome_files": outcome_files})
        _json(run / "status.json", {"run_id": run_id, "status": status, "error": str(exc),
                                   "completed_tasks": len(outcomes), "successful_tasks": successes,
                                   "failed_tasks": len(outcomes)-successes, "total_tasks": total})
        raise
    return {"run_id": run_id, "status": status, "path": str(run), "summary": summary}
