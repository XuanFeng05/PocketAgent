"""Local-only, preregistered fixed-rule validation on disjoint date windows.

Every symbol/window is a fresh normalized account. This module does not select
parameters, join NAV curves, download data, or label explored history untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median
from typing import Callable
from uuid import uuid4

from research_layer.triple_runner import normalize_triple_config, run_triple_experiment, _implementation_identity


SCOPE = "exploratory_validation"
_CONFIG_KEYS = {"schema_version", "mode", "source_type", "source_path", "calendar_path", "symbols",
                "settings", "adjust", "cost_scenarios_bps", "research_scope", "notes"}
_WINDOW_KEYS = {"label", "start", "end"}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_path(value, *, file_only=False) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("A nonempty existing local data path is required.")
    raw = str(value)
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise ValueError("Only local filesystem sources are allowed; no URL or network-share input.")
    path = Path(value).expanduser().resolve()
    if not path.exists() or (file_only and not path.is_file()):
        raise ValueError(f"Local input does not exist or has the wrong type: {path}")
    return str(path)


def _date(value, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must be a YYYY-MM-DD local date.")
    return date.fromisoformat(value).isoformat()


def normalize_validation_protocol(config: dict, windows: list[dict]) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Validation configuration must be an object.")
    if unknown := set(config) - _CONFIG_KEYS:
        raise ValueError(f"Unknown validation configuration fields: {sorted(unknown)}; dates belong to windows.")
    if config.get("research_scope", SCOPE) != SCOPE:
        raise ValueError("research_scope must be exploratory_validation, never an untouched holdout claim.")
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported validation schema_version.")
    if not isinstance(config.get("symbols"), list) or not config["symbols"]:
        raise ValueError("Validation requires an explicit nonempty symbols list.")
    source_type = config.get("source_type", "market")
    if source_type not in {"market", "csv"}:
        raise ValueError("Validation supports only existing local market or CSV data.")
    local = deepcopy(config)
    local["source_path"] = _local_path(config.get("source_path"), file_only=source_type == "csv")
    if config.get("calendar_path"):
        local["calendar_path"] = _local_path(config["calendar_path"], file_only=True)
    fixed = normalize_triple_config({**local, "research_scope": "exploratory"})
    fixed["research_scope"] = SCOPE
    if not isinstance(windows, list) or not 1 <= len(windows) <= 128:
        raise ValueError("Provide 1 to 128 explicit validation windows.")
    normalized, labels = [], set()
    for item in windows:
        if not isinstance(item, dict) or set(item) != _WINDOW_KEYS:
            raise ValueError("Each window requires exactly label, start and end.")
        label = item["label"]
        if (not isinstance(label, str) or not label.strip() or len(label) > 120
                or any(ord(character) < 32 for character in label)):
            raise ValueError("Window label must be nonempty text without control characters.")
        label = label.strip()
        if label in labels:
            raise ValueError("Duplicate window labels are not allowed.")
        labels.add(label)
        start, end = _date(item["start"], "window.start"), _date(item["end"], "window.end")
        if start > end:
            raise ValueError("Window start must not exceed end.")
        normalized.append({"label": label, "start": start, "end": end})
    normalized.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    for previous, current in zip(normalized, normalized[1:]):
        if previous["end"] >= current["start"]:
            raise ValueError("Evaluation windows must not overlap, including their end dates.")
    return {"schema_version": 1, "research_scope": SCOPE, "fixed_configuration": fixed, "windows": normalized,
            "selection_policy": "fixed_settings_no_candidate_selection",
            "account_policy": "independent_initial_nav_1_per_symbol_per_window",
            "aggregation_policy": "window_by_cost_distributions_no_concatenated_nav",
            "source_policy": "local_files_only_no_downloads",
            "warmup_policy": "past_data_before_each_requested_start_not_traded",
            "coverage_claim": "explicit_requested_symbols_only_not_certified_full_market"}


def _window_rows(summary: dict, window: dict, fixed: dict) -> list[dict]:
    expected = {(symbol, cost) for symbol in fixed["symbols"] for cost in fixed["cost_scenarios_bps"]}
    raw = summary.get("outcomes")
    if not isinstance(raw, list):
        raise ValueError("Child experiment is missing its full outcomes list.")
    keys = [(row.get("symbol"), row.get("cost_bps")) for row in raw]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Child outcome list has missing, duplicate or unexpected symbol/cost tasks.")
    rows = []
    for original in raw:
        row = deepcopy(original)
        if row.get("status") not in {"completed", "failed"}:
            raise ValueError("Child outcome has a nonterminal task status.")
        if row["status"] == "completed":
            start, end = _date(row.get("start"), "effective.start"), _date(row.get("end"), "effective.end")
            if not window["start"] <= start <= end <= window["end"]:
                raise ValueError("Child effective evaluation range escapes the registered window.")
            for key in ("total_return", "maximum_drawdown"):
                value = row.get("metrics", {}).get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not float("-inf") < value < float("inf"):
                    raise ValueError(f"Child completed outcome lacks finite {key}.")
            for key in ("buy_hold_return", "excess_return_vs_buy_hold"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not float("-inf") < value < float("inf"):
                    raise ValueError(f"Child completed outcome lacks finite {key}.")
            row.update(effective_start=start, effective_end=end,
                       effective_range_differs=(start != window["start"] or end != window["end"]))
        else:
            row.update(effective_start=None, effective_end=None, effective_range_differs=None)
        row.update(window_label=window["label"], requested_start=window["start"], requested_end=window["end"], research_scope=SCOPE)
        rows.append(row)
    return rows


def _failed_rows(window: dict, fixed: dict, error: str) -> list[dict]:
    return [{"window_label": window["label"], "symbol": symbol, "cost_bps": cost,
             "status": "failed", "error": error, "research_scope": SCOPE,
             "requested_start": window["start"], "requested_end": window["end"],
             "effective_start": None, "effective_end": None, "effective_range_differs": None}
            for symbol in fixed["symbols"] for cost in fixed["cost_scenarios_bps"]]


def _by_cost(window: dict, fixed: dict, rows: list[dict]) -> list[dict]:
    result = []
    for cost in fixed["cost_scenarios_bps"]:
        group = [row for row in rows if row["cost_bps"] == cost]
        valid = [row for row in group if row["status"] == "completed"]
        returns = [row["metrics"]["total_return"] for row in valid]
        excess = [row["excess_return_vs_buy_hold"] for row in valid]
        result.append({"window_label": window["label"], "requested_start": window["start"], "requested_end": window["end"],
                       "cost_bps": cost, "requested_symbols": len(group), "completed_symbols": len(valid),
                       "failed_symbols": len(group) - len(valid), "mean_return": mean(returns) if valid else None,
                       "median_return": median(returns) if valid else None,
                       "mean_buy_hold_return": mean(r["buy_hold_return"] for r in valid) if valid else None,
                       "mean_excess_return": mean(excess) if valid else None,
                       "beat_cash_ratio": mean(value > 0 for value in returns) if valid else None,
                       "beat_buy_hold_ratio": mean(value > 0 for value in excess) if valid else None,
                       "worst_maximum_drawdown": max(r["metrics"]["maximum_drawdown"] for r in valid) if valid else None,
                       "effective_ranges_aligned": len({(r["effective_start"], r["effective_end"]) for r in valid}) <= 1 if valid else None,
                       "effective_ranges": [{"symbol": r["symbol"], "start": r["effective_start"], "end": r["effective_end"],
                                             "differs_from_request": r["effective_range_differs"]} for r in valid],
                       "aggregation": "independent_symbol_distribution_not_continuous_portfolio"})
    return result


def run_triple_validation(config: dict, windows: list[dict], output_root: str | Path, *,
                          progress_callback: Callable[[dict], None] | None = None,
                          cancel_check: Callable[[], bool] | None = None) -> dict:
    """Report parent-level task counts and propagate cancellation to children.

    Cancelled windows retain their child path and completed outcome records in
    partial_result.json; they are not converted into failed strategy attempts.
    """
    protocol = normalize_validation_protocol(config, windows)
    fixed = protocol["fixed_configuration"]
    implementation = _implementation_identity()
    protocol.update(registered_at=datetime.now(timezone.utc).isoformat(),
                    implementation_sha256=implementation, orchestration_sha256=_sha256(Path(__file__)))
    run_id = f"validation_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run = Path(output_root).resolve() / run_id
    run.mkdir(parents=True, exist_ok=False)
    # Registration is complete before the first child can inspect prices or returns.
    _write_json(run / "protocol.json", protocol)
    protocol_hash = _sha256(run / "protocol.json")
    (run / "protocol.sha256").write_text(protocol_hash + "\n", encoding="ascii")
    window_results, all_rows = [], []
    expected_tasks = len(protocol["windows"]) * len(fixed["symbols"]) * len(fixed["cost_scenarios_bps"])
    active_child = {}
    active_rows = []
    _write_json(run / "status.json", {"status": "running", "completed_windows": 0, "total_windows": len(protocol["windows"])})

    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Annual validation cancelled; completed windows and child tasks are preserved.")

    try:
        if progress_callback:
            progress_callback({"event": "run_started", "run_id": run_id, "path": str(run),
                               "completed": 0, "total": expected_tasks, "message": "Registered annual validation"})
        check_cancelled()
        for index, window in enumerate(protocol["windows"]):
            check_cancelled()
            active_child = {"window_index": index, "window_label": window["label"]}
            active_rows = []
            record = {**window, "window_index": index, "run_id": None, "path": None, "status": "running"}

            def child_progress(event):
                if event.get("run_id"):
                    active_child["run_id"] = event["run_id"]
                if event.get("path"):
                    active_child["path"] = event["path"]
                if event.get("event") == "task_completed":
                    active_rows.append(deepcopy(event["outcome"]))
                completed = len(all_rows) + len(active_rows)
                _write_json(run / "status.json", {"status": "running", "completed_windows": len(window_results),
                                                   "total_windows": len(protocol["windows"]),
                                                   "recorded_tasks": completed, "total_tasks": expected_tasks,
                                                   "active_child": active_child})
                if progress_callback:
                    progress_callback({"event": "task_completed" if event.get("event") == "task_completed" else "child_started",
                                       "run_id": run_id, "path": str(run), "completed": completed, "total": expected_tasks,
                                       "window_label": window["label"], "child_run_id": active_child.get("run_id"),
                                       "child_path": active_child.get("path"),
                                       "message": f"{window['label']}: {event.get('message', 'Running')}"})

            try:
                if _sha256(run / "protocol.json") != protocol_hash:
                    raise ValueError("Registered validation protocol changed during execution.")
                if _implementation_identity() != implementation:
                    raise ValueError("Research implementation changed after protocol registration.")
                child_config = {**deepcopy(fixed), "research_scope": "exploratory", "start": window["start"], "end": window["end"]}
                options = {}
                if progress_callback is not None or cancel_check is not None:
                    options["progress_callback"] = child_progress
                if cancel_check is not None:
                    options["cancel_check"] = cancel_check
                output = run_triple_experiment(child_config, run / "windows" / f"window_{index:03d}", **options)
                record.update(run_id=output["run_id"], path=output["path"])
                if _implementation_identity() != implementation or _sha256(run / "protocol.json") != protocol_hash:
                    raise ValueError("Implementation or protocol changed while a validation window was running.")
                rows = _window_rows(output["summary"], window, fixed)
                success = sum(row["status"] == "completed" for row in rows)
                record["status"] = "completed" if success == len(rows) else "partial" if success else "failed"
            except InterruptedError:
                raise
            except Exception as exc:
                record.update(status="failed", error=str(exc))
                rows = _failed_rows(window, fixed, str(exc))
            record["by_cost"] = _by_cost(window, fixed, rows)
            record["outcomes"] = rows
            all_rows.extend(rows)
            window_results.append(record)
            active_child, active_rows = {}, []
            _write_json(run / "window_results" / f"window_{index:03d}.json", record)
            _write_json(run / "status.json", {"status": "running", "completed_windows": len(window_results),
                                               "total_windows": len(protocol["windows"]), "recorded_tasks": len(all_rows)})
            if progress_callback:
                progress_callback({"event": "window_completed", "run_id": run_id, "path": str(run),
                                   "completed": len(all_rows), "total": expected_tasks,
                                   "completed_windows": len(window_results), "window": record,
                                   "message": f"Window {window['label']}: {record['status']}"})
            if len(window_results) < len(protocol["windows"]):
                check_cancelled()
        successes = sum(row["status"] == "completed" for row in all_rows)
        if len(all_rows) != expected_tasks:
            raise ValueError("Validation accounting does not reconcile with the registered task count.")
        if _sha256(run / "protocol.json") != protocol_hash:
            raise ValueError("Registered protocol integrity was lost; results cannot be certified against it.")
        status = "completed" if successes == expected_tasks else "partial" if successes else "failed"
        summary = {"run_id": run_id, "status": status, "research_scope": SCOPE, "price_basis": "pre",
                   "protocol_sha256": protocol_hash, "selection_policy": protocol["selection_policy"],
                   "aggregation_policy": protocol["aggregation_policy"], "coverage_claim": protocol["coverage_claim"],
                   "requested_windows": len(protocol["windows"]), "requested_symbols": len(fixed["symbols"]),
                   "total_tasks": expected_tasks, "completed_tasks": successes, "failed_tasks": expected_tasks - successes,
                   "windows": window_results, "by_cost_window": [row for record in window_results for row in record["by_cost"]]}
        _write_json(run / "summary.json", summary)
        artifacts = {path.relative_to(run).as_posix(): _sha256(path)
                     for path in [run / "protocol.json", run / "protocol.sha256", run / "summary.json", *sorted((run / "window_results").glob("*.json"))]}
        _write_json(run / "artifacts.json", {"sha256": artifacts})
        _write_json(run / "status.json", {"status": status, "completed_windows": len(window_results),
                                           "total_windows": len(protocol["windows"]), "successful_tasks": successes,
                                           "failed_tasks": expected_tasks - successes, "total_tasks": expected_tasks})
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, InterruptedError) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        known_rows = all_rows + active_rows
        successes = sum(row["status"] == "completed" for row in known_rows)
        partial = {"run_id": run_id, "status": status, "research_scope": SCOPE, "price_basis": "pre",
                   "protocol_sha256": protocol_hash, "total_tasks": expected_tasks,
                   "recorded_tasks": len(known_rows), "completed_tasks": successes,
                   "failed_tasks": len(known_rows)-successes, "unfinished_tasks": expected_tasks-len(known_rows),
                   "windows": window_results, "active_child": active_child, "active_outcomes": active_rows,
                   "by_cost_window": [row for item in window_results for row in item["by_cost"]]}
        _write_json(run / "partial_result.json", partial)
        paths = [run / "protocol.json", run / "protocol.sha256", run / "partial_result.json",
                 *sorted((run / "window_results").glob("*.json"))]
        _write_json(run / "artifacts.json", {"sha256": {path.relative_to(run).as_posix(): _sha256(path) for path in paths}})
        _write_json(run / "status.json", {"status": status,
                                           "error": str(exc), "completed_windows": len(window_results),
                                           "total_windows": len(protocol["windows"]), "recorded_tasks": len(known_rows),
                                           "total_tasks": expected_tasks, "active_child": active_child})
        raise
    return {"run_id": run_id, "status": status, "path": str(run), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="本地前复权三重滤网：预登记、不重叠窗口、固定参数验证。")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path, help="JSON array with label/start/end")
    parser.add_argument("--output", type=Path, default=Path("runtime_layer/reports/triple_validation"))
    args = parser.parse_args()
    config_path, window_path = Path(_local_path(args.config, file_only=True)), Path(_local_path(args.windows, file_only=True))
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    windows = json.loads(window_path.read_text(encoding="utf-8-sig"))
    output = run_triple_validation(config, windows, args.output)
    print(json.dumps({"run_id": output["run_id"], "status": output["status"], "path": output["path"],
                      "research_scope": SCOPE, "by_cost_window": output["summary"]["by_cost_window"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
