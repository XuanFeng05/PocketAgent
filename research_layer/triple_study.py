"""Preregistered local sensitivity/ablation cases with matched-stock diagnostics.

Cases are experiments, not a leaderboard. Each has its own independent
normalized accounts; only equal periods and frozen inputs enter paired deltas.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from typing import Callable
from uuid import uuid4

from research_layer.triple_runner import normalize_triple_config, run_triple_experiment, _implementation_identity
from strategy_layer.triple_screen import resolve_triple_settings


SCOPE = "exploratory_study"
VARIANTS = {"full", "drop_weekly", "drop_pullback", "drop_trigger"}
OVERRIDES = {"macd_fast", "macd_slow", "macd_signal", "force_span"}
CONFIG_KEYS = {"schema_version", "mode", "source_type", "source_path", "calendar_path", "symbols",
               "settings", "adjust", "cost_scenarios_bps", "research_scope", "notes", "start", "end", "max_tasks"}


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _local(value, *, exists=True, file_only=False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("An explicit local filesystem path is required.")
    raw = str(value)
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise ValueError("Study input/output must be local; URLs and network shares are forbidden.")
    path = Path(value).expanduser().resolve()
    if str(path).startswith(("\\\\", "//")):
        raise ValueError("Resolved network shares are forbidden.")
    if exists and (not path.exists() or (file_only and not path.is_file())):
        raise ValueError(f"Local input is missing or has an invalid type: {path}")
    return path


def _date(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Study dates must be explicit YYYY-MM-DD dates.")
    return date.fromisoformat(value).isoformat()


def normalize_study_protocol(config: dict, cases: list[dict]) -> dict:
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise ValueError("Study configuration must be an object with known fields only.")
    if config.get("research_scope", SCOPE) != SCOPE:
        raise ValueError("Study scope must be exploratory_study, never holdout or winner selection.")
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported study schema.")
    if not isinstance(config.get("symbols"), list) or not config["symbols"]:
        raise ValueError("Provide an explicit symbols list.")
    start, end = _date(config.get("start")), _date(config.get("end"))
    if start > end:
        raise ValueError("Study start must not exceed end.")
    source = config.get("source_type", "market")
    if source not in {"market", "csv"}:
        raise ValueError("Only existing local market/CSV sources are supported.")
    local = deepcopy(config)
    budget = local.pop("max_tasks", 2000)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 100000:
        raise ValueError("max_tasks must be an integer from 1 to 100000.")
    local["source_path"] = str(_local(local.get("source_path"), file_only=source == "csv"))
    if local.get("calendar_path"):
        local["calendar_path"] = str(_local(local["calendar_path"], file_only=True))
    fixed = normalize_triple_config({**local, "research_scope": "exploratory", "start": start, "end": end})
    fixed["research_scope"] = SCOPE
    if not isinstance(cases, list) or not 1 <= len(cases) <= 16:
        raise ValueError("Provide 1 to 16 explicit study cases.")
    normalized, ids = [], set()
    for item in cases:
        if not isinstance(item, dict) or set(item) - {"id", "label", "settings", "variant"}:
            raise ValueError("Cases accept only id, label, settings and variant.")
        case_id, label = item.get("id"), item.get("label")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", case_id) or case_id in ids:
            raise ValueError("Case IDs must be unique safe identifiers.")
        if not isinstance(label, str) or not label.strip() or len(label) > 120 or any(ord(c) < 32 for c in label):
            raise ValueError("Each case needs a nonempty display label without control characters.")
        override, variant = item.get("settings", {}), item.get("variant", "full")
        if not isinstance(override, dict) or set(override) - OVERRIDES:
            raise ValueError("Cases may override only MACD periods and force_span; version/warmup/cost stay fixed.")
        if not isinstance(variant, str) or variant not in VARIANTS:
            raise ValueError("Unknown study filter variant.")
        if case_id == "primary" and (override or variant != "full"):
            raise ValueError("The primary case requires empty settings overrides and variant=full.")
        settings = resolve_triple_settings({**fixed["settings"], **override}).to_dict()
        normalized.append({"id": case_id, "label": label.strip(), "settings": deepcopy(override),
                           "resolved_settings": settings, "variant": variant})
        ids.add(case_id)
    if "primary" not in ids:
        raise ValueError("A primary case with empty overrides and variant=full is required.")
    normalized.sort(key=lambda item: item["id"] != "primary")
    tasks = len(normalized) * len(fixed["symbols"]) * len(fixed["cost_scenarios_bps"])
    if tasks > budget:
        raise ValueError(f"Study needs {tasks} tasks, over max_tasks={budget}; no cases are truncated.")
    return {"schema_version": 1, "research_scope": SCOPE, "fixed_configuration": fixed,
            "cases": normalized, "total_tasks": tasks, "max_tasks": budget,
            "selection_policy": "no_winner_selection_no_ranking", "primary_case_id": "primary",
            "pairing_policy": "same_symbol_cost_effective_dates_input_sha256_calendar_sha256",
            "account_policy": "independent_initial_nav_1_per_symbol_per_case",
            "source_policy": "local_files_only_no_downloads", "coverage_claim": "requested_list_not_full_A_share_certification"}


def _failure_rows(fixed, error):
    return [{"symbol": symbol, "cost_bps": cost, "status": "failed", "error": error}
            for symbol in fixed["symbols"] for cost in fixed["cost_scenarios_bps"]]


def _finite(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _child_rows(summary, fixed):
    rows = summary.get("outcomes")
    expected = {(s, c) for s in fixed["symbols"] for c in fixed["cost_scenarios_bps"]}
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Child outcome list is missing or malformed.")
    keys = [(r.get("symbol"), r.get("cost_bps")) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Child task identities do not reconcile with the registered symbol/cost set.")
    compact = []
    for raw in rows:
        status = raw.get("status")
        if status not in {"completed", "failed"}:
            raise ValueError("Child outcomes must have terminal statuses.")
        row = {key: deepcopy(raw[key]) for key in ("symbol", "cost_bps", "status", "error", "start", "end", "metrics",
                                                  "data_sha256", "calendar_sha256", "buy_hold_return", "excess_return_vs_buy_hold", "artifact_path") if key in raw}
        if status == "completed":
            start, end = _date(raw.get("start")), _date(raw.get("end"))
            if not fixed["start"] <= start <= end <= fixed["end"]:
                raise ValueError("Child effective dates escape the registered research interval.")
            for metric in ("total_return", "maximum_drawdown"):
                if not _finite(raw.get("metrics", {}).get(metric)):
                    raise ValueError(f"Child outcome lacks finite {metric}.")
            for metric in ("buy_hold_return", "excess_return_vs_buy_hold"):
                if not _finite(raw.get(metric)):
                    raise ValueError(f"Child outcome lacks finite {metric}.")
            for identity in ("data_sha256", "calendar_sha256"):
                if not isinstance(raw.get(identity), str) or not re.fullmatch(r"[0-9a-f]{64}", raw[identity]):
                    raise ValueError(f"Child outcome lacks a valid {identity}.")
        compact.append(row)
    return compact


def _paired(case_rows: list[dict], primary_rows: list[dict], fixed: dict) -> list[dict]:
    primary = {(r["symbol"], r["cost_bps"]): r for r in primary_rows}
    output = []
    for cost in fixed["cost_scenarios_bps"]:
        rows = [r for r in case_rows if r["cost_bps"] == cost]
        pairs, exclusions = [], []
        for row in rows:
            ref = primary.get((row["symbol"], cost))
            reason = None
            if ref is None or ref["status"] != "completed": reason = "primary_failed_or_missing"
            elif row["status"] != "completed": reason = "case_failed"
            elif (row["start"], row["end"]) != (ref["start"], ref["end"]): reason = "effective_period_mismatch"
            elif row["data_sha256"] != ref["data_sha256"]: reason = "input_identity_mismatch"
            elif row["calendar_sha256"] != ref["calendar_sha256"]: reason = "calendar_identity_mismatch"
            if reason:
                exclusions.append({"symbol": row["symbol"], "reason": reason})
                continue
            pairs.append({"symbol": row["symbol"], "start": row["start"], "end": row["end"],
                          "case_return": row["metrics"]["total_return"], "primary_return": ref["metrics"]["total_return"],
                          "return_delta": row["metrics"]["total_return"] - ref["metrics"]["total_return"],
                          "drawdown_delta": row["metrics"]["maximum_drawdown"] - ref["metrics"]["maximum_drawdown"]})
        deltas = [pair["return_delta"] for pair in pairs]
        same_period = len({(pair["start"], pair["end"]) for pair in pairs}) <= 1 if pairs else None
        # Heterogeneous horizons are not pooled; return one distribution per period.
        groups = []
        periods = sorted({(p["start"], p["end"]) for p in pairs})
        for start, end in periods:
            subset = [p for p in pairs if (p["start"], p["end"]) == (start, end)]
            values = [p["return_delta"] for p in subset]
            groups.append({"start": start, "end": end, "paired_symbols": len(subset),
                           "mean_return_delta": mean(values), "median_return_delta": median(values),
                           "positive_delta_ratio": mean(v > 0 for v in values),
                           "mean_drawdown_delta": mean(p["drawdown_delta"] for p in subset)})
        output.append({"cost_bps": cost, "requested_symbols": len(rows), "paired_symbols": len(pairs),
                       "excluded_symbols": len(exclusions), "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
                       "periods_aligned": same_period,
                       "mean_return_delta": mean(deltas) if pairs and same_period else None,
                       "median_return_delta": median(deltas) if pairs and same_period else None,
                       "positive_delta_ratio": mean(v > 0 for v in deltas) if pairs and same_period else None,
                       "by_effective_period": groups, "pairs": pairs, "exclusions": exclusions,
                       "aggregation": "matched_independent_stock_differences_not_portfolio_returns"})
    return output


def _distributions(rows: list[dict], fixed: dict) -> list[dict]:
    result = []
    for cost in fixed["cost_scenarios_bps"]:
        group = [r for r in rows if r["cost_bps"] == cost]
        valid = [r for r in group if r["status"] == "completed"]
        periods = sorted({(r["start"], r["end"]) for r in valid})
        by_period = []
        for start, end in periods:
            subset = [r for r in valid if (r["start"], r["end"]) == (start, end)]
            values = [r["metrics"]["total_return"] for r in subset]
            by_period.append({"start": start, "end": end, "symbol_count": len(subset), "mean_return": mean(values),
                              "median_return": median(values), "beat_cash_ratio": mean(v > 0 for v in values),
                              "mean_buy_hold_return": mean(r["buy_hold_return"] for r in subset),
                              "beat_buy_hold_ratio": mean(r["excess_return_vs_buy_hold"] > 0 for r in subset),
                              "worst_maximum_drawdown": max(r["metrics"]["maximum_drawdown"] for r in subset)})
        result.append({"cost_bps": cost, "requested_symbols": len(group), "completed_symbols": len(valid),
                       "failed_symbols": len(group) - len(valid), "periods_aligned": len(periods) <= 1 if valid else None,
                       "by_effective_period": by_period})
    return result


def _verified_child(output, case_root: Path, request: dict, implementation: str):
    child = Path(output["path"]).resolve()
    if not child.is_relative_to(case_root.resolve()):
        raise ValueError("Child output escapes the case directory.")
    manifest_path = child / "artifacts.json"
    manifest = _read(manifest_path)
    index = manifest.get("sha256", {})
    for name in ("config.json", "summary.json"):
        if name not in index or _sha(child / name) != index[name]:
            raise ValueError(f"Unverified child artifact: {name}")
    snapshot, summary = _read(child / "config.json"), _read(child / "summary.json")
    if snapshot.get("configuration") != normalize_triple_config(request) or snapshot.get("implementation_sha256") != implementation:
        raise ValueError("Child configuration or implementation differs from the registered case.")
    if summary != output.get("summary") or summary.get("price_basis") != "pre" or summary.get("research_scope") != "exploratory":
        raise ValueError("Child summary differs from its saved exploratory result.")
    return summary, manifest_path, _sha(manifest_path)


def run_triple_study(config: dict, cases: list[dict], output_root: str | Path, *,
                     progress_callback: Callable[[dict], None] | None = None) -> dict:
    protocol = normalize_study_protocol(config, cases)
    fixed = protocol["fixed_configuration"]
    implementation = _implementation_identity()
    protocol.update(registered_at=datetime.now(timezone.utc).isoformat(), implementation_sha256=implementation,
                    orchestration_sha256=_sha(Path(__file__)))
    root = _local(output_root, exists=False)
    run_id = f"study_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run = root / run_id
    run.mkdir(parents=True, exist_ok=False)
    _json(run / "protocol.json", protocol)
    protocol_hash = _sha(run / "protocol.json")
    (run / "protocol.sha256").write_text(protocol_hash + "\n", encoding="ascii")
    _json(run / "status.json", {"status": "running", "completed_cases": 0, "total_cases": len(protocol["cases"])})
    records, child_manifests, primary_rows = [], {}, []

    def check_identity():
        if _sha(run / "protocol.json") != protocol_hash or _implementation_identity() != implementation or _sha(Path(__file__)) != protocol["orchestration_sha256"]:
            raise ValueError("Study protocol or implementation changed after registration.")

    try:
        for case in protocol["cases"]:
            record = {"id": case["id"], "label": case["label"], "variant": case["variant"],
                      "settings": case["resolved_settings"], "status": "running", "child_run_id": None, "child_path": None}
            try:
                check_identity()
                request = {**deepcopy(fixed), "research_scope": "exploratory", "settings": case["resolved_settings"], "variant": case["variant"]}
                case_root = run / "cases" / case["id"]
                output = run_triple_experiment(request, case_root)
                record.update(child_run_id=output["run_id"], child_path=output["path"])
                check_identity()
                summary, manifest, digest = _verified_child(output, case_root, request, implementation)
                rows = _child_rows(summary, fixed)
                child_manifests[manifest.relative_to(run).as_posix()] = digest
                successes = sum(row["status"] == "completed" for row in rows)
                record["status"] = "completed" if successes == len(rows) else "partial" if successes else "failed"
            except Exception as exc:
                record.update(status="failed", error=str(exc))
                rows = _failure_rows(fixed, str(exc))
            if case["id"] == "primary":
                primary_rows = rows
            record.update(outcomes=rows, by_cost=_distributions(rows, fixed), paired_vs_primary=_paired(rows, primary_rows, fixed))
            _json(run / "case_results" / f"{case['id']}.json", record)
            records.append(record)
            _json(run / "status.json", {"status": "running", "completed_cases": len(records), "total_cases": len(protocol["cases"])})
            if progress_callback:
                progress_callback({"event": "case_completed", "completed": len(records), "total": len(protocol["cases"]), "case": record})
        check_identity()
        total = sum(len(case["outcomes"]) for case in records)
        successes = sum(row["status"] == "completed" for case in records for row in case["outcomes"])
        if total != protocol["total_tasks"]:
            raise ValueError("Study task counts do not reconcile with the registered budget.")
        status = "completed" if successes == total else "partial" if successes else "failed"
        summary = {"run_id": run_id, "status": status, "research_scope": SCOPE, "price_basis": "pre",
                   "protocol_sha256": protocol_hash, "selection_policy": protocol["selection_policy"],
                   "coverage_claim": protocol["coverage_claim"], "total_tasks": total, "completed_tasks": successes,
                   "failed_tasks": total-successes, "cases": [{key: value for key, value in case.items() if key != "outcomes"} for case in records]}
        _json(run / "summary.json", summary)
        files = [run / "protocol.json", run / "protocol.sha256", run / "summary.json", *sorted((run / "case_results").glob("*.json"))]
        _json(run / "artifacts.json", {"sha256": {**{p.relative_to(run).as_posix(): _sha(p) for p in files}, **child_manifests},
                                       "child_manifests": child_manifests,
                                       "verification_note": "Child manifest identities are pinned here; verify their referenced artifacts recursively for full acceptance."})
        _json(run / "status.json", {"status": status, "completed_cases": len(records), "total_cases": len(protocol["cases"]),
                                    "successful_tasks": successes, "failed_tasks": total-successes, "total_tasks": total})
    except BaseException as exc:
        _json(run / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                    "error": str(exc), "completed_cases": len(records), "total_cases": len(protocol["cases"])})
        raise
    return {"run_id": run_id, "status": status, "path": str(run), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="本地三重滤网：预登记参数邻域与滤网移除对照，不选赢家。")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runtime_layer/reports/triple_study"))
    args = parser.parse_args()
    config = _read(_local(args.config, file_only=True))
    cases = _read(_local(args.cases, file_only=True))
    def progress(event):
        case = event["case"]
        print(json.dumps({"event": event["event"], "completed": event["completed"], "total": event["total"],
                          "case_id": case["id"], "status": case["status"]}, ensure_ascii=False), flush=True)
    output = run_triple_study(config, cases, args.output, progress_callback=progress)
    print(json.dumps({"run_id": output["run_id"], "status": output["status"], "path": output["path"],
                      "total_tasks": output["summary"]["total_tasks"], "failed_tasks": output["summary"]["failed_tasks"]}, ensure_ascii=False))
    if output["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
