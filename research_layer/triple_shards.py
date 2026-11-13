"""Local, bounded universe shards with frozen identity and verified recovery."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import psutil

from research_layer.triple_runner import normalize_triple_config, run_triple_experiment, _implementation_identity


CONFIG_KEYS = {"schema_version", "mode", "source_type", "source_path", "calendar_path", "symbols", "settings",
               "adjust", "cost_scenarios_bps", "research_scope", "notes", "start", "end", "variant"}
MAX_CSV_BYTES = 64 * 1024 * 1024


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _local(value, *, exists=True):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Explicit local paths are required.")
    raw = str(value)
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise ValueError("Network paths and URLs are not allowed.")
    path = Path(raw).expanduser().resolve()
    if str(path).startswith(("\\\\", "//")) or (exists and not path.exists()):
        raise ValueError("Local path is missing or resolves to a network share.")
    return path


def _inside(root: Path, relative: str):
    if not isinstance(relative, str):
        raise ValueError("Artifact paths must be strings.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes the owning run.")
    return path


def _verify_manifest(root: Path):
    manifest = _read(root / "artifacts.json")
    index = manifest.get("sha256")
    if not isinstance(index, dict):
        raise ValueError("Artifact hash index is missing.")
    for relative, digest in index.items():
        path = _inside(root, relative)
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not path.is_file() or _sha(path) != digest):
            raise ValueError(f"Artifact integrity failure: {relative}")
    return manifest


def make_shard_protocol(config: dict, shard_size: int = 100):
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise ValueError("Shard configuration must contain only known fields.")
    if not isinstance(config.get("symbols"), list) or not config["symbols"]:
        raise ValueError("Provide an explicit nonempty symbols list.")
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or not 1 <= shard_size <= 500:
        raise ValueError("shard_size must be an integer from 1 to 500.")
    local = deepcopy(config)
    source_path = _local(local.get("source_path"))
    if local.get("source_type", "market") == "csv":
        if not source_path.is_file() or source_path.stat().st_size > MAX_CSV_BYTES:
            raise ValueError("CSV sharding accepts files up to 64 MiB; use per-symbol market storage for larger universes.")
    local["source_path"] = str(source_path)
    if local.get("calendar_path"):
        local["calendar_path"] = str(_local(local["calendar_path"]))
    normalized = normalize_triple_config(local)
    symbols = normalized["symbols"]
    costs = normalized["cost_scenarios_bps"]
    shards = [{"id": f"shard_{index // shard_size:05d}", "start_index": index,
               "end_index_exclusive": min(index + shard_size, len(symbols)),
               "symbols": symbols[index:index + shard_size], "tasks": len(symbols[index:index + shard_size]) * len(costs)}
              for index in range(0, len(symbols), shard_size)]
    return {"schema_version": 1, "configuration": normalized, "shard_size": shard_size, "shards": shards,
            "total_symbols": len(symbols), "total_tasks": len(symbols) * len(costs),
            "symbol_order": "explicit_input_order", "research_scope": "exploratory",
            "csv_memory_guard_bytes": MAX_CSV_BYTES,
            "bounded_memory_verified": False,
            "memory_policy": "market_reads_one_symbol_per_child_task; CSV_has_64_MiB_input_guard_but_still_uses_existing_whole_file_reader; peak_memory_not_profiled",
            "source_policy": "local_only_no_downloads", "coverage_claim": "requested_symbols_not_certified_all_A_shares"}


def _request(protocol, shard):
    return {**deepcopy(protocol["configuration"]), "symbols": shard["symbols"]}


def _discover_child(shard_root: Path):
    if not shard_root.exists():
        return None
    paths = [p.resolve() for p in shard_root.iterdir() if p.is_dir() and (p / "config.json").exists()]
    if len(paths) > 1:
        raise ValueError("Multiple child runs in one shard; recovery is ambiguous.")
    if paths and not paths[0].is_relative_to(shard_root.resolve()):
        raise ValueError("Child output escapes the shard directory.")
    return paths[0] if paths else None


def _child_record(child: Path, shard_root: Path, request: dict, implementation: str):
    child = child.resolve()
    if not child.is_relative_to(shard_root.resolve()):
        raise ValueError("Child output escapes the shard directory.")
    manifest = _verify_manifest(child)
    index = manifest["sha256"]
    if "config.json" not in index:
        raise ValueError("Child configuration is not hash-verified.")
    saved = _read(child / "config.json")
    if saved.get("configuration") != normalize_triple_config(request) or saved.get("implementation_sha256") != implementation:
        raise ValueError("Child configuration/implementation identity mismatch.")
    events = manifest.get("outcome_files")
    if not isinstance(events, list) or len(set(events)) != len(events):
        raise ValueError("Child immutable outcome events are missing or duplicated.")
    expected = [(s, c) for s in request["symbols"] for c in request["cost_scenarios_bps"]]
    observed = {}
    for relative in events:
        if relative not in index:
            raise ValueError("Child event lacks a verified hash.")
        event = _read(_inside(child, relative))
        key = (event.get("symbol"), event.get("cost_bps"))
        if key not in expected or key in observed or event.get("status") not in {"completed", "failed"}:
            raise ValueError("Child events do not match the shard plan.")
        if event["status"] == "completed" and event.get("artifact_path") not in index:
            raise ValueError("Completed child result is not hash-verified.")
        observed[key] = {k: deepcopy(v) for k, v in event.items() if k in {
            "task_id", "symbol", "cost_bps", "status", "reused", "error", "start", "end",
            "artifact_path", "data_sha256", "calendar_sha256", "metrics", "buy_hold_return", "excess_return_vs_buy_hold"}}
    rows = [observed.get(key, {"symbol": key[0], "cost_bps": key[1], "status": "unknown",
                               "reason": "no_verified_outcome_event"}) for key in expected]
    if "summary.json" in index:
        summary = _read(child / "summary.json")
        if summary.get("outcomes") != [_read(_inside(child, event)) for event in events]:
            raise ValueError("Child terminal summary differs from verified outcome events.")
        if len(events) != len(expected):
            raise ValueError("Terminal child summary is missing planned tasks.")
    return rows, _sha(child / "artifacts.json")


def _counts(rows):
    return {state: sum(row["status"] == state for row in rows) for state in ("completed", "failed", "unknown")}


def _status(counts):
    if counts["unknown"]:
        return "incomplete"
    return "completed" if not counts["failed"] else "partial" if counts["completed"] else "failed"


def _assert_source_stopped(root: Path):
    status = _read(root / "status.json")
    if status.get("status") != "running":
        return
    owner = _read(root / "owner.json")
    try:
        process = psutil.Process(owner["pid"])
        if process.create_time() == owner["process_created_at"] and process.is_running():
            raise ValueError("Source shard coordinator is still running; do not start a competing recovery.")
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return


def run_triple_shards(config: dict, output_root: str | Path, *, shard_size=100, resume_from=None) -> dict:
    plan = make_shard_protocol(config, shard_size)
    implementation, orchestration = _implementation_identity(), _sha(Path(__file__))
    plan.update(implementation_sha256=implementation, orchestration_sha256=orchestration)
    prior = _local(resume_from) if resume_from else None
    prior_children = {}
    if prior:
        _assert_source_stopped(prior)
        index = _verify_manifest(prior)["sha256"]
        if "protocol.json" not in index or _read(prior / "protocol.json") != plan:
            raise ValueError("Resume protocol/configuration/implementation identity mismatch.")
        # Each discovered child is checked recursively before any new run starts.
        for shard in plan["shards"]:
            child = _discover_child(prior / "shards" / shard["id"])
            if child:
                _child_record(child, prior / "shards" / shard["id"], _request(plan, shard), implementation)
                prior_children[shard["id"]] = child
    root = _local(output_root, exists=False)
    run_id = f"shards_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run = root / run_id
    run.mkdir(parents=True, exist_ok=False)
    _json(run / "protocol.json", plan)
    (run / "protocol.sha256").write_text(_sha(run / "protocol.json") + "\n", encoding="ascii")
    _json(run / "owner.json", {"pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
                                "started_at": datetime.now(timezone.utc).isoformat(), "resumed_from": str(prior) if prior else None})
    artifacts = {name: _sha(run / name) for name in ("protocol.json", "protocol.sha256", "owner.json")}
    records = []

    def publish(status="running"):
        seen = {record["id"]: record for record in records}
        tally = {"completed": 0, "failed": 0, "unknown": 0}
        for shard in plan["shards"]:
            count = seen[shard["id"]]["counts"] if shard["id"] in seen else {"completed": 0, "failed": 0, "unknown": shard["tasks"]}
            for state in tally:
                tally[state] += count[state]
        if sum(tally.values()) != plan["total_tasks"]:
            raise ValueError("Shard task accounting does not reconcile.")
        _json(run / "artifacts.json", {"sha256": artifacts, "shard_records": [f"shard_results/{r['id']}.json" for r in records]})
        _json(run / "status.json", {"status": status, "recorded_shards": len(records), "total_shards": len(plan["shards"]),
                                   "total_tasks": plan["total_tasks"], **tally})
        return tally

    def store_record(record):
        path = run / "shard_results" / f"{record['id']}.json"
        _json(path, record)
        artifacts[path.relative_to(run).as_posix()] = _sha(path)
        if record.get("child_manifest"):
            artifacts[record["child_manifest"]] = record["child_manifest_sha256"]
        records.append(record)
        publish()

    def check_identity():
        if _sha(run / "protocol.json") != artifacts["protocol.json"] or _implementation_identity() != implementation or _sha(Path(__file__)) != orchestration:
            raise ValueError("Protocol or implementation changed during sharded execution.")

    publish()
    try:
        for shard in plan["shards"]:
            check_identity()
            shard_root = run / "shards" / shard["id"]
            request = _request(plan, shard)
            record = {**shard, "child_path": None, "child_manifest": None, "resumed_from": str(prior_children[shard["id"]]) if shard["id"] in prior_children else None}
            interruption = None
            try:
                output = run_triple_experiment(request, shard_root, resume_from=prior_children.get(shard["id"]))
                child = Path(output["path"]).resolve()
                rows, digest = _child_record(child, shard_root, request, implementation)
                record.update(child_path=str(child), child_manifest=(child / "artifacts.json").relative_to(run).as_posix(), child_manifest_sha256=digest)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    interruption = exc
                record["error"] = str(exc)
                rows = [{"symbol": s, "cost_bps": c, "status": "unknown", "reason": "shard_execution_error"}
                        for s in shard["symbols"] for c in request["cost_scenarios_bps"]]
                try:
                    child = _discover_child(shard_root)
                    if child:
                        rows, digest = _child_record(child, shard_root, request, implementation)
                        record.update(child_path=str(child), child_manifest=(child / "artifacts.json").relative_to(run).as_posix(), child_manifest_sha256=digest)
                except Exception as recovery_error:
                    record["recovery_error"] = str(recovery_error)
            record.update(counts=_counts(rows), outcomes=rows)
            record["status"] = _status(record["counts"])
            store_record(record)
            if interruption:
                raise interruption
        check_identity()
        counts = publish()
        status = _status(counts)
        summary = {"run_id": run_id, "status": status, "research_scope": "exploratory", "price_basis": "pre",
                   "research_variant": plan["configuration"]["variant"], "total_symbols": plan["total_symbols"],
                   "total_tasks": plan["total_tasks"], "shard_size": shard_size, "total_shards": len(plan["shards"]),
                   "counts": counts, "shards": records, "coverage_claim": plan["coverage_claim"],
                   "aggregation": "independent_stock_tasks_no_combined_nav", "source_policy": plan["source_policy"]}
        _json(run / "summary.json", summary)
        artifacts["summary.json"] = _sha(run / "summary.json")
        publish(status)
    except BaseException as exc:
        publish("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed")
        raise
    return {"run_id": run_id, "status": status, "path": str(run), "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="本地三重滤网分片运行与校验恢复；不下载数据。")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runtime_layer/reports/triple_shards"))
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.config and not args.resume:
        parser.error("Provide --config or --resume.")
    old = _read(_local(args.resume) / "protocol.json") if args.resume else None
    config = _read(_local(args.config)) if args.config else old["configuration"]
    size = args.shard_size if args.shard_size is not None else old["shard_size"] if old else 100
    result = run_triple_shards(config, args.output, shard_size=size, resume_from=args.resume)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"], "path": result["path"], "counts": result["summary"]["counts"]}, ensure_ascii=False))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
