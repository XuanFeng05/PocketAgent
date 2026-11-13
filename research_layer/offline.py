"""Checksummed, offline task exchange. Bundled source is never executed.

One task is a complete account simulation (or a complete walk-forward job).
Machines need the same installed calculation code and no network connection.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import tempfile
import time
from typing import Any, Callable
from uuid import uuid4
import zipfile

import psutil


SCHEMA_VERSION = 1
MAX_FILES = 20_000
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_TASK_KIND = "pocketagent.task_plan"
_RESULT_KIND = "pocketagent.task_results"


def _json(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    return _loads(path.read_bytes())


def _loads(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("Manifest/result must be a JSON object")
    return value


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(_json(data))
    deadline = time.monotonic() + 2
    try:
        while True:
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"Unsafe package path: {name!r}")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(name).is_absolute():
        raise ValueError(f"Unsafe package path: {name!r}")
    if any(part.casefold() == ".git" for part in parts):
        raise ValueError("Git metadata is not a portable task payload")
    return name


def _safe_id(value: Any, prefix: str) -> str:
    text = str(value)
    if not text.startswith(prefix) or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in text) or len(text) > 100:
        raise ValueError(f"Invalid {prefix} identifier")
    return text


def _fingerprint(manifest: dict, key: str) -> str:
    return _sha(_json({k: v for k, v in manifest.items() if k not in {key, "created_at"}}))


def _environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "duckdb", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}", "packages": versions}


def _current_implementation_hash() -> str:
    from research_layer.runner import _implementation_hash
    return _implementation_hash()


@contextmanager
def _bundle(path: str | Path, expected_kind: str):
    """Validate every entry before yielding an open, non-executable archive."""
    with zipfile.ZipFile(Path(path)) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_FILES:
            raise ValueError("Bundle entry count is outside the supported limit")
        names = set()
        total = 0
        for info in infos:
            name = _safe_relative(info.filename)
            if name.casefold() in names:
                raise ValueError("Duplicate or case-colliding ZIP destination")
            names.add(name.casefold())
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                raise ValueError("Directory, symlink and encrypted ZIP entries are not accepted")
            if info.file_size < 0 or info.file_size > MAX_FILE_BYTES:
                raise ValueError("Bundle member exceeds the size limit")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Bundle expanded size exceeds the CPU exchange budget")
        manifest_info = archive.getinfo("manifest.json")
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ValueError("Bundle manifest is too large")
        manifest = _loads(archive.read(manifest_info))
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != expected_kind:
            raise ValueError("Unsupported package kind or schema")
        id_key, prefix = ("plan_id", "plan_") if expected_kind == _TASK_KIND else ("bundle_id", "results_")
        if manifest.get(id_key) != prefix + _fingerprint(manifest, id_key):
            raise ValueError("Bundle manifest identity mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {i.filename for i in infos if i.filename != "manifest.json"}:
            raise ValueError("Manifest and archive file inventories differ")
        for name, expected in files.items():
            _safe_relative(name)
            info = archive.getinfo(name)
            if expected.get("size") != info.file_size:
                raise ValueError(f"Size mismatch: {name}")
            h = hashlib.sha256()
            with archive.open(info) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected.get("sha256"):
                raise ValueError(f"SHA-256 mismatch: {name}")
        yield archive, manifest


def inspect_task_bundle(path: str | Path) -> dict:
    with _bundle(path, _TASK_KIND) as (_, manifest):
        _validate_plan(manifest)
        return manifest


def _validate_plan(plan: dict) -> None:
    _safe_id(plan.get("plan_id"), "plan_")
    if plan["plan_id"] != "plan_" + _fingerprint(plan, "plan_id"):
        raise ValueError("Task plan fingerprint differs from its contents")
    if plan.get("files", {}).get("input.csv", {}).get("sha256") != plan.get("identity", {}).get("input_sha256"):
        raise ValueError("Input snapshot is not bound to the plan identity")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 256:
        raise ValueError("A task plan must contain 1 to 256 complete tasks")
    ids = set()
    for task in tasks:
        identifier = _safe_id(task.get("task_id"), "task_")
        if identifier in ids:
            raise ValueError("Duplicate task identifier")
        ids.add(identifier)
        if not isinstance(task.get("config"), dict):
            raise ValueError("Task configuration must be an object")
        if task.get("config_sha256") != _sha(_json(task.get("config"))):
            raise ValueError("Task configuration fingerprint mismatch")
        expected = "task_" + _sha(_json({"identity": plan["identity"], "config": task["config"]}))
        if identifier != expected:
            raise ValueError("Task identity mismatch")
    if not isinstance(plan.get("shard_count"), int) or isinstance(plan["shard_count"], bool) or not 1 <= plan["shard_count"] <= 64:
        raise ValueError("shard_count must be an integer from 1 to 64")


def _write_bundle(destination: Path, manifest: dict, members: dict[str, bytes]) -> dict:
    if destination.exists():
        raise FileExistsError(f"Bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data in members.items():
        _safe_relative(name)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("A payload exceeds the exchange size limit")
        files[name] = {"size": len(data), "sha256": _sha(data)}
    if sum(item["size"] for item in files.values()) > MAX_TOTAL_BYTES or len(files) + 1 > MAX_FILES:
        raise ValueError("Task/result bundle exceeds the exchange budget")
    manifest = {**manifest, "files": files}
    key, prefix = ("plan_id", "plan_") if manifest["kind"] == _TASK_KIND else ("bundle_id", "results_")
    manifest[key] = prefix + _fingerprint(manifest, key)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    created_destination = False
    try:
        with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("manifest.json", _json(manifest))
            for name, data in sorted(members.items()):
                archive.writestr(name, data)
        with _bundle(temporary, manifest["kind"]):
            pass
        # Exclusive creation prevents an accidental overwrite even if another
        # process creates the intended destination while validation runs.
        with destination.open("xb") as output:
            created_destination = True
            with temporary.open("rb") as source:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
    except BaseException:
        if created_destination:
            destination.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def prepare_task_bundle(config: dict, destination_zip: str | Path, *, shard_count: int = 1) -> dict:
    """Freeze input once; batch candidates become complete independent tasks."""
    from research_layer.runner import _load_input
    from research_layer.batch import preflight_experiment
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or not 1 <= shard_count <= 64:
        raise ValueError("shard_count must be from 1 to 64")
    bars = _load_input(config)
    preflight = preflight_experiment(config, len(bars))
    input_bytes = bars.to_csv(index=False).encode("utf-8")
    implementation = _current_implementation_hash()
    frozen = {**config, "source_type": "csv", "source_path": "input.csv", "snapshot_ready": True,
              "context": None, "synthetic": str(config.get("source_type") or "demo") == "demo"}
    frozen.pop("context_sources", None)
    frozen.pop("start", None)
    frozen.pop("end", None)
    identity = {"input_sha256": _sha(input_bytes), "implementation_sha256": implementation,
                "data_sha256": _sha(bars.to_json(orient="records", date_format="iso", double_precision=15).encode()),
                "environment": _environment()}
    task_configs = []
    if preflight["mode"] == "batch":
        for candidate in preflight["candidates"]:
            task = {**frozen, "mode": "single", "parameters": candidate["parameters"]}
            task.pop("parameter_grid", None)
            task_configs.append(task)
    else:
        task_configs.append(frozen)
    tasks = []
    for index, task in enumerate(task_configs):
        tasks.append({"task_id": "task_" + _sha(_json({"identity": identity, "config": task})),
                      "config": task, "config_sha256": _sha(_json(task)), "shard_index": index % shard_count})
    # The installed project remains authoritative. Source is included only as
    # a reviewable snapshot; workers never import or execute it from this ZIP.
    root = Path(__file__).resolve().parents[1]
    members = {"input.csv": input_bytes}
    for module in ("backtest_layer", "strategy_layer", "research_layer", "data_layer", "feature_layer", "evaluation_layer"):
        for path in (root / module).rglob("*.py"):
            members["source/" + path.relative_to(root).as_posix()] = path.read_bytes()
    for filename in ("requirements.txt", "requirements-dev.txt"):
        path = root / filename
        if path.exists():
            members["source/" + filename] = path.read_bytes()
    lock_names = ("numpy", "pandas", "duckdb", "pyarrow", "PyYAML", "psutil", "baostock", "python-dateutil", "pytz", "tzdata")
    locked = []
    for name in lock_names:
        try:
            locked.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    members["requirements-cpu-lock.txt"] = ("\n".join(locked) + "\n").encode()
    members["README.txt"] = (
        "Install/review the same PocketAgent code and CPU dependencies before running.\n"
        "No code in this package is automatically executed.\n"
        "Each task is a complete independent account experiment; do not concatenate account segments.\n"
    ).encode()
    if implementation != _current_implementation_hash():
        raise RuntimeError("Calculation source changed while packaging; retry after edits finish")
    manifest = {"schema_version": SCHEMA_VERSION, "kind": _TASK_KIND, "identity": identity,
                "config": frozen, "original_mode": preflight["mode"], "tasks": tasks, "shard_count": shard_count,
                "candidate_count": preflight["candidate_count"], "preflight": preflight,
                "source_execution": "installed_code_only", "created_at": datetime.now(timezone.utc).isoformat()}
    return _write_bundle(Path(destination_zip), manifest, members)


def _bind_plan(workspace: Path, plan: dict) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if workspace.is_symlink() or (hasattr(workspace, "is_junction") and workspace.is_junction()):
        raise ValueError("Workspace must not be a linked directory")
    destination = workspace / "plan.json"
    if destination.exists():
        existing = _read_json(destination)
        if existing.get("plan_id") != plan["plan_id"]:
            raise ValueError("Workspace is already bound to a different plan")
    else:
        _write_json(destination, plan)


def _claim_task(task_dir: Path, task_id: str) -> Path:
    lock = task_dir / "running.lock"
    for _ in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = _read_json(lock)
                process = psutil.Process(int(owner["pid"]))
                alive = abs(process.create_time() - float(owner["process_started"])) < .01
            except psutil.NoSuchProcess:
                alive = False
            except (psutil.AccessDenied, KeyError, ValueError, TypeError):
                raise RuntimeError(f"Cannot verify task lock owner; preserve and inspect {lock}")
            if alive:
                raise RuntimeError(f"Task is already running: {task_id}")
            # Keep the stale owner's record; never terminate an unrelated process.
            lock.replace(task_dir / f"abandoned-lock-{uuid4().hex}.json")
            continue
        os.write(descriptor, _json({"pid": os.getpid(), "process_started": psutil.Process().create_time(), "task_id": task_id}))
        os.fsync(descriptor)
        os.close(descriptor)
        return lock
    raise RuntimeError("Could not acquire task lock")


def _semantic_outcome(result: dict) -> dict:
    # Run ids and a path-sensitive runner config hash identify executions, not
    # their deterministic computation. The task already binds canonical config.
    return {key: value for key, value in result.items() if key not in {"run_id", "config_hash"}}


def _verify_attempt(workspace: Path, plan: dict, task: dict, marker: dict) -> dict:
    if marker.get("plan_id") != plan["plan_id"] or marker.get("task_id") != task["task_id"]:
        raise ValueError("Task completion marker belongs to a different plan/task")
    relative = _safe_relative(marker["outcome_path"])
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace.resolve()):
        raise ValueError("Task outcome escapes workspace")
    raw = path.read_bytes()
    if _sha(raw) != marker.get("outcome_sha256"):
        raise ValueError("Saved task outcome checksum differs; no automatic reuse")
    outcome = _loads(raw)
    if outcome.get("task_id") != task["task_id"] or outcome.get("config_sha256") != task["config_sha256"]:
        raise ValueError("Saved task outcome has wrong configuration identity")
    return outcome


def run_task_bundle(
    bundle_zip: str | Path, workspace: str | Path, *, shard_index: int = 0,
    shard_count: int | None = None, task_ids: list[str] | None = None, retry_failed: bool = False,
    progress_callback: Callable[[dict], None] | None = None, cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Execute selected whole tasks; reuse verified task completions on restart."""
    from research_layer.runner import run_experiment
    with _bundle(bundle_zip, _TASK_KIND) as (archive, plan):
        _validate_plan(plan)
        if _current_implementation_hash() != plan["identity"]["implementation_sha256"]:
            raise ValueError("Installed calculation source differs from this plan; install its reviewed version")
        if _environment() != plan["identity"]["environment"]:
            raise ValueError("CPU dependency/Python versions differ from the task plan")
        if shard_count is not None and shard_count != plan["shard_count"]:
            raise ValueError("shard_count must match the exported plan")
        if isinstance(shard_index, bool) or not isinstance(shard_index, int) or not 0 <= shard_index < plan["shard_count"]:
            raise ValueError("Invalid shard_index")
        if task_ids is not None:
            if not isinstance(task_ids, list) or len(set(task_ids)) != len(task_ids) or not set(task_ids) <= {t["task_id"] for t in plan["tasks"]}:
                raise ValueError("Unknown or repeated task IDs")
            selected = [task for task in plan["tasks"] if task["task_id"] in task_ids]
        else:
            selected = [task for task in plan["tasks"] if task["shard_index"] == shard_index]
        work = Path(workspace).resolve()
        _bind_plan(work, plan)
        input_path = work / "input.csv"
        if input_path.exists():
            if _digest_file(input_path) != plan["identity"]["input_sha256"]:
                raise ValueError("Existing input snapshot was modified")
        else:
            with input_path.open("xb") as output:
                output.write(archive.read("input.csv"))
        outcomes = []
        reused = 0
        for number, task in enumerate(selected, start=1):
            if cancel_check and cancel_check():
                raise InterruptedError("Offline worker cancelled; completed task markers retained")
            task_dir = work / "tasks" / task["task_id"]
            marker_path = task_dir / "completion.json"
            if marker_path.exists():
                previous = _verify_attempt(work, plan, task, _read_json(marker_path))
                if previous["status"] == "completed" or not retry_failed:
                    outcomes.append(previous)
                    reused += 1
                    if progress_callback:
                        progress_callback({"task_id": task["task_id"], "completed": number, "total": len(selected), "reused": True})
                    continue
            task_dir.mkdir(parents=True, exist_ok=True)
            lock = _claim_task(task_dir, task["task_id"])
            attempt_id = uuid4().hex
            attempt = task_dir / "attempts" / attempt_id
            attempt.mkdir(parents=True, exist_ok=False)
            outcome = {"plan_id": plan["plan_id"], "task_id": task["task_id"], "config_sha256": task["config_sha256"],
                       "implementation_sha256": plan["identity"]["implementation_sha256"],
                       "input_sha256": plan["identity"]["input_sha256"], "status": "running", "parameters": task["config"].get("parameters", {})}
            try:
                config = {**task["config"], "source_path": str(input_path), "snapshot_ready": True}
                config.pop("context_sources", None)
                config["context"] = None
                result = run_experiment(config, attempt / "run", cancel_check=cancel_check)
                outcome.update(status="completed", result=_semantic_outcome(result["result"]))
                outcome["result"]["synthetic"] = bool(task["config"].get("synthetic"))
            except InterruptedError:
                _write_json(attempt / "interrupted.json", {**outcome, "status": "interrupted"})
                lock.unlink(missing_ok=True)
                raise
            except Exception as exc:
                outcome.update(status="failed", error=str(exc))
            try:
                raw = _json(outcome)
                path = attempt / "outcome.json"
                path.write_bytes(raw)
                marker = {"plan_id": plan["plan_id"], "task_id": task["task_id"], "status": outcome["status"],
                          "attempt_id": attempt_id, "outcome_path": path.relative_to(work).as_posix(), "outcome_sha256": _sha(raw)}
                _write_json(marker_path, marker)
            finally:
                lock.unlink(missing_ok=True)
            outcomes.append(outcome)
            if progress_callback:
                progress_callback({"task_id": task["task_id"], "completed": number, "total": len(selected), "reused": False})
        return {"plan_id": plan["plan_id"], "selected_tasks": len(selected), "completed_tasks": sum(x["status"] == "completed" for x in outcomes),
                "failed_tasks": sum(x["status"] == "failed" for x in outcomes), "reused_tasks": reused, "workspace": str(work)}


def export_result_bundle(workspace: str | Path, destination_zip: str | Path) -> dict:
    work = Path(workspace).resolve()
    plan = _read_json(work / "plan.json")
    _validate_plan(plan)
    members = {}
    outcomes = []
    for task in plan["tasks"]:
        marker_path = work / "tasks" / task["task_id"] / "completion.json"
        if not marker_path.exists():
            continue
        outcome = _verify_attempt(work, plan, task, _read_json(marker_path))
        filename = f"outcomes/{task['task_id']}.json"
        members[filename] = _json(outcome)
        outcomes.append({"task_id": task["task_id"], "status": outcome["status"], "file": filename})
    manifest = {"schema_version": SCHEMA_VERSION, "kind": _RESULT_KIND, "plan_id": plan["plan_id"],
                "identity": plan["identity"], "outcomes": outcomes, "total_planned_tasks": len(plan["tasks"])}
    return _write_bundle(Path(destination_zip), manifest, members)


def import_result_bundle(result_zip: str | Path, collection_dir: str | Path, expected_plan_zip: str | Path) -> dict:
    plan = inspect_task_bundle(expected_plan_zip)
    tasks = {task["task_id"]: task for task in plan["tasks"]}
    collection = Path(collection_dir).resolve()
    collection.mkdir(parents=True, exist_ok=True)
    lock = _claim_task(collection, "result_collection")
    try:
        with _bundle(result_zip, _RESULT_KIND) as (archive, manifest):
            if manifest.get("plan_id") != plan["plan_id"] or manifest.get("identity") != plan["identity"]:
                raise ValueError("Result bundle belongs to a different source/data/plan identity")
            if (collection / "plan.json").exists() and _read_json(collection / "plan.json").get("plan_id") != plan["plan_id"]:
                raise ValueError("Collection is bound to another plan")
            index_path = collection / "imported-index.json"
            index_record = _read_json(index_path) if index_path.exists() else {"plan_id": plan["plan_id"], "files": {}}
            if index_record.get("plan_id") != plan["plan_id"]:
                raise ValueError("Collection integrity index belongs to another plan")
            index = dict(index_record.get("files", {}))
            for relative, digest in index.items():
                path = (collection / _safe_relative(relative)).resolve()
                if not path.is_relative_to(collection) or not path.is_file() or _digest_file(path) != digest:
                    raise ValueError("Existing imported outcome was modified")
            pending = []
            duplicated = 0
            seen = set()
            for entry in manifest.get("outcomes", []):
                task_id = entry.get("task_id")
                if task_id not in tasks or task_id in seen:
                    raise ValueError("Unknown or duplicated result task ID")
                seen.add(task_id)
                raw = archive.read(_safe_relative(entry["file"]))
                outcome = _loads(raw)
                task = tasks[task_id]
                if (outcome.get("task_id") != task_id or outcome.get("plan_id") != plan["plan_id"]
                    or outcome.get("config_sha256") != task["config_sha256"]
                    or outcome.get("implementation_sha256") != plan["identity"]["implementation_sha256"]
                    or outcome.get("input_sha256") != plan["identity"]["input_sha256"]
                    or outcome.get("status") not in {"completed", "failed"} or outcome["status"] != entry.get("status")):
                    raise ValueError("Result task identity/status mismatch")
                path = collection / "outcomes" / f"{task_id}.json"
                canonical = _json(outcome)
                if path.exists():
                    if index.get(path.relative_to(collection).as_posix()) != _digest_file(path):
                        raise ValueError("Existing outcome is not checksummed")
                    if _json(_read_json(path)) != canonical:
                        raise ValueError(f"Conflicting result for {task_id}; collection unchanged")
                    duplicated += 1
                else:
                    pending.append((path, canonical))
            if set(manifest["files"]) != {item["file"] for item in manifest.get("outcomes", [])}:
                raise ValueError("Unreferenced result payloads are not allowed")
            # Validate the entire packet, including all conflicts, before
            # changing collection data. Publication failure removes only files
            # created by this import; pre-existing outcomes remain untouched.
            _bind_plan(collection, plan)
            published = []
            try:
                for path, raw in pending:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as output:
                        published.append(path)
                        output.write(raw)
                    index[path.relative_to(collection).as_posix()] = _sha(raw)
                _write_json(index_path, {"plan_id": plan["plan_id"], "files": index})
            except BaseException:
                for path in published:
                    path.unlink(missing_ok=True)
                raise
            return {"plan_id": plan["plan_id"], "imported": len(pending), "duplicates": duplicated,
                    "summary": _aggregate_unlocked(collection, plan)}
    finally:
        lock.unlink(missing_ok=True)


def aggregate_results(collection_dir: str | Path, expected_plan_zip: str | Path) -> dict:
    plan = inspect_task_bundle(expected_plan_zip)
    collection = Path(collection_dir).resolve()
    collection.mkdir(parents=True, exist_ok=True)
    lock = _claim_task(collection, "result_collection")
    try:
        _bind_plan(collection, plan)
        return _aggregate_unlocked(collection, plan)
    finally:
        lock.unlink(missing_ok=True)


def _aggregate_unlocked(collection: Path, plan: dict) -> dict:
    from research_layer.batch import rank_trials
    index_path = collection / "imported-index.json"
    index = _read_json(index_path) if index_path.exists() else {"plan_id": plan["plan_id"], "files": {}}
    if index.get("plan_id") != plan["plan_id"]:
        raise ValueError("Collection integrity index belongs to another plan")
    trials = []
    missing = []
    for task in plan["tasks"]:
        path = collection / "outcomes" / f"{task['task_id']}.json"
        if not path.exists():
            missing.append(task["task_id"])
            continue
        if index.get("files", {}).get(path.relative_to(collection).as_posix()) != _digest_file(path):
            raise ValueError("Collection outcome is missing its checksum or has changed")
        outcome = _read_json(path)
        if outcome.get("plan_id") != plan["plan_id"] or outcome.get("config_sha256") != task["config_sha256"]:
            raise ValueError("Stored collection outcome identity mismatch")
        report = outcome.get("result", {}).get("strategy") or {}
        trials.append({"task_id": task["task_id"], "candidate_id": task["task_id"], "parameters": task["config"].get("parameters", {}),
                       "status": outcome["status"], "metrics": report.get("metrics", {}), "error": outcome.get("error"),
                       "outcome_path": str(path)})
    ranking = rank_trials(trials, plan["config"].get("selection_metric", "total_return")) if plan["original_mode"] == "batch" else []
    summary = {"plan_id": plan["plan_id"], "status": "complete" if not missing else "incomplete", "task_count": len(plan["tasks"]),
               "received_tasks": len(trials), "missing_task_ids": missing, "trials": trials,
               "ranking_scope": "in_sample" if plan["original_mode"] == "batch" else "not_ranked", "ranking": ranking,
               "completed_tasks": sum(t["status"] == "completed" for t in trials), "failed_tasks": sum(t["status"] == "failed" for t in trials)}
    _write_json(collection / "aggregate.json", summary)
    with (collection / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "status", "parameters", "total_return", "maximum_drawdown", "error"])
        writer.writeheader()
        for trial in trials:
            writer.writerow({"task_id": trial["task_id"], "status": trial["status"], "parameters": json.dumps(trial["parameters"], sort_keys=True),
                             "total_return": trial["metrics"].get("total_return"), "maximum_drawdown": trial["metrics"].get("maximum_drawdown"), "error": trial["error"]})
    return summary
