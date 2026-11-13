from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from agent_layer.data import (
    AgentCacheBuildConfig,
    AgentTimelineLoader,
    build_agent_cache,
    inspect_agent_cache,
    is_agent_cache_dir,
    open_agent_timeline_loader,
    validate_agent_cache_dataset,
)
from agent_layer.environment import ExecutionConfig, SingleSymbolTradingEnv
from agent_layer.experiments import (
    AgentRunStore,
    DEFAULT_AGENT_FREQUENCIES,
    ValidationQueue,
    agent_run_config_from_payload,
    build_walk_forward_splits,
    formal_run_config,
    server_run_config,
    save_agent_checkpoint,
    select_representative_symbols,
    select_symbols,
)
from agent_layer.models import SingleSymbolMultiFrequencyPolicy, SingleSymbolMultiFrequencyPolicyConfig
from agent_layer.training import PPOConfig, PPOTrainer, TrainingCancelled
from evaluation_layer.backtest import evaluate_policy
from agent_layer.data.feature_parts import FeaturePartsDataset, validate_feature_parts_dataset
from agent_layer.data.cache_schema import CACHE_MANIFEST_NAME, REQUIRED_RUNTIME_CONTRACT_FIELDS

from app_layer.backend.data_controller import (
    _relative_or_absolute,
    _resolve_symbols,
    resolve_project_path,
)


DEFAULT_FEATURE_DATASET_DIR = "runtime_layer/reports/feature_dataset"
FEATURE_PARTS_DIR_NAME = "feature_parts"
FEATURE_PARTS_MANIFEST_NAME = "feature_parts_manifest.json"
DEFAULT_AGENT_MODEL_DIR = "runtime_layer/models/agent"
DEFAULT_SMOKE_SYMBOLS = "config/universe/smoke_symbols.txt"
DEFAULT_AGENT_RUN_DIR = "runtime_layer/runs/agent"
DEFAULT_AGENT_CACHE_ROOT = "runtime_layer/agent_cache"
AGENT_CACHE_CURRENT_NAME = "current.json"




def start_agent_cache_job(payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    active = jobs.active_job("agent_cache_build")
    if active:
        raise ValueError(f"An Agent cache build job is already active: {active['job_id']}")
    _cleanup_pending_agent_caches()
    request = _agent_cache_request(payload)
    job_id = jobs.create_job("agent_cache_build", title="Build Agent Cache")
    jobs.update_job(
        job_id,
        total=0,
        message="Queued Agent cache build",
        current="queued",
    )
    jobs.submit(job_id, _run_agent_cache_job, job_id, request, jobs)
    return {"job_id": job_id, "status": "queued", "output_dir": str(request["output_dir"])}


def inspect_agent_cache_request(payload: dict[str, Any]) -> dict[str, Any]:
    cache_dir = resolve_project_path(
        payload.get("cache_dir") or payload.get("output_dir") or "runtime_layer/agent_cache/latest"
    )
    return inspect_agent_cache(cache_dir)


def _run_agent_cache_job(job_id: str, request: dict[str, Any], jobs: Any) -> dict[str, Any]:
    add_log = getattr(jobs, "add_log", None)
    final_output_dir = Path(request["output_dir"]).resolve()
    staging_output_dir = _agent_cache_staging_dir(final_output_dir)
    if callable(add_log):
        add_log(
            job_id,
            f"CONFIG feature_dir={request['feature_dir']}, output_dir={final_output_dir}, workers={request['workers']}",
        )

    def progress(item: dict[str, Any]) -> None:
        if jobs.is_cancel_requested(job_id):
            raise InterruptedError("Agent cache build cancellation requested.")
        phase = str(item.get("phase") or "")
        if phase == "starting":
            total = int(item.get("symbols") or 0)
            jobs.update_job(
                job_id,
                status="running",
                total=total,
                completed=0,
                progress=0.01,
                current="building",
                message=f"Building index-based Agent cache for {total} symbols",
            )
            if callable(add_log):
                add_log(job_id, f"START symbols={total}, workers={item.get('workers')}, storage={item.get('storage') or '-'}")
            return
        if phase == "symbol_done":
            summary = dict(item.get("summary") or {})
            index = int(item.get("index") or 0)
            current = jobs.get_job(job_id) or {}
            total = int(current.get("total") or 0)
            failed = int(current.get("failed") or 0) + (1 if summary.get("error") else 0)
            succeeded = int(current.get("succeeded") or 0) + (0 if summary.get("error") else 1)
            message = f"Cached {index}/{total}: {summary.get('symbol')} decisions={summary.get('decision_count', 0)}"
            if summary.get("error"):
                message += f" error={summary.get('error')}"
            jobs.update_job(
                job_id,
                completed=index,
                saved_rows=index,
                succeeded=succeeded,
                failed=failed,
                progress=index / max(1, total),
                current="building",
                message=message,
            )
            if callable(add_log) and (index == 1 or index == total or index % 10 == 0 or summary.get("error")):
                add_log(job_id, message, level="error" if summary.get("error") else "info")

    config = AgentCacheBuildConfig(
        feature_dir=request["feature_dir"],
        output_dir=staging_output_dir,
        symbols=request.get("symbols") or None,
        frequencies=request.get("frequencies") or None,
        start=request.get("start") or None,
        end=request.get("end") or None,
        stages=request.get("stages") or None,
        workers=int(request.get("workers") or 1),
        chunk_size=int(request.get("chunk_size") or 256),
        reset=bool(request.get("reset")),
        max_decisions_per_symbol=request.get("max_decisions_per_symbol") or None,
    )
    _prepare_agent_cache_staging(staging_output_dir)
    if final_output_dir.exists() and not bool(request.get("reset")):
        raise ValueError(
            f"Agent Cache output already exists: {final_output_dir}. "
            "Enable reset or choose a new output directory."
        )
    if bool(request.get("reset")) and final_output_dir.exists() and _agent_cache_referenced_by_resumable_run(final_output_dir):
        raise ValueError(
            "Refusing to reset an Agent Cache directory referenced by a queued/running/paused/interrupted run. "
            "Choose a new output directory under runtime_layer/agent_cache/builds."
        )
    summary = build_agent_cache(
        config,
        progress_callback=progress,
        cancel_check=lambda: jobs.is_cancel_requested(job_id),
    )
    if summary.get("ok"):
        _finalize_agent_cache_staging(staging_output_dir, final_output_dir, reset=bool(request.get("reset")))
        summary = {**summary, "output_dir": str(final_output_dir)}
    inspected = inspect_agent_cache(final_output_dir) if summary.get("ok") else {}
    if summary.get("ok"):
        _publish_current_agent_cache(final_output_dir, feature_dir=config.feature_dir)
    result = {**summary, "inspect": inspected}
    jobs.update_job(
        job_id,
        progress=1.0 if summary.get("ok") else 0.99,
        current="completed" if summary.get("ok") else "failed",
        message="Agent cache build completed" if summary.get("ok") else "Agent cache build completed with errors",
        result=result,
    )
    if callable(add_log):
        add_log(job_id, f"DONE ok={summary.get('ok')} output={summary.get('output_dir')} decisions={summary.get('decision_count')}")
    return result


def _agent_cache_request(payload: dict[str, Any]) -> dict[str, Any]:
    feature_dir = resolve_project_path(
        payload.get("feature_dir") or payload.get("source_feature_dir") or DEFAULT_FEATURE_DATASET_DIR
    )
    output_dir = resolve_project_path(
        payload.get("output_dir") or payload.get("cache_dir") or f"{DEFAULT_AGENT_CACHE_ROOT}/builds"
    )
    if output_dir.name.lower() == "builds":
        output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _assert_agent_cache_child(output_dir)
    requested_symbols = _resolve_symbols(payload)
    symbol_limit = max(0, int(payload.get("symbol_limit") or 0))
    symbol_seed = int(payload.get("symbol_seed") or 42)
    if requested_symbols:
        symbols = select_symbols(requested_symbols, limit=symbol_limit, seed=symbol_seed)
    else:
        symbols = select_symbols(
            FeaturePartsDataset(feature_dir).symbols,
            limit=symbol_limit,
            seed=symbol_seed,
        )
    frequencies = tuple(str(value) for value in (payload.get("frequencies") or ())) or None
    stages = tuple(str(value).strip() for value in (payload.get("stages") or []) if str(value).strip()) or None
    max_decisions_raw = payload.get("max_decisions_per_symbol")
    max_decisions = int(max_decisions_raw or 0)
    return {
        "feature_dir": feature_dir,
        "output_dir": output_dir,
        "symbols": tuple(symbols) if symbols else None,
        "symbol_limit": symbol_limit,
        "symbol_seed": symbol_seed,
        "frequencies": frequencies,
        "start": payload.get("start") or None,
        "end": payload.get("end") or None,
        "stages": stages,
        "workers": max(1, int(payload.get("workers") or 1)),
        "chunk_size": max(1, int(payload.get("chunk_size") or 256)),
        "reset": bool(payload.get("reset", True)),
        "max_decisions_per_symbol": max_decisions if max_decisions > 0 else None,
    }

def agent_spec_payload() -> dict[str, Any]:
    model = SingleSymbolMultiFrequencyPolicyConfig(
        frequency_channels={freq: 40 for freq in DEFAULT_AGENT_FREQUENCIES},
        decision_context_size=12,
        runtime_state_size=14,
        size_distribution=agent_run_config_from_payload({}).runtime.size_distribution,
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_config = agent_run_config_from_payload({"device": default_device})
    formal_payload = formal_run_config().payload()
    formal_payload.pop("config_hash", None)
    formal_payload["device"] = default_device
    formal_config = agent_run_config_from_payload(formal_payload)
    server_payload = server_run_config().payload()
    server_payload.pop("config_hash", None)
    server_payload["device"] = default_device
    server_config = agent_run_config_from_payload(server_payload)
    return {
        "algorithm": "Single-symbol multi-frequency PPO",
        "encoder": "Independent two-layer LSTM per frequency with masked attention pooling",
        "actor": "Shared single-stock SELL/HOLD/BUY categorical head plus configurable size head",
        "critic": "Single-symbol value critic over the sampled stock episode",
        "model": asdict(model),
        "frequency_options": [
            "5min", "15min", "30min", "60min", "daily", "weekly", "monthly",
        ],
        "ppo": asdict(PPOConfig()),
        "execution": asdict(ExecutionConfig()),
        "runtime": asdict(smoke_config.runtime),
        "defaults": {
            "store": DEFAULT_FEATURE_DATASET_DIR,
            "output_dir": DEFAULT_AGENT_MODEL_DIR,
            "symbols_file": DEFAULT_SMOKE_SYMBOLS,
            "fold": smoke_config.fold,
            "total_steps": 128,
            "episode_days": 5,
            "validation_days": 5,
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "parallel_envs": smoke_config.parallel_envs,
            "use_agent_cache": smoke_config.use_agent_cache,
            "frequencies": list(smoke_config.frequencies),
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ],
        "torch_version": torch.__version__,
        "profiles": {
            "smoke": smoke_config.payload(),
            "formal": formal_config.payload(),
            "server": server_config.payload(),
        },
    }


def agent_run_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    config, loader, splits, validation = _prepare_agent_run(payload)
    fold = splits.folds[config.fold - 1]
    checks = list(validation["checks"])
    checks.append(
        {
            "name": "immutable_run_config",
            "status": "pass",
            "message": (
                f"Resolved {len(config.resolved_symbols)} symbols and locked config "
                f"{config.config_hash[:12]}."
            ),
        }
    )
    topology = _distributed_topology(config)
    checks.append(
        {
            "name": "distributed_topology",
            "status": "pass",
            "message": topology["message"],
            "details": topology,
        }
    )
    checks.append(
        {
            "name": "time_split",
            "status": "pass",
            "message": (
                f"Fold {config.fold} trains through {fold.train.end.date()}, validates "
                f"{fold.validation.start.date()} to {fold.validation.end.date()}, and keeps "
                f"{splits.test.start.date()} to {splits.test.end.date()} untouched for testing."
            ),
        }
    )
    return {
        "ok": True,
        "status": "pass",
        "checks": checks,
        "store": validation,
        "symbols": len(config.resolved_symbols),
        "validation_symbols": len(config.resolved_validation_symbols),
        "frequencies": list(loader.frequencies),
        "schema_hash": loader.schema_hash,
        "config_hash": config.config_hash,
        "selected_fold": config.fold,
        "splits": splits.payload(),
        "device": config.device,
        "runtime": asdict(config.runtime),
        "distributed": topology,
        "resolved_config": config.payload(),
    }


def start_agent_run(payload: dict[str, Any]) -> dict[str, Any]:
    config, _, _, _ = _prepare_agent_run(payload)
    store = _agent_run_store()
    active = _active_agent_run(store)
    if active:
        raise ValueError(f"Agent run {active['run_id']} is already active.")
    created = store.create(config)
    spawned = _spawn_agent_worker(store, created["run_id"])
    return {"run_id": created["run_id"], "status": spawned}


def list_agent_runs() -> list[dict[str, Any]]:
    _cleanup_pending_agent_caches()
    store = _agent_run_store()
    result = []
    for item in store.list_runs():
        refreshed = _refresh_run_status(store, item)
        queue = ValidationQueue(store.run_dir(refreshed["run_id"]))
        validation = queue.refresh_status(restart_pending=False)
        refreshed = {**refreshed, "validation_status": validation.get("status")}
        result.append(refreshed)
    return result


def agent_run_detail(run_id: str, *, include_records: bool = True) -> dict[str, Any]:
    store = _agent_run_store()
    status = _refresh_run_status(store, store.read_status(run_id))
    metric_after = max(0, int(status.get("metrics_seq") or 0) - 5000)
    log_after = max(0, int(status.get("logs_seq") or 0) - 300)
    result = {
        "status": status,
        "config": store.read_config(run_id),
        "checkpoints": _checkpoint_records(store.run_dir(run_id)),
        "validation_status": ValidationQueue(store.run_dir(run_id)).refresh_status(restart_pending=False),
        "validation_results": ValidationQueue(store.run_dir(run_id)).list_results()[:20],
    }
    result["metrics"] = (
        store.read_records(run_id, "metrics", after=metric_after, limit=5000)
        if include_records else []
    )
    result["logs"] = (
        store.read_records(run_id, "logs", after=log_after, limit=300)
        if include_records else []
    )
    return result


def agent_run_records(
    run_id: str,
    kind: str,
    *,
    after: int = 0,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    return _agent_run_store().read_records(run_id, kind, after=after, limit=limit)


def agent_run_action(run_id: str, action: str) -> dict[str, Any]:
    store = _agent_run_store()
    status = _refresh_run_status(store, store.read_status(run_id))
    current = str(status.get("status") or "")
    if action == "resume":
        if current not in {"paused", "failed", "interrupted", "stopped"}:
            raise ValueError(f"Run cannot resume while status is {current}.")
        latest = status.get("latest_checkpoint")
        if not latest or not Path(str(latest)).exists():
            raise ValueError("Run has no safe checkpoint to resume from.")
        active = _active_agent_run(store, exclude=run_id)
        if active:
            raise ValueError(f"Agent run {active['run_id']} is already active.")
        store.clear_control(run_id)
        return _spawn_agent_worker(store, run_id)
    if action == "pause":
        if current not in {"queued", "running"}:
            raise ValueError(f"Run cannot pause while status is {current}.")
        store.request(run_id, action)
        return store.update_status(run_id, message="Pause requested; saving a safe checkpoint")
    if action == "checkpoint":
        if current != "running":
            raise ValueError("Manual checkpoint requires a running worker.")
        store.request(run_id, action)
        return store.update_status(run_id, message="Manual checkpoint requested")
    if action == "stop":
        if current in {"paused", "interrupted", "failed"}:
            store.request(run_id, action)
            return store.update_status(
                run_id,
                status="stopped",
                phase="stopped",
                pid=None,
                message="Run stopped",
            )
        if current not in {"queued", "running"}:
            raise ValueError(f"Run cannot stop while status is {current}.")
        store.request(run_id, action)
        pid = int(status.get("pid") or 0)
        if not _agent_worker_alive(pid, store.run_dir(run_id)):
            return store.update_status(
                run_id,
                status="stopped",
                phase="stopped",
                pid=None,
                message="Stale run stopped; no live Agent worker was found.",
            )
        return store.update_status(run_id, message="Stop requested; saving a safe checkpoint")
    raise ValueError(f"Unsupported Agent run action: {action}")



def export_agent_model_to_evaluation_pool(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    store = _agent_run_store()
    run_dir = store.run_dir(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Agent run not found: {run_id}")
    checkpoint_value = str(payload.get("checkpoint") or payload.get("checkpoint_path") or "").strip()
    if not checkpoint_value:
        raise ValueError("Select a checkpoint to export.")
    checkpoint = resolve_project_path(checkpoint_value)
    checkpoint_root = (run_dir / "checkpoints").resolve()
    checkpoint_resolved = checkpoint.resolve()
    try:
        relative_checkpoint = checkpoint_resolved.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError("Only numbered step checkpoints from this run can be exported.") from exc
    checkpoint_name = checkpoint_resolved.name
    if (
        len(relative_checkpoint.parts) != 1
        or not checkpoint_name.startswith("step_")
        or checkpoint_resolved.suffix.lower() != ".pt"
    ):
        raise ValueError("Only numbered step checkpoints from this run can be exported.")
    if not checkpoint_resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_value}")
    model_name = _safe_model_name(str(payload.get("model_name") or ""))
    if not model_name:
        raise ValueError("Model name is required.")
    overwrite = bool(payload.get("overwrite"))
    pool_dir = resolve_project_path("runtime_layer/models/evaluation_pool")
    model_dir = pool_dir / model_name
    if model_dir.exists() and not overwrite:
        raise FileExistsError(f"Evaluation model already exists: {model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / "model.pt"
    tmp = destination.with_suffix(".pt.tmp")
    shutil.copy2(checkpoint_resolved, tmp)
    tmp.replace(destination)

    manifest = _checkpoint_manifest(checkpoint_resolved)
    config = store.read_config(run_id)
    notes = str(payload.get("notes") or "")
    metadata = {
        "name": model_name,
        "notes": notes,
        "exported_at": datetime.now().isoformat(),
        "source_run_id": run_id,
        "source_checkpoint": _relative_or_absolute(checkpoint_resolved),
        "source_step": (manifest.get("training_state") or {}).get("total_steps"),
        "source_reason": (manifest.get("experiment") or {}).get("checkpoint_reason"),
        "schema_hash": manifest.get("schema_hash"),
        "universe_hash": manifest.get("universe_hash"),
        "model_config": manifest.get("model_config"),
        "action_space": "sell/hold/buy + continuous size",
    }
    _write_json_atomic(model_dir / "model_metadata.json", metadata)
    _write_json_atomic(model_dir / "training_config.json", config)
    _write_json_atomic(model_dir / "source_checkpoint.json", manifest)
    _write_json_atomic(
        model_dir / "model_input_contract.json",
        _model_input_contract_for_export(config, manifest),
    )
    return {
        "name": model_name,
        "path": _relative_or_absolute(model_dir),
        "model_path": _relative_or_absolute(destination),
        "source_run_id": run_id,
        "source_checkpoint": _relative_or_absolute(checkpoint_resolved),
        "overwritten": overwrite,
    }


def _safe_model_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return cleaned.strip("._-")[:80]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    deadline = time.monotonic() + 2.0
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.025)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


def _model_input_contract_for_export(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    source_dataset = str(config.get("store_path") or config.get("store") or DEFAULT_FEATURE_DATASET_DIR)
    try:
        feature_input, resolution = _resolve_export_feature_dataset(source_dataset)
        for candidate in (
            feature_input / "model_input_contract.json",
            feature_input.parent / "model_input_contract.json",
        ):
            if candidate.exists():
                loaded = _read_json_file(candidate)
                if loaded:
                    contract = loaded
                    break
        if not contract:
            root = feature_input.parent if feature_input.name == FEATURE_PARTS_DIR_NAME else feature_input
            manifest_payload = _read_json_file(root / "manifest.json")
            if isinstance(manifest_payload.get("model_input"), dict):
                contract = dict(manifest_payload["model_input"])
        if not contract:
            parts = FeaturePartsDataset.maybe(feature_input)
            if parts is not None:
                contract = parts.compiled_model_input()
        contract["source_feature_dataset"] = _relative_or_absolute(feature_input)
        contract["source_resolution"] = resolution
    except Exception as exc:
        contract = {
            "source": "checkpoint_manifest_only",
            "source_error": str(exc),
        }

    contract.setdefault("schema_hash", manifest.get("schema_hash"))
    contract.setdefault("model_config", manifest.get("model_config"))
    contract.setdefault("universe_hash", manifest.get("universe_hash"))
    contract.setdefault("frequencies", list((manifest.get("model_config") or {}).get("frequency_channels", {}).keys()))
    return contract


def _resolve_export_feature_dataset(raw_path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    requested = resolve_project_path(raw_path or DEFAULT_FEATURE_DATASET_DIR)
    if is_agent_cache_dir(requested):
        cache_manifest = _read_json_file(requested / CACHE_MANIFEST_NAME)
        source_feature_dir = str(cache_manifest.get("source_feature_dir") or "").strip()
        if not source_feature_dir:
            raise ValueError(f"Agent Cache manifest has no source_feature_dir: {requested / CACHE_MANIFEST_NAME}")
        feature_input, resolution = _resolve_agent_feature_input(source_feature_dir)
        return feature_input, {
            **resolution,
            "source": "agent_cache_source_feature_parts",
            "agent_cache_dir": str(requested),
            "agent_cache_manifest": str(requested / CACHE_MANIFEST_NAME),
        }
    return _resolve_agent_feature_input(requested)


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _assert_agent_cache_child(path: Path) -> Path:
    resolved = path.resolve()
    root = _agent_cache_root().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Agent Cache path must be under {root}: {resolved}"
        ) from exc
    if resolved == root:
        raise ValueError(f"Refusing to use Agent Cache root as an output directory: {resolved}")
    return resolved


def _agent_cache_staging_dir(final_output_dir: Path) -> Path:
    final = _assert_agent_cache_child(final_output_dir)
    return final.with_name(f"{final.name}.building")


def _prepare_agent_cache_staging(staging_dir: Path) -> None:
    staging = _assert_agent_cache_child(staging_dir)
    if staging.exists():
        if not staging.name.endswith(".building"):
            raise ValueError(f"Refusing to clear non-staging cache directory: {staging}")
        _remove_tree_with_retry(staging)


def _finalize_agent_cache_staging(staging_dir: Path, final_output_dir: Path, *, reset: bool) -> None:
    staging = _assert_agent_cache_child(staging_dir)
    final = _assert_agent_cache_child(final_output_dir)
    if not is_agent_cache_dir(staging):
        raise ValueError(f"Built Agent Cache staging directory is incomplete: {staging}")
    if final.exists():
        if not reset:
            raise ValueError(f"Agent Cache output already exists: {final}")
        if final.is_dir() and not any(final.iterdir()):
            _remove_tree_with_retry(final)
        elif is_agent_cache_dir(final):
            _remove_tree_with_retry(final)
        else:
            raise ValueError(
                f"Refusing to replace a non-cache, non-empty Agent Cache output directory: {final}"
            )
    _replace_directory_with_retry(staging, final)


def _remove_tree_with_retry(path: Path, *, attempts: int = 8) -> None:
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            if not path.exists():
                return
        except OSError as exc:
            last_error = exc
        time.sleep(min(2.0, 0.25 * (attempt + 1)))
    raise PermissionError(f"Could not remove Agent Cache directory after retries: {path}") from last_error


def _replace_directory_with_retry(staging: Path, final: Path, *, attempts: int = 8) -> None:
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            staging.replace(final)
            return
        except OSError as exc:
            last_error = exc
            if final.exists() or not staging.exists():
                break
            time.sleep(min(2.0, 0.25 * (attempt + 1)))
    raise PermissionError(
        f"Could not publish Agent Cache after retries: {staging} -> {final}. "
        "On Windows this usually means another process still holds a file handle in the cache directory."
    ) from last_error


def _agent_cache_root() -> Path:
    return resolve_project_path(DEFAULT_AGENT_CACHE_ROOT)


def _agent_cache_current_file() -> Path:
    return _agent_cache_root() / AGENT_CACHE_CURRENT_NAME


def _publish_current_agent_cache(cache_dir: str | Path, *, feature_dir: str | Path) -> None:
    output = _agent_cache_current_file()
    previous = _read_json_file(output)
    previous_cache_dir = previous.get("cache_dir") or previous.get("output_dir")
    cache_path = Path(cache_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_dir": str(cache_path),
        "source_feature_dir": str(Path(feature_dir).resolve()),
        "published_at": datetime.now().isoformat(),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    if previous_cache_dir:
        _cleanup_unreferenced_agent_cache(previous_cache_dir, keep=cache_path)
    _cleanup_pending_agent_caches(keep=cache_path)


def _cleanup_unreferenced_agent_cache(cache_dir: str | Path, *, keep: Path) -> None:
    candidate = resolve_project_path(cache_dir)
    try:
        candidate = candidate.resolve()
        keep = keep.resolve()
    except OSError:
        return
    if candidate == keep or not candidate.exists():
        return
    root = _agent_cache_root().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    if candidate == root:
        return
    if _agent_cache_referenced_by_resumable_run(candidate):
        pending = candidate.with_suffix(candidate.suffix + ".pending_delete")
        marker = pending if candidate.suffix else candidate.parent / f"{candidate.name}.pending_delete"
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
        return
    if is_agent_cache_dir(candidate):
        shutil.rmtree(candidate, ignore_errors=True)


def _cleanup_pending_agent_caches(*, keep: Path | None = None) -> None:
    root = _agent_cache_root()
    if not root.exists():
        return
    keep_resolved = keep.resolve() if keep is not None else None
    for marker in root.rglob("*.pending_delete"):
        if not marker.is_file():
            continue
        target_name = marker.name[: -len(".pending_delete")]
        target = marker.with_name(target_name)
        try:
            target = target.resolve()
            if keep_resolved is not None and target == keep_resolved:
                continue
            if target.exists() and _agent_cache_referenced_by_resumable_run(target):
                continue
            if target.exists() and is_agent_cache_dir(target):
                shutil.rmtree(target, ignore_errors=True)
            if not target.exists():
                marker.unlink(missing_ok=True)
        except Exception:
            continue


def _agent_cache_referenced_by_resumable_run(cache_dir: Path) -> bool:
    try:
        store = _agent_run_store()
        target = cache_dir.resolve()
        for item in store.list_runs():
            if item.get("status") not in {"queued", "running", "paused", "interrupted"}:
                continue
            config = store.read_config(str(item.get("run_id") or ""))
            raw_path = config.get("store_path")
            if not raw_path:
                continue
            if resolve_project_path(raw_path).resolve() == target:
                return True
    except Exception:
        return True
    return False


def _resolve_current_agent_cache() -> tuple[Path, dict[str, Any]]:
    current_file = _agent_cache_current_file()
    payload = _read_json_file(current_file)
    raw_cache_dir = payload.get("cache_dir") or payload.get("output_dir")
    candidates: list[Path] = []
    if raw_cache_dir:
        candidates.append(resolve_project_path(raw_cache_dir))
    root = _agent_cache_root()
    candidates.extend([root / "current", root / "latest", root / "full"])
    for candidate in candidates:
        if is_agent_cache_dir(candidate):
            return candidate.resolve(), {
                "source": "agent_cache",
                "requested_path": str(current_file),
                "cache_dir": str(candidate.resolve()),
                "current_file": str(current_file),
                "current": payload,
                "legacy_fallback": candidate.name in {"current", "latest", "full"} and not raw_cache_dir,
            }
    raise FileNotFoundError(
        "Agent Cache is enabled, but no current cache was found. "
        f"Build Agent Cache from the Agent Cache panel first. Expected current pointer at {current_file}."
    )


def _resolve_agent_training_input(
    raw_path: str | Path | None,
    *,
    use_agent_cache: bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    requested = resolve_project_path(raw_path or DEFAULT_FEATURE_DATASET_DIR)
    if use_agent_cache is True:
        if is_agent_cache_dir(requested):
            return requested, {
                "source": "agent_cache",
                "requested_path": str(requested),
                "cache_dir": str(requested),
                "legacy_fallback": False,
            }
        return _resolve_current_agent_cache()
    if use_agent_cache is False and is_agent_cache_dir(requested):
        raise ValueError(
            "Use Agent Cache is disabled, but Training Dataset points to an Agent Cache. "
            "Choose the Feature Dataset directory or enable Agent Cache."
        )
    if is_agent_cache_dir(requested):
        return requested, {
            "source": "agent_cache",
            "requested_path": str(requested),
            "cache_dir": str(requested),
            "legacy_fallback": False,
        }
    return _resolve_agent_feature_input(requested)


def _resolve_agent_feature_input(raw_path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    """Resolve the Agent input path to the canonical feature_parts dataset."""

    requested = resolve_project_path(raw_path or DEFAULT_FEATURE_DATASET_DIR)
    if requested.suffix.lower() == ".duckdb":
        raise ValueError(
            "Agent no longer accepts feature_store.duckdb as a training input. "
            "Use the Feature Dataset directory or its feature_parts directory instead."
        )

    dataset_dir = requested.parent if requested.name == FEATURE_PARTS_DIR_NAME else requested
    parts_root = dataset_dir / FEATURE_PARTS_DIR_NAME
    manifest_path = dataset_dir / FEATURE_PARTS_MANIFEST_NAME
    try:
        FeaturePartsDataset(dataset_dir)
    except Exception as parts_error:
        raise FileNotFoundError(
            "Agent Feature Dataset is not ready. Expected per-symbol feature_parts under "
            f"{parts_root}. Run Feature Build first. Details: {parts_error}"
        ) from parts_error
    return dataset_dir, {
        "source": "feature_parts",
        "requested_path": str(requested),
        "dataset_dir": str(dataset_dir),
        "parts_root": str(parts_root),
        "manifest_path": str(manifest_path),
        "legacy_fallback": False,
    }



def _validate_agent_training_input(
    path: Path,
    resolution: dict[str, Any],
    *,
    config: Any | None = None,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_schema_hash = _active_agent_model_input_schema_hash()
    source = str(resolution.get("source") or "")
    if source == "agent_cache":
        universe = None
        frequencies = None
        if config is not None:
            universe = getattr(config, "resolved_symbols", None) or None
            frequencies = getattr(config, "frequencies", None) or None
        if request is not None:
            universe = request.get("symbols") or universe
            frequencies = request.get("frequencies") or frequencies
        validation = validate_agent_cache_dataset(path, universe=universe, frequencies=frequencies)
        _append_schema_hash_check(validation, expected_schema_hash)
        return validation
    return validate_feature_parts_dataset(
        path,
        expected_schema_hash=expected_schema_hash,
        sample_limit=0,
    )


def _active_agent_model_input_schema_hash() -> str | None:
    try:
        from feature_layer.model_input_registry import (
            compile_model_input_blueprint,
            load_model_input_blueprint,
        )

        compiled = compile_model_input_blueprint(load_model_input_blueprint())
        return str(compiled.get("schema_hash") or "") or None
    except Exception:
        return None


def _append_schema_hash_check(validation: dict[str, Any], expected_schema_hash: str | None) -> None:
    if not expected_schema_hash:
        return
    actual = str(validation.get("schema_hash") or validation.get("summary", {}).get("schema_hash") or "")
    ok = bool(actual) and actual == expected_schema_hash
    validation.setdefault("checks", []).append(
        {
            "name": "model_schema",
            "status": "pass" if ok else "error",
            "message": (
                f"Model input schema is {actual[:12]}."
                if ok
                else (
                    "Agent cache schema does not match current model_input.json. "
                    "Rebuild Feature Dataset, then rebuild Agent Cache."
                )
            ),
            "schema_hash": actual,
            "expected_schema_hash": expected_schema_hash,
        }
    )
    if not ok:
        validation["ok"] = False
        validation["status"] = "error"

def _agent_validation_error_messages(validation: dict[str, Any]) -> list[str]:
    return [
        str(item.get("message") or item.get("name") or "Agent Feature Parts validation failed")
        for item in validation.get("checks", [])
        if item.get("status") == "error"
    ]


def _runtime_contract_check(runtime_contract: tuple[str, ...] | list[str]) -> dict[str, str]:
    missing = sorted(set(REQUIRED_RUNTIME_CONTRACT_FIELDS).difference(str(value) for value in runtime_contract))
    if missing:
        return {
            "name": "runtime_contract",
            "status": "error",
            "message": (
                "Agent runtime contract is outdated; missing "
                + ", ".join(missing)
                + ". Rebuild Feature Dataset, then rebuild Agent Cache if cache is enabled."
            ),
        }
    return {
        "name": "runtime_contract",
        "status": "pass",
        "message": f"{len(tuple(runtime_contract))} runtime fields available.",
    }


def _agent_validation_should_force_rebuild(validation: dict[str, Any]) -> bool:
    error_names = {
        str(item.get("name") or "")
        for item in validation.get("checks", [])
        if item.get("status") == "error"
    }
    return bool(error_names.intersection({"model_schema", "sample_batches"}))


def _agent_feature_part_dirs(dataset_dir: Path) -> list[Path]:
    parts_root = dataset_dir / FEATURE_PARTS_DIR_NAME
    manifest_path = dataset_dir / FEATURE_PARTS_MANIFEST_NAME
    ordered: list[Path] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        symbols = manifest.get("symbols_order") or sorted((manifest.get("symbols") or {}).keys())
        for symbol in symbols:
            entry = dict((manifest.get("symbols") or {}).get(str(symbol), {}))
            raw_parts = entry.get("parts_dir")
            if raw_parts:
                candidate = resolve_project_path(raw_parts)
            else:
                safe_symbol = str(symbol).upper().replace(".", "_")
                candidate = parts_root / f"symbol={safe_symbol}"
            if _agent_feature_part_dir_complete(candidate):
                ordered.append(candidate)
    if not ordered and parts_root.exists():
        ordered = [path for path in sorted(parts_root.glob("symbol=*")) if _agent_feature_part_dir_complete(path)]
    return ordered


def _agent_feature_part_dir_complete(path: Path) -> bool:
    required = (
        "decisions",
        "decision_context",
        "constraints",
        "decision_index",
        "market_bars",
        "market_features",
        "decision_snapshots",
        "dataset_metadata",
    )
    return path.exists() and all((path / table).exists() and any((path / table).glob("*.parquet")) for table in required)


def _prepare_agent_run(payload: dict[str, Any]):
    config = agent_run_config_from_payload(payload)
    feature_input_path, input_resolution = _resolve_agent_training_input(
        config.store_path,
        use_agent_cache=config.use_agent_cache,
    )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    _distributed_topology(config)
    validation = _validate_agent_training_input(feature_input_path, input_resolution, config=config)
    validation["resolution"] = input_resolution
    if not validation["ok"]:
        messages = _agent_validation_error_messages(validation)
        if input_resolution.get("source") == "agent_cache":
            messages.append(
                "Fix: rebuild Feature Dataset if model_input/frequency/schema changed, then build Agent Cache from the Agent Cache panel."
            )
        else:
            messages.append(
                "Fix: rebuild Feature Dataset so feature_parts contain the current model_input schema and all selected frequencies."
            )
        raise ValueError("Agent preflight failed: " + "; ".join(messages))
    requested = _resolve_symbols(payload)
    if not requested and config.symbols_file:
        requested = _resolve_symbols({"symbols_file": config.symbols_file})
    loader = open_agent_timeline_loader(
        feature_input_path,
        universe=requested or None,
        frequencies=config.frequencies or None,
        validate_store=False,
        use_market_cache=False,
        use_decision_cache=False,
    )
    symbols = select_symbols(
        loader.universe,
        limit=config.symbol_limit,
        seed=config.symbol_seed,
    )
    validation_symbols = select_representative_symbols(
        symbols,
        limit=min(config.validation.symbol_limit, len(symbols)),
        seed=config.validation.symbol_seed,
    )
    loader = open_agent_timeline_loader(
        feature_input_path,
        universe=symbols,
        frequencies=config.frequencies or None,
        validate_store=False,
        use_market_cache=False,
        use_decision_cache=False,
    )
    runtime_check = _runtime_contract_check(loader.runtime_contract)
    validation.setdefault("checks", []).append(runtime_check)
    if runtime_check["status"] == "error":
        raise ValueError("Agent preflight failed: " + runtime_check["message"])
    splits = build_walk_forward_splits(
        loader.trading_dates(),
        validation_days=config.split.validation_days,
        test_days=config.split.test_days,
        embargo_days=config.split.embargo_days,
        folds=config.split.folds,
    )
    locked = config.with_contract(
        symbols=loader.universe,
        validation_symbols=validation_symbols,
        schema_hash=loader.schema_hash,
    )
    locked_payload = locked.payload()
    locked_payload.pop("config_hash", None)
    locked_payload["store_path"] = str(feature_input_path)
    locked_payload["feature_input_resolution"] = input_resolution
    locked_payload["output_dir"] = str(resolve_project_path(locked.output_dir))
    locked = agent_run_config_from_payload(locked_payload)
    return locked, loader, splits, validation


def _agent_run_store() -> AgentRunStore:
    return AgentRunStore(resolve_project_path(DEFAULT_AGENT_RUN_DIR))


def _active_agent_run(
    store: AgentRunStore,
    *,
    exclude: str | None = None,
) -> dict[str, Any] | None:
    for item in store.list_runs():
        if item.get("run_id") == exclude:
            continue
        refreshed = _refresh_run_status(store, item)
        if refreshed.get("status") not in {"queued", "running"}:
            continue
        pid = int(refreshed.get("pid") or 0)
        if pid and _agent_worker_alive(pid, store.run_dir(refreshed["run_id"])):
            return refreshed
    return None


def _spawn_agent_worker(store: AgentRunStore, run_id: str) -> dict[str, Any]:
    run_dir = store.run_dir(run_id)
    config = agent_run_config_from_payload(store.read_config(run_id))
    topology = _distributed_topology(config)
    process_count = int(topology["processes_per_node"])
    stdout_path = run_dir / "worker.stdout.log"
    stderr_path = run_dir / "worker.stderr.log"
    if process_count > 1:
        command = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "agent_layer.cli.distributed_launcher",
            "--run-dir",
            str(run_dir),
            "--processes",
            str(process_count),
            "--master-port",
            str(_available_local_port()),
        ]
    else:
        command = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "agent_layer.cli.worker",
            "--run-dir",
            str(run_dir),
        ]
    popen_kwargs: dict[str, Any] = {
        "cwd": str(resolve_project_path(".")),
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
        "env": _distributed_worker_environment(config, process_count),
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_kwargs["start_new_session"] = True
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **popen_kwargs)
    store.clear_control(run_id)
    return store.update_status(
        run_id,
        status="queued",
        phase="launching",
        message=(
            f"Distributed training launched with {process_count} processes"
            if process_count > 1
            else "Independent training worker launched"
        ),
        pid=process.pid,
        error=None,
        distributed=topology,
    )


def _distributed_topology(config: Any) -> dict[str, Any]:
    policy = config.distributed
    visible_gpus = int(torch.cuda.device_count()) if config.device == "cuda" else 0
    requested = int(policy.processes_per_node)
    if policy.strategy == "off":
        processes = 1
    elif policy.strategy == "auto":
        processes = requested or (visible_gpus if visible_gpus > 1 else 1)
    elif config.device == "cuda":
        processes = requested or visible_gpus
    else:
        processes = requested
    if processes <= 0:
        raise ValueError(
            "DDP process count could not be resolved. Set distributed.processes_per_node."
        )
    if processes > 1 and config.device == "cuda" and processes > visible_gpus:
        raise ValueError(
            f"DDP requested {processes} processes but only {visible_gpus} CUDA devices are visible."
        )
    if policy.strategy == "ddp" and processes < 2:
        raise ValueError("DDP requires at least two worker processes.")
    if processes > 1 and config.total_steps % processes != 0:
        raise ValueError(
            f"total_steps={config.total_steps} must be divisible by DDP world size {processes}."
        )
    backend = policy.backend
    if backend == "auto":
        backend = "nccl" if config.device == "cuda" else "gloo"
    if processes > 1 and config.device == "cuda" and os.name == "nt" and backend == "nccl":
        raise ValueError("CUDA DDP with NCCL requires Linux; Windows supports the Gloo CPU diagnostic path only.")
    if processes > 1 and not torch.distributed.is_available():
        raise ValueError("This PyTorch build does not include torch.distributed.")
    if processes > 1 and backend == "nccl" and not torch.distributed.is_nccl_available():
        raise ValueError("This PyTorch build does not include NCCL support.")
    if processes > 1 and backend == "gloo" and not torch.distributed.is_gloo_available():
        raise ValueError("This PyTorch build does not include Gloo support.")
    cpu_count = max(1, int(os.cpu_count() or 1))
    cpu_threads = int(policy.cpu_threads_per_process) or max(1, cpu_count // processes)
    mode = "ddp" if processes > 1 else "single"
    return {
        "mode": mode,
        "strategy": policy.strategy,
        "processes_per_node": processes,
        "visible_gpus": visible_gpus,
        "backend": backend if processes > 1 else None,
        "cpu_threads_per_process": cpu_threads,
        "global_parallel_envs": int(config.parallel_envs) * processes,
        "env_backend": getattr(config.runtime, "env_backend", "in_process"),
        "local_rollout_samples": int(config.parallel_envs) * int(config.ppo.rollout_steps),
        "global_rollout_samples": (
            int(config.parallel_envs) * int(config.ppo.rollout_steps) * processes
        ),
        "local_minibatch_size": int(config.ppo.minibatch_size),
        "effective_global_minibatch_size": int(config.ppo.minibatch_size) * processes,
        "message": (
            f"DDP will use {processes} processes, {visible_gpus} visible GPUs, "
            f"{config.parallel_envs} environments per rank, backend={backend}."
            if processes > 1
            else f"Single-process training will use {config.parallel_envs} environments ({getattr(config.runtime, 'env_backend', 'in_process')} backend)."
        ),
    }


def _distributed_worker_environment(config: Any, process_count: int) -> dict[str, str]:
    environment = dict(os.environ)
    cpu_count = max(1, int(os.cpu_count() or 1))
    threads = int(config.distributed.cpu_threads_per_process) or max(
        1, cpu_count // max(1, process_count)
    )
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    if os.name == "nt":
        environment.setdefault("USE_LIBUV", "0")
    if config.device == "cuda" and process_count > 1:
        environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        environment.setdefault("NCCL_DEBUG", "WARN")
    return environment


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _refresh_run_status(store: AgentRunStore, status: dict[str, Any]) -> dict[str, Any]:
    if status.get("status") not in {"queued", "running"}:
        return status
    pid = int(status.get("pid") or 0)
    run_dir = store.run_dir(status["run_id"])
    if pid and _agent_worker_alive(pid, run_dir):
        # A live worker can spend a long time inside a PPO update or Windows I/O.
        # Treat stale heartbeats as a UI warning, not as proof that the worker died.
        if _agent_heartbeat_stale(status):
            return {**status, "heartbeat_warning": True}
        return status
    return store.update_status(
        status["run_id"],
        status="interrupted",
        phase="interrupted",
        pid=None,
        message="Training worker is no longer running; resume from the latest safe checkpoint.",
    )


def _agent_worker_alive(pid: int, run_dir: Path) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        process = psutil.Process(int(pid))
        if not process.is_running():
            return False
        try:
            command = " ".join(process.cmdline()).lower()
        except (psutil.AccessDenied, psutil.ZombieProcess):
            return _pid_alive(pid)
        if not command:
            return _pid_alive(pid)
        normalized_run_dir = str(run_dir.resolve()).replace("\\", "/").lower()
        normalized_command = command.replace("\\", "/")
        return (
            (
                "agent_layer.cli.worker" in normalized_command
                or "agent_layer.cli.distributed_launcher" in normalized_command
            )
            and "--run-dir" in normalized_command
            and (normalized_run_dir in normalized_command or run_dir.name.lower() in normalized_command)
        )
    except ImportError:
        return _pid_alive(pid)
    except Exception:
        return False


def _agent_heartbeat_stale(status: dict[str, Any]) -> bool:
    timestamp = _parse_status_datetime(status.get("heartbeat") or status.get("updated_at"))
    if timestamp is None:
        timestamp = _parse_status_datetime(status.get("created_at"))
    if timestamp is None:
        return False
    return (datetime.now(timestamp.tzinfo) - timestamp).total_seconds() > 600


def _parse_status_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.astimezone()
        return parsed
    except (TypeError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not process:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(
                    process, ctypes.byref(exit_code)
                ):
                    return False
                return exit_code.value == 259
            finally:
                ctypes.windll.kernel32.CloseHandle(process)
        except (AttributeError, OSError, ValueError):
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _checkpoint_records(run_dir: Path) -> list[dict[str, Any]]:
    checkpoint_dir = run_dir / "checkpoints"
    result = []
    for path in checkpoint_dir.glob("step_*.pt"):
        stat = path.stat()
        manifest = _checkpoint_manifest(path)
        training_state = manifest.get("training_state") or {}
        experiment = manifest.get("experiment") or {}
        validation = (experiment.get("validation") or {}) if isinstance(experiment, dict) else {}
        quick_validation = (experiment.get("quick_validation") or {}) if isinstance(experiment, dict) else {}
        display_validation = validation
        if (
            isinstance(quick_validation, dict)
            and quick_validation
            and (not isinstance(validation, dict) or validation.get("status") != "completed")
        ):
            display_validation = quick_validation
        validation = display_validation if isinstance(display_validation, dict) else {}
        validation_metrics = validation.get("metrics") if isinstance(validation, dict) else None
        result.append(
            {
                "name": path.name,
                "path": _relative_or_absolute(path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "step": training_state.get("total_steps"),
                "updates": training_state.get("updates"),
                "episodes": training_state.get("episodes"),
                "reason": experiment.get("checkpoint_reason") if isinstance(experiment, dict) else None,
                "schema_hash": manifest.get("schema_hash"),
                "validation": {
                    "status": validation.get("status") if isinstance(validation, dict) else None,
                    "total_return": (validation_metrics or {}).get("total_return") if isinstance(validation_metrics, dict) else None,
                    "maximum_drawdown": (validation_metrics or {}).get("maximum_drawdown") if isinstance(validation_metrics, dict) else None,
                    "sharpe": (validation_metrics or {}).get("sharpe") if isinstance(validation_metrics, dict) else None,
                    "calmar": (validation_metrics or {}).get("calmar") if isinstance(validation_metrics, dict) else None,
                },
            }
        )
    return sorted(result, key=lambda item: item["modified_at"], reverse=True)


def _checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".json")
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def agent_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    request = _agent_request(payload)
    feature_input_path, input_resolution = _resolve_agent_training_input(
        request["store_path"],
        use_agent_cache=bool(request.get("use_agent_cache")),
    )
    request["store_path"] = feature_input_path
    validation = _validate_agent_training_input(feature_input_path, input_resolution, request=request)
    validation["resolution"] = input_resolution
    checks = list(validation["checks"])
    if not validation["ok"]:
        return {
            "ok": False,
            "status": "error",
            "checks": checks,
            "store": validation,
            "splits": None,
        }
    try:
        loader = open_agent_timeline_loader(
            request["store_path"],
            universe=request["symbols"] or None,
            frequencies=request.get("frequencies") or None,
            validate_store=False,
            use_market_cache=False,
            use_decision_cache=False,
        )
        runtime_check = _runtime_contract_check(loader.runtime_contract)
        checks.append(runtime_check)
        if runtime_check["status"] == "error":
            raise ValueError(runtime_check["message"])
        dates = loader.trading_dates()
        splits = build_walk_forward_splits(
            dates,
            validation_days=request["split"]["validation_days"],
            test_days=request["split"]["test_days"],
            embargo_days=request["split"]["embargo_days"],
            folds=request["split"]["folds"],
        )
        fold = splits.folds[request["fold"] - 1]
        checks.append(
            {
                "name": "time_split",
                "status": "pass",
                "message": (
                    f"Fold {request['fold']} trains through {fold.train.end.date()}, validates "
                    f"{fold.validation.start.date()} to {fold.validation.end.date()}, and keeps "
                    f"{splits.test.start.date()} to {splits.test.end.date()} untouched for testing."
                ),
            }
        )
    except Exception as exc:
        checks.append({"name": "agent_contract", "status": "error", "message": str(exc)})
        return {
            "ok": False,
            "status": "error",
            "checks": checks,
            "store": validation,
            "splits": None,
        }
    return {
        "ok": True,
        "status": "pass",
        "checks": checks,
        "store": validation,
        "symbols": len(loader.universe),
        "frequencies": list(loader.frequencies),
        "schema_hash": loader.schema_hash,
        "selected_fold": request["fold"],
        "splits": splits.payload(),
        "device": request["device"],
    }


def start_agent_training_job(payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    active = [
        job for job in jobs.list_jobs()
        if job.get("type") == "agent_training" and job.get("status") in {"queued", "running"}
    ]
    if active:
        raise ValueError(f"An Agent training job is already active: {active[-1]['job_id']}")
    request = _agent_request(payload)
    job_id = jobs.create_job("agent_training", title="Train Agent")
    jobs.update_job(
        job_id,
        total=request["total_steps"],
        message="Queued Agent training",
    )
    jobs.submit(job_id, _run_agent_training_job, job_id, payload, jobs)
    return {"job_id": job_id, "status": "queued"}


def _run_agent_training_job(
    job_id: str,
    payload: dict[str, Any],
    jobs: Any,
) -> dict[str, Any]:
    request = _agent_request(payload)
    feature_input_path, input_resolution = _resolve_agent_training_input(
        request["store_path"],
        use_agent_cache=bool(request.get("use_agent_cache")),
    )
    request["store_path"] = feature_input_path
    add_log = getattr(jobs, "add_log", None)
    jobs.update_job(
        job_id,
        status="running",
        progress=0.01,
        current="preflight",
        message="Validating Feature Parts and time splits",
    )
    if callable(add_log):
        add_log(job_id, f"CONFIG store={request['store_path']}, fold={request['fold']}, steps={request['total_steps']}, seed={request['seed']}, device={request['device']}")
    preflight = agent_preflight(payload)
    if not preflight["ok"]:
        raise ValueError("Agent preflight failed: " + "; ".join(
            check["message"] for check in preflight["checks"] if check["status"] == "error"
        ))

    try:
        loader = open_agent_timeline_loader(
            request["store_path"],
            universe=request["symbols"] or None,
            frequencies=request["frequencies"] or None,
            validate_store=False,
        )
        dates = loader.trading_dates()
        splits = build_walk_forward_splits(
            dates,
            validation_days=request["split"]["validation_days"],
            test_days=request["split"]["test_days"],
            embargo_days=request["split"]["embargo_days"],
            folds=request["split"]["folds"],
        )
        fold = splits.folds[request["fold"] - 1]
        train_dates = [date for date in dates if fold.train.start <= date <= fold.train.end]
        rng = np.random.default_rng(request["seed"])
        np.random.seed(request["seed"])
        torch.manual_seed(request["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request["seed"])

        symbol_loaders: dict[str, AgentTimelineLoader] = {}

        def loader_for_symbol(symbol: str) -> AgentTimelineLoader:
            normalized = str(symbol).upper()
            selected = symbol_loaders.get(normalized)
            if selected is None:
                selected = loader.for_universe((normalized,))
                symbol_loaders[normalized] = selected
            return selected

        def environment_factory() -> SingleSymbolTradingEnv:
            days = min(request["episode_days"], len(train_dates))
            attempts = max(4, len(loader.universe) * 2)
            last_error: Exception | None = None
            for _ in range(attempts):
                symbol = str(rng.choice(loader.universe)).upper()
                start_index = int(rng.integers(0, len(train_dates) - days + 1))
                try:
                    return SingleSymbolTradingEnv(
                        loader_for_symbol(symbol),
                        start=train_dates[start_index],
                        end=train_dates[start_index + days - 1],
                        execution_config=ExecutionConfig(),
                        enforce_action_contract=True,
                    )
                except ValueError as exc:
                    last_error = exc
            raise ValueError("Unable to create a single-symbol training episode.") from last_error

        model = SingleSymbolMultiFrequencyPolicy(
            SingleSymbolMultiFrequencyPolicyConfig(
                frequency_channels={
                    freq: len(loader.feature_names[freq]) for freq in loader.frequencies
                },
                decision_context_size=len(loader.decision_context_names),
                runtime_state_size=len(loader.runtime_contract),
                size_distribution=request["runtime"]["size_distribution"],
            )
        )
        ppo_config = PPOConfig()
        trainer = PPOTrainer(model, config=ppo_config, device=request["device"])

        last_logged_step = -1

        def progress(item: dict[str, float]) -> None:
            nonlocal last_logged_step
            completed = int(item["steps"])
            phase = str(item.get("phase") or "training")
            jobs.update_job(
                job_id,
                completed=completed,
                saved_rows=completed,
                total=request["total_steps"],
                progress=0.02 + 0.83 * item["progress"],
                current=phase,
                message=f"{phase.title()} {completed}/{request['total_steps']} steps",
            )
            should_log = (
                phase == "updated"
                or completed == 1
                or completed - last_logged_step >= 10
            )
            if callable(add_log) and should_log:
                add_log(job_id, f"UPDATE steps={completed}, loss={item['loss']:.6f}, kl={item['approximate_kl']:.6f}, episodes={int(item['episodes'])}")
                last_logged_step = completed

        summary = trainer.train(
            environment_factory,
            total_steps=request["total_steps"],
            progress_callback=progress,
            cancel_check=lambda: jobs.is_cancel_requested(job_id),
        )
        if jobs.is_cancel_requested(job_id):
            raise TrainingCancelled("Agent training cancellation requested.")
        output = request["output_path"] or (
            resolve_project_path(DEFAULT_AGENT_MODEL_DIR)
            / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        checkpoint, manifest = save_agent_checkpoint(
            output,
            model=model,
            optimizer=trainer.optimizer,
            schema_hash=loader.schema_hash,
            universe=loader.universe,
            ppo_config=ppo_config,
            execution_config=ExecutionConfig(),
            experiment={
                "selected_fold": request["fold"],
                "splits": splits.payload(),
                "validation": {"status": "pending"},
                "episode_days": request["episode_days"],
                "validation_days": request["validation_days"],
                "seed": request["seed"],
            },
            training_state=asdict(summary),
        )
        pending_result = {
            "checkpoint": _relative_or_absolute(checkpoint),
            "manifest": _relative_or_absolute(manifest),
            "training": asdict(summary),
            "validation": {"status": "pending"},
            "schema_hash": loader.schema_hash,
            "symbols": len(loader.universe),
            "frequencies": list(loader.frequencies),
            "validation_days": request["validation_days"],
        }
        jobs.update_job(
            job_id,
            progress=0.855,
            current="checkpoint",
            message="Training checkpoint saved; validation pending",
            result=pending_result,
        )
        if callable(add_log):
            add_log(job_id, f"CHECKPOINT training saved to {pending_result['checkpoint']}; validation pending")
        jobs.update_job(
            job_id,
            progress=0.86,
            current="validation",
            message="Running deterministic validation",
        )
        validation_dates = [
            date for date in dates
            if fold.validation.start <= date <= fold.validation.end
        ][: request["validation_days"]]
        validation_symbol = loader.universe[0]
        validation_environment = SingleSymbolTradingEnv(
            loader_for_symbol(validation_symbol),
            start=validation_dates[0],
            end=validation_dates[-1],
            execution_config=ExecutionConfig(),
            enforce_action_contract=True,
        )
        last_validation_log = -1

        def validation_progress(completed: int, total: int) -> None:
            nonlocal last_validation_log
            ratio = completed / max(1, total)
            jobs.update_job(
                job_id,
                progress=0.86 + 0.11 * ratio,
                current="validation",
                message=f"Validating {completed}/{total} steps",
            )
            if callable(add_log) and (
                completed == 1 or completed == total or completed - last_validation_log >= 25
            ):
                add_log(job_id, f"VALIDATION steps={completed}/{total}")
                last_validation_log = completed

        validation = evaluate_policy(
            validation_environment,
            model,
            device=request["device"],
            deterministic=True,
            cancel_check=lambda: jobs.is_cancel_requested(job_id),
            progress_callback=validation_progress,
        )
        checkpoint, manifest = save_agent_checkpoint(
            output,
            model=model,
            optimizer=trainer.optimizer,
            schema_hash=loader.schema_hash,
            universe=loader.universe,
            ppo_config=ppo_config,
            execution_config=ExecutionConfig(),
            experiment={
                "selected_fold": request["fold"],
                "splits": splits.payload(),
                "validation": validation.payload(),
                "episode_days": request["episode_days"],
                "validation_days": request["validation_days"],
                "seed": request["seed"],
            },
            training_state=asdict(summary),
        )
    except (TrainingCancelled, InterruptedError) as exc:
        jobs.update_job(
            job_id,
            status="cancelled",
            message="Agent training cancelled safely",
        )
        if callable(add_log):
            add_log(job_id, str(exc), level="warn")
        return {"cancelled": True}

    result = {
        "checkpoint": _relative_or_absolute(checkpoint),
        "manifest": _relative_or_absolute(manifest),
        "training": asdict(summary),
        "validation": validation.payload(),
        "schema_hash": loader.schema_hash,
        "symbols": len(loader.universe),
        "frequencies": list(loader.frequencies),
        "validation_days": request["validation_days"],
    }
    jobs.update_job(
        job_id,
        completed=request["total_steps"],
        succeeded=1,
        progress=0.98,
        current="checkpoint",
        message="Agent checkpoint and validation report saved",
        result=result,
    )
    if callable(add_log):
        add_log(job_id, f"OK checkpoint={result['checkpoint']}, validation_calmar={validation.metrics['calmar']:.6f}")
    return result


def _agent_request(payload: dict[str, Any]) -> dict[str, Any]:
    run_config = agent_run_config_from_payload(payload)
    store_path = resolve_project_path(payload.get("store_path") or payload.get("store") or run_config.store_path)
    symbols = _resolve_symbols(payload)
    device = run_config.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    output_raw = payload.get("output_path") or payload.get("output")
    return {
        "store_path": store_path,
        "symbols": symbols,
        "frequencies": run_config.frequencies,
        "fold": run_config.fold,
        "total_steps": run_config.total_steps,
        "episode_days": run_config.episode_days,
        "validation_days": run_config.validation_days,
        "split": asdict(run_config.split),
        "runtime": asdict(run_config.runtime),
        "use_agent_cache": run_config.use_agent_cache,
        "seed": run_config.seed,
        "device": device,
        "output_path": resolve_project_path(output_raw) if output_raw else None,
    }
