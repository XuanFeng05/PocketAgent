from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
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
    save_agent_checkpoint,
    select_representative_symbols,
    select_symbols,
)
from agent_layer.models import SingleSymbolMultiFrequencyPolicy, SingleSymbolMultiFrequencyPolicyConfig
from agent_layer.training import PPOConfig, PPOTrainer, TrainingCancelled
from evaluation_layer.backtest import evaluate_policy
from agent_layer.data.feature_parts import FeaturePartsDataset, validate_feature_parts_dataset

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




def start_agent_cache_job(payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    active = jobs.active_job("agent_cache_build")
    if active:
        raise ValueError(f"An Agent cache build job is already active: {active['job_id']}")
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
    if callable(add_log):
        add_log(
            job_id,
            f"CONFIG feature_dir={request['feature_dir']}, output_dir={request['output_dir']}, workers={request['workers']}",
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
        output_dir=request["output_dir"],
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
    summary = build_agent_cache(
        config,
        progress_callback=progress,
        cancel_check=lambda: jobs.is_cancel_requested(job_id),
    )
    inspected = inspect_agent_cache(config.output_dir) if summary.get("ok") else {}
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
        payload.get("output_dir") or payload.get("cache_dir") or "runtime_layer/agent_cache/latest"
    )
    symbols = _resolve_symbols(payload)
    frequencies = tuple(str(value) for value in (payload.get("frequencies") or ())) or None
    stages = tuple(str(value).strip() for value in (payload.get("stages") or []) if str(value).strip()) or None
    max_decisions_raw = payload.get("max_decisions_per_symbol")
    max_decisions = int(max_decisions_raw or 0)
    return {
        "feature_dir": feature_dir,
        "output_dir": output_dir,
        "symbols": tuple(symbols) if symbols else None,
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
        runtime_state_size=10,
    )
    smoke_config = agent_run_config_from_payload({})
    formal_config = formal_run_config()
    return {
        "algorithm": "Single-symbol multi-frequency PPO",
        "encoder": "Independent two-layer LSTM per frequency with masked attention pooling",
        "actor": "Shared single-stock SELL/HOLD/BUY categorical head plus Beta size head",
        "critic": "Single-symbol value critic over the sampled stock episode",
        "model": asdict(model),
        "frequency_options": [
            "5min", "15min", "30min", "60min", "daily", "weekly", "monthly",
        ],
        "ppo": asdict(PPOConfig()),
        "execution": asdict(ExecutionConfig()),
        "defaults": {
            "store": DEFAULT_FEATURE_DATASET_DIR,
            "output_dir": DEFAULT_AGENT_MODEL_DIR,
            "symbols_file": DEFAULT_SMOKE_SYMBOLS,
            "fold": 3,
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
        "torch_version": torch.__version__,
        "profiles": {
            "smoke": smoke_config.payload(),
            "formal": formal_config.payload(),
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
        feature_input, resolution = _resolve_agent_feature_input(source_dataset)
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


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}

def _resolve_agent_training_input(raw_path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    requested = resolve_project_path(raw_path or DEFAULT_FEATURE_DATASET_DIR)
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
        return validate_agent_cache_dataset(path, universe=universe, frequencies=frequencies)
    return validate_feature_parts_dataset(path, sample_limit=2)

def _agent_validation_error_messages(validation: dict[str, Any]) -> list[str]:
    return [
        str(item.get("message") or item.get("name") or "Agent Feature Parts validation failed")
        for item in validation.get("checks", [])
        if item.get("status") == "error"
    ]


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
    feature_input_path, input_resolution = _resolve_agent_training_input(config.store_path)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    validation = _validate_agent_training_input(feature_input_path, input_resolution, config=config)
    validation["resolution"] = input_resolution
    if not validation["ok"]:
        messages = _agent_validation_error_messages(validation)
        messages.append(
            "Agent now prefers feature_parts directly. If this fails after changing Feature indicators/model_input/monthly settings, rebuild Feature Dataset so the per-symbol parts contain the current schema."
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
    splits = build_walk_forward_splits(loader.trading_dates())
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
    stdout_path = run_dir / "worker.stdout.log"
    stderr_path = run_dir / "worker.stderr.log"
    command = [sys.executable, "-m", "agent_layer.cli.worker", "--run-dir", str(run_dir)]
    popen_kwargs: dict[str, Any] = {
        "cwd": str(resolve_project_path(".")),
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
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
        message="Independent training worker launched",
        pid=process.pid,
        error=None,
    )


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
            "agent_layer.cli.worker" in normalized_command
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
    feature_input_path, input_resolution = _resolve_agent_training_input(request["store_path"])
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
        dates = loader.trading_dates()
        splits = build_walk_forward_splits(dates)
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
    feature_input_path, input_resolution = _resolve_agent_training_input(request["store_path"])
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
        splits = build_walk_forward_splits(dates)
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
    store_path = resolve_project_path(payload.get("store_path") or payload.get("store"), DEFAULT_FEATURE_DATASET_DIR)
    symbols = _resolve_symbols(payload)
    fold = int(payload.get("fold") or 3)
    if not 1 <= fold <= 3:
        raise ValueError("Agent fold must be 1, 2, or 3.")
    total_steps = int(payload.get("total_steps") or 5_000_000)
    episode_days = int(payload.get("episode_days") or 252)
    validation_days = int(payload.get("validation_days") or 126)
    if total_steps <= 0 or episode_days <= 0 or validation_days <= 0:
        raise ValueError(
            "Training steps, episode days, and validation days must be positive."
        )
    device = str(payload.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Agent device must be cpu or cuda.")
    output_raw = payload.get("output_path") or payload.get("output")
    seed_raw = payload.get("seed")
    return {
        "store_path": store_path,
        "symbols": symbols,
        "frequencies": tuple(payload.get("frequencies") or DEFAULT_AGENT_FREQUENCIES),
        "fold": fold,
        "total_steps": total_steps,
        "episode_days": episode_days,
        "validation_days": validation_days,
        "seed": int(42 if seed_raw is None or seed_raw == "" else seed_raw),
        "device": device,
        "output_path": resolve_project_path(output_raw) if output_raw else None,
    }
