from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import json
import os
from pathlib import Path
import time
from typing import Any

from agent_layer.environment import ExecutionConfig
from evaluation_layer.backtest.runner import run_policy_replay_to_files

from app_layer.backend.data_controller import _relative_or_absolute, resolve_project_path
from app_layer.backend.jobs import JobManager


DEFAULT_EVALUATION_MODEL_POOL = "runtime_layer/models/evaluation_pool"
DEFAULT_EVALUATION_RUN_DIR = "runtime_layer/reports/evaluation_runs"
DEFAULT_FEATURE_DATASET_DIR = "runtime_layer/reports/feature_dataset"


def list_evaluation_models() -> list[dict[str, Any]]:
    pool = resolve_project_path(DEFAULT_EVALUATION_MODEL_POOL)
    if not pool.exists():
        return []
    result: list[dict[str, Any]] = []
    for model_dir in sorted((p for p in pool.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        metadata = _read_json(model_dir / "model_metadata.json")
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            continue
        result.append({
            "name": model_dir.name,
            "path": _relative_or_absolute(model_dir),
            "model_path": _relative_or_absolute(model_path),
            "size_bytes": model_path.stat().st_size,
            "exported_at": metadata.get("exported_at"),
            "notes": metadata.get("notes"),
            "source_run_id": metadata.get("source_run_id"),
            "source_checkpoint": metadata.get("source_checkpoint"),
            "source_step": metadata.get("source_step"),
            "schema_hash": metadata.get("schema_hash"),
        })
    return result


def list_evaluation_runs() -> list[dict[str, Any]]:
    root = resolve_project_path(DEFAULT_EVALUATION_RUN_DIR)
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        status = _read_json(run_dir / "status.json")
        config = _read_json(run_dir / "config.json")
        summary = _read_json(run_dir / "summary.json")
        runs.append({
            "run_id": run_dir.name,
            "path": _relative_or_absolute(run_dir),
            "status": status.get("status") or "unknown",
            "progress": status.get("progress"),
            "job_id": status.get("job_id"),
            "model_name": config.get("model_name"),
            "symbol": config.get("symbol"),
            "start": config.get("start"),
            "end": config.get("end"),
            "updated_at": status.get("updated_at"),
            "total_return": (summary.get("metrics") or {}).get("total_return") if isinstance(summary, dict) else None,
            "maximum_drawdown": (summary.get("metrics") or {}).get("maximum_drawdown") if isinstance(summary, dict) else None,
            "total_actions": summary.get("total_actions") if isinstance(summary, dict) else None,
            "executed_trades": summary.get("executed_trades") if isinstance(summary, dict) else None,
            "blocked_actions": summary.get("blocked_actions") if isinstance(summary, dict) else None,
            "block_rate": summary.get("block_rate") if isinstance(summary, dict) else None,
        })
    return runs


def evaluation_run_detail(run_id: str, *, event_limit: int = 2000) -> dict[str, Any]:
    run_dir = _evaluation_run_dir(run_id)
    events = _read_events(run_dir / "events.jsonl", limit=event_limit)
    return {
        "run_id": run_id,
        "path": _relative_or_absolute(run_dir),
        "config": _read_json(run_dir / "config.json"),
        "status": _read_json(run_dir / "status.json"),
        "summary": _read_json(run_dir / "summary.json"),
        "events": events,
    }


def evaluation_run_events(run_id: str, *, after: int = 0, limit: int = 2000) -> list[dict[str, Any]]:
    run_dir = _evaluation_run_dir(run_id)
    return _read_events(run_dir / "events.jsonl", after=after, limit=limit)


def start_evaluation_run(payload: dict[str, Any], jobs: JobManager) -> dict[str, Any]:
    config = _evaluation_config(payload)
    run_id = _new_evaluation_run_id(config['symbol'])
    run_dir = resolve_project_path(DEFAULT_EVALUATION_RUN_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    job_id = jobs.create_job("evaluation_run", title=f"Evaluate {config['model_name']} / {config['symbol']}")
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "status.json", {
        "run_id": run_id,
        "job_id": job_id,
        "status": "queued",
        "phase": "queued",
        "progress": 0.0,
        "model_name": config["model_name"],
        "symbol": config["symbol"],
        "start": config["start"],
        "end": config["end"],
        "message": "Queued evaluation run",
        "created_at": _now(),
        "updated_at": _now(),
    })
    jobs.update_job(job_id, total=1, message="Queued evaluation run", result={"run_id": run_id})
    jobs.submit(job_id, _run_evaluation_job, job_id, run_id, run_dir, config, jobs)
    return {"run_id": run_id, "job_id": job_id, "status": "queued"}


def stop_evaluation_run(run_id: str, jobs: JobManager) -> dict[str, Any]:
    run_dir = _evaluation_run_dir(run_id)
    status = _read_json(run_dir / "status.json")
    current = str(status.get("status") or "").lower()
    if current in {"completed", "failed", "cancelled"}:
        return {"run_id": run_id, "status": current, "message": "Evaluation run is already terminal."}

    job_id = _find_evaluation_job_id(run_id, jobs)
    if job_id:
        jobs.request_cancel(job_id)
        next_status = {
            **status,
            "run_id": run_id,
            "job_id": job_id,
            "status": "running",
            "phase": "cancelling",
            "message": "Cancel requested",
            "updated_at": _now(),
        }
        _write_json(run_dir / "status.json", next_status)
        return {"run_id": run_id, "job_id": job_id, "status": "cancelling"}

    next_status = {
        **status,
        "run_id": run_id,
        "status": "cancelled",
        "phase": "cancelled",
        "message": "Evaluation cancelled; no live job was found.",
        "updated_at": _now(),
    }
    _write_json(run_dir / "status.json", next_status)
    return {"run_id": run_id, "status": "cancelled", "message": next_status["message"]}


def _find_evaluation_job_id(run_id: str, jobs: JobManager) -> str | None:
    for job in jobs.list_jobs():
        if job.get("type") != "evaluation_run":
            continue
        if job.get("status") not in {"queued", "running"}:
            continue
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        if result.get("run_id") == run_id:
            return str(job.get("job_id"))
    return None


def _run_evaluation_job(job_id: str, run_id: str, run_dir: Path, config: dict[str, Any], jobs: JobManager) -> dict[str, Any]:
    def progress(item: dict[str, Any]) -> None:
        payload = {
            "run_id": run_id,
            "job_id": job_id,
            "status": "running",
            "phase": "replaying",
            "progress": float(item.get("progress") or 0.0),
            "model_name": config["model_name"],
            "symbol": config["symbol"],
            "start": config["start"],
            "end": config["end"],
            "completed": item.get("completed"),
            "total": item.get("total"),
            "message": f"Replaying {item.get('completed', 0)}/{item.get('total', '-')}",
            "nav": item.get("nav"),
            "return": item.get("return"),
            "drawdown": item.get("drawdown"),
            "orders": item.get("orders"),
            "blocked_orders": item.get("blocked_orders"),
            "updated_at": _now(),
        }
        _write_json(run_dir / "status.json", payload)
        jobs.update_job(
            job_id,
            progress=payload["progress"],
            completed=payload["completed"],
            total=payload["total"],
            current="replaying",
            message=payload["message"],
            result={"run_id": run_id},
        )

    try:
        _write_json(run_dir / "status.json", {
            "run_id": run_id,
            "job_id": job_id,
            "status": "running",
            "phase": "loading",
            "progress": 0.01,
            "model_name": config["model_name"],
            "symbol": config["symbol"],
            "start": config["start"],
            "end": config["end"],
            "message": "Loading evaluation model and feature parts",
            "created_at": config["created_at"],
            "updated_at": _now(),
        })
        summary = run_policy_replay_to_files(
            run_dir=run_dir,
            model_path=resolve_project_path(config["model_path"]),
            dataset_dir=resolve_project_path(config["feature_dataset_dir"]),
            symbol=config["symbol"],
            start=config["start"],
            end=config["end"],
            execution_config=_execution_config(config),
            device=config.get("device") or "cpu",
            deterministic=True,
            stages=tuple(config["stages"]) if config.get("stages") else None,
            cancel_check=lambda: jobs.is_cancel_requested(job_id),
            progress_callback=progress,
        )
        final_status = {
            "run_id": run_id,
            "job_id": job_id,
            "status": "completed",
            "phase": "completed",
            "progress": 1.0,
            "model_name": config["model_name"],
            "symbol": config["symbol"],
            "start": config["start"],
            "end": config["end"],
            "message": "Evaluation completed",
            "completed": summary.get("steps"),
            "total": summary.get("steps"),
            "nav": (summary.get("metrics") or {}).get("final_nav"),
            "return": (summary.get("metrics") or {}).get("total_return"),
            "drawdown": (summary.get("metrics") or {}).get("maximum_drawdown"),
            "orders": summary.get("orders"),
            "blocked_orders": summary.get("blocked_orders"),
            "updated_at": _now(),
        }
        _write_json(run_dir / "status.json", final_status)
        return {"run_id": run_id, "summary": summary}
    except InterruptedError:
        _write_json(run_dir / "status.json", {
            "run_id": run_id,
            "job_id": job_id,
            "status": "cancelled",
            "phase": "cancelled",
            "progress": 0.0,
            "model_name": config["model_name"],
            "symbol": config["symbol"],
            "start": config["start"],
            "end": config["end"],
            "message": "Evaluation cancelled",
            "updated_at": _now(),
        })
        raise
    except Exception as exc:
        _write_json(run_dir / "status.json", {
            "run_id": run_id,
            "job_id": job_id,
            "status": "failed",
            "phase": "failed",
            "progress": 0.0,
            "model_name": config["model_name"],
            "symbol": config["symbol"],
            "start": config["start"],
            "end": config["end"],
            "message": f"Evaluation failed: {exc}",
            "error": str(exc),
            "updated_at": _now(),
        })
        raise


def _new_evaluation_run_id(symbol: str) -> str:
    base_symbol = str(symbol).replace('.', '_')
    root = resolve_project_path(DEFAULT_EVALUATION_RUN_DIR)
    for _ in range(10):
        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base_symbol}_{uuid4().hex[:6]}"
        if not (root / run_id).exists():
            return run_id
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base_symbol}_{uuid4().hex}"


def _evaluation_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_name = str(payload.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("Select an exported evaluation model.")
    if Path(model_name).name != model_name or model_name in {".", ".."}:
        raise ValueError("Invalid evaluation model name.")
    model_dir = resolve_project_path(DEFAULT_EVALUATION_MODEL_POOL) / model_name
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Evaluation model not found: {model_name}")
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Enter a symbol.")
    start = str(payload.get("start") or "").strip()
    end = str(payload.get("end") or "").strip()
    if not start or not end:
        raise ValueError("Start and end dates are required.")
    return {
        "model_name": model_name,
        "model_path": _relative_or_absolute(model_path),
        "feature_dataset_dir": str(payload.get("feature_dataset_dir") or DEFAULT_FEATURE_DATASET_DIR),
        "symbol": symbol,
        "start": start,
        "end": end,
        "device": str(payload.get("device") or "cpu"),
        "stages": _stages(payload.get("stages")),
        "created_at": _now(),
        "execution": {
            "initial_cash": _number(payload.get("initial_cash"), 1_000_000.0),
            "lot_size": int(_number(payload.get("lot_size"), 100)),
            "max_position_ratio": _number(payload.get("max_position_ratio"), 0.20),
            "commission_rate": _number(payload.get("commission_rate"), 0.0003),
            "minimum_commission": _number(payload.get("minimum_commission"), 5.0),
            "stamp_duty_rate": _number(payload.get("stamp_duty_rate"), 0.0005),
            "transfer_fee_rate": _number(payload.get("transfer_fee_rate"), 0.00001),
            "base_slippage_rate": _number(payload.get("base_slippage_rate"), 0.0002),
        },
    }


def _execution_config(config: dict[str, Any]) -> ExecutionConfig:
    values = dict(config.get("execution") or {})
    return ExecutionConfig(**values)


def _stages(value: Any) -> list[str] | None:
    if value in (None, "", "all"):
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    result = [item for item in items if item]
    return result or None


def _evaluation_run_dir(run_id: str) -> Path:
    safe = Path(str(run_id)).name
    run_dir = resolve_project_path(DEFAULT_EVALUATION_RUN_DIR) / safe
    if not run_dir.exists():
        raise FileNotFoundError(f"Evaluation run not found: {run_id}")
    return run_dir


def _read_events(path: Path, *, after: int = 0, limit: int = 2000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(event.get("seq") or 0) <= after:
                continue
            result.append(event)
            if len(result) >= limit:
                break
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    deadline = time.monotonic() + 2.0
    try:
        while True:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (PermissionError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.025)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
