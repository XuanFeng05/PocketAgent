"""CPU strategy research API with durable request records and isolated run outputs."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from threading import Lock
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

import psutil

from backtest_layer.execution import ExecutionConfig
from research_layer.runner import run_experiment, preflight_experiment
from strategy_layer.registry import strategy_catalog
from app_layer.backend.jobs import JobManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime_layer" / "research"
_LIVE_SESSIONS: WeakValueDictionary = WeakValueDictionary()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    deadline = time.monotonic() + 2
    try:
        while True:
            try:
                temporary.replace(path)
                return
            except PermissionError:
                # Windows readers briefly hold a non-delete-sharing handle.
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    deadline = time.monotonic() + 2
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


class ResearchService:
    def __init__(self, jobs: JobManager, *, output_root: str | Path | None = None):
        self.jobs = jobs
        self.output_root = Path(output_root or DEFAULT_OUTPUT).resolve()
        self.requests_root = self.output_root / "requests"
        self.session_id = uuid4().hex
        self.owner = {"pid": os.getpid(), "process_started": psutil.Process().create_time(), "session": self.session_id}
        _LIVE_SESSIONS[self.session_id] = self
        self._start_lock = Lock()

    def spec(self) -> dict:
        return {
            "strategies": strategy_catalog(),
            "execution_defaults": asdict(ExecutionConfig()),
            "source_types": ["demo", "csv", "market"],
            "market_path": str(PROJECT_ROOT / "runtime_layer" / "data"),
            "output_root": str(self.output_root),
            "frequency": "daily",
            "modes": ["single", "batch", "walk_forward", "screen", "universe", "portfolio"],
            "portfolio_strategies": __import__("strategy_layer.portfolio", fromlist=["portfolio_strategy_catalog"]).portfolio_strategy_catalog(),
            "research_defaults": {"max_candidates": 64, "max_backtests": 512, "workers": 1,
                "max_bar_evaluations": 2_000_000,
                "walk_forward": {"train_bars": 40, "validation_bars": 15, "gap_bars": 1,
                                 "holdout_bars": 15, "expanding": False, "evaluate_holdout": False,
                                 "selection_metric": "total_return"}},
            "execution_timing": "收盘信号在下一交易日开盘尝试成交；日终以收盘价估值。",
        }

    def _request_path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", job_id):
            raise ValueError("Invalid research run identifier")
        return self.requests_root / f"{job_id}.json"

    def _normalize(self, payload: dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Research configuration must be a JSON object")
        source_type = str(payload.get("source_type") or "demo")
        if source_type not in {"demo", "csv", "market"}:
            raise ValueError("请选择演示、CSV 或本地行情数据源")
        strategy = str(payload.get("strategy") or "sma_trend")
        mode = str(payload.get("mode") or "single")
        if mode not in {"single", "batch", "walk_forward", "screen", "portfolio", "universe"}:
            raise ValueError("Unknown research mode")
        if mode != "portfolio" and strategy not in {s.get("id", s["name"]) for s in strategy_catalog()}:
            raise ValueError("Unknown strategy")
        parameters = payload.get("parameters") or {}
        execution = payload.get("execution") or {}
        if not isinstance(parameters, dict) or not isinstance(execution, dict):
            raise ValueError("parameters and execution must be JSON objects")
        try:
            normalized_execution = asdict(ExecutionConfig(**execution))
        except TypeError as exc:
            raise ValueError(f"Invalid execution configuration: {exc}") from exc
        source_path = None
        if source_type != "demo":
            raw = payload.get("source_path")
            if not raw:
                raise ValueError("请填写已有 CSV 或行情目录路径")
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(f"数据路径不存在：{path}")
            if source_type == "csv" and not path.is_file():
                raise ValueError("CSV 数据源必须是文件")
            source_path = str(path)
        dates = {}
        for field in ("start", "end"):
            raw = str(payload.get(field) or "").strip()
            if raw:
                dates[field] = datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
        if dates.get("start", "") > dates.get("end", "9999-12-31"):
            raise ValueError("开始日期不能晚于结束日期")
        config = {
            "mode": mode,
            "source_type": source_type,
            "source_path": source_path,
            "symbol": str(payload.get("symbol") or "").strip().upper() or None,
            "symbols": [str(value).strip().upper() for value in (payload.get("symbols") or []) if str(value).strip()],
            **dates,
            "strategy": strategy,
            "parameters": parameters,
            "execution": normalized_execution,
            "include_baselines": True,
        }
        if mode != "single":
            config.update({"parameter_grid": payload.get("parameter_grid", {}),
                           "max_candidates": payload.get("max_candidates", 64),
                           "max_backtests": payload.get("max_backtests", 512),
                           "max_bar_evaluations": payload.get("max_bar_evaluations", 2_000_000),
                           "selection_metric": payload.get("selection_metric", "total_return")})
        if mode == "walk_forward":
            config["walk_forward"] = payload.get("walk_forward", {})
        if mode == "screen":
            config["screen"] = payload.get("screen", {})
        if mode == "portfolio":
            config["portfolio"] = payload.get("portfolio", {})
        if mode == "universe":
            config["max_symbols"] = payload.get("max_symbols", 20)
        if payload.get("context") is not None:
            config["context"] = payload.get("context")
        if payload.get("robustness") is not None:
            config["robustness"] = payload.get("robustness")
        if payload.get("random_seed") is not None:
            config["random_seed"] = payload.get("random_seed")
        preflight_experiment(config)
        return config

    def preview(self, payload: dict) -> dict:
        return preflight_experiment(self._normalize(payload))

    def prepare_offline(self, payload: dict) -> dict:
        from research_layer.offline import prepare_task_bundle
        raw = payload.get("config") if isinstance(payload, dict) and "config" in payload else payload
        config = self._normalize(raw)
        destination = Path(str(payload.get("output") or "")).expanduser().resolve()
        if destination.suffix.lower() != ".zip":
            raise ValueError("离线任务需要 output ZIP 路径")
        return prepare_task_bundle(config, destination, shard_count=int(payload.get("shards", 1)))

    def aggregate_offline(self, payload: dict) -> dict:
        from research_layer.offline import aggregate_results
        return aggregate_results(payload["collection"], payload["plan"])

    def resume(self, payload: dict) -> dict:
        from research_layer.resume import resume_experiment
        return resume_experiment(payload["run"], payload.get("output"), retry_failed=bool(payload.get("retry_failed", False)))

    def start(self, payload: dict) -> dict:
        config = self._normalize(payload)
        with self._start_lock:
            if self.jobs.active_job("strategy_backtest"):
                raise ValueError("当前已有研究任务，请等待完成或先取消；笔记本默认一次运行一个实验。")
            job_id = self.jobs.create_job("strategy_backtest", title=f"策略研究 · {config['strategy']}")
            record = {
                "job_id": job_id, "status": "queued", "created_at": _now(),
                "updated_at": _now(), "config": config, "result_path": None,
                "owner": self.owner,
            }
            _write_json(self._request_path(job_id), record)
            self.jobs.submit(job_id, self._run, record)
        return {"job_id": job_id, "status": "queued"}

    def _run(self, record: dict) -> dict:
        path = self._request_path(record["job_id"])
        record = {**record, "status": "running", "updated_at": _now()}
        _write_json(path, record)
        def progress(update: dict) -> None:
            record.update(progress=update, updated_at=_now())
            if update.get("run_id"):
                record["run_id"] = update["run_id"]
            if update.get("path"):
                record["result_path"] = str(update["path"])
            _write_json(path, record)
            self.jobs.update_job(record["job_id"], message=str(update.get("message", "研究中")),
                                 completed=update.get("completed"), total=update.get("total"))
        try:
            result = run_experiment(record["config"], self.output_root / "runs",
                                    progress_callback=progress,
                                    cancel_check=lambda: self.jobs.is_cancel_requested(record["job_id"]))
            record.update(status="completed", updated_at=_now(), run_id=result["run_id"], result_path=str(result["path"]))
            _write_json(path, record)
            return {"job_id": record["job_id"], "run_id": result["run_id"], "path": result["path"]}
        except InterruptedError as exc:
            record.update(status="cancelled", updated_at=_now(), error=str(exc))
            _write_json(path, record)
            raise
        except Exception as exc:
            record.update(status="failed", updated_at=_now(), error=str(exc))
            _write_json(path, record)
            raise

    def cancel(self, job_id: str) -> dict:
        record = self.detail(job_id)
        if record["status"] not in {"queued", "running"}:
            return {"job_id": job_id, "status": record["status"]}
        if not self.jobs.request_cancel(job_id):
            raise ValueError("任务属于另一个运行实例；请在启动该任务的实例中取消。")
        return {"job_id": job_id, "status": "cancelling"}

    def list_runs(self) -> list[dict]:
        if not self.requests_root.exists():
            return []
        runs = []
        for path in self.requests_root.glob("*.json"):
            try:
                record = _read_json(path)
            except (OSError, ValueError):
                continue
            runs.append(self._reconcile(record))
        return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)

    def detail(self, job_id: str) -> dict:
        path = self._request_path(job_id)
        if not path.exists():
            raise FileNotFoundError("研究记录不存在")
        record = self._reconcile(_read_json(path))
        if record.get("status") in {"completed", "failed", "cancelled", "interrupted"} and record.get("result_path"):
            run_path = Path(record["result_path"]).resolve()
            if not run_path.is_relative_to(self.output_root / "runs"):
                raise ValueError("Result path is outside the research output directory")
            result_file = run_path / ("result.json" if record["status"] == "completed" else "partial_result.json")
            if result_file.exists():
                record["result"] = _read_json(result_file)
        return record

    def trial_report(self, job_id: str, candidate_id: str) -> dict:
        record = self.detail(job_id)
        result = record.get("result") or {}
        if (result.get("mode") or record["config"].get("mode")) != "batch":
            raise ValueError("候选详情端点用于批量参数实验；滚动验证请查看对应窗口报告。")
        trials = (result.get("batch") or {}).get("trials", result.get("trials", []))
        trial = next((item for item in trials if item.get("candidate_id") == candidate_id), None)
        if trial is None or trial.get("status") != "completed" or not trial.get("artifact_path"):
            raise FileNotFoundError("候选报告尚未完成或不存在")
        run_path = Path(record["result_path"]).resolve()
        report_path = (run_path / trial["artifact_path"] / "result.json").resolve()
        if not report_path.is_relative_to(run_path / "trials"):
            raise ValueError("Invalid candidate report path")
        report = _read_json(report_path)
        return {"strategy": report, "baselines": result.get("baselines", {}),
                "parameters": trial["parameters"], "candidate_id": candidate_id}

    def _reconcile(self, record: dict) -> dict:
        """Recover orphaned requests without reclassifying another live process."""
        if record.get("status") not in {"queued", "running"}:
            return record
        owner = record.get("owner") or {}
        live_session = _LIVE_SESSIONS.get(owner.get("session"))
        if live_session is not None:
            job = live_session.jobs.get_job(record["job_id"])
            if job is not None and job.get("status") in {"queued", "running"}:
                return record
            if job is not None and job.get("status") == "completed":
                # A reader may have opened the old file just before its atomic replacement.
                return _read_json(self._request_path(record["job_id"]))
            if job is not None and job.get("status") == "failed":
                record = {**record, "status": "failed", "updated_at": _now(),
                          "error": job.get("error") or "Research worker failed"}
                _write_json(self._request_path(record["job_id"]), record)
                return record
            if job is not None and job.get("status") == "cancelled":
                record = {**record, "status": "cancelled", "updated_at": _now()}
                _write_json(self._request_path(record["job_id"]), record)
                return record
        elif owner.get("pid") and owner.get("pid") != os.getpid():
            try:
                process = psutil.Process(int(owner["pid"]))
                if abs(process.create_time() - float(owner.get("process_started", -1))) < 0.01:
                    return record
            except psutil.AccessDenied:
                return record  # Unknown ownership is not evidence of a dead worker.
            except (psutil.NoSuchProcess, ValueError, TypeError):
                pass
        record = {**record, "status": "interrupted", "updated_at": _now(),
                  "error": "实验所属进程或会话已结束；原始配置已保留，可重新运行。"}
        _write_json(self._request_path(record["job_id"]), record)
        return record
