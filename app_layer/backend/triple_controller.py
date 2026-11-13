"""Local dashboard service for the fixed Triple Screen research workflow.

This service only reads the existing project market shards. It never downloads,
refreshes, or installs data; the page exposes the currently available local
symbol inventory and records every request under a separate dashboard root.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any
from uuid import uuid4

from app_layer.backend.jobs import JobManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "runtime_layer" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime_layer" / "reports" / "triple_dashboard"
VALID_MODES = {"single", "batch", "annual", "study"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def local_symbols(data_root: str | Path = DEFAULT_DATA_ROOT) -> list[str]:
    root = Path(data_root).resolve()
    parts = root / "market_parts"
    values = []
    for path in parts.glob("bucket=*/*/daily_pre.parquet"):
        name = path.parent.name.replace("_", ".").upper()
        if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", name):
            values.append(name)
    return sorted(set(values))


class TripleDashboardService:
    def __init__(self, jobs: JobManager, *, output_root: str | Path | None = None,
                 data_root: str | Path = DEFAULT_DATA_ROOT):
        self.jobs = jobs
        self.data_root = Path(data_root).resolve()
        self.output_root = Path(output_root or DEFAULT_OUTPUT).resolve()
        self.requests_root = self.output_root / "requests"
        self._lock = Lock()
        self.owner = {"pid": __import__("os").getpid(), "session": uuid4().hex}

    def spec(self) -> dict:
        symbols = local_symbols(self.data_root)
        return {
            "mode": "triple_screen",
            "data_root": str(self.data_root),
            "price_basis": "pre",
            "symbols": symbols,
            "symbol_count": len(symbols),
            "default_start": "2022-01-04",
            "default_end": "2024-12-31",
            "default_costs_bps": [10],
            "default_annual_windows": [
                {"label": str(year), "start": f"{year}-01-01", "end": f"{year}-12-31"}
                for year in (2022, 2023, 2024, 2025)
            ],
            "rules": {
                "version": "triple_screen_pre_v1",
                "weekly": "MACD histogram slope (12/26/9) on completed weekly bars",
                "daily": "EFI2 below zero",
                "trigger": "break above prior completed daily high",
            },
            "modes": [
                {"id": "single", "label": "单次三重滤网"},
                {"id": "batch", "label": "股票批量回测"},
                {"id": "annual", "label": "年度验证"},
                {"id": "study", "label": "参数与滤网对照"},
            ],
            "limitations": [
                "只读取本地已有前复权数据，不下载数据。",
                "当前页面股票清单来自本地100股 daily_pre 分片。",
                "结果是独立单股标准化净值，不是共享现金组合。",
            ],
        }

    def _path(self, job_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,140}", job_id):
            raise ValueError("Invalid Triple Screen job identifier")
        return self.requests_root / f"{job_id}.json"

    def _normalize(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Triple Screen request must be an object")
        mode = str(payload.get("mode") or "batch")
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown Triple Screen mode: {mode}")
        available = set(local_symbols(self.data_root))
        raw_symbols = payload.get("symbols") or sorted(available)
        if not isinstance(raw_symbols, list) or not raw_symbols:
            raise ValueError("请选择至少一只本地股票")
        symbols = [str(value).strip().upper() for value in raw_symbols]
        if len(symbols) > 100 or len(set(symbols)) != len(symbols):
            raise ValueError("当前页面最多选择100只股票，且不能重复")
        unknown = sorted(set(symbols) - available)
        if unknown:
            raise ValueError(f"所选股票不在本地前复权清单中：{', '.join(unknown[:8])}")
        if mode == "single" and len(symbols) != 1:
            raise ValueError("单次模式只能选择一只股票")
        start = str(payload.get("start") or "2022-01-04")
        end = str(payload.get("end") or "2024-12-31")
        try:
            if datetime.fromisoformat(start) > datetime.fromisoformat(end):
                raise ValueError("开始日期不能晚于结束日期")
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("日期必须为 YYYY-MM-DD 且开始不晚于结束") from exc
        costs = payload.get("costs_bps") or [10]
        if not isinstance(costs, list) or not costs or len(costs) > 4:
            raise ValueError("成本情景需要1到4个数值")
        try:
            costs = [float(value) for value in costs]
        except (TypeError, ValueError) as exc:
            raise ValueError("成本情景必须为数值") from exc
        if any(value < 0 or value >= 10000 for value in costs) or len(set(costs)) != len(costs):
            raise ValueError("成本情景必须不重复且在0到10000bps之间")
        settings = deepcopy(payload.get("settings") or {})
        allowed_settings = {"macd_fast", "macd_slow", "macd_signal", "force_span", "warmup_weeks"}
        if set(settings) - allowed_settings:
            raise ValueError("只允许修改三重滤网参数，不允许修改版本或执行口径")
        windows = payload.get("windows")
        if mode == "annual":
            windows = windows or self.spec()["default_annual_windows"]
            if not isinstance(windows, list) or not windows:
                raise ValueError("年度验证需要时间窗口")
        if mode == "study":
            cases = payload.get("cases")
            if not isinstance(cases, list) or not cases:
                raise ValueError("参数与滤网对照需要案例清单")
            if len(cases) > 16:
                raise ValueError("最多16个对照案例")
        result = {
            "schema_version": 1, "mode": mode, "source_type": "market",
            "source_path": str(self.data_root), "adjust": "pre", "symbols": symbols,
            "start": start, "end": end, "settings": settings,
            "cost_scenarios_bps": costs, "research_scope": "exploratory",
            "variant": str(payload.get("variant") or "full"),
        }
        if mode == "annual":
            result["windows"] = windows
        if mode == "study":
            result["cases"] = cases
            result["max_tasks"] = int(payload.get("max_tasks") or 2000)
        return result

    def start(self, payload: dict) -> dict:
        config = self._normalize(payload)
        with self._lock:
            active = self.jobs.active_job("triple_screen")
            if active:
                raise ValueError("当前已有三重滤网任务，请等待完成")
            job_id = self.jobs.create_job("triple_screen", title=f"三重滤网 · {config['mode']}")
            request = {"job_id": job_id, "status": "queued", "created_at": _now(),
                       "config": config, "owner": self.owner, "result_path": None}
            _write(self._path(job_id), request)
            self.jobs.submit(job_id, self._run, request)
        return {"job_id": job_id, "status": "queued", "mode": config["mode"], "symbol_count": len(config["symbols"])}

    def _run(self, request: dict) -> dict:
        from research_layer.triple_runner import run_triple_experiment
        from research_layer.triple_study import run_triple_study
        from research_layer.triple_validation import run_triple_validation

        job_id = request["job_id"]
        path = self._path(job_id)
        request = {**request, "status": "running", "updated_at": _now()}
        _write(path, request)
        mode = request["config"]["mode"]
        try:
            child_config = {**request["config"], "mode": "triple_screen",
                            "research_scope": "exploratory"}
            if mode in {"single", "batch"}:
                result = run_triple_experiment(child_config, self.output_root / "runs")
                try:
                    from research_layer.triple_report import render_triple_report
                    render_triple_report(result["path"])
                except Exception:
                    # The saved JSON/CSV result remains authoritative if the
                    # optional derived HTML view cannot be rendered.
                    pass
            elif mode == "annual":
                config = {key: value for key, value in request["config"].items() if key not in {"mode", "start", "end", "windows", "variant"}}
                config.update({"mode": "triple_screen", "research_scope": "exploratory_validation"})
                result = run_triple_validation(config, request["config"]["windows"], self.output_root / "validation")
            else:
                config = {key: value for key, value in request["config"].items() if key not in {"mode", "cases", "variant"}}
                config.update({"mode": "triple_screen", "research_scope": "exploratory_study"})
                result = run_triple_study(config, request["config"]["cases"], self.output_root / "studies")
            request.update(status="completed", updated_at=_now(), run_id=result["run_id"], result_path=result["path"])
            _write(path, request)
            return {"job_id": job_id, "run_id": result["run_id"], "path": result["path"], "mode": mode}
        except Exception as exc:
            request.update(status="failed", updated_at=_now(), error=str(exc))
            _write(path, request)
            raise

    def list_runs(self) -> list[dict]:
        if not self.requests_root.exists():
            return []
        output = []
        for path in self.requests_root.glob("*.json"):
            try:
                output.append(_read(path))
            except (OSError, ValueError):
                continue
        return sorted(output, key=lambda item: item.get("created_at", ""), reverse=True)

    def detail(self, job_id: str) -> dict:
        record = _read(self._path(job_id))
        result_path = record.get("result_path")
        if record.get("status") == "completed" and result_path:
            root = Path(result_path).resolve()
            if not root.is_relative_to(self.output_root.resolve()):
                raise ValueError("Triple Screen result path is outside the dashboard output")
            summary = root / "summary.json"
            result = root / "result.json"
            if summary.exists():
                record["summary"] = _read(summary)
            elif result.exists():
                record["result"] = _read(result)
            report = root / "report.html"
            if report.exists():
                record["report_path"] = str(report)
                record["report_url"] = f"/api/triple/runs/{job_id}/report"
        return record

    def report(self, job_id: str) -> Path:
        record = _read(self._path(job_id))
        root = Path(record.get("result_path") or "").resolve()
        if not root.is_relative_to(self.output_root.resolve()):
            raise ValueError("Triple Screen result path is outside the dashboard output")
        path = root / "report.html"
        if not path.is_file():
            raise FileNotFoundError("该任务没有离线 HTML 报告")
        return path

    def cancel(self, job_id: str) -> dict:
        if not self.jobs.request_cancel(job_id):
            raise ValueError("三重滤网任务不存在")
        return {"job_id": job_id, "status": "cancelling"}
