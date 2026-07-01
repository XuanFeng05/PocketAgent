from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch

from agent_layer.data import AgentTimelineLoader, open_agent_timeline_loader
from agent_layer.environment import SingleSymbolTradingEnv
from agent_layer.experiments import (
    AgentRunStore,
    ValidationQueue,
    ValidationTask,
    agent_run_config_from_payload,
    build_walk_forward_splits,
    load_agent_checkpoint,
)
from evaluation_layer.backtest import evaluate_policy


class AgentValidationWorker:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        self.store = AgentRunStore(self.run_dir.parent)
        self.queue = ValidationQueue(self.run_dir)
        self.config = agent_run_config_from_payload(self.store.read_config(self.run_id))
        self._loader: AgentTimelineLoader | None = None
        self._loader_symbols: tuple[str, ...] = ()

    def run(self) -> int:
        self.queue.update_status(
            status="starting",
            pid=os.getpid(),
            message="Validation worker started",
            error=None,
        )
        self.store.append_log(
            self.run_id,
            f"VALIDATOR pid={os.getpid()} started",
        )
        idle_checks = 0
        while True:
            task = self.queue.claim()
            if task is None:
                idle_checks += 1
                run_status = str(self.store.read_status(self.run_id).get("status") or "")
                if run_status in {"queued", "running", "paused"}:
                    self.queue.update_status(
                        status="idle",
                        pid=os.getpid(),
                        message="Validation worker is waiting for a checkpoint",
                    )
                    time.sleep(1.0)
                    continue
                if idle_checks < 4:
                    time.sleep(0.5)
                    continue
                break
            idle_checks = 0
            self._run_task(task)
        final_status = self.queue.read_status()
        if final_status.get("status") in {"completed", "failed"}:
            self.queue.update_status(pid=None)
        else:
            self.queue.update_status(
                status="idle",
                pid=None,
                task_id=None,
                kind=None,
                message="Validation queue is idle",
            )
        return 0

    def _run_task(self, task: ValidationTask) -> None:
        started_at = _now()
        try:
            if task.kind == "quick":
                _lower_process_priority()
            device = self._resolve_device(task.device)
            self.queue.update_status(
                status="running",
                pid=os.getpid(),
                task_id=task.task_id,
                kind=task.kind,
                progress=0.0,
                completed=0,
                total=0,
                message=f"{task.kind.title()} validation loading checkpoint",
                started_at=started_at,
                error=None,
            )
            base_loader = self._loader_for(task.symbols)
            model, checkpoint_payload = load_agent_checkpoint(
                task.checkpoint_path,
                expected_schema_hash=self.config.schema_hash,
                map_location="cpu",
            )
            training_universe = set(checkpoint_payload.get("universe") or ())
            if not set(task.symbols).issubset(training_universe):
                raise ValueError("Validation subset is not contained in the checkpoint universe.")

            plans: list[tuple[str, AgentTimelineLoader, list[Any], int]] = []
            total_steps = 0
            for symbol in task.symbols:
                symbol_loader = base_loader.for_universe((symbol,))
                dates = symbol_loader.trading_dates()
                splits = build_walk_forward_splits(dates)
                fold = splits.folds[self.config.fold - 1]
                validation_dates = [
                    date for date in dates
                    if fold.validation.start <= date <= fold.validation.end
                ][: task.days]
                if not validation_dates:
                    continue
                environment = SingleSymbolTradingEnv(
                    symbol_loader,
                    start=validation_dates[0],
                    end=validation_dates[-1],
                    execution_config=self.config.execution,
                    reward_scale=self.config.reward.scale,
                    drawdown_penalty=self.config.reward.drawdown_penalty,
                    turnover_penalty=self.config.reward.turnover_penalty,
                    invalid_action_penalty=self.config.reward.invalid_action_penalty,
                )
                plans.append((str(symbol).upper(), symbol_loader, validation_dates, len(environment.keys)))
                total_steps += len(environment.keys)
            if not plans:
                raise ValueError("Validation task contains no single-symbol trading dates.")

            completed_steps = 0

            def progress(completed: int, total: int) -> None:
                current_completed = completed_steps + completed
                current_total = max(1, total_steps)
                self.queue.update_status(
                    status="running",
                    pid=os.getpid(),
                    task_id=task.task_id,
                    kind=task.kind,
                    progress=current_completed / current_total,
                    completed=current_completed,
                    total=current_total,
                    message=f"{task.kind.title()} single-symbol validation {current_completed}/{current_total}",
                )
                if current_completed == 1 or current_completed == current_total or current_completed % 25 == 0:
                    self.store.append_metric(
                        self.run_id,
                        {
                            "kind": "validation_progress",
                            "time": _now(),
                            "validation_kind": task.kind,
                            "task_id": task.task_id,
                            "completed": current_completed,
                            "total": current_total,
                        },
                    )

            symbol_results: list[dict[str, Any]] = []
            data_performance = {"chunks": 0, "steps": 0, "load_seconds": 0.0}
            for symbol, symbol_loader, validation_dates, _ in plans:
                environment = SingleSymbolTradingEnv(
                    symbol_loader,
                    start=validation_dates[0],
                    end=validation_dates[-1],
                    execution_config=self.config.execution,
                    reward_scale=self.config.reward.scale,
                    drawdown_penalty=self.config.reward.drawdown_penalty,
                    turnover_penalty=self.config.reward.turnover_penalty,
                    invalid_action_penalty=self.config.reward.invalid_action_penalty,
                )
                item = evaluate_policy(
                    environment,
                    model,
                    device=device,
                    deterministic=True,
                    progress_callback=progress,
                ).payload()
                item["symbol"] = symbol
                symbol_results.append(item)
                completed_steps += int(item.get("steps") or 0)
                perf = symbol_loader.performance_payload()
                data_performance["chunks"] += int(perf.get("chunks") or 0)
                data_performance["steps"] += int(perf.get("steps") or 0)
                data_performance["load_seconds"] += float(perf.get("load_seconds") or 0.0)

            result = _aggregate_single_symbol_results(symbol_results)
            result["data_performance"] = {
                **data_performance,
                "seconds_per_step": data_performance["load_seconds"] / max(1, data_performance["steps"]),
            }
            completed_at = _now()
            result_path = self.queue.write_result(
                task,
                {
                    "status": "completed",
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "device": device,
                    "result": result,
                },
            )
            best_path = self._consider_best(task, result)
            self.store.append_metric(
                self.run_id,
                {
                    "kind": "validation",
                    "time": completed_at,
                    "validation_kind": task.kind,
                    "task_id": task.task_id,
                    "symbols": len(task.symbols),
                    "days": task.days,
                    **result,
                },
            )
            self._record_result(task, result, result_path, best_path)
            self.queue.update_status(
                status="completed",
                pid=os.getpid(),
                task_id=task.task_id,
                kind=task.kind,
                progress=1.0,
                completed=result.get("steps", 0),
                total=result.get("steps", 0),
                message=f"{task.kind.title()} validation completed",
                last_result=str(result_path),
                error=None,
            )
            self.store.append_log(
                self.run_id,
                f"VALIDATION kind={task.kind}, step={task.step}, result={result_path}",
            )
        except Exception as exc:
            result_path = self.queue.write_result(
                task,
                {
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": _now(),
                    "error": str(exc),
                },
            )
            self.queue.update_status(
                status="failed",
                pid=os.getpid(),
                task_id=task.task_id,
                kind=task.kind,
                message=f"Validation failed: {exc}",
                last_result=str(result_path),
                error=str(exc),
            )
            self.store.append_log(self.run_id, str(exc), level="error")
        finally:
            self.queue.cleanup_task_checkpoint(task)
            self.queue.complete(task)

    def _consider_best(
        self,
        task: ValidationTask,
        result: dict[str, Any],
    ) -> Path | None:
        metric_name = self.config.checkpoint.best_metric
        value = float((result.get("metrics") or {}).get(metric_name) or 0.0)
        status = self.queue.read_status()
        field = "best_final_metric" if task.kind == "final" else "best_quick_metric"
        current = status.get(field)
        if current is not None and value <= float(current):
            return None
        name = "best.pt" if task.kind == "final" else "best_quick.pt"
        best = self.run_dir / "checkpoints" / name
        temporary = best.with_suffix(".pt.tmp")
        shutil.copy2(task.checkpoint_path, temporary)
        temporary.replace(best)
        self.queue.update_status(**{field: value})
        destination = Path(self.config.output_dir) / self.run_id
        destination.mkdir(parents=True, exist_ok=True)
        published = destination / name
        published_temp = published.with_suffix(".pt.tmp")
        shutil.copy2(best, published_temp)
        published_temp.replace(published)
        if task.kind == "final":
            self.store.update_status(
                self.run_id,
                best_checkpoint=str(best),
                best_metric=value,
            )
        else:
            self.store.update_status(
                self.run_id,
                best_quick_checkpoint=str(best),
                best_quick_metric=value,
            )
        return best

    def _record_result(
        self,
        task: ValidationTask,
        result: dict[str, Any],
        result_path: Path,
        best_path: Path | None,
    ) -> None:
        status = self.store.read_status(self.run_id)
        run_result = dict(status.get("result") or {})
        key = "validation" if task.kind == "final" else "quick_validation"
        run_result[key] = result
        run_result[f"{key}_task"] = {
            "task_id": task.task_id,
            "checkpoint": task.source_checkpoint,
            "symbols": list(task.symbols),
            "days": task.days,
            "result_path": str(result_path),
            "best_checkpoint": str(best_path) if best_path else None,
        }
        self.store.update_status(self.run_id, result=run_result)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _loader_for(self, symbols: tuple[str, ...]) -> AgentTimelineLoader:
        if self._loader is None or self._loader_symbols != tuple(symbols):
            self._loader = open_agent_timeline_loader(
                self.config.store_path,
                universe=symbols,
                frequencies=self.config.frequencies or None,
                validate_store=False,
            )
            self._loader_symbols = tuple(symbols)
        return self._loader


def _aggregate_single_symbol_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one single-symbol validation result is required.")
    metric_keys = sorted({
        key
        for item in results
        for key, value in (item.get("metrics") or {}).items()
        if isinstance(value, (int, float))
    })
    metrics = {
        key: sum(float((item.get("metrics") or {}).get(key) or 0.0) for item in results) / len(results)
        for key in metric_keys
    }
    return {
        "metrics": metrics,
        "daily_nav": (),
        "orders": sum(int(item.get("orders") or 0) for item in results),
        "blocked_orders": sum(int(item.get("blocked_orders") or 0) for item in results),
        "steps": sum(int(item.get("steps") or 0) for item in results),
        "symbols": len(results),
        "symbol_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run queued PocketAgent validations.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    return AgentValidationWorker(args.run_dir).run()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lower_process_priority() -> None:
    try:
        if os.name == "nt":
            import ctypes

            below_normal_priority = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                below_normal_priority,
            )
        else:
            os.nice(5)
    except (AttributeError, OSError):
        pass
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
