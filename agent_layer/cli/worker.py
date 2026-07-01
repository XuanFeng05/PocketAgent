from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch

from agent_layer.data import AgentTimelineLoader, open_agent_timeline_loader
from agent_layer.environment import SingleSymbolTradingEnv
from agent_layer.experiments import (
    AgentRunStore,
    ValidationQueue,
    agent_run_config_from_payload,
    build_walk_forward_splits,
    load_agent_checkpoint,
    save_agent_checkpoint,
)
from agent_layer.models import SingleSymbolMultiFrequencyPolicy, SingleSymbolMultiFrequencyPolicyConfig
from agent_layer.training import PPOTrainer, TrainingCancelled, TrainingSummary


class AgentTrainingWorker:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        self.store = AgentRunStore(self.run_dir.parent)
        self.config = agent_run_config_from_payload(self.store.read_config(self.run_id))
        self.started = perf_counter()
        self.current_steps = 0
        self.current_updates = 0
        self.current_episodes = 0
        self.safe_steps = 0
        self.safe_updates = 0
        self.safe_episodes = 0
        self.last_step_metric = -1
        self.last_status_write_at = 0.0
        self.last_metric_write_at = 0.0
        self.last_hardware_write_at = 0.0
        self.last_cache_status_at = 0.0
        self.last_progress_phase: str | None = None
        self.cached_hardware: dict[str, float | None] = _empty_hardware_snapshot()
        self.latest_checkpoint: Path | None = None
        self.rng = np.random.default_rng(self.config.seed)
        self.validation_queue = ValidationQueue(self.run_dir)

    def run(self) -> int:
        model: SingleSymbolMultiFrequencyPolicy | None = None
        trainer: PPOTrainer | None = None
        splits_payload: dict[str, Any] | None = None
        try:
            self._status(
                status="running",
                phase="preflight",
                message="Worker validating immutable run configuration",
                pid=os.getpid(),
                heartbeat=_now(),
            )
            self._log(
                f"WORKER pid={os.getpid()}, config={self.config.config_hash}, "
                f"device={self.config.device}, steps={self.config.total_steps}"
            )
            self._seed()

            self._status(
                status="running",
                phase="loading_feature_parts",
                message="Loading Feature Parts contract and symbol shards",
                heartbeat=_now(),
            )

            def cache_progress(item: dict[str, object]) -> None:
                completed = int(item.get("completed") or 0)
                total = int(item.get("total") or 0)
                cached = bool(item.get("cached"))
                phase_name = str(item.get("phase") or "data_cache")
                label = "Agent decision cache" if phase_name == "agent_cache" else "Market feature cache"
                message = (
                    f"{label} ready"
                    if cached or total == 0
                    else str(item.get("message") or f"Preparing {label.lower()} {completed}/{total}")
                )
                now_perf = perf_counter()
                force_status = bool(total and (completed == total or completed == 1))
                if force_status or now_perf - self.last_cache_status_at >= 2.0:
                    self._status(
                        status="running",
                        phase="preparing_data",
                        message=message,
                        data_cache_completed=completed,
                        data_cache_total=total,
                        heartbeat=_now(),
                    )
                    self.last_cache_status_at = now_perf
                if total and (completed == total or completed == 1 or completed % 25 == 0):
                    prefix = "AGENT CACHE" if phase_name == "agent_cache" else "DATA CACHE"
                    self._log(f"{prefix} {completed}/{total}")

            loader = open_agent_timeline_loader(
                self.config.store_path,
                universe=self.config.resolved_symbols or None,
                frequencies=self.config.frequencies or None,
                validate_store=False,
                stream_chunk_size=max(8, 64 // max(1, self.config.parallel_envs)),
                use_market_cache=self.config.use_agent_cache,
                use_decision_cache=self.config.use_agent_cache,
                market_cache_progress=cache_progress,
                decision_cache_progress=cache_progress,
            )
            if self.config.schema_hash and loader.schema_hash != self.config.schema_hash:
                raise ValueError("Feature Parts schema changed after preflight.")
            self._status(
                status="running",
                phase="building_splits",
                message="Building walk-forward training and validation splits",
                heartbeat=_now(),
            )
            dates = loader.trading_dates()
            splits = build_walk_forward_splits(dates)
            splits_payload = splits.payload()
            fold = splits.folds[self.config.fold - 1]
            train_dates = [date for date in dates if fold.train.start <= date <= fold.train.end]
            if not train_dates:
                raise ValueError("Run split contains no usable trading dates.")

            self._status(
                status="running",
                phase="building_model",
                message="Building single-symbol multi-frequency PPO model and optimizer",
                heartbeat=_now(),
            )
            model, resume_payload = self._model(loader)
            trainer = PPOTrainer(model, config=self.config.ppo, device=self.config.device)
            if resume_payload:
                optimizer_state = resume_payload.get("optimizer_state_dict")
                if optimizer_state:
                    trainer.optimizer.load_state_dict(optimizer_state)
                self._restore_runtime_state(resume_payload.get("runtime_state") or {})
                training_state = resume_payload.get("training_state") or {}
                self.current_steps = int(training_state.get("total_steps") or 0)
                self.current_updates = int(training_state.get("updates") or 0)
                self.current_episodes = int(training_state.get("episodes") or 0)
                self.safe_steps = self.current_steps
                self.safe_updates = self.current_updates
                self.safe_episodes = self.current_episodes
                self._log(
                    f"RESUME checkpoint={self.latest_checkpoint}, steps={self.current_steps}, "
                    f"updates={self.current_updates}"
                )

            symbol_loaders: dict[str, AgentTimelineLoader] = {}

            def loader_for_symbol(symbol: str) -> AgentTimelineLoader:
                normalized = str(symbol).upper()
                selected = symbol_loaders.get(normalized)
                if selected is None:
                    selected = loader.for_universe((normalized,))
                    symbol_loaders[normalized] = selected
                return selected

            def environment_factory() -> SingleSymbolTradingEnv:
                # Train a shared single-stock operator: every episode samples one
                # symbol shard and one contiguous date window.  Parallel envs still
                # diversify rollout collection, but they no longer share a
                # market-wide multi-symbol account.
                if not self.config.resolved_symbols:
                    raise ValueError("Resolved training symbols cannot be empty.")
                days = min(self.config.episode_days, len(train_dates))
                attempts = max(4, len(self.config.resolved_symbols) * 2)
                last_error: Exception | None = None
                for _ in range(attempts):
                    symbol = str(self.rng.choice(self.config.resolved_symbols)).upper()
                    start_index = int(self.rng.integers(0, len(train_dates) - days + 1))
                    try:
                        return SingleSymbolTradingEnv(
                            loader_for_symbol(symbol),
                            start=train_dates[start_index],
                            end=train_dates[start_index + days - 1],
                            execution_config=self.config.execution,
                            reward_scale=self.config.reward.scale,
                            hurdle_rate_annual=self.config.reward.hurdle_rate_annual,
                            drawdown_penalty=self.config.reward.drawdown_penalty,
                            turnover_penalty=self.config.reward.turnover_penalty,
                            invalid_action_penalty=self.config.reward.invalid_action_penalty,
                        )
                    except ValueError as exc:
                        last_error = exc
                raise ValueError(
                    "Unable to create a single-symbol training episode from the selected universe."
                ) from last_error

            def progress(item: dict[str, object]) -> None:
                now_perf = perf_counter()
                self.current_steps = int(float(item.get("steps") or self.current_steps))
                self.current_updates = int(float(item.get("updates") or self.current_updates))
                self.current_episodes = int(float(item.get("episodes") or self.current_episodes))
                phase = str(item.get("phase") or "training")
                if phase == "updated":
                    self.safe_steps = self.current_steps
                    self.safe_updates = self.current_updates
                    self.safe_episodes = self.current_episodes
                elapsed = max(now_perf - self.started, 1e-9)
                rate = self.current_steps / elapsed
                remaining = max(0, self.config.total_steps - self.current_steps)
                phase_changed = phase != self.last_progress_phase
                self.last_progress_phase = phase
                important_phase_change = phase_changed and phase not in {"collecting", "updated"}
                should_status = (
                    self.current_steps <= 1
                    or important_phase_change
                    or now_perf - self.last_status_write_at >= 5.0
                )
                should_record = (
                    self.current_steps <= 1
                    or now_perf - self.last_metric_write_at >= self._metric_interval_seconds(elapsed)
                )
                hardware = self._hardware_snapshot_cached(now_perf, force=should_record)
                status_updates = {
                    "status": "running",
                    "phase": phase,
                    "message": f"{phase} {self.current_steps}/{self.config.total_steps}",
                    "progress": self.current_steps / self.config.total_steps,
                    "steps": self.current_steps,
                    "updates": self.current_updates,
                    "episodes": self.current_episodes,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": rate,
                    "eta_seconds": remaining / rate if rate > 0 else None,
                    "parallel_envs": int(float(item.get("parallel_envs") or self.config.parallel_envs)),
                    "heartbeat": _now(),
                    **_training_diagnostics(item),
                    **hardware,
                }
                if should_status:
                    self.store.update_status(self.run_id, **status_updates)
                    self.last_status_write_at = now_perf
                if should_record:
                    self.store.append_metric(
                        self.run_id,
                        {"kind": "training", "time": _now(), **item, **status_updates},
                    )
                    self.last_step_metric = self.current_steps
                    self.last_metric_write_at = now_perf

            def after_update(item: dict[str, object]) -> None:
                update = int(float(item.get("updates") or 0))
                control = self._control()
                interval = self.config.checkpoint.validation_interval_updates
                validation_due = bool(interval and update % interval == 0)
                checkpoint_due = (
                    update % self.config.checkpoint.checkpoint_interval_updates == 0
                    or control == "checkpoint"
                    or validation_due
                )
                checkpoint = None
                if checkpoint_due:
                    checkpoint = self._save_checkpoint(
                        model,
                        trainer,
                        splits=splits_payload,
                        validation={"status": "pending"},
                        reason=(
                            "manual" if control == "checkpoint"
                            else "quick_validation" if validation_due
                            else "periodic"
                        ),
                    )
                    if control == "checkpoint":
                        self.store.clear_control(self.run_id)
                if validation_due and checkpoint is not None:
                    self._queue_validation(
                        checkpoint,
                        kind="quick",
                        days=min(
                            self.config.validation.quick_days,
                            self.config.validation_days,
                        ),
                        device=self.config.validation.periodic_device,
                    )

            self._status(
                status="running",
                phase="starting_rollout",
                message="Starting rollout collection",
                heartbeat=_now(),
            )
            # Validation workers are launched lazily when a checkpoint actually needs validation.
            # Cache construction is a one-time preparation phase. Training speed
            # and ETA should start when rollout collection actually begins.
            self.started = perf_counter()
            summary = trainer.train(
                environment_factory,
                total_steps=self.config.total_steps,
                progress_callback=progress,
                cancel_check=self._cancel_requested,
                initial_steps=self.current_steps,
                initial_updates=self.current_updates,
                update_callback=after_update,
                parallel_environments=self.config.parallel_envs,
            )
            self.current_steps = summary.total_steps
            self.current_updates = summary.updates
            self.current_episodes = summary.episodes
            self.safe_steps = summary.total_steps
            self.safe_updates = summary.updates
            self.safe_episodes = summary.episodes
            final_checkpoint = self._save_checkpoint(
                model,
                trainer,
                splits=splits_payload,
                validation={"status": "pending"},
                reason="training_complete",
                summary=summary,
            )
            self._queue_validation(
                final_checkpoint,
                kind="final",
                days=self.config.validation_days,
                device=self.config.validation.final_device,
                launch=False,
            )
            self._publish_models()
            self._status(
                status="completed",
                phase="training_completed",
                message="Training completed; final validation runs independently",
                progress=1.0,
                pid=None,
                result={
                    "checkpoint": str(final_checkpoint),
                    "training": asdict(summary),
                    "validation": {"status": "queued"},
                },
                heartbeat=_now(),
            )
            trainer = None
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._ensure_validation_worker()
            self._log(f"TRAINING COMPLETE checkpoint={final_checkpoint}; final validation queued")
            return 0
        except TrainingCancelled:
            action = self._control()
            status = "paused" if action == "pause" else "stopped"
            if model is not None and trainer is not None and splits_payload is not None:
                self._save_checkpoint(
                    model,
                    trainer,
                    splits=splits_payload,
                    validation={"status": "pending"},
                    reason=f"{status}_safe_boundary",
                )
            self._status(
                status=status,
                phase=status,
                message=f"Run {status} at the last completed PPO update",
                steps=self.safe_steps,
                updates=self.safe_updates,
                episodes=self.safe_episodes,
                progress=self.safe_steps / self.config.total_steps,
                pid=None,
                heartbeat=_now(),
            )
            self._log(f"CONTROL action={action or 'stop'}, status={status}", level="warn")
            return 0
        except Exception as exc:
            self._status(
                status="failed",
                phase="failed",
                message=f"Run failed: {exc}",
                error=str(exc),
                pid=None,
                heartbeat=_now(),
            )
            self._log(str(exc), level="error")
            raise


    def _metric_interval_seconds(self, elapsed_seconds: float) -> float:
        if elapsed_seconds >= 3 * 3600:
            return 300.0
        if elapsed_seconds >= 30 * 60:
            return 60.0
        return 30.0

    def _hardware_snapshot_cached(self, now_perf: float, *, force: bool = False) -> dict[str, float | None]:
        interval = 60.0 if (now_perf - self.started) >= 30 * 60 else 30.0
        if force or now_perf - self.last_hardware_write_at >= interval:
            self.cached_hardware = _hardware_snapshot(self.config.device)
            self.last_hardware_write_at = now_perf
        return dict(self.cached_hardware)

    def _model(
        self,
        loader: AgentTimelineLoader,
    ) -> tuple[SingleSymbolMultiFrequencyPolicy, dict[str, Any] | None]:
        status = self.store.read_status(self.run_id)
        checkpoint_value = status.get("latest_checkpoint")
        if checkpoint_value and Path(str(checkpoint_value)).exists():
            self.latest_checkpoint = Path(str(checkpoint_value))
            return load_agent_checkpoint(
                self.latest_checkpoint,
                expected_schema_hash=loader.schema_hash,
                expected_universe=loader.universe,
                map_location="cpu",
            )
        model = SingleSymbolMultiFrequencyPolicy(
            SingleSymbolMultiFrequencyPolicyConfig(
                frequency_channels={
                    freq: len(loader.feature_names[freq]) for freq in loader.frequencies
                },
                decision_context_size=len(loader.decision_context_names),
                runtime_state_size=len(loader.runtime_contract),
                **asdict(self.config.model),
            )
        )
        return model, None

    def _save_checkpoint(
        self,
        model: SingleSymbolMultiFrequencyPolicy,
        trainer: PPOTrainer,
        *,
        splits: dict[str, Any],
        validation: dict[str, Any],
        reason: str,
        summary: TrainingSummary | None = None,
    ) -> Path:
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint = checkpoint_dir / f"step_{self.safe_steps:09d}.pt"
        training_state = asdict(summary) if summary else {
            "total_steps": self.safe_steps,
            "updates": self.safe_updates,
            "episodes": self.safe_episodes,
        }
        save_agent_checkpoint(
            checkpoint,
            model=model,
            optimizer=trainer.optimizer,
            schema_hash=self.config.schema_hash,
            universe=self.config.resolved_symbols,
            ppo_config=self.config.ppo,
            execution_config=self.config.execution,
            experiment={
                "run_id": self.run_id,
                "config_hash": self.config.config_hash,
                "selected_fold": self.config.fold,
                "splits": splits,
                "validation": validation,
                "checkpoint_reason": reason,
            },
            training_state=training_state,
            runtime_state=self._runtime_state(),
        )
        latest = checkpoint_dir / "latest.pt"
        temporary = checkpoint_dir / "latest.pt.tmp"
        shutil.copy2(checkpoint, temporary)
        temporary.replace(latest)
        self.latest_checkpoint = latest
        self._prune_checkpoints(checkpoint_dir)
        self._status(
            latest_checkpoint=str(latest),
            phase="checkpointing",
            message=f"Checkpoint saved at step {self.safe_steps} ({reason})",
            heartbeat=_now(),
        )
        self._log(f"CHECKPOINT reason={reason}, safe_step={self.safe_steps}, path={checkpoint}")
        return checkpoint

    def _prune_checkpoints(self, checkpoint_dir: Path) -> None:
        numbered = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda path: path.stat().st_mtime)
        for path in numbered[: -self.config.checkpoint.keep_last]:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)

    def _publish_models(self) -> None:
        destination = Path(self.config.output_dir) / self.run_id
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("latest.pt", "best.pt", "best_quick.pt"):
            source = self.run_dir / "checkpoints" / name
            if source.exists():
                temporary = destination / f"{name}.tmp"
                shutil.copy2(source, temporary)
                temporary.replace(destination / name)
        self._status(published_model_dir=str(destination))

    def _queue_validation(
        self,
        checkpoint: Path,
        *,
        kind: str,
        days: int,
        device: str,
        launch: bool = True,
    ) -> None:
        task = self.validation_queue.create_task(
            kind=kind,
            checkpoint=checkpoint,
            step=self.safe_steps,
            updates=self.safe_updates,
            days=days,
            symbols=(
                self.config.resolved_validation_symbols
                or self.config.resolved_symbols
            ),
            device=device,
        )
        self.validation_queue.submit(task)
        self._log(
            f"VALIDATION QUEUED kind={kind}, task={task.task_id}, "
            f"symbols={len(task.symbols)}, days={days}, device={device}"
        )
        if launch:
            self._ensure_validation_worker()

    def _ensure_validation_worker(self) -> None:
        self.validation_queue.ensure_worker()

    def _runtime_state(self) -> dict[str, Any]:
        return {
            "numpy_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def _restore_runtime_state(self, state: dict[str, Any]) -> None:
        if state.get("numpy_rng"):
            self.rng.bit_generator.state = state["numpy_rng"]
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def _seed(self) -> None:
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _control(self) -> str | None:
        return self.store.read_control(self.run_id).get("action")

    def _cancel_requested(self) -> bool:
        return self._control() in {"pause", "stop"}

    def _status(self, **updates: Any) -> None:
        self.store.update_status(self.run_id, **updates)

    def _log(self, message: str, *, level: str = "info") -> None:
        self.store.append_log(self.run_id, message, level=level)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one persistent PocketAgent training run.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    return AgentTrainingWorker(args.run_dir).run()


def _empty_hardware_snapshot() -> dict[str, float | None]:
    return {
        "cpu_percent": None,
        "memory_percent": None,
        "gpu_percent": None,
        "gpu_memory_percent": None,
    }


_TRAINING_DIAGNOSTIC_KEYS = {
    "reward",
    "nav",
    "cash",
    "buy_actions",
    "hold_actions",
    "sell_actions",
    "executed_orders",
    "blocked_orders",
    "blocked_reasons",
    "turnover_value",
    "total_fees",
    "model_seconds",
    "environment_seconds",
    "data_load_seconds",
    "reset_seconds",
    "data_chunks",
    "data_steps",
    "collect_samples",
    "collect_seconds",
    "collect_steps_per_second",
    "collect_model_seconds",
    "collect_environment_seconds",
    "collect_data_load_seconds",
    "collect_reset_seconds",
    "collect_overhead_seconds",
    "optimize_seconds",
    "samples_per_update",
    "optimize_steps_per_second",
    "optimizer_steps",
    "optimizer_steps_per_second",
    "policy_loss",
    "value_loss",
    "entropy",
    "clip_fraction",
    "gradient_norm",
    "initialized_env_seconds",
}


def _training_diagnostics(item: dict[str, object]) -> dict[str, object]:
    return {key: item[key] for key in _TRAINING_DIAGNOSTIC_KEYS if key in item}


def _hardware_snapshot(device: str) -> dict[str, float | None]:
    payload = _empty_hardware_snapshot()
    try:
        import psutil  # type: ignore

        payload["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        payload["memory_percent"] = float(psutil.virtual_memory().percent)
    except Exception:
        pass
    if str(device).lower() != "cuda":
        return payload
    try:
        nvidia_smi_kwargs: dict[str, Any] = {
            "stderr": subprocess.DEVNULL,
            "text": True,
            "timeout": 1.0,
        }
        if os.name == "nt":
            nvidia_smi_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            **nvidia_smi_kwargs,
        )
        first = output.strip().splitlines()[0]
        gpu_util, memory_used, memory_total = [float(part.strip()) for part in first.split(",")[:3]]
        payload["gpu_percent"] = gpu_util
        payload["gpu_memory_percent"] = (memory_used / memory_total * 100.0) if memory_total else None
    except Exception:
        try:
            if torch.cuda.is_available():
                used = float(torch.cuda.memory_allocated())
                total = float(torch.cuda.get_device_properties(0).total_memory)
                payload["gpu_memory_percent"] = (used / total * 100.0) if total else None
        except Exception:
            pass
    return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
