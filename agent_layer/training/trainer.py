from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Any

import numpy as np
import torch
from torch import Tensor, nn

from agent_layer.actions import PortfolioAction
from agent_layer.environment import AgentObservation, AshareTradingEnv
from agent_layer.models import (
    MultiFrequencyMAPPO,
    market_steps_to_tensors,
    observation_state_to_tensors,
    observation_to_tensors,
)
from agent_layer.training.mappo import MAPPOConfig, compute_gae, compute_mappo_loss
from agent_layer.training.distributed import DistributedTrainingContext


@dataclass(frozen=True)
class TrainingSummary:
    total_steps: int
    updates: int
    episodes: int
    mean_reward: float
    final_loss: float
    final_approximate_kl: float
    final_policy_loss: float = 0.0
    final_value_loss: float = 0.0
    final_entropy: float = 0.0
    final_clip_fraction: float = 0.0
    final_gradient_norm: float = 0.0


@dataclass(frozen=True)
class TrainingUpdateMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    optimizer_steps: int


@dataclass
class RolloutDiagnostics:
    samples: int = 0
    model_seconds: float = 0.0
    environment_seconds: float = 0.0
    data_load_seconds: float = 0.0
    reset_seconds: float = 0.0
    chunk_count: int = 0
    data_steps: int = 0
    buy_actions: int = 0
    hold_actions: int = 0
    sell_actions: int = 0
    executed_orders: int = 0
    blocked_orders: int = 0
    turnover_value: float = 0.0
    blocked_reasons: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.blocked_reasons is None:
            self.blocked_reasons = {}

    def add_action_counts(self, action: PortfolioAction) -> None:
        self.buy_actions += int(np.count_nonzero(action.directions > 0))
        self.hold_actions += int(np.count_nonzero(action.directions == 0))
        self.sell_actions += int(np.count_nonzero(action.directions < 0))

    def add_execution(self, execution: object | None) -> None:
        if not execution:
            return
        self.executed_orders += len(execution.executed_fills)
        blocked = [fill for fill in execution.fills if fill.status == "blocked"]
        self.blocked_orders += len(blocked)
        for fill in blocked:
            reason = str(fill.reason or "unknown")
            assert self.blocked_reasons is not None
            self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
        self.turnover_value += float(execution.turnover_value)

    def payload(self, *, collect_seconds: float | None = None) -> dict[str, object]:
        samples = max(1, int(self.samples))
        payload: dict[str, object] = {
            "collect_samples": int(self.samples),
            "model_seconds": self.model_seconds / samples,
            "environment_seconds": self.environment_seconds / samples,
            "data_load_seconds": self.data_load_seconds / samples,
            "reset_seconds": self.reset_seconds / samples,
            "collect_model_seconds": self.model_seconds,
            "collect_environment_seconds": self.environment_seconds,
            "collect_data_load_seconds": self.data_load_seconds,
            "collect_reset_seconds": self.reset_seconds,
            "data_chunks": int(self.chunk_count),
            "data_steps": int(self.data_steps),
            "buy_actions": int(self.buy_actions),
            "hold_actions": int(self.hold_actions),
            "sell_actions": int(self.sell_actions),
            "executed_orders": int(self.executed_orders),
            "blocked_orders": int(self.blocked_orders),
            "blocked_reasons": dict(self.blocked_reasons or {}),
            "turnover_value": float(self.turnover_value),
        }
        if collect_seconds is not None:
            seconds = max(float(collect_seconds), 1e-9)
            payload.update(
                {
                    "collect_seconds": seconds,
                    "collect_steps_per_second": float(self.samples) / seconds,
                    "collect_overhead_seconds": max(
                        0.0,
                        seconds - self.model_seconds - self.environment_seconds - self.reset_seconds,
                    ),
                }
            )
        return payload


class TrainingCancelled(InterruptedError):
    pass


class _MAPPOUpdateModule(nn.Module):
    def __init__(self, policy: MultiFrequencyMAPPO) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        *,
        directions: Tensor,
        sizes: Tensor,
        **observation_inputs: object,
    ) -> object:
        return self.policy.evaluate_actions(
            directions=directions,
            sizes=sizes,
            **observation_inputs,
        )


class MAPPOTrainer:
    def __init__(
        self,
        model: MultiFrequencyMAPPO,
        *,
        config: MAPPOConfig | None = None,
        device: str | torch.device = "cpu",
        distributed: DistributedTrainingContext | None = None,
    ) -> None:
        self.model = model
        self.config = config or MAPPOConfig()
        self.distributed = distributed or DistributedTrainingContext(device=torch.device(device))
        self.device = self.distributed.device
        self.model.to(self.device)
        self.update_model = self.distributed.wrap(_MAPPOUpdateModule(self.model))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
            weight_decay=1e-4,
        )

    def train(
        self,
        environment_factory: Callable[[], AshareTradingEnv],
        *,
        total_steps: int,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        initial_steps: int = 0,
        initial_updates: int = 0,
        update_callback: Callable[[dict[str, object]], None] | None = None,
        parallel_environments: int = 1,
        env_backend: str = "in_process",
        vector_env_factory: Callable[[int], Any] | None = None,
    ) -> TrainingSummary:
        if total_steps <= 0:
            raise ValueError("Training steps must be positive.")
        parallel_environments = max(1, int(parallel_environments))
        backend = str(env_backend or "in_process").strip().lower()
        if backend == "subprocess":
            if self.distributed.enabled:
                raise ValueError("Subprocess environment backend is not supported with DDP yet.")
            if vector_env_factory is None:
                raise ValueError("Subprocess environment backend requires a vector_env_factory.")
            return self._train_vector_parallel(
                vector_env_factory,
                total_steps=total_steps,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                initial_steps=initial_steps,
                initial_updates=initial_updates,
                update_callback=update_callback,
                parallel_environments=parallel_environments,
            )
        if backend != "in_process":
            raise ValueError(f"Unsupported environment backend: {env_backend}")
        if parallel_environments > 1 or self.distributed.enabled:
            return self._train_parallel(
                environment_factory,
                total_steps=total_steps,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                initial_steps=initial_steps,
                initial_updates=initial_updates,
                update_callback=update_callback,
                parallel_environments=parallel_environments,
            )
        if progress_callback:
            progress_callback(
                {
                    "phase": "initializing_envs",
                    "steps": float(max(0, int(initial_steps))),
                    "total_steps": float(total_steps),
                    "progress": max(0, int(initial_steps)) / total_steps,
                    "updates": float(max(0, int(initial_updates))),
                    "episodes": 0.0,
                    "loss": 0.0,
                    "approximate_kl": 0.0,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "parallel_envs": 1.0,
                }
            )
        initialization_started = perf_counter()
        environment = environment_factory()
        observation, _ = environment.reset()
        initialization_seconds = perf_counter() - initialization_started
        if progress_callback:
            progress_callback(
                {
                    "phase": "initializing_envs",
                    "steps": float(max(0, int(initial_steps))),
                    "total_steps": float(total_steps),
                    "progress": max(0, int(initial_steps)) / total_steps,
                    "updates": float(max(0, int(initial_updates))),
                    "episodes": 0.0,
                    "loss": 0.0,
                    "approximate_kl": 0.0,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "parallel_envs": 1.0,
                    "initialized_envs": 1.0,
                    "initialized_env_seconds": initialization_seconds,
                }
            )
        completed_steps = max(0, int(initial_steps))
        updates = max(0, int(initial_updates))
        episodes = 0
        reward_sum = 0.0
        reward_count = 0
        final_loss = 0.0
        final_kl = 0.0
        final_update = TrainingUpdateMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        while completed_steps < total_steps:
            if cancel_check and cancel_check():
                raise TrainingCancelled("Agent training cancellation requested.")
            rollout_steps = min(self.config.rollout_steps, total_steps - completed_steps)
            collect_started = perf_counter()
            rollout, observation, environment, ended_episodes = self._collect_rollout(
                environment,
                observation,
                environment_factory,
                rollout_steps,
                cancel_check,
                step_callback=(
                    lambda collected, telemetry: progress_callback(
                        {
                            "phase": "collecting",
                            "steps": float(completed_steps + collected),
                            "total_steps": float(total_steps),
                            "progress": (completed_steps + collected) / total_steps,
                            "updates": float(updates),
                            "episodes": float(episodes),
                            "loss": final_loss,
                            "approximate_kl": final_kl,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            **telemetry,
                        }
                    )
                    if progress_callback
                    else None
                ),
            )
            collect_seconds = perf_counter() - collect_started
            collect_diagnostics = dict(rollout.get("_diagnostics") or {})
            collect_diagnostics.update(
                {
                    "collect_seconds": collect_seconds,
                    "collect_steps_per_second": rollout_steps / max(collect_seconds, 1e-9),
                    "collect_overhead_seconds": _collect_overhead_seconds(
                        collect_seconds, collect_diagnostics
                    ),
                }
            )
            episodes += ended_episodes
            reward_tensor = rollout["reward"]
            reward_sum += float(reward_tensor.sum().item())
            reward_count += int(reward_tensor.numel())
            if progress_callback:
                progress_callback(
                    {
                        "phase": "optimizing",
                        "steps": float(completed_steps + rollout_steps),
                        "total_steps": float(total_steps),
                        "progress": (completed_steps + rollout_steps) / total_steps,
                        "updates": float(updates),
                        "episodes": float(episodes),
                        "loss": final_loss,
                        "approximate_kl": final_kl,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        **collect_diagnostics,
                    }
                )
            optimize_started = perf_counter()
            final_update = self._update(rollout)
            optimize_seconds = perf_counter() - optimize_started
            final_loss = final_update.total_loss
            final_kl = final_update.approximate_kl
            completed_steps += rollout_steps
            updates += 1
            progress = completed_steps / total_steps
            learning_rate = self.config.learning_rate + progress * (
                self.config.final_learning_rate - self.config.learning_rate
            )
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            update_payload = {
                "phase": "updated",
                "steps": float(completed_steps),
                "total_steps": float(total_steps),
                "progress": progress,
                "updates": float(updates),
                "episodes": float(episodes),
                "loss": final_loss,
                "approximate_kl": final_kl,
                "learning_rate": learning_rate,
                "policy_loss": final_update.policy_loss,
                "value_loss": final_update.value_loss,
                "entropy": final_update.entropy,
                "clip_fraction": final_update.clip_fraction,
                "gradient_norm": final_update.gradient_norm,
                "optimizer_steps": float(final_update.optimizer_steps),
                "optimize_seconds": optimize_seconds,
                "samples_per_update": float(rollout_steps),
                "optimize_steps_per_second": rollout_steps / max(optimize_seconds, 1e-9),
                "optimizer_steps_per_second": final_update.optimizer_steps / max(optimize_seconds, 1e-9),
                **collect_diagnostics,
            }
            if progress_callback:
                progress_callback(update_payload)
            if update_callback:
                update_callback(update_payload)

        return TrainingSummary(
            total_steps=completed_steps,
            updates=updates,
            episodes=episodes,
            mean_reward=reward_sum / max(1, reward_count),
            final_loss=final_loss,
            final_approximate_kl=final_kl,
            final_policy_loss=final_update.policy_loss,
            final_value_loss=final_update.value_loss,
            final_entropy=final_update.entropy,
            final_clip_fraction=final_update.clip_fraction,
            final_gradient_norm=final_update.gradient_norm,
        )


    def _train_vector_parallel(
        self,
        vector_env_factory: Callable[[int], Any],
        *,
        total_steps: int,
        progress_callback: Callable[[dict[str, object]], None] | None,
        cancel_check: Callable[[], bool] | None,
        initial_steps: int,
        initial_updates: int,
        update_callback: Callable[[dict[str, object]], None] | None,
        parallel_environments: int,
    ) -> TrainingSummary:
        completed_steps = max(0, int(initial_steps))
        updates = max(0, int(initial_updates))
        episodes = 0
        reward_sum = 0.0
        reward_count = 0
        final_loss = 0.0
        final_kl = 0.0
        final_update = TrainingUpdateMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        vector_env = None
        try:
            if progress_callback:
                progress_callback(
                    {
                        "phase": "initializing_envs",
                        "steps": float(completed_steps),
                        "total_steps": float(total_steps),
                        "progress": completed_steps / total_steps,
                        "updates": float(updates),
                        "episodes": float(episodes),
                        "loss": final_loss,
                        "approximate_kl": final_kl,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "parallel_envs": float(parallel_environments),
                        "parallel_envs_per_rank": float(parallel_environments),
                        "world_size": 1.0,
                        "env_backend": "subprocess",
                        "initialized_envs": 0.0,
                        "initialized_env_seconds": 0.0,
                    }
                )
            initialization_started = perf_counter()
            vector_env = vector_env_factory(parallel_environments)
            observations, _infos, init_telemetry = vector_env.reset()
            initialized_env_seconds = perf_counter() - initialization_started
            if progress_callback:
                progress_callback(
                    {
                        "phase": "initializing_envs",
                        "steps": float(completed_steps),
                        "total_steps": float(total_steps),
                        "progress": completed_steps / total_steps,
                        "updates": float(updates),
                        "episodes": float(episodes),
                        "loss": final_loss,
                        "approximate_kl": final_kl,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "parallel_envs": float(parallel_environments),
                        "parallel_envs_per_rank": float(parallel_environments),
                        "world_size": 1.0,
                        "env_backend": "subprocess",
                        "initialized_envs": float(parallel_environments),
                        "initialized_env_seconds": initialized_env_seconds,
                        **{k: v for k, v in dict(init_telemetry or {}).items() if k not in {"initialized_env_seconds"}},
                    }
                )

            while completed_steps < total_steps:
                if cancel_check and cancel_check():
                    raise TrainingCancelled("Agent training cancellation requested.")
                remaining = max(1, total_steps - completed_steps)
                active_envs = min(parallel_environments, remaining)
                if active_envs != parallel_environments:
                    # Keep subprocess vector env stable; finish using at least one full vector step.
                    active_envs = parallel_environments
                per_env_steps = min(self.config.rollout_steps, max(1, remaining // parallel_environments))
                collect_started = perf_counter()
                rollout, observations, ended_episodes, samples = self._collect_rollout_vector(
                    vector_env,
                    observations,
                    per_env_steps,
                    cancel_check,
                    step_callback=(
                        lambda collected, telemetry: progress_callback(
                            {
                                "phase": "collecting",
                                "steps": float(min(total_steps, completed_steps + collected)),
                                "total_steps": float(total_steps),
                                "progress": min(1.0, (completed_steps + collected) / total_steps),
                                "updates": float(updates),
                                "episodes": float(episodes),
                                "loss": final_loss,
                                "approximate_kl": final_kl,
                                "learning_rate": self.optimizer.param_groups[0]["lr"],
                                "parallel_envs": float(parallel_environments),
                                "parallel_envs_per_rank": float(parallel_environments),
                                "world_size": 1.0,
                                "env_backend": "subprocess",
                                **telemetry,
                            }
                        )
                        if progress_callback
                        else None
                    ),
                )
                collect_seconds = perf_counter() - collect_started
                collect_diagnostics = dict(rollout.get("_diagnostics") or {})
                collect_diagnostics.update(
                    {
                        "collect_seconds": collect_seconds,
                        "collect_steps_per_second": samples / max(collect_seconds, 1e-9),
                        "collect_overhead_seconds": _collect_overhead_seconds(
                            collect_seconds, collect_diagnostics
                        ),
                        "env_backend": "subprocess",
                    }
                )
                episodes += int(ended_episodes)
                reward_tensor = rollout["reward"]
                reward_sum += float(reward_tensor.sum().item())
                reward_count += int(reward_tensor.numel())
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "optimizing",
                            "steps": float(min(total_steps, completed_steps + samples)),
                            "total_steps": float(total_steps),
                            "progress": min(1.0, (completed_steps + samples) / total_steps),
                            "updates": float(updates),
                            "episodes": float(episodes),
                            "loss": final_loss,
                            "approximate_kl": final_kl,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "parallel_envs": float(parallel_environments),
                            "parallel_envs_per_rank": float(parallel_environments),
                            "world_size": 1.0,
                            **collect_diagnostics,
                        }
                    )
                optimize_started = perf_counter()
                final_update = self._update(rollout)
                optimize_seconds = perf_counter() - optimize_started
                final_loss = final_update.total_loss
                final_kl = final_update.approximate_kl
                completed_steps += samples
                updates += 1
                progress = min(1.0, completed_steps / total_steps)
                learning_rate = self.config.learning_rate + progress * (
                    self.config.final_learning_rate - self.config.learning_rate
                )
                for group in self.optimizer.param_groups:
                    group["lr"] = learning_rate
                update_payload = {
                    "phase": "updated",
                    "steps": float(completed_steps),
                    "total_steps": float(total_steps),
                    "progress": progress,
                    "updates": float(updates),
                    "episodes": float(episodes),
                    "loss": final_loss,
                    "approximate_kl": final_kl,
                    "learning_rate": learning_rate,
                    "policy_loss": final_update.policy_loss,
                    "value_loss": final_update.value_loss,
                    "entropy": final_update.entropy,
                    "clip_fraction": final_update.clip_fraction,
                    "gradient_norm": final_update.gradient_norm,
                    "optimizer_steps": float(final_update.optimizer_steps),
                    "parallel_envs": float(parallel_environments),
                    "parallel_envs_per_rank": float(parallel_environments),
                    "world_size": 1.0,
                    "env_backend": "subprocess",
                    "optimize_seconds": optimize_seconds,
                    "samples_per_update": float(samples),
                    "local_samples_per_update": float(samples),
                    "optimize_steps_per_second": samples / max(optimize_seconds, 1e-9),
                    "optimizer_steps_per_second": final_update.optimizer_steps / max(optimize_seconds, 1e-9),
                    **collect_diagnostics,
                }
                if progress_callback:
                    progress_callback(update_payload)
                if update_callback:
                    update_callback(update_payload)

            return TrainingSummary(
                total_steps=completed_steps,
                updates=updates,
                episodes=episodes,
                mean_reward=reward_sum / max(1, reward_count),
                final_loss=final_loss,
                final_approximate_kl=final_kl,
                final_policy_loss=final_update.policy_loss,
                final_value_loss=final_update.value_loss,
                final_entropy=final_update.entropy,
                final_clip_fraction=final_update.clip_fraction,
                final_gradient_norm=final_update.gradient_norm,
            )
        finally:
            close = getattr(vector_env, "close", None)
            if callable(close):
                close()

    def _train_parallel(
        self,
        environment_factory: Callable[[], AshareTradingEnv],
        *,
        total_steps: int,
        progress_callback: Callable[[dict[str, object]], None] | None,
        cancel_check: Callable[[], bool] | None,
        initial_steps: int,
        initial_updates: int,
        update_callback: Callable[[dict[str, object]], None] | None,
        parallel_environments: int,
    ) -> TrainingSummary:
        completed_steps = max(0, int(initial_steps))
        remaining_steps = total_steps - completed_steps
        if remaining_steps % self.distributed.world_size != 0:
            raise ValueError(
                "Remaining training steps must be divisible by the distributed world size: "
                f"{remaining_steps} % {self.distributed.world_size} != 0."
            )
        updates = max(0, int(initial_updates))
        episodes = 0
        reward_sum = 0.0
        reward_count = 0
        final_loss = 0.0
        final_kl = 0.0
        final_update = TrainingUpdateMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        environments = []
        observations = []
        initialized_env_seconds = 0.0
        for env_index in range(parallel_environments):
            if progress_callback:
                progress_callback(
                    {
                        "phase": "initializing_envs",
                        "steps": float(completed_steps),
                        "total_steps": float(total_steps),
                        "progress": completed_steps / total_steps,
                        "updates": float(updates),
                        "episodes": float(episodes),
                        "loss": final_loss,
                        "approximate_kl": final_kl,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "parallel_envs": float(parallel_environments),
                        "parallel_envs_per_rank": float(parallel_environments),
                        "world_size": float(self.distributed.world_size),
                        "initialized_envs": float(env_index),
                        "initialized_env_seconds": initialized_env_seconds,
                    }
                )
            initialization_started = perf_counter()
            environment = environment_factory()
            observation, _ = environment.reset()
            initialized_env_seconds += perf_counter() - initialization_started
            environments.append(environment)
            observations.append(observation)
        if progress_callback:
            progress_callback(
                {
                    "phase": "initializing_envs",
                    "steps": float(completed_steps),
                    "total_steps": float(total_steps),
                    "progress": completed_steps / total_steps,
                    "updates": float(updates),
                    "episodes": float(episodes),
                    "loss": final_loss,
                    "approximate_kl": final_kl,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "parallel_envs": float(parallel_environments),
                    "parallel_envs_per_rank": float(parallel_environments),
                    "world_size": float(self.distributed.world_size),
                    "initialized_envs": float(parallel_environments),
                    "initialized_env_seconds": initialized_env_seconds,
                }
            )

        while completed_steps < total_steps:
            if cancel_check and cancel_check():
                raise TrainingCancelled("Agent training cancellation requested.")
            remaining = max(1, total_steps - completed_steps)
            local_remaining = remaining // self.distributed.world_size
            active_envs = min(parallel_environments, local_remaining)
            per_env_steps = min(
                self.config.rollout_steps,
                max(1, local_remaining // active_envs),
            )
            collect_started = perf_counter()
            rollout, next_observations, next_environments, ended_episodes, samples = (
                self._collect_rollout_parallel(
                    environments[:active_envs],
                    observations[:active_envs],
                    environment_factory,
                    per_env_steps,
                    cancel_check,
                    step_callback=(
                        lambda collected, telemetry: progress_callback(
                            {
                                "phase": "collecting",
                                "steps": float(
                                    completed_steps + collected * self.distributed.world_size
                                ),
                                "total_steps": float(total_steps),
                                "progress": min(
                                    1.0,
                                    (completed_steps + collected * self.distributed.world_size)
                                    / total_steps,
                                ),
                                "updates": float(updates),
                                "episodes": float(episodes),
                                "loss": final_loss,
                                "approximate_kl": final_kl,
                                "learning_rate": self.optimizer.param_groups[0]["lr"],
                                "parallel_envs": float(
                                    active_envs * self.distributed.world_size
                                ),
                                "parallel_envs_per_rank": float(active_envs),
                                "world_size": float(self.distributed.world_size),
                                **telemetry,
                            }
                        )
                        if progress_callback
                        else None
                    ),
                )
            )
            collect_seconds = self.distributed.all_reduce_float(
                perf_counter() - collect_started,
                reduction="max",
            )
            local_diagnostics = dict(rollout.get("_diagnostics") or {})
            collect_diagnostics = _aggregate_rollout_diagnostics(
                local_diagnostics,
                self.distributed,
            )
            global_samples = self.distributed.sum_int(samples)
            collect_diagnostics.update(
                {
                    "collect_seconds": collect_seconds,
                    "collect_steps_per_second": global_samples / max(collect_seconds, 1e-9),
                    "collect_overhead_seconds": _collect_overhead_seconds(
                        collect_seconds, collect_diagnostics
                    ),
                }
            )
            environments[:active_envs] = next_environments
            observations[:active_envs] = next_observations
            episodes += self.distributed.sum_int(ended_episodes)
            reward_tensor = rollout["reward"]
            reward_sum += self.distributed.all_reduce_float(
                float(reward_tensor.sum().item()),
                reduction="sum",
            )
            reward_count += self.distributed.sum_int(int(reward_tensor.numel()))
            if progress_callback:
                progress_callback(
                    {
                        "phase": "optimizing",
                        "steps": float(completed_steps + global_samples),
                        "total_steps": float(total_steps),
                        "progress": min(
                            1.0,
                            (completed_steps + global_samples) / total_steps,
                        ),
                        "updates": float(updates),
                        "episodes": float(episodes),
                        "loss": final_loss,
                        "approximate_kl": final_kl,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "parallel_envs": float(active_envs * self.distributed.world_size),
                        "parallel_envs_per_rank": float(active_envs),
                        "world_size": float(self.distributed.world_size),
                        **collect_diagnostics,
                    }
                )
            optimize_started = perf_counter()
            final_update = self._update(rollout)
            optimize_seconds = self.distributed.all_reduce_float(
                perf_counter() - optimize_started,
                reduction="max",
            )
            final_loss = final_update.total_loss
            final_kl = final_update.approximate_kl
            completed_steps += global_samples
            updates += 1
            progress = completed_steps / total_steps
            learning_rate = self.config.learning_rate + progress * (
                self.config.final_learning_rate - self.config.learning_rate
            )
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            update_payload = {
                "phase": "updated",
                "steps": float(completed_steps),
                "total_steps": float(total_steps),
                "progress": progress,
                "updates": float(updates),
                "episodes": float(episodes),
                "loss": final_loss,
                "approximate_kl": final_kl,
                "learning_rate": learning_rate,
                "policy_loss": final_update.policy_loss,
                "value_loss": final_update.value_loss,
                "entropy": final_update.entropy,
                "clip_fraction": final_update.clip_fraction,
                "gradient_norm": final_update.gradient_norm,
                "optimizer_steps": float(final_update.optimizer_steps),
                "parallel_envs": float(active_envs * self.distributed.world_size),
                "parallel_envs_per_rank": float(active_envs),
                "world_size": float(self.distributed.world_size),
                "optimize_seconds": optimize_seconds,
                "samples_per_update": float(global_samples),
                "local_samples_per_update": float(samples),
                "optimize_steps_per_second": global_samples / max(optimize_seconds, 1e-9),
                "optimizer_steps_per_second": final_update.optimizer_steps / max(optimize_seconds, 1e-9),
                **collect_diagnostics,
            }
            if progress_callback:
                progress_callback(update_payload)
            if update_callback:
                update_callback(update_payload)
            self.distributed.barrier()

        return TrainingSummary(
            total_steps=completed_steps,
            updates=updates,
            episodes=episodes,
            mean_reward=reward_sum / max(1, reward_count),
            final_loss=final_loss,
            final_approximate_kl=final_kl,
            final_policy_loss=final_update.policy_loss,
            final_value_loss=final_update.value_loss,
            final_entropy=final_update.entropy,
            final_clip_fraction=final_update.clip_fraction,
            final_gradient_norm=final_update.gradient_norm,
        )

    def _collect_rollout(
        self,
        environment: AshareTradingEnv,
        observation: AgentObservation,
        environment_factory: Callable[[], AshareTradingEnv],
        rollout_steps: int,
        cancel_check: Callable[[], bool] | None,
        step_callback: Callable[[int, dict[str, object]], None] | None = None,
    ) -> tuple[dict[str, object], AgentObservation, AshareTradingEnv, int]:
        self.model.eval()
        inputs_list: list[dict[str, object]] = []
        directions: list[Tensor] = []
        sizes: list[Tensor] = []
        log_prob: list[Tensor] = []
        values: list[Tensor] = []
        active_masks: list[Tensor] = []
        rewards: list[float] = []
        dones: list[bool] = []
        episodes = 0
        diagnostics = RolloutDiagnostics()

        for rollout_index in range(rollout_steps):
            if cancel_check and cancel_check():
                raise TrainingCancelled("Agent training cancellation requested.")
            model_started = perf_counter()
            inputs = observation_to_tensors(observation, device=self.device)
            with torch.inference_mode():
                action = self.model.act(**inputs)
            model_seconds = perf_counter() - model_started
            direction_cpu = action.directions.squeeze(0).detach().cpu()
            size_cpu = action.sizes.squeeze(0).detach().cpu()
            portfolio_action = PortfolioAction(
                direction_cpu.numpy(),
                size_cpu.numpy(),
            )
            loader_before = _loader_performance(environment)
            environment_started = perf_counter()
            next_observation, reward, terminated, truncated, info = environment.step(
                portfolio_action
            )
            environment_seconds = perf_counter() - environment_started
            loader_delta = _loader_performance_delta(loader_before, environment)
            diagnostics.samples += 1
            diagnostics.model_seconds += model_seconds
            diagnostics.environment_seconds += environment_seconds
            diagnostics.data_load_seconds += float(loader_delta.get("load_seconds") or 0.0)
            diagnostics.chunk_count += int(loader_delta.get("chunks") or 0)
            diagnostics.data_steps += int(loader_delta.get("steps") or 0)
            diagnostics.add_action_counts(portfolio_action)
            diagnostics.add_execution(info.get("execution"))
            inputs_list.append(_inputs_to_cpu(inputs))
            directions.append(direction_cpu)
            sizes.append(size_cpu)
            log_prob.append(action.log_prob.squeeze(0).detach().cpu())
            values.append(action.value.squeeze(0).detach().cpu())
            active_masks.append(inputs["active_mask"].squeeze(0).detach().cpu())
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))
            reset_seconds = 0.0
            reset_data_seconds = 0.0
            reset_chunks = 0
            reset_data_steps = 0
            if terminated or truncated:
                episodes += 1
                reset_started = perf_counter()
                old_environment = environment
                replacement = environment_factory()
                try:
                    reset_loader_before = _loader_performance(replacement)
                    observation, _ = replacement.reset()
                except Exception:
                    _close_environment(replacement)
                    raise
                environment = replacement
                _close_environment(old_environment)
                reset_seconds = perf_counter() - reset_started
                reset_delta = _loader_performance_delta(reset_loader_before, environment)
                reset_data_seconds = float(reset_delta.get("load_seconds") or 0.0)
                reset_chunks = int(reset_delta.get("chunks") or 0)
                reset_data_steps = int(reset_delta.get("steps") or 0)
                diagnostics.reset_seconds += reset_seconds
                diagnostics.data_load_seconds += reset_data_seconds
                diagnostics.chunk_count += reset_chunks
                diagnostics.data_steps += reset_data_steps
            else:
                if next_observation is None:
                    raise RuntimeError("A non-terminal environment step returned no observation.")
                observation = next_observation
            if step_callback:
                directions_np = portfolio_action.directions
                execution = info.get("execution")
                blocked_reasons: dict[str, int] = {}
                if execution:
                    for fill in execution.fills:
                        if fill.status == "blocked":
                            reason = str(fill.reason or "unknown")
                            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
                step_callback(
                    rollout_index + 1,
                    {
                        "reward": float(reward),
                        "nav": float(info.get("net_asset_value") or 0.0),
                        "cash": float(info.get("cash") or 0.0),
                        "buy_actions": int(np.count_nonzero(directions_np > 0)),
                        "hold_actions": int(np.count_nonzero(directions_np == 0)),
                        "sell_actions": int(np.count_nonzero(directions_np < 0)),
                        "executed_orders": len(execution.executed_fills) if execution else 0,
                        "blocked_orders": (
                            sum(fill.status == "blocked" for fill in execution.fills)
                            if execution else 0
                        ),
                        "blocked_reasons": blocked_reasons,
                        "turnover_value": float(execution.turnover_value) if execution else 0.0,
                        "total_fees": float(environment.account.total_fees),
                        "model_seconds": model_seconds,
                        "environment_seconds": environment_seconds,
                        "data_load_seconds": float(loader_delta.get("load_seconds") or 0.0) + reset_data_seconds,
                        "data_chunks": int(loader_delta.get("chunks") or 0) + reset_chunks,
                        "data_steps": int(loader_delta.get("steps") or 0) + reset_data_steps,
                        "reset_seconds": reset_seconds,
                    },
                )

        if dones[-1]:
            bootstrap = torch.zeros(())
        else:
            with torch.inference_mode():
                bootstrap_inputs = observation_to_tensors(observation, device=self.device)
                bootstrap = self.model(**bootstrap_inputs).value.squeeze(0).cpu()
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        done_tensor = torch.tensor(dones, dtype=torch.bool)
        value_tensor = torch.stack([*values, bootstrap])
        advantages, returns = compute_gae(
            reward_tensor,
            value_tensor,
            done_tensor,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        return (
            {
                "inputs": _stack_inputs(inputs_list),
                "directions": torch.stack(directions),
                "sizes": torch.stack(sizes),
                "old_log_prob": torch.stack(log_prob),
                "old_values": torch.stack(values),
                "active_mask": torch.stack(active_masks),
                "reward": reward_tensor,
                "advantages": advantages.detach(),
                "returns": returns.detach(),
                "_diagnostics": diagnostics.payload(),
            },
            observation,
            environment,
            episodes,
        )

    def _collect_rollout_parallel(
        self,
        environments: list[AshareTradingEnv],
        observations: list[AgentObservation],
        environment_factory: Callable[[], AshareTradingEnv],
        per_env_steps: int,
        cancel_check: Callable[[], bool] | None,
        step_callback: Callable[[int, dict[str, object]], None] | None = None,
    ) -> tuple[
        dict[str, object],
        list[AgentObservation],
        list[AshareTradingEnv],
        int,
        int,
    ]:
        self.model.eval()
        inputs_list: list[dict[str, object]] = []
        directions: list[Tensor] = []
        sizes: list[Tensor] = []
        log_prob: list[Tensor] = []
        values: list[Tensor] = []
        active_masks: list[Tensor] = []
        rewards: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        episodes = 0
        env_count = len(environments)
        diagnostics = RolloutDiagnostics()

        for rollout_index in range(per_env_steps):
            if cancel_check and cancel_check():
                raise TrainingCancelled("Agent training cancellation requested.")
            model_started = perf_counter()
            inputs = _observations_to_tensors(observations, device=self.device)
            with torch.inference_mode():
                action = self.model.act(**inputs)
            model_seconds = perf_counter() - model_started
            directions_cpu = action.directions.detach().cpu()
            sizes_cpu = action.sizes.detach().cpu()
            step_rewards = np.zeros(env_count, dtype=np.float32)
            step_dones = np.zeros(env_count, dtype=np.bool_)
            telemetry = _ParallelTelemetry()
            step_data_load_seconds = 0.0
            step_data_chunks = 0
            step_data_steps = 0
            step_reset_seconds = 0.0
            environment_started = perf_counter()
            for env_index, environment in enumerate(environments):
                portfolio_action = PortfolioAction(
                    directions_cpu[env_index].numpy(),
                    sizes_cpu[env_index].numpy(),
                )
                loader_before = _loader_performance(environment)
                next_observation, reward, terminated, truncated, info = environment.step(
                    portfolio_action
                )
                loader_delta = _loader_performance_delta(loader_before, environment)
                step_data_load_seconds += float(loader_delta.get("load_seconds") or 0.0)
                step_data_chunks += int(loader_delta.get("chunks") or 0)
                step_data_steps += int(loader_delta.get("steps") or 0)
                step_rewards[env_index] = float(reward)
                done = bool(terminated or truncated)
                step_dones[env_index] = done
                telemetry.add(environment, portfolio_action, info)
                diagnostics.samples += 1
                diagnostics.add_action_counts(portfolio_action)
                diagnostics.add_execution(info.get("execution"))
                if done:
                    episodes += 1
                    reset_started = perf_counter()
                    old_environment = environment
                    replacement = environment_factory()
                    try:
                        reset_loader_before = _loader_performance(replacement)
                        observations[env_index], _ = replacement.reset()
                    except Exception:
                        _close_environment(replacement)
                        raise
                    _close_environment(old_environment)
                    reset_seconds = perf_counter() - reset_started
                    reset_delta = _loader_performance_delta(reset_loader_before, replacement)
                    step_reset_seconds += reset_seconds
                    step_data_load_seconds += float(reset_delta.get("load_seconds") or 0.0)
                    step_data_chunks += int(reset_delta.get("chunks") or 0)
                    step_data_steps += int(reset_delta.get("steps") or 0)
                    environments[env_index] = replacement
                else:
                    if next_observation is None:
                        raise RuntimeError(
                            "A non-terminal environment step returned no observation."
                        )
                    observations[env_index] = next_observation
            environment_seconds = perf_counter() - environment_started
            diagnostics.model_seconds += model_seconds
            diagnostics.environment_seconds += environment_seconds
            diagnostics.data_load_seconds += step_data_load_seconds
            diagnostics.chunk_count += step_data_chunks
            diagnostics.data_steps += step_data_steps
            diagnostics.reset_seconds += step_reset_seconds
            inputs_list.append(_inputs_to_cpu(inputs))
            directions.append(directions_cpu)
            sizes.append(sizes_cpu)
            log_prob.append(action.log_prob.detach().cpu())
            values.append(action.value.detach().cpu())
            active_masks.append(inputs["active_mask"].detach().cpu())
            rewards.append(step_rewards)
            dones.append(step_dones)
            if step_callback:
                payload = telemetry.payload()
                payload.update(
                    {
                        "model_seconds": model_seconds,
                        "environment_seconds": environment_seconds,
                        "data_load_seconds": step_data_load_seconds,
                        "data_chunks": step_data_chunks,
                        "data_steps": step_data_steps,
                        "reset_seconds": step_reset_seconds,
                        "parallel_envs": env_count,
                    }
                )
                step_callback((rollout_index + 1) * env_count, payload)

        with torch.inference_mode():
            bootstrap_inputs = _observations_to_tensors(observations, device=self.device)
            bootstrap = self.model(**bootstrap_inputs).value.cpu()
        if dones:
            bootstrap = torch.where(
                torch.as_tensor(dones[-1], dtype=torch.bool),
                torch.zeros_like(bootstrap),
                bootstrap,
            )
        reward_tensor = torch.as_tensor(np.stack(rewards), dtype=torch.float32)
        done_tensor = torch.as_tensor(np.stack(dones), dtype=torch.bool)
        value_tensor = torch.cat([torch.stack(values), bootstrap.unsqueeze(0)], dim=0)
        advantages, returns = compute_gae(
            reward_tensor,
            value_tensor,
            done_tensor,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        samples = int(reward_tensor.numel())
        return (
            {
                "inputs": _stack_inputs(inputs_list),
                "directions": torch.cat(directions, dim=0),
                "sizes": torch.cat(sizes, dim=0),
                "old_log_prob": torch.cat(log_prob, dim=0),
                "old_values": torch.cat(values, dim=0),
                "active_mask": torch.cat(active_masks, dim=0),
                "reward": reward_tensor.reshape(-1),
                "advantages": advantages.reshape(-1).detach(),
                "returns": returns.reshape(-1).detach(),
                "_diagnostics": diagnostics.payload(),
            },
            observations,
            environments,
            episodes,
            samples,
        )


    def _collect_rollout_vector(
        self,
        vector_env: Any,
        observations: list[AgentObservation],
        per_env_steps: int,
        cancel_check: Callable[[], bool] | None,
        step_callback: Callable[[int, dict[str, object]], None] | None = None,
    ) -> tuple[dict[str, object], list[AgentObservation], int, int]:
        self.model.eval()
        inputs_list: list[dict[str, object]] = []
        directions: list[Tensor] = []
        sizes: list[Tensor] = []
        log_prob: list[Tensor] = []
        values: list[Tensor] = []
        active_masks: list[Tensor] = []
        rewards: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        episodes = 0
        env_count = len(observations)
        diagnostics = RolloutDiagnostics()

        for rollout_index in range(per_env_steps):
            if cancel_check and cancel_check():
                raise TrainingCancelled("Agent training cancellation requested.")
            model_started = perf_counter()
            inputs = _observations_to_tensors(observations, device=self.device)
            with torch.inference_mode():
                action = self.model.act(**inputs)
            model_seconds = perf_counter() - model_started
            directions_cpu = action.directions.detach().cpu()
            sizes_cpu = action.sizes.detach().cpu()
            worker_actions = [
                (directions_cpu[index].numpy(), sizes_cpu[index].numpy())
                for index in range(env_count)
            ]
            environment_started = perf_counter()
            results = vector_env.step(worker_actions)
            environment_wall_seconds = perf_counter() - environment_started
            step_rewards = np.zeros(env_count, dtype=np.float32)
            step_dones = np.zeros(env_count, dtype=np.bool_)
            step_data_load_seconds = 0.0
            step_data_chunks = 0
            step_data_steps = 0
            step_reset_seconds = 0.0
            step_environment_seconds = 0.0
            telemetry = _VectorTelemetry()
            for env_index, result in enumerate(results):
                observation = result.get("observation")
                if observation is None:
                    raise RuntimeError("Subprocess vector env returned no observation.")
                observations[env_index] = observation
                reward = float(result.get("reward") or 0.0)
                done = bool(result.get("done") or result.get("terminated") or result.get("truncated"))
                info = dict(result.get("info") or {})
                step_rewards[env_index] = reward
                step_dones[env_index] = done
                if done:
                    episodes += 1
                portfolio_action = PortfolioAction(
                    directions_cpu[env_index].numpy(),
                    sizes_cpu[env_index].numpy(),
                )
                diagnostics.samples += 1
                diagnostics.add_action_counts(portfolio_action)
                diagnostics.add_execution(info.get("execution"))
                telemetry.add(portfolio_action, info)
                step_data_load_seconds += float(result.get("data_load_seconds") or 0.0)
                step_data_chunks += int(result.get("data_chunks") or 0)
                step_data_steps += int(result.get("data_steps") or 0)
                step_reset_seconds += float(result.get("reset_seconds") or 0.0)
                step_environment_seconds += float(result.get("environment_seconds") or 0.0)
            diagnostics.model_seconds += model_seconds
            diagnostics.environment_seconds += step_environment_seconds or environment_wall_seconds
            diagnostics.data_load_seconds += step_data_load_seconds
            diagnostics.chunk_count += step_data_chunks
            diagnostics.data_steps += step_data_steps
            diagnostics.reset_seconds += step_reset_seconds
            inputs_list.append(_inputs_to_cpu(inputs))
            directions.append(directions_cpu)
            sizes.append(sizes_cpu)
            log_prob.append(action.log_prob.detach().cpu())
            values.append(action.value.detach().cpu())
            active_masks.append(inputs["active_mask"].detach().cpu())
            rewards.append(step_rewards)
            dones.append(step_dones)
            if step_callback:
                payload = telemetry.payload()
                payload.update(
                    {
                        "model_seconds": model_seconds,
                        "environment_seconds": step_environment_seconds or environment_wall_seconds,
                        "environment_wall_seconds": environment_wall_seconds,
                        "data_load_seconds": step_data_load_seconds,
                        "data_chunks": step_data_chunks,
                        "data_steps": step_data_steps,
                        "reset_seconds": step_reset_seconds,
                        "parallel_envs": env_count,
                        "env_backend": "subprocess",
                    }
                )
                step_callback((rollout_index + 1) * env_count, payload)

        with torch.inference_mode():
            bootstrap_inputs = _observations_to_tensors(observations, device=self.device)
            bootstrap = self.model(**bootstrap_inputs).value.cpu()
        if dones:
            bootstrap = torch.where(
                torch.as_tensor(dones[-1], dtype=torch.bool),
                torch.zeros_like(bootstrap),
                bootstrap,
            )
        reward_tensor = torch.as_tensor(np.stack(rewards), dtype=torch.float32)
        done_tensor = torch.as_tensor(np.stack(dones), dtype=torch.bool)
        value_tensor = torch.cat([torch.stack(values), bootstrap.unsqueeze(0)], dim=0)
        advantages, returns = compute_gae(
            reward_tensor,
            value_tensor,
            done_tensor,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        samples = int(reward_tensor.numel())
        return (
            {
                "inputs": _stack_inputs(inputs_list),
                "directions": torch.cat(directions, dim=0),
                "sizes": torch.cat(sizes, dim=0),
                "old_log_prob": torch.cat(log_prob, dim=0),
                "old_values": torch.cat(values, dim=0),
                "active_mask": torch.cat(active_masks, dim=0),
                "reward": reward_tensor.reshape(-1),
                "advantages": advantages.reshape(-1).detach(),
                "returns": returns.reshape(-1).detach(),
                "_diagnostics": diagnostics.payload(),
            },
            observations,
            episodes,
            samples,
        )

    def _update(self, rollout: dict[str, object]) -> TrainingUpdateMetrics:
        steps = int(rollout["directions"].shape[0])
        final_loss = 0.0
        final_kl = 0.0
        final_policy = 0.0
        final_value = 0.0
        final_entropy = 0.0
        final_clip = 0.0
        final_gradient = 0.0
        optimizer_steps = 0
        # cuDNN LSTMs require training mode for backward. The model contract fixes
        # dropout at zero, so PPO likelihood-ratio re-evaluation stays deterministic.
        self.model.train()
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(steps)
            stop_for_kl = False
            for start in range(0, steps, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                loss = compute_mappo_loss(
                    self.update_model,
                    observation_inputs=_inputs_to_device(
                        _slice_inputs(rollout["inputs"], indices), self.device
                    ),
                    directions=rollout["directions"][indices].to(self.device),
                    sizes=rollout["sizes"][indices].to(self.device),
                    old_log_prob=rollout["old_log_prob"][indices].to(self.device),
                    advantages=rollout["advantages"][indices].to(self.device),
                    returns=rollout["returns"][indices].to(self.device),
                    old_values=rollout["old_values"][indices].to(self.device),
                    active_mask=rollout["active_mask"][indices].to(self.device),
                    config=self.config,
                )
                final_loss = float(loss.total.detach().item())
                final_policy = float(loss.policy.detach().item())
                final_value = float(loss.value.detach().item())
                final_entropy = float(loss.entropy.detach().item())
                final_clip = float(loss.clip_fraction.detach().item())
                final_kl = self.distributed.all_reduce_float(
                    float(loss.approximate_kl.detach().item()),
                    reduction="max",
                )
                if final_kl > self.config.target_kl:
                    stop_for_kl = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.maximum_gradient_norm
                )
                final_gradient = float(gradient_norm.detach().item())
                self.optimizer.step()
                optimizer_steps += 1
            if stop_for_kl:
                break
        self.model.eval()
        reduced_metrics = self.distributed.all_reduce_floats(
            [final_loss, final_policy, final_value, final_entropy, final_clip],
            reduction="sum",
        )
        (
            final_loss,
            final_policy,
            final_value,
            final_entropy,
            final_clip,
        ) = [value / self.distributed.world_size for value in reduced_metrics]
        final_gradient = self.distributed.mean_float(final_gradient)
        optimizer_steps = int(round(self.distributed.mean_float(float(optimizer_steps))))
        return TrainingUpdateMetrics(
            final_loss,
            final_policy,
            final_value,
            final_entropy,
            final_kl,
            final_clip,
            final_gradient,
            optimizer_steps,
        )


def _stack_inputs(inputs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "market_sequences": {
            freq: torch.cat([item["market_sequences"][freq] for item in inputs], dim=0)
            for freq in inputs[0]["market_sequences"]
        },
        "sequence_masks": {
            freq: torch.cat([item["sequence_masks"][freq] for item in inputs], dim=0)
            for freq in inputs[0]["sequence_masks"]
        },
        **{
            name: torch.cat([item[name] for item in inputs], dim=0)
            for name in (
                "decision_context",
                "runtime_state",
                "active_mask",
                "can_buy",
                "can_sell",
                "buy_min_size",
                "sell_min_size",
            )
        },
    }


def _observations_to_tensors(
    observations: list[AgentObservation],
    *,
    device: torch.device,
) -> dict[str, object]:
    if not observations:
        raise ValueError("At least one Agent observation is required.")
    state_inputs = [
        observation_state_to_tensors(observation, device=device)
        for observation in observations
    ]
    return {
        **market_steps_to_tensors(
            [observation.market for observation in observations],
            device=device,
        ),
        **{
            name: torch.cat([item[name] for item in state_inputs], dim=0)
            for name in (
                "decision_context",
                "runtime_state",
                "active_mask",
                "can_buy",
                "can_sell",
                "buy_min_size",
                "sell_min_size",
            )
        },
    }


class _ParallelTelemetry:
    def __init__(self) -> None:
        self.reward = 0.0
        self.nav = 0.0
        self.cash = 0.0
        self.envs = 0
        self.buy_actions = 0
        self.hold_actions = 0
        self.sell_actions = 0
        self.executed_orders = 0
        self.blocked_orders = 0
        self.blocked_reasons: dict[str, int] = {}
        self.turnover_value = 0.0
        self.total_fees = 0.0

    def add(
        self,
        environment: AshareTradingEnv,
        action: PortfolioAction,
        info: dict[str, object],
    ) -> None:
        self.envs += 1
        self.reward += float(info.get("reward") or 0.0)
        self.nav += float(info.get("net_asset_value") or 0.0)
        self.cash += float(info.get("cash") or 0.0)
        self.buy_actions += int(np.count_nonzero(action.directions > 0))
        self.hold_actions += int(np.count_nonzero(action.directions == 0))
        self.sell_actions += int(np.count_nonzero(action.directions < 0))
        execution = info.get("execution")
        if execution:
            self.executed_orders += len(execution.executed_fills)
            blocked = [fill for fill in execution.fills if fill.status == "blocked"]
            self.blocked_orders += len(blocked)
            for fill in blocked:
                reason = str(fill.reason or "unknown")
                self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
            self.turnover_value += float(execution.turnover_value)
        self.total_fees += float(environment.account.total_fees)

    def payload(self) -> dict[str, object]:
        divisor = max(1, self.envs)
        return {
            "reward": self.reward / divisor,
            "nav": self.nav / divisor,
            "cash": self.cash / divisor,
            "buy_actions": self.buy_actions,
            "hold_actions": self.hold_actions,
            "sell_actions": self.sell_actions,
            "executed_orders": self.executed_orders,
            "blocked_orders": self.blocked_orders,
            "blocked_reasons": self.blocked_reasons,
            "turnover_value": self.turnover_value,
            "total_fees": self.total_fees,
        }



class _VectorTelemetry:
    def __init__(self) -> None:
        self.reward = 0.0
        self.nav = 0.0
        self.cash = 0.0
        self.envs = 0
        self.buy_actions = 0
        self.hold_actions = 0
        self.sell_actions = 0
        self.executed_orders = 0
        self.blocked_orders = 0
        self.blocked_reasons: dict[str, int] = {}
        self.turnover_value = 0.0
        self.total_fees = 0.0

    def add(self, action: PortfolioAction, info: dict[str, object]) -> None:
        self.envs += 1
        self.reward += float(info.get("reward") or 0.0)
        self.nav += float(info.get("net_asset_value") or 0.0)
        self.cash += float(info.get("cash") or 0.0)
        self.buy_actions += int(np.count_nonzero(action.directions > 0))
        self.hold_actions += int(np.count_nonzero(action.directions == 0))
        self.sell_actions += int(np.count_nonzero(action.directions < 0))
        execution = info.get("execution")
        if execution:
            self.executed_orders += len(execution.executed_fills)
            blocked = [fill for fill in execution.fills if fill.status == "blocked"]
            self.blocked_orders += len(blocked)
            for fill in blocked:
                reason = str(fill.reason or "unknown")
                self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
            self.turnover_value += float(execution.turnover_value)
        self.total_fees += float(info.get("total_fees") or 0.0)

    def payload(self) -> dict[str, object]:
        divisor = max(1, self.envs)
        return {
            "reward": self.reward / divisor,
            "nav": self.nav / divisor,
            "cash": self.cash / divisor,
            "buy_actions": self.buy_actions,
            "hold_actions": self.hold_actions,
            "sell_actions": self.sell_actions,
            "executed_orders": self.executed_orders,
            "blocked_orders": self.blocked_orders,
            "blocked_reasons": self.blocked_reasons,
            "turnover_value": self.turnover_value,
            "total_fees": self.total_fees,
        }


def _close_environment(environment: AshareTradingEnv | None) -> None:
    if environment is None:
        return
    close = getattr(environment, "close", None)
    if callable(close):
        close()


def _loader_performance(environment: AshareTradingEnv) -> dict[str, float | int]:
    payload = getattr(environment.loader, "performance_payload", None)
    if not callable(payload):
        return {"chunks": 0, "steps": 0, "load_seconds": 0.0}
    try:
        data = payload()
    except Exception:
        return {"chunks": 0, "steps": 0, "load_seconds": 0.0}
    return {
        "chunks": int(data.get("chunks") or 0),
        "steps": int(data.get("steps") or 0),
        "load_seconds": float(data.get("load_seconds") or 0.0),
    }


def _loader_performance_delta(
    before: dict[str, float | int],
    environment: AshareTradingEnv,
) -> dict[str, float | int]:
    after = _loader_performance(environment)
    return {
        "chunks": max(0, int(after.get("chunks") or 0) - int(before.get("chunks") or 0)),
        "steps": max(0, int(after.get("steps") or 0) - int(before.get("steps") or 0)),
        "load_seconds": max(
            0.0,
            float(after.get("load_seconds") or 0.0)
            - float(before.get("load_seconds") or 0.0),
        ),
    }


def _aggregate_rollout_diagnostics(
    local: dict[str, object],
    distributed: DistributedTrainingContext,
) -> dict[str, object]:
    gathered = [dict(item or {}) for item in distributed.all_gather_object(local)]
    samples = sum(int(item.get("collect_samples") or 0) for item in gathered)
    result: dict[str, object] = {
        "collect_samples": samples,
    }
    for name in (
        "data_chunks",
        "data_steps",
        "buy_actions",
        "hold_actions",
        "sell_actions",
        "executed_orders",
        "blocked_orders",
    ):
        result[name] = sum(int(item.get(name) or 0) for item in gathered)
    result["turnover_value"] = sum(
        float(item.get("turnover_value") or 0.0) for item in gathered
    )
    for name in (
        "collect_model_seconds",
        "collect_environment_seconds",
        "collect_data_load_seconds",
        "collect_reset_seconds",
    ):
        result[name] = max((float(item.get(name) or 0.0) for item in gathered), default=0.0)
    divisor = max(1, samples)
    result["model_seconds"] = float(result["collect_model_seconds"]) / divisor
    result["environment_seconds"] = float(result["collect_environment_seconds"]) / divisor
    result["data_load_seconds"] = float(result["collect_data_load_seconds"]) / divisor
    result["reset_seconds"] = float(result["collect_reset_seconds"]) / divisor
    blocked_reasons: dict[str, int] = {}
    for item in gathered:
        for reason, count in dict(item.get("blocked_reasons") or {}).items():
            blocked_reasons[str(reason)] = blocked_reasons.get(str(reason), 0) + int(count)
    result["blocked_reasons"] = blocked_reasons
    return result


def _collect_overhead_seconds(
    collect_seconds: float,
    diagnostics: dict[str, object],
) -> float:
    accounted = (
        float(diagnostics.get("collect_model_seconds") or 0.0)
        + float(diagnostics.get("collect_environment_seconds") or 0.0)
        + float(diagnostics.get("collect_reset_seconds") or 0.0)
    )
    return max(0.0, float(collect_seconds) - accounted)


def _slice_inputs(inputs: dict[str, object], indices: Tensor) -> dict[str, object]:
    return {
        "market_sequences": {
            freq: values[indices] for freq, values in inputs["market_sequences"].items()
        },
        "sequence_masks": {
            freq: values[indices] for freq, values in inputs["sequence_masks"].items()
        },
        **{
            name: inputs[name][indices]
            for name in (
                "decision_context",
                "runtime_state",
                "active_mask",
                "can_buy",
                "can_sell",
                "buy_min_size",
                "sell_min_size",
            )
        },
    }


def _inputs_to_cpu(inputs: dict[str, object]) -> dict[str, object]:
    return _inputs_to_device(inputs, torch.device("cpu"))


def _inputs_to_device(
    inputs: dict[str, object], device: torch.device
) -> dict[str, object]:
    return {
        "market_sequences": {
            freq: values.detach().to(device)
            for freq, values in inputs["market_sequences"].items()
        },
        "sequence_masks": {
            freq: values.detach().to(device)
            for freq, values in inputs["sequence_masks"].items()
        },
        **{
            name: inputs[name].detach().to(device)
            for name in (
                "decision_context",
                "runtime_state",
                "active_mask",
                "can_buy",
                "can_sell",
                "buy_min_size",
                "sell_min_size",
            )
        },
    }
