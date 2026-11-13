from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from agent_layer.environment import ExecutionConfig
from agent_layer.training import MAPPOConfig


DEFAULT_AGENT_FREQUENCIES: tuple[str, ...] = (
    "5min",
    "15min",
    "30min",
    "60min",
    "daily",
    "weekly",
    "monthly",
)


@dataclass(frozen=True)
class ModelHyperparameters:
    input_projection_size: int = 64
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    fused_market_size: int = 128
    context_embedding_size: int = 32
    runtime_embedding_size: int = 32
    local_state_size: int = 128
    global_state_size: int = 128
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name == "dropout":
                if float(value) != 0.0:
                    raise ValueError("PPO model dropout is fixed at zero.")
            elif int(value) <= 0:
                raise ValueError(f"Model parameter {name} must be positive.")


@dataclass(frozen=True)
class RewardConfig:
    kind: str = "net_asset_log_return"
    scale: float = 1.0
    hurdle_rate_annual: float = 0.0
    drawdown_penalty: float = 0.0
    turnover_penalty: float = 0.0
    invalid_action_penalty: float = 0.0

    def validate(self) -> None:
        if self.kind != "net_asset_log_return":
            raise ValueError("Agent v1 only supports net_asset_log_return.")
        if self.scale <= 0:
            raise ValueError("Reward scale must be positive.")
        if self.hurdle_rate_annual < 0:
            raise ValueError("Reward hurdle rate cannot be negative.")
        for name in ("drawdown_penalty", "turnover_penalty", "invalid_action_penalty"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"Reward parameter {name} cannot be negative.")


@dataclass(frozen=True)
class CheckpointPolicy:
    checkpoint_interval_updates: int = 5
    validation_interval_updates: int = 50
    keep_last: int = 5
    best_metric: str = "sharpe"

    def validate(self) -> None:
        if self.checkpoint_interval_updates <= 0:
            raise ValueError("Checkpoint interval must be positive.")
        if self.validation_interval_updates < 0:
            raise ValueError("Validation interval cannot be negative.")
        if self.keep_last <= 0:
            raise ValueError("Checkpoint retention must be positive.")
        if self.best_metric not in {"sharpe", "calmar", "total_return"}:
            raise ValueError("Unsupported best-checkpoint metric.")


@dataclass(frozen=True)
class ValidationPolicy:
    symbol_limit: int = 5
    symbol_seed: int = 2026
    quick_days: int = 5
    periodic_device: str = "cpu"
    final_device: str = "auto"

    def validate(self) -> None:
        if self.symbol_limit <= 0:
            raise ValueError("Validation symbol limit must be positive.")
        if self.quick_days <= 0:
            raise ValueError("Quick validation days must be positive.")
        for name in ("periodic_device", "final_device"):
            if getattr(self, name) not in {"auto", "cpu", "cuda"}:
                raise ValueError(f"Validation {name} must be auto, cpu, or cuda.")


@dataclass(frozen=True)
class SplitPolicy:
    validation_days: int = 20
    test_days: int = 20
    embargo_days: int = 5
    folds: int = 1
    min_train_days: int = 30

    def validate(self) -> None:
        for name in ("validation_days", "test_days", "folds", "min_train_days"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"Split {name} must be positive.")
        if int(self.embargo_days) < 0:
            raise ValueError("Split embargo_days cannot be negative.")


@dataclass(frozen=True)
class RuntimePolicy:
    amp_enabled: bool = False
    amp_dtype: str = "float16"
    size_distribution: str = "beta"
    env_backend: str = "in_process"

    def validate(self) -> None:
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("AMP dtype must be float16 or bfloat16.")
        if self.size_distribution not in {"beta", "logit_normal"}:
            raise ValueError("Size distribution must be beta or logit_normal.")
        if self.env_backend not in {"in_process", "subprocess"}:
            raise ValueError("Environment backend must be in_process or subprocess.")


@dataclass(frozen=True)
class DistributedPolicy:
    strategy: str = "off"
    processes_per_node: int = 0
    backend: str = "auto"
    cpu_threads_per_process: int = 0
    timeout_seconds: int = 1800

    def validate(self) -> None:
        if self.strategy not in {"off", "auto", "ddp"}:
            raise ValueError("Distributed strategy must be off, auto, or ddp.")
        if int(self.processes_per_node) < 0:
            raise ValueError("Distributed processes_per_node cannot be negative.")
        if self.backend not in {"auto", "nccl", "gloo"}:
            raise ValueError("Distributed backend must be auto, nccl, or gloo.")
        if int(self.cpu_threads_per_process) < 0:
            raise ValueError("Distributed CPU threads per process cannot be negative.")
        if int(self.timeout_seconds) < 60:
            raise ValueError("Distributed timeout must be at least 60 seconds.")


@dataclass(frozen=True)
class AgentRunConfig:
    profile: str = "smoke"
    run_name: str = ""
    store_path: str = "runtime_layer/reports/feature_dataset"
    symbols_file: str | None = "config/universe/smoke_symbols.txt"
    symbol_limit: int = 0
    symbol_seed: int = 42
    fold: int = 1
    total_steps: int = 128
    episode_days: int = 5
    validation_days: int = 5
    seed: int = 42
    device: str = "cuda"
    parallel_envs: int = 1
    use_agent_cache: bool = True
    frequencies: tuple[str, ...] = DEFAULT_AGENT_FREQUENCIES
    output_dir: str = "runtime_layer/models/agent"
    model: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    ppo: MAPPOConfig = field(default_factory=MAPPOConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    checkpoint: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    split: SplitPolicy = field(default_factory=SplitPolicy)
    runtime: RuntimePolicy = field(default_factory=RuntimePolicy)
    distributed: DistributedPolicy = field(default_factory=DistributedPolicy)
    resolved_symbols: tuple[str, ...] = ()
    resolved_validation_symbols: tuple[str, ...] = ()
    schema_hash: str = ""

    def validate(self) -> None:
        if self.profile not in {"smoke", "formal", "server", "custom"}:
            raise ValueError("Run profile must be smoke, formal, server, or custom.")
        self.split.validate()
        if not 1 <= int(self.fold) <= int(self.split.folds):
            raise ValueError(f"Walk-forward fold must be between 1 and {self.split.folds}.")
        for name in ("total_steps", "episode_days", "validation_days"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.symbol_limit < 0:
            raise ValueError("Symbol limit cannot be negative.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Device must be cpu or cuda.")
        if int(self.parallel_envs) <= 0:
            raise ValueError("Parallel environments must be positive.")
        if self.ppo.rollout_steps <= 0 or self.ppo.minibatch_size <= 0:
            raise ValueError("Rollout and minibatch sizes must be positive.")
        if self.ppo.rollout_steps % self.ppo.minibatch_size != 0:
            raise ValueError("Rollout steps must be divisible by minibatch size.")
        if self.ppo.update_epochs <= 0:
            raise ValueError("PPO update epochs must be positive.")
        for name in ("gamma", "gae_lambda"):
            value = float(getattr(self.ppo, name))
            if not 0 < value <= 1:
                raise ValueError(f"PPO {name} must be in (0, 1].")
        for name in ("clip_ratio", "value_clip", "maximum_gradient_norm", "target_kl"):
            if float(getattr(self.ppo, name)) <= 0:
                raise ValueError(f"PPO {name} must be positive.")
        for name in ("value_coefficient", "entropy_coefficient"):
            if float(getattr(self.ppo, name)) < 0:
                raise ValueError(f"PPO {name} cannot be negative.")
        if self.ppo.learning_rate <= 0:
            raise ValueError("Initial learning rate must be positive.")
        if not 0 < self.ppo.final_learning_rate <= self.ppo.learning_rate:
            raise ValueError("Final learning rate must be positive and no larger than initial learning rate.")
        self.model.validate()
        self.execution.__post_init__()
        self.reward.validate()
        self.checkpoint.validate()
        self.validation.validate()
        self.runtime.validate()
        self.distributed.validate()
        if self.distributed.strategy == "ddp" and self.distributed.processes_per_node == 1:
            raise ValueError("DDP requires zero (auto) or at least two processes per node.")
        if self.device == "cpu" and self.distributed.backend == "nccl":
            raise ValueError("NCCL requires CUDA training.")
        if self.resolved_validation_symbols and not set(
            self.resolved_validation_symbols
        ).issubset(self.resolved_symbols):
            raise ValueError("Validation symbols must be contained in the training universe.")

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["resolved_symbols"] = list(self.resolved_symbols)
        result["resolved_validation_symbols"] = list(self.resolved_validation_symbols)
        result["config_hash"] = self.config_hash
        return result

    @property
    def config_hash(self) -> str:
        raw = asdict(self)
        raw["resolved_symbols"] = list(self.resolved_symbols)
        raw["resolved_validation_symbols"] = list(self.resolved_validation_symbols)
        serialized = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def with_contract(
        self,
        *,
        symbols: Iterable[str],
        schema_hash: str,
        validation_symbols: Iterable[str] | None = None,
    ) -> "AgentRunConfig":
        payload = self.payload()
        payload.pop("config_hash", None)
        payload["resolved_symbols"] = tuple(str(symbol).upper() for symbol in symbols)
        payload["resolved_validation_symbols"] = tuple(
            str(symbol).upper()
            for symbol in (validation_symbols or payload["resolved_symbols"])
        )
        payload["schema_hash"] = str(schema_hash)
        return agent_run_config_from_payload(payload)


def agent_run_config_from_payload(payload: dict[str, Any]) -> AgentRunConfig:
    source = dict(payload or {})
    profile = str(source.get("profile") or "smoke")
    if profile == "formal":
        defaults = formal_run_config()
    elif profile == "server":
        defaults = server_run_config()
    else:
        defaults = AgentRunConfig()
    model = ModelHyperparameters(**_merged(asdict(defaults.model), source.get("model")))
    ppo_values = _merged(asdict(defaults.ppo), source.get("ppo"))
    for key in asdict(defaults.ppo):
        if key in source:
            ppo_values[key] = source[key]
    execution_values = _merged(asdict(defaults.execution), source.get("execution"))
    reward = RewardConfig(**_merged(asdict(defaults.reward), source.get("reward")))
    checkpoint = CheckpointPolicy(
        **_merged(asdict(defaults.checkpoint), source.get("checkpoint"))
    )
    validation = ValidationPolicy(
        **_merged(asdict(defaults.validation), source.get("validation"))
    )
    split = SplitPolicy(**_merged(asdict(defaults.split), source.get("split")))
    runtime = RuntimePolicy(**_merged(asdict(defaults.runtime), source.get("runtime")))
    distributed = DistributedPolicy(
        **_merged(asdict(defaults.distributed), source.get("distributed"))
    )
    values = {
        key: source.get(key, getattr(defaults, key))
        for key in (
            "profile", "run_name", "store_path", "symbols_file", "symbol_limit",
            "symbol_seed", "fold", "total_steps", "episode_days", "validation_days",
            "seed", "device", "parallel_envs", "use_agent_cache", "frequencies",
            "output_dir", "schema_hash",
        )
    }
    values["frequencies"] = tuple(
        dict.fromkeys(str(value) for value in (values.get("frequencies") or ()))
    )
    values["resolved_symbols"] = tuple(source.get("resolved_symbols") or ())
    values["resolved_validation_symbols"] = tuple(
        source.get("resolved_validation_symbols") or ()
    )
    config = AgentRunConfig(
        **values,
        model=model,
        ppo=MAPPOConfig(**ppo_values),
        execution=ExecutionConfig(**execution_values),
        reward=reward,
        checkpoint=checkpoint,
        validation=validation,
        split=split,
        runtime=runtime,
        distributed=distributed,
    )
    config.validate()
    return config


def formal_run_config() -> AgentRunConfig:
    return AgentRunConfig(
        profile="formal",
        symbols_file="config/universe/available_universe.txt",
        fold=3,
        total_steps=1_000_000,
        episode_days=252,
        validation_days=126,
        parallel_envs=4,
        split=SplitPolicy(
            validation_days=126,
            test_days=252,
            embargo_days=20,
            folds=3,
            min_train_days=252,
        ),
        ppo=MAPPOConfig(
            gamma=0.9995,
            gae_lambda=0.995,
            clip_ratio=0.20,
            value_clip=0.20,
            value_coefficient=0.50,
            entropy_coefficient=0.008,
            maximum_gradient_norm=0.50,
            target_kl=0.02,
            learning_rate=5e-5,
            final_learning_rate=5e-6,
            rollout_steps=256,
            minibatch_size=32,
            update_epochs=6,
        ),
        checkpoint=CheckpointPolicy(
            checkpoint_interval_updates=50,
            validation_interval_updates=250,
            keep_last=10,
            best_metric="sharpe",
        ),
        validation=ValidationPolicy(
            symbol_limit=20,
            symbol_seed=2026,
            quick_days=5,
            periodic_device="cpu",
            final_device="auto",
        ),
        distributed=DistributedPolicy(strategy="auto"),
    )


def server_run_config() -> AgentRunConfig:
    base = formal_run_config()
    return replace(
        base,
        profile="server",
        parallel_envs=16,
        ppo=replace(base.ppo, minibatch_size=128),
        runtime=replace(base.runtime, env_backend="subprocess"),
        distributed=DistributedPolicy(strategy="auto"),
    )


def select_symbols(
    symbols: Iterable[str],
    *,
    limit: int,
    seed: int,
) -> tuple[str, ...]:
    available = sorted(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if symbol))
    if not available:
        raise ValueError("Agent universe cannot be empty.")
    if limit <= 0 or limit >= len(available):
        return tuple(available)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.asarray(available, dtype=object), size=int(limit), replace=False)
    return tuple(sorted(str(value) for value in selected))


def select_representative_symbols(
    symbols: Iterable[str],
    *,
    limit: int,
    seed: int,
) -> tuple[str, ...]:
    available = sorted(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if symbol))
    if not available:
        raise ValueError("Validation universe cannot be empty.")
    if limit >= len(available):
        return tuple(available)
    groups: dict[str, list[str]] = {}
    for symbol in available:
        groups.setdefault(_board_group(symbol), []).append(symbol)
    rng = np.random.default_rng(int(seed))
    for values in groups.values():
        rng.shuffle(values)
    ordered_groups = sorted(groups)
    selected: list[str] = []
    while len(selected) < limit:
        progressed = False
        for group in ordered_groups:
            values = groups[group]
            if values:
                selected.append(values.pop())
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return tuple(sorted(selected))


def _board_group(symbol: str) -> str:
    code, _, market = symbol.partition(".")
    if market == "BJ":
        return "beijing"
    if code.startswith("688"):
        return "star"
    if code.startswith(("300", "301")):
        return "chinext"
    return market.lower() or "other"


def _merged(defaults: dict[str, Any], supplied: Any) -> dict[str, Any]:
    result = dict(defaults)
    if isinstance(supplied, dict):
        result.update(supplied)
    return result
