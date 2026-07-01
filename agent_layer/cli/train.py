from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from agent_layer.data import AgentTimelineLoader, open_agent_timeline_loader
from agent_layer.environment import ExecutionConfig, SingleSymbolTradingEnv
from agent_layer.experiments import build_walk_forward_splits, save_agent_checkpoint
from agent_layer.models import SingleSymbolMultiFrequencyPolicy, SingleSymbolMultiFrequencyPolicyConfig
from agent_layer.training import PPOConfig, PPOTrainer
from evaluation_layer.backtest import evaluate_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PocketAgent with a single-symbol multi-frequency PPO policy.")
    parser.add_argument(
        "--store",
        default="runtime_layer/reports/feature_dataset",
        help="Validated Feature Parts Dataset.",
    )
    parser.add_argument("--symbols-file", default=None, help="Optional fixed Agent universe file.")
    parser.add_argument("--fold", type=int, default=3, help="Walk-forward fold to train and validate.")
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--episode-days", type=int, default=252)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print splits without training.")
    parser.add_argument(
        "--output",
        default=None,
        help="Checkpoint path. Default: runtime_layer/models/agent/agent_<timestamp>.pt",
    )
    args = parser.parse_args()

    universe = _read_symbols(Path(args.symbols_file)) if args.symbols_file else None
    loader = open_agent_timeline_loader(args.store, universe=universe)
    dates = loader.trading_dates()
    splits = build_walk_forward_splits(dates)
    if not 1 <= args.fold <= len(splits.folds):
        raise ValueError(f"Fold must be between 1 and {len(splits.folds)}.")
    fold = splits.folds[args.fold - 1]
    configuration = {
        "store": str(args.store),
        "schema_hash": loader.schema_hash,
        "symbols": len(loader.universe),
        "frequencies": list(loader.frequencies),
        "selected_fold": args.fold,
        "splits": splits.payload(),
        "seed": args.seed,
        "total_steps": args.total_steps,
        "episode_days": args.episode_days,
    }
    print(json.dumps(configuration, ensure_ascii=False, indent=2, default=str))
    if args.dry_run:
        return 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    rng = np.random.default_rng(args.seed)
    train_dates = [date for date in dates if fold.train.start <= date <= fold.train.end]

    symbol_loaders: dict[str, AgentTimelineLoader] = {}

    def loader_for_symbol(symbol: str) -> AgentTimelineLoader:
        normalized = str(symbol).upper()
        selected = symbol_loaders.get(normalized)
        if selected is None:
            selected = loader.for_universe((normalized,))
            symbol_loaders[normalized] = selected
        return selected

    def environment_factory() -> SingleSymbolTradingEnv:
        episode_days = min(max(1, args.episode_days), len(train_dates))
        attempts = max(4, len(loader.universe) * 2)
        last_error: Exception | None = None
        for _ in range(attempts):
            symbol = str(rng.choice(loader.universe)).upper()
            start_index = int(rng.integers(0, len(train_dates) - episode_days + 1))
            try:
                return SingleSymbolTradingEnv(
                    loader_for_symbol(symbol),
                    start=train_dates[start_index],
                    end=train_dates[start_index + episode_days - 1],
                    execution_config=ExecutionConfig(),
                )
            except ValueError as exc:
                last_error = exc
        raise ValueError("Unable to create a single-symbol training episode.") from last_error

    sample_environment = environment_factory()
    sample_observation, _ = sample_environment.reset()
    model = SingleSymbolMultiFrequencyPolicy(
        SingleSymbolMultiFrequencyPolicyConfig.from_observation(sample_observation)
    )
    ppo_config = PPOConfig()
    trainer = PPOTrainer(model, config=ppo_config, device=device)
    summary = trainer.train(
        environment_factory,
        total_steps=args.total_steps,
        progress_callback=lambda item: print(json.dumps(item, ensure_ascii=False)),
    )

    validation_symbol = loader.universe[0]
    validation_environment = SingleSymbolTradingEnv(
        loader_for_symbol(validation_symbol),
        start=fold.validation.start,
        end=fold.validation.end,
        execution_config=ExecutionConfig(),
    )
    validation = evaluate_policy(
        validation_environment,
        model,
        device=device,
        deterministic=True,
    )
    output = Path(args.output) if args.output else Path(
        "runtime_layer/models/agent"
    ) / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    checkpoint, manifest = save_agent_checkpoint(
        output,
        model=model,
        optimizer=trainer.optimizer,
        schema_hash=loader.schema_hash,
        universe=loader.universe,
        ppo_config=ppo_config,
        execution_config=ExecutionConfig(),
        experiment={
            "selected_fold": args.fold,
            "splits": splits.payload(),
            "validation": validation.payload(),
        },
        training_state=asdict(summary),
    )
    print(json.dumps({"checkpoint": str(checkpoint), "manifest": str(manifest), "validation": validation.metrics}, ensure_ascii=False, indent=2))
    return 0


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return torch.device(value)


def _read_symbols(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [item.strip().upper() for item in text.replace("\n", ",").split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
