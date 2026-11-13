from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch

from agent_layer.data import open_agent_timeline_loader
from agent_layer.environment import ExecutionConfig, SingleSymbolTradingEnv
from agent_layer.experiments import build_walk_forward_splits, load_agent_checkpoint
from evaluation_layer.backtest.runner import buy_hold_baseline, evaluate_policy


@dataclass(frozen=True)
class SingleSymbolEvaluationSummary:
    symbol_results: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        returns = [float(item.get("metrics", {}).get("total_return", 0.0)) for item in self.symbol_results]
        excess = [
            float(item["excess_return"])
            for item in self.symbol_results
            if isinstance(item.get("excess_return"), (int, float))
        ]
        return {
            "symbols": len(self.symbol_results),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "median_return": float(np.median(returns)) if returns else 0.0,
            "mean_excess_return": float(np.mean(excess)) if excess else 0.0,
            "win_rate_vs_buy_hold": float(np.mean([value > 0 for value in excess])) if excess else 0.0,
            "symbol_results": list(self.symbol_results),
        }


@torch.inference_mode()
def evaluate_checkpoint_by_symbol(
    *,
    checkpoint: str,
    store: str,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
    max_symbols: int | None = None,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> SingleSymbolEvaluationSummary:
    model, payload = load_agent_checkpoint(checkpoint, map_location="cpu")
    universe = tuple(str(symbol).upper() for symbol in (symbols or payload.get("universe") or ()))
    if not universe:
        raise ValueError("Evaluation requires at least one checkpoint symbol.")
    selected = universe[: int(max_symbols)] if max_symbols else universe
    loader = open_agent_timeline_loader(store, universe=selected, frequencies=model.frequencies, validate_store=False)
    expected_schema = str(payload.get("schema_hash") or "")
    if expected_schema and loader.schema_hash != expected_schema:
        raise ValueError(f"Feature schema mismatch: model {expected_schema} != dataset {loader.schema_hash}")
    split = payload.get("experiment", {}).get("splits", {}).get("test", {})
    eval_start = start or split.get("start")
    eval_end = end or split.get("end")
    if not eval_start or not eval_end:
        dates = loader.trading_dates()
        splits = build_walk_forward_splits(dates)
        eval_start = str(splits.test.start.date())
        eval_end = str(splits.test.end.date())

    target = torch.device(device)
    model.to(target)
    model.eval()
    results: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(selected, start=1):
        symbol_loader = loader.for_universe((symbol,))
        environment = SingleSymbolTradingEnv(
            symbol_loader,
            start=eval_start,
            end=eval_end,
            execution_config=ExecutionConfig(**payload.get("execution_config", {})),
            enforce_action_contract=True,
        )
        result = evaluate_policy(environment, model, device=target, deterministic=deterministic).payload()
        initial_nav = float(result.get("metrics", {}).get("initial_nav") or 0.0)
        baseline = buy_hold_baseline(
            symbol_loader,
            symbol=symbol,
            start=str(eval_start),
            end=str(eval_end),
            initial_nav=initial_nav,
            execution_config=environment.executor.config,
        )
        buy_hold_return = baseline.get("total_return")
        total_return = float(result.get("metrics", {}).get("total_return", 0.0))
        result.update(
            {
                "symbol": symbol,
                "buy_hold_return": buy_hold_return,
                "excess_return": (
                    total_return - float(buy_hold_return)
                    if isinstance(buy_hold_return, (int, float))
                    else None
                ),
            }
        )
        results.append(result)
        if progress_callback:
            progress_callback(symbol_index, len(selected), result)
    return SingleSymbolEvaluationSummary(tuple(results))
