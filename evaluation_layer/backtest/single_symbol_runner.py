from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from agent_layer.data import open_agent_timeline_loader
from agent_layer.environment import ExecutionConfig, SingleSymbolTradingEnv
from agent_layer.experiments import build_walk_forward_splits, load_agent_checkpoint
from evaluation_layer.backtest.runner import evaluate_policy


@dataclass(frozen=True)
class SingleSymbolEvaluationSummary:
    symbol_results: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        returns = [float(item.get("metrics", {}).get("total_return", 0.0)) for item in self.symbol_results]
        excess = [float(item.get("excess_return", 0.0)) for item in self.symbol_results]
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
    for symbol in selected:
        symbol_loader = loader.for_universe((symbol,))
        environment = SingleSymbolTradingEnv(
            symbol_loader,
            start=eval_start,
            end=eval_end,
            execution_config=ExecutionConfig(**payload.get("execution_config", {})),
        )
        result = evaluate_policy(environment, model, device=target, deterministic=deterministic).payload()
        navs = [float(row.get("nav") or 0.0) for row in result.get("daily_nav", [])]
        buy_hold_return = 0.0
        if len(navs) >= 2 and navs[0] > 0:
            # Placeholder benchmark until evaluation has direct close-price baseline from cache.
            buy_hold_return = 0.0
        result.update(
            {
                "symbol": symbol,
                "buy_hold_return": buy_hold_return,
                "excess_return": float(result.get("metrics", {}).get("total_return", 0.0)) - buy_hold_return,
            }
        )
        results.append(result)
    return SingleSymbolEvaluationSummary(tuple(results))
