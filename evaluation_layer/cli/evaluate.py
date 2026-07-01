from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from agent_layer.data import open_agent_timeline_loader
from agent_layer.environment import ExecutionConfig, SingleSymbolTradingEnv
from agent_layer.experiments import load_agent_checkpoint
from evaluation_layer.backtest import evaluate_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen PocketAgent checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--store",
        default="runtime_layer/reports/feature_dataset",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbol", default=None, help="Symbol to evaluate. Defaults to the first checkpoint universe symbol.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    model, payload = load_agent_checkpoint(args.checkpoint, map_location=device)
    checkpoint_universe = tuple(str(symbol).upper() for symbol in (payload.get("universe") or ()))
    if not checkpoint_universe:
        raise ValueError("Checkpoint does not contain a training universe.")
    selected_symbol = str(args.symbol or checkpoint_universe[0]).upper()
    if selected_symbol not in checkpoint_universe:
        raise ValueError(f"Symbol {selected_symbol} is not in the checkpoint universe.")
    loader = open_agent_timeline_loader(args.store, universe=(selected_symbol,), frequencies=model.frequencies, validate_store=False)
    if loader.schema_hash != payload["schema_hash"]:
        raise ValueError("Checkpoint and Feature Parts schema hashes do not match.")
    test_range = payload.get("experiment", {}).get("splits", {}).get("test", {})
    start = args.start or test_range.get("start")
    end = args.end or test_range.get("end")
    if not start or not end:
        raise ValueError("Evaluation requires start/end or an embedded frozen test split.")
    environment = SingleSymbolTradingEnv(
        loader,
        start=start,
        end=end,
        execution_config=ExecutionConfig(**payload["execution_config"]),
    )
    result = evaluate_policy(
        environment,
        model,
        device=device,
        deterministic=not args.stochastic,
    )
    output = Path(args.output) if args.output else Path(
        "runtime_layer/reports/evaluation"
    ) / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "schema_hash": loader.schema_hash,
                "symbol": selected_symbol,
                "date_range": {"start": start, "end": end},
                **result.payload(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "metrics": result.metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
