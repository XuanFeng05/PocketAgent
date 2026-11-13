from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_layer.runner import run_experiment
from strategy_layer.registry import strategy_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU strategy research; demo data is synthetic.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--csv", type=Path)
    source.add_argument("--db", type=Path)
    parser.add_argument("--symbol")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--strategy", default="sma_trend", choices=[item["name"] for item in strategy_catalog()])
    parser.add_argument("--parameters", default="{}", help="JSON object, e.g. fast_window / slow_window")
    parser.add_argument("--execution", default="{}", help="Explicit ExecutionConfig overrides as JSON")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runtime_layer/reports/research"))
    args = parser.parse_args()
    config = {"source_type": "demo" if args.demo else "csv" if args.csv else "market",
              "source_path": str(args.csv or args.db or ""), "symbol": args.symbol,
              "start": args.start, "end": args.end, "strategy": args.strategy,
              "parameters": json.loads(args.parameters), "execution": json.loads(args.execution),
              "include_baselines": not args.no_baselines}
    result = run_experiment(config, args.output)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"], "path": result["path"],
                      "synthetic": result["result"]["synthetic"], "metrics": result["result"]["strategy"]["metrics"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
