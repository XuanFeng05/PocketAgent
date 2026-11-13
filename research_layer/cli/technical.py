"""Forward-adjusted technical research, using normalized price returns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_layer.technical_runner import run_technical_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="前复权日线技术研究：标准化净值，收盘信号、下一开盘切换仓位。")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", type=Path)
    source.add_argument("--csv", type=Path, help="CSV must explicitly contain adjust=pre")
    parser.add_argument("--symbol")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--strategy", default="sma_trend", choices=["sma_trend", "breakout", "mean_reversion", "cash", "buy_hold"])
    parser.add_argument("--parameters", default="{}", help="JSON object of strategy parameters")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="单边比例摩擦，默认 0；1 bp = 0.01%%")
    parser.add_argument("--output", type=Path, default=Path("runtime_layer/reports/technical"))
    args = parser.parse_args()
    parameters = json.loads(args.parameters)
    if not isinstance(parameters, dict):
        parser.error("--parameters must be a JSON object")
    config = {"source_type": "csv" if args.csv else "market", "source_path": str(args.csv or args.db),
              "adjust": "pre", "symbol": args.symbol, "start": args.start, "end": args.end,
              "strategy": args.strategy, "parameters": parameters, "cost_bps": args.cost_bps}
    output = run_technical_experiment(config, args.output)
    print(json.dumps({"run_id": output["run_id"], "status": output["status"], "path": output["path"],
                      "price_basis": "pre", "comparison": output["result"]["comparison"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
