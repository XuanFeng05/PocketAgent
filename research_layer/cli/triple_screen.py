"""Run a frozen Triple Screen specification over an explicit symbol list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_layer.triple_runner import run_triple_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="前复权三重滤网技术研究：显式股票清单、标准化净值、完整失败记录。")
    parser.add_argument("--config", type=Path, help="JSON configuration, required unless --resume is supplied")
    parser.add_argument("--symbols-file", type=Path, help="One symbol per line; replaces the config symbol list")
    parser.add_argument("--resume", type=Path, help="Verify and reuse completed artifacts into a new run")
    parser.add_argument("--output", type=Path, default=Path("runtime_layer/reports/triple_screen"))
    args = parser.parse_args()
    if not args.config and not args.resume:
        parser.error("Provide --config or --resume.")
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    else:
        config = json.loads((args.resume / "config.json").read_text(encoding="utf-8-sig"))["configuration"]
    if args.symbols_file:
        config["symbols"] = [s.strip() for s in args.symbols_file.read_text(encoding="utf-8-sig").splitlines()
                             if s.strip() and not s.lstrip().startswith("#")]
        config.pop("symbol", None)

    def progress(event):
        if event.get("event") != "task_completed":
            return
        row = event["outcome"]
        print(json.dumps({"event": event["event"], "completed": event["completed"], "total": event["total"],
                          "symbol": row["symbol"], "cost_bps": row["cost_bps"], "status": row["status"],
                          "reused": row.get("reused", False), "error": row.get("error")}, ensure_ascii=False), flush=True)

    output = run_triple_experiment(config, args.output, progress_callback=progress, resume_from=args.resume)
    print(json.dumps({"run_id": output["run_id"], "status": output["status"], "path": output["path"],
                      "by_cost": output["summary"]["by_cost"]}, ensure_ascii=False, allow_nan=False))
    if output["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
