from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_layer.offline import (prepare_task_bundle, inspect_task_bundle, run_task_bundle,
                                    export_result_bundle, import_result_bundle, aggregate_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline, checksummed CPU task exchange; no package code is executed")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--shards", type=int, default=1)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--bundle", required=True, type=Path)
    worker = commands.add_parser("run")
    worker.add_argument("--bundle", required=True, type=Path)
    worker.add_argument("--workspace", required=True, type=Path)
    worker.add_argument("--shard", type=int, default=0)
    worker.add_argument("--task-id", action="append")
    worker.add_argument("--retry-failed", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--workspace", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    merge = commands.add_parser("import")
    merge.add_argument("--results", required=True, type=Path)
    merge.add_argument("--plan", required=True, type=Path)
    merge.add_argument("--collection", required=True, type=Path)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--plan", required=True, type=Path)
    aggregate.add_argument("--collection", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_task_bundle(json.loads(args.config.read_text(encoding="utf-8-sig")), args.output, shard_count=args.shards)
        result = {"plan_id": result["plan_id"], "tasks": len(result["tasks"]), "path": str(args.output)}
    elif args.command == "inspect":
        result = inspect_task_bundle(args.bundle)
    elif args.command == "run":
        result = run_task_bundle(args.bundle, args.workspace, shard_index=args.shard, task_ids=args.task_id, retry_failed=args.retry_failed)
    elif args.command == "export":
        result = export_result_bundle(args.workspace, args.output)
    elif args.command == "import":
        result = import_result_bundle(args.results, args.collection, args.plan)
    else:
        result = aggregate_results(args.collection, args.plan)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
