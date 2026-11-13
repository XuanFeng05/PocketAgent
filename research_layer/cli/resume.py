from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_layer.resume import resume_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume verified completed CPU trial/fold checkpoints into a new run")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    result = resume_experiment(args.run, args.output, retry_failed=args.retry_failed)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"], "path": result["path"],
                      "reused_run": result.get("reused_run", False), "resume": result.get("resume")}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
