from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_layer.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded CPU parameter research and walk-forward validation")
    parser.add_argument("--config", required=True, type=Path, help="JSON experiment configuration")
    parser.add_argument("--output", type=Path, default=Path("runtime_layer/reports/research"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    output = run_experiment(config, args.output)
    result = output["result"]
    print(json.dumps({"run_id": output["run_id"], "status": output["status"], "path": output["path"],
                      "mode": result["mode"], "synthetic": result["synthetic"],
                      "selection_scope": result["selection_scope"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
