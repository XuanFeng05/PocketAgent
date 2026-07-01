from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

from agent_layer.data.single_symbol_reader import SingleSymbolReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Agent single-symbol tensor cache reads.")
    parser.add_argument("--cache", required=True, help="Agent cache directory.")
    parser.add_argument("--symbol", default="", help="Optional symbol to benchmark.")
    parser.add_argument("--max-steps", type=int, default=512, help="Maximum episode steps to materialize.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated materialization reads.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cache_dir = _resolve_path(args.cache)
    reader = SingleSymbolReader(cache_dir)
    symbol = args.symbol.strip().upper() if args.symbol else reader.universe[0]
    repeats = max(1, int(args.repeats))
    max_steps = max(1, int(args.max_steps))
    first_started = perf_counter()
    buffer = reader.episode_buffer(symbol, max_steps=max_steps)
    first_seconds = perf_counter() - first_started
    step_started = perf_counter()
    for index in range(len(buffer)):
        buffer.market_step(index)
    step_seconds = perf_counter() - step_started
    repeat_seconds = []
    for _ in range(repeats):
        started = perf_counter()
        reader.episode_buffer(symbol, max_steps=max_steps)
        repeat_seconds.append(perf_counter() - started)
    payload = {
        "cache": str(cache_dir),
        "symbol": symbol,
        "steps": len(buffer),
        "frequencies": list(reader.frequencies),
        "schema_hash": reader.schema_hash,
        "first_materialize_seconds": first_seconds,
        "market_step_seconds": step_seconds,
        "market_steps_per_second": len(buffer) / max(step_seconds, 1e-9),
        "repeat_materialize_seconds": repeat_seconds,
        "arrays": {
            freq: list(values.shape)
            for freq, values in buffer.market_sequences.items()
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
