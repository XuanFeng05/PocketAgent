from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from agent_layer.data.cache_builder import (
    AgentCacheBuildConfig,
    build_agent_cache,
    inspect_agent_cache,
)
from agent_layer.data.cache_schema import normalize_symbol_list


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fast single-symbol Agent tensor cache from feature_parts."
    )
    parser.add_argument(
        "--feature-dir",
        default=str(PROJECT_ROOT / "runtime_layer" / "features"),
        help="Feature dataset directory containing feature_parts/ and feature_parts_manifest.json.",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "runtime_layer" / "agent_cache" / "latest"),
        help="Output Agent cache directory.",
    )
    parser.add_argument("--symbols", default="", help="Comma-separated symbols or a txt file path.")
    parser.add_argument("--frequencies", default="", help="Comma-separated frequencies. Defaults to model input contract order.")
    parser.add_argument("--start", default=None, help="Optional decision start date/time.")
    parser.add_argument("--end", default=None, help="Optional decision end date/time.")
    parser.add_argument("--stages", default="", help="Comma-separated stages, e.g. open_auction,bar_close.")
    parser.add_argument("--workers", type=int, default=1, help="Number of symbol workers.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Decision ids per batch while reading feature_parts.")
    parser.add_argument("--max-decisions-per-symbol", type=int, default=0, help="Debug limit per symbol. 0 means no limit.")
    parser.add_argument("--reset", action="store_true", help="Delete the output cache directory before building.")
    parser.add_argument("--inspect", action="store_true", help="Inspect an existing cache instead of building it.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out = _resolve_path(args.out)
    if args.inspect:
        print(json.dumps(inspect_agent_cache(out), ensure_ascii=False, indent=2))
        return 0

    config = AgentCacheBuildConfig(
        feature_dir=_resolve_path(args.feature_dir),
        output_dir=out,
        symbols=_parse_symbols(args.symbols),
        frequencies=_parse_csv(args.frequencies),
        start=args.start,
        end=args.end,
        stages=_parse_csv(args.stages),
        workers=max(1, int(args.workers)),
        chunk_size=max(1, int(args.chunk_size)),
        reset=bool(args.reset),
        max_decisions_per_symbol=(
            int(args.max_decisions_per_symbol)
            if int(args.max_decisions_per_symbol or 0) > 0
            else None
        ),
    )

    def progress(item: dict[str, object]) -> None:
        phase = str(item.get("phase") or "")
        if phase == "starting":
            print(
                f"[agent-cache] building {item.get('symbols')} symbols "
                f"with {item.get('workers')} worker(s) "
                f"storage={item.get('storage') or '-'} -> {item.get('output_dir')}",
                flush=True,
            )
        elif phase == "symbol_done":
            summary = dict(item.get("summary") or {})
            status = "error" if summary.get("error") else "ok"
            message = (
                f"[agent-cache] {item.get('index')}: {summary.get('symbol')} "
                f"{status} decisions={summary.get('decision_count', 0)}"
            )
            if summary.get("error"):
                message += f" error={summary.get('error')}"
            print(message, flush=True)

    summary = build_agent_cache(config, progress_callback=progress)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("ok") else 1


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_csv(value: str) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip()) or None


def _parse_symbols(value: str) -> tuple[str, ...] | None:
    if not value:
        return None
    candidate = _resolve_path(value)
    if candidate.exists() and candidate.is_file():
        return normalize_symbol_list(candidate.read_text(encoding="utf-8").splitlines())
    return normalize_symbol_list(value.split(","))


if __name__ == "__main__":
    raise SystemExit(main())
