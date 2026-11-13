from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from agent_layer.experiments import load_agent_checkpoint
from evaluation_layer.backtest import SingleSymbolEvaluationSummary, evaluate_checkpoint_by_symbol


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PocketAgent checkpoint one symbol at a time.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbols", default=None, help="Optional comma-separated checkpoint symbols.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-symbols", type=int, default=0, help="0 evaluates the full checkpoint universe.")
    parser.add_argument("--workers", type=int, default=1, help="CPU symbol workers. CUDA evaluation requires 1.")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.workers <= 0 or args.threads_per_worker <= 0:
        parser.error("--workers and --threads-per-worker must be positive.")
    if args.device == "cuda" and args.workers != 1:
        parser.error("CUDA evaluation supports exactly one worker.")
    requested_symbols = tuple(
        symbol.strip().upper()
        for symbol in str(args.symbols or "").split(",")
        if symbol.strip()
    ) or None

    output = Path(args.output) if args.output else Path("runtime_layer/reports/evaluation") / (
        f"universe_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    progress_path = output.with_suffix(".progress.json")
    completed_results: list[dict[str, Any]] = []

    def progress(completed: int, total: int, result: dict[str, Any]) -> None:
        completed_results.append(result)
        payload = {
            "status": "running",
            "completed": completed,
            "total": total,
            "latest_symbol": result.get("symbol"),
            "latest_total_return": (result.get("metrics") or {}).get("total_return"),
            "latest_excess_return": result.get("excess_return"),
            "symbol_results": completed_results,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(progress_path, payload)
        print(
            json.dumps(
                {key: payload[key] for key in ("completed", "total", "latest_symbol", "latest_total_return", "latest_excess_return")},
                ensure_ascii=False,
            ),
            flush=True,
        )

    try:
        if args.workers == 1:
            torch.set_num_threads(args.threads_per_worker)
            summary = evaluate_checkpoint_by_symbol(
                checkpoint=args.checkpoint,
                store=args.store,
                symbols=requested_symbols,
                start=args.start,
                end=args.end,
                device=args.device,
                max_symbols=args.max_symbols or None,
                progress_callback=progress,
            ).payload()
        else:
            symbols = _checkpoint_symbols(
                args.checkpoint,
                args.max_symbols or None,
                requested_symbols=requested_symbols,
            )
            worker_args = [
                {
                    "checkpoint": args.checkpoint,
                    "store": args.store,
                    "symbol": symbol,
                    "start": args.start,
                    "end": args.end,
                    "device": args.device,
                }
                for symbol in symbols
            ]
            results_by_symbol: dict[str, dict[str, Any]] = {}
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(worker_args)),
                mp_context=get_context("spawn"),
                initializer=_configure_worker,
                initargs=(args.threads_per_worker,),
            ) as pool:
                futures = {pool.submit(_evaluate_one_symbol, item): item["symbol"] for item in worker_args}
                for future in as_completed(futures):
                    result = future.result()
                    results_by_symbol[str(result["symbol"])] = result
                    progress(len(results_by_symbol), len(symbols), result)
            ordered = tuple(results_by_symbol[symbol] for symbol in symbols)
            summary = SingleSymbolEvaluationSummary(ordered).payload()
    except BaseException as exc:
        _write_json(
            progress_path,
            {
                "status": "failed",
                "completed": len(completed_results),
                "symbol_results": completed_results,
                "error": str(exc),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        output,
        {
            "checkpoint": args.checkpoint,
            "store": args.store,
            "date_range": {"start": args.start, "end": args.end},
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **summary,
        },
    )
    _write_json(
        progress_path,
        {
            "status": "completed",
            "completed": summary["symbols"],
            "total": summary["symbols"],
            "output": str(output),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps({"output": str(output), **{k: summary[k] for k in summary if k != "symbol_results"}}, ensure_ascii=False, indent=2))
    return 0


def _checkpoint_symbols(
    checkpoint: str,
    max_symbols: int | None,
    *,
    requested_symbols: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    model, payload = load_agent_checkpoint(checkpoint, map_location="cpu")
    del model
    symbols = tuple(str(symbol).upper() for symbol in (payload.get("universe") or ()))
    if not symbols:
        raise ValueError("Checkpoint does not contain a training universe.")
    if requested_symbols:
        unknown = sorted(set(requested_symbols).difference(symbols))
        if unknown:
            raise ValueError("Requested symbols are absent from the checkpoint: " + ", ".join(unknown))
        symbols = requested_symbols
    return symbols[:max_symbols] if max_symbols else symbols


def _evaluate_one_symbol(args: dict[str, Any]) -> dict[str, Any]:
    summary = evaluate_checkpoint_by_symbol(
        checkpoint=str(args["checkpoint"]),
        store=str(args["store"]),
        symbols=(str(args["symbol"]),),
        start=args.get("start"),
        end=args.get("end"),
        device=str(args["device"]),
        max_symbols=1,
    )
    return summary.symbol_results[0]


def _configure_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
