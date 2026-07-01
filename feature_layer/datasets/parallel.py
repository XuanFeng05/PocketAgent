from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd

from feature_layer.datasets.builder import FeatureDatasetConfig
from feature_layer.datasets.compact import (
    CompactFeatureParquetWriter,
    CompactFeatureStoreWriter,
    build_compact_feature_dataset_from_duckdb,
    write_compact_feature_store,
)
from feature_layer.specs import DEFAULT_FEATURE_SPEC, FeatureSpec


@dataclass(frozen=True)
class FeatureChunkBuildTask:
    chunk_index: int
    db_path: str
    chunk_store_path: str
    symbols: tuple[str, ...]
    adjust: str
    start: str | None
    end: str | None
    config: FeatureDatasetConfig
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC
    preview_limit: int = 1000
    low_memory_mode: bool = True
    chunk_parts_dir: str | None = None
    output_format: str = "duckdb"
    source_cache_dir: str | None = None


@dataclass(frozen=True)
class FeatureChunkBuildResult:
    chunk_index: int
    chunk_store_path: str
    symbols: tuple[str, ...]
    summary: dict[str, Any]
    compiled_model_input: dict[str, Any]
    preview_decisions: pd.DataFrame
    preview_decision_context: pd.DataFrame
    preview_constraints: pd.DataFrame
    timings: dict[str, float]
    chunk_parts_dir: str | None = None
    output_format: str = "duckdb"


def build_compact_feature_chunk(task: FeatureChunkBuildTask) -> FeatureChunkBuildResult:
    """Worker entrypoint: build one independent compact feature chunk store.

    In low-memory mode, a worker receives a chunk of symbols but builds and
    appends them one symbol at a time.  This keeps the parallel design while
    preventing each worker from holding a full multi-symbol compact dataset in
    memory.
    """

    if task.low_memory_mode and len(task.symbols) > 1:
        return _build_compact_feature_chunk_low_memory(task)
    return _build_compact_feature_chunk_batch(task)


def _empty_preview_frames() -> dict[str, list[pd.DataFrame]]:
    return {"decisions": [], "decision_context": [], "constraints": []}


def _append_preview(
    previews: dict[str, list[pd.DataFrame]],
    key: str,
    frame: pd.DataFrame,
    *,
    preview_limit: int,
) -> None:
    captured = sum(len(part) for part in previews[key])
    if captured < preview_limit:
        previews[key].append(frame.head(preview_limit - captured).copy())


def _blank_preview_result(previews: dict[str, list[pd.DataFrame]], key: str) -> pd.DataFrame:
    return pd.concat(previews[key], ignore_index=True) if previews[key] else pd.DataFrame()


def _accumulate_chunk_summary(
    totals: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    totals["decisions"] = int(totals.get("decisions", 0)) + int(summary.get("decisions", 0))
    totals["decision_context_rows"] = int(totals.get("decision_context_rows", 0)) + int(summary.get("decision_context_rows", 0))
    totals["constraint_rows"] = int(totals.get("constraint_rows", 0)) + int(summary.get("constraint_rows", 0))
    totals["decision_index_rows"] = int(totals.get("decision_index_rows", 0)) + int(summary.get("decision_index_rows", 0))
    totals["snapshot_rows"] = int(totals.get("snapshot_rows", 0)) + int(summary.get("snapshot_rows", 0))
    totals["st_decisions"] = int(totals.get("st_decisions", 0)) + int(summary.get("st_decisions", 0))
    market_rows = totals.setdefault("market_rows", {})
    for freq, rows in dict(summary.get("market_rows", {})).items():
        market_rows[str(freq)] = int(market_rows.get(str(freq), 0)) + int(rows)


def _compact_summary(compact: Any) -> dict[str, Any]:
    dataset = compact.dataset
    summary = compact.summary()
    summary.update({
        "constraint_rows": int(len(dataset.constraints)),
        "decision_context_rows": int(len(dataset.decision_context)),
        "decision_index_rows": int(len(compact.decision_index)),
        "snapshot_rows": int(len(compact.decision_snapshots)),
    })
    return summary


def _build_compact_feature_chunk_low_memory(task: FeatureChunkBuildTask) -> FeatureChunkBuildResult:
    start_total = perf_counter()
    chunk_path = Path(task.chunk_store_path)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    if chunk_path.exists():
        chunk_path.unlink()

    preview_limit = max(0, int(task.preview_limit))
    previews = _empty_preview_frames()
    totals: dict[str, Any] = {
        "decisions": 0,
        "decision_context_rows": 0,
        "constraint_rows": 0,
        "decision_index_rows": 0,
        "snapshot_rows": 0,
        "st_decisions": 0,
        "market_rows": {},
    }
    processed_symbols: list[str] = []
    compiled_model_input: dict[str, Any] | None = None
    build_seconds = 0.0
    write_seconds = 0.0

    if task.output_format == "parquet":
        parts_dir = Path(task.chunk_parts_dir or chunk_path.with_suffix(".parts"))
        writer = CompactFeatureParquetWriter(parts_dir, reset=True)
    else:
        parts_dir = None
        writer = CompactFeatureStoreWriter(chunk_path, reset=True)
    try:
        for symbol in task.symbols:
            symbol_start = perf_counter()
            compact = build_compact_feature_dataset_from_duckdb(
                task.db_path,
                symbols=(symbol,),
                adjust=task.adjust,
                start=task.start,
                end=task.end,
                config=task.config,
                spec=task.spec,
                source_cache_dir=task.source_cache_dir,
            )
            build_seconds += perf_counter() - symbol_start

            write_start = perf_counter()
            writer.append(compact)
            write_seconds += perf_counter() - write_start

            dataset = compact.dataset
            summary = _compact_summary(compact)
            _accumulate_chunk_summary(totals, summary)
            for produced_symbol in tuple(dataset.requested_symbols or (symbol,)):
                if str(produced_symbol) not in processed_symbols:
                    processed_symbols.append(str(produced_symbol))
            compiled_model_input = compact.compiled_model_input
            _append_preview(previews, "decisions", dataset.decisions, preview_limit=preview_limit)
            _append_preview(previews, "decision_context", dataset.decision_context, preview_limit=preview_limit)
            _append_preview(previews, "constraints", dataset.constraints, preview_limit=preview_limit)

            del dataset
            del compact
            gc.collect()
        if isinstance(writer, CompactFeatureStoreWriter):
            writer.finalize(create_indexes=False)
    except Exception:
        if isinstance(writer, CompactFeatureStoreWriter):
            writer.close()
        if chunk_path.exists():
            try:
                chunk_path.unlink()
            except OSError:
                pass
        raise
    finally:
        if isinstance(writer, CompactFeatureStoreWriter):
            writer.close()

    if compiled_model_input is None:
        raise ValueError(f"Feature chunk {task.chunk_index} produced no model input metadata.")
    summary = {
        "spec": task.spec.name,
        "frequencies": list(task.config.frequencies),
        "requested_symbols": list(processed_symbols),
        "decisions": int(totals["decisions"]),
        "decision_context_rows": int(totals["decision_context_rows"]),
        "constraint_rows": int(totals["constraint_rows"]),
        "market_rows": dict(totals["market_rows"]),
        "decision_index_rows": int(totals["decision_index_rows"]),
        "snapshot_rows": int(totals["snapshot_rows"]),
        "st_decisions": int(totals["st_decisions"]),
        "model_input_schema_hash": compiled_model_input.get("schema_hash"),
    }
    return FeatureChunkBuildResult(
        chunk_index=task.chunk_index,
        chunk_store_path=str(chunk_path),
        symbols=tuple(processed_symbols or task.symbols),
        chunk_parts_dir=str(parts_dir) if parts_dir is not None else None,
        output_format=str(task.output_format),
        summary=summary,
        compiled_model_input=compiled_model_input,
        preview_decisions=_blank_preview_result(previews, "decisions"),
        preview_decision_context=_blank_preview_result(previews, "decision_context"),
        preview_constraints=_blank_preview_result(previews, "constraints"),
        timings={
            "build_seconds": float(build_seconds),
            "write_seconds": float(write_seconds),
            "total_seconds": float(perf_counter() - start_total),
            "low_memory_symbols": float(len(processed_symbols or task.symbols)),
        },
    )


def _build_compact_feature_chunk_batch(task: FeatureChunkBuildTask) -> FeatureChunkBuildResult:
    start_total = perf_counter()
    chunk_path = Path(task.chunk_store_path)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    if chunk_path.exists():
        chunk_path.unlink()

    start_build = perf_counter()
    compact = build_compact_feature_dataset_from_duckdb(
        task.db_path,
        symbols=task.symbols,
        adjust=task.adjust,
        start=task.start,
        end=task.end,
        config=task.config,
        spec=task.spec,
        source_cache_dir=task.source_cache_dir,
    )
    build_seconds = perf_counter() - start_build

    start_write = perf_counter()
    parts_dir = Path(task.chunk_parts_dir or chunk_path.with_suffix(".parts")) if task.output_format == "parquet" else None
    if parts_dir is not None:
        writer = CompactFeatureParquetWriter(parts_dir, reset=True)
        writer.append(compact)
    else:
        write_compact_feature_store(compact, chunk_path, append=False, create_indexes=False)
    write_seconds = perf_counter() - start_write

    dataset = compact.dataset
    summary = _compact_summary(compact)
    preview_limit = max(0, int(task.preview_limit))
    return FeatureChunkBuildResult(
        chunk_index=task.chunk_index,
        chunk_store_path=str(chunk_path),
        symbols=tuple(dataset.requested_symbols or task.symbols),
        chunk_parts_dir=str(parts_dir) if parts_dir is not None else None,
        output_format=str(task.output_format),
        summary=summary,
        compiled_model_input=compact.compiled_model_input,
        preview_decisions=dataset.decisions.head(preview_limit).copy(),
        preview_decision_context=dataset.decision_context.head(preview_limit).copy(),
        preview_constraints=dataset.constraints.head(preview_limit).copy(),
        timings={
            "build_seconds": float(build_seconds),
            "write_seconds": float(write_seconds),
            "total_seconds": float(perf_counter() - start_total),
            "low_memory_symbols": 0.0,
        },
    )


def chunk_symbols(symbols: Iterable[str], chunk_size: int) -> list[tuple[int, tuple[str, ...]]]:
    values = tuple(str(symbol) for symbol in symbols)
    size = max(1, int(chunk_size))
    return [
        (index // size, tuple(values[index:index + size]))
        for index in range(0, len(values), size)
    ]
