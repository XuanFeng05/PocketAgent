from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_layer.data.cache_schema import (
    CONSTRAINTS_NAME,
    DECISION_CONTEXT_NAME,
    DECISIONS_NAME,
    EXECUTION_NAME,
    SYMBOL_METADATA_NAME,
    SymbolCacheMetadata,
    read_json,
    safe_frequency_name,
    write_json,
)


def write_array(path: str | Path, values: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(output)
    return output


def read_array(path: str | Path, *, mmap_mode: str | None = None) -> np.ndarray:
    return np.load(Path(path), mmap_mode=mmap_mode, allow_pickle=False)


def sequence_path(symbol_dir: str | Path, freq: str) -> Path:
    # Index-based cache v2 stores the continuous feature matrix here:
    # [feature_bar_count, channels].  It no longer stores expanded windows.
    return Path(symbol_dir) / f"features_{safe_frequency_name(freq)}.npy"


def feature_time_path(symbol_dir: str | Path, freq: str) -> Path:
    return Path(symbol_dir) / f"feature_time_{safe_frequency_name(freq)}.npy"


def end_index_path(symbol_dir: str | Path, freq: str) -> Path:
    return Path(symbol_dir) / f"end_index_{safe_frequency_name(freq)}.npy"


def valid_rows_path(symbol_dir: str | Path, freq: str) -> Path:
    return Path(symbol_dir) / f"valid_rows_{safe_frequency_name(freq)}.npy"


def snapshot_required_path(symbol_dir: str | Path, freq: str) -> Path:
    return Path(symbol_dir) / f"snapshot_required_{safe_frequency_name(freq)}.npy"


def snapshot_path(symbol_dir: str | Path, freq: str) -> Path:
    return Path(symbol_dir) / f"snapshot_{safe_frequency_name(freq)}.npy"


# Compatibility names.  Old v1 cache used these for expanded windows.  New code
# keeps the helpers importable but does not write mask_*.npy anymore.
def mask_path(symbol_dir: str | Path, freq: str) -> Path:
    return Path(symbol_dir) / f"mask_{safe_frequency_name(freq)}.npy"


def valid_ratio_path(symbol_dir: str | Path, freq: str) -> Path:
    return Path(symbol_dir) / f"valid_ratio_{safe_frequency_name(freq)}.npy"


def write_symbol_cache(
    symbol_dir: str | Path,
    *,
    metadata: SymbolCacheMetadata,
    decisions: pd.DataFrame,
    decision_context: np.ndarray,
    execution: np.ndarray,
    constraints: np.ndarray,
    feature_matrices: dict[str, np.ndarray],
    feature_times: dict[str, np.ndarray],
    end_indices: dict[str, np.ndarray],
    valid_rows: dict[str, np.ndarray],
    snapshot_required: dict[str, np.ndarray],
    snapshot_features: dict[str, np.ndarray],
    valid_ratios: dict[str, np.ndarray],
) -> Path:
    output_dir = Path(symbol_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(output_dir / DECISIONS_NAME, index=False)
    write_array(output_dir / DECISION_CONTEXT_NAME, np.asarray(decision_context, dtype=np.float32))
    write_array(output_dir / EXECUTION_NAME, np.asarray(execution, dtype=np.float32))
    write_array(output_dir / CONSTRAINTS_NAME, np.asarray(constraints, dtype=np.uint8))
    for freq in metadata.frequencies:
        write_array(sequence_path(output_dir, freq), np.asarray(feature_matrices[freq], dtype=np.float32))
        write_array(feature_time_path(output_dir, freq), np.asarray(feature_times[freq], dtype="datetime64[us]"))
        write_array(end_index_path(output_dir, freq), np.asarray(end_indices[freq], dtype=np.int64))
        write_array(valid_rows_path(output_dir, freq), np.asarray(valid_rows[freq], dtype=np.int16))
        write_array(snapshot_required_path(output_dir, freq), np.asarray(snapshot_required[freq], dtype=np.uint8))
        write_array(snapshot_path(output_dir, freq), np.asarray(snapshot_features[freq], dtype=np.float32))
        write_array(valid_ratio_path(output_dir, freq), np.asarray(valid_ratios[freq], dtype=np.float32))
    write_json(output_dir / SYMBOL_METADATA_NAME, metadata.to_json_dict())
    return output_dir


def read_symbol_metadata(symbol_dir: str | Path) -> SymbolCacheMetadata:
    return SymbolCacheMetadata.from_json_dict(read_json(Path(symbol_dir) / SYMBOL_METADATA_NAME))


def describe_symbol_cache(symbol_dir: str | Path) -> dict[str, Any]:
    metadata = read_symbol_metadata(symbol_dir)
    payload: dict[str, Any] = {
        "symbol": metadata.symbol,
        "decision_count": metadata.decision_count,
        "first_decision_time": metadata.first_decision_time,
        "last_decision_time": metadata.last_decision_time,
        "frequencies": list(metadata.frequencies),
        "schema_hash": metadata.schema_hash,
        "storage": metadata.storage,
        "arrays": {},
    }
    arrays: dict[str, Any] = payload["arrays"]  # type: ignore[assignment]
    for name in (DECISION_CONTEXT_NAME, EXECUTION_NAME, CONSTRAINTS_NAME):
        path = Path(symbol_dir) / name
        if path.exists():
            arrays[name] = list(read_array(path, mmap_mode="r").shape)
    for freq in metadata.frequencies:
        for label, path in (
            (f"features_{freq}", sequence_path(symbol_dir, freq)),
            (f"end_index_{freq}", end_index_path(symbol_dir, freq)),
            (f"valid_rows_{freq}", valid_rows_path(symbol_dir, freq)),
            (f"snapshot_required_{freq}", snapshot_required_path(symbol_dir, freq)),
            (f"snapshot_{freq}", snapshot_path(symbol_dir, freq)),
            (f"valid_ratio_{freq}", valid_ratio_path(symbol_dir, freq)),
        ):
            if path.exists():
                arrays[label] = list(read_array(path, mmap_mode="r").shape)
    return payload
