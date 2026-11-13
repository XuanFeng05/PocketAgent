from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from agent_layer.data.cache_schema import (
    AGENT_CACHE_VERSION,
    CACHE_MANIFEST_NAME,
    CACHE_STORAGE_INDEX_BASED,
    CONSTRAINT_COLUMNS,
    CONSTRAINTS_NAME,
    DECISION_CONTEXT_NAME,
    DECISIONS_NAME,
    EPISODE_INDEX_NAME,
    EXECUTION_COLUMNS,
    EXECUTION_NAME,
    REQUIRED_RUNTIME_CONTRACT_FIELDS,
    SOURCE_DATASET_FINGERPRINT_VERSION,
    AgentCacheManifest,
    SymbolCacheMetadata,
    feature_parts_source_fingerprint,
    symbol_cache_dir,
    write_json,
)
from agent_layer.data.single_symbol_reader import validate_agent_cache_dataset
from agent_layer.data.tensor_store import (
    end_index_path,
    feature_time_path,
    sequence_path,
    snapshot_path,
    snapshot_required_path,
    valid_ratio_path,
    valid_rows_path,
)

SYMBOL = "000001.SZ"
FREQUENCY = "5min"
SCHEMA_HASH = "schema-v1"
DECISION_COUNT = 3
FEATURE_ROWS = 4
CHANNELS = 2


def test_agent_cache_validation_accepts_bound_complete_cache(tmp_path: Path) -> None:
    cache_dir, _ = _build_valid_cache(tmp_path)

    result = validate_agent_cache_dataset(cache_dir)

    assert result["ok"] is True
    assert {item["name"] for item in result["checks"]} >= {
        "manifest",
        "source_dataset",
        "symbols",
        "episode_index",
    }


def test_agent_cache_validation_rejects_changed_source_dataset(tmp_path: Path) -> None:
    cache_dir, source_dir = _build_valid_cache(tmp_path)
    source_manifest_path = source_dir / "feature_parts_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["symbols"][SYMBOL]["fingerprint"] = "source-symbol-v2"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = validate_agent_cache_dataset(cache_dir)

    assert result["ok"] is False
    assert "source dataset changed" in result["checks"][-1]["message"]


def test_agent_cache_validation_rejects_partial_build_manifest(tmp_path: Path) -> None:
    cache_dir, _ = _build_valid_cache(tmp_path)
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest["failed_symbols"] = [SYMBOL]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = validate_agent_cache_dataset(cache_dir)

    assert result["ok"] is False
    assert "build is incomplete" in result["checks"][-1]["message"]


def test_agent_cache_validation_rejects_structural_shape_mismatch(tmp_path: Path) -> None:
    cache_dir, _ = _build_valid_cache(tmp_path)
    np.save(
        valid_ratio_path(symbol_cache_dir(cache_dir, SYMBOL), FREQUENCY),
        np.zeros(DECISION_COUNT - 1, dtype=np.float32),
    )

    result = validate_agent_cache_dataset(cache_dir)

    assert result["ok"] is False
    assert "valid_ratio_5min.npy shape" in result["checks"][-1]["message"]


def test_agent_cache_validation_requires_full_episode_index_coverage(tmp_path: Path) -> None:
    cache_dir, _ = _build_valid_cache(tmp_path)
    _write_parquet(
        cache_dir / EPISODE_INDEX_NAME,
        "SELECT '000002.SZ' AS symbol, '000002_SZ' AS safe_symbol, "
        "TIMESTAMP '2024-01-02 09:30:00' AS first_decision_time, "
        "TIMESTAMP '2024-01-02 09:40:00' AS last_decision_time, "
        "3::BIGINT AS decision_count, 0::BIGINT AS start_offset, 2::BIGINT AS end_offset",
    )

    result = validate_agent_cache_dataset(cache_dir)

    assert result["ok"] is False
    assert "episode index is missing selected symbols" in result["checks"][-1]["message"]


def _build_valid_cache(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "feature_dataset"
    source_dir.mkdir()
    source_manifest = {
        "version": 1,
        "build_signature": {
            "frequencies": [FREQUENCY],
            "model_input_schema_hash": SCHEMA_HASH,
        },
        "symbols_order": [SYMBOL],
        "symbols": {
            SYMBOL: {
                "fingerprint": "source-symbol-v1",
                "summary": {
                    "decisions": DECISION_COUNT,
                    "model_input_schema_hash": SCHEMA_HASH,
                },
            }
        },
    }
    (source_dir / "feature_parts_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cache_dir = tmp_path / "agent_cache"
    symbol_dir = symbol_cache_dir(cache_dir, SYMBOL)
    symbol_dir.mkdir(parents=True)
    metadata = SymbolCacheMetadata(
        symbol=SYMBOL,
        safe_symbol="000001_SZ",
        decision_count=DECISION_COUNT,
        first_decision_time="2024-01-02T09:30:00",
        last_decision_time="2024-01-02T09:40:00",
        frequencies=(FREQUENCY,),
        feature_names={FREQUENCY: ("feature_a", "feature_b")},
        sequence_shapes={FREQUENCY: (2, CHANNELS)},
        decision_context_names=("context",),
        runtime_contract=REQUIRED_RUNTIME_CONTRACT_FIELDS,
        schema_hash=SCHEMA_HASH,
    )
    write_json(symbol_dir / "metadata.json", metadata.to_json_dict())

    _write_parquet(
        symbol_dir / DECISIONS_NAME,
        f"SELECT range AS decision_id FROM range({DECISION_COUNT})",
    )
    np.save(
        symbol_dir / DECISION_CONTEXT_NAME,
        np.zeros((DECISION_COUNT, 1), dtype=np.float32),
    )
    np.save(
        symbol_dir / EXECUTION_NAME,
        np.zeros((DECISION_COUNT, len(EXECUTION_COLUMNS)), dtype=np.float32),
    )
    np.save(
        symbol_dir / CONSTRAINTS_NAME,
        np.zeros((DECISION_COUNT, len(CONSTRAINT_COLUMNS)), dtype=np.uint8),
    )
    np.save(
        sequence_path(symbol_dir, FREQUENCY),
        np.zeros((FEATURE_ROWS, CHANNELS), dtype=np.float32),
    )
    np.save(
        feature_time_path(symbol_dir, FREQUENCY),
        np.arange(FEATURE_ROWS).astype("datetime64[us]"),
    )
    np.save(
        end_index_path(symbol_dir, FREQUENCY),
        np.arange(DECISION_COUNT, dtype=np.int64),
    )
    np.save(
        valid_rows_path(symbol_dir, FREQUENCY),
        np.ones(DECISION_COUNT, dtype=np.int16),
    )
    np.save(
        snapshot_required_path(symbol_dir, FREQUENCY),
        np.zeros(DECISION_COUNT, dtype=np.uint8),
    )
    np.save(
        snapshot_path(symbol_dir, FREQUENCY),
        np.zeros((DECISION_COUNT, CHANNELS), dtype=np.float32),
    )
    np.save(
        valid_ratio_path(symbol_dir, FREQUENCY),
        np.ones(DECISION_COUNT, dtype=np.float32),
    )

    _write_parquet(
        cache_dir / EPISODE_INDEX_NAME,
        f"SELECT '{SYMBOL}' AS symbol, '000001_SZ' AS safe_symbol, "
        "TIMESTAMP '2024-01-02 09:30:00' AS first_decision_time, "
        "TIMESTAMP '2024-01-02 09:40:00' AS last_decision_time, "
        f"{DECISION_COUNT}::BIGINT AS decision_count, 0::BIGINT AS start_offset, "
        f"{DECISION_COUNT - 1}::BIGINT AS end_offset",
    )
    source_fingerprint = feature_parts_source_fingerprint(
        source_dir,
        symbols=(SYMBOL,),
        frequencies=(FREQUENCY,),
    )
    manifest = AgentCacheManifest(
        cache_version=AGENT_CACHE_VERSION,
        source_feature_dir=str(source_dir.resolve()),
        schema_hash=SCHEMA_HASH,
        frequencies=(FREQUENCY,),
        symbols=(SYMBOL,),
        symbol_count=1,
        decision_count=DECISION_COUNT,
        created_at="2024-01-02T00:00:00+00:00",
        storage=CACHE_STORAGE_INDEX_BASED,
        requested_symbols=(SYMBOL,),
        failed_symbols=(),
        source_dataset_fingerprint=source_fingerprint,
        source_dataset_fingerprint_version=SOURCE_DATASET_FINGERPRINT_VERSION,
        complete=True,
    )
    write_json(cache_dir / CACHE_MANIFEST_NAME, manifest.to_json_dict())
    return cache_dir, source_dir


def _write_parquet(path: Path, select_sql: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = str(path).replace("'", "''")
    with duckdb.connect(":memory:") as conn:
        conn.execute(f"COPY ({select_sql}) TO '{escaped}' (FORMAT PARQUET)")
