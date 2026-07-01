from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json


AGENT_CACHE_VERSION = 2
CACHE_STORAGE_INDEX_BASED = "index_based_features"
CACHE_MANIFEST_NAME = "cache_manifest.json"
EPISODE_INDEX_NAME = "episode_index.parquet"
SYMBOLS_DIR_NAME = "symbols"
SYMBOL_METADATA_NAME = "metadata.json"
DECISIONS_NAME = "decisions.parquet"
DECISION_CONTEXT_NAME = "decision_context.npy"
EXECUTION_NAME = "execution.npy"
CONSTRAINTS_NAME = "constraints.npy"

# Execution values are kept as float32 for fast tensor conversion.
EXECUTION_COLUMNS: tuple[str, ...] = (
    "execution_price",
    "limit_reference_close",
    "limit_pct",
    "liquidity_volume",
    "liquidity_amount",
)

# These are booleans at training time.  The cache stores them as uint8.
CONSTRAINT_COLUMNS: tuple[str, ...] = (
    "is_st",
    "market_can_buy",
    "market_can_sell",
    "is_tradeable",
    "is_limit_up",
    "is_limit_down",
    "is_zero_volume",
)


@dataclass(frozen=True)
class SymbolCacheMetadata:
    symbol: str
    safe_symbol: str
    decision_count: int
    first_decision_time: str | None
    last_decision_time: str | None
    frequencies: tuple[str, ...]
    feature_names: dict[str, tuple[str, ...]]
    sequence_shapes: dict[str, tuple[int, ...]]
    decision_context_names: tuple[str, ...]
    runtime_contract: tuple[str, ...]
    schema_hash: str
    storage: str = CACHE_STORAGE_INDEX_BASED
    execution_columns: tuple[str, ...] = EXECUTION_COLUMNS
    constraint_columns: tuple[str, ...] = CONSTRAINT_COLUMNS

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frequencies"] = list(self.frequencies)
        payload["feature_names"] = {
            key: list(value) for key, value in self.feature_names.items()
        }
        payload["sequence_shapes"] = {
            key: list(value) for key, value in self.sequence_shapes.items()
        }
        payload["decision_context_names"] = list(self.decision_context_names)
        payload["runtime_contract"] = list(self.runtime_contract)
        payload["execution_columns"] = list(self.execution_columns)
        payload["constraint_columns"] = list(self.constraint_columns)
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "SymbolCacheMetadata":
        return cls(
            symbol=str(payload["symbol"]),
            safe_symbol=str(payload["safe_symbol"]),
            decision_count=int(payload.get("decision_count") or 0),
            first_decision_time=(
                str(payload["first_decision_time"])
                if payload.get("first_decision_time") is not None
                else None
            ),
            last_decision_time=(
                str(payload["last_decision_time"])
                if payload.get("last_decision_time") is not None
                else None
            ),
            frequencies=tuple(str(value) for value in payload.get("frequencies", [])),
            feature_names={
                str(key): tuple(str(item) for item in value)
                for key, value in dict(payload.get("feature_names", {})).items()
            },
            sequence_shapes={
                str(key): tuple(int(item) for item in value)
                for key, value in dict(payload.get("sequence_shapes", {})).items()
            },
            decision_context_names=tuple(
                str(value) for value in payload.get("decision_context_names", [])
            ),
            runtime_contract=tuple(
                str(value) for value in payload.get("runtime_contract", [])
            ),
            schema_hash=str(payload.get("schema_hash") or ""),
            storage=str(payload.get("storage") or "precomputed_windows"),
            execution_columns=tuple(
                str(value) for value in payload.get("execution_columns", EXECUTION_COLUMNS)
            ),
            constraint_columns=tuple(
                str(value) for value in payload.get("constraint_columns", CONSTRAINT_COLUMNS)
            ),
        )


@dataclass(frozen=True)
class AgentCacheManifest:
    cache_version: int
    source_feature_dir: str
    schema_hash: str
    frequencies: tuple[str, ...]
    symbols: tuple[str, ...]
    symbol_count: int
    decision_count: int
    created_at: str
    storage: str = CACHE_STORAGE_INDEX_BASED

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frequencies"] = list(self.frequencies)
        payload["symbols"] = list(self.symbols)
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "AgentCacheManifest":
        return cls(
            cache_version=int(payload.get("cache_version") or 0),
            source_feature_dir=str(payload.get("source_feature_dir") or ""),
            schema_hash=str(payload.get("schema_hash") or ""),
            frequencies=tuple(str(value) for value in payload.get("frequencies", [])),
            symbols=tuple(str(value) for value in payload.get("symbols", [])),
            symbol_count=int(payload.get("symbol_count") or 0),
            decision_count=int(payload.get("decision_count") or 0),
            created_at=str(payload.get("created_at") or ""),
            storage=str(payload.get("storage") or "precomputed_windows"),
        )


def safe_symbol_name(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "_").replace("/", "_").replace("\\", "_")


def safe_frequency_name(freq: str) -> str:
    return str(freq).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def symbol_cache_dir(cache_dir: str | Path, symbol: str) -> Path:
    return Path(cache_dir) / SYMBOLS_DIR_NAME / safe_symbol_name(symbol)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_symbol_list(values: Iterable[str] | None) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or []:
        normalized = str(value).strip()
        if not normalized or normalized.startswith("#"):
            continue
        normalized = normalized.upper() if "." in normalized else normalized
        if normalized not in result:
            result.append(normalized)
    return tuple(result)
