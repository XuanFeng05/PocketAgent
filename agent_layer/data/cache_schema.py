from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

AGENT_CACHE_VERSION = 3
SOURCE_DATASET_FINGERPRINT_VERSION = 1
CACHE_STORAGE_INDEX_BASED = "index_based_features"
CACHE_MANIFEST_NAME = "cache_manifest.json"
FEATURE_PARTS_MANIFEST_NAME = "feature_parts_manifest.json"
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

REQUIRED_RUNTIME_CONTRACT_FIELDS: tuple[str, ...] = (
    "cash_ratio",
    "position_ratio",
    "available_position_ratio",
    "unrealized_pnl_ratio",
    "holding_bars_norm",
    "one_lot_nav_ratio",
    "buy_one_lot_required_ratio",
    "sell_one_lot_required_ratio",
    "max_buy_value_ratio",
    "max_sell_value_ratio",
    "can_buy",
    "can_sell",
    "can_buy_one_lot",
    "can_sell_one_lot",
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
    requested_symbols: tuple[str, ...] = ()
    failed_symbols: tuple[str, ...] = ()
    source_dataset_fingerprint: str = ""
    source_dataset_fingerprint_version: int = 0
    complete: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frequencies"] = list(self.frequencies)
        payload["symbols"] = list(self.symbols)
        payload["requested_symbols"] = list(self.requested_symbols)
        payload["failed_symbols"] = list(self.failed_symbols)
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
            requested_symbols=tuple(
                str(value) for value in payload.get("requested_symbols", [])
            ),
            failed_symbols=tuple(
                str(value) for value in payload.get("failed_symbols", [])
            ),
            source_dataset_fingerprint=str(
                payload.get("source_dataset_fingerprint") or ""
            ),
            source_dataset_fingerprint_version=int(
                payload.get("source_dataset_fingerprint_version") or 0
            ),
            complete=bool(payload.get("complete", False)),
        )


def feature_parts_source_fingerprint(
    feature_dir: str | Path,
    *,
    symbols: Iterable[str],
    frequencies: Iterable[str],
) -> str:
    """Return a stable identity for the Feature Parts subset used by Agent Cache.

    The Feature Parts manifest already owns per-symbol content fingerprints and
    the full build signature.  Hashing those canonical records binds an Agent
    Cache to the exact selected source subset without re-reading every parquet
    byte during training preflight.
    """

    dataset_dir = Path(feature_dir).resolve()
    manifest_path = dataset_dir / FEATURE_PARTS_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Feature Parts manifest required for Agent Cache binding: {manifest_path}"
        )
    manifest = read_json(manifest_path)
    symbol_entries = manifest.get("symbols")
    if not isinstance(symbol_entries, dict):
        raise ValueError(
            f"Feature Parts manifest has no symbol fingerprint map: {manifest_path}"
        )

    selected_symbols = normalize_symbol_list(symbols)
    if not selected_symbols:
        raise ValueError("Agent Cache source fingerprint requires at least one symbol.")
    selected_frequencies = tuple(str(value).strip() for value in frequencies if str(value).strip())
    if not selected_frequencies:
        raise ValueError("Agent Cache source fingerprint requires at least one frequency.")

    selected_entries: list[dict[str, Any]] = []
    for symbol in selected_symbols:
        raw_entry = symbol_entries.get(symbol)
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"Feature Parts manifest is missing selected symbol {symbol}: {manifest_path}"
            )
        fingerprint = str(raw_entry.get("fingerprint") or "").strip()
        if not fingerprint:
            raise ValueError(
                f"Feature Parts manifest symbol {symbol} has no content fingerprint. "
                "Rebuild Feature Dataset before Agent Cache."
            )
        summary = raw_entry.get("summary") if isinstance(raw_entry.get("summary"), dict) else {}
        selected_entries.append(
            {
                "symbol": symbol,
                "fingerprint": fingerprint,
                "decisions": int(summary.get("decisions") or 0),
                "model_input_schema_hash": str(
                    summary.get("model_input_schema_hash") or ""
                ),
            }
        )

    payload = {
        "fingerprint_version": SOURCE_DATASET_FINGERPRINT_VERSION,
        "feature_parts_manifest_version": manifest.get("version"),
        "build_signature": manifest.get("build_signature"),
        "frequencies": list(selected_frequencies),
        "symbols": selected_entries,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
