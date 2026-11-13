from __future__ import annotations

import gc
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator

import duckdb
import numpy as np
import pandas as pd

from agent_layer.data.cache_schema import (
    AGENT_CACHE_VERSION,
    CACHE_MANIFEST_NAME,
    CACHE_STORAGE_INDEX_BASED,
    CONSTRAINT_COLUMNS,
    CONSTRAINTS_NAME,
    DECISION_CONTEXT_NAME,
    DECISIONS_NAME,
    EXECUTION_COLUMNS,
    EXECUTION_NAME,
    REQUIRED_RUNTIME_CONTRACT_FIELDS,
    SOURCE_DATASET_FINGERPRINT_VERSION,
    SYMBOL_METADATA_NAME,
    SYMBOLS_DIR_NAME,
    AgentCacheManifest,
    SymbolCacheMetadata,
    feature_parts_source_fingerprint,
    normalize_symbol_list,
    symbol_cache_dir,
)
from agent_layer.data.episode_index import read_episode_index
from agent_layer.data.single_symbol_episode import (
    IndexBasedMaskView,
    IndexBasedWindowView,
    SingleSymbolEpisodeBuffer,
)
from agent_layer.data.tensor_store import (
    end_index_path,
    feature_time_path,
    read_array,
    read_symbol_metadata,
    sequence_path,
    snapshot_path,
    snapshot_required_path,
    valid_ratio_path,
    valid_rows_path,
)
from agent_layer.data.timeline import STAGE_ORDER, AgentMarketStep
from agent_layer.data.timeline_types import TimelineKey


class AgentCacheError(ValueError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _validate_cache_manifest_contract(
    cache_dir: Path,
    manifest: AgentCacheManifest,
) -> Path:
    if manifest.cache_version != AGENT_CACHE_VERSION:
        raise AgentCacheError(
            f"Agent cache version {manifest.cache_version} is unsupported; "
            f"version {AGENT_CACHE_VERSION} is required for source-dataset binding. "
            "Rebuild Agent Cache from the selected Feature Dataset."
        )
    if manifest.storage != CACHE_STORAGE_INDEX_BASED:
        raise AgentCacheError(
            "Unsupported Agent cache storage "
            f"{manifest.storage!r}. Rebuild Agent Cache with the index-based cache builder."
        )

    requested = normalize_symbol_list(manifest.requested_symbols)
    available = normalize_symbol_list(manifest.symbols)
    if not manifest.complete:
        failed = normalize_symbol_list(manifest.failed_symbols)
        suffix = f" Failed symbols: {', '.join(failed[:20])}." if failed else ""
        raise AgentCacheError(
            "Agent cache build is incomplete and cannot be used for training or evaluation."
            + suffix
            + " Rebuild Agent Cache."
        )
    if not requested or requested != available:
        missing = [symbol for symbol in requested if symbol not in available]
        detail = f" Missing: {', '.join(missing[:20])}." if missing else ""
        raise AgentCacheError(
            "Agent cache does not contain the complete requested symbol set."
            + detail
            + " Rebuild Agent Cache."
        )
    if manifest.failed_symbols:
        raise AgentCacheError("A complete Agent cache manifest cannot contain failed symbols.")
    if manifest.symbol_count != len(available):
        raise AgentCacheError(
            "Agent cache manifest symbol_count does not match its symbol list. Rebuild Agent Cache."
        )
    if manifest.decision_count <= 0:
        raise AgentCacheError("Agent cache manifest contains no decisions. Rebuild Agent Cache.")
    if (
        manifest.source_dataset_fingerprint_version
        != SOURCE_DATASET_FINGERPRINT_VERSION
    ):
        raise AgentCacheError(
            "Agent cache source fingerprint version is missing or unsupported. Rebuild Agent Cache."
        )
    if not manifest.source_dataset_fingerprint:
        raise AgentCacheError(
            "Agent cache is not bound to a Feature Parts source dataset. Rebuild Agent Cache."
        )

    source_dir = _resolve_manifest_source_dir(cache_dir, manifest.source_feature_dir)
    actual_fingerprint = feature_parts_source_fingerprint(
        source_dir,
        symbols=requested,
        frequencies=manifest.frequencies,
    )
    if actual_fingerprint != manifest.source_dataset_fingerprint:
        raise AgentCacheError(
            "Agent cache source dataset changed after the cache was built. "
            "Rebuild Agent Cache from the currently selected Feature Dataset."
        )
    return source_dir


def _resolve_manifest_source_dir(cache_dir: Path, raw_value: str) -> Path:
    normalized = str(raw_value or "").strip()
    if not normalized:
        raise AgentCacheError("Agent cache manifest has no source_feature_dir.")
    raw = Path(normalized)
    candidates = (
        (raw,)
        if raw.is_absolute()
        else (PROJECT_ROOT / raw, cache_dir.parent / raw, Path.cwd() / raw)
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise AgentCacheError(
        "Agent cache source Feature Parts dataset is unavailable: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


@dataclass
class _CachedSymbol:
    symbol_dir: Path
    metadata: SymbolCacheMetadata
    decisions: pd.DataFrame
    decision_context: np.ndarray
    execution: np.ndarray
    constraints: np.ndarray
    feature_matrices: dict[str, np.ndarray]
    end_indices: dict[str, np.ndarray]
    valid_rows: dict[str, np.ndarray]
    snapshot_required: dict[str, np.ndarray]
    snapshot_features: dict[str, np.ndarray]
    valid_ratios: dict[str, np.ndarray]
    index: dict[tuple[pd.Timestamp, str], int]


class SingleSymbolReader:
    """Fast reader for the Agent single-symbol tensor cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        universe: Iterable[str] | None = None,
        frequencies: Iterable[str] | None = None,
        mmap: bool = False,
        symbol_cache_size: int = 8,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / CACHE_MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Agent cache manifest not found: {manifest_path}")
        self.manifest = AgentCacheManifest.from_json_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        self.source_feature_dir = _validate_cache_manifest_contract(
            self.cache_dir,
            self.manifest,
        )
        available_symbols = tuple(str(symbol).upper() for symbol in self.manifest.symbols)
        selected_symbols = (
            normalize_symbol_list(universe)
            if universe is not None
            else available_symbols
        )
        unknown = sorted(set(selected_symbols).difference(available_symbols))
        if unknown:
            raise AgentCacheError(
                "Agent cache is missing requested symbols: " + ", ".join(unknown[:20])
            )
        if not selected_symbols:
            raise AgentCacheError("Agent cache universe cannot be empty.")
        available_frequencies = tuple(str(freq) for freq in self.manifest.frequencies)
        selected_frequencies = (
            tuple(str(freq) for freq in frequencies)
            if frequencies is not None
            else available_frequencies
        )
        missing = sorted(set(selected_frequencies).difference(available_frequencies))
        if missing:
            raise AgentCacheError(
                "Agent cache is missing requested frequencies: " + ", ".join(missing)
            )
        self.universe = tuple(selected_symbols)
        self.frequencies = tuple(selected_frequencies)
        self.mmap_mode = "r" if _truthy(mmap) else None
        self.symbol_cache_size = max(1, int(symbol_cache_size))
        self._symbols: OrderedDict[str, _CachedSymbol] = OrderedDict()

        first_metadata = read_symbol_metadata(symbol_cache_dir(self.cache_dir, self.universe[0]))
        if first_metadata.schema_hash != self.manifest.schema_hash:
            raise AgentCacheError(
                f"Agent cache schema mismatch for {self.universe[0]}: "
                f"{first_metadata.schema_hash} != {self.manifest.schema_hash}"
            )
        if first_metadata.storage != CACHE_STORAGE_INDEX_BASED:
            raise AgentCacheError(
                f"Unsupported Agent cache storage for {self.universe[0]}: "
                f"{first_metadata.storage!r}. Rebuild Agent Cache."
            )
        missing_first_freqs = sorted(set(self.frequencies).difference(first_metadata.frequencies))
        if missing_first_freqs:
            raise AgentCacheError(
                f"Agent cache symbol {self.universe[0]} is missing frequencies: "
                + ", ".join(missing_first_freqs)
            )
        self.schema_hash = first_metadata.schema_hash
        self.feature_names = {
            freq: tuple(first_metadata.feature_names.get(freq, ()))
            for freq in self.frequencies
        }
        self.sequence_shapes = {
            freq: tuple(int(value) for value in first_metadata.sequence_shapes.get(freq, ()))
            for freq in self.frequencies
        }
        self.decision_context_names = first_metadata.decision_context_names
        self.runtime_contract = first_metadata.runtime_contract

    def for_universe(self, universe: Iterable[str]) -> "SingleSymbolReader":
        clone = object.__new__(SingleSymbolReader)
        clone.cache_dir = self.cache_dir
        clone.manifest = self.manifest
        clone.source_feature_dir = self.source_feature_dir
        selected = normalize_symbol_list(universe)
        if not selected:
            raise AgentCacheError("Agent cache universe view cannot be empty.")
        unknown = sorted(set(selected).difference(self.manifest.symbols))
        if unknown:
            raise AgentCacheError(
                "Agent cache universe view contains missing symbols: "
                + ", ".join(unknown[:20])
            )
        clone.universe = tuple(selected)
        clone.frequencies = self.frequencies
        clone.mmap_mode = self.mmap_mode
        clone.symbol_cache_size = self.symbol_cache_size
        clone._symbols = self._symbols
        clone.schema_hash = self.schema_hash
        clone.feature_names = self.feature_names
        clone.sequence_shapes = self.sequence_shapes
        clone.decision_context_names = self.decision_context_names
        clone.runtime_contract = self.runtime_contract
        return clone

    def trading_dates(self, *, universe: Iterable[str] | None = None) -> list[pd.Timestamp]:
        symbols = normalize_symbol_list(universe) if universe is not None else self.universe
        values: set[pd.Timestamp] = set()
        for symbol in symbols:
            decisions_path = symbol_cache_dir(self.cache_dir, symbol) / DECISIONS_NAME
            if not decisions_path.exists():
                continue
            values.update(_read_decision_dates(decisions_path))
        return sorted(values)

    def timeline(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
    ) -> list[TimelineKey]:
        selected_stages = set(str(stage) for stage in stages) if stages is not None else None
        counts: dict[tuple[pd.Timestamp, str], int] = {}
        for symbol in self.universe:
            frame = self._filtered_decisions(symbol, start=start, end=end, stages=selected_stages)
            for row in frame.itertuples(index=False):
                key = (pd.Timestamp(row.decision_time), str(row.stage))
                counts[key] = counts.get(key, 0) + 1
        return [
            TimelineKey(decision_time=decision_time, stage=stage, active_symbols=count)
            for (decision_time, stage), count in sorted(
                counts.items(), key=lambda item: (item[0][0], STAGE_ORDER.get(item[0][1], 99), item[0][1])
            )
        ]

    def episode_buffer(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
        max_steps: int | None = None,
    ) -> SingleSymbolEpisodeBuffer:
        normalized = normalize_symbol_list((symbol,))[0]
        cached = self._load_symbol(normalized)
        stage_set = set(str(stage) for stage in stages) if stages is not None else None
        frame = self._filtered_decisions(normalized, start=start, end=end, stages=stage_set)
        if max_steps is not None and int(max_steps) > 0:
            frame = frame.head(int(max_steps)).copy()
        if frame.empty:
            raise ValueError(f"Agent cache episode is empty for {normalized}.")
        indices = np.asarray(frame["_cache_index"].to_numpy(), dtype=np.int64)
        return SingleSymbolEpisodeBuffer(
            symbol=normalized,
            decision_times=tuple(pd.Timestamp(value) for value in frame["decision_time"]),
            stages=tuple(str(value) for value in frame["stage"]),
            decision_ids=tuple(
                str(value) if value is not None and not pd.isna(value) else None
                for value in frame.get("decision_id", pd.Series([None] * len(frame)))
            ),
            execution=np.asarray(cached.execution[indices], dtype=np.float32),
            constraints=np.asarray(cached.constraints[indices], dtype=np.uint8),
            decision_context=np.asarray(cached.decision_context[indices], dtype=np.float32),
            market_sequences={
                freq: IndexBasedWindowView(
                    features=cached.feature_matrices[freq],
                    end_indices=cached.end_indices[freq],
                    snapshot_required=cached.snapshot_required[freq],
                    snapshot_features=cached.snapshot_features[freq],
                    episode_indices=indices,
                    window=int(self.sequence_shapes[freq][0]),
                    channels=int(self.sequence_shapes[freq][1]),
                )
                for freq in self.frequencies
            },
            sequence_masks={
                freq: IndexBasedMaskView(
                    valid_rows=cached.valid_rows[freq],
                    episode_indices=indices,
                    window=int(self.sequence_shapes[freq][0]),
                )
                for freq in self.frequencies
            },
            valid_ratios={
                freq: np.asarray(cached.valid_ratios[freq][indices], dtype=np.float32)
                for freq in self.frequencies
            },
            feature_names=self.feature_names,
            decision_context_names=self.decision_context_names,
            runtime_contract=self.runtime_contract,
            schema_hash=self.schema_hash,
            execution_columns=cached.metadata.execution_columns,
            constraint_columns=cached.metadata.constraint_columns,
        )

    def market_step(self, symbol: str, decision_time: pd.Timestamp, stage: str) -> AgentMarketStep:
        return self.episode_buffer(symbol, start=decision_time, end=decision_time, stages=(stage,)).market_step(0)

    def _filtered_decisions(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
        stages: set[str] | None,
    ) -> pd.DataFrame:
        cached = self._load_symbol(symbol)
        frame = cached.decisions
        mask = np.ones(len(frame), dtype=bool)
        if start is not None:
            mask &= frame["decision_time"].ge(pd.Timestamp(start)).to_numpy()
        if end is not None:
            end_timestamp = pd.Timestamp(end)
            if end_timestamp == end_timestamp.normalize():
                end_timestamp += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            mask &= frame["decision_time"].le(end_timestamp).to_numpy()
        if stages:
            mask &= frame["stage"].astype(str).isin(stages).to_numpy()
        result = frame.loc[mask].copy()
        return result.sort_values(["decision_time", "_stage_order", "stage"]).reset_index(drop=True)

    def _load_symbol(self, symbol: str) -> _CachedSymbol:
        normalized = normalize_symbol_list((symbol,))[0]
        cached = self._symbols.get(normalized)
        if cached is not None:
            self._symbols.move_to_end(normalized)
            return cached
        symbol_dir = symbol_cache_dir(self.cache_dir, normalized)
        metadata = read_symbol_metadata(symbol_dir)
        if metadata.schema_hash != self.manifest.schema_hash:
            raise AgentCacheError(
                f"Agent cache schema mismatch for {normalized}: "
                f"{metadata.schema_hash} != {self.manifest.schema_hash}"
            )
        missing_freqs = sorted(set(self.frequencies).difference(metadata.frequencies))
        if missing_freqs:
            raise AgentCacheError(
                f"Agent cache symbol {normalized} is missing frequencies: "
                + ", ".join(missing_freqs)
            )
        decisions_path = symbol_dir / DECISIONS_NAME
        if not decisions_path.exists():
            raise FileNotFoundError(f"Agent cache decisions not found: {decisions_path}")
        decisions = _read_parquet_frame(decisions_path).reset_index(drop=True)
        if decisions.empty:
            raise AgentCacheError(f"Agent cache symbol {normalized} has no decisions.")
        decisions["decision_time"] = pd.to_datetime(decisions["decision_time"], errors="coerce")
        decisions["stage"] = decisions["stage"].astype(str)
        decisions["_cache_index"] = np.arange(len(decisions), dtype=np.int64)
        decisions["_stage_order"] = decisions["stage"].map(STAGE_ORDER).fillna(99).astype(int)
        decisions = decisions.sort_values(["decision_time", "_stage_order", "stage"]).reset_index(drop=True)
        # Preserve the original row id after sorting because arrays are stored in
        # the same order as decisions were written by the cache builder.
        index = {
            (pd.Timestamp(row["decision_time"]), str(row["stage"])): int(row["_cache_index"])
            for _, row in decisions.iterrows()
        }
        if metadata.storage != CACHE_STORAGE_INDEX_BASED:
            raise AgentCacheError(
                f"Unsupported Agent cache storage for {normalized}: {metadata.storage!r}. "
                "Rebuild Agent Cache."
            )
        loaded = _CachedSymbol(
            symbol_dir=symbol_dir,
            metadata=metadata,
            decisions=decisions,
            decision_context=read_array(symbol_dir / DECISION_CONTEXT_NAME, mmap_mode=self.mmap_mode),
            execution=read_array(symbol_dir / EXECUTION_NAME, mmap_mode=self.mmap_mode),
            constraints=read_array(symbol_dir / CONSTRAINTS_NAME, mmap_mode=self.mmap_mode),
            feature_matrices={
                freq: read_array(sequence_path(symbol_dir, freq), mmap_mode=self.mmap_mode)
                for freq in self.frequencies
            },
            end_indices={
                freq: read_array(end_index_path(symbol_dir, freq), mmap_mode=self.mmap_mode)
                for freq in self.frequencies
            },
            valid_rows={
                freq: read_array(valid_rows_path(symbol_dir, freq), mmap_mode=self.mmap_mode)
                for freq in self.frequencies
            },
            snapshot_required={
                freq: read_array(snapshot_required_path(symbol_dir, freq), mmap_mode=self.mmap_mode)
                for freq in self.frequencies
            },
            snapshot_features={
                freq: read_array(snapshot_path(symbol_dir, freq), mmap_mode=self.mmap_mode)
                for freq in self.frequencies
            },
            valid_ratios={
                freq: read_array(valid_ratio_path(symbol_dir, freq), mmap_mode=self.mmap_mode)
                for freq in self.frequencies
            },
            index=index,
        )
        self._symbols[normalized] = loaded
        self._symbols.move_to_end(normalized)
        while len(self._symbols) > self.symbol_cache_size:
            self._symbols.popitem(last=False)
            # Dropping the last Python reference is enough for normal ndarray
            # loads and avoids manually closing mmap objects that may still be
            # referenced by active episode buffers.  The training default is
            # mmap=False, so this primarily bounds memory, not file handles.
            gc.collect()
        return loaded

    def close(self) -> None:
        """Release cached symbol arrays owned by this reader view.

        Reader views share the same underlying LRU cache.  Closing is therefore
        intended for the root training loader at worker shutdown, not for every
        individual environment reset.
        """
        self._symbols.clear()
        gc.collect()


class CacheBackedAgentTimelineLoader:
    """AgentTimelineLoader-compatible adapter backed by Agent tensor cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        universe: Iterable[str] | None = None,
        frequencies: Iterable[str] | None = None,
        validate_store: bool = True,
        cache_size: int = 8192,
        stream_chunk_size: int = 256,
        use_market_cache: bool = True,
        use_decision_cache: bool = True,
        market_cache_workers: int | None = None,
        market_cache_progress=None,
        decision_cache_progress=None,
    ) -> None:
        del use_market_cache, use_decision_cache
        del market_cache_workers, market_cache_progress, decision_cache_progress
        self.cache_dir = Path(cache_dir)
        if validate_store:
            validation = validate_agent_cache_dataset(
                self.cache_dir,
                universe=universe,
                frequencies=frequencies,
            )
            if not validation.get("ok"):
                messages = [
                    str(item.get("message") or "Agent cache validation failed.")
                    for item in validation.get("checks", [])
                    if item.get("status") == "error"
                ]
                raise AgentCacheError("; ".join(messages) or "Agent cache validation failed.")
        mmap_enabled = _env_bool("POCKETAGENT_AGENT_CACHE_MMAP", default=False)
        symbol_cache_size = _env_int("POCKETAGENT_SYMBOL_CACHE_SIZE", default=8)
        # The legacy cache_size argument was a decision-batch cache size.  When
        # callers pass it through unchanged it is often 8192, which would be a
        # dangerous number of open memmaps.  Treat it only as an upper bound and
        # keep a conservative default for training stability.
        if cache_size is not None:
            try:
                symbol_cache_size = min(symbol_cache_size, max(1, int(cache_size)))
            except (TypeError, ValueError):
                pass
        self.reader = SingleSymbolReader(
            self.cache_dir,
            universe=universe,
            frequencies=frequencies,
            mmap=mmap_enabled,
            symbol_cache_size=symbol_cache_size,
        )
        self.mmap_enabled = bool(mmap_enabled)
        self.symbol_cache_size = int(symbol_cache_size)
        self.store_path = self.cache_dir
        self.universe = self.reader.universe
        self.frequencies = self.reader.frequencies
        self.schema_hash = self.reader.schema_hash
        self.feature_names = self.reader.feature_names
        self.sequence_shapes = self.reader.sequence_shapes
        self.decision_context_names = self.reader.decision_context_names
        self.runtime_contract = self.reader.runtime_contract
        self.stream_chunk_size = max(1, int(stream_chunk_size))
        self._performance = {"chunks": 0, "steps": 0, "load_seconds": 0.0}

    def for_universe(self, universe: Iterable[str]) -> "CacheBackedAgentTimelineLoader":
        clone = object.__new__(CacheBackedAgentTimelineLoader)
        clone.cache_dir = self.cache_dir
        clone.reader = self.reader.for_universe(universe)
        clone.store_path = self.store_path
        clone.universe = clone.reader.universe
        clone.frequencies = clone.reader.frequencies
        clone.schema_hash = clone.reader.schema_hash
        clone.feature_names = clone.reader.feature_names
        clone.sequence_shapes = clone.reader.sequence_shapes
        clone.decision_context_names = clone.reader.decision_context_names
        clone.runtime_contract = clone.reader.runtime_contract
        clone.stream_chunk_size = self.stream_chunk_size
        clone.mmap_enabled = self.mmap_enabled
        clone.symbol_cache_size = self.symbol_cache_size
        clone._performance = {"chunks": 0, "steps": 0, "load_seconds": 0.0}
        return clone

    def trading_dates(self) -> list[pd.Timestamp]:
        return self.reader.trading_dates(universe=self.universe)

    def timeline(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
    ) -> list[TimelineKey]:
        return self.reader.timeline(start=start, end=end, stages=stages)

    def iter_steps(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
    ) -> Iterator[AgentMarketStep]:
        for key in self.timeline(start=start, end=end, stages=stages):
            yield self.load_step(key.decision_time, key.stage)

    def stream_steps(self, keys: Iterable[TimelineKey]) -> "CacheBackedTimelineStream":
        return CacheBackedTimelineStream(self, list(keys), chunk_size=self.stream_chunk_size)

    def load_steps(self, keys: Iterable[TimelineKey]) -> list[AgentMarketStep]:
        selected = list(keys)
        if not selected:
            return []
        return [self.load_step(key.decision_time, key.stage) for key in selected]

    def load_step(self, decision_time: str | pd.Timestamp, stage: str) -> AgentMarketStep:
        timestamp = pd.Timestamp(decision_time)
        markets: list[AgentMarketStep | None] = []
        for symbol in self.universe:
            try:
                markets.append(self.reader.market_step(symbol, timestamp, str(stage)))
            except ValueError:
                markets.append(None)
        active_markets = [market for market in markets if market is not None]
        if not active_markets:
            raise KeyError(f"No cached Agent decisions at {timestamp} / {stage}.")
        return _merge_single_symbol_markets(
            timestamp,
            str(stage),
            self.universe,
            active_markets,
            self,
        )

    def close(self) -> None:
        self.reader.close()

    def performance_payload(self) -> dict[str, float | int]:
        steps = int(self._performance["steps"])
        seconds = float(self._performance["load_seconds"])
        return {
            **self._performance,
            "seconds_per_step": seconds / max(1, steps),
        }


class CacheBackedTimelineStream:
    def __init__(
        self,
        loader: CacheBackedAgentTimelineLoader,
        keys: list[TimelineKey],
        *,
        chunk_size: int,
    ) -> None:
        self.loader = loader
        self.keys = keys
        self.chunk_size = max(1, int(chunk_size))
        self._next_index = 0
        self._buffer: list[AgentMarketStep] = []
        self._closed = False

    def __iter__(self) -> "CacheBackedTimelineStream":
        return self

    def __next__(self) -> AgentMarketStep:
        if self._closed:
            raise StopIteration
        if not self._buffer:
            self._load_next_chunk()
        if not self._buffer:
            raise StopIteration
        return self._buffer.pop(0)

    def close(self) -> None:
        self._closed = True
        self._buffer.clear()

    def peek_buffer(self) -> tuple[AgentMarketStep, ...]:
        return tuple(self._buffer)

    def _load_next_chunk(self) -> None:
        if self._next_index >= len(self.keys):
            return
        chunk = self.keys[self._next_index : self._next_index + self.chunk_size]
        self._next_index += len(chunk)
        started = perf_counter()
        self._buffer = self.loader.load_steps(chunk)
        self.loader._performance["chunks"] += 1
        self.loader._performance["steps"] += len(self._buffer)
        self.loader._performance["load_seconds"] += perf_counter() - started


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def is_agent_cache_dir(path: str | Path) -> bool:
    root = Path(path)
    return (root / CACHE_MANIFEST_NAME).exists() and (root / SYMBOLS_DIR_NAME).exists()


def _read_parquet_frame(path: str | Path) -> pd.DataFrame:
    with duckdb.connect(":memory:") as conn:
        return conn.execute("SELECT * FROM read_parquet(?)", [str(Path(path))]).fetchdf()


def _read_decision_dates(path: str | Path) -> set[pd.Timestamp]:
    with duckdb.connect(":memory:") as conn:
        rows = conn.execute(
            "SELECT DISTINCT CAST(decision_time AS DATE) AS decision_date "
            "FROM read_parquet(?) WHERE decision_time IS NOT NULL",
            [str(Path(path))],
        ).fetchall()
    return {
        pd.Timestamp(value).normalize()
        for (value,) in rows
        if value is not None
    }


def _npy_shape(path: Path) -> tuple[int, ...]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return tuple(int(value) for value in values.shape)
    finally:
        mmap = getattr(values, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _parquet_row_count(path: Path) -> int:
    with duckdb.connect(":memory:") as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
    return int(row[0] if row else 0)


def _validate_symbol_cache_structure(
    root: Path,
    symbol: str,
    frequencies: tuple[str, ...],
    manifest: AgentCacheManifest,
) -> tuple[list[str], SymbolCacheMetadata | None]:
    errors: list[str] = []
    symbol_dir = symbol_cache_dir(root, symbol)
    metadata_path = symbol_dir / SYMBOL_METADATA_NAME
    if not metadata_path.exists():
        return [f"{symbol}: missing {SYMBOL_METADATA_NAME}"], None
    try:
        metadata = read_symbol_metadata(symbol_dir)
    except Exception as exc:
        return [f"{symbol}: invalid {SYMBOL_METADATA_NAME}: {exc}"], None

    if metadata.symbol.upper() != symbol.upper():
        errors.append(f"{symbol}: metadata symbol is {metadata.symbol}")
    if metadata.schema_hash != manifest.schema_hash:
        errors.append(f"{symbol}: schema hash differs from cache manifest")
    if metadata.storage != CACHE_STORAGE_INDEX_BASED:
        errors.append(f"{symbol}: unsupported storage {metadata.storage!r}")
    if metadata.decision_count <= 0:
        errors.append(f"{symbol}: decision_count must be positive")
    if tuple(metadata.execution_columns) != tuple(EXECUTION_COLUMNS):
        errors.append(f"{symbol}: execution column contract is outdated")
    if tuple(metadata.constraint_columns) != tuple(CONSTRAINT_COLUMNS):
        errors.append(f"{symbol}: constraint column contract is outdated")

    missing_symbol_freqs = sorted(set(frequencies).difference(metadata.frequencies))
    if missing_symbol_freqs:
        errors.append(f"{symbol}: missing frequencies {', '.join(missing_symbol_freqs)}")

    decision_count = int(metadata.decision_count)
    decisions_path = symbol_dir / DECISIONS_NAME
    if not decisions_path.exists():
        errors.append(f"{symbol}: missing {DECISIONS_NAME}")
    else:
        try:
            rows = _parquet_row_count(decisions_path)
            if rows != decision_count:
                errors.append(
                    f"{symbol}: {DECISIONS_NAME} has {rows} rows, expected {decision_count}"
                )
        except Exception as exc:
            errors.append(f"{symbol}: unreadable {DECISIONS_NAME}: {exc}")

    fixed_shapes = {
        symbol_dir / DECISION_CONTEXT_NAME: (
            decision_count,
            len(metadata.decision_context_names),
        ),
        symbol_dir / EXECUTION_NAME: (decision_count, len(EXECUTION_COLUMNS)),
        symbol_dir / CONSTRAINTS_NAME: (decision_count, len(CONSTRAINT_COLUMNS)),
    }
    for path, expected_shape in fixed_shapes.items():
        if not path.exists():
            errors.append(f"{symbol}: missing {path.name}")
            continue
        try:
            actual_shape = _npy_shape(path)
            if actual_shape != expected_shape:
                errors.append(
                    f"{symbol}: {path.name} shape {actual_shape}, expected {expected_shape}"
                )
        except Exception as exc:
            errors.append(f"{symbol}: unreadable {path.name}: {exc}")

    for freq in frequencies:
        names = tuple(metadata.feature_names.get(freq, ()))
        sequence_shape = tuple(metadata.sequence_shapes.get(freq, ()))
        channels = int(sequence_shape[1]) if len(sequence_shape) == 2 else len(names)
        if len(sequence_shape) != 2 or channels <= 0 or len(names) != channels:
            errors.append(f"{symbol}: invalid sequence contract for {freq}")
            continue
        feature_path = sequence_path(symbol_dir, freq)
        feature_rows = 0
        if not feature_path.exists():
            errors.append(f"{symbol}: missing {feature_path.name}")
        else:
            try:
                feature_shape = _npy_shape(feature_path)
                if (
                    len(feature_shape) != 2
                    or feature_shape[0] <= 0
                    or feature_shape[1] != channels
                ):
                    errors.append(
                        f"{symbol}: {feature_path.name} shape {feature_shape}, "
                        f"expected [positive rows, {channels}]"
                    )
                else:
                    feature_rows = feature_shape[0]
            except Exception as exc:
                errors.append(f"{symbol}: unreadable {feature_path.name}: {exc}")

        per_frequency_shapes = {
            feature_time_path(symbol_dir, freq): (feature_rows,),
            end_index_path(symbol_dir, freq): (decision_count,),
            valid_rows_path(symbol_dir, freq): (decision_count,),
            snapshot_required_path(symbol_dir, freq): (decision_count,),
            snapshot_path(symbol_dir, freq): (decision_count, channels),
            valid_ratio_path(symbol_dir, freq): (decision_count,),
        }
        for path, expected_shape in per_frequency_shapes.items():
            if not path.exists():
                errors.append(f"{symbol}: missing {path.name}")
                continue
            try:
                actual_shape = _npy_shape(path)
                if actual_shape != expected_shape:
                    errors.append(
                        f"{symbol}: {path.name} shape {actual_shape}, expected {expected_shape}"
                    )
            except Exception as exc:
                errors.append(f"{symbol}: unreadable {path.name}: {exc}")
    return errors, metadata


def validate_agent_cache_dataset(
    cache_dir: str | Path,
    *,
    universe: Iterable[str] | None = None,
    frequencies: Iterable[str] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    try:
        root = Path(cache_dir)
        manifest_path = root / CACHE_MANIFEST_NAME
        manifest = AgentCacheManifest.from_json_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        source_dir = _validate_cache_manifest_contract(root, manifest)
        selected_symbols = (
            normalize_symbol_list(universe)
            if universe is not None
            else tuple(str(symbol).upper() for symbol in manifest.symbols)
        )
        selected_frequencies = (
            tuple(str(freq) for freq in frequencies)
            if frequencies is not None
            else tuple(str(freq) for freq in manifest.frequencies)
        )
        missing_symbols = sorted(set(selected_symbols).difference(str(symbol).upper() for symbol in manifest.symbols))
        if missing_symbols:
            raise AgentCacheError(
                "Agent cache is missing requested symbols: "
                + ", ".join(missing_symbols[:20])
                + ". Build Agent Cache from the selected Feature Dataset and symbols."
            )
        missing_frequencies = sorted(set(selected_frequencies).difference(str(freq) for freq in manifest.frequencies))
        if missing_frequencies:
            raise AgentCacheError(
                "Agent cache is missing requested frequencies: "
                + ", ".join(missing_frequencies)
                + ". Rebuild Agent Cache after rebuilding Feature Dataset with all required frequencies."
            )
        structural_errors: list[str] = []
        runtime_contract: tuple[str, ...] | None = None
        reference_metadata: SymbolCacheMetadata | None = None
        decision_counts: dict[str, int] = {}
        for symbol in selected_symbols:
            symbol_errors, metadata = _validate_symbol_cache_structure(
                root,
                symbol,
                tuple(selected_frequencies),
                manifest,
            )
            structural_errors.extend(symbol_errors)
            if metadata is None:
                continue
            decision_counts[symbol] = int(metadata.decision_count)
            runtime_contract = runtime_contract or metadata.runtime_contract
            if reference_metadata is None:
                reference_metadata = metadata
            else:
                if metadata.decision_context_names != reference_metadata.decision_context_names:
                    structural_errors.append(
                        f"{symbol}: decision context contract differs from other cache symbols"
                    )
                if metadata.runtime_contract != reference_metadata.runtime_contract:
                    structural_errors.append(
                        f"{symbol}: runtime contract differs from other cache symbols"
                    )
                for freq in selected_frequencies:
                    if metadata.feature_names.get(freq) != reference_metadata.feature_names.get(freq):
                        structural_errors.append(
                            f"{symbol}: feature names for {freq} differ from other cache symbols"
                        )
                    if metadata.sequence_shapes.get(freq) != reference_metadata.sequence_shapes.get(freq):
                        structural_errors.append(
                            f"{symbol}: sequence shape for {freq} differs from other cache symbols"
                        )
        if structural_errors:
            raise AgentCacheError(
                "Agent cache is incomplete. "
                + "; ".join(structural_errors[:10])
                + ". Rebuild Agent Cache from the Agent Cache panel."
            )
        if set(selected_symbols) == set(normalize_symbol_list(manifest.symbols)):
            actual_decision_count = sum(decision_counts.values())
            if actual_decision_count != manifest.decision_count:
                raise AgentCacheError(
                    "Agent cache manifest decision_count does not match symbol metadata: "
                    f"{manifest.decision_count} != {actual_decision_count}. Rebuild Agent Cache."
                )
        if runtime_contract is not None:
            missing_runtime = sorted(set(REQUIRED_RUNTIME_CONTRACT_FIELDS).difference(runtime_contract))
            if missing_runtime:
                raise AgentCacheError(
                    "Agent cache runtime contract is outdated; missing "
                    + ", ".join(missing_runtime)
                    + ". Rebuild Feature Dataset, then rebuild Agent Cache."
                )
        try:
            episode_index = read_episode_index(root)
        except Exception:
            episode_index = pd.DataFrame()
        indexed_symbols = set(
            str(value).upper()
            for value in episode_index.get("symbol", pd.Series(dtype=str)).tolist()
        )
        missing_index_symbols = sorted(set(selected_symbols).difference(indexed_symbols))
        if missing_index_symbols:
            raise AgentCacheError(
                "Agent cache episode index is missing selected symbols: "
                + ", ".join(missing_index_symbols[:20])
                + ". Rebuild Agent Cache."
            )
        indexed_selected = sorted(set(selected_symbols).intersection(indexed_symbols))
        selected_rows = episode_index.loc[
            episode_index["symbol"].astype(str).str.upper().isin(indexed_selected)
        ]
        duplicate_index_symbols = sorted(
            symbol
            for symbol, count in selected_rows["symbol"].astype(str).str.upper().value_counts().items()
            if int(count) != 1
        )
        if duplicate_index_symbols:
            raise AgentCacheError(
                "Agent cache episode index must contain exactly one row per selected symbol: "
                + ", ".join(duplicate_index_symbols[:20])
                + ". Rebuild Agent Cache."
            )
        if "decision_count" in selected_rows:
            indexed_counts = {
                str(row.symbol).upper(): int(row.decision_count)
                for row in selected_rows.itertuples(index=False)
            }
            count_mismatches = [
                symbol
                for symbol, count in decision_counts.items()
                if indexed_counts.get(symbol) != count
            ]
            if count_mismatches:
                raise AgentCacheError(
                    "Agent cache episode index decision counts differ from symbol metadata: "
                    + ", ".join(count_mismatches[:20])
                    + ". Rebuild Agent Cache."
                )
        dates: set[pd.Timestamp] = set()
        if not episode_index.empty and indexed_selected:
            for column in ("first_decision_time", "last_decision_time"):
                if column not in selected_rows:
                    continue
                for value in pd.to_datetime(selected_rows[column], errors="coerce").dropna():
                    dates.add(pd.Timestamp(value).normalize())
        schema_hash = manifest.schema_hash
        checks.append(
            {
                "name": "manifest",
                "status": "pass",
                "message": f"Agent cache v{manifest.cache_version} manifest is complete.",
            }
        )
        checks.append(
            {
                "name": "source_dataset",
                "status": "pass",
                "message": f"Agent cache is bound to {source_dir}.",
            }
        )
        checks.append(
            {
                "name": "symbols",
                "status": "pass",
                "message": f"All {len(selected_symbols)} selected symbols are structurally complete.",
            }
        )
        checks.append({"name": "frequencies", "status": "pass", "message": ", ".join(selected_frequencies)})
        checks.append(
            {
                "name": "episode_index",
                "status": "pass",
                "message": f"All {len(indexed_selected)} selected symbols have one matching episode index row.",
            }
        )
        return {
            "ok": True,
            "status": "pass",
            "type": "agent_cache",
            "checks": checks,
            "schema_hash": schema_hash,
            "source_dataset_fingerprint": manifest.source_dataset_fingerprint,
            "symbols": len(selected_symbols),
            "frequencies": list(selected_frequencies),
            "trading_date_bounds": {
                "start": str(min(dates).date()) if dates else None,
                "end": str(max(dates).date()) if dates else None,
            },
        }
    except Exception as exc:
        checks.append({"name": "agent_cache", "status": "error", "message": str(exc)})
        return {"ok": False, "status": "error", "type": "agent_cache", "checks": checks}


def _merge_single_symbol_markets(
    decision_time: pd.Timestamp,
    stage: str,
    universe: tuple[str, ...],
    active_markets: list[AgentMarketStep],
    loader: CacheBackedAgentTimelineLoader,
) -> AgentMarketStep:
    active_by_symbol = {market.symbols[0]: market for market in active_markets}
    size = len(universe)
    active_mask = np.zeros(size, dtype=np.bool_)
    decision_ids: list[str | None] = [None] * size
    execution_prices = np.full(size, np.nan, dtype=np.float64)
    limit_reference_close = np.full(size, np.nan, dtype=np.float64)
    limit_pct = np.full(size, np.nan, dtype=np.float32)
    liquidity_volume = np.zeros(size, dtype=np.float64)
    liquidity_amount = np.zeros(size, dtype=np.float64)
    is_st = np.zeros(size, dtype=np.bool_)
    market_can_buy = np.zeros(size, dtype=np.bool_)
    market_can_sell = np.zeros(size, dtype=np.bool_)
    is_tradeable = np.zeros(size, dtype=np.bool_)
    is_limit_up = np.zeros(size, dtype=np.bool_)
    is_limit_down = np.zeros(size, dtype=np.bool_)
    is_zero_volume = np.ones(size, dtype=np.bool_)
    decision_context = np.zeros((size, len(loader.decision_context_names)), dtype=np.float32)
    market_sequences = {
        freq: np.zeros((size, *loader.sequence_shapes[freq]), dtype=np.float32)
        for freq in loader.frequencies
    }
    sequence_masks = {
        freq: np.zeros((size, loader.sequence_shapes[freq][0]), dtype=np.float32)
        for freq in loader.frequencies
    }
    valid_ratios = {freq: np.zeros(size, dtype=np.float32) for freq in loader.frequencies}
    for index, symbol in enumerate(universe):
        market = active_by_symbol.get(symbol)
        if market is None:
            continue
        active_mask[index] = True
        decision_ids[index] = market.decision_ids[0]
        execution_prices[index] = market.execution_prices[0]
        limit_reference_close[index] = market.limit_reference_close[0]
        limit_pct[index] = market.limit_pct[0]
        liquidity_volume[index] = market.liquidity_volume[0]
        liquidity_amount[index] = market.liquidity_amount[0]
        is_st[index] = market.is_st[0]
        market_can_buy[index] = market.market_can_buy[0]
        market_can_sell[index] = market.market_can_sell[0]
        is_tradeable[index] = market.is_tradeable[0]
        is_limit_up[index] = market.is_limit_up[0]
        is_limit_down[index] = market.is_limit_down[0]
        is_zero_volume[index] = market.is_zero_volume[0]
        decision_context[index] = market.decision_context[0]
        for freq in loader.frequencies:
            market_sequences[freq][index] = market.market_sequences[freq][0]
            sequence_masks[freq][index] = market.sequence_masks[freq][0]
            valid_ratios[freq][index] = market.valid_ratios[freq][0]
    return AgentMarketStep(
        decision_time=decision_time,
        stage=stage,
        symbols=universe,
        decision_ids=tuple(decision_ids),
        active_mask=active_mask,
        execution_prices=execution_prices,
        limit_reference_close=limit_reference_close,
        limit_pct=limit_pct,
        liquidity_volume=liquidity_volume,
        liquidity_amount=liquidity_amount,
        is_st=is_st,
        market_can_buy=market_can_buy,
        market_can_sell=market_can_sell,
        is_tradeable=is_tradeable,
        is_limit_up=is_limit_up,
        is_limit_down=is_limit_down,
        is_zero_volume=is_zero_volume,
        market_sequences=market_sequences,
        sequence_masks=sequence_masks,
        valid_ratios=valid_ratios,
        decision_context=decision_context,
        feature_names=loader.feature_names,
        decision_context_names=loader.decision_context_names,
        runtime_contract=loader.runtime_contract,
        schema_hash=loader.schema_hash,
    )
