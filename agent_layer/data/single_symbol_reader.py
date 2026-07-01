from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator, Mapping
import json

import numpy as np
import pandas as pd

from agent_layer.data.cache_schema import (
    CACHE_MANIFEST_NAME,
    CACHE_STORAGE_INDEX_BASED,
    CONSTRAINTS_NAME,
    DECISION_CONTEXT_NAME,
    DECISIONS_NAME,
    EXECUTION_NAME,
    SYMBOLS_DIR_NAME,
    AgentCacheManifest,
    SymbolCacheMetadata,
    normalize_symbol_list,
    symbol_cache_dir,
)
from agent_layer.data.single_symbol_episode import (
    IndexBasedMaskView,
    IndexBasedWindowView,
    SingleSymbolEpisodeBuffer,
)
from agent_layer.data.tensor_store import (
    end_index_path,
    read_array,
    read_symbol_metadata,
    sequence_path,
    snapshot_path,
    snapshot_required_path,
    valid_ratio_path,
    valid_rows_path,
)
from agent_layer.data.timeline import AgentMarketStep, STAGE_ORDER
from agent_layer.data.timeline_types import TimelineKey


class AgentCacheError(ValueError):
    pass


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
        mmap: bool = True,
        symbol_cache_size: int = 32,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / CACHE_MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Agent cache manifest not found: {manifest_path}")
        self.manifest = AgentCacheManifest.from_json_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if self.manifest.cache_version <= 0:
            raise AgentCacheError("Invalid Agent cache manifest version.")
        if self.manifest.storage != CACHE_STORAGE_INDEX_BASED:
            raise AgentCacheError(
                "Unsupported Agent cache storage "
                f"{self.manifest.storage!r}. Rebuild Agent Cache with the index-based cache builder."
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
        self.mmap_mode = "r" if mmap else None
        self.symbol_cache_size = max(1, int(symbol_cache_size))
        self._symbols: OrderedDict[str, _CachedSymbol] = OrderedDict()

        first = self._load_symbol(self.universe[0])
        self.schema_hash = first.metadata.schema_hash
        self.feature_names = {
            freq: tuple(first.metadata.feature_names.get(freq, ()))
            for freq in self.frequencies
        }
        self.sequence_shapes = {
            freq: tuple(int(value) for value in first.metadata.sequence_shapes.get(freq, ()))
            for freq in self.frequencies
        }
        self.decision_context_names = first.metadata.decision_context_names
        self.runtime_contract = first.metadata.runtime_contract

    def for_universe(self, universe: Iterable[str]) -> "SingleSymbolReader":
        clone = object.__new__(SingleSymbolReader)
        clone.cache_dir = self.cache_dir
        clone.manifest = self.manifest
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
            frame = self._load_symbol(symbol).decisions
            if "decision_time" not in frame:
                continue
            for value in pd.to_datetime(frame["decision_time"], errors="coerce").dropna():
                values.add(pd.Timestamp(value).normalize())
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
        decisions = pd.read_parquet(decisions_path).reset_index(drop=True)
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
        return loaded


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
        del validate_store, cache_size, use_market_cache, use_decision_cache
        del market_cache_workers, market_cache_progress, decision_cache_progress
        self.cache_dir = Path(cache_dir)
        self.reader = SingleSymbolReader(
            self.cache_dir,
            universe=universe,
            frequencies=frequencies,
            mmap=True,
        )
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


def is_agent_cache_dir(path: str | Path) -> bool:
    root = Path(path)
    return (root / CACHE_MANIFEST_NAME).exists() and (root / SYMBOLS_DIR_NAME).exists()


def validate_agent_cache_dataset(
    cache_dir: str | Path,
    *,
    universe: Iterable[str] | None = None,
    frequencies: Iterable[str] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    try:
        loader = CacheBackedAgentTimelineLoader(
            cache_dir,
            universe=universe,
            frequencies=frequencies,
            validate_store=False,
        )
        checks.append({"name": "manifest", "status": "pass", "message": "Agent cache manifest exists."})
        checks.append({"name": "symbols", "status": "pass", "message": f"{len(loader.universe)} symbols selected."})
        checks.append({"name": "frequencies", "status": "pass", "message": ", ".join(loader.frequencies)})
        dates = loader.trading_dates()
        status = "pass" if dates else "error"
        checks.append({"name": "trading_dates", "status": status, "message": f"{len(dates)} trading dates available."})
        return {
            "ok": bool(dates),
            "status": "pass" if dates else "error",
            "type": "agent_cache",
            "checks": checks,
            "schema_hash": loader.schema_hash,
            "symbols": len(loader.universe),
            "frequencies": list(loader.frequencies),
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
