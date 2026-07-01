from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from feature_layer.datasets import ModelInputBatch

from agent_layer.data.feature_parts import FeaturePartsDataset
from agent_layer.data.timeline_types import TimelineKey


STAGE_ORDER: dict[str, int] = {"open_auction": 0, "bar_close": 1}


@dataclass(frozen=True)
class AgentMarketStep:
    decision_time: pd.Timestamp
    stage: str
    symbols: tuple[str, ...]
    decision_ids: tuple[str | None, ...]
    active_mask: np.ndarray
    execution_prices: np.ndarray
    limit_reference_close: np.ndarray
    limit_pct: np.ndarray
    liquidity_volume: np.ndarray
    liquidity_amount: np.ndarray
    is_st: np.ndarray
    market_can_buy: np.ndarray
    market_can_sell: np.ndarray
    is_tradeable: np.ndarray
    is_limit_up: np.ndarray
    is_limit_down: np.ndarray
    is_zero_volume: np.ndarray
    market_sequences: dict[str, np.ndarray]
    sequence_masks: dict[str, np.ndarray]
    valid_ratios: dict[str, np.ndarray]
    decision_context: np.ndarray
    feature_names: dict[str, tuple[str, ...]]
    decision_context_names: tuple[str, ...]
    runtime_contract: tuple[str, ...]
    schema_hash: str

    @property
    def active_symbols(self) -> tuple[str, ...]:
        return tuple(
            symbol for symbol, active in zip(self.symbols, self.active_mask) if bool(active)
        )


class AgentTimelineLoader:
    """Expose feature_parts as Agent market steps.

    The canonical training path can use a full universe for metadata/preflight,
    then derive cheap single-symbol views with :meth:`for_universe` for each
    episode.  This keeps the multi-frequency model input contract intact while
    avoiding market-wide action batches when the Agent is trained as a
    single-stock operator.
    """

    def __init__(
        self,
        store_path: str | Path,
        *,
        universe: Iterable[str] | None = None,
        frequencies: Iterable[str] | None = None,
        validate_store: bool = True,
        cache_size: int = 8192,
        stream_chunk_size: int = 64,
        use_market_cache: bool = True,
        use_decision_cache: bool = True,
        market_cache_workers: int | None = None,
        market_cache_progress=None,
        decision_cache_progress=None,
    ) -> None:
        self.store_path = Path(store_path)
        if self.store_path.suffix.lower() == ".duckdb":
            raise ValueError(
                "AgentTimelineLoader no longer accepts feature_store.duckdb. "
                "Use the Feature Dataset directory or feature_parts directory."
            )
        self._parts_source = FeaturePartsDataset.maybe(self.store_path)
        if self._parts_source is None:
            raise FileNotFoundError(
                "Feature parts dataset not found. Expected a directory containing "
                "feature_parts/ and feature_parts_manifest.json."
            )
        if validate_store:
            validation = self._parts_source.validate(sample_limit=0)
            if not validation["ok"]:
                failures = [
                    check["message"]
                    for check in validation["checks"]
                    if check["status"] == "error"
                ]
                raise ValueError("Invalid Feature Parts Dataset: " + "; ".join(failures))
        compiled = self._parts_source.compiled_model_input()
        stored_symbols = list(self._parts_source.symbols)
        indexed_frequencies = set(self._parts_source.frequencies())

        requested_universe = _normalize_unique(universe) if universe is not None else stored_symbols
        if not requested_universe:
            raise ValueError("Agent universe cannot be empty.")
        unknown_symbols = sorted(set(requested_universe).difference(stored_symbols))
        if unknown_symbols:
            raise ValueError(
                "Agent universe contains symbols absent from the Feature Parts Dataset: "
                + ", ".join(unknown_symbols[:20])
            )

        compiled_order = [
            str(freq)
            for freq in compiled.get("channels_by_frequency", {})
            if str(freq) in indexed_frequencies
        ]
        selected_frequencies = (
            _normalize_unique(frequencies) if frequencies is not None else compiled_order
        )
        unavailable_frequencies = sorted(set(selected_frequencies).difference(indexed_frequencies))
        if unavailable_frequencies:
            raise ValueError(
                "Requested frequencies are absent from the Feature Parts Dataset: "
                + ", ".join(unavailable_frequencies)
            )

        self.universe = tuple(requested_universe)
        self.frequencies = tuple(selected_frequencies)
        self.compiled_model_input = compiled
        self.schema_hash = str(compiled.get("schema_hash") or "")
        self.feature_names = {
            freq: tuple(compiled.get("channels_by_frequency", {}).get(freq, []))
            for freq in self.frequencies
        }
        self.sequence_shapes = {
            freq: tuple(int(value) for value in compiled.get("shapes", {}).get(freq, []))
            for freq in self.frequencies
        }
        self.decision_context_names = tuple(compiled.get("decision_context", []))
        self.runtime_contract = tuple(compiled.get("runtime_state", []))
        self.cache_size = max(0, int(cache_size))
        self.stream_chunk_size = max(1, int(stream_chunk_size))
        self._batch_cache: OrderedDict[str, ModelInputBatch] = OrderedDict()
        self._step_rows_cache: dict[tuple[pd.Timestamp, str], pd.DataFrame] = {}
        # Compact DuckDB caches are intentionally not used on the canonical
        # feature_parts input path. DuckDB is still used internally only as an
        # in-memory parquet query engine inside FeaturePartsDataset.
        self._performance = {
            "chunks": 0,
            "steps": 0,
            "load_seconds": 0.0,
        }

    def for_universe(self, universe: Iterable[str]) -> "AgentTimelineLoader":
        """Return a lightweight loader view restricted to ``universe``.

        The view shares immutable Feature Parts metadata with the parent loader
        but owns its own small step/batch caches.  It avoids re-validating the
        dataset and, more importantly, lets training create one-symbol episodes
        without querying or materializing every symbol in the run universe.
        """
        requested = tuple(_normalize_unique(universe))
        if not requested:
            raise ValueError("Agent universe view cannot be empty.")
        unknown = sorted(set(requested).difference(self._parts_source.symbols))
        if unknown:
            raise ValueError(
                "Agent universe view contains symbols absent from the Feature Parts Dataset: "
                + ", ".join(unknown[:20])
            )
        clone = object.__new__(AgentTimelineLoader)
        clone.store_path = self.store_path
        clone._parts_source = self._parts_source
        clone.universe = requested
        clone.frequencies = self.frequencies
        clone.compiled_model_input = self.compiled_model_input
        clone.schema_hash = self.schema_hash
        clone.feature_names = self.feature_names
        clone.sequence_shapes = self.sequence_shapes
        clone.decision_context_names = self.decision_context_names
        clone.runtime_contract = self.runtime_contract
        clone.cache_size = self.cache_size
        clone.stream_chunk_size = self.stream_chunk_size
        clone._batch_cache = OrderedDict()
        clone._step_rows_cache = {}
        clone._performance = {
            "chunks": 0,
            "steps": 0,
            "load_seconds": 0.0,
        }
        return clone

    def timeline(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
    ) -> list[TimelineKey]:
        return self._parts_source.timeline(
            universe=self.universe,
            start=start,
            end=end,
            stages=stages,
        )

    def trading_dates(self) -> list[pd.Timestamp]:
        return self._parts_source.trading_dates(universe=self.universe)

    def iter_steps(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
    ) -> Iterator[AgentMarketStep]:
        for key in self.timeline(start=start, end=end, stages=stages):
            yield self.load_step(key.decision_time, key.stage)

    def stream_steps(
        self,
        keys: Iterable[TimelineKey],
    ) -> "AgentTimelineStream":
        """Stream ordered market steps while prefetching one bounded chunk ahead."""
        return AgentTimelineStream(
            self,
            list(keys),
            chunk_size=self.stream_chunk_size,
        )

    def load_steps(self, keys: Iterable[TimelineKey]) -> list[AgentMarketStep]:
        selected = list(keys)
        if not selected:
            return []
        key_pairs = {(pd.Timestamp(key.decision_time), str(key.stage)) for key in selected}
        start = min(pair[0] for pair in key_pairs)
        end = max(pair[0] for pair in key_pairs)
        rows = self._query_decision_rows(start=start, end=end)
        rows = rows.loc[
            [
                (pd.Timestamp(decision_time), str(stage)) in key_pairs
                for decision_time, stage in zip(rows["decision_time"], rows["stage"])
            ]
        ].copy()
        for (decision_time, stage), frame in rows.groupby(
            ["decision_time", "stage"], sort=False
        ):
            self._step_rows_cache[(pd.Timestamp(decision_time), str(stage))] = (
                frame.sort_values("symbol").reset_index(drop=True)
            )
        decision_ids = [
            str(decision_id) for decision_id in rows["decision_id"]
        ]
        self._prefill_batches(decision_ids)
        return [self.load_step(key.decision_time, key.stage) for key in selected]

    def performance_payload(self) -> dict[str, float | int]:
        steps = int(self._performance["steps"])
        seconds = float(self._performance["load_seconds"])
        return {
            **self._performance,
            "seconds_per_step": seconds / max(1, steps),
        }

    def _prefill_batches(self, decision_ids: Iterable[str]) -> None:
        requested = list(dict.fromkeys(str(value) for value in decision_ids if value))
        missing = [value for value in requested if value not in self._batch_cache]
        if not missing:
            return
        loaded = self._parts_source.load_model_input_batches(
            decision_ids=missing,
            frequencies=self.frequencies,
        )
        for decision_id in missing:
            self._remember_batch(decision_id, loaded[decision_id])

    def load_step(self, decision_time: str | pd.Timestamp, stage: str) -> AgentMarketStep:
        timestamp = pd.Timestamp(decision_time)
        normalized_stage = str(stage)
        rows = self._step_rows_cache.pop((timestamp, normalized_stage), None)
        if rows is None:
            rows = self._query_decision_rows(
                start=timestamp,
                end=timestamp,
                stage=normalized_stage,
            )
        if rows.empty:
            raise KeyError(f"No Agent decisions at {timestamp} / {normalized_stage}.")

        symbol_to_row = {
            str(row.symbol): row for row in rows.itertuples(index=False)
        }
        decision_ids_for_step = rows["decision_id"].astype(str).tolist()
        batches = {
            decision_id: self._batch_cache[decision_id]
            for decision_id in decision_ids_for_step
            if decision_id in self._batch_cache
        }
        for decision_id in batches:
            self._batch_cache.move_to_end(decision_id)
        missing_ids = [
            decision_id for decision_id in decision_ids_for_step
            if decision_id not in batches
        ]
        if missing_ids:
            batches.update(self._parts_source.load_model_input_batches(
                decision_ids=missing_ids,
                frequencies=self.frequencies,
            ))
        for decision_id in missing_ids:
            self._remember_batch(decision_id, batches[decision_id])
        size = len(self.universe)
        decision_ids: list[str | None] = [None] * size
        active_mask = np.zeros(size, dtype=np.bool_)
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
        decision_context = np.zeros(
            (size, len(self.decision_context_names)), dtype=np.float32
        )
        market_sequences = {
            freq: np.zeros((size, *self.sequence_shapes[freq]), dtype=np.float32)
            for freq in self.frequencies
        }
        sequence_masks = {
            freq: np.zeros((size, self.sequence_shapes[freq][0]), dtype=np.float32)
            for freq in self.frequencies
        }
        valid_ratios = {
            freq: np.zeros(size, dtype=np.float32) for freq in self.frequencies
        }

        for index, symbol in enumerate(self.universe):
            row = symbol_to_row.get(symbol)
            if row is None:
                continue
            decision_id = str(row.decision_id)
            batch = batches[decision_id]
            if batch.schema_hash != self.schema_hash:
                raise ValueError(
                    f"Model schema changed while loading {decision_id}: "
                    f"{batch.schema_hash} != {self.schema_hash}"
                )
            decision_ids[index] = decision_id
            active_mask[index] = True
            execution_prices[index] = float(row.execution_price)
            limit_reference_close[index] = float(row.limit_reference_close)
            limit_pct[index] = float(row.limit_pct)
            liquidity_volume[index] = float(row.liquidity_volume or 0.0)
            liquidity_amount[index] = float(row.liquidity_amount or 0.0)
            is_st[index] = bool(row.is_st)
            market_can_buy[index] = bool(row.market_can_buy)
            market_can_sell[index] = bool(row.market_can_sell)
            is_tradeable[index] = bool(row.is_tradeable)
            is_limit_up[index] = bool(row.is_limit_up)
            is_limit_down[index] = bool(row.is_limit_down)
            is_zero_volume[index] = bool(row.is_zero_volume)
            row_values = row._asdict()
            if all(name in row_values for name in self.decision_context_names):
                decision_context[index] = np.asarray(
                    [row_values.get(name, 0.0) for name in self.decision_context_names],
                    dtype=np.float32,
                )
            else:
                decision_context[index] = batch.decision_context
            for freq in self.frequencies:
                market_sequences[freq][index] = batch.market_sequences[freq]
                sequence_masks[freq][index] = batch.sequence_masks[freq]
                valid_ratios[freq][index] = batch.valid_ratios[freq]

        return AgentMarketStep(
            decision_time=timestamp,
            stage=normalized_stage,
            symbols=self.universe,
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
            feature_names=self.feature_names,
            decision_context_names=self.decision_context_names,
            runtime_contract=self.runtime_contract,
            schema_hash=self.schema_hash,
        )

    def _query_decision_rows(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        stage: str | None = None,
    ) -> pd.DataFrame:
        return self._parts_source.query_decision_rows(
            universe=self.universe,
            start=start,
            end=end,
            stage=stage,
        )

    def _remember_batch(self, decision_id: str, batch: ModelInputBatch) -> None:
        if self.cache_size <= 0:
            return
        self._batch_cache[decision_id] = batch
        self._batch_cache.move_to_end(decision_id)
        while len(self._batch_cache) > self.cache_size:
            self._batch_cache.popitem(last=False)

    def _decision_filters(
        self,
        *,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
        stages: Iterable[str] | None,
    ) -> tuple[str, list[object]]:
        clauses = ["symbol IN (SELECT UNNEST(?))"]
        params: list[object] = [list(self.universe)]
        if start is not None:
            clauses.append("decision_time >= ?")
            params.append(pd.Timestamp(start))
        if end is not None:
            end_timestamp = pd.Timestamp(end)
            if end_timestamp == end_timestamp.normalize():
                end_timestamp += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            clauses.append("decision_time <= ?")
            params.append(end_timestamp)
        selected_stages = _normalize_unique(stages) if stages is not None else []
        if selected_stages:
            unknown = sorted(set(selected_stages).difference(STAGE_ORDER))
            if unknown:
                raise ValueError("Unsupported decision stages: " + ", ".join(unknown))
            clauses.append("stage IN (SELECT UNNEST(?))")
            params.append(selected_stages)
        return " AND ".join(clauses), params


def _normalize_unique(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        normalized = str(value).strip().upper() if "." in str(value) else str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


class AgentTimelineStream:
    """Bounded producer/consumer stream with one asynchronously prefetched chunk."""

    def __init__(
        self,
        loader: AgentTimelineLoader,
        keys: list[TimelineKey],
        *,
        chunk_size: int,
    ) -> None:
        self.loader = loader
        self.keys = keys
        self.chunk_size = max(1, int(chunk_size))
        self._next_index = 0
        self._buffer: list[AgentMarketStep] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-data")
        self._future: Future[list[AgentMarketStep]] | None = None
        self._closed = False
        self._schedule()

    def __iter__(self) -> "AgentTimelineStream":
        return self

    def __next__(self) -> AgentMarketStep:
        if self._closed:
            raise StopIteration
        if not self._buffer:
            if self._future is None:
                raise StopIteration
            self._buffer = self._future.result()
            self._future = None
            self._schedule()
        if not self._buffer:
            raise StopIteration
        return self._buffer.pop(0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._future = None
        self._buffer.clear()

    def peek_buffer(self) -> tuple[AgentMarketStep, ...]:
        return tuple(self._buffer)

    def _schedule(self) -> None:
        if self._closed or self._next_index >= len(self.keys):
            return
        chunk = self.keys[self._next_index : self._next_index + self.chunk_size]
        self._next_index += len(chunk)
        self._future = self._executor.submit(
            self._load_chunk,
            chunk,
        )

    def _load_chunk(self, keys: list[TimelineKey]) -> list[AgentMarketStep]:
        started = perf_counter()
        result = self.loader.load_steps(keys)
        self.loader._performance["chunks"] += 1
        self.loader._performance["steps"] += len(result)
        self.loader._performance["load_seconds"] += perf_counter() - started
        return result
