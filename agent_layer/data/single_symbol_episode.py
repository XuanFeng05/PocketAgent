from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from agent_layer.data.timeline import AgentMarketStep
from agent_layer.data.timeline_types import TimelineKey


class IndexBasedWindowView:
    """Lazy [T, window, channels] view backed by continuous features + indices."""

    def __init__(
        self,
        *,
        features: np.ndarray,
        end_indices: np.ndarray,
        snapshot_required: np.ndarray,
        snapshot_features: np.ndarray,
        episode_indices: np.ndarray,
        window: int,
        channels: int,
    ) -> None:
        self.features = features
        self.end_indices = end_indices
        self.snapshot_required = snapshot_required.astype(bool, copy=False)
        self.snapshot_features = snapshot_features
        self.episode_indices = episode_indices.astype(np.int64, copy=False)
        self.window = int(window)
        self.channels = int(channels)
        self.shape = (len(self.episode_indices), self.window, self.channels)
        self.dtype = np.dtype("float32")

    def __len__(self) -> int:
        return len(self.episode_indices)

    def __getitem__(self, key):
        positions = self._positions(key)
        values = np.zeros((len(positions), self.window, self.channels), dtype=np.float32)
        for output_index, episode_position in enumerate(positions):
            values[output_index] = self._window_for_episode_position(int(episode_position))
        if isinstance(key, (int, np.integer)):
            return values[0]
        return values

    def __array__(self, dtype=None):
        array = self[:]
        return array.astype(dtype, copy=False) if dtype is not None else array

    def _positions(self, key) -> np.ndarray:
        if isinstance(key, slice):
            return np.arange(len(self), dtype=np.int64)[key]
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            return np.asarray([index], dtype=np.int64)
        return np.asarray(key, dtype=np.int64)

    def _window_for_episode_position(self, episode_position: int) -> np.ndarray:
        decision_index = int(self.episode_indices[episode_position])
        end_index = int(self.end_indices[decision_index]) if decision_index < len(self.end_indices) else -1
        needs_snapshot = bool(self.snapshot_required[decision_index]) if decision_index < len(self.snapshot_required) else False
        stable_limit = max(0, self.window - (1 if needs_snapshot else 0))
        pieces: list[np.ndarray] = []
        if end_index >= 0 and stable_limit > 0 and len(self.features):
            start = max(0, end_index - stable_limit + 1)
            pieces.append(np.asarray(self.features[start : end_index + 1], dtype=np.float32))
        if needs_snapshot and decision_index < len(self.snapshot_features):
            pieces.append(np.asarray(self.snapshot_features[decision_index : decision_index + 1], dtype=np.float32))
        if pieces:
            rows = np.concatenate(pieces, axis=0)[-self.window :]
        else:
            rows = np.zeros((0, self.channels), dtype=np.float32)
        values = np.zeros((self.window, self.channels), dtype=np.float32)
        valid = min(self.window, len(rows))
        if valid:
            values[-valid:] = rows[-valid:]
        return values


class IndexBasedMaskView:
    """Lazy [T, window] mask view derived from cached valid row counts."""

    def __init__(self, *, valid_rows: np.ndarray, episode_indices: np.ndarray, window: int) -> None:
        self.valid_rows = valid_rows.astype(np.int32, copy=False)
        self.episode_indices = episode_indices.astype(np.int64, copy=False)
        self.window = int(window)
        self.shape = (len(self.episode_indices), self.window)
        self.dtype = np.dtype("float32")

    def __len__(self) -> int:
        return len(self.episode_indices)

    def __getitem__(self, key):
        positions = self._positions(key)
        values = np.zeros((len(positions), self.window), dtype=np.float32)
        for output_index, episode_position in enumerate(positions):
            decision_index = int(self.episode_indices[int(episode_position)])
            valid = int(self.valid_rows[decision_index]) if decision_index < len(self.valid_rows) else 0
            valid = max(0, min(self.window, valid))
            if valid:
                values[output_index, -valid:] = 1.0
        if isinstance(key, (int, np.integer)):
            return values[0]
        return values

    def __array__(self, dtype=None):
        array = self[:]
        return array.astype(dtype, copy=False) if dtype is not None else array

    def _positions(self, key) -> np.ndarray:
        if isinstance(key, slice):
            return np.arange(len(self), dtype=np.int64)[key]
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            return np.asarray([index], dtype=np.int64)
        return np.asarray(key, dtype=np.int64)


@dataclass(frozen=True)
class SingleSymbolEpisodeBuffer:
    """Training-time view for one stock episode.

    Index-based Agent Cache v2 keeps only continuous feature matrices and
    decision end-index arrays on disk.  This buffer therefore stays lightweight:
    it carries lazy window views and constructs the single current model window
    only when ``market_step(index)`` asks for it.
    """

    symbol: str
    decision_times: tuple[pd.Timestamp, ...]
    stages: tuple[str, ...]
    decision_ids: tuple[str | None, ...]
    execution: np.ndarray
    constraints: np.ndarray
    decision_context: np.ndarray
    market_sequences: Mapping[str, object]
    sequence_masks: Mapping[str, object]
    valid_ratios: Mapping[str, np.ndarray]
    feature_names: Mapping[str, tuple[str, ...]]
    decision_context_names: tuple[str, ...]
    runtime_contract: tuple[str, ...]
    schema_hash: str
    execution_columns: tuple[str, ...]
    constraint_columns: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.decision_times)

    def keys(self) -> list[TimelineKey]:
        return [
            TimelineKey(
                decision_time=decision_time,
                stage=stage,
                active_symbols=1,
            )
            for decision_time, stage in zip(self.decision_times, self.stages)
        ]

    def market_step(self, index: int) -> AgentMarketStep:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        execution = self.execution[index]
        constraints = self.constraints[index]
        execution_lookup = {
            column: offset for offset, column in enumerate(self.execution_columns)
        }
        constraint_lookup = {
            column: offset for offset, column in enumerate(self.constraint_columns)
        }
        return AgentMarketStep(
            decision_time=self.decision_times[index],
            stage=self.stages[index],
            symbols=(self.symbol,),
            decision_ids=(self.decision_ids[index],),
            active_mask=np.asarray([True], dtype=np.bool_),
            execution_prices=np.asarray(
                [_float_column(execution, execution_lookup, "execution_price")],
                dtype=np.float64,
            ),
            limit_reference_close=np.asarray(
                [_float_column(execution, execution_lookup, "limit_reference_close")],
                dtype=np.float64,
            ),
            limit_pct=np.asarray(
                [_float_column(execution, execution_lookup, "limit_pct")],
                dtype=np.float32,
            ),
            liquidity_volume=np.asarray(
                [_float_column(execution, execution_lookup, "liquidity_volume")],
                dtype=np.float64,
            ),
            liquidity_amount=np.asarray(
                [_float_column(execution, execution_lookup, "liquidity_amount")],
                dtype=np.float64,
            ),
            is_st=np.asarray([_bool_column(constraints, constraint_lookup, "is_st")]),
            market_can_buy=np.asarray(
                [_bool_column(constraints, constraint_lookup, "market_can_buy")]
            ),
            market_can_sell=np.asarray(
                [_bool_column(constraints, constraint_lookup, "market_can_sell")]
            ),
            is_tradeable=np.asarray(
                [_bool_column(constraints, constraint_lookup, "is_tradeable")]
            ),
            is_limit_up=np.asarray(
                [_bool_column(constraints, constraint_lookup, "is_limit_up")]
            ),
            is_limit_down=np.asarray(
                [_bool_column(constraints, constraint_lookup, "is_limit_down")]
            ),
            is_zero_volume=np.asarray(
                [_bool_column(constraints, constraint_lookup, "is_zero_volume")]
            ),
            market_sequences={
                freq: np.asarray(values[index : index + 1], dtype=np.float32)
                for freq, values in self.market_sequences.items()
            },
            sequence_masks={
                freq: np.asarray(values[index : index + 1], dtype=np.float32)
                for freq, values in self.sequence_masks.items()
            },
            valid_ratios={
                freq: np.asarray(values[index : index + 1], dtype=np.float32)
                for freq, values in self.valid_ratios.items()
            },
            decision_context=np.asarray(
                self.decision_context[index : index + 1], dtype=np.float32
            ),
            feature_names=dict(self.feature_names),
            decision_context_names=self.decision_context_names,
            runtime_contract=self.runtime_contract,
            schema_hash=self.schema_hash,
        )


def _float_column(values: np.ndarray, lookup: dict[str, int], column: str) -> float:
    index = lookup.get(column)
    if index is None or index >= values.shape[0]:
        return 0.0
    value = float(values[index])
    return value if np.isfinite(value) else 0.0


def _bool_column(values: np.ndarray, lookup: dict[str, int], column: str) -> bool:
    index = lookup.get(column)
    if index is None or index >= values.shape[0]:
        return False
    return bool(values[index])
