from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import hashlib
import json
import os

import duckdb
import numpy as np
import pandas as pd

from agent_layer.data.timeline_types import TimelineKey


DecisionCacheProgress = Callable[[dict[str, object]], None]


STAGE_TO_CODE = {"open_auction": 0, "bar_close": 1}
CODE_TO_STAGE = {value: key for key, value in STAGE_TO_CODE.items()}


@dataclass(frozen=True)
class DecisionCacheRowSchema:
    numeric: tuple[str, ...] = (
        "execution_price",
        "limit_reference_close",
        "limit_pct",
        "liquidity_volume",
        "liquidity_amount",
    )
    flags: tuple[str, ...] = (
        "is_st",
        "market_can_buy",
        "market_can_sell",
        "is_tradeable",
        "is_limit_up",
        "is_limit_down",
        "is_zero_volume",
    )


class AgentDecisionCache:
    """Precompiled decision metadata for Agent rollout collection.

    The Feature Store remains the durable source of truth. This sidecar turns
    per-step execution metadata, constraints, and decision context into compact
    memory-mapped arrays so training does not repeatedly re-run the same DuckDB
    joins while sampling episodes.
    """

    VERSION = 1

    def __init__(
        self,
        store_path: str | Path,
        *,
        universe: Iterable[str],
        schema_hash: str,
        context_names: Iterable[str],
        progress_callback: DecisionCacheProgress | None = None,
    ) -> None:
        self.store_path = Path(store_path).resolve()
        self.universe = tuple(str(symbol).upper() for symbol in universe)
        self.schema_hash = str(schema_hash)
        self.context_names = tuple(str(name) for name in context_names)
        self.row_schema = DecisionCacheRowSchema()
        stat = self.store_path.stat()
        fingerprint_source = json.dumps(
            {
                "version": self.VERSION,
                "store": str(self.store_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "schema_hash": self.schema_hash,
                "universe": self.universe,
                "context": self.context_names,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_source).hexdigest()[:20]
        self.cache_dir = self.store_path.parent / ".agent_decision_cache" / fingerprint
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self._manifest: dict[str, object] | None = None
        self._arrays: dict[str, np.ndarray] = {}
        if not self._is_ready():
            self._build(progress_callback=progress_callback)
        elif progress_callback:
            progress_callback(
                {
                    "phase": "agent_cache",
                    "completed": 0,
                    "total": 0,
                    "cached": True,
                    "message": "Agent decision cache ready",
                }
            )
        self._load_manifest()

    def timeline(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
    ) -> list[TimelineKey]:
        step_times = self._array("step_times")
        step_stage_codes = self._array("step_stage_codes")
        step_starts = self._array("step_starts")
        step_ends = self._array("step_ends")
        mask = np.ones(len(step_times), dtype=np.bool_)
        if start is not None:
            mask &= step_times >= int(pd.Timestamp(start).value)
        if end is not None:
            end_timestamp = pd.Timestamp(end)
            if end_timestamp == end_timestamp.normalize():
                end_timestamp += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            mask &= step_times <= int(end_timestamp.value)
        stage_codes = _stage_codes(stages)
        if stage_codes:
            mask &= np.isin(step_stage_codes, np.asarray(stage_codes, dtype=np.int8))
        selected = np.flatnonzero(mask)
        return [
            TimelineKey(
                pd.Timestamp(int(step_times[index])),
                CODE_TO_STAGE[int(step_stage_codes[index])],
                int(step_ends[index] - step_starts[index]),
            )
            for index in selected
        ]

    def query_rows(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        stage: str | None = None,
    ) -> pd.DataFrame:
        step_times = self._array("step_times")
        step_stage_codes = self._array("step_stage_codes")
        step_starts = self._array("step_starts")
        step_ends = self._array("step_ends")
        start_ns = int(pd.Timestamp(start).value)
        end_ns = int(pd.Timestamp(end).value)
        mask = (step_times >= start_ns) & (step_times <= end_ns)
        if stage is not None:
            mask &= step_stage_codes == STAGE_TO_CODE[str(stage)]
        step_indices = np.flatnonzero(mask)
        if not len(step_indices):
            return self._empty_frame()
        row_indices = np.concatenate(
            [
                np.arange(int(step_starts[index]), int(step_ends[index]), dtype=np.int64)
                for index in step_indices
            ]
        )
        return self._rows_frame(row_indices)

    def _rows_frame(self, row_indices: np.ndarray) -> pd.DataFrame:
        manifest = self._load_manifest()
        symbols = tuple(str(value) for value in manifest["symbols"])
        adjusts = tuple(str(value) for value in manifest["adjusts"])
        times = self._array("row_times")[row_indices]
        stage_codes = self._array("row_stage_codes")[row_indices]
        symbol_codes = self._array("row_symbol_codes")[row_indices]
        adjust_codes = self._array("row_adjust_codes")[row_indices]
        numeric = self._array("numeric")[row_indices]
        flags = self._array("flags")[row_indices]
        context = self._array("context")[row_indices]
        symbol_values = [symbols[int(code)] for code in symbol_codes]
        adjust_values = [adjusts[int(code)] for code in adjust_codes]
        stage_values = [CODE_TO_STAGE[int(code)] for code in stage_codes]
        timestamp_values = pd.to_datetime(times)
        data: dict[str, object] = {
            "decision_time": timestamp_values,
            "stage": stage_values,
            "decision_id": [
                _decision_id(symbol, adjust, timestamp, stage_value)
                for symbol, adjust, timestamp, stage_value in zip(
                    symbol_values,
                    adjust_values,
                    timestamp_values,
                    stage_values,
                )
            ],
            "symbol": symbol_values,
            "adjust": adjust_values,
        }
        for column_index, name in enumerate(self.row_schema.numeric):
            data[name] = numeric[:, column_index]
        for column_index, name in enumerate(self.row_schema.flags):
            data[name] = flags[:, column_index]
        for column_index, name in enumerate(self.context_names):
            data[name] = context[:, column_index]
        return pd.DataFrame(data)

    def _empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "decision_time",
                "stage",
                "decision_id",
                "symbol",
                "adjust",
                *self.row_schema.numeric,
                *self.row_schema.flags,
                *self.context_names,
            ]
        )

    def _build(
        self,
        *,
        progress_callback: DecisionCacheProgress | None,
    ) -> None:
        if progress_callback:
            progress_callback(
                {
                    "phase": "agent_cache",
                    "completed": 0,
                    "total": 2,
                    "cached": False,
                    "message": "Preparing agent decision cache",
                }
            )
        frame = self._query_source_frame()
        if progress_callback:
            progress_callback(
                {
                    "phase": "agent_cache",
                    "completed": 1,
                    "total": 2,
                    "cached": False,
                    "message": "Writing agent decision cache",
                }
            )
        self._write_frame(frame)
        if progress_callback:
            progress_callback(
                {
                    "phase": "agent_cache",
                    "completed": 2,
                    "total": 2,
                    "cached": False,
                    "message": "Agent decision cache ready",
                }
            )

    def _query_source_frame(self) -> pd.DataFrame:
        context_columns = _context_select_list(self.context_names)
        with duckdb.connect(str(self.store_path), read_only=True) as conn:
            conn.execute("SET threads = 4")
            frame = conn.execute(
                "SELECT d.decision_time, d.stage, d.symbol, d.adjust, "
                "d.source_bar_end, "
                "d.execution_price, d.limit_reference_close, d.limit_pct, "
                "d.is_st, d.market_can_buy, d.market_can_sell, d.is_tradeable, "
                "d.is_limit_up, d.is_limit_down, d.is_zero_volume, "
                f"{context_columns} "
                "FROM decisions d "
                "LEFT JOIN decision_context dc USING (decision_id) "
                "LEFT JOIN constraints c USING (decision_id) "
                "WHERE d.symbol IN (SELECT UNNEST(?)) "
                "ORDER BY d.decision_time, CASE d.stage WHEN 'open_auction' THEN 0 ELSE 1 END, d.symbol",
                [list(self.universe)],
            ).fetchdf()
            minimum_source = frame["source_bar_end"].min()
            maximum_source = frame["source_bar_end"].max()
            intraday = conn.execute(
                "SELECT symbol, adjust, bar_datetime, volume, amount FROM market_bars "
                "WHERE symbol IN (SELECT UNNEST(?)) AND freq = '5min' "
                "AND bar_datetime >= ? AND bar_datetime <= ?",
                [list(self.universe), minimum_source, maximum_source],
            ).fetchdf()
            daily = conn.execute(
                "SELECT symbol, adjust, bar_datetime, volume, amount FROM market_bars "
                "WHERE symbol IN (SELECT UNNEST(?)) AND freq = 'daily' "
                "ORDER BY symbol, adjust, bar_datetime",
                [list(self.universe)],
            ).fetchdf()
        return _attach_liquidity(frame, intraday=intraday, daily=daily)

    def _write_frame(self, frame: pd.DataFrame) -> None:
        frame["decision_time"] = pd.to_datetime(frame["decision_time"], errors="coerce")
        frame = frame.loc[frame["decision_time"].notna()].reset_index(drop=True)
        symbols = list(self.universe)
        symbol_lookup = {symbol: index for index, symbol in enumerate(symbols)}
        adjusts = sorted(dict.fromkeys(frame["adjust"].astype(str)))
        adjust_lookup = {adjust: index for index, adjust in enumerate(adjusts)}
        row_times = frame["decision_time"].to_numpy(dtype="datetime64[ns]").view(np.int64)
        row_stage_codes = np.asarray(
            [STAGE_TO_CODE[str(value)] for value in frame["stage"]],
            dtype=np.int8,
        )
        row_symbol_codes = np.asarray(
            [symbol_lookup[str(value)] for value in frame["symbol"]],
            dtype=np.int32,
        )
        row_adjust_codes = np.asarray(
            [adjust_lookup[str(value)] for value in frame["adjust"]],
            dtype=np.int16,
        )
        numeric = (
            frame.loc[:, list(self.row_schema.numeric)]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=True)
        )
        flags = (
            frame.loc[:, list(self.row_schema.flags)]
            .fillna(False)
            .astype(bool)
            .to_numpy(dtype=np.bool_, copy=True)
        )
        context = (
            frame.loc[:, [f"__ctx_{index}" for index in range(len(self.context_names))]]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=True)
        )
        grouped = frame.groupby(["decision_time", "stage"], sort=False).indices
        step_times: list[int] = []
        step_stage_codes: list[int] = []
        step_starts: list[int] = []
        step_ends: list[int] = []
        for (timestamp, stage), indices in grouped.items():
            values = np.asarray(indices, dtype=np.int64)
            step_times.append(int(pd.Timestamp(timestamp).value))
            step_stage_codes.append(STAGE_TO_CODE[str(stage)])
            step_starts.append(int(values.min()))
            step_ends.append(int(values.max()) + 1)

        _atomic_numpy_save(self.cache_dir / "row_times.npy", row_times)
        _atomic_numpy_save(self.cache_dir / "row_stage_codes.npy", row_stage_codes)
        _atomic_numpy_save(self.cache_dir / "row_symbol_codes.npy", row_symbol_codes)
        _atomic_numpy_save(self.cache_dir / "row_adjust_codes.npy", row_adjust_codes)
        _atomic_numpy_save(self.cache_dir / "numeric.npy", numeric)
        _atomic_numpy_save(self.cache_dir / "flags.npy", flags)
        _atomic_numpy_save(self.cache_dir / "context.npy", context)
        _atomic_numpy_save(
            self.cache_dir / "step_times.npy",
            np.asarray(step_times, dtype=np.int64),
        )
        _atomic_numpy_save(
            self.cache_dir / "step_stage_codes.npy",
            np.asarray(step_stage_codes, dtype=np.int8),
        )
        _atomic_numpy_save(
            self.cache_dir / "step_starts.npy",
            np.asarray(step_starts, dtype=np.int64),
        )
        _atomic_numpy_save(
            self.cache_dir / "step_ends.npy",
            np.asarray(step_ends, dtype=np.int64),
        )
        manifest = {
            "version": self.VERSION,
            "store_path": str(self.store_path),
            "schema_hash": self.schema_hash,
            "symbols": symbols,
            "adjusts": adjusts,
            "context_names": list(self.context_names),
            "numeric_names": list(self.row_schema.numeric),
            "flag_names": list(self.row_schema.flags),
            "rows": int(len(frame)),
            "steps": int(len(step_times)),
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def _is_ready(self) -> bool:
        required = [
            "manifest.json",
            "row_times.npy",
            "row_stage_codes.npy",
            "row_symbol_codes.npy",
            "row_adjust_codes.npy",
            "numeric.npy",
            "flags.npy",
            "context.npy",
            "step_times.npy",
            "step_stage_codes.npy",
            "step_starts.npy",
            "step_ends.npy",
        ]
        return all((self.cache_dir / name).exists() for name in required)

    def _array(self, name: str) -> np.ndarray:
        cached = self._arrays.get(name)
        if cached is not None:
            return cached
        values = np.load(self.cache_dir / f"{name}.npy", mmap_mode="r")
        self._arrays[name] = values
        return values

    def _load_manifest(self) -> dict[str, object]:
        if self._manifest is None:
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest


def _context_select_list(context_names: tuple[str, ...]) -> str:
    parts = []
    for index, name in enumerate(context_names):
        quoted = _quote_identifier(name)
        source = "dc" if name in {
            "bar_slot_norm",
            "day_progress",
            "is_morning_session",
            "is_afternoon_session",
            "minutes_to_close_norm",
            "is_open_auction",
        } else "c"
        parts.append(f"COALESCE({source}.{quoted}, 0) AS __ctx_{index}")
    return ", ".join(parts) if parts else "0 AS __ctx_empty"


def _attach_liquidity(
    frame: pd.DataFrame,
    *,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["decision_time"] = pd.to_datetime(result["decision_time"], errors="coerce")
    result["source_bar_end"] = pd.to_datetime(result["source_bar_end"], errors="coerce")
    result["liquidity_volume"] = 0.0
    result["liquidity_amount"] = 0.0
    if not intraday.empty:
        intraday = intraday.copy()
        intraday["bar_datetime"] = pd.to_datetime(
            intraday["bar_datetime"], errors="coerce"
        )
        intraday = intraday.rename(
            columns={
                "bar_datetime": "source_bar_end",
                "volume": "__volume",
                "amount": "__amount",
            }
        )
        subset = result.loc[
            ~result["stage"].astype(str).eq("open_auction"),
            ["symbol", "adjust", "source_bar_end"],
        ].copy()
        subset["__row"] = subset.index.to_numpy()
        merged = subset.merge(
            intraday.loc[:, ["symbol", "adjust", "source_bar_end", "__volume", "__amount"]],
            on=["symbol", "adjust", "source_bar_end"],
            how="left",
        )
        rows = merged["__row"].to_numpy(dtype=np.int64)
        result.loc[rows, "liquidity_volume"] = (
            pd.to_numeric(merged["__volume"], errors="coerce").fillna(0.0).to_numpy()
        )
        result.loc[rows, "liquidity_amount"] = (
            pd.to_numeric(merged["__amount"], errors="coerce").fillna(0.0).to_numpy()
        )
    if not daily.empty:
        daily = daily.copy()
        daily["bar_datetime"] = pd.to_datetime(daily["bar_datetime"], errors="coerce")
        daily = daily.loc[daily["bar_datetime"].notna()].sort_values(
            ["symbol", "adjust", "bar_datetime"]
        )
        lookup: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for (symbol, adjust), group in daily.groupby(["symbol", "adjust"], sort=False):
            lookup[(str(symbol), str(adjust))] = (
                group["bar_datetime"].to_numpy(dtype="datetime64[ns]").view(np.int64),
                pd.to_numeric(group["volume"], errors="coerce").fillna(0.0).to_numpy(
                    dtype=np.float64
                ),
                pd.to_numeric(group["amount"], errors="coerce").fillna(0.0).to_numpy(
                    dtype=np.float64
                ),
            )
        open_rows = result.loc[
            result["stage"].astype(str).eq("open_auction"),
            ["symbol", "adjust", "decision_time"],
        ].copy()
        open_rows["__row"] = open_rows.index.to_numpy()
        open_rows["__day"] = open_rows["decision_time"].dt.normalize()
        for (symbol, adjust), group in open_rows.groupby(["symbol", "adjust"], sort=False):
            arrays = lookup.get((str(symbol), str(adjust)))
            if arrays is None:
                continue
            times, volumes, amounts = arrays
            days = group["__day"].to_numpy(dtype="datetime64[ns]").view(np.int64)
            positions = np.searchsorted(times, days, side="left") - 1
            valid = positions >= 0
            if not np.any(valid):
                continue
            rows = group["__row"].to_numpy(dtype=np.int64)[valid]
            result.loc[rows, "liquidity_volume"] = volumes[positions[valid]]
            result.loc[rows, "liquidity_amount"] = amounts[positions[valid]]
    return result.drop(columns=["source_bar_end"])


def _stage_codes(stages: Iterable[str] | None) -> tuple[int, ...]:
    result: list[int] = []
    for stage in stages or []:
        normalized = str(stage)
        if normalized not in STAGE_TO_CODE:
            raise ValueError(f"Unsupported decision stage: {normalized}")
        result.append(STAGE_TO_CODE[normalized])
    return tuple(dict.fromkeys(result))


def _decision_id(
    symbol: str,
    adjust: str,
    timestamp: pd.Timestamp,
    stage: str,
) -> str:
    return f"{symbol}|{adjust}|{pd.Timestamp(timestamp).strftime('%Y%m%d%H%M%S')}|{stage}"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _atomic_numpy_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)
