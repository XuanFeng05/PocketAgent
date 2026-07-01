from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock
from typing import Callable, Iterable

import duckdb
import numpy as np
import pandas as pd


CacheProgress = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class MarketCacheKey:
    symbol: str
    freq: str
    adjust: str


class CompactMarketFeatureCache:
    """Memory-mapped market features used by the Agent timeline loader.

    The compact DuckDB store is optimized for durable storage, not for repeatedly
    expanding overlapping rolling windows. This sidecar keeps one float32 matrix
    per symbol/frequency and lets training slice those windows without another SQL
    join or Pandas groupby.
    """

    VERSION = 3
    RAW_COLUMNS = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pctChg",
        "turn",
        "progress",
    )

    def __init__(
        self,
        store_path: str | Path,
        *,
        feature_names: dict[str, tuple[str, ...]],
        keys: Iterable[MarketCacheKey],
        schema_hash: str,
        workers: int | None = None,
        progress_callback: CacheProgress | None = None,
    ) -> None:
        self.store_path = Path(store_path).resolve()
        self.feature_names = {
            str(freq): tuple(str(name) for name in names)
            for freq, names in feature_names.items()
        }
        self.schema_hash = str(schema_hash)
        self.keys = tuple(dict.fromkeys(keys))
        stat = self.store_path.stat()
        fingerprint_source = json.dumps(
            {
                "version": self.VERSION,
                "store": str(self.store_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "schema_hash": self.schema_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_source).hexdigest()[:20]
        self.cache_dir = self.store_path.parent / ".agent_market_cache" / fingerprint
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._arrays: dict[
            MarketCacheKey, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._snapshot_engines: dict[MarketCacheKey, object] = {}
        self._arrays_lock = Lock()
        self._ensure_files(
            workers=workers,
            progress_callback=progress_callback,
        )

    def load_values(
        self,
        *,
        symbol: str,
        freq: str,
        adjust: str,
        end: object,
        limit: int,
    ) -> np.ndarray:
        names = self.feature_names[str(freq)]
        if end is None or int(limit) <= 0:
            return np.empty((0, len(names)), dtype=np.float32)
        try:
            end_ns = int(pd.Timestamp(end).value)
        except (TypeError, ValueError):
            return np.empty((0, len(names)), dtype=np.float32)
        if end_ns == np.iinfo(np.int64).min:
            return np.empty((0, len(names)), dtype=np.float32)

        key = MarketCacheKey(str(symbol), str(freq), str(adjust))
        times, values, _ = self._arrays_for(key)
        stop = int(np.searchsorted(times, end_ns, side="right"))
        start = max(0, stop - int(limit))
        return np.asarray(values[start:stop], dtype=np.float32)

    def load_raw_values(
        self,
        *,
        symbol: str,
        freq: str,
        adjust: str,
        end: object,
        limit: int,
    ) -> np.ndarray:
        if end is None or int(limit) <= 0:
            return np.empty((0, len(self.RAW_COLUMNS)), dtype=np.float64)
        try:
            end_ns = int(pd.Timestamp(end).value)
        except (TypeError, ValueError):
            return np.empty((0, len(self.RAW_COLUMNS)), dtype=np.float64)
        if end_ns == np.iinfo(np.int64).min:
            return np.empty((0, len(self.RAW_COLUMNS)), dtype=np.float64)
        key = MarketCacheKey(str(symbol), str(freq), str(adjust))
        times, _, raw = self._arrays_for(key)
        stop = int(np.searchsorted(times, end_ns, side="right"))
        start = max(0, stop - int(limit))
        return np.asarray(raw[start:stop], dtype=np.float64)

    def snapshot_engine(
        self,
        *,
        symbol: str,
        freq: str,
        adjust: str,
        spec,
    ):
        from feature_layer.builders.snapshot_features import SnapshotFeatureEngine

        key = MarketCacheKey(str(symbol), str(freq), str(adjust))
        with self._arrays_lock:
            engine = self._snapshot_engines.get(key)
        if engine is not None:
            return engine
        times, _, raw = self._arrays_for(key)
        engine = SnapshotFeatureEngine(
            np.asarray(times),
            np.asarray(raw, dtype=np.float64),
            spec=spec,
            freq=str(freq),
        )
        with self._arrays_lock:
            self._snapshot_engines[key] = engine
        return engine

    def _ensure_files(
        self,
        *,
        workers: int | None,
        progress_callback: CacheProgress | None,
    ) -> None:
        missing = [key for key in self.keys if not self._is_ready(key)]
        total = len(missing)
        if not missing:
            if progress_callback:
                progress_callback(
                    {"phase": "data_cache", "completed": 0, "total": 0, "cached": True}
                )
            return

        maximum_workers = workers or min(8, max(1, (os.cpu_count() or 2) // 2))
        maximum_workers = max(1, min(int(maximum_workers), total))
        completed = 0
        with ThreadPoolExecutor(
            max_workers=maximum_workers,
            thread_name_prefix="feature-cache",
        ) as executor:
            futures = {executor.submit(self._build_key, key): key for key in missing}
            for future in as_completed(futures):
                future.result()
                completed += 1
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "data_cache",
                            "completed": completed,
                            "total": total,
                            "cached": False,
                        }
                    )

    def _build_key(self, key: MarketCacheKey) -> None:
        columns = self.feature_names[key.freq]
        quoted = ", ".join(f"mf.{_quote_identifier(name)}" for name in columns)
        raw_select = ", ".join(
            f"mb.{_quote_identifier(name)} AS {_quote_identifier('__raw_' + name)}"
            for name in self.RAW_COLUMNS
        )
        with duckdb.connect(str(self.store_path), read_only=True) as conn:
            conn.execute("SET threads = 1")
            frame = conn.execute(
                f"SELECT mf.bar_datetime, {quoted}, {raw_select} "
                "FROM market_features mf JOIN market_bars mb USING (market_row_id) "
                "WHERE mf.symbol = ? AND mf.freq = ? AND mf.adjust = ? "
                "ORDER BY mf.bar_datetime",
                [key.symbol, key.freq, key.adjust],
            ).fetchdf()

        times = pd.to_datetime(frame.pop("bar_datetime"), errors="coerce")
        valid = times.notna().to_numpy()
        time_values = times.loc[valid].to_numpy(dtype="datetime64[ns]").view(np.int64)
        raw_names = [f"__raw_{name}" for name in self.RAW_COLUMNS]
        raw = (
            frame.loc[valid, raw_names]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64, copy=True)
        )
        values = (
            frame.loc[valid, list(columns)]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=True)
        )
        times_path, values_path, raw_path = self._paths(key)
        _atomic_numpy_save(times_path, time_values)
        _atomic_numpy_save(values_path, values)
        _atomic_numpy_save(raw_path, raw)

    def _arrays_for(
        self, key: MarketCacheKey
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._arrays_lock:
            cached = self._arrays.get(key)
            if cached is not None:
                return cached
            if not self._is_ready(key):
                self._build_key(key)
            times_path, values_path, raw_path = self._paths(key)
            arrays = (
                np.load(times_path, mmap_mode="r"),
                np.load(values_path, mmap_mode="r"),
                np.load(raw_path, mmap_mode="r"),
            )
            self._arrays[key] = arrays
            return arrays

    def _is_ready(self, key: MarketCacheKey) -> bool:
        times_path, values_path, raw_path = self._paths(key)
        return times_path.exists() and values_path.exists() and raw_path.exists()

    def _paths(self, key: MarketCacheKey) -> tuple[Path, Path, Path]:
        stem = "__".join(
            _safe_component(value) for value in (key.symbol, key.freq, key.adjust)
        )
        return (
            self.cache_dir / f"{stem}.times.npy",
            self.cache_dir / f"{stem}.values.npy",
            self.cache_dir / f"{stem}.raw.npy",
        )


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _atomic_numpy_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)
