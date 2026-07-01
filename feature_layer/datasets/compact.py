from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from data_layer.storage.duckdb_storage import load_kline_from_duckdb
from feature_layer.builders.aggregation import aggregate_ohlcv_from_base, normalize_frequency
from feature_layer.builders.bar_features import build_bar_features
from feature_layer.datasets.builder import (
    FeatureDataset,
    FeatureDatasetConfig,
    _current_partial_daily,
    _prepare_kline_frame,
    build_feature_dataset_from_frames,
    load_feature_source_frames_from_duckdb,
    load_feature_source_frames_from_parquet_cache,
)
from feature_layer.datasets.sequence import PaddedMarketSequence, pad_market_sequence
from feature_layer.indicator_registry import (
    feature_spec_from_indicators,
    indicator_lookback,
    indicator_to_payload,
    validate_indicator_specs,
)
from feature_layer.specs import DEFAULT_FEATURE_SPEC, FeatureSpec
from feature_layer.model_input_registry import (
    compile_model_input_blueprint,
    load_model_input_blueprint,
)
from feature_layer.datasets.market_cache import CompactMarketFeatureCache


@dataclass(frozen=True)
class CompactFeatureDataset:
    spec: FeatureSpec
    dataset: FeatureDataset
    market_bars: pd.DataFrame
    market_features: pd.DataFrame
    decision_index: pd.DataFrame
    decision_snapshots: pd.DataFrame
    model_input_blueprint: dict[str, object]
    compiled_model_input: dict[str, object]

    def summary(self) -> dict[str, object]:
        result = self.dataset.summary()
        result.update(
            {
                "market_rows": {
                    freq: (
                        int(self.market_features["freq"].eq(freq).sum())
                        if "freq" in self.market_features
                        else 0
                    )
                    for freq in self.dataset.frequencies
                },
                "decision_index_rows": int(len(self.decision_index)),
                "snapshot_rows": int(len(self.decision_snapshots)),
                "model_input_schema_hash": self.compiled_model_input.get("schema_hash"),
            }
        )
        return result


@dataclass(frozen=True)
class ModelInputBatch:
    market_sequences: dict[str, np.ndarray]
    sequence_masks: dict[str, np.ndarray]
    valid_ratios: dict[str, float]
    decision_context: np.ndarray
    runtime_contract: tuple[str, ...]
    feature_names: dict[str, tuple[str, ...]]
    decision_context_names: tuple[str, ...]
    schema_hash: str


def build_compact_feature_dataset_from_duckdb(
    db_path: str | Path,
    *,
    symbols: Iterable[str],
    adjust: str,
    start: str | None,
    end: str | None,
    config: FeatureDatasetConfig,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    model_input_blueprint: dict[str, object] | None = None,
    source_cache_dir: str | Path | None = None,
) -> CompactFeatureDataset:
    blueprint = model_input_blueprint or load_model_input_blueprint(spec=spec)
    compiled_model_input = compile_model_input_blueprint(blueprint, spec=spec)
    compact_config = replace(config, materialize_sequences=False)
    if source_cache_dir is not None:
        source = load_feature_source_frames_from_parquet_cache(
            source_cache_dir,
            symbols=symbols,
            adjust=adjust,
            start=start,
            end=end,
            config=compact_config,
            spec=spec,
        )
    else:
        source = load_feature_source_frames_from_duckdb(
            db_path,
            symbols=symbols,
            adjust=adjust,
            start=start,
            end=end,
            config=compact_config,
            spec=spec,
        )
    dataset = build_feature_dataset_from_frames(
        source,
        adjust=adjust,
        start=start,
        end=end,
        config=compact_config,
        spec=spec,
    )
    eligible = sorted(set(dataset.requested_symbols))
    base = _prepare_kline_frame(source.base, freq=normalize_frequency(config.trade_freq), adjust=adjust)
    daily = _prepare_kline_frame(source.daily, freq="daily", adjust=adjust)

    stable_by_freq = _load_stable_bars(
        db_path,
        symbols=eligible,
        adjust=adjust,
        end=end,
        base=base,
        daily=daily,
        frequencies=dataset.frequencies,
        trade_freq=normalize_frequency(config.trade_freq),
    )
    market_bar_parts: list[pd.DataFrame] = []
    market_feature_parts: list[pd.DataFrame] = []
    for freq, bars in stable_by_freq.items():
        prepared = _standardize_market_bars(bars, freq=freq)
        if prepared.empty:
            continue
        features = build_bar_features(prepared, spec=spec).rename(
            columns={"datetime": "bar_datetime"}
        )
        prepared = prepared.rename(columns={"datetime": "bar_datetime"})
        prepared.insert(0, "market_row_id", _market_row_ids(prepared))
        features.insert(0, "market_row_id", prepared["market_row_id"].to_numpy())
        market_bar_parts.append(prepared)
        market_feature_parts.append(features)

    market_bars = (
        pd.concat(market_bar_parts, ignore_index=True)
        if market_bar_parts
        else pd.DataFrame(columns=["market_row_id", *_market_bar_columns(bar_datetime=True)])
    )
    market_features = (
        pd.concat(market_feature_parts, ignore_index=True)
        if market_feature_parts
        else pd.DataFrame(
            columns=["market_row_id", "symbol", "bar_datetime", "freq", "adjust", *spec.market_feature_names]
        )
    )
    decision_index, snapshots = _build_decision_index_fast(
        dataset.decisions,
        stable_by_freq=stable_by_freq,
        frequencies=dataset.frequencies,
        sequence_windows=spec.sequence_windows,
        trade_freq=normalize_frequency(config.trade_freq),
    )
    dataset = replace(
        dataset,
        market={
            freq: market_features.loc[market_features["freq"].eq(freq)].reset_index(drop=True)
            for freq in dataset.frequencies
        } if not market_features.empty else {freq: pd.DataFrame() for freq in dataset.frequencies},
    )
    return CompactFeatureDataset(
        spec=spec,
        dataset=dataset,
        market_bars=market_bars,
        market_features=market_features,
        decision_index=decision_index,
        decision_snapshots=snapshots,
        model_input_blueprint=blueprint,
        compiled_model_input=compiled_model_input,
    )


COMPACT_FEATURE_STORE_TABLES: tuple[str, ...] = (
    "decisions",
    "decision_context",
    "constraints",
    "market_bars",
    "market_features",
    "decision_index",
    "decision_snapshots",
    "dataset_metadata",
)


class CompactFeatureStoreWriter:
    """Write compact feature chunks through one DuckDB connection."""

    def __init__(self, path: str | Path, *, reset: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self._conn = duckdb.connect(str(self.path))
        self._has_written = False
        self._closed = False

    def append(self, compact: CompactFeatureDataset) -> None:
        if self._closed:
            raise RuntimeError("CompactFeatureStoreWriter is already closed.")
        _write_compact_tables_to_connection(
            self._conn,
            _compact_feature_store_tables(compact),
            append=self._has_written,
        )
        self._has_written = True

    def finalize(self, *, create_indexes: bool = True) -> Path:
        if self._closed:
            return self.path
        if create_indexes:
            _create_compact_feature_store_indexes(self._conn)
        self.close()
        return self.path

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "CompactFeatureStoreWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class CompactFeatureParquetWriter:
    """Write compact feature table parts as independent parquet files.

    This is the distributed-ready output path: every worker writes only its own
    part files, and the driver later merges them into the final DuckDB store.
    """

    def __init__(self, root_dir: str | Path, *, reset: bool = True) -> None:
        self.root_dir = Path(root_dir)
        if reset and self.root_dir.exists():
            import shutil
            shutil.rmtree(self.root_dir, ignore_errors=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for table in COMPACT_FEATURE_STORE_TABLES:
            (self.root_dir / table).mkdir(parents=True, exist_ok=True)
        self._part_index = 0
        self._metadata_written = False

    def append(self, compact: CompactFeatureDataset) -> None:
        tables = _compact_feature_store_tables(compact)
        part_name = f"part_{self._part_index:06d}.parquet"
        for table_name in COMPACT_FEATURE_STORE_TABLES:
            if table_name == "dataset_metadata" and self._metadata_written:
                continue
            frame = tables[table_name]
            if frame.empty and self._part_index > 0:
                continue
            _write_dataframe_parquet(frame, self.root_dir / table_name / part_name)
            if table_name == "dataset_metadata":
                self._metadata_written = True
        self._part_index += 1

    @property
    def part_count(self) -> int:
        return self._part_index


def _write_dataframe_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    with duckdb.connect(":memory:") as conn:
        conn.execute("SET threads = 1")
        conn.register("_feature_part_df", frame)
        try:
            conn.execute(
                f"COPY (SELECT * FROM _feature_part_df) TO {_duckdb_string_literal(temp)} (FORMAT PARQUET)"
            )
        finally:
            conn.unregister("_feature_part_df")
    temp.replace(output)
    return output


def merge_compact_feature_parquet_parts(
    part_roots: Iterable[str | Path],
    output_path: str | Path,
    *,
    create_indexes: bool = True,
    metadata_frame: pd.DataFrame | None = None,
) -> Path:
    roots = [Path(root) for root in part_roots]
    if not roots:
        raise ValueError("No compact feature parquet part directories were provided for merge.")
    store_path = Path(output_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        store_path.unlink()
    with duckdb.connect(str(store_path)) as conn:
        metadata_written = False
        if metadata_frame is not None:
            registered = "_dataset_metadata_df"
            conn.register(registered, metadata_frame)
            try:
                conn.execute(
                    f'CREATE OR REPLACE TABLE {_quote_identifier("dataset_metadata")} AS '
                    f'SELECT * FROM {registered}'
                )
            finally:
                conn.unregister(registered)
            metadata_written = True
        for table_name in COMPACT_FEATURE_STORE_TABLES:
            if table_name == "dataset_metadata" and metadata_written:
                continue
            files: list[Path] = []
            for root in roots:
                files.extend(sorted((root / table_name).glob("*.parquet")))
            if not files:
                continue
            if table_name == "dataset_metadata":
                first = files[0]
                conn.execute(
                    f'CREATE OR REPLACE TABLE {_quote_identifier(table_name)} AS '
                    f'SELECT * FROM read_parquet({_duckdb_string_literal(first)})'
                )
            else:
                conn.execute(
                    f'CREATE OR REPLACE TABLE {_quote_identifier(table_name)} AS '
                    f'SELECT * FROM read_parquet({_duckdb_list_literal(files)})'
                )
        if create_indexes:
            _create_compact_feature_store_indexes(conn)
    return store_path


def _duckdb_list_literal(paths: Iterable[str | Path]) -> str:
    return "[" + ", ".join(_duckdb_string_literal(path) for path in paths) + "]"


def _compact_feature_store_tables(compact: CompactFeatureDataset) -> dict[str, pd.DataFrame]:
    return {
        "decisions": compact.dataset.decisions,
        "decision_context": compact.dataset.decision_context,
        "constraints": compact.dataset.constraints,
        "market_bars": compact.market_bars,
        "market_features": compact.market_features,
        "decision_index": compact.decision_index,
        "decision_snapshots": compact.decision_snapshots,
        "dataset_metadata": _metadata_frame(
            compact.spec,
            compact.model_input_blueprint,
            compact.compiled_model_input,
        ),
    }


def _write_compact_tables_to_connection(
    conn: duckdb.DuckDBPyConnection,
    tables: dict[str, pd.DataFrame],
    *,
    append: bool,
) -> None:
    for table_name in COMPACT_FEATURE_STORE_TABLES:
        frame = tables[table_name]
        if append and table_name == "dataset_metadata":
            continue
        if append and frame.empty:
            continue
        registered = f"_{table_name}_df"
        conn.register(registered, frame)
        try:
            if append:
                columns = ", ".join(f'"{column}"' for column in frame.columns)
                conn.execute(
                    f'INSERT INTO "{table_name}" ({columns}) SELECT {columns} FROM {registered}'
                )
            else:
                conn.execute(
                    f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM {registered}'
                )
        finally:
            conn.unregister(registered)


def _duckdb_string_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def merge_compact_feature_stores(
    chunk_store_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    create_indexes: bool = True,
) -> Path:
    """Merge worker-owned chunk stores into one final compact feature store."""

    chunks = [Path(path) for path in chunk_store_paths]
    if not chunks:
        raise ValueError("No compact feature chunk stores were provided for merge.")
    store_path = Path(output_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        store_path.unlink()
    with duckdb.connect(str(store_path)) as conn:
        first_chunk = True
        for index, chunk_path in enumerate(chunks):
            if not chunk_path.exists():
                raise FileNotFoundError(f"Missing compact feature chunk store: {chunk_path}")
            alias = f"chunk_{index}"
            conn.execute(f"ATTACH {_duckdb_string_literal(chunk_path)} AS {_quote_identifier(alias)}")
            try:
                for table_name in COMPACT_FEATURE_STORE_TABLES:
                    if not first_chunk and table_name == "dataset_metadata":
                        continue
                    table_ref = f'{_quote_identifier(alias)}.{_quote_identifier(table_name)}'
                    if first_chunk:
                        conn.execute(
                            f'CREATE OR REPLACE TABLE {_quote_identifier(table_name)} AS SELECT * FROM {table_ref}'
                        )
                    else:
                        columns = [
                            str(row[1])
                            for row in conn.execute(
                                f"PRAGMA table_info({_duckdb_string_literal(table_name)})"
                            ).fetchall()
                        ]
                        column_sql = ", ".join(_quote_identifier(column) for column in columns)
                        conn.execute(
                            f'INSERT INTO {_quote_identifier(table_name)} ({column_sql}) '
                            f'SELECT {column_sql} FROM {table_ref}'
                        )
                first_chunk = False
            finally:
                conn.execute(f"DETACH {_quote_identifier(alias)}")
        if create_indexes:
            _create_compact_feature_store_indexes(conn)
    return store_path


def write_compact_feature_store(
    compact: CompactFeatureDataset,
    path: str | Path,
    *,
    append: bool = False,
    create_indexes: bool = True,
) -> Path:
    store_path = Path(path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(store_path)) as conn:
        _write_compact_tables_to_connection(
            conn,
            _compact_feature_store_tables(compact),
            append=append,
        )
        if create_indexes:
            _create_compact_feature_store_indexes(conn)
    return store_path


def create_compact_feature_store_indexes(path: str | Path) -> Path:
    store_path = Path(path)
    with duckdb.connect(str(store_path)) as conn:
        _create_compact_feature_store_indexes(conn)
    return store_path


def _create_compact_feature_store_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    tables = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
    if "market_features" in tables:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_features_slice "
            "ON market_features(symbol, freq, adjust, bar_datetime)"
        )
    if "decision_index" in tables:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decision_index_lookup "
            "ON decision_index(decision_id, freq)"
        )


def load_compact_market_sequence(
    store_path: str | Path,
    *,
    decision_id: str,
    freq: str,
    spec: FeatureSpec | None = None,
) -> PaddedMarketSequence:
    """Resolve one indexed sequence and return a padded model tensor plus mask."""
    normalized = normalize_frequency(freq)
    with duckdb.connect(str(store_path), read_only=True) as conn:
        if spec is None:
            spec = _spec_from_store(conn)
        compiled_model_input = _compiled_model_input_from_store(conn, spec)
        index_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info('decision_index')").fetchall()
        }
        return _load_compact_market_sequence_from_connection(
            conn,
            decision_id=decision_id,
            normalized=normalized,
            spec=spec,
            compiled_model_input=compiled_model_input,
            index_columns=index_columns,
        )


def _load_compact_market_sequence_from_connection(
    conn: duckdb.DuckDBPyConnection,
    *,
    decision_id: str,
    normalized: str,
    spec: FeatureSpec,
    compiled_model_input: dict[str, object],
    index_columns: set[str],
) -> PaddedMarketSequence:
    feature_names = tuple(
        compiled_model_input.get("channels_by_frequency", {}).get(normalized, [])
    )
    if not feature_names:
        raise ValueError(f"Model input blueprint has no market channels for {normalized}.")
    snapshot_required_expr = (
        "snapshot_required" if "snapshot_required" in index_columns else "FALSE"
    )
    index = conn.execute(
        "SELECT symbol, adjust, sequence_window, stable_end_datetime, snapshot_id, "
        f"{snapshot_required_expr} AS snapshot_required "
        "FROM decision_index WHERE decision_id = ? AND freq = ?",
        [decision_id, normalized],
    ).fetchone()
    if index is None:
        raise KeyError(f"Decision index not found: {decision_id} / {normalized}")
    symbol, adjust, window, stable_end, snapshot_id, snapshot_required = index
    snapshot_count = 1 if snapshot_id or snapshot_required else 0
    stable_limit = max(0, int(window) - snapshot_count)
    if stable_end is not None and stable_limit:
        stable = conn.execute(
            "SELECT * EXCLUDE (market_row_id) FROM market_features "
            "WHERE symbol = ? AND freq = ? AND adjust = ? AND bar_datetime <= ? "
            "ORDER BY bar_datetime DESC LIMIT ?",
            [symbol, normalized, adjust, stable_end, stable_limit],
        ).fetchdf().sort_values("bar_datetime")
    else:
        stable = pd.DataFrame(columns=[*feature_names])

    if snapshot_id:
        snapshot = conn.execute(
            "SELECT * EXCLUDE (snapshot_id, decision_id) FROM decision_snapshots "
            "WHERE snapshot_id = ?",
            [snapshot_id],
        ).fetchdf()
        if snapshot.empty and snapshot_required:
            snapshot = _build_lazy_decision_snapshot(
                conn,
                decision_id=decision_id,
                freq=normalized,
                trade_freq=normalize_frequency(spec.trade_frequency),
            )
    elif snapshot_required:
        snapshot = _build_lazy_decision_snapshot(
            conn,
            decision_id=decision_id,
            freq=normalized,
            trade_freq=normalize_frequency(spec.trade_frequency),
        )
    else:
        snapshot = pd.DataFrame()
    if not snapshot.empty:
        lookback = max(
            [
                20,
                *(
                    indicator_lookback(item)
                    for item in spec.indicators
                    if item.enabled and normalized in item.frequencies
                ),
            ]
        )
        history_limit = max(int(window), lookback * 4)
        history = (
            conn.execute(
                "SELECT * EXCLUDE (market_row_id) FROM market_bars "
                "WHERE symbol = ? AND freq = ? AND adjust = ? AND bar_datetime <= ? "
                "ORDER BY bar_datetime DESC LIMIT ?",
                [symbol, normalized, adjust, stable_end, history_limit],
            ).fetchdf().sort_values("bar_datetime")
            if stable_end is not None
            else pd.DataFrame()
        )
        raw_parts = [
            frame.rename(columns={"bar_datetime": "datetime"}).dropna(axis=1, how="all")
            for frame in (history, snapshot)
            if frame is not None and not frame.empty
        ]
        raw = pd.concat(raw_parts, ignore_index=True)
        snapshot_features = build_bar_features(raw, spec=spec).tail(1).rename(
            columns={"datetime": "bar_datetime"}
        )
        stable = (
            snapshot_features.reset_index(drop=True)
            if stable.empty
            else pd.concat([stable, snapshot_features], ignore_index=True)
        )

    valid_ratio = min(1.0, len(stable) / float(max(1, int(window))))
    stable["sequence_valid_ratio"] = valid_ratio
    return pad_market_sequence(
        stable,
        feature_names=feature_names,
        window=int(window),
    )


def load_model_input_batch(
    store_path: str | Path,
    *,
    decision_id: str,
    frequencies: Iterable[str] | None = None,
) -> ModelInputBatch:
    """Load every blueprint-defined stream for one decision."""
    path = Path(store_path)
    with duckdb.connect(str(path), read_only=True) as conn:
        spec = _spec_from_store(conn)
        compiled = _compiled_model_input_from_store(conn, spec)
        available = [
            str(row[0])
            for row in conn.execute(
                "SELECT freq FROM decision_index WHERE decision_id = ? ORDER BY freq",
                [decision_id],
            ).fetchall()
        ]
        selected = [normalize_frequency(value) for value in (frequencies or available)]
        selected = list(dict.fromkeys(value for value in selected if value in available))
        index_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info('decision_index')").fetchall()
        }
        context_row: dict[str, Any] = {}
        for table_name in ("decision_context", "constraints"):
            frame = conn.execute(
                f"SELECT * FROM {table_name} WHERE decision_id = ? LIMIT 1",
                [decision_id],
            ).fetchdf()
            if not frame.empty:
                context_row.update(frame.iloc[0].to_dict())
        sequences: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        valid_ratios: dict[str, float] = {}
        for freq in selected:
            sequence = _load_compact_market_sequence_from_connection(
                conn,
                decision_id=decision_id,
                normalized=freq,
                spec=spec,
                compiled_model_input=compiled,
                index_columns=index_columns,
            )
            sequences[freq] = sequence.values
            masks[freq] = sequence.sequence_mask
            valid_ratios[freq] = sequence.valid_ratio

    feature_names = {
        freq: tuple(compiled.get("channels_by_frequency", {}).get(freq, []))
        for freq in selected
    }

    context_names = tuple(compiled.get("decision_context", []))
    context_values = np.asarray(
        [_numeric_context_value(context_row.get(name, 0.0)) for name in context_names],
        dtype=np.float32,
    )
    return ModelInputBatch(
        market_sequences=sequences,
        sequence_masks=masks,
        valid_ratios=valid_ratios,
        decision_context=context_values,
        runtime_contract=tuple(compiled.get("runtime_state", [])),
        feature_names=feature_names,
        decision_context_names=context_names,
        schema_hash=str(compiled.get("schema_hash") or ""),
    )


def load_model_input_batches(
    store_path: str | Path,
    *,
    decision_ids: Iterable[str],
    frequencies: Iterable[str] | None = None,
    market_cache: CompactMarketFeatureCache | None = None,
) -> dict[str, ModelInputBatch]:
    """Load one market-wide decision step with batched DuckDB scans."""
    path = Path(store_path)
    requested_ids = list(dict.fromkeys(str(value) for value in decision_ids if value))
    if not requested_ids:
        return {}

    with duckdb.connect(str(path), read_only=True) as conn:
        spec = _spec_from_store(conn)
        compiled = _compiled_model_input_from_store(conn, spec)
        configured = [
            normalize_frequency(value)
            for value in compiled.get("channels_by_frequency", {})
        ]
        selected = [
            normalize_frequency(value) for value in (frequencies or configured)
        ]
        selected = list(dict.fromkeys(value for value in selected if value in configured))
        if not selected:
            raise ValueError("No model input frequencies were selected.")

        index = conn.execute(
            "SELECT decision_id, symbol, adjust, decision_time, freq, sequence_window, "
            "stable_end_datetime, snapshot_id, snapshot_required, valid_rows, "
            "sequence_valid_ratio FROM decision_index "
            "WHERE decision_id IN (SELECT UNNEST(?)) "
            "AND freq IN (SELECT UNNEST(?))",
            [requested_ids, selected],
        ).fetchdf()
        expected = len(requested_ids) * len(selected)
        if len(index) != expected:
            found = set(index["decision_id"].astype(str)) if not index.empty else set()
            missing = [value for value in requested_ids if value not in found]
            raise KeyError(
                "Decision index is incomplete for the requested market step: "
                + ", ".join(missing[:20])
            )

        contexts = conn.execute(
            "SELECT dc.*, c.* EXCLUDE (decision_id) FROM decision_context dc "
            "LEFT JOIN constraints c USING (decision_id) "
            "WHERE dc.decision_id IN (SELECT UNNEST(?))",
            [requested_ids],
        ).fetchdf()
        decisions = conn.execute(
            "SELECT * FROM decisions WHERE decision_id IN (SELECT UNNEST(?))",
            [requested_ids],
        ).fetchdf()
        stable = (
            pd.DataFrame()
            if market_cache is not None
            else conn.execute(
                "WITH selected AS ("
                "  SELECT decision_id, symbol, adjust, freq, sequence_window, "
                "    stable_end_datetime, snapshot_required FROM decision_index "
                "  WHERE decision_id IN (SELECT UNNEST(?)) "
                "    AND freq IN (SELECT UNNEST(?))"
                "), ranked AS ("
                "  SELECT selected.decision_id, selected.freq, "
                "    market_features.* EXCLUDE (market_row_id, freq), "
                "    ROW_NUMBER() OVER ("
                "      PARTITION BY selected.decision_id, selected.freq "
                "      ORDER BY market_features.bar_datetime DESC"
                "    ) AS row_number, "
                "    GREATEST(0, selected.sequence_window - "
                "      CASE WHEN selected.snapshot_required THEN 1 ELSE 0 END"
                "    ) AS stable_limit "
                "  FROM selected JOIN market_features ON "
                "    market_features.symbol = selected.symbol "
                "    AND market_features.adjust = selected.adjust "
                "    AND market_features.freq = selected.freq "
                "    AND market_features.bar_datetime <= selected.stable_end_datetime"
                ") SELECT * EXCLUDE (row_number, stable_limit) FROM ranked "
                "WHERE row_number <= stable_limit ORDER BY decision_id, freq, bar_datetime",
                [requested_ids, selected],
            ).fetchdf()
        )
        snapshots = _load_snapshot_feature_rows_batched(
            conn,
            index=index,
            decisions=decisions,
            spec=spec,
            trade_freq=normalize_frequency(spec.trade_frequency),
            market_cache=market_cache,
        )

    context_names = tuple(compiled.get("decision_context", []))
    feature_names = {
        freq: tuple(compiled.get("channels_by_frequency", {}).get(freq, []))
        for freq in selected
    }
    context_lookup = {
        str(row.decision_id): row._asdict()
        for row in contexts.itertuples(index=False)
    }
    index_lookup = {
        (str(row.decision_id), str(row.freq)): row
        for row in index.itertuples(index=False)
    }
    stable_groups = {
        (str(decision_id), str(freq)): group.drop(columns=["decision_id"]).copy()
        for (decision_id, freq), group in stable.groupby(
            ["decision_id", "freq"], sort=False
        )
    } if not stable.empty else {}

    result: dict[str, ModelInputBatch] = {}
    for decision_id in requested_ids:
        sequences: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        valid_ratios: dict[str, float] = {}
        for freq in selected:
            index_row = index_lookup[(decision_id, freq)]
            stable_limit = max(
                0,
                int(index_row.sequence_window)
                - (1 if bool(index_row.snapshot_required) else 0),
            )
            cached_values = None
            if market_cache is not None:
                cached_values = market_cache.load_values(
                    symbol=str(index_row.symbol),
                    freq=freq,
                    adjust=str(index_row.adjust),
                    end=index_row.stable_end_datetime,
                    limit=stable_limit,
                )
                frame = pd.DataFrame()
            else:
                frame = stable_groups.get((decision_id, freq), pd.DataFrame()).copy()
            snapshot = snapshots.get((decision_id, freq))
            if market_cache is not None:
                snapshot_values = (
                    np.asarray(
                        [_numeric_context_value(snapshot.get(name, 0.0)) for name in feature_names[freq]],
                        dtype=np.float32,
                    ).reshape(1, -1)
                    if snapshot is not None
                    else np.empty((0, len(feature_names[freq])), dtype=np.float32)
                )
                values = (
                    np.concatenate([cached_values, snapshot_values], axis=0)
                    if len(snapshot_values)
                    else cached_values
                )
                window = int(index_row.sequence_window)
                valid_ratio = min(1.0, len(values) / float(max(1, window)))
                sequence = _pad_cached_market_sequence(
                    values,
                    feature_names=feature_names[freq],
                    window=window,
                    valid_ratio=valid_ratio,
                )
            else:
                if snapshot is not None:
                    snapshot_frame = pd.DataFrame([snapshot])
                    frame = (
                        snapshot_frame
                        if frame.empty
                        else pd.concat([frame, snapshot_frame], ignore_index=True)
                    )
                valid_ratio = min(
                    1.0,
                    len(frame) / float(max(1, int(index_row.sequence_window))),
                )
                frame["sequence_valid_ratio"] = valid_ratio
                sequence = pad_market_sequence(
                    frame,
                    feature_names=feature_names[freq],
                    window=int(index_row.sequence_window),
                )
            sequences[freq] = sequence.values
            masks[freq] = sequence.sequence_mask
            valid_ratios[freq] = sequence.valid_ratio

        context = context_lookup.get(decision_id, {})
        result[decision_id] = ModelInputBatch(
            market_sequences=sequences,
            sequence_masks=masks,
            valid_ratios=valid_ratios,
            decision_context=np.asarray(
                [_numeric_context_value(context.get(name, 0.0)) for name in context_names],
                dtype=np.float32,
            ),
            runtime_contract=tuple(compiled.get("runtime_state", [])),
            feature_names=feature_names,
            decision_context_names=context_names,
            schema_hash=str(compiled.get("schema_hash") or ""),
        )
    return result


def _pad_cached_market_sequence(
    values: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    window: int,
    valid_ratio: float,
) -> PaddedMarketSequence:
    width = len(feature_names)
    selected = np.asarray(values[-window:], dtype=np.float32)
    output = np.zeros((window, width), dtype=np.float32)
    mask = np.zeros(window, dtype=np.float32)
    if len(selected):
        output[-len(selected) :] = selected
        mask[-len(selected) :] = 1.0
    if len(selected) and "sequence_valid_ratio" in feature_names:
        column = feature_names.index("sequence_valid_ratio")
        output[-len(selected) :, column] = float(valid_ratio)
    return PaddedMarketSequence(
        values=output,
        sequence_mask=mask,
        valid_rows=int(mask.sum()),
        valid_ratio=float(valid_ratio),
    )


def _load_snapshot_feature_rows_batched(
    conn: duckdb.DuckDBPyConnection,
    *,
    index: pd.DataFrame,
    decisions: pd.DataFrame,
    spec: FeatureSpec,
    trade_freq: str,
    market_cache: CompactMarketFeatureCache | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    required = index.loc[index["snapshot_required"].fillna(False).astype(bool)].copy()
    if required.empty:
        return {}

    decisions = decisions.copy()
    decisions["decision_time"] = pd.to_datetime(decisions["decision_time"], errors="coerce")
    decisions["visible_bar_end"] = pd.to_datetime(decisions["visible_bar_end"], errors="coerce")
    decision_lookup = {
        str(row.decision_id): row for row in decisions.itertuples(index=False)
    }
    symbols = list(dict.fromkeys(required["symbol"].astype(str)))
    minimum_day = decisions["decision_time"].min().normalize()
    maximum_day = decisions["decision_time"].max().normalize()
    maximum_cutoff = decisions["visible_bar_end"].fillna(decisions["decision_time"]).max()
    base = conn.execute(
        "SELECT * EXCLUDE (market_row_id) FROM market_bars "
        "WHERE symbol IN (SELECT UNNEST(?)) AND freq = ? "
        "AND bar_datetime >= ? AND bar_datetime <= ? ORDER BY symbol, bar_datetime",
        [symbols, trade_freq, minimum_day, maximum_cutoff],
    ).fetchdf().rename(columns={"bar_datetime": "datetime"})
    week_start = minimum_day - pd.Timedelta(days=minimum_day.weekday())
    daily = conn.execute(
        "SELECT * EXCLUDE (market_row_id) FROM market_bars "
        "WHERE symbol IN (SELECT UNNEST(?)) AND freq = 'daily' "
        "AND bar_datetime >= ? AND bar_datetime < ? ORDER BY symbol, bar_datetime",
        [symbols, week_start, maximum_day + pd.Timedelta(days=1)],
    ).fetchdf().rename(columns={"bar_datetime": "datetime"})
    if not base.empty:
        base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
    if not daily.empty:
        daily["datetime"] = pd.to_datetime(daily["datetime"], errors="coerce")
    base_lookup = _raw_day_lookup(base)
    daily_lookup = _raw_symbol_lookup(daily)

    raw_snapshots: dict[str, list[dict[str, Any]]] = {}
    for row in required.itertuples(index=False):
        decision = decision_lookup[str(row.decision_id)]
        cutoff = (
            pd.Timestamp(decision.visible_bar_end)
            if pd.notna(decision.visible_bar_end)
            else pd.Timestamp(decision.decision_time)
        )
        session_day = pd.Timestamp(decision.decision_time).normalize()
        if market_cache is not None:
            raw = _fast_snapshot_row_from_arrays(
                decision,
                freq=str(row.freq),
                cutoff=cutoff,
                day_values=base_lookup.get(
                    (str(row.symbol), str(row.adjust), session_day)
                ),
                daily_values=daily_lookup.get((str(row.symbol), str(row.adjust))),
            )
            if raw is None:
                continue
        else:
            visible_source = base_lookup.get(
                (str(row.symbol), str(row.adjust), session_day)
            )
            visible_day = _raw_values_frame(visible_source, cutoff=cutoff)
            daily_values = daily_lookup.get((str(row.symbol), str(row.adjust)))
            daily_symbol = _raw_values_frame(daily_values)
            snapshot = _decision_snapshot(
                decision,
                freq=str(row.freq),
                trade_freq=trade_freq,
                visible_day=visible_day,
                daily_symbol=daily_symbol,
                stable_end=(
                    pd.Timestamp(row.stable_end_datetime)
                    if pd.notna(row.stable_end_datetime)
                    else None
                ),
            )
            if snapshot is None or snapshot.empty:
                continue
            raw = _standardize_market_bars(
                snapshot.tail(1), freq=str(row.freq)
            ).iloc[0].to_dict()
        raw["decision_id"] = str(row.decision_id)
        raw_snapshots.setdefault(str(row.freq), []).append(raw)

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for freq, snapshot_rows in raw_snapshots.items():
        frequency_index = required.loc[required["freq"].astype(str).eq(freq)]
        ids = frequency_index["decision_id"].astype(str).tolist()
        active_indicators = [
            item for item in spec.indicators if item.enabled and freq in item.frequencies
        ]
        lookback = max([20, *(indicator_lookback(item) for item in active_indicators)])
        history_limit = max(
            int(frequency_index["sequence_window"].max()),
            lookback * 4,
        )
        if market_cache is not None:
            frequency_lookup = {
                str(row.decision_id): row
                for row in frequency_index.itertuples(index=False)
            }
            grouped_snapshots: dict[
                tuple[str, str], list[dict[str, Any]]
            ] = {}
            for snapshot_row in snapshot_rows:
                index_row = frequency_lookup[str(snapshot_row["decision_id"])]
                grouped_snapshots.setdefault(
                    (str(index_row.symbol), str(index_row.adjust)), []
                ).append(snapshot_row)
            for (symbol, adjust), symbol_snapshots in grouped_snapshots.items():
                engine = market_cache.snapshot_engine(
                    symbol=symbol,
                    freq=freq,
                    adjust=adjust,
                    spec=spec,
                )
                stable_ends = [
                    frequency_lookup[str(item["decision_id"])].stable_end_datetime
                    for item in symbol_snapshots
                ]
                feature_rows = engine.build(
                    stable_ends,
                    symbol_snapshots,
                    feature_names=spec.market_feature_names,
                )
                for snapshot_row, features in zip(symbol_snapshots, feature_rows):
                    decision_id = str(snapshot_row["decision_id"])
                    features["symbol"] = symbol
                    features["bar_datetime"] = snapshot_row.get("bar_datetime")
                    result[(decision_id, freq)] = features
            continue
        history = conn.execute(
            "WITH selected AS ("
            " SELECT decision_id, symbol, adjust, freq, stable_end_datetime "
            " FROM decision_index WHERE decision_id IN (SELECT UNNEST(?)) AND freq = ?"
            "), ranked AS ("
            " SELECT selected.decision_id, market_bars.* EXCLUDE (market_row_id), "
            " ROW_NUMBER() OVER (PARTITION BY selected.decision_id "
            " ORDER BY market_bars.bar_datetime DESC) AS row_number "
            " FROM selected JOIN market_bars ON market_bars.symbol = selected.symbol "
            " AND market_bars.adjust = selected.adjust AND market_bars.freq = selected.freq "
            " AND market_bars.bar_datetime <= selected.stable_end_datetime"
            ") SELECT * EXCLUDE (row_number) FROM ranked WHERE row_number <= ? "
            "ORDER BY decision_id, bar_datetime",
            [ids, freq, history_limit],
        ).fetchdf().rename(columns={"bar_datetime": "datetime"})
        parts: list[pd.DataFrame] = []
        if not history.empty:
            history["symbol"] = history["decision_id"].astype(str)
            parts.append(history.drop(columns=["decision_id"]))
        snapshot_frame = pd.DataFrame(snapshot_rows)
        snapshot_frame["symbol"] = snapshot_frame["decision_id"].astype(str)
        parts.append(snapshot_frame.drop(columns=["decision_id"]))
        features = build_bar_features(
            pd.concat(
                [part.dropna(axis=1, how="all") for part in parts],
                ignore_index=True,
            ),
            spec=spec,
        )
        for synthetic_id, group in features.groupby("symbol", sort=False):
            row = group.sort_values("datetime").iloc[-1].to_dict()
            row["symbol"] = str(decision_lookup[str(synthetic_id)].symbol)
            row["bar_datetime"] = row.pop("datetime")
            result[(str(synthetic_id), freq)] = row
    return result


def _fast_snapshot_row(
    decision: object,
    *,
    freq: str,
    visible_day: pd.DataFrame,
    daily_symbol: pd.DataFrame,
) -> dict[str, Any] | None:
    """Build one partial OHLCV row without invoking Pandas groupby machinery."""
    decision_time = pd.Timestamp(decision.decision_time)
    symbol = str(decision.symbol)
    adjust = str(decision.adjust)
    if freq in {"15min", "30min", "60min"}:
        if visible_day.empty:
            return None
        source_rows = {"15min": 3, "30min": 6, "60min": 12}[freq]
        partial_rows = len(visible_day) % source_rows
        if partial_rows == 0:
            return None
        current = visible_day.tail(partial_rows)
        previous_close = (
            float(visible_day.iloc[-partial_rows - 1]["close"])
            if len(visible_day) > partial_rows
            else np.nan
        )
        close = float(current.iloc[-1]["close"])
        open_ = float(current.iloc[0]["open"])
        pct_chg = (
            close / previous_close - 1.0
            if np.isfinite(previous_close) and previous_close != 0
            else close / open_ - 1.0
        )
        return _snapshot_row_values(
            current,
            symbol=symbol,
            adjust=adjust,
            freq=freq,
            bar_datetime=pd.Timestamp(current.iloc[-1]["datetime"]),
            progress=partial_rows / float(source_rows),
            pct_chg=pct_chg,
        )

    if visible_day.empty:
        if str(decision.stage) != "open_auction":
            return None
        price = float(decision.execution_price)
        partial = pd.DataFrame(
            [{
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
                "amount": 0.0,
                "datetime": decision_time.normalize(),
            }]
        )
        daily_progress = 0.0
    else:
        partial = visible_day
        daily_progress = min(1.0, len(visible_day) / 48.0)

    daily_row = _snapshot_row_values(
        partial,
        symbol=symbol,
        adjust=adjust,
        freq="daily",
        bar_datetime=decision_time.normalize(),
        progress=daily_progress,
        pct_chg=None,
    )
    daily_row["pctChg"] = _safe_snapshot_return(daily_row["close"], daily_row["open"])
    if freq == "daily":
        return daily_row
    if freq != "weekly":
        return None

    week_start = decision_time.normalize() - pd.Timedelta(days=decision_time.weekday())
    prior = daily_symbol.loc[
        (daily_symbol["datetime"] >= week_start)
        & (daily_symbol["datetime"].dt.date < decision_time.date())
    ]
    current = pd.DataFrame([daily_row])
    weekly_source = pd.concat([prior, current], ignore_index=True, sort=False)
    weekly = _snapshot_row_values(
        weekly_source,
        symbol=symbol,
        adjust=adjust,
        freq="weekly",
        bar_datetime=decision_time.normalize(),
        progress=min(1.0, len(weekly_source) / 5.0),
        pct_chg=None,
    )
    weekly["pctChg"] = _safe_snapshot_return(weekly["close"], weekly["open"])
    return weekly


def _fast_snapshot_row_from_arrays(
    decision: object,
    *,
    freq: str,
    cutoff: pd.Timestamp,
    day_values: dict[str, np.ndarray] | None,
    daily_values: dict[str, np.ndarray] | None,
) -> dict[str, Any] | None:
    decision_time = pd.Timestamp(decision.decision_time)
    symbol = str(decision.symbol)
    adjust = str(decision.adjust)
    visible_count = (
        int(np.searchsorted(day_values["datetime"], int(cutoff.value), side="right"))
        if day_values is not None
        else 0
    )
    if freq in {"15min", "30min", "60min"}:
        if not visible_count or day_values is None:
            return None
        expected = {"15min": 3, "30min": 6, "60min": 12}[freq]
        partial = visible_count % expected
        if partial == 0:
            return None
        start = visible_count - partial
        previous = day_values["close"][start - 1] if start else np.nan
        close = day_values["close"][visible_count - 1]
        open_ = day_values["open"][start]
        pct_chg = (
            close / previous - 1.0
            if np.isfinite(previous) and previous != 0
            else close / open_ - 1.0
        )
        return _snapshot_values_from_arrays(
            day_values,
            start=start,
            stop=visible_count,
            symbol=symbol,
            adjust=adjust,
            freq=freq,
            bar_datetime=pd.Timestamp(day_values["datetime"][visible_count - 1]),
            progress=partial / float(expected),
            pct_chg=pct_chg,
        )

    if visible_count and day_values is not None:
        daily_row = _snapshot_values_from_arrays(
            day_values,
            start=0,
            stop=visible_count,
            symbol=symbol,
            adjust=adjust,
            freq="daily",
            bar_datetime=decision_time.normalize(),
            progress=min(1.0, visible_count / 48.0),
            pct_chg=None,
        )
    elif str(decision.stage) == "open_auction":
        price = float(decision.execution_price)
        daily_row = {
            "symbol": symbol,
            "bar_datetime": decision_time.normalize(),
            "freq": "daily",
            "adjust": adjust,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0.0,
            "amount": 0.0,
            "pctChg": 0.0,
            "turn": np.nan,
            "progress": 0.0,
        }
    else:
        return None
    daily_row["pctChg"] = _safe_snapshot_return(daily_row["close"], daily_row["open"])
    if freq == "daily":
        return daily_row
    if freq != "weekly":
        return None

    week_start_ns = int(
        (decision_time.normalize() - pd.Timedelta(days=decision_time.weekday())).value
    )
    day_start_ns = int(decision_time.normalize().value)
    if daily_values is None:
        prior_start = prior_stop = 0
    else:
        prior_start = int(np.searchsorted(daily_values["datetime"], week_start_ns, side="left"))
        prior_stop = int(np.searchsorted(daily_values["datetime"], day_start_ns, side="left"))
    prior_count = max(0, prior_stop - prior_start)
    if prior_count:
        weekly_open = daily_values["open"][prior_start]
        weekly_high = max(float(np.nanmax(daily_values["high"][prior_start:prior_stop])), daily_row["high"])
        weekly_low = min(float(np.nanmin(daily_values["low"][prior_start:prior_stop])), daily_row["low"])
        weekly_volume = float(np.nansum(daily_values["volume"][prior_start:prior_stop])) + daily_row["volume"]
        weekly_amount = float(np.nansum(daily_values["amount"][prior_start:prior_stop])) + daily_row["amount"]
    else:
        weekly_open = daily_row["open"]
        weekly_high = daily_row["high"]
        weekly_low = daily_row["low"]
        weekly_volume = daily_row["volume"]
        weekly_amount = daily_row["amount"]
    return {
        "symbol": symbol,
        "bar_datetime": decision_time.normalize(),
        "freq": "weekly",
        "adjust": adjust,
        "open": float(weekly_open),
        "high": float(weekly_high),
        "low": float(weekly_low),
        "close": float(daily_row["close"]),
        "volume": float(weekly_volume),
        "amount": float(weekly_amount),
        "pctChg": _safe_snapshot_return(daily_row["close"], weekly_open),
        "turn": np.nan,
        "progress": min(1.0, (prior_count + 1) / 5.0),
    }


def _raw_day_lookup(frame: pd.DataFrame) -> dict[tuple[str, str, pd.Timestamp], dict[str, np.ndarray]]:
    if frame.empty:
        return {}
    source = frame.assign(session_day=frame["datetime"].dt.normalize())
    return {
        (str(symbol), str(adjust), pd.Timestamp(day).normalize()): _raw_group_arrays(group)
        for (symbol, adjust, day), group in source.groupby(
            ["symbol", "adjust", "session_day"], sort=False
        )
    }


def _raw_symbol_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    if frame.empty:
        return {}
    return {
        (str(symbol), str(adjust)): _raw_group_arrays(group)
        for (symbol, adjust), group in frame.groupby(["symbol", "adjust"], sort=False)
    }


def _raw_group_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ordered = frame.sort_values("datetime")
    result = {
        "datetime": pd.to_datetime(ordered["datetime"]).to_numpy(dtype="datetime64[ns]").view(np.int64),
        "symbol": np.asarray([str(ordered.iloc[0].get("symbol", ""))], dtype=object),
        "adjust": np.asarray([str(ordered.iloc[0].get("adjust", ""))], dtype=object),
    }
    for name in ("open", "high", "low", "close", "volume", "amount"):
        result[name] = pd.to_numeric(ordered[name], errors="coerce").to_numpy(dtype=np.float64)
    return result


def _raw_values_frame(
    values: dict[str, np.ndarray] | None,
    *,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if values is None:
        return pd.DataFrame()
    stop = len(values["datetime"])
    if cutoff is not None:
        stop = int(np.searchsorted(values["datetime"], int(cutoff.value), side="right"))
    data = {
        name: (
                pd.to_datetime(values[name][:stop])
                if name == "datetime"
                else values[name][:stop]
            )
        for name in ("datetime", "open", "high", "low", "close", "volume", "amount")
    }
    data["symbol"] = str(values["symbol"][0])
    data["adjust"] = str(values["adjust"][0])
    return pd.DataFrame(data)


def _snapshot_values_from_arrays(
    values: dict[str, np.ndarray],
    *,
    start: int,
    stop: int,
    symbol: str,
    adjust: str,
    freq: str,
    bar_datetime: pd.Timestamp,
    progress: float,
    pct_chg: float | None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "bar_datetime": bar_datetime,
        "freq": freq,
        "adjust": adjust,
        "open": float(values["open"][start]),
        "high": float(np.nanmax(values["high"][start:stop])),
        "low": float(np.nanmin(values["low"][start:stop])),
        "close": float(values["close"][stop - 1]),
        "volume": float(np.nansum(values["volume"][start:stop])),
        "amount": float(np.nansum(values["amount"][start:stop])),
        "pctChg": pct_chg,
        "turn": np.nan,
        "progress": float(progress),
    }


def _snapshot_row_values(
    rows: pd.DataFrame,
    *,
    symbol: str,
    adjust: str,
    freq: str,
    bar_datetime: pd.Timestamp,
    progress: float,
    pct_chg: float | None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "bar_datetime": bar_datetime,
        "freq": freq,
        "adjust": adjust,
        "open": float(rows.iloc[0]["open"]),
        "high": float(pd.to_numeric(rows["high"], errors="coerce").max()),
        "low": float(pd.to_numeric(rows["low"], errors="coerce").min()),
        "close": float(rows.iloc[-1]["close"]),
        "volume": float(pd.to_numeric(rows["volume"], errors="coerce").fillna(0.0).sum()),
        "amount": float(pd.to_numeric(rows["amount"], errors="coerce").fillna(0.0).sum()),
        "pctChg": pct_chg,
        "turn": np.nan,
        "progress": float(progress),
    }


def _safe_snapshot_return(close: object, open_: object) -> float:
    close_value = float(close)
    open_value = float(open_)
    return close_value / open_value - 1.0 if open_value else 0.0


def _numeric_context_value(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return 0.0 if pd.isna(number) else float(number)


def _build_lazy_decision_snapshot(
    conn: duckdb.DuckDBPyConnection,
    *,
    decision_id: str,
    freq: str,
    trade_freq: str,
) -> pd.DataFrame:
    decision_frame = conn.execute(
        "SELECT * FROM decisions WHERE decision_id = ? LIMIT 1",
        [decision_id],
    ).fetchdf()
    if decision_frame.empty:
        return pd.DataFrame()
    values = decision_frame.iloc[0].to_dict()
    decision = SimpleNamespace(**values)
    decision_time = pd.Timestamp(values["decision_time"])
    visible_bar_end = values.get("visible_bar_end")
    cutoff = pd.Timestamp(visible_bar_end) if pd.notna(visible_bar_end) else decision_time
    day_start = decision_time.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    base = conn.execute(
        "SELECT * EXCLUDE (market_row_id) FROM market_bars "
        "WHERE symbol = ? AND freq = ? AND adjust = ? "
        "AND bar_datetime >= ? AND bar_datetime < ? AND bar_datetime <= ? "
        "ORDER BY bar_datetime",
        [values["symbol"], trade_freq, values["adjust"], day_start, day_end, cutoff],
    ).fetchdf().rename(columns={"bar_datetime": "datetime"})

    if freq in {"15min", "30min", "60min"}:
        if base.empty:
            return pd.DataFrame()
        return aggregate_ohlcv_from_base(base, freq, base_freq=trade_freq).tail(1)

    partial_daily = _current_partial_daily(
        decision=decision,
        current_day_base=base,
        trade_freq=trade_freq,
    )
    if freq == "daily":
        return partial_daily

    week_start = day_start - pd.Timedelta(days=day_start.weekday())
    history = conn.execute(
        "SELECT * EXCLUDE (market_row_id) FROM market_bars "
        "WHERE symbol = ? AND freq = 'daily' AND adjust = ? "
        "AND bar_datetime >= ? AND bar_datetime < ? ORDER BY bar_datetime",
        [values["symbol"], values["adjust"], week_start, day_start],
    ).fetchdf().rename(columns={"bar_datetime": "datetime"})
    parts = [part for part in (history, partial_daily) if part is not None and not part.empty]
    if not parts:
        return pd.DataFrame()
    return aggregate_ohlcv_from_base(
        pd.concat([part.dropna(axis=1, how="all") for part in parts], ignore_index=True),
        "weekly",
        base_freq="daily",
    ).tail(1)


def _metadata_frame(
    spec: FeatureSpec,
    blueprint: dict[str, object],
    compiled_model_input: dict[str, object],
) -> pd.DataFrame:
    values = {
        "spec_name": spec.name,
        "spec_version": spec.version,
        "sequence_windows": json.dumps(spec.sequence_windows, ensure_ascii=False),
        "indicators": json.dumps(
            [indicator_to_payload(item) for item in spec.indicators], ensure_ascii=False
        ),
        "market_feature_names": json.dumps(list(spec.market_feature_names), ensure_ascii=False),
        "context_feature_names": json.dumps(list(spec.context_feature_names), ensure_ascii=False),
        "market_rule_version": "dated_st_v1",
        "st_policy": "5pct_limit_no_new_buys_legal_exits",
        "limit_reference": "previous_trading_day_close",
        "is_st_model_feature": "false",
        "model_input_blueprint": json.dumps(blueprint, ensure_ascii=False),
        "compiled_model_input": json.dumps(compiled_model_input, ensure_ascii=False),
        "model_input_schema_hash": str(compiled_model_input.get("schema_hash") or ""),
        "model_input_schema_version": str(compiled_model_input.get("schema_version") or 1),
    }
    return pd.DataFrame([{"key": key, "value": value} for key, value in values.items()])


def build_compact_metadata_frame(
    spec: FeatureSpec,
    blueprint: dict[str, object],
    compiled_model_input: dict[str, object],
) -> pd.DataFrame:
    """Build the canonical dataset_metadata table for a compact feature store."""

    return _metadata_frame(spec, blueprint, compiled_model_input)


def _spec_from_store(conn: duckdb.DuckDBPyConnection) -> FeatureSpec:
    rows = dict(conn.execute("SELECT key, value FROM dataset_metadata").fetchall())
    indicators = validate_indicator_specs(json.loads(rows["indicators"]))
    spec = feature_spec_from_indicators(indicators)
    windows = {str(key): int(value) for key, value in json.loads(rows["sequence_windows"]).items()}
    return replace(spec, sequence_windows=windows)


def _compiled_model_input_from_store(
    conn: duckdb.DuckDBPyConnection,
    spec: FeatureSpec,
) -> dict[str, object]:
    try:
        rows = dict(conn.execute("SELECT key, value FROM dataset_metadata").fetchall())
    except Exception:
        rows = {}
    serialized = rows.get("compiled_model_input")
    if serialized:
        return json.loads(serialized)
    frequencies = (spec.base_frequency, *spec.derived_frequencies)
    channels_by_frequency = {
        freq: list(spec.market_feature_names)
        for freq in frequencies
    }
    decision_context = [*spec.context_feature_names, *spec.constraint_feature_names]
    runtime_state = [*spec.portfolio_feature_names, *spec.environment_feature_names]
    return {
        "schema_version": 0,
        "schema_hash": "legacy",
        "channels_by_frequency": channels_by_frequency,
        "decision_context": decision_context,
        "runtime_state": runtime_state,
        "shapes": {
            freq: [int(spec.sequence_windows.get(freq, 0)), len(channels)]
            for freq, channels in channels_by_frequency.items()
        },
        "decision_context_shape": [len(decision_context)],
        "runtime_state_shape": [len(runtime_state)],
    }


def _load_stable_bars(
    db_path: str | Path,
    *,
    symbols: list[str],
    adjust: str,
    end: str | None,
    base: pd.DataFrame,
    daily: pd.DataFrame,
    frequencies: tuple[str, ...],
    trade_freq: str,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for freq in frequencies:
        if freq == trade_freq:
            result[freq] = base.copy()
        elif freq == "daily":
            result[freq] = daily.copy()
        else:
            stored = load_kline_from_duckdb(
                db_path,
                symbols=symbols,
                freq=freq,
                adjust=adjust,
                start=None,
                end=end,
            )
            stored = _prepare_kline_frame(stored, freq=freq, adjust=adjust)
            if not stored.empty:
                result[freq] = stored
            elif freq in {"15min", "30min", "60min"}:
                result[freq] = aggregate_ohlcv_from_base(base, freq, base_freq=trade_freq)
            elif freq == "weekly":
                result[freq] = aggregate_ohlcv_from_base(daily, "weekly", base_freq="daily")
            else:
                raise ValueError(f"Unsupported compact frequency: {freq}")
    return result


def _build_decision_index_fast(
    decisions: pd.DataFrame,
    *,
    stable_by_freq: dict[str, pd.DataFrame],
    frequencies: tuple[str, ...],
    sequence_windows: dict[str, int],
    trade_freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "decision_id", "symbol", "adjust", "decision_time", "freq",
        "sequence_window", "stable_end_datetime", "snapshot_id",
        "snapshot_required", "valid_rows", "sequence_valid_ratio",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=["snapshot_id", "decision_id", *_market_bar_columns(bar_datetime=True)])

    source = decisions.copy()
    source["decision_time"] = pd.to_datetime(source["decision_time"], errors="coerce")
    source["visible_bar_end"] = pd.to_datetime(source.get("visible_bar_end"), errors="coerce")
    source["_cutoff"] = source["visible_bar_end"].fillna(source["decision_time"])
    parts: list[pd.DataFrame] = []
    for freq in frequencies:
        frame = source[["decision_id", "symbol", "adjust", "decision_time", "stage", "_cutoff"]].copy()
        frame["freq"] = freq
        frame["sequence_window"] = max(1, int(sequence_windows.get(freq, 1)))
        frame["stable_end_datetime"] = pd.NaT
        frame["_stable_count"] = 0
        for symbol, positions in frame.groupby("symbol", sort=False).groups.items():
            stable_times = _stable_times(stable_by_freq.get(freq, pd.DataFrame()), str(symbol))
            if freq == "daily":
                cutoffs = frame.loc[positions, "decision_time"].dt.normalize() - pd.Timedelta(microseconds=1)
            elif freq == "weekly":
                decision_times = frame.loc[positions, "decision_time"]
                cutoffs = decision_times.dt.normalize() - pd.to_timedelta(decision_times.dt.weekday, unit="D") - pd.Timedelta(microseconds=1)
            else:
                cutoffs = frame.loc[positions, "_cutoff"]
            cutoff_values = cutoffs.to_numpy(dtype="datetime64[ns]")
            counts = np.searchsorted(stable_times, cutoff_values, side="right")
            stable_ends = np.full(len(counts), np.datetime64("NaT"), dtype="datetime64[ns]")
            has_stable = counts > 0
            stable_ends[has_stable] = stable_times[counts[has_stable] - 1]
            frame.loc[positions, "_stable_count"] = counts
            frame.loc[positions, "stable_end_datetime"] = pd.to_datetime(stable_ends)

        is_open = frame["stage"].astype(str).str.endswith("open_auction")
        if freq == trade_freq:
            snapshot_required = pd.Series(False, index=frame.index)
        elif freq in {"15min", "30min", "60min"}:
            stable_end = pd.to_datetime(frame["stable_end_datetime"], errors="coerce")
            snapshot_required = (~is_open) & (stable_end.isna() | stable_end.lt(frame["_cutoff"]))
        else:
            snapshot_required = pd.Series(True, index=frame.index)
        frame["snapshot_required"] = snapshot_required.astype(bool)
        frame["snapshot_id"] = [
            f"{decision_id}|{freq}" if required else None
            for decision_id, required in zip(frame["decision_id"], frame["snapshot_required"])
        ]
        frame["valid_rows"] = np.minimum(
            frame["sequence_window"].to_numpy(dtype=int),
            frame["_stable_count"].to_numpy(dtype=int) + frame["snapshot_required"].to_numpy(dtype=int),
        )
        frame["sequence_valid_ratio"] = frame["valid_rows"] / frame["sequence_window"].astype(float)
        parts.append(frame[columns])

    return (
        pd.concat(parts, ignore_index=True),
        pd.DataFrame(columns=["snapshot_id", "decision_id", *_market_bar_columns(bar_datetime=True)]),
    )


def _standardize_market_bars(frame: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if "datetime" not in result and "bar_end" in result:
        result["datetime"] = result["bar_end"]
    result["freq"] = freq
    for column in ["turn", "progress"]:
        if column not in result:
            result[column] = np.nan if column == "turn" else 1.0
    columns = [
        "symbol", "datetime", "freq", "adjust", "open", "high", "low", "close",
        "volume", "amount", "pctChg", "turn", "progress",
    ]
    for column in columns:
        if column not in result:
            result[column] = None
    return result[columns].sort_values(["symbol", "datetime"]).reset_index(drop=True)


def _market_row_ids(frame: pd.DataFrame) -> list[str]:
    return [
        f"{row.symbol}|{row.freq}|{row.adjust}|{pd.Timestamp(row.bar_datetime).strftime('%Y%m%d%H%M%S')}"
        for row in frame.itertuples(index=False)
    ]


def _build_decision_index(
    decisions: pd.DataFrame,
    *,
    base: pd.DataFrame,
    daily: pd.DataFrame,
    stable_by_freq: dict[str, pd.DataFrame],
    frequencies: tuple[str, ...],
    sequence_windows: dict[str, int],
    trade_freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []
    stable_lookup = {
        (freq, symbol): _stable_times(frame, symbol)
        for freq, frame in stable_by_freq.items()
        for symbol in decisions["symbol"].dropna().astype(str).unique()
    }
    base_days = {
        (symbol, day): group.copy()
        for (symbol, day), group in base.groupby(
            ["symbol", pd.to_datetime(base["datetime"]).dt.date], sort=False
        )
    } if not base.empty else {}
    daily_symbols = {
        symbol: group.copy()
        for symbol, group in daily.groupby("symbol", sort=False)
    } if not daily.empty else {}

    for decision in decisions.itertuples(index=False):
        symbol = str(decision.symbol)
        adjust = str(decision.adjust)
        decision_time = pd.Timestamp(decision.decision_time)
        cutoff = pd.Timestamp(decision.visible_bar_end) if pd.notna(decision.visible_bar_end) else decision_time
        session_date = decision_time.date()
        day_base = base_days.get((symbol, session_date), pd.DataFrame())
        visible_day = day_base.loc[pd.to_datetime(day_base.get("datetime"), errors="coerce") <= cutoff].copy() if not day_base.empty else pd.DataFrame()

        for freq in frequencies:
            stable_times = stable_lookup.get((freq, symbol), np.array([], dtype="datetime64[ns]"))
            if freq == "daily":
                stable_cutoff = decision_time.normalize() - pd.Timedelta(microseconds=1)
            elif freq == "weekly":
                stable_cutoff = decision_time.normalize() - pd.Timedelta(days=decision_time.weekday(), microseconds=1)
            else:
                stable_cutoff = cutoff
            count = int(np.searchsorted(stable_times, np.datetime64(stable_cutoff), side="right"))
            stable_end = pd.Timestamp(stable_times[count - 1]) if count else None
            snapshot = _decision_snapshot(
                decision,
                freq=freq,
                trade_freq=trade_freq,
                visible_day=visible_day,
                daily_symbol=daily_symbols.get(symbol, pd.DataFrame()),
                stable_end=stable_end,
            )
            snapshot_id = None
            if snapshot is not None and not snapshot.empty:
                snapshot_id = f"{decision.decision_id}|{freq}"
                row = _standardize_market_bars(snapshot.tail(1), freq=freq).iloc[0].to_dict()
                row["snapshot_id"] = snapshot_id
                row["decision_id"] = decision.decision_id
                row["bar_datetime"] = row.pop("datetime")
                snapshot_rows.append(row)
            window = max(1, int(sequence_windows.get(freq, 1)))
            valid_rows = min(window, count + (1 if snapshot_id else 0))
            index_rows.append(
                {
                    "decision_id": decision.decision_id,
                    "symbol": symbol,
                    "adjust": adjust,
                    "decision_time": decision_time,
                    "freq": freq,
                    "sequence_window": window,
                    "stable_end_datetime": stable_end,
                    "snapshot_id": snapshot_id,
                    "valid_rows": valid_rows,
                    "sequence_valid_ratio": valid_rows / float(window),
                }
            )
    index_columns = [
        "decision_id", "symbol", "adjust", "decision_time", "freq",
        "sequence_window", "stable_end_datetime", "snapshot_id", "valid_rows",
        "sequence_valid_ratio",
    ]
    snapshot_columns = ["snapshot_id", "decision_id", *_market_bar_columns(bar_datetime=True)]
    return (
        pd.DataFrame(index_rows, columns=index_columns),
        pd.DataFrame(snapshot_rows, columns=snapshot_columns),
    )


def _stable_times(frame: pd.DataFrame, symbol: str) -> np.ndarray:
    if frame is None or frame.empty:
        return np.array([], dtype="datetime64[ns]")
    column = "datetime" if "datetime" in frame else "bar_end"
    values = pd.to_datetime(
        frame.loc[frame["symbol"].astype(str).eq(symbol), column], errors="coerce"
    ).dropna().sort_values()
    return values.to_numpy(dtype="datetime64[ns]")


def _market_bar_columns(*, bar_datetime: bool) -> list[str]:
    return [
        "symbol",
        "bar_datetime" if bar_datetime else "datetime",
        "freq",
        "adjust",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pctChg",
        "turn",
        "progress",
    ]


def _decision_snapshot(
    decision: object,
    *,
    freq: str,
    trade_freq: str,
    visible_day: pd.DataFrame,
    daily_symbol: pd.DataFrame,
    stable_end: pd.Timestamp | None,
) -> pd.DataFrame | None:
    if freq == trade_freq:
        return None
    if freq in {"15min", "30min", "60min"}:
        if visible_day.empty:
            return None
        current = aggregate_ohlcv_from_base(visible_day, freq, base_freq=trade_freq).tail(1)
        if current.empty:
            return None
        current_end = pd.Timestamp(current.iloc[0]["bar_end"])
        return current if stable_end is None or current_end > stable_end else None

    partial_daily = _current_partial_daily(
        decision=decision,
        current_day_base=visible_day,
        trade_freq=trade_freq,
    )
    if freq == "daily":
        return partial_daily

    decision_time = pd.Timestamp(decision.decision_time)
    week_start = decision_time.normalize() - pd.Timedelta(days=decision_time.weekday())
    history = daily_symbol.copy()
    history["datetime"] = pd.to_datetime(history["datetime"], errors="coerce")
    history = history.loc[
        (history["datetime"] >= week_start)
        & (history["datetime"].dt.date < decision_time.date())
    ].copy()
    parts = [part for part in (history, partial_daily) if part is not None and not part.empty]
    if not parts:
        return None
    return aggregate_ohlcv_from_base(
        pd.concat([part.dropna(axis=1, how="all") for part in parts], ignore_index=True),
        "weekly",
        base_freq="daily",
    ).tail(1)
