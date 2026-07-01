from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
import shutil

import numpy as np
import pandas as pd

from agent_layer.data.cache_schema import (
    AGENT_CACHE_VERSION,
    CACHE_MANIFEST_NAME,
    CACHE_STORAGE_INDEX_BASED,
    CONSTRAINT_COLUMNS,
    EXECUTION_COLUMNS,
    AgentCacheManifest,
    SymbolCacheMetadata,
    normalize_symbol_list,
    safe_symbol_name,
    symbol_cache_dir,
    write_json,
)
from agent_layer.data.episode_index import build_episode_index_frame, write_episode_index
from agent_layer.data.feature_parts import FeaturePartsDataset
from agent_layer.data.tensor_store import describe_symbol_cache, write_symbol_cache


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class AgentCacheBuildConfig:
    feature_dir: Path
    output_dir: Path
    symbols: tuple[str, ...] | None = None
    frequencies: tuple[str, ...] | None = None
    start: str | None = None
    end: str | None = None
    stages: tuple[str, ...] | None = None
    workers: int = 1
    chunk_size: int = 256
    reset: bool = False
    max_decisions_per_symbol: int | None = None


def build_agent_cache(
    config: AgentCacheBuildConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    feature_dir = Path(config.feature_dir)
    output_dir = Path(config.output_dir)
    if config.reset and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = FeaturePartsDataset(feature_dir)
    available_symbols = tuple(source.symbols)
    selected_symbols = normalize_symbol_list(config.symbols) if config.symbols else available_symbols
    unknown = sorted(set(selected_symbols).difference(available_symbols))
    if unknown:
        raise ValueError(
            "Agent cache symbols are absent from the Feature Parts Dataset: "
            + ", ".join(unknown[:20])
        )
    if not selected_symbols:
        raise ValueError("No symbols were selected for Agent cache build.")
    compiled = source.compiled_model_input()
    available_frequencies = tuple(source.frequencies())
    selected_frequencies = (
        normalize_symbol_list(config.frequencies)
        if config.frequencies
        else tuple(
            str(freq)
            for freq in compiled.get("channels_by_frequency", {})
            if str(freq) in available_frequencies
        )
    )
    missing_freqs = sorted(set(selected_frequencies).difference(available_frequencies))
    if missing_freqs:
        raise ValueError(
            "Agent cache frequencies are absent from the Feature Parts Dataset: "
            + ", ".join(missing_freqs)
        )
    worker_count = max(1, int(config.workers))
    jobs = [
        _SymbolCacheJob(
            feature_dir=str(feature_dir),
            output_dir=str(output_dir),
            symbol=symbol,
            frequencies=tuple(selected_frequencies),
            start=config.start,
            end=config.end,
            stages=config.stages,
            chunk_size=max(1, int(config.chunk_size)),
            max_decisions=config.max_decisions_per_symbol,
        )
        for symbol in selected_symbols
    ]
    summaries: list[dict[str, Any]] = []
    if progress_callback:
        progress_callback(
            {
                "phase": "starting",
                "symbols": len(jobs),
                "workers": worker_count,
                "output_dir": str(output_dir),
                "storage": CACHE_STORAGE_INDEX_BASED,
            }
        )
    _raise_if_cancelled(cancel_check)
    if worker_count == 1:
        for index, job in enumerate(jobs, start=1):
            _raise_if_cancelled(cancel_check)
            summary = _build_symbol_cache_worker(job)
            summaries.append(summary)
            if progress_callback:
                progress_callback({"phase": "symbol_done", "index": index, "summary": summary})
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_build_symbol_cache_worker, job): job for job in jobs}
            try:
                for index, future in enumerate(as_completed(futures), start=1):
                    _raise_if_cancelled(cancel_check)
                    job = futures[future]
                    try:
                        summary = future.result()
                    except Exception as exc:
                        summary = {
                            "symbol": job.symbol,
                            "safe_symbol": safe_symbol_name(job.symbol),
                            "decision_count": 0,
                            "error": str(exc),
                        }
                    summaries.append(summary)
                    if progress_callback:
                        progress_callback({"phase": "symbol_done", "index": index, "summary": summary})
            except InterruptedError:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    failures = [item for item in summaries if item.get("error")]
    success = [item for item in summaries if not item.get("error") and int(item.get("decision_count") or 0) > 0]
    episode_index = build_episode_index_frame(success)
    write_episode_index(output_dir, episode_index)
    manifest = AgentCacheManifest(
        cache_version=AGENT_CACHE_VERSION,
        source_feature_dir=str(feature_dir),
        schema_hash=str(compiled.get("schema_hash") or ""),
        frequencies=tuple(selected_frequencies),
        symbols=tuple(str(item["symbol"]) for item in success),
        symbol_count=len(success),
        decision_count=int(sum(int(item.get("decision_count") or 0) for item in success)),
        created_at=datetime.now(timezone.utc).isoformat(),
        storage=CACHE_STORAGE_INDEX_BASED,
    )
    write_json(output_dir / CACHE_MANIFEST_NAME, manifest.to_json_dict())
    return {
        "ok": not failures and bool(success),
        "output_dir": str(output_dir),
        "symbols": len(success),
        "failed_symbols": len(failures),
        "decision_count": manifest.decision_count,
        "frequencies": list(selected_frequencies),
        "schema_hash": manifest.schema_hash,
        "storage": CACHE_STORAGE_INDEX_BASED,
        "failures": failures,
        "episode_index_rows": int(len(episode_index)),
    }


def inspect_agent_cache(cache_dir: str | Path, *, sample_symbols: int = 3) -> dict[str, Any]:
    root = Path(cache_dir)
    manifest_path = root / CACHE_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Agent cache manifest not found: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols = [str(value) for value in manifest.get("symbols", [])]
    samples = []
    for symbol in symbols[: max(0, int(sample_symbols))]:
        samples.append(describe_symbol_cache(symbol_cache_dir(root, symbol)))
    episode_index_path = root / "episode_index.parquet"
    episode_rows = int(len(pd.read_parquet(episode_index_path))) if episode_index_path.exists() else 0
    return {
        "manifest": manifest,
        "episode_index_rows": episode_rows,
        "samples": samples,
    }


@dataclass(frozen=True)
class _SymbolCacheJob:
    feature_dir: str
    output_dir: str
    symbol: str
    frequencies: tuple[str, ...]
    start: str | None
    end: str | None
    stages: tuple[str, ...] | None
    chunk_size: int
    max_decisions: int | None


def _build_symbol_cache_worker(job: _SymbolCacheJob) -> dict[str, Any]:
    source = FeaturePartsDataset(job.feature_dir)
    rows = _load_symbol_decision_rows(source, job)
    if rows.empty:
        return {
            "symbol": job.symbol,
            "safe_symbol": safe_symbol_name(job.symbol),
            "decision_count": 0,
            "first_decision_time": None,
            "last_decision_time": None,
        }
    decision_ids = rows["decision_id"].astype(str).tolist()
    compiled = source.compiled_model_input()
    feature_names = {
        freq: tuple(str(value) for value in compiled.get("channels_by_frequency", {}).get(freq, []))
        for freq in job.frequencies
    }
    sequence_shapes = {
        freq: tuple(int(value) for value in compiled.get("shapes", {}).get(freq, []))
        for freq in job.frequencies
    }
    context_names = tuple(str(value) for value in compiled.get("decision_context", []))
    runtime_contract = tuple(str(value) for value in compiled.get("runtime_state", []))
    count = len(decision_ids)
    decision_context = _load_decision_context(source, decision_ids, context_names)
    execution = _rows_to_float_array(rows, EXECUTION_COLUMNS)
    constraints = _rows_to_bool_array(rows, CONSTRAINT_COLUMNS)
    index_rows = _load_decision_index_rows(source, decision_ids, job.frequencies)
    feature_matrices: dict[str, np.ndarray] = {}
    feature_times: dict[str, np.ndarray] = {}
    end_indices: dict[str, np.ndarray] = {}
    valid_rows: dict[str, np.ndarray] = {}
    snapshot_required: dict[str, np.ndarray] = {}
    snapshot_features: dict[str, np.ndarray] = {}
    valid_ratios: dict[str, np.ndarray] = {}
    for freq in job.frequencies:
        names = feature_names[freq]
        features_frame = _load_market_feature_frame(source, job.symbol, freq, names)
        feature_matrices[freq] = _feature_matrix(features_frame, names)
        feature_times[freq] = pd.to_datetime(
            features_frame.get("bar_datetime", pd.Series(dtype="datetime64[ns]")),
            errors="coerce",
        ).to_numpy(dtype="datetime64[us]")
        freq_index = index_rows.loc[index_rows["freq"].astype(str) == freq].copy()
        if freq_index.empty:
            raise KeyError(f"Decision index has no rows for {job.symbol} / {freq}")
        freq_index = freq_index.set_index("decision_id", drop=False).reindex(decision_ids)
        missing = freq_index["decision_id"].isna()
        if bool(missing.any()):
            raise KeyError(f"Decision index is incomplete for {job.symbol} / {freq}.")
        freq_index = freq_index.reset_index(drop=True)
        end_indices[freq] = _compute_end_indices(features_frame, freq_index)
        valid_rows[freq] = pd.to_numeric(freq_index.get("valid_rows", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int16)
        snapshot_required[freq] = freq_index.get("snapshot_required", False).fillna(False).astype(bool).to_numpy(dtype=np.uint8)
        valid_ratios[freq] = pd.to_numeric(freq_index.get("sequence_valid_ratio", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        channels = int(sequence_shapes[freq][1]) if len(sequence_shapes[freq]) >= 2 else len(names)
        snapshot_features[freq] = np.zeros((count, channels), dtype=np.float32)
    _fill_snapshot_features(
        source,
        decision_ids=decision_ids,
        frequencies=job.frequencies,
        snapshot_required=snapshot_required,
        snapshot_features=snapshot_features,
        chunk_size=job.chunk_size,
    )
    metadata = SymbolCacheMetadata(
        symbol=job.symbol,
        safe_symbol=safe_symbol_name(job.symbol),
        decision_count=count,
        first_decision_time=_timestamp_to_string(rows["decision_time"].iloc[0]),
        last_decision_time=_timestamp_to_string(rows["decision_time"].iloc[-1]),
        frequencies=job.frequencies,
        feature_names=feature_names,
        sequence_shapes=sequence_shapes,
        decision_context_names=context_names,
        runtime_contract=runtime_contract,
        schema_hash=str(compiled.get("schema_hash") or ""),
        storage=CACHE_STORAGE_INDEX_BASED,
    )
    out_dir = symbol_cache_dir(job.output_dir, job.symbol)
    write_symbol_cache(
        out_dir,
        metadata=metadata,
        decisions=rows.reset_index(drop=True),
        decision_context=decision_context,
        execution=execution,
        constraints=constraints,
        feature_matrices=feature_matrices,
        feature_times=feature_times,
        end_indices=end_indices,
        valid_rows=valid_rows,
        snapshot_required=snapshot_required,
        snapshot_features=snapshot_features,
        valid_ratios=valid_ratios,
    )
    return {
        "symbol": job.symbol,
        "safe_symbol": metadata.safe_symbol,
        "decision_count": count,
        "first_decision_time": metadata.first_decision_time,
        "last_decision_time": metadata.last_decision_time,
        "output_dir": str(out_dir),
        "storage": CACHE_STORAGE_INDEX_BASED,
    }


def _load_symbol_decision_rows(source: FeaturePartsDataset, job: _SymbolCacheJob) -> pd.DataFrame:
    start, end = _decision_time_bounds(job.start, job.end)
    rows = source.query_decision_rows(
        universe=(job.symbol,),
        start=start,
        end=end,
        stage=None,
    )
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["decision_time"] = pd.to_datetime(rows["decision_time"], errors="coerce")
    if job.stages:
        selected_stages = set(job.stages)
        rows = rows.loc[rows["stage"].astype(str).isin(selected_stages)].copy()
    rows = rows.sort_values(["decision_time", "stage", "symbol"]).reset_index(drop=True)
    if job.max_decisions is not None and int(job.max_decisions) > 0:
        rows = rows.head(int(job.max_decisions)).copy()
    return rows


def _load_decision_context(source: FeaturePartsDataset, decision_ids: list[str], context_names: tuple[str, ...]) -> np.ndarray:
    if not decision_ids:
        return np.zeros((0, len(context_names)), dtype=np.float32)
    symbols = source._symbols_for_decision_ids(decision_ids)  # noqa: SLF001 - single-purpose cache builder fast path.
    with source.connect(symbols=symbols) as conn:
        frame = conn.execute(
            "SELECT * FROM decision_context WHERE decision_id IN (SELECT UNNEST(?))",
            [decision_ids],
        ).fetchdf()
    lookup = {str(row.get("decision_id")): row for row in frame.to_dict("records")} if not frame.empty else {}
    values = np.zeros((len(decision_ids), len(context_names)), dtype=np.float32)
    for row_index, decision_id in enumerate(decision_ids):
        row = lookup.get(decision_id, {})
        for col_index, name in enumerate(context_names):
            values[row_index, col_index] = _numeric_value(row.get(name, 0.0))
    return values


def _load_decision_index_rows(source: FeaturePartsDataset, decision_ids: list[str], frequencies: tuple[str, ...]) -> pd.DataFrame:
    symbols = source._symbols_for_decision_ids(decision_ids)  # noqa: SLF001
    with source.connect(symbols=symbols) as conn:
        return conn.execute(
            "SELECT decision_id, symbol, adjust, decision_time, freq, sequence_window, "
            "stable_end_datetime, snapshot_id, snapshot_required, valid_rows, "
            "sequence_valid_ratio FROM decision_index "
            "WHERE decision_id IN (SELECT UNNEST(?)) "
            "AND freq IN (SELECT UNNEST(?))",
            [decision_ids, list(frequencies)],
        ).fetchdf()


def _load_market_feature_frame(source: FeaturePartsDataset, symbol: str, freq: str, feature_names: tuple[str, ...]) -> pd.DataFrame:
    columns = list(dict.fromkeys(["adjust", "bar_datetime", *feature_names]))
    select_sql = ", ".join(_quote_ident(column) for column in columns)
    with source.connect(symbols=(symbol,)) as conn:
        frame = conn.execute(
            f"SELECT {select_sql} FROM market_features "
            "WHERE symbol = ? AND freq = ? ORDER BY adjust, bar_datetime",
            [symbol, freq],
        ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["bar_datetime"] = pd.to_datetime(frame["bar_datetime"], errors="coerce")
    for column in feature_names:
        if column not in frame:
            frame[column] = 0.0
    return frame[columns].reset_index(drop=True)


def _feature_matrix(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    if frame.empty:
        return np.zeros((0, len(feature_names)), dtype=np.float32)
    return (
        frame[list(feature_names)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _compute_end_indices(features_frame: pd.DataFrame, freq_index: pd.DataFrame) -> np.ndarray:
    result = np.full(len(freq_index), -1, dtype=np.int64)
    if features_frame.empty:
        return result
    feature_adjust = features_frame["adjust"].astype(str).to_numpy()
    feature_times = pd.to_datetime(features_frame["bar_datetime"], errors="coerce")
    for adjust, positions in freq_index.groupby(freq_index["adjust"].astype(str), sort=False).groups.items():
        feature_positions = np.flatnonzero(feature_adjust == str(adjust))
        if len(feature_positions) == 0:
            continue
        times = feature_times.iloc[feature_positions].to_numpy(dtype="datetime64[ns]")
        stable_series = pd.to_datetime(freq_index.loc[positions, "stable_end_datetime"], errors="coerce")
        target_positions = np.asarray(list(positions), dtype=np.int64)
        valid_mask = stable_series.notna().to_numpy(dtype=bool)
        if not bool(valid_mask.any()):
            continue
        stable_end = stable_series.loc[valid_mask].to_numpy(dtype="datetime64[ns]")
        counts = np.searchsorted(times, stable_end, side="right")
        has = counts > 0
        result[target_positions[valid_mask][has]] = feature_positions[counts[has] - 1]
    return result


def _fill_snapshot_features(
    source: FeaturePartsDataset,
    *,
    decision_ids: list[str],
    frequencies: tuple[str, ...],
    snapshot_required: dict[str, np.ndarray],
    snapshot_features: dict[str, np.ndarray],
    chunk_size: int,
) -> None:
    if not decision_ids:
        return
    for offset in range(0, len(decision_ids), max(1, int(chunk_size))):
        chunk_ids = decision_ids[offset : offset + max(1, int(chunk_size))]
        requested_freqs = tuple(
            freq
            for freq in frequencies
            if bool(snapshot_required[freq][offset : offset + len(chunk_ids)].any())
        )
        if not requested_freqs:
            continue
        batches = source.load_model_input_batches(decision_ids=chunk_ids, frequencies=requested_freqs)
        for local_index, decision_id in enumerate(chunk_ids):
            target_index = offset + local_index
            batch = batches.get(decision_id)
            if batch is None:
                continue
            for freq in requested_freqs:
                if not bool(snapshot_required[freq][target_index]):
                    continue
                sequence = batch.market_sequences.get(freq)
                if sequence is not None and len(sequence):
                    snapshot_features[freq][target_index] = np.asarray(sequence[-1], dtype=np.float32)


def _decision_time_bounds(start_value: str | None, end_value: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _coerce_timestamp_bound(start_value, default=pd.Timestamp("1900-01-01"), is_end=False)
    end = _coerce_timestamp_bound(end_value, default=_safe_pandas_max_timestamp(), is_end=True)
    if start > end:
        raise ValueError(f"Agent cache start must be <= end: {start} > {end}")
    return start, end


def _coerce_timestamp_bound(value: str | None, *, default: pd.Timestamp, is_end: bool) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        return default
    raw = str(value).strip()
    timestamp = pd.Timestamp(raw)
    safe_max = _safe_pandas_max_timestamp()
    if timestamp > safe_max:
        timestamp = safe_max
    safe_min = pd.Timestamp.min.ceil("us")
    if timestamp < safe_min:
        timestamp = safe_min
    if is_end and _looks_like_date_only(raw):
        timestamp = _end_of_day(timestamp)
    return timestamp


def _looks_like_date_only(value: str) -> bool:
    text = value.strip()
    return len(text) <= 10 and "T" not in text and ":" not in text


def _end_of_day(value: pd.Timestamp) -> pd.Timestamp:
    safe_max = _safe_pandas_max_timestamp()
    day_start = value.normalize()
    next_day_start = day_start + pd.Timedelta(days=1) if day_start < safe_max.normalize() else safe_max
    end = next_day_start - pd.Timedelta(microseconds=1)
    return min(end, safe_max)


def _safe_pandas_max_timestamp() -> pd.Timestamp:
    return pd.Timestamp.max.floor("us")


def _rows_to_float_array(rows: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    cols = tuple(columns)
    values = np.zeros((len(rows), len(cols)), dtype=np.float32)
    for index, column in enumerate(cols):
        if column in rows:
            values[:, index] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return values


def _rows_to_bool_array(rows: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    cols = tuple(columns)
    values = np.zeros((len(rows), len(cols)), dtype=np.uint8)
    for index, column in enumerate(cols):
        if column in rows:
            values[:, index] = rows[column].fillna(False).astype(bool).to_numpy(dtype=np.uint8)
    return values


def _timestamp_to_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _numeric_value(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return number if np.isfinite(number) else 0.0


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Agent cache build cancellation requested.")
