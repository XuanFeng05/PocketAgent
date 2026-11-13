from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from agent_layer.data.cache_schema import (
    AGENT_CACHE_VERSION,
    CACHE_MANIFEST_NAME,
    CACHE_STORAGE_INDEX_BASED,
    CONSTRAINT_COLUMNS,
    EXECUTION_COLUMNS,
    SOURCE_DATASET_FINGERPRINT_VERSION,
    AgentCacheManifest,
    SymbolCacheMetadata,
    feature_parts_source_fingerprint,
    normalize_symbol_list,
    safe_symbol_name,
    symbol_cache_dir,
    write_json,
)
from agent_layer.data.decision_cache import _attach_liquidity
from agent_layer.data.episode_index import (
    build_episode_index_frame,
    read_episode_index,
    write_episode_index,
)
from agent_layer.data.feature_parts import FeaturePartsDataset
from agent_layer.data.tensor_store import describe_symbol_cache, write_symbol_cache
from feature_layer.builders.aggregation import normalize_frequency
from feature_layer.builders.snapshot_features import RAW_COLUMNS as SNAPSHOT_RAW_COLUMNS
from feature_layer.builders.snapshot_features import SnapshotFeatureEngine
from feature_layer.datasets.compact import (
    _fast_snapshot_row_from_arrays,
    _raw_day_lookup,
    _raw_symbol_lookup,
    _spec_from_store,
)

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
    feature_dir = Path(config.feature_dir).resolve()
    output_dir = Path(config.output_dir)
    if config.reset and output_dir.exists():
        _ensure_safe_cache_reset_target(output_dir)
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
    source_fingerprint = feature_parts_source_fingerprint(
        source.dataset_dir,
        symbols=selected_symbols,
        frequencies=selected_frequencies,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    # Invalidate any older manifest before touching symbol files.  If a build is
    # cancelled or crashes, readers see an explicitly incomplete v3 cache rather
    # than silently combining a stale manifest with partially replaced arrays.
    write_json(
        output_dir / CACHE_MANIFEST_NAME,
        AgentCacheManifest(
            cache_version=AGENT_CACHE_VERSION,
            source_feature_dir=str(source.dataset_dir),
            schema_hash=str(compiled.get("schema_hash") or ""),
            frequencies=tuple(selected_frequencies),
            symbols=(),
            symbol_count=0,
            decision_count=0,
            created_at=created_at,
            storage=CACHE_STORAGE_INDEX_BASED,
            requested_symbols=tuple(selected_symbols),
            failed_symbols=tuple(selected_symbols),
            source_dataset_fingerprint=source_fingerprint,
            source_dataset_fingerprint_version=SOURCE_DATASET_FINGERPRINT_VERSION,
            complete=False,
        ).to_json_dict(),
    )
    worker_count = max(1, min(int(config.workers), len(selected_symbols)))
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
    summary_by_symbol = {
        str(item.get("symbol") or "").upper(): item
        for item in summaries
        if str(item.get("symbol") or "").strip()
    }
    ordered_summaries: list[dict[str, Any]] = []
    for symbol in selected_symbols:
        summary = summary_by_symbol.get(symbol)
        if summary is None:
            summary = {
                "symbol": symbol,
                "safe_symbol": safe_symbol_name(symbol),
                "decision_count": 0,
                "error": "Agent cache worker returned no result.",
            }
        elif not summary.get("error") and int(summary.get("decision_count") or 0) <= 0:
            summary = {**summary, "error": "No decisions were cached for the selected range."}
        ordered_summaries.append(summary)
    failures = [item for item in ordered_summaries if item.get("error")]
    success = [item for item in ordered_summaries if not item.get("error")]
    episode_index = build_episode_index_frame(success)
    write_episode_index(output_dir, episode_index)
    complete = not failures and len(success) == len(selected_symbols)
    manifest = AgentCacheManifest(
        cache_version=AGENT_CACHE_VERSION,
        source_feature_dir=str(source.dataset_dir),
        schema_hash=str(compiled.get("schema_hash") or ""),
        frequencies=tuple(selected_frequencies),
        symbols=tuple(str(item["symbol"]) for item in success),
        symbol_count=len(success),
        decision_count=int(sum(int(item.get("decision_count") or 0) for item in success)),
        created_at=created_at,
        storage=CACHE_STORAGE_INDEX_BASED,
        requested_symbols=tuple(selected_symbols),
        failed_symbols=tuple(str(item.get("symbol") or "") for item in failures),
        source_dataset_fingerprint=source_fingerprint,
        source_dataset_fingerprint_version=SOURCE_DATASET_FINGERPRINT_VERSION,
        complete=complete,
    )
    write_json(output_dir / CACHE_MANIFEST_NAME, manifest.to_json_dict())
    return {
        "ok": complete,
        "output_dir": str(output_dir),
        "symbols": len(success),
        "failed_symbols": len(failures),
        "decision_count": manifest.decision_count,
        "frequencies": list(selected_frequencies),
        "schema_hash": manifest.schema_hash,
        "source_dataset_fingerprint": manifest.source_dataset_fingerprint,
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
    episode_rows = int(len(read_episode_index(root))) if episode_index_path.exists() else 0
    return {
        "manifest": manifest,
        "episode_index_rows": episode_rows,
        "samples": samples,
    }


def _ensure_safe_cache_reset_target(path: Path) -> None:
    resolved = path.resolve()
    parts = tuple(part.lower() for part in resolved.parts)
    if "agent_cache" not in parts:
        raise ValueError(
            f"Refusing to reset non-Agent-cache directory: {resolved}. "
            "Choose a directory under runtime_layer/agent_cache."
        )
    cache_index = parts.index("agent_cache")
    if len(parts) <= cache_index + 1:
        raise ValueError(f"Refusing to reset the Agent cache root directory: {resolved}")
    if resolved.parent == resolved:
        raise ValueError(f"Refusing to reset filesystem root: {resolved}")


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
        decisions=rows,
        index_rows=index_rows,
        decision_ids=decision_ids,
        frequencies=job.frequencies,
        feature_names=feature_names,
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
    stage_values = tuple(str(stage) for stage in (job.stages or ()) if str(stage))
    stage_clause = "AND d.stage IN (SELECT UNNEST(?)) " if stage_values else ""
    limit_clause = "LIMIT ? " if job.max_decisions is not None and int(job.max_decisions) > 0 else ""
    params: list[object] = [start, end, [job.symbol]]
    if stage_values:
        params.append(list(stage_values))
    if limit_clause:
        params.append(int(job.max_decisions or 0))
    with source.connect(symbols=(job.symbol,)) as conn:
        rows = conn.execute(
            "SELECT d.* FROM decisions d "
            "WHERE d.decision_time >= ? AND d.decision_time <= ? "
            "AND d.symbol IN (SELECT UNNEST(?)) "
            f"{stage_clause}"
            "ORDER BY d.decision_time, CASE d.stage WHEN 'open_auction' THEN 0 ELSE 1 END, d.symbol "
            f"{limit_clause}",
            params,
        ).fetchdf()
        if rows.empty:
            return rows
        rows["source_bar_end"] = pd.to_datetime(rows["source_bar_end"], errors="coerce")
        source_bounds = rows["source_bar_end"].dropna()
        if source_bounds.empty:
            intraday = pd.DataFrame(columns=["symbol", "adjust", "bar_datetime", "volume", "amount"])
        else:
            intraday = conn.execute(
                "SELECT symbol, adjust, bar_datetime, volume, amount FROM market_bars "
                "WHERE symbol = ? AND freq = '5min' "
                "AND bar_datetime >= ? AND bar_datetime <= ?",
                [job.symbol, source_bounds.min(), source_bounds.max()],
            ).fetchdf()
        daily = conn.execute(
            "SELECT symbol, adjust, bar_datetime, volume, amount FROM market_bars "
            "WHERE symbol = ? AND freq = 'daily' "
            "ORDER BY symbol, adjust, bar_datetime",
            [job.symbol],
        ).fetchdf()
    rows = _attach_liquidity(rows, intraday=intraday, daily=daily)
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["decision_time"] = pd.to_datetime(rows["decision_time"], errors="coerce")
    rows["_stage_order"] = rows["stage"].astype(str).map({"open_auction": 0, "bar_close": 1}).fillna(99)
    rows = (
        rows.sort_values(["decision_time", "_stage_order", "symbol"])
        .drop(columns=["_stage_order"])
        .reset_index(drop=True)
    )
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
    decisions: pd.DataFrame,
    index_rows: pd.DataFrame,
    decision_ids: list[str],
    frequencies: tuple[str, ...],
    feature_names: dict[str, tuple[str, ...]],
    snapshot_required: dict[str, np.ndarray],
    snapshot_features: dict[str, np.ndarray],
    chunk_size: int,
) -> None:
    if not decision_ids:
        return
    if not any(bool(values.any()) for values in snapshot_required.values()):
        return

    row_by_decision = {decision_id: index for index, decision_id in enumerate(decision_ids)}
    source_symbols = source._symbols_for_decision_ids(decision_ids)  # noqa: SLF001 - cache build fast path.
    if not source_symbols:
        return
    with source.connect(symbols=source_symbols) as conn:
        spec = _spec_from_store(conn)
        normalized_trade_freq = normalize_frequency(spec.trade_frequency)
        full_decisions = conn.execute(
            "SELECT * FROM decisions WHERE decision_id IN (SELECT UNNEST(?))",
            [decision_ids],
        ).fetchdf()
        if full_decisions.empty:
            raise KeyError("Cannot build snapshot features because decision rows are missing.")
        full_decisions["decision_id"] = full_decisions["decision_id"].astype(str)
        decision_lookup = {
            str(row.decision_id): row
            for row in full_decisions.itertuples(index=False)
        }
        base_lookup = _raw_day_lookup(
            _load_market_bars_for_snapshots(conn, source_symbols, normalized_trade_freq)
        )
        daily_lookup = _raw_symbol_lookup(
            _load_market_bars_for_snapshots(conn, source_symbols, "daily")
        )
        raw_history = _load_snapshot_engine_inputs(conn, source_symbols, frequencies)
        engines: dict[tuple[str, str, str], SnapshotFeatureEngine] = {}

        for freq in frequencies:
            if not bool(snapshot_required[freq].any()):
                continue
            freq_index = index_rows.loc[index_rows["freq"].astype(str).eq(freq)].copy()
            if freq_index.empty:
                continue
            freq_index["decision_id"] = freq_index["decision_id"].astype(str)
            names = feature_names[freq]
            grouped: dict[tuple[str, str], list[tuple[str, object, dict[str, Any]]]] = {}
            missing: list[str] = []
            for index_row in freq_index.itertuples(index=False):
                decision_id = str(index_row.decision_id)
                target_index = row_by_decision.get(decision_id)
                if target_index is None or not bool(snapshot_required[freq][target_index]):
                    continue
                decision = decision_lookup.get(decision_id)
                if decision is None:
                    missing.append(decision_id)
                    continue
                decision_time = pd.Timestamp(decision.decision_time)
                visible_bar_end = getattr(decision, "visible_bar_end", None)
                cutoff = pd.Timestamp(visible_bar_end) if pd.notna(visible_bar_end) else decision_time
                adjust = str(index_row.adjust)
                raw = _fast_snapshot_row_from_arrays(
                    decision,
                    freq=freq,
                    cutoff=cutoff,
                    stable_end=(
                        pd.Timestamp(index_row.stable_end_datetime)
                        if pd.notna(index_row.stable_end_datetime)
                        else None
                    ),
                    day_values=base_lookup.get((str(index_row.symbol), adjust, decision_time.normalize())),
                    daily_values=daily_lookup.get((str(index_row.symbol), adjust)),
                )
                if raw is None:
                    missing.append(decision_id)
                    continue
                grouped.setdefault((str(index_row.symbol), adjust), []).append((decision_id, index_row.stable_end_datetime, raw))
            if missing:
                raise ValueError(
                    "Feature Dataset cannot materialize required snapshot features "
                    f"for {freq}; missing decision_ids: {', '.join(missing[:10])}. "
                    "Rebuild Feature Dataset after verifying raw bars for this frequency."
                )
            for (symbol, adjust), items in grouped.items():
                engine_key = (symbol, freq, adjust)
                engine = engines.get(engine_key)
                if engine is None:
                    times, raw_values = raw_history.get(
                        (str(source_symbols[0]), freq, adjust),
                        (np.empty(0, dtype=np.int64), np.empty((0, len(SNAPSHOT_RAW_COLUMNS)), dtype=np.float64)),
                    )
                    engine = SnapshotFeatureEngine(times, raw_values, spec=spec, freq=freq)
                    engines[engine_key] = engine
                built_matrix = engine.build(
                    [item[1] for item in items],
                    [item[2] for item in items],
                    feature_names=names,
                    as_matrix=True,
                )
                for row_index, (decision_id, _, _) in enumerate(items):
                    target_index = row_by_decision[decision_id]
                    snapshot_features[freq][target_index] = built_matrix[row_index]


def _load_market_bars_for_snapshots(conn: Any, symbols: Iterable[str], freq: str) -> pd.DataFrame:
    frame = conn.execute(
        "SELECT * EXCLUDE (market_row_id) FROM market_bars "
        "WHERE symbol IN (SELECT UNNEST(?)) AND freq = ? "
        "ORDER BY symbol, adjust, bar_datetime",
        [list(symbols), str(freq)],
    ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "datetime", "adjust", *SNAPSHOT_RAW_COLUMNS])
    frame = frame.rename(columns={"bar_datetime": "datetime"}).copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return frame


def _load_snapshot_engine_inputs(
    conn: Any,
    symbols: Iterable[str],
    frequencies: Iterable[str],
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    frame = conn.execute(
        "SELECT * EXCLUDE (market_row_id) FROM market_bars "
        "WHERE symbol IN (SELECT UNNEST(?)) AND freq IN (SELECT UNNEST(?)) "
        "ORDER BY symbol, freq, adjust, bar_datetime",
        [list(symbols), list(frequencies)],
    ).fetchdf()
    if frame.empty:
        return {}
    frame["bar_datetime"] = pd.to_datetime(frame["bar_datetime"], errors="coerce")
    frame = frame.dropna(subset=["bar_datetime"]).copy()
    for column in SNAPSHOT_RAW_COLUMNS:
        if column not in frame:
            frame[column] = 1.0 if column == "progress" else np.nan
    result: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (symbol, freq, adjust), group in frame.groupby(["symbol", "freq", "adjust"], sort=False):
        ordered = group.sort_values("bar_datetime")
        times = ordered["bar_datetime"].to_numpy(dtype="datetime64[ns]").view(np.int64)
        raw = (
            ordered[list(SNAPSHOT_RAW_COLUMNS)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64, copy=True)
        )
        result[(str(symbol), str(freq), str(adjust))] = (times, raw)
    return result


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
