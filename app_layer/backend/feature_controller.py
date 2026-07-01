from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from feature_layer import DEFAULT_FEATURE_SPEC
from feature_layer.indicator_registry import (
    active_feature_spec,
    indicator_config_payload,
    indicator_lookback,
    indicator_to_payload,
    save_indicator_specs,
)
from feature_layer.rules import price_limit_pct_for_symbol
from feature_layer.model_input_registry import (
    compile_model_input_blueprint,
    load_model_input_blueprint,
    model_input_blueprint_payload,
    save_model_input_blueprint,
    validate_model_input_blueprint,
)
from feature_layer.builders.indicator_visualization import (
    build_indicator_visualization_payload,
)
from feature_layer.datasets import (
    DEFAULT_DATASET_FREQUENCIES,
    CompactFeatureStoreWriter,
    build_compact_metadata_frame,
    FeatureChunkBuildTask,
    FeatureDatasetConfig,
    build_compact_feature_chunk,
    build_compact_feature_dataset_from_duckdb,
    build_feature_dataset_from_duckdb,
    chunk_symbols,
    evaluate_market_status_coverage,
    evaluate_frequency_warmup,
    merge_compact_feature_stores,
    merge_compact_feature_parquet_parts,
)
from feature_layer.datasets.market_parquet import (
    ensure_market_parquet_cache,
    resolve_market_parquet_cache_dir,
)

from data_layer.storage.duckdb_storage import (
    KLINE_TABLE_NAME,
    STOCK_STATUS_DAILY_TABLE_NAME,
    connect_duckdb,
    load_kline_from_duckdb,
    load_stock_status_daily_from_duckdb,
)
from data_layer.storage.partitioned_storage import KLINE_DATASET, DAILY_LIQUIDITY_DATASET, DAILY_STATUS_DATASET, get_market_catalog_record, has_market_shard_storage, resolve_market_data_root

from app_layer.backend.data_controller import (
    DEFAULT_DB_PATH,
    DEFAULT_REPORT_DIR,
    _relative_or_absolute,
    _resolve_symbols,
    resolve_project_path,
)
from app_layer.backend.json_utils import dataframe_to_records


DEFAULT_FEATURE_OUTPUT_DIR = f"{DEFAULT_REPORT_DIR}/feature_dataset"
FEATURE_PARTS_DIR_NAME = "feature_parts"
FEATURE_PARTS_MANIFEST_NAME = "feature_parts_manifest.json"
FEATURE_PART_TABLES = (
    "decisions",
    "decision_context",
    "constraints",
    "market_bars",
    "market_features",
    "decision_index",
    "decision_snapshots",
    "dataset_metadata",
)



@dataclass(frozen=True)
class _FeatureBuildSummary:
    spec_name: str
    frequencies: tuple[str, ...]
    requested_symbols: tuple[str, ...]
    decisions: int
    decision_context_rows: int
    constraint_rows: int
    market_rows: dict[str, int]
    decision_index_rows: int
    snapshot_rows: int
    st_decisions: int
    model_input_schema_hash: str

    def summary(self) -> dict[str, object]:
        return {
            "spec": self.spec_name,
            "decisions": self.decisions,
            "decision_context_rows": self.decision_context_rows,
            "constraint_rows": self.constraint_rows,
            "frequencies": list(self.frequencies),
            "market_rows": self.market_rows,
            "requested_symbols": list(self.requested_symbols),
            "decision_index_rows": self.decision_index_rows,
            "snapshot_rows": self.snapshot_rows,
            "st_decisions": self.st_decisions,
            "model_input_schema_hash": self.model_input_schema_hash,
        }


def feature_spec_payload() -> dict[str, Any]:
    spec = active_feature_spec()
    return {
        "name": spec.name,
        "version": spec.version,
        "base_frequency": spec.base_frequency,
        "trade_frequency": spec.trade_frequency,
        "default_frequencies": ["5min", "30min", "daily", "weekly"],
        "available_frequencies": [spec.base_frequency, *spec.derived_frequencies],
        "sequence_windows": spec.sequence_windows,
        "indicators": [indicator_to_payload(item) for item in spec.indicators],
        "decision_stages": [stage.value for stage in spec.decision_stages],
        "frequency_policy": [
            {"freq": "5min", "source": "downloaded 5min bars", "visible_rule": "completed bar only"},
            {"freq": "15min/30min/60min", "source": "aggregated from visible 5min bars", "visible_rule": "may include partial higher-period bar"},
            {"freq": "daily", "source": "official completed daily bars + current partial daily from visible 5min", "visible_rule": "today official daily is never exposed before close"},
            {"freq": "weekly", "source": "aggregated from official completed daily bars + current partial daily", "visible_rule": "current week is partial until completed"},
            {"freq": "monthly", "source": "aggregated from official completed daily bars + current partial daily", "visible_rule": "current month is partial until completed"},
        ],
        "market_fields": [_field_payload(field, _formula_for_field(field.name)) for field in spec.market_fields],
        "context_fields": [_field_payload(field, _formula_for_field(field.name)) for field in spec.context_fields],
        "portfolio_fields": [_field_payload(field, _formula_for_field(field.name)) for field in spec.portfolio_fields],
        "constraint_fields": [_field_payload(field, _formula_for_field(field.name)) for field in spec.constraint_fields],
        "environment_fields": [_field_payload(field, _formula_for_field(field.name)) for field in spec.environment_fields],
        "dataset_outputs": ["feature_parts/", "feature_parts_manifest.json", "model_input_contract.json", "preview_*.csv", "manifest.json", "quality_report.json"],
        "ema_channels": [
            {"name": channel.name, "fast_period": channel.fast_period, "slow_period": channel.slow_period}
            for channel in spec.ema_channels
        ],
        "indicator_details": _indicator_details(spec),
        "trading_rules": [
            {"rule": "price_limit_main_board", "value": "10%", "note": "Default for symbols outside special board prefixes."},
            {"rule": "price_limit_star_chinext", "value": "20%", "note": "688/689 and 300/301 prefixes."},
            {"rule": "price_limit_beijing_reserved", "value": "30%", "note": "8/4/920 prefixes reserved for BSE rows."},
            {"rule": "price_limit_st", "value": "5%", "note": "Dated isST status overrides the board limit for that trading day."},
            {"rule": "limit_reference", "value": "previous trading-day close", "note": "The reference is fixed for every intraday decision in the session."},
            {"rule": "T+1", "value": "fixed", "note": "Represented by environment-provided portfolio fields and enforced outside Feature Layer."},
            {"rule": "zero_volume", "value": "not tradeable", "note": "market_can_buy/market_can_sell become false when completed bar volume is zero."},
            {"rule": "ST_policy", "value": "no new position", "note": "ST days block market_can_buy but retain legal exits; is_st is audit metadata, not a model feature."},
        ],
        "example_price_limits": {
            "000001.SZ": price_limit_pct_for_symbol("000001.SZ"),
            "300001.SZ": price_limit_pct_for_symbol("300001.SZ"),
            "688001.SH": price_limit_pct_for_symbol("688001.SH"),
            "000001.SZ@ST": price_limit_pct_for_symbol("000001.SZ", is_st=True),
        },
        "table_formats": [
            {"file": "feature_parts/", "grain": "symbol-part parquet training dataset", "columns": ["decisions", "decision_context", "constraints", "market_bars", "market_features", "decision_index", "decision_snapshots"]},
            {"file": "feature_parts_manifest.json", "grain": "one manifest for symbol-part reuse and dataset discovery", "columns": ["symbols", "parts_dir", "fingerprint", "source_signature", "summary"]},
            {"file": "model_input_contract.json", "grain": "one model-input schema contract for Agent Layer", "columns": ["schema_hash", "frequencies", "market_fields", "context_fields", "constraints"]},
            {"file": "preview_*.csv", "grain": "bounded human-review samples only", "columns": ["first 1000 decisions", "decision context", "constraints"]},
            {"file": "model batch", "grain": "one fixed-size tensor per decision and frequency", "columns": ["left-zero-padded feature values", "sequence_mask (0=padding, 1=real row)", "valid_ratio"]},
            {"file": "portfolio contract", "grain": "environment-provided at training/simulation time, not generated by Feature Layer", "columns": [*spec.portfolio_feature_names, *spec.environment_feature_names]},
            {"file": "manifest.json", "grain": "one file per dataset build", "columns": ["spec", "request", "feature_columns", "outputs", "generated_at", "git_commit"]},
            {"file": "quality_report.json", "grain": "one file per dataset build", "columns": ["rows", "nan_counts", "inf_counts", "clipped_counts", "missing_sequence_count"]},
        ],
    }


def feature_indicator_config() -> dict[str, Any]:
    return indicator_config_payload()


def update_feature_indicator_config(payload: dict[str, Any]) -> dict[str, Any]:
    return save_indicator_specs(payload)


def feature_model_input_blueprint() -> dict[str, Any]:
    return _enrich_model_input_payload(model_input_blueprint_payload())


def update_feature_model_input_blueprint(payload: dict[str, Any]) -> dict[str, Any]:
    return _enrich_model_input_payload(save_model_input_blueprint(payload))


def validate_feature_model_input_blueprint(payload: dict[str, Any]) -> dict[str, Any]:
    blueprint = payload.get("blueprint") if isinstance(payload.get("blueprint"), dict) else payload
    validation = validate_model_input_blueprint(blueprint)
    return {
        "validation": validation,
        "compiled": compile_model_input_blueprint(blueprint) if validation["valid"] else None,
    }


def _enrich_model_input_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for item in payload.get("catalog", []):
        item["formula"] = _formula_for_field(str(item.get("name") or ""))
    return payload




def _market_source_available(path: Path) -> bool:
    return path.exists() or has_market_shard_storage(resolve_market_data_root(path))


def _market_source_label(path: Path) -> str:
    shard_root = resolve_market_data_root(path)
    if has_market_shard_storage(shard_root):
        return f"partitioned shards: {_relative_or_absolute(shard_root) or shard_root}"
    return f"DuckDB: {_relative_or_absolute(path) or path}"

def feature_visualization_payload(payload: dict[str, Any]) -> dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("Symbol is required.")
    offset = payload.get("offset")
    return build_indicator_visualization_payload(
        resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH),
        symbol=symbol,
        freq=str(payload.get("freq") or "daily"),
        adjust=str(payload.get("adjust") or "none"),
        offset=None if offset in (None, "") else int(offset),
        limit=int(payload.get("limit") or 240),
    )


def build_feature_dataset(
    payload: dict[str, Any],
    *,
    progress_callback: Callable[[int, int, str, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    request = _feature_request(payload)
    if not request["symbols"]:
        raise ValueError("At least one symbol is required to build a feature dataset.")
    db_path = request["db_path"]
    output_dir = request["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_feature_outputs(output_dir)

    preview_limit = 1000
    previews: dict[str, list[pd.DataFrame]] = {
        "decisions": [],
        "decision_context": [],
        "constraints": [],
    }
    totals = {
        "decisions": 0,
        "decision_context_rows": 0,
        "constraint_rows": 0,
        "decision_index_rows": 0,
        "snapshot_rows": 0,
        "st_decisions": 0,
    }
    market_rows = {freq: 0 for freq in request["config"].frequencies}
    processed_symbols: list[str] = []
    compiled_model_input: dict[str, Any] | None = None
    partial_store_path = output_dir / "feature_store.partial.duckdb"
    if partial_store_path.exists():
        partial_store_path.unlink()

    shard_root = resolve_market_data_root(db_path)
    if has_market_shard_storage(shard_root):
        # Partitioned market storage is already the canonical parquet source.
        # Prefer it even when a stale compatibility market.duckdb file still exists
        # beside the shards; otherwise feature builds may accidentally export old
        # DuckDB rows and produce zero decisions for the new shard dataset.
        request.pop("source_cache_dir", None)
        request["market_source"] = "market_shards"
        request["market_cache_info"] = {
            "source": _market_source_label(db_path),
            "seconds": 0.0,
        }
    elif request.get("market_parquet_cache_enabled") and db_path.exists():
        if progress_callback:
            progress_callback(0, len(request["symbols"]), "preparing_market_parquet_cache", 0)
        cache_started = perf_counter()
        cache_info = ensure_market_parquet_cache(
            db_path,
            cache_root=request["market_parquet_cache_root"],
            symbols=request["symbols"],
            adjust=request["adjust"],
            trade_freq=request["trade_freq"],
            end=request["end"],
            force=bool(request.get("market_parquet_cache_force", False)),
        )
        request["source_cache_dir"] = str(cache_info.cache_dir)
        request["market_source"] = "parquet"
        request["market_cache_info"] = {
            "cache_dir": _relative_or_absolute(cache_info.cache_dir) or str(cache_info.cache_dir),
            "fingerprint": cache_info.fingerprint,
            "exported_files": cache_info.exported_files,
            "reused_files": cache_info.reused_files,
            "seconds": cache_info.seconds,
        }
        request["market_cache_seconds"] = float(perf_counter() - cache_started)
    elif request.get("market_parquet_cache_enabled") and not db_path.exists():
        request["market_source"] = "missing"
        request["market_cache_info"] = {"source": _market_source_label(db_path), "seconds": 0.0}

    build_state = _build_feature_store_chunks(
        request=request,
        db_path=db_path,
        output_dir=output_dir,
        partial_store_path=partial_store_path,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        preview_limit=preview_limit,
    )
    if build_state.get("cancelled"):
        if partial_store_path.exists():
            partial_store_path.unlink()
        cancelled_totals = build_state.get("totals", totals)
        cancelled_market_rows = build_state.get("market_rows", market_rows)
        return {
            "cancelled": True,
            "summary": {**cancelled_totals, "market_rows": cancelled_market_rows},
            "output_dir": _relative_or_absolute(output_dir),
            "decisions": cancelled_totals["decisions"],
            "market_rows": cancelled_market_rows,
            "frequencies": list(request["config"].frequencies),
        }

    totals = build_state["totals"]
    market_rows = build_state["market_rows"]
    previews = build_state["previews"]
    processed_symbols = build_state["processed_symbols"]
    compiled_model_input = build_state["compiled_model_input"]
    performance = build_state.get("performance", {})
    if request.get("market_cache_info"):
        performance["market_cache"] = request.get("market_cache_info")
        performance["market_cache_seconds"] = request.get("market_cache_seconds")

    if cancel_check and cancel_check():
        if partial_store_path.exists():
            partial_store_path.unlink()
        return {
            "cancelled": True,
            "summary": {**totals, "market_rows": market_rows},
            "output_dir": _relative_or_absolute(output_dir),
            "decisions": totals["decisions"],
            "market_rows": market_rows,
            "frequencies": list(request["config"].frequencies),
        }
    if not processed_symbols or compiled_model_input is None:
        if partial_store_path.exists():
            partial_store_path.unlink()
        raise ValueError("No feature data was produced for the selected symbols.")
    if partial_store_path.exists():
        partial_store_path.unlink()


    def preview_frame(name: str) -> pd.DataFrame:
        return pd.concat(previews[name], ignore_index=True) if previews[name] else pd.DataFrame()

    outputs: dict[str, str] = {}
    outputs["preview_decisions"] = _write_csv(preview_frame("decisions"), output_dir / "preview_decisions.csv")
    outputs["preview_decision_context"] = _write_csv(preview_frame("decision_context"), output_dir / "preview_decision_context.csv")
    outputs["preview_constraints"] = _write_csv(preview_frame("constraints"), output_dir / "preview_constraints.csv")
    parts_root = output_dir / FEATURE_PARTS_DIR_NAME
    if parts_root.exists():
        outputs["feature_parts"] = _relative_or_absolute(parts_root) or str(parts_root)
    summary_dataset = _FeatureBuildSummary(
        spec_name=request["spec"].name,
        frequencies=tuple(request["config"].frequencies),
        requested_symbols=tuple(processed_symbols),
        decisions=totals["decisions"],
        decision_context_rows=totals["decision_context_rows"],
        constraint_rows=totals["constraint_rows"],
        market_rows=market_rows,
        decision_index_rows=totals["decision_index_rows"],
        snapshot_rows=totals["snapshot_rows"],
        st_decisions=totals["st_decisions"],
        model_input_schema_hash=str(compiled_model_input.get("schema_hash") or ""),
    )
    quality = _streaming_quality_report(summary_dataset, spec=request["spec"])
    outputs["quality_report"] = _write_json(quality, output_dir / "quality_report.json")

    manifest = _dataset_manifest(
        summary_dataset,
        request=request,
        outputs=outputs,
        output_dir=output_dir,
        model_input=compiled_model_input,
    )
    outputs["manifest"] = _write_json(manifest, output_dir / "manifest.json")

    summary = summary_dataset.summary()
    return {
        "summary": summary,
        "output_dir": _relative_or_absolute(output_dir),
        "outputs": outputs,
        "manifest": outputs["manifest"],
        "quality_report": outputs["quality_report"],
        "decisions": totals["decisions"],
        "market_rows": summary.get("market_rows", {}),
        "frequencies": list(summary_dataset.frequencies),
        "performance": performance,
    }


def _build_feature_store_chunks(
    *,
    request: dict[str, Any],
    db_path: Path,
    output_dir: Path,
    partial_store_path: Path,
    progress_callback: Callable[[int, int, str, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    preview_limit: int,
) -> dict[str, Any]:
    chunk_size = max(1, int(request.get("feature_build_chunk_size") or 16))
    workers = max(1, int(request.get("feature_build_workers") or 1))
    low_memory_mode = bool(request.get("feature_low_memory", True))
    if request["config"].max_decisions:
        workers = 1
    if _should_use_incremental_feature_parts(request):
        return _build_feature_store_chunks_incremental(
            request=request,
            db_path=db_path,
            output_dir=output_dir,
            partial_store_path=partial_store_path,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            preview_limit=preview_limit,
            workers=workers,
        )
    raise ValueError(
        "Feature Dataset build now only supports parquet feature_parts output. "
        "Keep Incremental build enabled, keep intermediate format as parquet, and do not use Max Decisions for a full build. "
        "Use Feature Preview for lightweight samples instead of materializing feature_store.duckdb."
    )
    if workers <= 1 or len(request["symbols"]) <= chunk_size:
        return _build_feature_store_chunks_serial(
            request=request,
            db_path=db_path,
            partial_store_path=partial_store_path,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            preview_limit=preview_limit,
            chunk_size=chunk_size,
            low_memory_mode=low_memory_mode,
        )
    return _build_feature_store_chunks_parallel(
        request=request,
        db_path=db_path,
        output_dir=output_dir,
        partial_store_path=partial_store_path,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        preview_limit=preview_limit,
        chunk_size=chunk_size,
        workers=workers,
        low_memory_mode=low_memory_mode,
    )


def _initial_feature_build_state(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "totals": {
            "decisions": 0,
            "decision_context_rows": 0,
            "constraint_rows": 0,
            "decision_index_rows": 0,
            "snapshot_rows": 0,
            "st_decisions": 0,
        },
        "market_rows": {freq: 0 for freq in request["config"].frequencies},
        "previews": {"decisions": [], "decision_context": [], "constraints": []},
        "processed_symbols": [],
        "compiled_model_input": None,
        "performance": {},
    }


def _accumulate_preview(
    previews: dict[str, list[pd.DataFrame]],
    key: str,
    frame: pd.DataFrame,
    *,
    preview_limit: int,
) -> None:
    captured = sum(len(part) for part in previews[key])
    if captured < preview_limit:
        previews[key].append(frame.head(preview_limit - captured).copy())


def _accumulate_compact_result(
    state: dict[str, Any],
    *,
    summary: dict[str, Any],
    symbols: Iterable[str],
    compiled_model_input: dict[str, Any],
    preview_decisions: pd.DataFrame,
    preview_decision_context: pd.DataFrame,
    preview_constraints: pd.DataFrame,
    preview_limit: int,
) -> None:
    totals = state["totals"]
    market_rows = state["market_rows"]
    totals["decisions"] += int(summary.get("decisions", 0))
    totals["decision_context_rows"] += int(summary.get("decision_context_rows", 0))
    totals["constraint_rows"] += int(summary.get("constraint_rows", 0))
    totals["decision_index_rows"] += int(summary.get("decision_index_rows", 0))
    totals["snapshot_rows"] += int(summary.get("snapshot_rows", 0))
    totals["st_decisions"] += int(summary.get("st_decisions", 0))
    for freq in market_rows:
        market_rows[freq] += int(summary.get("market_rows", {}).get(freq, 0))
    processed = state["processed_symbols"]
    for symbol in symbols:
        if symbol not in processed:
            processed.append(str(symbol))
    state["compiled_model_input"] = compiled_model_input
    _accumulate_preview(state["previews"], "decisions", preview_decisions, preview_limit=preview_limit)
    _accumulate_preview(state["previews"], "decision_context", preview_decision_context, preview_limit=preview_limit)
    _accumulate_preview(state["previews"], "constraints", preview_constraints, preview_limit=preview_limit)



def _canonical_dataset_metadata_frame(request: dict[str, Any], compiled_model_input: dict[str, Any]) -> pd.DataFrame:
    return build_compact_metadata_frame(
        request["spec"],
        load_model_input_blueprint(spec=request["spec"]),
        compiled_model_input,
    )


def _feature_store_metadata_compatible(feature_store_path: Path, compiled_model_input: dict[str, Any]) -> bool:
    if not feature_store_path.exists():
        return False
    required_keys = {
        "spec_name",
        "spec_version",
        "sequence_windows",
        "indicators",
        "market_feature_names",
        "context_feature_names",
        "model_input_blueprint",
        "compiled_model_input",
        "model_input_schema_hash",
        "model_input_schema_version",
    }
    try:
        with connect_duckdb(feature_store_path, read_only=True) as conn:
            tables = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
            if "dataset_metadata" not in tables:
                return False
            rows = dict(conn.execute("SELECT key, value FROM dataset_metadata").fetchall())
    except Exception:
        return False
    if not required_keys.issubset(rows.keys()):
        return False
    expected_hash = str(compiled_model_input.get("schema_hash") or "")
    if str(rows.get("model_input_schema_hash") or "") != expected_hash:
        return False
    try:
        compiled = json.loads(str(rows.get("compiled_model_input") or "{}"))
    except Exception:
        return False
    return str(compiled.get("schema_hash") or "") == expected_hash


def _should_use_incremental_feature_parts(request: dict[str, Any]) -> bool:
    return (
        bool(request.get("feature_incremental_enabled", True))
        and not request["config"].max_decisions
        and str(request.get("feature_intermediate_format") or "parquet").lower() == "parquet"
    )


def _build_feature_store_chunks_incremental(
    *,
    request: dict[str, Any],
    db_path: Path,
    output_dir: Path,
    partial_store_path: Path,
    progress_callback: Callable[[int, int, str, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    preview_limit: int,
    workers: int,
) -> dict[str, Any]:
    """Incremental symbol-part build.

    Canonical cache is feature_parts/symbol=<symbol>/... parquet tables. If a
    symbol fingerprint is unchanged, its parquet parts are reused and the worker
    build is skipped.
    """

    state = _initial_feature_build_state(request)
    started = perf_counter()
    total_symbols = len(request["symbols"])
    parts_root = output_dir / FEATURE_PARTS_DIR_NAME
    parts_root.mkdir(parents=True, exist_ok=True)
    old_manifest = _load_feature_parts_manifest(output_dir)
    build_signature = _feature_parts_build_signature(request)
    old_symbols = set(str(symbol) for symbol in old_manifest.get("symbols", {}).keys())
    current_symbols = [str(symbol) for symbol in request["symbols"]]
    current_symbol_set = set(current_symbols)

    removed_symbols = sorted(old_symbols.difference(current_symbol_set))
    if request.get("feature_force_rebuild_parts"):
        removed_symbols = sorted(old_symbols)
    for symbol in removed_symbols:
        shutil.rmtree(_feature_symbol_parts_dir(parts_root, symbol), ignore_errors=True)

    compiled_model_input = compile_model_input_blueprint(load_model_input_blueprint(spec=request["spec"]), spec=request["spec"])
    changed_symbols: list[str] = []
    reused_symbols: list[str] = []
    symbol_entries: dict[str, dict[str, Any]] = {}

    for symbol in current_symbols:
        fingerprint = _feature_symbol_fingerprint(request, symbol=symbol, build_signature=build_signature)
        old_entry = dict(old_manifest.get("symbols", {}).get(symbol, {}))
        parts_dir = _feature_symbol_parts_dir(parts_root, symbol)
        reusable = (
            not request.get("feature_force_rebuild_parts")
            and old_entry.get("fingerprint") == fingerprint
            and _feature_symbol_parts_complete(parts_dir)
        )
        if reusable:
            reused_symbols.append(symbol)
            symbol_entries[symbol] = old_entry
            _accumulate_incremental_summary(
                state,
                summary=dict(old_entry.get("summary", {})),
                symbols=[symbol],
                compiled_model_input=compiled_model_input,
            )
        else:
            changed_symbols.append(symbol)
            symbol_entries[symbol] = {
                "symbol": symbol,
                "fingerprint": fingerprint,
                "parts_dir": _relative_or_absolute(parts_dir) or str(parts_dir),
                "source_signature": _feature_symbol_source_signature(request, symbol),
                "summary": {},
            }

    perf = state["performance"]
    perf["mode"] = "incremental_parallel" if workers > 1 else "incremental_serial"
    perf["workers"] = int(workers)
    perf["reused_symbols"] = len(reused_symbols)
    perf["changed_symbols"] = len(changed_symbols)
    perf["removed_symbols"] = len(removed_symbols)
    perf["parts_root"] = _relative_or_absolute(parts_root) or str(parts_root)
    request["feature_parts_reuse"] = {
        "enabled": True,
        "reused_symbols": len(reused_symbols),
        "changed_symbols": len(changed_symbols),
        "removed_symbols": len(removed_symbols),
        "reuse_ratio": round(len(reused_symbols) / max(1, total_symbols), 6),
    }

    if progress_callback:
        progress_callback(len(reused_symbols), total_symbols, f"checking_feature_parts reuse={len(reused_symbols)} changed={len(changed_symbols)}", state["totals"]["decisions"])

    if cancel_check and cancel_check():
        state["cancelled"] = True
        return state

    tmp_dir = output_dir / f"_tmp_feature_incremental_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    built_symbols = 0
    try:
        if changed_symbols:
            tasks = []
            for index, symbol in enumerate(changed_symbols):
                task_parts = tmp_dir / f"symbol_{index:05d}_{_safe_feature_part_name(symbol)}" / "parts"
                task_store = tmp_dir / f"symbol_{index:05d}_{_safe_feature_part_name(symbol)}" / "feature_chunk.duckdb"
                tasks.append(
                    FeatureChunkBuildTask(
                        chunk_index=index,
                        db_path=str(db_path),
                        chunk_store_path=str(task_store),
                        symbols=(symbol,),
                        adjust=request["adjust"],
                        start=request["start"],
                        end=request["end"],
                        config=request["config"],
                        spec=request["spec"],
                        preview_limit=preview_limit,
                        low_memory_mode=True,
                        chunk_parts_dir=str(task_parts),
                        output_format="parquet",
                        source_cache_dir=request.get("source_cache_dir"),
                    )
                )
            if workers > 1 and len(tasks) > 1:
                executor = ProcessPoolExecutor(max_workers=max(1, int(workers)))
                futures = {executor.submit(build_compact_feature_chunk, task): task for task in tasks}
                pending = set(futures)
                try:
                    while pending:
                        if cancel_check and cancel_check():
                            for future in pending:
                                future.cancel()
                            state["cancelled"] = True
                            return state
                        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        for future in done:
                            task = futures[future]
                            result = future.result()
                            symbol = task.symbols[0]
                            results[symbol] = result
                            built_symbols += 1
                            _accumulate_compact_result(
                                state,
                                summary=result.summary,
                                symbols=result.symbols,
                                compiled_model_input=result.compiled_model_input,
                                preview_decisions=result.preview_decisions,
                                preview_decision_context=result.preview_decision_context,
                                preview_constraints=result.preview_constraints,
                                preview_limit=preview_limit,
                            )
                            _accumulate_incremental_performance(state, result)
                            if progress_callback:
                                completed = len(reused_symbols) + built_symbols
                                progress_callback(min(completed, total_symbols), total_symbols, f"built_changed {symbol}", state["totals"]["decisions"])
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
            else:
                for task in tasks:
                    if cancel_check and cancel_check():
                        state["cancelled"] = True
                        return state
                    result = build_compact_feature_chunk(task)
                    symbol = task.symbols[0]
                    results[symbol] = result
                    built_symbols += 1
                    _accumulate_compact_result(
                        state,
                        summary=result.summary,
                        symbols=result.symbols,
                        compiled_model_input=result.compiled_model_input,
                        preview_decisions=result.preview_decisions,
                        preview_decision_context=result.preview_decision_context,
                        preview_constraints=result.preview_constraints,
                        preview_limit=preview_limit,
                    )
                    _accumulate_incremental_performance(state, result)
                    if progress_callback:
                        completed = len(reused_symbols) + built_symbols
                        progress_callback(min(completed, total_symbols), total_symbols, f"built_changed {symbol}", state["totals"]["decisions"])

        if state.get("cancelled"):
            return state
        for symbol, result in results.items():
            final_dir = _feature_symbol_parts_dir(parts_root, symbol)
            temp_parts = Path(result.chunk_parts_dir or "")
            if not temp_parts.exists():
                raise FileNotFoundError(f"Missing incremental parquet parts for {symbol}: {temp_parts}")
            shutil.rmtree(final_dir, ignore_errors=True)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_parts), str(final_dir))
            entry = symbol_entries[symbol]
            entry["parts_dir"] = _relative_or_absolute(final_dir) or str(final_dir)
            entry["summary"] = result.summary
            entry["built_at"] = datetime.now(timezone.utc).isoformat()
            entry["timings"] = result.timings

        ordered_dirs = [_feature_symbol_parts_dir(parts_root, symbol) for symbol in current_symbols]
        if not ordered_dirs:
            raise ValueError("No feature symbol parts are available.")
        state["skip_replace"] = True
        state["feature_store_materialized"] = False
        state["performance"]["merge_seconds"] = 0.0
        if progress_callback:
            progress_callback(total_symbols, total_symbols, "feature_parts_ready_no_merge", state["totals"]["decisions"])

        state["processed_symbols"] = current_symbols
        state["compiled_model_input"] = compiled_model_input
        state["performance"]["total_seconds"] = float(perf_counter() - started)
        new_manifest = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "build_signature": build_signature,
            "symbols_order": current_symbols,
            "reused_symbols": reused_symbols,
            "changed_symbols": changed_symbols,
            "removed_symbols": removed_symbols,
            "symbols": {symbol: symbol_entries[symbol] for symbol in current_symbols},
        }
        manifest_path = _feature_parts_manifest_path(output_dir)
        manifest_path.write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        request["feature_parts_manifest"] = _relative_or_absolute(manifest_path) or str(manifest_path)
        return state
    except Exception:
        if partial_store_path.exists():
            try:
                partial_store_path.unlink()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _accumulate_incremental_summary(
    state: dict[str, Any],
    *,
    summary: dict[str, Any],
    symbols: Iterable[str],
    compiled_model_input: dict[str, Any],
) -> None:
    _accumulate_compact_result(
        state,
        summary=summary,
        symbols=symbols,
        compiled_model_input=compiled_model_input,
        preview_decisions=pd.DataFrame(),
        preview_decision_context=pd.DataFrame(),
        preview_constraints=pd.DataFrame(),
        preview_limit=0,
    )


def _accumulate_incremental_performance(state: dict[str, Any], result: Any) -> None:
    perf = state["performance"]
    perf["worker_total_seconds_sum"] = float(perf.get("worker_total_seconds_sum", 0.0)) + float(result.timings.get("total_seconds", 0.0))
    perf["worker_build_seconds_sum"] = float(perf.get("worker_build_seconds_sum", 0.0)) + float(result.timings.get("build_seconds", 0.0))
    perf["worker_write_seconds_sum"] = float(perf.get("worker_write_seconds_sum", 0.0)) + float(result.timings.get("write_seconds", 0.0))
    perf["chunks_completed"] = int(perf.get("chunks_completed", 0)) + 1


def _feature_parts_manifest_path(output_dir: Path) -> Path:
    return output_dir / FEATURE_PARTS_MANIFEST_NAME


def _load_feature_parts_manifest(output_dir: Path) -> dict[str, Any]:
    path = _feature_parts_manifest_path(output_dir)
    if not path.exists():
        return {"version": 1, "symbols": {}, "symbols_order": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "symbols": {}, "symbols_order": []}
    if not isinstance(payload, dict):
        return {"version": 1, "symbols": {}, "symbols_order": []}
    payload.setdefault("symbols", {})
    payload.setdefault("symbols_order", [])
    return payload


def _feature_symbol_parts_dir(parts_root: Path, symbol: str) -> Path:
    return parts_root / f"symbol={_safe_feature_part_name(symbol)}"


def _feature_symbol_parts_complete(parts_dir: Path) -> bool:
    if not parts_dir.exists():
        return False
    for table_name in FEATURE_PART_TABLES:
        table_dir = parts_dir / table_name
        if not table_dir.exists() or not any(table_dir.glob("*.parquet")):
            return False
    return True


def _feature_parts_build_signature(request: dict[str, Any]) -> dict[str, Any]:
    spec = request["spec"]
    model_input = compile_model_input_blueprint(load_model_input_blueprint(spec=spec), spec=spec)
    return {
        "version": 1,
        "adjust": request["adjust"],
        "trade_freq": request["trade_freq"],
        "frequencies": list(request["config"].frequencies),
        "include_open_auction": bool(request["config"].include_open_auction),
        "sequence_windows": dict(spec.sequence_windows),
        "date_range": {"start": request["start"], "end": request["end"]},
        "spec": {"name": spec.name, "version": spec.version},
        "market_feature_names": list(spec.market_feature_names),
        "context_feature_names": list(spec.context_feature_names),
        "constraint_feature_names": list(spec.constraint_feature_names),
        "model_input_schema_hash": model_input.get("schema_hash"),
        "market_source": request.get("market_source", "duckdb"),
        "source_cache_dir": str(request.get("source_cache_dir") or ""),
    }


def _feature_symbol_fingerprint(
    request: dict[str, Any],
    *,
    symbol: str,
    build_signature: dict[str, Any],
) -> str:
    payload = {
        "symbol": str(symbol),
        "build_signature": build_signature,
        "source_signature": _feature_symbol_source_signature(request, str(symbol)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _feature_symbol_source_signature(request: dict[str, Any], symbol: str) -> dict[str, Any]:
    source_cache = request.get("source_cache_dir")
    if source_cache:
        cache = Path(source_cache)
        files = [
            cache / "kline" / f"freq={_safe_feature_part_name(request['trade_freq'])}" / f"adjust={_safe_feature_part_name(request['adjust'])}" / f"{_safe_feature_part_name(symbol)}.parquet",
            cache / "kline" / "freq=daily" / f"adjust={_safe_feature_part_name(request['adjust'])}" / f"{_safe_feature_part_name(symbol)}.parquet",
            cache / "stock_liquidity_daily" / f"{_safe_feature_part_name(symbol)}.parquet",
            cache / "stock_status_daily" / f"{_safe_feature_part_name(symbol)}.parquet",
        ]
        return {
            "kind": "market_parquet_cache",
            "cache_dir": str(cache),
            "files": [_path_stat_signature(path) for path in files],
        }
    shard_root = resolve_market_data_root(Path(request["db_path"]))
    if has_market_shard_storage(shard_root):
        records = []
        for dataset, freq, adjust in (
            (KLINE_DATASET, request["trade_freq"], request["adjust"]),
            (KLINE_DATASET, "daily", request["adjust"]),
            (DAILY_LIQUIDITY_DATASET, DAILY_LIQUIDITY_DATASET, "-"),
            (DAILY_STATUS_DATASET, DAILY_STATUS_DATASET, "-"),
        ):
            record = get_market_catalog_record(
                shard_root,
                dataset=dataset,
                symbol=str(symbol),
                freq=str(freq),
                adjust=str(adjust),
            )
            records.append({
                "dataset": dataset,
                "freq": freq,
                "adjust": adjust,
                "rows": None if record is None else int(record.get("rows") or 0),
                "hash": None if record is None else record.get("data_hash"),
                "updated_at": None if record is None else str(record.get("updated_at")),
                "path": None if record is None else record.get("shard_path"),
            })
        return {
            "kind": "market_shards",
            "root": str(shard_root),
            "records": records,
        }
    return {
        "kind": "market_duckdb",
        "db": _path_stat_signature(Path(request["db_path"])),
    }


def _path_stat_signature(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "size": 0, "mtime_ns": 0}


def _safe_feature_part_name(value: object) -> str:
    return str(value).strip().replace("/", "_").replace("\\", "_").replace(":", "_")


def _build_feature_store_chunks_serial(
    *,
    request: dict[str, Any],
    db_path: Path,
    partial_store_path: Path,
    progress_callback: Callable[[int, int, str, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    preview_limit: int,
    chunk_size: int,
    low_memory_mode: bool,
) -> dict[str, Any]:
    state = _initial_feature_build_state(request)
    total_symbols = len(request["symbols"])
    start_total = perf_counter()
    writer = CompactFeatureStoreWriter(partial_store_path, reset=True)
    completed_symbols = 0
    try:
        serial_chunk_size = 1 if low_memory_mode else chunk_size
        for _, batch_symbols in chunk_symbols(request["symbols"], serial_chunk_size):
            if cancel_check and cancel_check():
                state["cancelled"] = True
                return state
            remaining = None
            if request["config"].max_decisions:
                remaining = max(0, int(request["config"].max_decisions) - state["totals"]["decisions"])
                if remaining == 0:
                    break
            batch_config = replace(request["config"], max_decisions=remaining)
            start_build = perf_counter()
            compact = build_compact_feature_dataset_from_duckdb(
                db_path,
                symbols=batch_symbols,
                adjust=request["adjust"],
                start=request["start"],
                end=request["end"],
                config=batch_config,
                spec=request["spec"],
                source_cache_dir=request.get("source_cache_dir"),
            )
            build_seconds = perf_counter() - start_build
            start_write = perf_counter()
            writer.append(compact)
            write_seconds = perf_counter() - start_write
            dataset = compact.dataset
            summary = compact.summary()
            summary.update({
                "constraint_rows": int(len(dataset.constraints)),
                "decision_context_rows": int(len(dataset.decision_context)),
                "decision_index_rows": int(len(compact.decision_index)),
                "snapshot_rows": int(len(compact.decision_snapshots)),
            })
            _accumulate_compact_result(
                state,
                summary=summary,
                symbols=dataset.requested_symbols or batch_symbols,
                compiled_model_input=compact.compiled_model_input,
                preview_decisions=dataset.decisions,
                preview_decision_context=dataset.decision_context,
                preview_constraints=dataset.constraints,
                preview_limit=preview_limit,
            )
            perf = state["performance"]
            perf["mode"] = "serial"
            perf["chunk_size"] = int(chunk_size)
            perf["workers"] = 1
            perf["low_memory_mode"] = bool(low_memory_mode)
            perf["chunks_completed"] = int(perf.get("chunks_completed", 0)) + 1
            perf["worker_build_seconds_sum"] = float(perf.get("worker_build_seconds_sum", 0.0)) + build_seconds
            perf["worker_write_seconds_sum"] = float(perf.get("worker_write_seconds_sum", 0.0)) + write_seconds
            completed_symbols += len(batch_symbols)
            if progress_callback:
                label = batch_symbols[0] if len(batch_symbols) == 1 else f"{batch_symbols[0]}..{batch_symbols[-1]}"
                progress_callback(min(completed_symbols, total_symbols), total_symbols, label, state["totals"]["decisions"])
            del dataset
            del compact
            if low_memory_mode:
                gc.collect()
        if cancel_check and cancel_check():
            state["cancelled"] = True
            return state
        writer.finalize(create_indexes=True)
        state["performance"]["total_seconds"] = float(perf_counter() - start_total)
        return state
    except Exception:
        if partial_store_path.exists():
            try:
                partial_store_path.unlink()
            except OSError:
                pass
        raise
    finally:
        writer.close()


def _build_feature_store_chunks_parallel(
    *,
    request: dict[str, Any],
    db_path: Path,
    output_dir: Path,
    partial_store_path: Path,
    progress_callback: Callable[[int, int, str, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    preview_limit: int,
    chunk_size: int,
    workers: int,
    low_memory_mode: bool,
) -> dict[str, Any]:
    state = _initial_feature_build_state(request)
    total_symbols = len(request["symbols"])
    started = perf_counter()
    tmp_dir = output_dir / f"_tmp_feature_chunks_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[FeatureChunkBuildTask] = []
    for chunk_index, batch_symbols in chunk_symbols(request["symbols"], chunk_size):
        chunk_store = tmp_dir / f"chunk_{chunk_index:04d}" / "feature_chunk.duckdb"
        chunk_parts = tmp_dir / f"chunk_{chunk_index:04d}" / "parts"
        tasks.append(
            FeatureChunkBuildTask(
                chunk_index=chunk_index,
                db_path=str(db_path),
                chunk_store_path=str(chunk_store),
                symbols=tuple(batch_symbols),
                adjust=request["adjust"],
                start=request["start"],
                end=request["end"],
                config=request["config"],
                spec=request["spec"],
                preview_limit=preview_limit,
                low_memory_mode=low_memory_mode,
                chunk_parts_dir=str(chunk_parts),
                output_format="parquet" if request.get("feature_intermediate_format") == "parquet" else "duckdb",
                source_cache_dir=request.get("source_cache_dir"),
            )
        )
    completed_symbols = 0
    results = {}
    executor = ProcessPoolExecutor(max_workers=max(1, int(workers)))
    futures = {executor.submit(build_compact_feature_chunk, task): task for task in tasks}
    pending = set(futures)
    cancelled = False
    try:
        while pending:
            if cancel_check and cancel_check():
                cancelled = True
                break
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures[future]
                result = future.result()
                results[result.chunk_index] = result
                completed_symbols += len(task.symbols)
                _accumulate_compact_result(
                    state,
                    summary=result.summary,
                    symbols=result.symbols,
                    compiled_model_input=result.compiled_model_input,
                    preview_decisions=result.preview_decisions,
                    preview_decision_context=result.preview_decision_context,
                    preview_constraints=result.preview_constraints,
                    preview_limit=preview_limit,
                )
                perf = state["performance"]
                perf["mode"] = "parallel"
                perf["chunk_size"] = int(chunk_size)
                perf["workers"] = int(workers)
                perf["low_memory_mode"] = bool(low_memory_mode)
                perf["chunks_completed"] = int(perf.get("chunks_completed", 0)) + 1
                perf["worker_total_seconds_sum"] = float(perf.get("worker_total_seconds_sum", 0.0)) + float(result.timings.get("total_seconds", 0.0))
                perf["worker_build_seconds_sum"] = float(perf.get("worker_build_seconds_sum", 0.0)) + float(result.timings.get("build_seconds", 0.0))
                perf["worker_write_seconds_sum"] = float(perf.get("worker_write_seconds_sum", 0.0)) + float(result.timings.get("write_seconds", 0.0))
                if progress_callback:
                    label = task.symbols[0] if len(task.symbols) == 1 else f"{task.symbols[0]}..{task.symbols[-1]}"
                    progress_callback(min(completed_symbols, total_symbols), total_symbols, label, state["totals"]["decisions"])
        if cancelled:
            for future in pending:
                future.cancel()
            state["cancelled"] = True
            return state
        if len(results) != len(tasks):
            raise RuntimeError(f"Only {len(results)} of {len(tasks)} feature chunks completed.")
        start_merge = perf_counter()
        if request.get("feature_intermediate_format") == "parquet":
            ordered_part_roots = [results[index].chunk_parts_dir for index in sorted(results)]
            if not all(ordered_part_roots):
                raise RuntimeError("Feature parquet part output is incomplete; missing chunk part directory.")
            merge_compact_feature_parquet_parts(
                ordered_part_roots,
                partial_store_path,
                create_indexes=True,
                metadata_frame=_canonical_dataset_metadata_frame(request, state["compiled_model_input"]),
            )
        else:
            ordered_chunk_paths = [results[index].chunk_store_path for index in sorted(results)]
            merge_compact_feature_stores(ordered_chunk_paths, partial_store_path, create_indexes=True)
        state["performance"]["merge_seconds"] = float(perf_counter() - start_merge)
        state["performance"]["total_seconds"] = float(perf_counter() - started)
        return state
    except Exception:
        if partial_store_path.exists():
            try:
                partial_store_path.unlink()
            except OSError:
                pass
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)



def preflight_feature_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    request = _feature_request(payload, require_output_dir=False)
    db_path = request["db_path"]
    symbols = request["symbols"]
    spec = request["spec"]
    config = request["config"]
    warmup_bars = max(
        [20, *(indicator_lookback(item) for item in spec.indicators if item.enabled)]
    )
    checks: list[dict[str, Any]] = []

    def add_check(name: str, status: str, message: str, **extra: Any) -> None:
        checks.append({"name": name, "status": status, "message": message, **extra})

    blueprint = load_model_input_blueprint(spec=spec)
    blueprint_validation = validate_model_input_blueprint(blueprint, spec=spec)
    compiled_model_input = (
        compile_model_input_blueprint(blueprint, spec=spec)
        if blueprint_validation["valid"]
        else None
    )
    add_check(
        "model_input_blueprint",
        "pass" if blueprint_validation["valid"] else "error",
        (
            f"Model input schema v{compiled_model_input['schema_version']} is valid: "
            f"{compiled_model_input['schema_hash'][:12]}."
            if compiled_model_input
            else f"Model input blueprint has {len(blueprint_validation['errors'])} validation errors."
        ),
        schema_hash=compiled_model_input["schema_hash"] if compiled_model_input else None,
        shapes=compiled_model_input["shapes"] if compiled_model_input else {},
        decision_context_shape=compiled_model_input["decision_context_shape"] if compiled_model_input else [0],
        runtime_state_shape=compiled_model_input["runtime_state_shape"] if compiled_model_input else [0],
        errors=blueprint_validation["errors"],
    )

    market_available = _market_source_available(db_path)
    add_check(
        "market_source",
        "pass" if market_available else "error",
        f"Market source exists ({_market_source_label(db_path)})." if market_available else f"Market source not found: {_relative_or_absolute(db_path) or db_path}",
        path=_relative_or_absolute(db_path),
        source=_market_source_label(db_path) if market_available else None,
    )
    add_check(
        "symbols",
        "pass" if symbols else "error",
        f"{len(symbols)} symbols selected." if symbols else "No symbols selected. Provide manual symbols or a symbols file.",
        count=len(symbols),
    )

    shard_preflight = has_market_shard_storage(resolve_market_data_root(db_path))
    if symbols and market_available and not shard_preflight:
        return _preflight_feature_dataset_fast(
            request,
            warmup_bars=warmup_bars,
            checks=checks,
            add_check=add_check,
            compiled_model_input=compiled_model_input,
        )

    if symbols and market_available:
        base = load_kline_from_duckdb(
            db_path,
            symbols=symbols,
            freq=config.trade_freq,
            adjust=config.adjust,
            start=None,
            end=request["end"],
        )
        daily = load_kline_from_duckdb(
            db_path,
            symbols=symbols,
            freq="daily",
            adjust=config.adjust,
            start=None,
            end=request["end"],
        )
        status = load_stock_status_daily_from_duckdb(
            db_path,
            symbols=symbols,
            start=request["start"],
            end=request["end"],
        )
    else:
        base = pd.DataFrame()
        daily = pd.DataFrame()
        status = pd.DataFrame()

    status_coverage = evaluate_market_status_coverage(
        base,
        status,
        symbols=symbols,
        start=request["start"],
        end=request["end"],
    )
    missing_status_days = int(status_coverage["missing_days"])
    st_status = "error" if missing_status_days else "pass"
    add_check(
        "st_status_coverage",
        st_status,
        (
            f"ST status covers all {status_coverage['required_days']} symbol-days; "
            f"{status_coverage['st_days']} ST symbol-days will use the 5% limit and block new buys."
            if not missing_status_days
            else (
                f"Historical ST status is missing for {missing_status_days} of "
                f"{status_coverage['required_days']} symbol-days. In Download, use the same date range "
                "with Skip Existing enabled; existing K-lines will be skipped while daily extensions are backfilled."
            )
        ),
        **status_coverage,
    )

    add_check("base_rows", "pass" if len(base) else "error", f"{len(base)} {config.trade_freq} rows found.", rows=int(len(base)))
    add_check("daily_rows", "pass" if len(daily) else "warn", f"{len(daily)} official daily rows found.", rows=int(len(daily)))
    reference_coverage = _limit_reference_coverage(
        base,
        start=request["start"],
        end=request["end"],
    )
    add_check(
        "limit_reference_coverage",
        "pass",
        (
            f"All {reference_coverage['decision_sessions']} decision sessions have a previous trading-day close."
            if not reference_coverage["missing_sessions"]
            else (
                f"{reference_coverage['missing_sessions']} first-history sessions lack an external previous close; "
                "features use open/first-close seeded references instead of dropping these rows."
            )
        ),
        **reference_coverage,
    )

    warmup_by_frequency = evaluate_frequency_warmup(
        base,
        daily,
        frequencies=config.frequencies,
        trade_freq=config.trade_freq,
        start=request["start"],
        spec=spec,
        symbols=symbols,
    )
    for item in warmup_by_frequency:
        short_symbols = list(item["short_symbols"])
        source_missing = (
            base.empty
            if item["freq"] in {config.trade_freq, "15min", "30min", "60min"}
            else daily.empty
        )
        if symbols and source_missing:
            warmup_status = "error"
            message = f"No {item['freq']} bars are available for the selected symbols and date range."
        elif short_symbols:
            warmup_status = "warn"
            message = (
                f"{item['freq']} has fewer than {item['required_bars']} bars before the formal training start for "
                f"{len(short_symbols)} symbols; available bars will be used with coverage features."
                if request["start"]
                else f"{item['freq']} has incomplete maturity for {len(short_symbols)} symbols; available bars will be used with coverage features."
            )
        else:
            warmup_status = "pass"
            message = f"{item['freq']} has {item['required_bars']} pre-training warm-up bars for every symbol."
        add_check(
            f"warmup_{item['freq']}",
            warmup_status,
            message,
            required_bars=item["required_bars"],
            short_symbols=short_symbols[:20],
            available_bars=item["available_bars"],
            feature_requirements=item["feature_requirements"],
        )

    numeric_columns = [column for column in ("open", "high", "low", "close", "volume", "amount", "pctChg") if column in base.columns]
    nan_counts = {column: int(base[column].isna().sum()) for column in numeric_columns} if numeric_columns else {}
    blocking_nan_count = sum(value for column, value in nan_counts.items() if column != "pctChg")
    pct_chg_nan_count = nan_counts.get("pctChg", 0)
    if base.empty:
        add_check("nan_scan", "warn", "NaN scan was skipped because no base rows are available.", nan_counts={})
    else:
        add_check(
            "nan_scan",
            "pass" if blocking_nan_count == 0 else "warn",
            (
                f"Required numeric columns are complete; {pct_chg_nan_count} missing pctChg values will be recomputed from prices."
                if blocking_nan_count == 0 and pct_chg_nan_count
                else (
                    "No NaN values in base numeric columns."
                    if blocking_nan_count == 0
                    else "NaN values exist in required base numeric columns."
                )
            ),
            nan_counts=nan_counts,
        )

    try:
        decision_base = base[[column for column in ("symbol", "adjust", "datetime", "open", "close") if column in base]].copy()
        decision_base["datetime"] = pd.to_datetime(decision_base["datetime"], errors="coerce")
        decision_base = decision_base.dropna(subset=["datetime", "open", "close"])
        if request["start"]:
            decision_base = decision_base.loc[decision_base["datetime"] >= pd.Timestamp(request["start"])]
        decisions_estimate = int(reference_coverage["eligible_bar_rows"])
        if config.include_open_auction and not decision_base.empty:
            decisions_estimate += int(reference_coverage["eligible_sessions"])
        if config.max_decisions:
            decisions_estimate = min(decisions_estimate, int(config.max_decisions))
        add_check("decision_points", "pass" if decisions_estimate else "error", f"Estimated {decisions_estimate} decision rows.", decisions=decisions_estimate)
    except Exception as exc:
        decisions_estimate = 0
        add_check("decision_points", "error", f"Decision point generation failed: {exc}", decisions=0)

    expanded_market_rows_estimate = {
        freq: int(decisions_estimate * int(spec.sequence_windows.get(freq, 0)))
        for freq in config.frequencies
    }
    inventory_by_frequency = evaluate_frequency_warmup(
        base,
        daily,
        frequencies=config.frequencies,
        trade_freq=config.trade_freq,
        start=None,
        spec=spec,
        symbols=symbols,
    )
    unique_market_rows_estimate = {
        str(item["freq"]): int(sum(item["available_bars"].values()))
        for item in inventory_by_frequency
    }
    decision_index_rows = int(decisions_estimate * len(config.frequencies))
    snapshot_rows_estimate = int(
        decisions_estimate * sum(1 for freq in config.frequencies if freq != config.trade_freq)
    )
    estimated_mb = round(
        (
            sum(unique_market_rows_estimate.values()) * max(1, len(spec.market_feature_names)) * 8
            + decision_index_rows * 64
            + snapshot_rows_estimate * 96
        ) / 1024 / 1024,
        2,
    )
    add_check(
        "size_estimate",
        "pass",
        f"Compact store estimates {sum(unique_market_rows_estimate.values())} unique market rows, "
        f"{decision_index_rows} decision-index rows, and about {estimated_mb} MB before DuckDB compression.",
        market_rows=unique_market_rows_estimate,
        expanded_rows_avoided=int(sum(expanded_market_rows_estimate.values())),
        decision_index_rows=decision_index_rows,
        snapshot_rows=snapshot_rows_estimate,
        estimated_mb=estimated_mb,
    )
    add_check(
        "future_leakage",
        "pass",
        "Daily/weekly/monthly current-period states are built only from visible intraday bars; official current daily close is not exposed before close.",
    )

    ok = not any(item["status"] == "error" for item in checks)
    return {
        "ok": ok,
        "status": "pass" if ok and not any(item["status"] == "warn" for item in checks) else ("warn" if ok else "error"),
        "checks": checks,
        "symbols": symbols[:100],
        "symbol_count": len(symbols),
        "warmup_bars": warmup_bars,
        "warmup_by_frequency": warmup_by_frequency,
        "st_status_coverage": status_coverage,
        "limit_reference_coverage": reference_coverage,
        "decisions_estimate": decisions_estimate,
        "market_rows_estimate": unique_market_rows_estimate,
        "expanded_market_rows_avoided": int(sum(expanded_market_rows_estimate.values())),
        "decision_index_rows_estimate": decision_index_rows,
        "snapshot_rows_estimate": snapshot_rows_estimate,
        "estimated_numeric_mb": estimated_mb,
        "model_input": compiled_model_input,
    }


def _preflight_feature_dataset_fast(
    request: dict[str, Any],
    *,
    warmup_bars: int,
    checks: list[dict[str, Any]],
    add_check: Callable[..., None],
    compiled_model_input: dict[str, Any] | None,
) -> dict[str, Any]:
    db_path = request["db_path"]
    symbols = request["symbols"]
    spec = request["spec"]
    config = request["config"]
    minutes = {"5min": 5, "15min": 15, "30min": 30, "60min": 60}

    with connect_duckdb(db_path) as conn:
        conn.register("_feature_symbols", pd.DataFrame({"symbol": symbols}))
        tables = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}

        def count_rows(freq: str) -> int:
            return int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {KLINE_TABLE_NAME} k JOIN _feature_symbols s USING(symbol) "
                    "WHERE k.freq = ? AND k.adjust = ? "
                    "AND (? IS NULL OR CAST(k.datetime AS DATE) <= CAST(? AS DATE))",
                    [freq, config.adjust, request["end"], request["end"]],
                ).fetchone()[0]
            )

        def stored_counts(freq: str, *, before_start: bool) -> dict[str, int]:
            rows = conn.execute(
                f"SELECT k.symbol, COUNT(*) FROM {KLINE_TABLE_NAME} k "
                "JOIN _feature_symbols s USING(symbol) "
                "WHERE k.freq = ? AND k.adjust = ? "
                "AND (? IS NULL OR CAST(k.datetime AS DATE) <= CAST(? AS DATE)) "
                "AND (? IS NULL OR k.datetime < CAST(? AS TIMESTAMP)) "
                "GROUP BY k.symbol",
                [
                    freq,
                    config.adjust,
                    request["end"],
                    request["end"],
                    request["start"] if before_start else None,
                    request["start"] if before_start else None,
                ],
            ).fetchall()
            return {str(symbol): int(count) for symbol, count in rows}

        def derived_counts(freq: str, *, before_start: bool) -> dict[str, int]:
            direct = stored_counts(freq, before_start=before_start)
            if sum(direct.values()) > 0:
                return {symbol: int(direct.get(symbol, 0)) for symbol in symbols}
            cutoff = request["start"] if before_start else None
            if freq in minutes and config.trade_freq in minutes:
                source_rows = max(1, minutes[freq] // minutes[config.trade_freq])
                rows = conn.execute(
                    f"WITH sessions AS ("
                    f" SELECT k.symbol, CAST(k.datetime AS DATE) AS session_date, COUNT(*) AS rows"
                    f" FROM {KLINE_TABLE_NAME} k JOIN _feature_symbols s USING(symbol)"
                    " WHERE k.freq = ? AND k.adjust = ?"
                    " AND (? IS NULL OR CAST(k.datetime AS DATE) <= CAST(? AS DATE))"
                    " AND (? IS NULL OR k.datetime < CAST(? AS TIMESTAMP))"
                    " GROUP BY k.symbol, session_date)"
                    " SELECT symbol, CAST(SUM(CEIL(rows::DOUBLE / ?)) AS BIGINT) FROM sessions GROUP BY symbol",
                    [
                        config.trade_freq,
                        config.adjust,
                        request["end"],
                        request["end"],
                        cutoff,
                        cutoff,
                        source_rows,
                    ],
                ).fetchall()
            elif freq == "weekly":
                rows = conn.execute(
                    f"SELECT k.symbol, COUNT(DISTINCT DATE_TRUNC('week', k.datetime)) "
                    f"FROM {KLINE_TABLE_NAME} k JOIN _feature_symbols s USING(symbol) "
                    "WHERE k.freq = 'daily' AND k.adjust = ? "
                    "AND (? IS NULL OR CAST(k.datetime AS DATE) <= CAST(? AS DATE)) "
                    "AND (? IS NULL OR k.datetime < CAST(? AS TIMESTAMP)) GROUP BY k.symbol",
                    [config.adjust, request["end"], request["end"], cutoff, cutoff],
                ).fetchall()
            else:
                rows = []
            values = {str(symbol): int(count) for symbol, count in rows}
            return {symbol: int(values.get(symbol, 0)) for symbol in symbols}

        base_rows = count_rows(config.trade_freq)
        daily_rows = count_rows("daily")
        add_check("base_rows", "pass" if base_rows else "error", f"{base_rows} {config.trade_freq} rows found.", rows=base_rows)
        add_check("daily_rows", "pass" if daily_rows else "warn", f"{daily_rows} official daily rows found.", rows=daily_rows)

        reference = conn.execute(
            f"WITH session_counts AS ("
            f" SELECT k.symbol, k.adjust, CAST(k.datetime AS DATE) AS session_date, COUNT(*) AS bar_rows"
            f" FROM {KLINE_TABLE_NAME} k JOIN _feature_symbols s USING(symbol)"
            " WHERE k.freq = ? AND k.adjust = ?"
            " AND (? IS NULL OR CAST(k.datetime AS DATE) <= CAST(? AS DATE))"
            " GROUP BY k.symbol, k.adjust, session_date),"
            " ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, adjust ORDER BY session_date) AS rn FROM session_counts)"
            " SELECT COUNT(*), COALESCE(SUM(CASE WHEN rn > 1 THEN 1 ELSE 0 END), 0),"
            " COALESCE(SUM(CASE WHEN rn = 1 THEN 1 ELSE 0 END), 0),"
            " COALESCE(SUM(CASE WHEN rn > 1 THEN bar_rows ELSE 0 END), 0) FROM ranked"
            " WHERE (? IS NULL OR session_date >= CAST(? AS DATE))"
            " AND (? IS NULL OR session_date <= CAST(? AS DATE))",
            [
                config.trade_freq,
                config.adjust,
                request["end"],
                request["end"],
                request["start"],
                request["start"],
                request["end"],
                request["end"],
            ],
        ).fetchone()
        reference_coverage = {
            "decision_sessions": int(reference[0]),
            "eligible_sessions": int(reference[1]),
            "missing_sessions": int(reference[2]),
            "eligible_bar_rows": int(reference[3]),
        }
        add_check(
            "limit_reference_coverage",
            "pass",
            (
                f"All {reference_coverage['decision_sessions']} decision sessions have a previous trading-day close."
                if not reference_coverage["missing_sessions"]
                else (
                    f"{reference_coverage['missing_sessions']} first-history sessions lack an external previous close; "
                    "features use open/first-close seeded references instead of dropping these rows."
                )
            ),
            **reference_coverage,
        )

        status_join = (
            f"LEFT JOIN (SELECT symbol, date, BOOL_OR(is_st) AS is_st FROM {STOCK_STATUS_DAILY_TABLE_NAME} GROUP BY symbol, date) st "
            "ON st.symbol = sessions.symbol AND st.date = sessions.session_date"
            if STOCK_STATUS_DAILY_TABLE_NAME in tables
            else "LEFT JOIN (SELECT NULL::VARCHAR AS symbol, NULL::DATE AS date, NULL::BOOLEAN AS is_st) st ON FALSE"
        )
        status_cte = (
            f"WITH sessions AS (SELECT DISTINCT k.symbol, CAST(k.datetime AS DATE) AS session_date "
            f"FROM {KLINE_TABLE_NAME} k JOIN _feature_symbols s USING(symbol) "
            "WHERE k.freq = ? AND k.adjust = ? "
            "AND (? IS NULL OR CAST(k.datetime AS DATE) >= CAST(? AS DATE)) "
            "AND (? IS NULL OR CAST(k.datetime AS DATE) <= CAST(? AS DATE))) "
        )
        coverage_row = conn.execute(
            status_cte
            + "SELECT COUNT(*), COUNT(st.date), COALESCE(SUM(CASE WHEN st.is_st THEN 1 ELSE 0 END), 0) "
            + "FROM sessions " + status_join,
            [config.trade_freq, config.adjust, request["start"], request["start"], request["end"], request["end"]],
        ).fetchone()
        missing_rows = conn.execute(
            status_cte
            + "SELECT sessions.symbol, sessions.session_date FROM sessions " + status_join
            + " WHERE st.date IS NULL ORDER BY sessions.symbol, sessions.session_date LIMIT 20",
            [config.trade_freq, config.adjust, request["start"], request["start"], request["end"], request["end"]],
        ).fetchall()
        st_symbol_rows = conn.execute(
            status_cte
            + "SELECT DISTINCT sessions.symbol FROM sessions " + status_join
            + " WHERE st.is_st ORDER BY sessions.symbol",
            [config.trade_freq, config.adjust, request["start"], request["start"], request["end"], request["end"]],
        ).fetchall()
        missing_symbols = sorted({str(symbol) for symbol, _ in missing_rows})
        status_coverage = {
            "required_days": int(coverage_row[0]),
            "covered_days": int(coverage_row[1]),
            "missing_days": int(coverage_row[0] - coverage_row[1]),
            "missing_symbols": missing_symbols,
            "missing_examples": [
                {"symbol": str(symbol), "date": str(date)} for symbol, date in missing_rows
            ],
            "st_days": int(coverage_row[2]),
            "st_symbols": [str(row[0]) for row in st_symbol_rows],
        }
        missing_status_days = status_coverage["missing_days"]
        add_check(
            "st_status_coverage",
            "error" if missing_status_days else "pass",
            (
                f"ST status covers all {status_coverage['required_days']} symbol-days; "
                f"{status_coverage['st_days']} ST symbol-days will use the 5% limit and block new buys."
                if not missing_status_days
                else f"Historical ST status is missing for {missing_status_days} of {status_coverage['required_days']} symbol-days. "
                "In Download, use the same date range with Skip Existing enabled; existing K-lines will be skipped while daily extensions are backfilled."
            ),
            **status_coverage,
        )

        warmup_by_frequency: list[dict[str, Any]] = []
        inventory_by_frequency: dict[str, dict[str, int]] = {}
        for freq in config.frequencies:
            feature_requirements = {"returns_and_rolling": 20}
            for indicator in spec.indicators:
                if indicator.enabled and freq in indicator.frequencies:
                    feature_requirements[indicator.id] = indicator_lookback(indicator)
            required = max(feature_requirements.values())
            available = derived_counts(freq, before_start=True)
            inventory = derived_counts(freq, before_start=False)
            inventory_by_frequency[freq] = inventory
            short_symbols = [symbol for symbol in symbols if int(available.get(symbol, 0)) < required]
            item = {
                "freq": freq,
                "required_bars": required,
                "available_bars": available,
                "short_symbols": short_symbols,
                "feature_requirements": feature_requirements,
            }
            warmup_by_frequency.append(item)
            source_missing = sum(inventory.values()) == 0
            if source_missing:
                warmup_status = "error"
                message = f"No {freq} bars are available for the selected symbols and date range."
            elif short_symbols:
                warmup_status = "warn"
                message = (
                    f"{freq} has fewer than {required} bars before the formal training start for {len(short_symbols)} symbols; "
                    "available bars will be used with coverage features."
                )
            else:
                warmup_status = "pass"
                message = f"{freq} has {required} pre-training warm-up bars for every symbol."
            add_check(
                f"warmup_{freq}", warmup_status, message,
                required_bars=required, short_symbols=short_symbols[:20],
                available_bars=available, feature_requirements=feature_requirements,
            )

        nan_row = conn.execute(
            f"SELECT " + ", ".join(
                f"COALESCE(SUM(CASE WHEN {column} IS NULL OR NOT ISFINITE({column}) THEN 1 ELSE 0 END), 0)"
                for column in ("open", "high", "low", "close", "volume", "amount", "pctChg")
            )
            + f" FROM {KLINE_TABLE_NAME} k JOIN _feature_symbols s USING(symbol) "
            "WHERE k.freq = ? AND k.adjust = ? AND (? IS NULL OR CAST(k.datetime AS DATE) <= CAST(? AS DATE))",
            [config.trade_freq, config.adjust, request["end"], request["end"]],
        ).fetchone()
        nan_counts = dict(zip(("open", "high", "low", "close", "volume", "amount", "pctChg"), map(int, nan_row)))
        blocking_nan_count = sum(value for column, value in nan_counts.items() if column != "pctChg")
        pct_chg_nan_count = nan_counts["pctChg"]
        add_check(
            "nan_scan",
            "pass" if blocking_nan_count == 0 else "warn",
            (
                f"Required numeric columns are complete; {pct_chg_nan_count} missing pctChg values will be recomputed from prices."
                if blocking_nan_count == 0 and pct_chg_nan_count
                else ("No NaN values in base numeric columns." if blocking_nan_count == 0 else "NaN values exist in required base numeric columns.")
            ),
            nan_counts=nan_counts,
        )

    decisions_estimate = reference_coverage["eligible_bar_rows"]
    if config.include_open_auction:
        decisions_estimate += reference_coverage["eligible_sessions"]
    if config.max_decisions:
        decisions_estimate = min(decisions_estimate, int(config.max_decisions))
    add_check("decision_points", "pass" if decisions_estimate else "error", f"Estimated {decisions_estimate} decision rows.", decisions=decisions_estimate)
    unique_market_rows_estimate = {
        freq: int(sum(counts.values())) for freq, counts in inventory_by_frequency.items()
    }
    expanded_market_rows_estimate = {
        freq: int(decisions_estimate * int(spec.sequence_windows.get(freq, 0)))
        for freq in config.frequencies
    }
    decision_index_rows = int(decisions_estimate * len(config.frequencies))
    estimated_mb = round(
        (sum(unique_market_rows_estimate.values()) * max(1, len(spec.market_feature_names)) * 8 + decision_index_rows * 64)
        / 1024 / 1024,
        2,
    )
    add_check(
        "size_estimate", "pass",
        f"Compact store estimates {sum(unique_market_rows_estimate.values())} unique market rows, "
        f"{decision_index_rows} decision-index rows, and about {estimated_mb} MB before DuckDB compression.",
        market_rows=unique_market_rows_estimate,
        expanded_rows_avoided=int(sum(expanded_market_rows_estimate.values())),
        decision_index_rows=decision_index_rows,
        snapshot_rows=0,
        estimated_mb=estimated_mb,
    )
    add_check(
        "future_leakage", "pass",
        "Daily/weekly/monthly current-period states are built only from visible intraday bars; official current daily close is not exposed before close.",
    )
    ok = not any(item["status"] == "error" for item in checks)
    return {
        "ok": ok,
        "status": "pass" if ok and not any(item["status"] == "warn" for item in checks) else ("warn" if ok else "error"),
        "checks": checks,
        "symbols": symbols[:100],
        "symbol_count": len(symbols),
        "warmup_bars": warmup_bars,
        "warmup_by_frequency": warmup_by_frequency,
        "st_status_coverage": status_coverage,
        "limit_reference_coverage": reference_coverage,
        "decisions_estimate": decisions_estimate,
        "market_rows_estimate": unique_market_rows_estimate,
        "expanded_market_rows_avoided": int(sum(expanded_market_rows_estimate.values())),
        "decision_index_rows_estimate": decision_index_rows,
        "snapshot_rows_estimate": 0,
        "estimated_numeric_mb": estimated_mb,
        "model_input": compiled_model_input,
    }


def preview_feature_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a small, real decision sample without running a full feature build.

    Earlier versions reused ``build_feature_dataset_from_duckdb`` with the full
    selected universe and then displayed ``head(5)``.  That made Preview almost
    as heavy as Build: it loaded every selected symbol before showing five rows.
    The preview endpoint must stay synchronous and lightweight, so it now probes
    a bounded number of symbols one by one and stops as soon as it has enough
    real decisions.
    """

    max_preview_decisions = min(_optional_positive_int(payload.get("max_decisions")) or 5, 5)
    request = _feature_request({**payload, "max_decisions": max_preview_decisions}, require_output_dir=False)
    symbols = [str(symbol).upper() for symbol in request["symbols"]]
    if not symbols:
        raise ValueError("At least one symbol is required to preview a feature dataset.")
    if not _market_source_available(request["db_path"]):
        raise FileNotFoundError(f"Market source not found: {_market_source_label(request['db_path'])}")

    started = perf_counter()
    decisions_parts: list[pd.DataFrame] = []
    available_market_rows: dict[str, dict[str, int]] = {}
    sampled_symbols: list[str] = []
    skipped_symbols: list[str] = []
    errors: list[str] = []
    remaining = max_preview_decisions

    # Bound the probe so a broken first-page universe cannot make Preview scan a
    # full 5,000-symbol list.  Build still uses the complete requested universe.
    probe_symbols = symbols[: min(len(symbols), max(20, max_preview_decisions * 4))]
    for symbol in probe_symbols:
        if remaining <= 0:
            break
        sampled_symbols.append(symbol)
        symbol_config = replace(request["config"], max_decisions=remaining, materialize_sequences=True)
        try:
            dataset = build_feature_dataset_from_duckdb(
                request["db_path"],
                symbols=[symbol],
                adjust=request["adjust"],
                start=request["start"],
                end=request["end"],
                config=symbol_config,
                spec=request["spec"],
            )
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
            continue

        if dataset.decisions.empty:
            skipped_symbols.append(symbol)
            continue

        decisions = dataset.decisions.head(remaining).copy()
        decisions_parts.append(decisions)
        for decision_id in decisions["decision_id"].tolist() if "decision_id" in decisions else []:
            available_market_rows[str(decision_id)] = {
                freq: int(frame.loc[frame["decision_id"].eq(decision_id)].shape[0]) if not frame.empty else 0
                for freq, frame in dataset.market.items()
            }
        remaining = max_preview_decisions - sum(len(part) for part in decisions_parts)

    preview_decisions = (
        pd.concat(decisions_parts, ignore_index=True).head(max_preview_decisions)
        if decisions_parts
        else pd.DataFrame()
    )
    elapsed = round(float(perf_counter() - started), 3)

    if preview_decisions.empty and errors:
        examples = "; ".join(errors[:5])
        raise ValueError(
            "Preview could not produce decision rows from the sampled symbols. "
            f"First errors: {examples}"
        )

    issue_note = ""
    if errors:
        issue_note = f" {len(errors)} sampled symbol(s) raised errors; first: {errors[0]}"
    elif skipped_symbols:
        issue_note = f" {len(skipped_symbols)} sampled symbol(s) had no decisions in range."

    return {
        "decisions": dataframe_to_records(preview_decisions),
        "available_market_rows": available_market_rows,
        "sampled_symbols": sampled_symbols,
        "skipped_symbols": skipped_symbols[:20],
        "errors": errors[:20],
        "seconds": elapsed,
        "note": (
            "Lightweight preview sampled at most 20 symbols and stopped after "
            f"{max_preview_decisions} real decisions; no feature_parts or feature_store files were written. "
            "Preview uses only visible completed bars. Official current daily close is not exposed before the daily bar is complete."
            f"{issue_note}"
        ),
    }


def start_feature_dataset_job(payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    active = [
        job for job in jobs.list_jobs()
        if job.get("type") == "feature_dataset" and job.get("status") in {"queued", "running"}
    ]
    if active:
        raise ValueError(f"A feature dataset build is already active: {active[-1]['job_id']}")
    job_id = jobs.create_job("feature_dataset", title="Build feature dataset")
    jobs.update_job(job_id, total=0, message="Queued feature dataset build")
    jobs.submit(job_id, _run_feature_dataset_job, job_id, payload, jobs)
    return {"job_id": job_id, "status": "queued"}


def _run_feature_dataset_job(job_id: str, payload: dict[str, Any], jobs: Any) -> dict[str, Any]:
    output_raw = payload.get("output_dir")
    if not output_raw:
        payload = {**payload, "output_dir": f"{DEFAULT_FEATURE_OUTPUT_DIR}/{job_id}"}

    requested = _feature_request(payload, require_output_dir=False)
    symbol_total = len(requested["symbols"])
    jobs.update_job(
        job_id,
        status="running",
        current="preparing",
        completed=0,
        total=symbol_total,
        progress=0.05,
        message="Building feature dataset",
    )
    add_log = getattr(jobs, "add_log", None)
    if callable(add_log):
        add_log(
            job_id,
            (
                f"CONFIG db={payload.get('db_path') or DEFAULT_DB_PATH}, "
                f"adjust={payload.get('adjust') or 'pre'}, trade_freq={payload.get('trade_freq') or '5min'}, "
                f"freqs={payload.get('frequencies') or payload.get('freqs') or '5min,30min,daily,weekly'}"
            ),
        )

    def update_progress(completed: int, total: int, symbol: str, decisions: int) -> None:
        progress = 0.05 + (0.85 * completed / max(1, total))
        jobs.update_job(
            job_id,
            current=symbol,
            completed=completed,
            total=total,
            saved_rows=decisions,
            progress=progress,
            message=f"Built {completed}/{total} symbols",
        )
        if callable(add_log):
            add_log(job_id, f"OK {symbol}: {completed}/{total}, decisions={decisions}")

    result = build_feature_dataset(
        payload,
        progress_callback=update_progress,
        cancel_check=lambda: jobs.is_cancel_requested(job_id),
    )
    if result.get("cancelled"):
        jobs.update_job(
            job_id,
            status="cancelled",
            message="Feature dataset build cancelled safely",
            saved_rows=int(result.get("decisions") or 0),
        )
        if callable(add_log):
            add_log(job_id, "CANCELLED feature dataset build; the previous completed store was kept.", level="warn")
        return result
    saved_rows = int(result.get("decisions") or 0)
    jobs.update_job(
        job_id,
        completed=1,
        succeeded=1,
        failed=0,
        saved_rows=saved_rows,
        progress=0.95,
        message="Feature dataset built",
    )
    if callable(add_log):
        add_log(job_id, f"OK feature dataset: decisions={saved_rows}, output={result.get('output_dir')}")
    return result


def _write_csv(frame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return _relative_or_absolute(path) or str(path)


def _cleanup_legacy_feature_outputs(output_dir: Path) -> None:
    """Remove only obsolete Feature Layer files from a reused output directory."""
    legacy_names = {
        "decisions.csv",
        "decision_context.csv",
        "constraints.csv",
        "portfolio.csv",
        "feature_store.duckdb",
        "feature_store.partial.duckdb",
    }
    candidates = [output_dir / name for name in legacy_names]
    candidates.extend(output_dir.glob("market_*.csv"))
    for path in candidates:
        if path.is_file():
            path.unlink()


def _write_json(payload: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return _relative_or_absolute(path) or str(path)


def _feature_request(payload: dict[str, Any], *, require_output_dir: bool = True) -> dict[str, Any]:
    db_path = resolve_project_path(payload.get("db_path"), DEFAULT_DB_PATH)
    output_dir = resolve_project_path(payload.get("output_dir"), DEFAULT_FEATURE_OUTPUT_DIR) if require_output_dir else None
    symbols = _resolve_symbols(payload)
    adjust = str(payload.get("adjust") or "pre")
    trade_freq = str(payload.get("trade_freq") or "5min")
    frequencies = _normalize_csv_values(payload.get("frequencies") or payload.get("freqs")) or list(DEFAULT_DATASET_FREQUENCIES)
    allowed_frequencies = {DEFAULT_FEATURE_SPEC.base_frequency, *DEFAULT_FEATURE_SPEC.derived_frequencies}
    unsupported_frequencies = sorted(set(frequencies).difference(allowed_frequencies))
    if unsupported_frequencies:
        raise ValueError(
            f"Unsupported feature frequencies: {', '.join(unsupported_frequencies)}. "
            f"Available: {', '.join(sorted(allowed_frequencies))}."
        )
    include_open_auction = _as_bool(payload.get("include_open_auction"), default=True)
    sequence_windows = _sequence_windows(payload.get("sequence_windows"))
    active_spec = active_feature_spec()
    spec = replace(active_spec, sequence_windows={**active_spec.sequence_windows, **sequence_windows})
    config = FeatureDatasetConfig(
        trade_freq=trade_freq,
        frequencies=tuple(frequencies),
        adjust=adjust,
        include_open_auction=include_open_auction,
        max_decisions=_optional_positive_int(payload.get("max_decisions")),
    )
    feature_build_chunk_size = _bounded_positive_int(
        payload.get("feature_build_chunk_size", payload.get("chunk_size")),
        default=16,
        minimum=1,
        maximum=256,
    )
    feature_build_workers = _bounded_positive_int(
        payload.get("feature_build_workers", payload.get("build_workers")),
        default=1,
        minimum=1,
        maximum=64,
    )
    feature_low_memory = _as_bool(payload.get("feature_low_memory"), default=True)
    market_parquet_cache_enabled = _as_bool(payload.get("market_parquet_cache_enabled"), default=True)
    market_parquet_cache_root = resolve_project_path(
        payload.get("market_parquet_cache_root") or "runtime_layer/data/market_parquet_cache"
    )
    market_parquet_cache_force = _as_bool(payload.get("market_parquet_cache_force"), default=False)
    # The canonical Feature Dataset output is symbol-part parquet plus metadata.
    # Ignore stale frontend/API flags that request a compact feature_store.duckdb export.
    feature_incremental_enabled = True
    feature_force_rebuild_parts = _as_bool(payload.get("feature_force_rebuild_parts"), default=False)
    feature_materialize_store = False
    feature_intermediate_format = "parquet"
    return {
        "db_path": db_path,
        "output_dir": output_dir,
        "symbols": symbols,
        "adjust": adjust,
        "trade_freq": trade_freq,
        "frequencies": frequencies,
        "include_open_auction": include_open_auction,
        "sequence_windows": spec.sequence_windows,
        "start": payload.get("start") or None,
        "end": payload.get("end") or None,
        "spec": spec,
        "config": config,
        "feature_build_chunk_size": feature_build_chunk_size,
        "feature_build_workers": feature_build_workers,
        "feature_low_memory": feature_low_memory,
        "market_parquet_cache_enabled": market_parquet_cache_enabled,
        "market_parquet_cache_root": market_parquet_cache_root,
        "market_parquet_cache_force": market_parquet_cache_force,
        "feature_incremental_enabled": feature_incremental_enabled,
        "feature_force_rebuild_parts": feature_force_rebuild_parts,
        "feature_materialize_store": feature_materialize_store,
        "feature_intermediate_format": feature_intermediate_format,
    }


def _quality_report(
    dataset: Any,
    *,
    spec: Any,
    decision_index: pd.DataFrame | None = None,
) -> dict[str, Any]:
    frame_reports: dict[str, Any] = {
        "decisions": _frame_quality(dataset.decisions),
        "decision_context": _frame_quality(dataset.decision_context, clip_fields=spec.context_fields),
        "constraints": _frame_quality(dataset.constraints, clip_fields=spec.constraint_fields),
    }
    market_reports: dict[str, Any] = {}
    missing_sequence_count = 0
    for freq, frame in dataset.market.items():
        market_reports[freq] = _frame_quality(frame, clip_fields=spec.market_fields)
        expected = int(spec.sequence_windows.get(freq, 0))
        if expected > 0 and not frame.empty and "decision_id" in frame:
            sizes = frame.groupby("decision_id").size()
            missing_sequence_count += int((sizes < expected).sum())
    if decision_index is not None and not decision_index.empty:
        missing_sequence_count = int(
            (
                pd.to_numeric(decision_index["valid_rows"], errors="coerce")
                < pd.to_numeric(decision_index["sequence_window"], errors="coerce")
            ).sum()
        )
    return {
        "spec": spec.name,
        "version": spec.version,
        "rows": {
            "decisions": int(len(dataset.decisions)),
            "decision_context": int(len(dataset.decision_context)),
            "constraints": int(len(dataset.constraints)),
            "market": {freq: int(len(frame)) for freq, frame in dataset.market.items()},
            "decision_index": int(len(decision_index)) if decision_index is not None else 0,
        },
        "frames": frame_reports,
        "market": market_reports,
        "missing_sequence_count": int(missing_sequence_count),
        "warmup_dropped_decisions": 0,
        "market_rule_audit": {
            "st_decisions": int(dataset.decisions.get("is_st", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "unknown_status_decisions": int((~dataset.decisions.get("status_known", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if not dataset.decisions.empty else 0,
            "missing_limit_reference_decisions": int((~dataset.decisions.get("has_limit_reference", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if not dataset.decisions.empty else 0,
        },
    }


def _streaming_quality_report(
    dataset: _FeatureBuildSummary,
    *,
    spec: Any,
) -> dict[str, Any]:
    return {
        "spec": spec.name,
        "version": spec.version,
        "rows": {
            "decisions": dataset.decisions,
            "decision_context": dataset.decision_context_rows,
            "constraints": dataset.constraint_rows,
            "market": dataset.market_rows,
            "decision_index": dataset.decision_index_rows,
        },
        "frames": {},
        "market": {},
        "missing_sequence_count": None,
        "warmup_dropped_decisions": 0,
        "market_rule_audit": {
            "st_decisions": dataset.st_decisions,
            "unknown_status_decisions": None,
            "missing_limit_reference_decisions": None,
        },
        "note": "Large builds are written symbol by symbol; detailed row diagnostics remain available through Preflight and preview files.",
    }


def _frame_quality(frame: Any, *, clip_fields: tuple[Any, ...] = ()) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"rows": 0, "nan_counts": {}, "inf_counts": {}, "clipped_counts": {}}
    numeric = frame.select_dtypes(include=["number"])
    nan_counts = {column: int(numeric[column].isna().sum()) for column in numeric.columns}
    inf_counts = {
        column: int((numeric[column] == float("inf")).sum() + (numeric[column] == float("-inf")).sum())
        for column in numeric.columns
    }
    clipped_counts: dict[str, int] = {}
    for field in clip_fields:
        if not field.clip or field.name not in numeric:
            continue
        lower, upper = field.clip
        values = numeric[field.name]
        clipped_counts[field.name] = int((values.eq(lower) | values.eq(upper)).sum())
    return {
        "rows": int(len(frame)),
        "nan_counts": {key: value for key, value in nan_counts.items() if value},
        "inf_counts": {key: value for key, value in inf_counts.items() if value},
        "clipped_counts": {key: value for key, value in clipped_counts.items() if value},
    }


def _dataset_manifest(
    dataset: Any,
    *,
    request: dict[str, Any],
    outputs: dict[str, str],
    output_dir: Path,
    model_input: dict[str, Any],
) -> dict[str, Any]:
    spec = request["spec"]
    return {
        "spec": {"name": spec.name, "version": spec.version},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "db_path": _relative_or_absolute(request["db_path"]),
        "output_dir": _relative_or_absolute(output_dir),
        "symbols": request["symbols"],
        "symbol_count": len(request["symbols"]),
        "feature_build_chunk_size": request.get("feature_build_chunk_size"),
        "feature_build_workers": request.get("feature_build_workers"),
        "feature_low_memory": request.get("feature_low_memory"),
        "market_source": request.get("market_source", "duckdb"),
        "market_parquet_cache_enabled": request.get("market_parquet_cache_enabled"),
        "market_parquet_cache": request.get("market_cache_info"),
        "feature_intermediate_format": request.get("feature_intermediate_format"),
        "feature_incremental_enabled": request.get("feature_incremental_enabled"),
        "feature_force_rebuild_parts": request.get("feature_force_rebuild_parts"),
        "feature_output_mode": "feature_parts",
        "feature_parts_manifest": request.get("feature_parts_manifest"),
        "feature_parts_reuse": request.get("feature_parts_reuse"),
        "adjust": request["adjust"],
        "trade_freq": request["trade_freq"],
        "enabled_frequencies": list(dataset.frequencies),
        "sequence_windows": spec.sequence_windows,
        "date_range": {"start": request["start"], "end": request["end"]},
        "feature_columns": {
            "market": list(spec.market_feature_names),
            "decision_context": list(spec.context_feature_names),
            "market_constraints": list(spec.constraint_feature_names),
            "portfolio_contract": list(spec.portfolio_feature_names),
            "environment_dynamic": list(spec.environment_feature_names),
        },
        "model_input": model_input,
        "market_rule_policy": {
            "st": "dated is_st overrides the board limit to 5% and blocks new buys without removing symbol history",
            "limit_reference": "previous trading-day close, fixed throughout the intraday session",
            "is_st_model_feature": False,
            "requested_symbols": list(dataset.requested_symbols),
        },
        "clip_rules": {
            field.name: field.clip
            for field in (*spec.market_fields, *spec.context_fields, *spec.constraint_fields, *spec.portfolio_fields, *spec.environment_fields)
            if field.clip is not None
        },
        "summary": dataset.summary(),
        "outputs": {
            key: {"path": value, "sha256": _hash_file(resolve_project_path(value))}
            for key, value in outputs.items()
            if value and str(value).endswith((".csv", ".json", ".duckdb"))
        },
    }


def _hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _field_payload(field: Any, formula: str) -> dict[str, Any]:
    return {
        "name": field.name,
        "group": field.group,
        "description": field.description,
        "clip": field.clip,
        "formula": formula,
    }


def _indicator_details(spec: Any) -> list[dict[str, Any]]:
    label_by_kind = {
        "macd": "MACD",
        "kd": "KD/KDJ core",
        "efi": "EFI",
        "ema_channel": "EMA channel",
    }
    details = []
    for indicator in spec.indicators:
        outputs = [
            field.name
            for field in spec.market_fields
            if (
                field.name.startswith(f"{indicator.id}_")
                or field.name.startswith(f"ema_channel_{indicator.id}_")
            )
        ]
        details.append(
            {
                "name": label_by_kind.get(indicator.kind, indicator.kind),
                "id": indicator.id,
                "kind": indicator.kind,
                "included": indicator.enabled,
                "frequencies": list(indicator.frequencies),
                "outputs": outputs,
                "formula": ", ".join(f"{key}={value}" for key, value in indicator.params.items()),
            }
        )
    details.extend(
        [
            {"name": "Data maturity", "included": True, "outputs": ["history_coverage", "sequence_valid_ratio", "sequence_mask at batch time"], "formula": "Insufficient history is used as-is; coverage values and a structural mask identify valid data."},
            {"name": "RSI", "included": False, "outputs": [], "formula": "Not configured in feature_v1."},
        ]
    )
    return details


def _formula_for_field(name: str) -> str:
    formulas = {
        "pctChg": "close / previous_close - 1; first visible row uses close / open - 1.",
        "log_ret": "log(close / previous_close).",
        "open_close_ret": "close / open - 1.",
        "gap_ret": "open / previous_close - 1.",
        "high_low_range": "(high - low) / previous_close.",
        "body_range": "abs(close - open) / previous_close.",
        "close_position": "(close - low) / (high - low).",
        "upper_shadow": "(high - max(open, close)) / previous_close.",
        "lower_shadow": "(min(open, close) - low) / previous_close.",
        "ret_3": "close / close.shift(3) - 1.",
        "ret_5": "close / close.shift(5) - 1.",
        "ret_10": "close / close.shift(10) - 1.",
        "ret_20": "close / close.shift(20) - 1.",
        "volatility_5": "rolling std of pctChg over 5 bars.",
        "volatility_20": "rolling std of pctChg over 20 bars.",
        "range_mean_20": "rolling mean of high_low_range over 20 bars.",
        "volume_ratio_20": "volume / prior 20-bar average volume.",
        "amount_ratio_20": "amount / prior 20-bar average amount.",
        "turn_lag1": "previous visible daily turnover.",
        "turn_ma20": "prior 20-day average turnover.",
        "turn_z20": "(turn_lag1 - turn_ma20) / prior 20-day turnover std.",
        "macd_dif_pct": "(EMA12(close) - EMA26(close)) / close.",
        "macd_dea_pct": "EMA9(DIF) / close.",
        "macd_hist_pct": "(DIF - DEA) / close.",
        "kd_k_norm": "K smoothing of RSV(9), scaled 0-1.",
        "kd_d_norm": "D smoothing of K, scaled 0-1.",
        "kd_diff_norm": "K - D.",
        "efi2_norm": "EMA2((close - REF(close, 1)) * volume) / prior 20-bar average abs force.",
        "efi13_norm": "EMA13((close - REF(close, 1)) * volume) / prior 20-bar average abs force.",
        "history_coverage": "min(available bars / longest active indicator lookback, 1).",
        "sequence_valid_ratio": "min(real sequence rows / configured sequence window, 1).",
        "progress": "current period source_rows / expected_source_rows, clipped to 1.",
        "bar_slot_norm": "completed intraday slot index normalized to 0..1 inside the trading day.",
        "day_progress": "visible trading-day elapsed minutes / full trading minutes.",
        "is_morning_session": "1 when the bar end is in 09:30-11:30.",
        "is_afternoon_session": "1 when the bar end is in 13:00-15:00.",
        "minutes_to_close_norm": "remaining trading minutes / full trading minutes.",
        "is_open_auction": "1 for the daily open-auction decision, otherwise 0.",
        "cash_ratio": "cash / account equity.",
        "position_ratio": "current symbol market value / account equity.",
        "available_position_ratio": "T+1 sellable position value / account equity.",
        "unrealized_pnl_ratio": "unrealized PnL / account equity.",
        "holding_bars_norm": "holding duration / configured cap.",
        "one_lot_nav_ratio": "one-lot notional / account equity.",
        "max_buy_value_ratio": "maximum buyable notional / account equity.",
        "max_sell_value_ratio": "maximum sellable notional / account equity.",
        "market_can_buy": "1 when fixed market rules allow increasing position before account limits.",
        "market_can_sell": "1 when fixed market rules allow reducing position before account limits.",
        "can_buy": "environment-provided field: market_can_buy plus current account cash/position limits.",
        "can_sell": "environment-provided field: market_can_sell plus current T+1 sellable position limits.",
        "is_tradeable": "1 when completed bar can be traded.",
        "is_limit_up": "1 when execution price is at/above fixed upper limit.",
        "is_limit_down": "1 when execution price is at/below fixed lower limit.",
        "is_zero_volume": "1 when completed bar volume is zero.",
    }
    if name.startswith("ema_channel_"):
        if name.endswith("_position"):
            return "(close - lower_ema) / (upper_ema - lower_ema)."
        if name.endswith("_gap"):
            return "close / EMA channel midpoint - 1."
        if name.endswith("_width"):
            return "(upper_ema - lower_ema) / EMA channel midpoint."
        if name.endswith("_slope"):
            return "EMA channel midpoint / previous midpoint - 1."
    return formulas.get(name, "")


def _limit_reference_coverage(
    base: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> dict[str, int]:
    if base is None or base.empty or not {"symbol", "datetime"}.issubset(base.columns):
        return {
            "decision_sessions": 0,
            "eligible_sessions": 0,
            "missing_sessions": 0,
            "eligible_bar_rows": 0,
        }
    columns = [column for column in ("symbol", "adjust", "datetime") if column in base.columns]
    frame = base[columns].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "datetime"])
    frame["_session"] = frame["datetime"].dt.normalize()
    group_columns = [column for column in ("symbol", "adjust") if column in frame.columns]
    sessions = frame[[*group_columns, "_session"]].drop_duplicates().sort_values(
        [*group_columns, "_session"]
    )
    sessions["_has_reference"] = sessions.groupby(group_columns, sort=False).cumcount().gt(0)
    target_sessions = sessions
    target_rows = frame
    if start:
        cutoff = pd.Timestamp(start).normalize()
        target_sessions = target_sessions.loc[target_sessions["_session"] >= cutoff]
        target_rows = target_rows.loc[target_rows["_session"] >= cutoff]
    if end:
        cutoff = pd.Timestamp(end).normalize()
        target_sessions = target_sessions.loc[target_sessions["_session"] <= cutoff]
        target_rows = target_rows.loc[target_rows["_session"] <= cutoff]
    eligible_sessions = target_sessions.loc[target_sessions["_has_reference"]]
    eligible_rows = target_rows.merge(
        eligible_sessions[[*group_columns, "_session"]],
        on=[*group_columns, "_session"],
        how="inner",
    )
    return {
        "decision_sessions": int(len(target_sessions)),
        "eligible_sessions": int(len(eligible_sessions)),
        "missing_sessions": int((~target_sessions["_has_reference"]).sum()),
        "eligible_bar_rows": int(len(eligible_rows)),
    }


def _normalize_csv_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = value.replace("\n", ",").split(",")
    else:
        raw_values = [str(item) for item in value]
    result: list[str] = []
    for item in raw_values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _sequence_windows(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, raw in value.items():
        try:
            number = int(raw)
        except (TypeError, ValueError):
            continue
        if number > 0:
            result[str(key)] = number
    return result


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bounded_positive_int(
    value: Any,
    *,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(default)
    number = max(int(minimum), number)
    if maximum is not None:
        number = min(int(maximum), number)
    return number
