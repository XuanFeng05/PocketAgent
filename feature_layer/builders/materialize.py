from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from data_layer.storage.duckdb_storage import (
    DERIVED_BAR_MANIFEST_TABLE_NAME,
    KLINE_SELECT_COLUMNS,
    KLINE_TABLE_NAME,
    connect_duckdb,
    init_duckdb,
)
from feature_layer.builders.aggregation import normalize_frequency


DERIVED_TARGETS_BY_BASE: dict[str, tuple[str, ...]] = {
    "5min": ("15min",),
    "30min": ("60min",),
    "daily": ("weekly", "monthly"),
}
DEFAULT_DERIVED_TARGETS: tuple[str, ...] = DERIVED_TARGETS_BY_BASE["5min"]
DERIVED_MANIFEST_TABLE_NAME = DERIVED_BAR_MANIFEST_TABLE_NAME
DERIVED_SHARD_MANIFEST_NAME = "derived_bars_manifest.json"
DERIVED_SHARD_RULE_VERSION = "derived_shards_v1"


def list_base_symbols(
    db_path: str | Path,
    *,
    base_freq: str = "5min",
    adjust: str = "none",
) -> list[str]:
    db_file = Path(db_path)
    base = normalize_frequency(base_freq)

    try:
        from data_layer.storage.partitioned_storage import (
            get_market_shard_inventory,
            has_market_shard_storage,
            resolve_market_data_root,
        )

        shard_root = resolve_market_data_root(db_file)
        if has_market_shard_storage(shard_root):
            inventory = get_market_shard_inventory(shard_root)
            if inventory.empty:
                return []
            mask = inventory["freq"].astype(str).eq(base) & inventory["adjust"].astype(str).eq(str(adjust))
            return sorted(inventory.loc[mask, "symbol"].dropna().astype(str).str.upper().unique().tolist())
    except Exception:
        # Fall back to legacy DuckDB below.  Materialize should stay usable even
        # if partitioned optional dependencies are unavailable in a local env.
        pass

    if not db_file.exists():
        return []

    with connect_duckdb(db_file) as conn:
        if KLINE_TABLE_NAME not in {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}:
            return []
        rows = conn.execute(
            f"""
            SELECT DISTINCT symbol
            FROM {KLINE_TABLE_NAME}
            WHERE COALESCE(freq, '') = COALESCE(?, '')
              AND COALESCE(adjust, '') = COALESCE(?, '')
            ORDER BY symbol
            """,
            [base, adjust],
        ).fetchall()
    return [str(row[0]) for row in rows]


def materialize_derived_bars(
    db_path: str | Path,
    *,
    symbols: Iterable[str] | None = None,
    base_freq: str = "5min",
    adjust: str = "none",
    targets: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
    chunk_size: int = 10,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Build canonical derived bars inside DuckDB and save them atomically."""
    db_file = Path(db_path)
    base = normalize_frequency(base_freq)
    target_list = list(
        dict.fromkeys(
            normalize_frequency(target)
            for target in (targets if targets is not None else DERIVED_TARGETS_BY_BASE.get(base, ()))
        )
    )
    _validate_targets(base, target_list)
    output_source = source or f"derived_{base}"
    symbol_list = [str(symbol).strip().upper() for symbol in symbols or [] if str(symbol).strip()]
    if not symbol_list:
        symbol_list = list_base_symbols(db_file, base_freq=base, adjust=adjust)

    # Kept for CLI/API compatibility. SQL aggregation no longer copies symbol
    # chunks into Pandas, which was the main large-history bottleneck.
    _ = chunk_size
    rows: list[dict[str, object]] = []
    total_units = len(target_list)
    completed_units = 0
    succeeded_units = 0
    failed_units = 0
    total_saved = 0
    cancelled = False
    target_totals = {
        target: {"saved_rows": 0, "base_rows": 0, "failed": False}
        for target in target_list
    }

    def emit(**updates: Any) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "completed": completed_units,
                "total": total_units,
                "succeeded": succeeded_units,
                "failed": failed_units,
                "saved_rows": total_saved,
                "progress": completed_units / total_units if total_units else 1.0,
                **updates,
            }
        )

    emit(
        current="preparing",
        message=f"Prepared {len(symbol_list)} symbols x {len(target_list)} targets",
        progress=0.0,
    )

    try:
        from data_layer.storage.partitioned_storage import has_market_shard_storage, resolve_market_data_root
    except Exception:
        has_market_shard_storage = None  # type: ignore[assignment]
        resolve_market_data_root = None  # type: ignore[assignment]

    if has_market_shard_storage is not None and resolve_market_data_root is not None:
        shard_root = resolve_market_data_root(db_file)
        if has_market_shard_storage(shard_root):
            return _materialize_derived_bars_in_shards(
                shard_root,
                symbols=symbol_list,
                base_freq=base,
                adjust=adjust,
                target_list=target_list,
                start=start,
                end=end,
                source=output_source,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

    init_duckdb(db_file)
    with connect_duckdb(db_file) as conn:
        for target_index, target in enumerate(target_list):
            if cancel_check and cancel_check():
                cancelled = True
                break

            current = f"{base} -> {target}"

            def stage_callback(stage: str, fraction: float) -> None:
                emit(
                    current=current,
                    message=f"{stage}: {current}",
                    progress=(target_index + fraction) / max(1, total_units),
                )

            try:
                base_counts, saved_counts, saved, stopped, skipped_symbols = _materialize_target_in_duckdb(
                    conn,
                    symbols=symbol_list,
                    base_freq=base,
                    target_freq=target,
                    adjust=adjust,
                    start=start,
                    end=end,
                    source=output_source,
                    cancel_check=cancel_check,
                    stage_callback=stage_callback,
                )
                if stopped:
                    cancelled = True
                    break

                for symbol in symbol_list:
                    if symbol not in base_counts:
                        rows.append(
                            _summary_row(
                                symbol=symbol,
                                target=target,
                                saved_rows=0,
                                base_rows=0,
                                status="missing_base",
                            )
                        )
                    else:
                        rows.append(
                            _summary_row(
                                symbol=symbol,
                                target=target,
                                saved_rows=int(saved_counts.get(symbol, 0)),
                                base_rows=int(base_counts[symbol]),
                                status="skipped" if symbol in skipped_symbols else "ok",
                            )
                        )
                target_totals[target]["saved_rows"] += saved
                target_totals[target]["base_rows"] += int(sum(base_counts.values()))
                total_saved += saved
                succeeded_units += 1
            except Exception as exc:
                target_totals[target]["failed"] = True
                failed_units += 1
                for symbol in symbol_list:
                    rows.append(
                        _summary_row(
                            symbol=symbol,
                            target=target,
                            saved_rows=0,
                            base_rows=0,
                            status="failed",
                            error=str(exc),
                        )
                    )
            completed_units += 1
            emit(
                current=current,
                message=f"Finished {current}",
                progress=completed_units / max(1, total_units),
            )

    for target, totals in target_totals.items():
        rows.append(
            _summary_row(
                symbol="__TOTAL__",
                target=target,
                saved_rows=int(totals["saved_rows"]),
                base_rows=int(totals["base_rows"]),
                status="cancelled" if cancelled else ("failed" if totals["failed"] else "ok"),
            )
        )
    result = pd.DataFrame(rows)
    result.attrs.update(
        {
            "cancelled": cancelled,
            "completed_units": completed_units,
            "total_units": total_units,
            "succeeded_units": succeeded_units,
            "failed_units": failed_units,
            "saved_rows": total_saved,
        }
    )
    return result



def _materialize_derived_bars_in_shards(
    root: str | Path,
    *,
    symbols: list[str],
    base_freq: str,
    adjust: str,
    target_list: list[str],
    start: str | None,
    end: str | None,
    source: str,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> pd.DataFrame:
    """Build derived bars directly in partitioned market shards.

    The old materializer writes into one global DuckDB table.  In the sharded
    storage layout the expensive final merge is intentionally gone, so derived
    bars are generated per symbol and saved back to that symbol's target shard.
    """
    rows: list[dict[str, object]] = []
    total_units = len(target_list)
    completed_units = 0
    succeeded_units = 0
    failed_units = 0
    total_saved = 0
    cancelled = False
    target_totals = {
        target: {"saved_rows": 0, "base_rows": 0, "failed": False}
        for target in target_list
    }

    def emit(**updates: Any) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "completed": completed_units,
                "total": total_units,
                "succeeded": succeeded_units,
                "failed": failed_units,
                "saved_rows": total_saved,
                "progress": completed_units / total_units if total_units else 1.0,
                **updates,
            }
        )

    emit(
        current="preparing",
        message=f"Prepared {len(symbols)} symbols x {len(target_list)} targets in shard storage",
        progress=0.0,
    )

    for target_index, target in enumerate(target_list):
        if cancel_check and cancel_check():
            cancelled = True
            break
        current = f"{base_freq} -> {target}"

        def stage_callback(stage: str, fraction: float) -> None:
            emit(
                current=current,
                message=f"{stage}: {current}",
                progress=(target_index + fraction) / max(1, total_units),
            )

        try:
            base_counts, saved_counts, saved, stopped, skipped_symbols = _materialize_target_in_shards(
                root,
                symbols=symbols,
                base_freq=base_freq,
                target_freq=target,
                adjust=adjust,
                start=start,
                end=end,
                source=source,
                cancel_check=cancel_check,
                stage_callback=stage_callback,
            )
            if stopped:
                cancelled = True
                break

            for symbol in symbols:
                if symbol not in base_counts:
                    rows.append(
                        _summary_row(
                            symbol=symbol,
                            target=target,
                            saved_rows=0,
                            base_rows=0,
                            status="missing_base",
                        )
                    )
                else:
                    rows.append(
                        _summary_row(
                            symbol=symbol,
                            target=target,
                            saved_rows=int(saved_counts.get(symbol, 0)),
                            base_rows=int(base_counts[symbol]),
                            status="skipped" if symbol in skipped_symbols else "ok",
                        )
                    )
            target_totals[target]["saved_rows"] += saved
            target_totals[target]["base_rows"] += int(sum(base_counts.values()))
            total_saved += saved
            succeeded_units += 1
        except Exception as exc:
            target_totals[target]["failed"] = True
            failed_units += 1
            for symbol in symbols:
                rows.append(
                    _summary_row(
                        symbol=symbol,
                        target=target,
                        saved_rows=0,
                        base_rows=0,
                        status="failed",
                        error=str(exc),
                    )
                )
        completed_units += 1
        emit(
            current=current,
            message=f"Finished {current}",
            progress=completed_units / max(1, total_units),
        )

    for target, totals in target_totals.items():
        rows.append(
            _summary_row(
                symbol="__TOTAL__",
                target=target,
                saved_rows=int(totals["saved_rows"]),
                base_rows=int(totals["base_rows"]),
                status="cancelled" if cancelled else ("failed" if totals["failed"] else "ok"),
            )
        )
    result = pd.DataFrame(rows)
    result.attrs.update(
        {
            "cancelled": cancelled,
            "completed_units": completed_units,
            "total_units": total_units,
            "succeeded_units": succeeded_units,
            "failed_units": failed_units,
            "saved_rows": total_saved,
        }
    )
    return result


def _materialize_target_in_shards(
    root: str | Path,
    *,
    symbols: list[str],
    base_freq: str,
    target_freq: str,
    adjust: str,
    start: str | None,
    end: str | None,
    source: str,
    cancel_check: Callable[[], bool] | None,
    stage_callback: Callable[[str, float], None],
) -> tuple[dict[str, int], dict[str, int], int, bool, set[str]]:
    from data_layer.storage.partitioned_storage import (
        KLINE_DATASET,
        _completed_catalog_df,
        delete_symbol_market_slice,
        get_market_catalog_record,
        load_kline_from_market_shards,
        save_kline_to_market_shard,
    )

    data_root = Path(root)
    base_counts: dict[str, int] = {}
    saved_counts: dict[str, int] = {}
    skipped_symbols: set[str] = set()
    total_saved = 0
    bounded_run = bool(start or end)

    stage_callback("Planning shard skip manifest", 0.03)
    catalog = _completed_catalog_df(data_root, dataset=KLINE_DATASET)
    base_records = _kline_catalog_lookup(catalog, symbols=symbols, freq=base_freq, adjust=adjust)
    target_records = _kline_catalog_lookup(catalog, symbols=symbols, freq=target_freq, adjust=adjust)
    manifest = _load_shard_derived_manifest(data_root)
    manifest_records = manifest.setdefault("records", {})
    manifest_dirty = False
    planned_symbols: list[str] = []

    for symbol in symbols:
        base_record = base_records.get(symbol)
        if not _catalog_record_is_usable(data_root, base_record):
            continue
        base_counts[symbol] = int(base_record.get("rows") or 0)
        target_record = target_records.get(symbol)
        key = _shard_manifest_key(
            symbol=symbol,
            base_freq=base_freq,
            target_freq=target_freq,
            adjust=adjust,
            source=source,
            start=start,
            end=end,
        )
        existing_manifest = manifest_records.get(key)
        if _shard_manifest_matches(
            data_root,
            existing_manifest,
            base_record=base_record,
            target_record=target_record,
            source=source,
            start=start,
            end=end,
        ):
            skipped_symbols.add(symbol)
            continue

        # Bootstrap one manifest row for derived targets generated before this
        # manifest existed.  This avoids forcing a one-time full rebuild after
        # upgrading, while future skips are strict: base hash, target hash,
        # rule version, and requested bounds must all keep matching.
        if _catalog_record_is_usable(data_root, target_record) and _target_not_older_than_base(target_record, base_record):
            manifest_records[key] = _shard_manifest_record(
                symbol=symbol,
                base_freq=base_freq,
                target_freq=target_freq,
                adjust=adjust,
                source=source,
                start=start,
                end=end,
                base_record=base_record,
                target_record=target_record,
            )
            manifest_dirty = True
            skipped_symbols.add(symbol)
            continue

        planned_symbols.append(symbol)

    if manifest_dirty:
        _write_shard_derived_manifest(data_root, manifest)

    if not planned_symbols:
        stage_callback(f"Up to date ({len(skipped_symbols)} symbols skipped)", 0.96)
        return base_counts, saved_counts, total_saved, False, skipped_symbols

    stage_callback(
        f"Aggregating {len(planned_symbols)} symbols; skipped {len(skipped_symbols)}",
        0.08,
    )
    total_symbols = max(1, len(planned_symbols))
    for index, symbol in enumerate(planned_symbols):
        if cancel_check and cancel_check():
            if manifest_dirty:
                _write_shard_derived_manifest(data_root, manifest)
            return base_counts, saved_counts, total_saved, True, skipped_symbols

        symbol_fraction = index / total_symbols
        stage_callback(f"Aggregating {symbol}", 0.08 + symbol_fraction * 0.70)
        base_record = base_records.get(symbol)
        base_df = load_kline_from_market_shards(
            data_root,
            symbol=symbol,
            freq=base_freq,
            adjust=adjust,
        )
        if base_df.empty:
            base_counts.pop(symbol, None)
            continue
        base_counts[symbol] = int(len(base_df))

        generated = _aggregate_derived_frame(
            base_df,
            base_freq=base_freq,
            target_freq=target_freq,
            source=source,
        )
        if generated.empty:
            saved_counts[symbol] = 0
            continue
        if start is not None or end is not None:
            generated = _filter_generated_target_range(generated, start=start, end=end)
            if generated.empty:
                saved_counts[symbol] = 0
                continue

        if not bounded_run:
            delete_symbol_market_slice(data_root, symbol, freq=target_freq, adjust=adjust)
        result = save_kline_to_market_shard(generated, data_root, replace_symbol=False)
        saved = int(result.rows or len(generated))
        saved_counts[symbol] = saved
        total_saved += saved

        fresh_base_record = base_record or get_market_catalog_record(
            data_root,
            dataset=KLINE_DATASET,
            symbol=symbol,
            freq=base_freq,
            adjust=adjust,
        )
        fresh_target_record = get_market_catalog_record(
            data_root,
            dataset=KLINE_DATASET,
            symbol=symbol,
            freq=target_freq,
            adjust=adjust,
        )
        if _catalog_record_is_usable(data_root, fresh_base_record) and _catalog_record_is_usable(data_root, fresh_target_record):
            key = _shard_manifest_key(
                symbol=symbol,
                base_freq=base_freq,
                target_freq=target_freq,
                adjust=adjust,
                source=source,
                start=start,
                end=end,
            )
            manifest_records[key] = _shard_manifest_record(
                symbol=symbol,
                base_freq=base_freq,
                target_freq=target_freq,
                adjust=adjust,
                source=source,
                start=start,
                end=end,
                base_record=fresh_base_record,
                target_record=fresh_target_record,
            )
            manifest_dirty = True

    if manifest_dirty:
        _write_shard_derived_manifest(data_root, manifest)
    stage_callback("Committed shard writes", 0.96)
    return base_counts, saved_counts, total_saved, False, skipped_symbols


def _shard_manifest_path(root: str | Path) -> Path:
    return Path(root) / DERIVED_SHARD_MANIFEST_NAME


def _load_shard_derived_manifest(root: str | Path) -> dict[str, Any]:
    path = _shard_manifest_path(root)
    if not path.exists():
        return {
            "schema_version": 1,
            "rule_version": DERIVED_SHARD_RULE_VERSION,
            "records": {},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema_version": 1,
            "rule_version": DERIVED_SHARD_RULE_VERSION,
            "records": {},
        }
    if not isinstance(data, dict):
        data = {}
    data.setdefault("schema_version", 1)
    data.setdefault("rule_version", DERIVED_SHARD_RULE_VERSION)
    records = data.get("records")
    if not isinstance(records, dict):
        data["records"] = {}
    return data


def _write_shard_derived_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    path = _shard_manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["schema_version"] = 1
    payload["rule_version"] = DERIVED_SHARD_RULE_VERSION
    payload["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _shard_manifest_key(
    *,
    symbol: str,
    base_freq: str,
    target_freq: str,
    adjust: str,
    source: str,
    start: str | None,
    end: str | None,
) -> str:
    parts = [
        str(symbol).upper(),
        normalize_frequency(base_freq),
        normalize_frequency(target_freq),
        str(adjust or "").lower(),
        str(source or ""),
        _bound_text(start),
        _bound_text(end),
    ]
    return "|".join(parts)


def _bound_text(value: object | None) -> str:
    if value is None or value == "":
        return ""
    return str(value)


def _kline_catalog_lookup(
    catalog: pd.DataFrame,
    *,
    symbols: list[str],
    freq: str,
    adjust: str,
) -> dict[str, dict[str, object]]:
    if catalog.empty:
        return {}
    symbol_set = {str(symbol).upper() for symbol in symbols}
    frame = catalog.copy()
    frame = frame[
        frame["symbol"].astype(str).str.upper().isin(symbol_set)
        & frame["freq"].fillna("").astype(str).eq(str(freq))
        & frame["adjust"].fillna("").astype(str).eq(str(adjust))
    ]
    records: dict[str, dict[str, object]] = {}
    for item in frame.to_dict(orient="records"):
        records[str(item.get("symbol") or "").upper()] = item
    return records


def _catalog_record_is_usable(root: str | Path, record: dict[str, object] | None) -> bool:
    if not record:
        return False
    if int(record.get("rows") or 0) <= 0:
        return False
    if not str(record.get("data_hash") or "").strip():
        return False
    shard_path = str(record.get("shard_path") or "").strip()
    if not shard_path:
        return False
    return (Path(root) / shard_path).exists()


def _target_not_older_than_base(
    target_record: dict[str, object] | None,
    base_record: dict[str, object] | None,
) -> bool:
    """Allow bootstrap only when the existing target is not older than base.

    This prevents trusting a pre-existing derived shard if base data was updated
    after the target shard was generated.  Once a manifest exists, content hashes
    are the source of truth and updated_at is no longer used for skip decisions.
    """
    if not target_record or not base_record:
        return False
    target_ts = pd.to_datetime(target_record.get("updated_at"), errors="coerce")
    base_ts = pd.to_datetime(base_record.get("updated_at"), errors="coerce")
    if pd.isna(target_ts) or pd.isna(base_ts):
        return False
    return bool(target_ts >= base_ts)


def _shard_manifest_matches(
    root: str | Path,
    record: object,
    *,
    base_record: dict[str, object],
    target_record: dict[str, object] | None,
    source: str,
    start: str | None,
    end: str | None,
) -> bool:
    if not isinstance(record, dict):
        return False
    if str(record.get("rule_version") or "") != DERIVED_SHARD_RULE_VERSION:
        return False
    if str(record.get("source") or "") != str(source or ""):
        return False
    if _bound_text(record.get("start")) != _bound_text(start):
        return False
    if _bound_text(record.get("end")) != _bound_text(end):
        return False
    if not _catalog_record_is_usable(root, target_record):
        return False

    return bool(
        int(base_record.get("rows") or 0) == int(record.get("base_rows") or -1)
        and str(base_record.get("data_hash") or "") == str(record.get("base_hash") or "")
        and _timestamp_text(base_record.get("start_datetime")) == _timestamp_text(record.get("base_start"))
        and _timestamp_text(base_record.get("end_datetime")) == _timestamp_text(record.get("base_end"))
        and int(target_record.get("rows") or 0) == int(record.get("target_rows") or -1)
        and str(target_record.get("data_hash") or "") == str(record.get("target_hash") or "")
        and _timestamp_text(target_record.get("start_datetime")) == _timestamp_text(record.get("target_start"))
        and _timestamp_text(target_record.get("end_datetime")) == _timestamp_text(record.get("target_end"))
    )


def _shard_manifest_record(
    *,
    symbol: str,
    base_freq: str,
    target_freq: str,
    adjust: str,
    source: str,
    start: str | None,
    end: str | None,
    base_record: dict[str, object],
    target_record: dict[str, object],
) -> dict[str, object]:
    return {
        "rule_version": DERIVED_SHARD_RULE_VERSION,
        "symbol": str(symbol).upper(),
        "base_freq": normalize_frequency(base_freq),
        "target_freq": normalize_frequency(target_freq),
        "adjust": str(adjust or "").lower(),
        "source": str(source or ""),
        "start": _bound_text(start),
        "end": _bound_text(end),
        "base_start": _timestamp_text(base_record.get("start_datetime")),
        "base_end": _timestamp_text(base_record.get("end_datetime")),
        "base_rows": int(base_record.get("rows") or 0),
        "base_hash": str(base_record.get("data_hash") or ""),
        "target_start": _timestamp_text(target_record.get("start_datetime")),
        "target_end": _timestamp_text(target_record.get("end_datetime")),
        "target_rows": int(target_record.get("rows") or 0),
        "target_hash": str(target_record.get("data_hash") or ""),
        "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
    }


def _timestamp_text(value: object | None) -> str:
    if value is None or value == "":
        return ""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)
    return str(timestamp)[:19]


def _aggregate_derived_frame(
    base_df: pd.DataFrame,
    *,
    base_freq: str,
    target_freq: str,
    source: str,
) -> pd.DataFrame:
    expected_rows = {
        ("5min", "15min"): 3,
        ("30min", "60min"): 2,
        ("daily", "weekly"): 5,
        ("daily", "monthly"): 21,
    }[(base_freq, target_freq)]
    if base_df.empty:
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS)

    df = base_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["symbol", "datetime"]).sort_values(["symbol", "adjust", "datetime"])
    if df.empty:
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS)

    if base_freq in {"5min", "30min"}:
        df["_period_key"] = df["datetime"].dt.normalize()
        df["_period_bucket"] = (
            df.groupby(["symbol", "adjust", "_period_key"], dropna=False).cumcount() // expected_rows
        )
    elif target_freq == "weekly":
        df["_period_key"] = df["datetime"].dt.normalize() - pd.to_timedelta(df["datetime"].dt.weekday, unit="D")
        df["_period_bucket"] = 0
    else:
        df["_period_key"] = df["datetime"].dt.to_period("M").dt.start_time
        df["_period_bucket"] = 0

    grouped = df.groupby(["symbol", "adjust", "_period_key", "_period_bucket"], sort=True, dropna=False)
    result = grouped.agg(
        datetime=("datetime", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).reset_index()
    result = result.sort_values(["symbol", "adjust", "datetime"]).reset_index(drop=True)

    close = pd.to_numeric(result["close"], errors="coerce")
    open_ = pd.to_numeric(result["open"], errors="coerce")
    previous_close = close.groupby([result["symbol"], result["adjust"]]).shift(1)
    pct = close / previous_close - 1
    fallback = close / open_ - 1
    result["pctChg"] = pct.where(previous_close.notna() & (previous_close != 0), fallback.where(open_.notna() & (open_ != 0), 0.0))
    result["source"] = source
    result["freq"] = target_freq

    for column in KLINE_SELECT_COLUMNS:
        if column not in result.columns:
            result[column] = None
    return result[KLINE_SELECT_COLUMNS].sort_values(["symbol", "datetime", "freq", "adjust"]).reset_index(drop=True)


def _filter_generated_target_range(df: pd.DataFrame, *, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    values = pd.to_datetime(result["datetime"], errors="coerce")
    if start is not None:
        result = result.loc[values >= pd.to_datetime(start, errors="coerce")]
        values = pd.to_datetime(result["datetime"], errors="coerce")
    if end is not None:
        end_value = f"{end} 23:59:59.999999" if isinstance(end, str) and len(end) == 10 else end
        result = result.loc[values <= pd.to_datetime(end_value, errors="coerce")]
    return result.reset_index(drop=True)


def _validate_targets(base_freq: str, targets: list[str]) -> None:
    allowed = set(DERIVED_TARGETS_BY_BASE.get(base_freq, ()))
    if not allowed:
        raise ValueError(f"Unsupported base frequency: {base_freq}.")
    if not targets:
        raise ValueError(f"At least one derived target is required for {base_freq}.")
    unsupported = sorted(set(targets).difference(allowed))
    if unsupported:
        raise ValueError(
            f"Unsupported derived targets for {base_freq}: {', '.join(unsupported)}. "
            f"Available: {', '.join(sorted(allowed))}."
        )


def _materialize_target_in_duckdb(
    conn: Any,
    *,
    symbols: list[str],
    base_freq: str,
    target_freq: str,
    adjust: str,
    start: str | None,
    end: str | None,
    source: str,
    cancel_check: Callable[[], bool] | None,
    stage_callback: Callable[[str, float], None],
) -> tuple[dict[str, int], dict[str, int], int, bool, set[str]]:
    _ensure_derived_manifest(conn)
    expected_rows = {
        ("5min", "15min"): 3,
        ("30min", "60min"): 2,
        ("daily", "weekly"): 5,
        ("daily", "monthly"): 21,
    }[(base_freq, target_freq)]
    base_stats = _base_slice_stats(
        conn,
        symbols=symbols,
        base_freq=base_freq,
        adjust=adjust,
    )
    base_counts = {symbol: int(values["rows"]) for symbol, values in base_stats.items()}
    bounded_run = bool(start or end)
    skipped_symbols: set[str] = set()
    plans: dict[str, dict[str, object]] = {}

    if bounded_run:
        rebuild_start = _period_start(start, base_freq=base_freq, target_freq=target_freq) if start else None
        plans = {
            symbol: {"rebuild_start": rebuild_start, "full_rebuild": False}
            for symbol in base_stats
        }
    else:
        manifests = _load_derived_manifests(
            conn,
            symbols=symbols,
            base_freq=base_freq,
            target_freq=target_freq,
            adjust=adjust,
        )
        target_stats = _target_slice_stats(
            conn,
            symbols=symbols,
            target_freq=target_freq,
            adjust=adjust,
            source=source,
        )
        missing_manifest_symbols = [symbol for symbol in base_stats if symbol not in manifests]
        expected_coverage = (
            _expected_target_coverage(
                conn,
                symbols=missing_manifest_symbols,
                base_freq=base_freq,
                target_freq=target_freq,
                adjust=adjust,
                expected_rows=expected_rows,
            )
            if missing_manifest_symbols
            else {}
        )
        bootstrap_rows: list[dict[str, object]] = []
        append_candidates: list[str] = []
        for symbol, current in base_stats.items():
            manifest = manifests.get(symbol)
            target = target_stats.get(symbol)
            if manifest is None:
                if _coverage_matches(target, expected_coverage.get(symbol)):
                    skipped_symbols.add(symbol)
                    bootstrap_rows.append(
                        _manifest_row(
                            symbol=symbol,
                            base_freq=base_freq,
                            target_freq=target_freq,
                            adjust=adjust,
                            source=source,
                            base=current,
                            target_rows=int(target["rows"]),
                        )
                    )
                else:
                    plans[symbol] = {"rebuild_start": None, "full_rebuild": True}
                continue

            target_is_intact = bool(
                target
                and int(target["rows"]) == int(manifest["target_rows"])
                and _same_timestamp(target["end"], manifest["base_end"])
            )
            base_is_unchanged = bool(
                int(current["rows"]) == int(manifest["base_rows"])
                and str(current["hash"]) == str(manifest["base_hash"])
                and _same_timestamp(current["start"], manifest["base_start"])
                and _same_timestamp(current["end"], manifest["base_end"])
            )
            if base_is_unchanged and target_is_intact:
                skipped_symbols.add(symbol)
                continue

            if (
                target_is_intact
                and _same_timestamp(current["start"], manifest["base_start"])
                and pd.Timestamp(current["end"]) > pd.Timestamp(manifest["base_end"])
                and int(current["rows"]) >= int(manifest["base_rows"])
            ):
                append_candidates.append(symbol)
            else:
                plans[symbol] = {"rebuild_start": None, "full_rebuild": True}

        prefix_stats = _prefix_slice_stats(
            conn,
            symbols=append_candidates,
            manifests=manifests,
            base_freq=base_freq,
            adjust=adjust,
        )
        for symbol in append_candidates:
            manifest = manifests[symbol]
            prefix = prefix_stats.get(symbol)
            if (
                prefix
                and int(prefix["rows"]) == int(manifest["base_rows"])
                and str(prefix["hash"]) == str(manifest["base_hash"])
            ):
                plans[symbol] = {
                    "rebuild_start": _period_start(
                        manifest["base_end"],
                        base_freq=base_freq,
                        target_freq=target_freq,
                    ),
                    "full_rebuild": False,
                }
            else:
                plans[symbol] = {"rebuild_start": None, "full_rebuild": True}

        if bootstrap_rows:
            conn.execute("BEGIN TRANSACTION")
            try:
                _replace_manifest_rows(conn, bootstrap_rows)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    if not plans:
        stage_callback(f"Up to date ({len(skipped_symbols)} symbols skipped)", 0.96)
        return base_counts, {}, 0, False, skipped_symbols

    _register_materialize_plan(conn, plans)
    if base_freq in {"5min", "30min"}:
        period_sql = "CAST(bars.datetime AS DATE)"
        bucket_sql = (
            "FLOOR((ROW_NUMBER() OVER ("
            "PARTITION BY bars.symbol, bars.adjust, CAST(bars.datetime AS DATE) ORDER BY bars.datetime"
            f") - 1) / {expected_rows})"
        )
    else:
        period_sql = (
            "DATE_TRUNC('week', bars.datetime)"
            if target_freq == "weekly"
            else "DATE_TRUNC('month', bars.datetime)"
        )
        bucket_sql = "0"

    stage_callback("Reading and aggregating", 0.10)
    end_condition = ""
    query_params: list[object] = [base_freq, adjust]
    if end:
        end_value = str(end)
        end_condition = " AND bars.datetime <= ?"
        query_params.append(f"{end_value} 23:59:59.999999" if len(end_value) == 10 else end_value)
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _materialized_derived AS
        WITH filtered AS (
            SELECT
                bars.symbol, bars.datetime, bars.open, bars.high, bars.low, bars.close,
                bars.volume, bars.amount, bars.adjust, plan.full_rebuild, plan.rebuild_start,
                {period_sql} AS period_key,
                {bucket_sql} AS period_bucket
            FROM {KLINE_TABLE_NAME} AS bars
            JOIN _materialize_plan AS plan ON plan.symbol = bars.symbol
            WHERE bars.freq = ?
              AND bars.adjust = ?
              AND (plan.rebuild_start IS NULL OR bars.datetime >= plan.rebuild_start)
              {end_condition}
        ), aggregated AS (
            SELECT
                symbol,
                adjust,
                full_rebuild,
                rebuild_start,
                MAX(datetime) AS datetime,
                FIRST(open ORDER BY datetime) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                LAST(close ORDER BY datetime) AS close,
                SUM(volume) AS volume,
                SUM(amount) AS amount,
                COUNT(*)::BIGINT AS source_rows
            FROM filtered
            GROUP BY symbol, adjust, full_rebuild, rebuild_start, period_key, period_bucket
        ), with_previous AS (
            SELECT
                *,
                COALESCE(
                    LAG(close) OVER (PARTITION BY symbol, adjust ORDER BY datetime),
                    CASE WHEN NOT full_rebuild THEN (
                        SELECT previous.close
                        FROM {KLINE_TABLE_NAME} AS previous
                        WHERE previous.symbol = aggregated.symbol
                          AND previous.freq = ?
                          AND previous.adjust = aggregated.adjust
                          AND previous.datetime < COALESCE(aggregated.rebuild_start, aggregated.datetime)
                        ORDER BY previous.datetime DESC
                        LIMIT 1
                    ) END
                ) AS previous_close
            FROM aggregated
        )
        SELECT
            symbol,
            datetime,
            open,
            high,
            low,
            close,
            volume,
            amount,
            COALESCE(
                close / NULLIF(previous_close, 0) - 1,
                close / NULLIF(open, 0) - 1,
                0.0
            ) AS pctChg,
            ?::VARCHAR AS source,
            ?::VARCHAR AS freq,
            adjust,
            source_rows
        FROM with_previous
        ORDER BY symbol, datetime
        """,
        [*query_params, target_freq, source, target_freq],
    )
    count_rows = conn.execute(
        "SELECT symbol, SUM(source_rows)::BIGINT, COUNT(*)::BIGINT "
        "FROM _materialized_derived GROUP BY symbol ORDER BY symbol"
    ).fetchall()
    saved_counts = {str(symbol): int(saved_rows) for symbol, _, saved_rows in count_rows}
    saved = int(sum(saved_counts.values()))
    stage_callback("Aggregated locally", 0.72)
    if cancel_check and cancel_check():
        return base_counts, saved_counts, 0, True, skipped_symbols

    stage_callback("Replacing derived rows", 0.82)
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            f"""
            DELETE FROM {KLINE_TABLE_NAME}
            USING _materialize_plan AS plan
            WHERE plan.full_rebuild
              AND {KLINE_TABLE_NAME}.symbol = plan.symbol
              AND {KLINE_TABLE_NAME}.freq = ?
              AND {KLINE_TABLE_NAME}.adjust = ?
            """,
            [target_freq, adjust],
        )
        conn.execute(
            f"""
            DELETE FROM {KLINE_TABLE_NAME}
            USING (
                SELECT
                    generated.symbol,
                    COALESCE(plan.rebuild_start, MIN(generated.datetime)) AS start_datetime,
                    MAX(generated.datetime) AS end_datetime
                FROM _materialized_derived AS generated
                JOIN _materialize_plan AS plan ON plan.symbol = generated.symbol
                WHERE NOT plan.full_rebuild
                GROUP BY generated.symbol, plan.rebuild_start
            ) AS bounds
            WHERE {KLINE_TABLE_NAME}.symbol = bounds.symbol
              AND {KLINE_TABLE_NAME}.freq = ?
              AND {KLINE_TABLE_NAME}.adjust = ?
              AND {KLINE_TABLE_NAME}.datetime BETWEEN bounds.start_datetime AND bounds.end_datetime
            """,
            [target_freq, adjust],
        )
        insert_columns = ", ".join(KLINE_SELECT_COLUMNS)
        conn.execute(
            f"""
            INSERT INTO {KLINE_TABLE_NAME} (
                {insert_columns}, created_at, updated_at, row_hash
            )
            SELECT
                symbol, datetime, open, high, low, close, volume, amount, pctChg,
                source, freq, adjust,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
            FROM _materialized_derived
            """
        )
        if not bounded_run:
            refreshed_target_stats = _target_slice_stats(
                conn,
                symbols=list(plans),
                target_freq=target_freq,
                adjust=adjust,
                source=source,
            )
            _replace_manifest_rows(
                conn,
                [
                    _manifest_row(
                        symbol=symbol,
                        base_freq=base_freq,
                        target_freq=target_freq,
                        adjust=adjust,
                        source=source,
                        base=base_stats[symbol],
                        target_rows=int(refreshed_target_stats.get(symbol, {}).get("rows", 0)),
                    )
                    for symbol in plans
                ],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    stage_callback("Committed", 0.96)
    return base_counts, saved_counts, saved, False, skipped_symbols


def _ensure_derived_manifest(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DERIVED_MANIFEST_TABLE_NAME} (
            symbol VARCHAR NOT NULL,
            base_freq VARCHAR NOT NULL,
            target_freq VARCHAR NOT NULL,
            adjust VARCHAR NOT NULL,
            source VARCHAR NOT NULL,
            base_start TIMESTAMP,
            base_end TIMESTAMP,
            base_rows BIGINT,
            base_hash VARCHAR,
            target_rows BIGINT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _base_slice_stats(
    conn: Any,
    *,
    symbols: list[str],
    base_freq: str,
    adjust: str,
) -> dict[str, dict[str, object]]:
    if not symbols:
        return {}
    placeholders = ", ".join(["?"] * len(symbols))
    rows = conn.execute(
        f"""
        SELECT
            symbol,
            COUNT(*)::BIGINT,
            MIN(datetime),
            MAX(datetime),
            CAST(BIT_XOR(HASH(datetime, open, high, low, close, volume, amount)) AS VARCHAR)
        FROM {KLINE_TABLE_NAME}
        WHERE freq = ? AND adjust = ? AND symbol IN ({placeholders})
        GROUP BY symbol
        """,
        [base_freq, adjust, *symbols],
    ).fetchall()
    return {
        str(symbol): {"rows": int(count), "start": start, "end": end, "hash": str(data_hash)}
        for symbol, count, start, end, data_hash in rows
    }


def _target_slice_stats(
    conn: Any,
    *,
    symbols: list[str],
    target_freq: str,
    adjust: str,
    source: str,
) -> dict[str, dict[str, object]]:
    if not symbols:
        return {}
    placeholders = ", ".join(["?"] * len(symbols))
    rows = conn.execute(
        f"""
        SELECT symbol, COUNT(*)::BIGINT, MIN(datetime), MAX(datetime)
        FROM {KLINE_TABLE_NAME}
        WHERE freq = ? AND adjust = ? AND source = ? AND symbol IN ({placeholders})
        GROUP BY symbol
        """,
        [target_freq, adjust, source, *symbols],
    ).fetchall()
    return {
        str(symbol): {"rows": int(count), "start": start, "end": end}
        for symbol, count, start, end in rows
    }


def _load_derived_manifests(
    conn: Any,
    *,
    symbols: list[str],
    base_freq: str,
    target_freq: str,
    adjust: str,
) -> dict[str, dict[str, object]]:
    if not symbols:
        return {}
    placeholders = ", ".join(["?"] * len(symbols))
    rows = conn.execute(
        f"""
        SELECT symbol, base_start, base_end, base_rows, base_hash, target_rows
        FROM {DERIVED_MANIFEST_TABLE_NAME}
        WHERE base_freq = ? AND target_freq = ? AND adjust = ?
          AND symbol IN ({placeholders})
        """,
        [base_freq, target_freq, adjust, *symbols],
    ).fetchall()
    return {
        str(symbol): {
            "base_start": base_start,
            "base_end": base_end,
            "base_rows": int(base_rows),
            "base_hash": str(base_hash),
            "target_rows": int(target_rows),
        }
        for symbol, base_start, base_end, base_rows, base_hash, target_rows in rows
    }


def _expected_target_coverage(
    conn: Any,
    *,
    symbols: list[str],
    base_freq: str,
    target_freq: str,
    adjust: str,
    expected_rows: int,
) -> dict[str, dict[str, object]]:
    if not symbols:
        return {}
    placeholders = ", ".join(["?"] * len(symbols))
    if base_freq in {"5min", "30min"}:
        period_sql = "CAST(datetime AS DATE)"
        bucket_sql = (
            "FLOOR((ROW_NUMBER() OVER (PARTITION BY symbol, CAST(datetime AS DATE) "
            f"ORDER BY datetime) - 1) / {expected_rows})"
        )
    else:
        period_sql = "DATE_TRUNC('week', datetime)" if target_freq == "weekly" else "DATE_TRUNC('month', datetime)"
        bucket_sql = "0"
    rows = conn.execute(
        f"""
        WITH base AS (
            SELECT symbol, datetime, {period_sql} AS period_key, {bucket_sql} AS period_bucket
            FROM {KLINE_TABLE_NAME}
            WHERE freq = ? AND adjust = ? AND symbol IN ({placeholders})
        ), targets AS (
            SELECT symbol, MAX(datetime) AS target_datetime
            FROM base
            GROUP BY symbol, period_key, period_bucket
        )
        SELECT symbol, COUNT(*)::BIGINT, MIN(target_datetime), MAX(target_datetime)
        FROM targets
        GROUP BY symbol
        """,
        [base_freq, adjust, *symbols],
    ).fetchall()
    return {
        str(symbol): {"rows": int(count), "start": start, "end": end}
        for symbol, count, start, end in rows
    }


def _prefix_slice_stats(
    conn: Any,
    *,
    symbols: list[str],
    manifests: dict[str, dict[str, object]],
    base_freq: str,
    adjust: str,
) -> dict[str, dict[str, object]]:
    if not symbols:
        return {}
    frame = pd.DataFrame(
        [{"symbol": symbol, "base_end": manifests[symbol]["base_end"]} for symbol in symbols]
    )
    conn.register("_manifest_prefix_df", frame)
    try:
        rows = conn.execute(
            f"""
            SELECT
                bars.symbol,
                COUNT(*)::BIGINT,
                CAST(BIT_XOR(HASH(datetime, open, high, low, close, volume, amount)) AS VARCHAR)
            FROM {KLINE_TABLE_NAME} AS bars
            JOIN _manifest_prefix_df AS manifest ON manifest.symbol = bars.symbol
            WHERE bars.freq = ? AND bars.adjust = ? AND bars.datetime <= manifest.base_end
            GROUP BY bars.symbol
            """,
            [base_freq, adjust],
        ).fetchall()
    finally:
        conn.unregister("_manifest_prefix_df")
    return {
        str(symbol): {"rows": int(count), "hash": str(data_hash)}
        for symbol, count, data_hash in rows
    }


def _register_materialize_plan(conn: Any, plans: dict[str, dict[str, object]]) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "rebuild_start": values["rebuild_start"],
                "full_rebuild": bool(values["full_rebuild"]),
            }
            for symbol, values in plans.items()
        ]
    )
    frame["rebuild_start"] = pd.to_datetime(frame["rebuild_start"], errors="coerce")
    conn.register("_materialize_plan_df", frame)
    try:
        conn.execute(
            "CREATE OR REPLACE TEMP TABLE _materialize_plan AS "
            "SELECT symbol, rebuild_start::TIMESTAMP AS rebuild_start, full_rebuild "
            "FROM _materialize_plan_df"
        )
    finally:
        conn.unregister("_materialize_plan_df")


def _replace_manifest_rows(conn: Any, rows: list[dict[str, object]]) -> None:
    for item in rows:
        conn.execute(
            f"""
            DELETE FROM {DERIVED_MANIFEST_TABLE_NAME}
            WHERE symbol = ? AND base_freq = ? AND target_freq = ? AND adjust = ?
            """,
            [item["symbol"], item["base_freq"], item["target_freq"], item["adjust"]],
        )
        conn.execute(
            f"""
            INSERT INTO {DERIVED_MANIFEST_TABLE_NAME} (
                symbol, base_freq, target_freq, adjust, source,
                base_start, base_end, base_rows, base_hash, target_rows, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                item["symbol"], item["base_freq"], item["target_freq"], item["adjust"], item["source"],
                item["base_start"], item["base_end"], item["base_rows"], item["base_hash"], item["target_rows"],
            ],
        )


def _manifest_row(
    *,
    symbol: str,
    base_freq: str,
    target_freq: str,
    adjust: str,
    source: str,
    base: dict[str, object],
    target_rows: int,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "base_freq": base_freq,
        "target_freq": target_freq,
        "adjust": adjust,
        "source": source,
        "base_start": base["start"],
        "base_end": base["end"],
        "base_rows": int(base["rows"]),
        "base_hash": str(base["hash"]),
        "target_rows": int(target_rows),
    }


def _coverage_matches(
    actual: dict[str, object] | None,
    expected: dict[str, object] | None,
) -> bool:
    return bool(
        actual
        and expected
        and int(actual["rows"]) == int(expected["rows"])
        and _same_timestamp(actual["start"], expected["start"])
        and _same_timestamp(actual["end"], expected["end"])
    )


def _same_timestamp(left: object, right: object) -> bool:
    return pd.Timestamp(left) == pd.Timestamp(right)


def _period_start(value: object, *, base_freq: str, target_freq: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value).normalize()
    if base_freq in {"5min", "30min"}:
        return timestamp
    if target_freq == "weekly":
        return timestamp - pd.Timedelta(days=timestamp.weekday())
    return timestamp.replace(day=1)


def _summary_row(
    *,
    symbol: str,
    target: str,
    saved_rows: int,
    base_rows: int,
    status: str,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "target_freq": target,
        "base_rows": int(base_rows),
        "saved_rows": int(saved_rows),
        "status": status,
        "error": error,
    }
