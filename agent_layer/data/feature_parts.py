from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb
import numpy as np
import pandas as pd

from feature_layer.builders.aggregation import normalize_frequency
from feature_layer.datasets.compact import (
    COMPACT_FEATURE_STORE_TABLES,
    ModelInputBatch,
    _compiled_model_input_from_store,
    _load_snapshot_feature_rows_batched,
    _numeric_context_value,
    _pad_cached_market_sequence,
    _spec_from_store,
)
from feature_layer.datasets.sequence import pad_market_sequence

from agent_layer.data.timeline_types import TimelineKey


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PARTS_DIR_NAME = "feature_parts"
FEATURE_PARTS_MANIFEST_NAME = "feature_parts_manifest.json"
STAGE_ORDER: dict[str, int] = {"open_auction": 0, "bar_close": 1}
REQUIRED_PART_TABLES: tuple[str, ...] = tuple(COMPACT_FEATURE_STORE_TABLES)


@dataclass(frozen=True)
class FeaturePartEntry:
    symbol: str
    parts_dir: Path


class FeaturePartsDataset:
    """Read Agent model input directly from per-symbol feature_parts.

    This is the distributed-ready Agent input path.  It exposes the same logical
    tables as the compact DuckDB Feature Store, but every query reads the
    selected symbol parquet parts directly through temporary DuckDB views.  No
    global feature_store.duckdb materialization is required.
    """

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = _resolve_path(dataset_dir)
        if self.dataset_dir.name == FEATURE_PARTS_DIR_NAME:
            self.dataset_dir = self.dataset_dir.parent
        self.parts_root = self.dataset_dir / FEATURE_PARTS_DIR_NAME
        self.manifest_path = self.dataset_dir / FEATURE_PARTS_MANIFEST_NAME
        self._manifest = _load_json(self.manifest_path)
        self._entries = self._load_entries()
        if not self._entries:
            raise FileNotFoundError(
                f"No complete feature_parts were found under {self.parts_root}. "
                "Run Feature Build first."
            )
        self._symbols = tuple(entry.symbol for entry in self._entries)
        self._entry_by_symbol = {entry.symbol: entry for entry in self._entries}
        self._compiled_model_input: dict[str, Any] | None = None
        self._frequencies: tuple[str, ...] | None = None

    @classmethod
    def maybe(cls, path: str | Path) -> "FeaturePartsDataset | None":
        raw = _resolve_path(path)
        dataset_dir = raw.parent if raw.name == FEATURE_PARTS_DIR_NAME else raw
        if raw.suffix.lower() == ".duckdb":
            return None
        parts_root = dataset_dir / FEATURE_PARTS_DIR_NAME
        manifest = dataset_dir / FEATURE_PARTS_MANIFEST_NAME
        if parts_root.exists() and (manifest.exists() or any(parts_root.glob("symbol=*"))):
            return cls(dataset_dir)
        return None

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def compiled_model_input(self) -> dict[str, Any]:
        if self._compiled_model_input is None:
            with self.connect(symbols=self._symbols[:1]) as conn:
                spec = _spec_from_store(conn)
                self._compiled_model_input = _compiled_model_input_from_store(conn, spec)
        return dict(self._compiled_model_input)

    @property
    def schema_hash(self) -> str:
        return str(self.compiled_model_input().get("schema_hash") or "")

    def frequencies(self) -> tuple[str, ...]:
        if self._frequencies is None:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT freq FROM decision_index ORDER BY freq"
                ).fetchall()
            self._frequencies = tuple(str(row[0]) for row in rows)
        return self._frequencies

    @contextmanager
    def connect(
        self,
        *,
        symbols: Iterable[str] | None = None,
        metadata_only: bool = False,
    ) -> Iterator[duckdb.DuckDBPyConnection]:
        selected = self._selected_entries(symbols)
        if not selected:
            raise ValueError("No feature part symbols were selected.")
        conn = duckdb.connect(":memory:")
        try:
            conn.execute("SET threads = 4")
            tables = ("dataset_metadata",) if metadata_only else REQUIRED_PART_TABLES
            for table_name in tables:
                files = self._table_files(table_name, selected)
                if not files:
                    continue
                conn.execute(
                    f'CREATE TEMP VIEW "{table_name}" AS '
                    f"SELECT * FROM read_parquet({_duckdb_list_literal(files)})"
                )
            yield conn
        finally:
            conn.close()

    def timeline(
        self,
        *,
        universe: Iterable[str],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
    ) -> list[TimelineKey]:
        where, params = _decision_filters(
            universe=universe,
            start=start,
            end=end,
            stages=stages,
        )
        with self.connect(symbols=universe) as conn:
            rows = conn.execute(
                "SELECT decision_time, stage, COUNT(*) AS active_symbols FROM decisions "
                f"WHERE {where} GROUP BY decision_time, stage "
                "ORDER BY decision_time, CASE stage WHEN 'open_auction' THEN 0 ELSE 1 END",
                params,
            ).fetchall()
        return [
            TimelineKey(pd.Timestamp(decision_time), str(stage), int(active_symbols))
            for decision_time, stage, active_symbols in rows
        ]

    def trading_dates(self, *, universe: Iterable[str]) -> list[pd.Timestamp]:
        selected = _normalize_unique(universe)
        with self.connect(symbols=selected) as conn:
            rows = conn.execute(
                "SELECT DISTINCT CAST(decision_time AS DATE) FROM decisions "
                "WHERE symbol IN (SELECT UNNEST(?)) ORDER BY 1",
                [list(selected)],
            ).fetchall()
        return [pd.Timestamp(row[0]).normalize() for row in rows]

    def query_decision_rows(
        self,
        *,
        universe: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        stage: str | None = None,
    ) -> pd.DataFrame:
        selected = _normalize_unique(universe)
        stage_clause = "AND d.stage = ? " if stage is not None else ""
        params: list[object] = [start, end, list(selected)]
        if stage is not None:
            params.append(stage)
        with self.connect(symbols=selected) as conn:
            return conn.execute(
                "SELECT d.decision_time, d.stage, d.decision_id, d.symbol, "
                "d.execution_price, d.limit_reference_close, d.limit_pct, "
                "d.is_st, d.market_can_buy, d.market_can_sell, d.is_tradeable, "
                "d.is_limit_up, d.is_limit_down, d.is_zero_volume, "
                "CASE WHEN d.stage = 'open_auction' THEN COALESCE(previous_daily.volume, 0) "
                "ELSE COALESCE(intraday.volume, 0) END AS liquidity_volume, "
                "CASE WHEN d.stage = 'open_auction' THEN COALESCE(previous_daily.amount, 0) "
                "ELSE COALESCE(intraday.amount, 0) END AS liquidity_amount "
                "FROM decisions d "
                "LEFT JOIN market_bars intraday ON d.stage <> 'open_auction' "
                "AND intraday.symbol = d.symbol AND intraday.adjust = d.adjust "
                "AND intraday.freq = '5min' AND intraday.bar_datetime = d.source_bar_end "
                "LEFT JOIN LATERAL ("
                " SELECT mb.volume, mb.amount FROM market_bars mb "
                " WHERE d.stage = 'open_auction' AND mb.symbol = d.symbol "
                " AND mb.adjust = d.adjust AND mb.freq = 'daily' "
                " AND mb.bar_datetime < DATE_TRUNC('day', d.decision_time) "
                " ORDER BY mb.bar_datetime DESC LIMIT 1"
                ") previous_daily ON TRUE "
                "WHERE d.decision_time >= ? AND d.decision_time <= ? "
                "AND d.symbol IN (SELECT UNNEST(?)) "
                f"{stage_clause}ORDER BY d.decision_time, d.stage, d.symbol",
                params,
            ).fetchdf()

    def load_model_input_batches(
        self,
        *,
        decision_ids: Iterable[str],
        frequencies: Iterable[str] | None = None,
    ) -> dict[str, ModelInputBatch]:
        requested_ids = list(dict.fromkeys(str(value) for value in decision_ids if value))
        if not requested_ids:
            return {}
        symbols = self._symbols_for_decision_ids(requested_ids)
        if not symbols:
            return {}
        with self.connect(symbols=symbols) as conn:
            return _load_model_input_batches_from_connection(
                conn,
                decision_ids=requested_ids,
                frequencies=frequencies,
            )

    def validate(self, *, expected_schema_hash: str | None = None, sample_limit: int = 2) -> dict[str, Any]:
        return validate_feature_parts_dataset(
            self.dataset_dir,
            expected_schema_hash=expected_schema_hash,
            sample_limit=sample_limit,
        )

    def _symbols_for_decision_ids(self, decision_ids: list[str]) -> tuple[str, ...]:
        # Decision ids are generated as symbol|adjust|timestamp|stage in current
        # feature builds.  Use the prefix as a fast path and fall back to all
        # symbols for older/unknown ids.
        symbols: list[str] = []
        for decision_id in decision_ids:
            symbol = str(decision_id).split("|", 1)[0].upper()
            if symbol in self._entry_by_symbol and symbol not in symbols:
                symbols.append(symbol)
        return tuple(symbols) if symbols else self._symbols

    def _load_entries(self) -> tuple[FeaturePartEntry, ...]:
        symbols = self._manifest.get("symbols") if isinstance(self._manifest, dict) else {}
        order = self._manifest.get("symbols_order") if isinstance(self._manifest, dict) else []
        entries: list[FeaturePartEntry] = []
        seen: set[str] = set()
        if isinstance(symbols, dict):
            ordered = [str(value) for value in order or symbols.keys()]
            for symbol in ordered:
                entry = dict(symbols.get(symbol, {}))
                raw_parts = entry.get("parts_dir")
                path = _resolve_path(raw_parts) if raw_parts else self.parts_root / f"symbol={_safe_symbol(symbol)}"
                normalized = str(symbol).upper()
                if normalized not in seen and _part_dir_complete(path):
                    entries.append(FeaturePartEntry(normalized, path))
                    seen.add(normalized)
        if not entries and self.parts_root.exists():
            for path in sorted(self.parts_root.glob("symbol=*")):
                if not _part_dir_complete(path):
                    continue
                symbol = _symbol_from_part_dir(path)
                if symbol not in seen:
                    entries.append(FeaturePartEntry(symbol, path))
                    seen.add(symbol)
        return tuple(entries)

    def _selected_entries(self, symbols: Iterable[str] | None) -> tuple[FeaturePartEntry, ...]:
        if symbols is None:
            return self._entries
        selected: list[FeaturePartEntry] = []
        for symbol in _normalize_unique(symbols):
            entry = self._entry_by_symbol.get(str(symbol).upper())
            if entry is not None:
                selected.append(entry)
        return tuple(selected)

    def _table_files(self, table_name: str, entries: Iterable[FeaturePartEntry]) -> list[Path]:
        files: list[Path] = []
        for entry in entries:
            files.extend(sorted((entry.parts_dir / table_name).glob("*.parquet")))
        return files


def validate_feature_parts_dataset(
    dataset_dir: str | Path,
    *,
    expected_schema_hash: str | None = None,
    sample_limit: int = 2,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    def add(name: str, status: str, message: str, **details: Any) -> None:
        checks.append({"name": name, "status": status, "message": message, **details})

    try:
        source = FeaturePartsDataset(dataset_dir)
    except Exception as exc:
        add("feature_parts", "error", f"Feature parts are not readable: {exc}")
        return _validation_result(_resolve_path(dataset_dir), checks, summary, source="feature_parts")

    add(
        "feature_parts",
        "pass",
        f"Feature parts dataset exists with {len(source.symbols)} symbols.",
        symbols=len(source.symbols),
        dataset_dir=str(source.dataset_dir),
        parts_root=str(source.parts_root),
    )

    sample_ids: list[str] = []
    compiled: dict[str, Any] = {}
    frequencies: list[str] = []
    try:
        with source.connect() as conn:
            compiled = _compiled_model_input_from_store(conn, _spec_from_store(conn))
            schema_hash = str(compiled.get("schema_hash") or "")
            schema_ok = bool(schema_hash) and (
                expected_schema_hash is None or schema_hash == expected_schema_hash
            )
            add(
                "model_schema",
                "pass" if schema_ok else "error",
                (
                    f"Model input schema is {schema_hash[:12]}."
                    if schema_ok
                    else "Model input schema is missing or does not match the expected hash."
                ),
                schema_hash=schema_hash,
                expected_schema_hash=expected_schema_hash,
            )
            frequencies = [
                str(row[0])
                for row in conn.execute("SELECT DISTINCT freq FROM decision_index ORDER BY freq").fetchall()
            ]
            counts = {
                table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                for table in (
                    "decisions",
                    "decision_context",
                    "constraints",
                    "decision_index",
                    "market_bars",
                    "market_features",
                )
            }
            aligned = (
                counts["decisions"] > 0
                and counts["decision_context"] == counts["decisions"]
                and counts["constraints"] == counts["decisions"]
            )
            add(
                "decision_alignment",
                "pass" if aligned else "error",
                "Decision, context, and constraint rows are aligned." if aligned else "Decision row tables are not aligned.",
                counts=counts,
            )
            expected_index_rows = counts["decisions"] * len(frequencies)
            index_aligned = bool(frequencies) and counts["decision_index"] == expected_index_rows
            add(
                "decision_index",
                "pass" if index_aligned else "error",
                "Every decision has one index row per enabled frequency." if index_aligned else "Decision index coverage is incomplete.",
                frequencies=frequencies,
                rows=counts["decision_index"],
                expected_rows=expected_index_rows,
            )
            symbol_count, first_time, last_time, unknown_status, missing_reference = conn.execute(
                "SELECT COUNT(DISTINCT symbol), MIN(decision_time), MAX(decision_time), "
                "SUM(CASE WHEN NOT status_known THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN NOT has_limit_reference THEN 1 ELSE 0 END) FROM decisions"
            ).fetchone()
            rules_ok = int(unknown_status or 0) == 0 and int(missing_reference or 0) == 0
            add(
                "market_rules",
                "pass" if rules_ok else "error",
                "All decision rows have dated market status and a previous-close reference." if rules_ok else "Some decisions lack required market-rule facts.",
                unknown_status=int(unknown_status or 0),
                missing_limit_reference=int(missing_reference or 0),
            )
            market_rows = {
                str(freq): int(rows)
                for freq, rows in conn.execute(
                    "SELECT freq, COUNT(*) FROM market_features GROUP BY freq ORDER BY freq"
                ).fetchall()
            }
            market_ok = all(market_rows.get(freq, 0) > 0 for freq in frequencies)
            add(
                "market_rows",
                "pass" if market_ok else "error",
                "Every enabled frequency has market feature rows." if market_ok else "One or more enabled frequencies have no market rows.",
                rows=market_rows,
            )
            requested_samples = max(0, int(sample_limit))
            if requested_samples:
                sample_ids = [
                    str(row[0])
                    for row in conn.execute(
                        "(SELECT decision_id FROM decisions ORDER BY decision_time ASC LIMIT 1) "
                        "UNION ALL "
                        "(SELECT decision_id FROM decisions ORDER BY decision_time DESC LIMIT 1)"
                    ).fetchall()
                ][:requested_samples]
            summary = {
                "counts": counts,
                "symbols": int(symbol_count or 0),
                "first_decision": first_time,
                "last_decision": last_time,
                "frequencies": frequencies,
                "market_rows": market_rows,
                "schema_hash": schema_hash,
            }
    except Exception as exc:
        add("feature_parts_read", "error", f"Feature parts could not be read: {exc}")
        return _validation_result(source.dataset_dir, checks, summary, source="feature_parts")

    expected_shapes = compiled.get("shapes", {}) if isinstance(compiled, dict) else {}
    sample_failures: list[str] = []
    sample_details: list[dict[str, Any]] = []
    for decision_id in sample_ids:
        try:
            batch = source.load_model_input_batches(
                decision_ids=[decision_id],
                frequencies=frequencies,
            )[decision_id]
            shapes = {freq: list(values.shape) for freq, values in batch.market_sequences.items()}
            finite = all(np.isfinite(values).all() for values in batch.market_sequences.values())
            shape_ok = all(shapes.get(freq) == list(expected_shapes.get(freq, [])) for freq in frequencies)
            context_finite = bool(np.isfinite(batch.decision_context).all())
            if not (finite and shape_ok and context_finite):
                sample_failures.append(decision_id)
            sample_details.append(
                {
                    "decision_id": decision_id,
                    "shapes": shapes,
                    "valid_rows": {
                        freq: int(mask.sum()) for freq, mask in batch.sequence_masks.items()
                    },
                    "finite": finite and context_finite,
                }
            )
        except Exception as exc:
            sample_failures.append(decision_id)
            sample_details.append({"decision_id": decision_id, "error": str(exc)})
    add(
        "sample_batches",
        "pass" if sample_ids and not sample_failures else ("warn" if not sample_ids else "error"),
        (
            f"Loaded {len(sample_ids)} finite model batches directly from feature_parts."
            if sample_ids and not sample_failures
            else ("Sample batch validation was skipped." if not sample_ids else "One or more model batches failed validation.")
        ),
        samples=sample_details,
        failures=sample_failures,
    )
    return _validation_result(source.dataset_dir, checks, summary, source="feature_parts")


def _load_model_input_batches_from_connection(
    conn: duckdb.DuckDBPyConnection,
    *,
    decision_ids: Iterable[str],
    frequencies: Iterable[str] | None,
) -> dict[str, ModelInputBatch]:
    requested_ids = list(dict.fromkeys(str(value) for value in decision_ids if value))
    if not requested_ids:
        return {}
    spec = _spec_from_store(conn)
    compiled = _compiled_model_input_from_store(conn, spec)
    configured = [normalize_frequency(value) for value in compiled.get("channels_by_frequency", {})]
    selected = [normalize_frequency(value) for value in (frequencies or configured)]
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
    stable = conn.execute(
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
    snapshots = _load_snapshot_feature_rows_batched(
        conn,
        index=index,
        decisions=decisions,
        spec=spec,
        trade_freq=normalize_frequency(spec.trade_frequency),
        market_cache=None,
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
        for (decision_id, freq), group in stable.groupby(["decision_id", "freq"], sort=False)
    } if not stable.empty else {}

    result: dict[str, ModelInputBatch] = {}
    for decision_id in requested_ids:
        sequences: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        valid_ratios: dict[str, float] = {}
        for freq in selected:
            index_row = index_lookup[(decision_id, freq)]
            frame = stable_groups.get((decision_id, freq), pd.DataFrame()).copy()
            snapshot = snapshots.get((decision_id, freq))
            if snapshot is not None:
                snapshot_frame = pd.DataFrame([snapshot])
                frame = snapshot_frame if frame.empty else pd.concat([frame, snapshot_frame], ignore_index=True)
            valid_ratio = min(1.0, len(frame) / float(max(1, int(index_row.sequence_window))))
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


def _decision_filters(
    *,
    universe: Iterable[str],
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    stages: Iterable[str] | None,
) -> tuple[str, list[object]]:
    clauses = ["symbol IN (SELECT UNNEST(?))"]
    params: list[object] = [list(_normalize_unique(universe))]
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


def _validation_result(
    path: Path,
    checks: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    ok = not any(check["status"] == "error" for check in checks)
    status = "error" if not ok else ("warn" if any(check["status"] == "warn" for check in checks) else "pass")
    return {
        "ok": ok,
        "status": status,
        "store_path": str(path),
        "source": source,
        "checks": checks,
        "summary": summary,
    }


def _part_dir_complete(path: Path) -> bool:
    return path.exists() and all(
        (path / table).exists() and any((path / table).glob("*.parquet"))
        for table in REQUIRED_PART_TABLES
    )


def _symbol_from_part_dir(path: Path) -> str:
    raw = path.name.split("symbol=", 1)[-1]
    if "_" in raw and "." not in raw:
        base, suffix = raw.rsplit("_", 1)
        if suffix in {"SZ", "SH", "BJ"}:
            return f"{base}.{suffix}"
    return raw.upper()


def _safe_symbol(symbol: str) -> str:
    return str(symbol).upper().replace(".", "_")


def _normalize_unique(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        normalized = str(value).strip().upper() if "." in str(value) else str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return PROJECT_ROOT / "runtime_layer" / "reports" / "feature_dataset"
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _duckdb_string_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_list_literal(paths: Iterable[str | Path]) -> str:
    return "[" + ", ".join(_duckdb_string_literal(path) for path in paths) + "]"
