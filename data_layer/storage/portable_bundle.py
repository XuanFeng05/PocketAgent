from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable
import zlib

import duckdb

from data_layer.storage.partitioned_storage import (
    CATALOG_DB_NAME,
    MARKET_PARTS_DIR,
    has_market_shard_storage,
    resolve_market_data_root,
)

BUNDLE_SCHEMA_VERSION = 1
DEFAULT_BUNDLE_PATH = Path("runtime_layer/bundles/pocketagent_bundle.duckdb")
DEFAULT_FEATURE_OUTPUT_DIR = Path("runtime_layer/reports/feature_dataset")
FEATURE_PARTS_DIR_NAME = "feature_parts"
FEATURE_PARTS_MANIFEST_NAME = "feature_parts_manifest.json"
DERIVED_BARS_MANIFEST_NAME = "derived_bars_manifest.json"
MATERIALIZE_REPORT_NAME = "materialize_dashboard_report.csv"
FEATURE_REPORT_SUFFIXES = {".csv", ".json"}

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class BundleFile:
    path: Path
    relative_path: str
    layer: str
    item_type: str


def now_utc_text() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_project_path(project_root: str | Path, path: str | Path | None) -> Path:
    root = Path(project_root).resolve()
    raw = Path(path or "")
    if raw.is_absolute():
        return raw.resolve()
    return (root / raw).resolve()


def _relative_to_project(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return (item for item in path.rglob("*") if item.is_file())


def _add_file(files: list[BundleFile], project_root: Path, path: Path, *, layer: str, item_type: str) -> None:
    if not path.exists() or not path.is_file():
        return
    if not _is_inside(path, project_root):
        raise ValueError(f"Bundle path is outside project root: {path}")
    files.append(BundleFile(path=path, relative_path=_relative_to_project(project_root, path), layer=layer, item_type=item_type))


def collect_portable_bundle_files(
    *,
    project_root: str | Path,
    db_path: str | Path | None = None,
    feature_output_dir: str | Path | None = None,
    include_data: bool = True,
    include_feature: bool = True,
) -> list[BundleFile]:
    root = Path(project_root).resolve()
    files: list[BundleFile] = []

    if include_data:
        data_root = resolve_market_data_root(_resolve_project_path(root, db_path) if db_path else None)
        data_root = _resolve_project_path(root, data_root)
        if has_market_shard_storage(data_root):
            _add_file(files, root, data_root / CATALOG_DB_NAME, layer="data", item_type="market_catalog")
            market_parts = data_root / MARKET_PARTS_DIR
            if market_parts.exists():
                for path in sorted(_iter_files(market_parts), key=lambda item: item.as_posix()):
                    _add_file(files, root, path, layer="data", item_type="market_part")
            # Auxiliary data-layer metadata/reports that are cheap to carry and
            # important for preserving skip/progress state after migration.
            # Legacy runtime stores such as market.duckdb and cache folders are
            # intentionally not bundled.
            for aux in (
                data_root / DERIVED_BARS_MANIFEST_NAME,
                root / "runtime_layer" / "reports" / MATERIALIZE_REPORT_NAME,
            ):
                _add_file(files, root, aux, layer="data", item_type="data_metadata")

    if include_feature:
        output_dir = _resolve_project_path(root, feature_output_dir or DEFAULT_FEATURE_OUTPUT_DIR)
        parts_dir = output_dir / FEATURE_PARTS_DIR_NAME
        if parts_dir.exists():
            for path in sorted(_iter_files(parts_dir), key=lambda item: item.as_posix()):
                _add_file(files, root, path, layer="feature", item_type="feature_part")
        # Keep feature metadata/reports lightweight and exact.  The restored
        # feature_parts directory should be immediately readable by
        # AgentTimelineLoader, and UI previews should survive migration.
        for path in sorted(output_dir.iterdir() if output_dir.exists() else [], key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in FEATURE_REPORT_SUFFIXES:
                item_type = "feature_metadata" if path.suffix.lower() == ".json" else "feature_report"
                _add_file(files, root, path, layer="feature", item_type=item_type)
        # Do not bundle feature_store.duckdb. It is a rebuildable legacy compact
        # export and can be very large or locked by DuckDB/Windows readers. The
        # portable runtime package is the distributed-ready dataset: per-symbol
        # feature_parts plus manifest/contract/report metadata.

    # Deduplicate in case a metadata file was discovered twice.
    deduped: dict[str, BundleFile] = {}
    for item in files:
        deduped[item.relative_path] = item
    return list(deduped.values())


def _init_bundle_db(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE bundle_meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE bundle_items (
            relative_path VARCHAR PRIMARY KEY,
            layer VARCHAR NOT NULL,
            item_type VARCHAR NOT NULL,
            size BIGINT NOT NULL,
            sha256 VARCHAR NOT NULL,
            mtime_ns BIGINT,
            compression VARCHAR NOT NULL,
            compressed_size BIGINT NOT NULL,
            payload BLOB NOT NULL
        )
        """
    )


def _set_meta(conn: duckdb.DuckDBPyConnection, key: str, value: object) -> None:
    conn.execute("INSERT INTO bundle_meta VALUES (?, ?)", [key, str(value)])


def export_portable_bundle(
    *,
    project_root: str | Path,
    bundle_path: str | Path,
    db_path: str | Path | None = None,
    feature_output_dir: str | Path | None = None,
    include_data: bool = True,
    include_feature: bool = True,
    overwrite: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    target = _resolve_project_path(root, bundle_path)
    if not _is_inside(target, root):
        raise ValueError(f"Bundle output must be inside project root: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Bundle already exists: {target}")

    files = collect_portable_bundle_files(
        project_root=root,
        db_path=db_path,
        feature_output_dir=feature_output_dir,
        include_data=include_data,
        include_feature=include_feature,
    )
    if not files:
        raise ValueError("No shard files were found to export.")

    tmp_path = target.with_name(f".{target.name}.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    total = len(files)
    total_bytes = 0
    compressed_bytes = 0
    layer_counts: dict[str, int] = {}
    item_type_counts: dict[str, int] = {}

    def progress(completed: int, message: str, current: str | None = None) -> None:
        if progress_callback:
            progress_callback({
                "completed": completed,
                "total": total,
                "progress": completed / max(1, total),
                "current": current,
                "message": message,
            })

    progress(0, "Preparing portable bundle")
    conn = duckdb.connect(str(tmp_path))
    try:
        _init_bundle_db(conn)
        _set_meta(conn, "schema_version", BUNDLE_SCHEMA_VERSION)
        _set_meta(conn, "created_at", now_utc_text())
        _set_meta(conn, "project_root_name", root.name)
        _set_meta(conn, "include_data", int(bool(include_data)))
        _set_meta(conn, "include_feature", int(bool(include_feature)))
        _set_meta(conn, "db_path", db_path or "")
        _set_meta(conn, "feature_output_dir", feature_output_dir or DEFAULT_FEATURE_OUTPUT_DIR)
        _set_meta(conn, "file_count", total)
        conn.execute("BEGIN TRANSACTION")
        for index, item in enumerate(files, start=1):
            if cancel_check and cancel_check():
                raise InterruptedError("Bundle export cancelled")
            raw = item.path.read_bytes()
            digest = sha256_bytes(raw)
            payload = zlib.compress(raw, level=6)
            stat = item.path.stat()
            total_bytes += int(stat.st_size)
            compressed_bytes += len(payload)
            layer_counts[item.layer] = layer_counts.get(item.layer, 0) + 1
            item_type_counts[item.item_type] = item_type_counts.get(item.item_type, 0) + 1
            conn.execute(
                """
                INSERT INTO bundle_items (
                    relative_path, layer, item_type, size, sha256, mtime_ns,
                    compression, compressed_size, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    item.relative_path,
                    item.layer,
                    item.item_type,
                    int(stat.st_size),
                    digest,
                    int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
                    "zlib",
                    len(payload),
                    payload,
                ],
            )
            if index == 1 or index == total or index % 25 == 0:
                progress(index, f"Packed {index}/{total} files", item.relative_path)
        _set_meta(conn, "total_bytes", total_bytes)
        _set_meta(conn, "compressed_payload_bytes", compressed_bytes)
        _set_meta(conn, "layer_counts", layer_counts)
        _set_meta(conn, "item_type_counts", item_type_counts)
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        conn.close()
        tmp_path.unlink(missing_ok=True)
        raise
    else:
        conn.close()
        if target.exists():
            target.unlink()
        tmp_path.replace(target)

    bundle_size = target.stat().st_size if target.exists() else 0
    return {
        "bundle_path": _relative_to_project(root, target),
        "file_count": total,
        "total_bytes": total_bytes,
        "compressed_payload_bytes": compressed_bytes,
        "bundle_size": bundle_size,
        "layer_counts": layer_counts,
        "item_type_counts": item_type_counts,
        "cancelled": False,
    }


def inspect_portable_bundle(*, project_root: str | Path, bundle_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    bundle = _resolve_project_path(root, bundle_path)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle}")
    with duckdb.connect(str(bundle), read_only=True) as conn:
        meta_rows = conn.execute("SELECT key, value FROM bundle_meta ORDER BY key").fetchall()
        meta = {str(key): value for key, value in meta_rows}
        summary = conn.execute(
            """
            SELECT layer, item_type, COUNT(*) AS files, SUM(size) AS bytes, SUM(compressed_size) AS compressed_bytes
            FROM bundle_items
            GROUP BY layer, item_type
            ORDER BY layer, item_type
            """
        ).fetchdf()
        total = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size), 0), COALESCE(SUM(compressed_size), 0) FROM bundle_items"
        ).fetchone()
    return {
        "bundle_path": _relative_to_project(root, bundle) if _is_inside(bundle, root) else str(bundle),
        "bundle_size": bundle.stat().st_size,
        "meta": meta,
        "file_count": int(total[0] or 0),
        "total_bytes": int(total[1] or 0),
        "compressed_payload_bytes": int(total[2] or 0),
        "summary": summary.to_dict(orient="records"),
    }


def _layer_roots_from_bundle(conn: duckdb.DuckDBPyConnection, root: Path, layers: set[str]) -> list[Path]:
    rows = conn.execute(
        "SELECT DISTINCT relative_path, layer FROM bundle_items ORDER BY relative_path"
    ).fetchall()
    roots: set[Path] = set()
    for relative_path, layer in rows:
        if str(layer) not in layers:
            continue
        path = root / str(relative_path)
        parts = Path(str(relative_path)).parts
        if str(layer) == "data":
            # Restore data as an exact package: catalog + market_parts, plus
            # lightweight derived-bar metadata/reports.
            if MARKET_PARTS_DIR in parts:
                roots.add((root / Path(*parts[:parts.index(MARKET_PARTS_DIR) + 1])).resolve())
            elif path.name in {CATALOG_DB_NAME, DERIVED_BARS_MANIFEST_NAME, MATERIALIZE_REPORT_NAME}:
                roots.add(path.resolve())
        elif str(layer) == "feature":
            if FEATURE_PARTS_DIR_NAME in parts:
                roots.add((root / Path(*parts[:parts.index(FEATURE_PARTS_DIR_NAME) + 1])).resolve())
            elif path.name == FEATURE_PARTS_MANIFEST_NAME or path.suffix.lower() in FEATURE_REPORT_SUFFIXES:
                roots.add(path.resolve())
    return sorted(roots, key=lambda item: len(item.parts), reverse=True)


def import_portable_bundle(
    *,
    project_root: str | Path,
    bundle_path: str | Path,
    include_data: bool = True,
    include_feature: bool = True,
    replace_existing: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    bundle = _resolve_project_path(root, bundle_path)
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle}")

    layers = set()
    if include_data:
        layers.add("data")
    if include_feature:
        layers.add("feature")
    if not layers:
        raise ValueError("Select at least one layer to import.")

    restored_files = 0
    restored_bytes = 0
    skipped_files = 0
    verified_files = 0
    failed_files: list[str] = []

    layer_values = ", ".join("'" + layer.replace("'", "''") + "'" for layer in sorted(layers))

    with duckdb.connect(str(bundle), read_only=True) as conn:
        total = int(conn.execute(
            f"""
            SELECT COUNT(*) FROM bundle_items
            WHERE layer IN ({layer_values})
              AND item_type <> 'feature_store_optional'
              AND relative_path NOT LIKE '%/feature_store.duckdb'
            """
        ).fetchone()[0] or 0)
        if total <= 0:
            raise ValueError("No selected layer files were found in the bundle.")

        if replace_existing:
            for target in _layer_roots_from_bundle(conn, root, layers):
                if cancel_check and cancel_check():
                    raise InterruptedError("Bundle import cancelled")
                if not _is_inside(target, root):
                    raise ValueError(f"Refusing to clean path outside project root: {target}")
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()

        if progress_callback:
            progress_callback({"completed": 0, "total": total, "progress": 0.0, "message": "Restoring portable bundle", "current": None})

        rows = conn.execute(
            f"""
            SELECT relative_path, layer, item_type, size, sha256, compression, payload
            FROM bundle_items
            WHERE layer IN ({layer_values})
              AND item_type <> 'feature_store_optional'
              AND relative_path NOT LIKE '%/feature_store.duckdb'
            ORDER BY relative_path
            """
        ).fetchall()
        for index, (relative_path, layer, item_type, size, digest, compression, payload) in enumerate(rows, start=1):
            if cancel_check and cancel_check():
                raise InterruptedError("Bundle import cancelled")
            target = (root / str(relative_path)).resolve()
            if not _is_inside(target, root):
                raise ValueError(f"Refusing to restore path outside project root: {relative_path}")
            if target.exists() and not replace_existing:
                skipped_files += 1
                continue
            raw = bytes(payload)
            if str(compression).lower() == "zlib":
                raw = zlib.decompress(raw)
            elif str(compression).lower() not in {"none", ""}:
                raise ValueError(f"Unsupported bundle compression {compression!r} for {relative_path}")
            actual_digest = sha256_bytes(raw)
            if actual_digest != str(digest):
                failed_files.append(str(relative_path))
                raise ValueError(f"Hash verification failed for {relative_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.restore_tmp")
            tmp.write_bytes(raw)
            tmp.replace(target)
            restored_files += 1
            restored_bytes += int(size or len(raw))
            verified_files += 1
            if progress_callback and (index == 1 or index == total or index % 25 == 0):
                progress_callback({
                    "completed": index,
                    "total": total,
                    "progress": index / max(1, total),
                    "message": f"Restored {index}/{total} files",
                    "current": str(relative_path),
                })

    return {
        "bundle_path": _relative_to_project(root, bundle) if _is_inside(bundle, root) else str(bundle),
        "restored_files": restored_files,
        "skipped_files": skipped_files,
        "verified_files": verified_files,
        "restored_bytes": restored_bytes,
        "failed_files": failed_files,
        "cancelled": False,
    }
