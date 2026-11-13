from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tempfile
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
BUNDLE_TRANSACTION_FILE_LIMIT = 25
BUNDLE_TRANSACTION_BYTE_LIMIT = 256 * 1024 * 1024
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


def _quote_sql_string(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _compact_bundle_db(source: Path, target: Path) -> None:
    target.unlink(missing_ok=True)
    target.with_name(f"{target.name}.wal").unlink(missing_ok=True)
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("SET preserve_insertion_order = false")
        conn.execute("SET memory_limit = '4GB'")
        conn.execute(
            f"ATTACH '{_quote_sql_string(source)}' AS source_bundle (READ_ONLY)"
        )
        conn.execute(f"ATTACH '{_quote_sql_string(target)}' AS target_bundle")
        conn.execute("COPY FROM DATABASE source_bundle TO target_bundle")
        source_count = int(
            conn.execute("SELECT COUNT(*) FROM source_bundle.bundle_items").fetchone()[0]
        )
        target_count = int(
            conn.execute("SELECT COUNT(*) FROM target_bundle.bundle_items").fetchone()[0]
        )
        if source_count != target_count:
            raise ValueError(
                f"Bundle compaction row mismatch: {source_count} != {target_count}"
            )
        conn.execute("DETACH target_bundle")
        conn.execute("DETACH source_bundle")
    finally:
        conn.close()


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
    compact_path = tmp_path.with_name(f"{tmp_path.name}.compact")
    if tmp_path.exists():
        tmp_path.unlink()
    compact_path.unlink(missing_ok=True)

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
        conn.execute("SET preserve_insertion_order = false")
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
        transaction_files = 0
        transaction_bytes = 0
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
            transaction_files += 1
            transaction_bytes += len(payload)
            if index < total and (
                transaction_files >= BUNDLE_TRANSACTION_FILE_LIMIT
                or transaction_bytes >= BUNDLE_TRANSACTION_BYTE_LIMIT
            ):
                conn.execute("COMMIT")
                conn.execute("BEGIN TRANSACTION")
                transaction_files = 0
                transaction_bytes = 0
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
        try:
            progress(total, "Compacting portable bundle", str(target))
            _compact_bundle_db(tmp_path, compact_path)
            tmp_path.unlink(missing_ok=True)
            compact_path.replace(target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            compact_path.unlink(missing_ok=True)
            raise

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

    def check_cancelled() -> None:
        if cancel_check and cancel_check():
            raise InterruptedError("Bundle import cancelled")

    # Stage on the same volume so publishing and restoring an original are
    # renames. No original path is changed until every selected file verifies.
    stage = Path(tempfile.mkdtemp(prefix=".bundle-import-", dir=root))
    incoming = stage / "incoming"
    originals = stage / "originals"
    preserve_recovery = False
    installed: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []
    created_parents: list[Path] = []
    try:
        with duckdb.connect(str(bundle), read_only=True) as conn:
            selection = (
                f"layer IN ({layer_values}) AND item_type <> 'feature_store_optional' "
                "AND relative_path NOT LIKE '%/feature_store.duckdb'"
            )
            metadata = conn.execute(
                f"SELECT relative_path FROM bundle_items WHERE {selection} ORDER BY relative_path"
            ).fetchall()
            total = len(metadata)
            if total <= 0:
                raise ValueError("No selected layer files were found in the bundle.")
            targets = [_bundle_import_target(root, str(row[0])) for row in metadata]
            if len(set(targets)) != len(targets):
                raise ValueError("Bundle contains duplicate destination paths")
            candidates = _layer_roots_from_bundle(conn, root, layers) if replace_existing else []
            candidates = [p for p in candidates if any(p == t or p in t.parents for t in targets)]
            candidates.extend(targets)
            # Use disjoint outermost roots: an exact feature_parts replacement
            # removes stale shards, while sibling reports are replaced singly.
            publish_roots: list[Path] = []
            for target in sorted(set(candidates), key=lambda p: (len(p.parts), str(p))):
                _bundle_import_target(root, target.relative_to(root).as_posix())
                if not any(parent == target or parent in target.parents for parent in publish_roots):
                    publish_roots.append(target)
            if any(target == bundle or target in bundle.parents for target in publish_roots):
                raise ValueError("The imported bundle cannot be inside a replacement target")

            check_cancelled()
            if progress_callback:
                progress_callback({"completed": 0, "total": total, "progress": 0.0, "message": "Validating and staging portable bundle", "current": None})
            cursor = conn.execute(
                f"SELECT relative_path, size, sha256, compression, payload FROM bundle_items "
                f"WHERE {selection} ORDER BY relative_path"
            )
            # Fetch one payload at a time; a bundle need not fit in RAM.
            for index in range(1, total + 1):
                check_cancelled()
                relative_path, size, digest, compression, payload = cursor.fetchone()
                target = _bundle_import_target(root, str(relative_path))
                raw = bytes(payload)
                if str(compression).lower() == "zlib":
                    raw = zlib.decompress(raw)
                elif str(compression).lower() not in {"none", ""}:
                    raise ValueError(f"Unsupported bundle compression {compression!r} for {relative_path}")
                if len(raw) != int(size) or sha256_bytes(raw) != str(digest):
                    failed_files.append(str(relative_path))
                    raise ValueError(f"Hash or size verification failed for {relative_path}")
                verified_files += 1
                if target.exists() and not replace_existing:
                    skipped_files += 1
                    continue
                staged = incoming / target.relative_to(root)
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(raw)
                restored_files += 1
                restored_bytes += len(raw)
                if progress_callback and (index == 1 or index == total or index % 25 == 0):
                    progress_callback({
                        "completed": index, "total": total, "progress": 0.9 * index / total,
                        "message": f"Validated {index}/{total} files", "current": str(relative_path),
                    })

        # The bundle connection is closed before any rename. Each rename has
        # a rollback record; cancellation and failures restore all old roots.
        check_cancelled()
        for target in publish_roots:
            staged = incoming / target.relative_to(root)
            if not staged.exists():
                continue
            check_cancelled()
            _bundle_import_target(root, target.relative_to(root).as_posix())
            if target.exists() and not replace_existing:
                raise FileExistsError(f"Import destination appeared during staging: {target}")
            missing = []
            parent = target.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for parent in reversed(missing):
                parent.mkdir()
                created_parents.append(parent)
            if target.exists():
                saved = originals / target.relative_to(root)
                saved.parent.mkdir(parents=True, exist_ok=True)
                target.replace(saved)
                backed_up.append((target, saved))
            staged.replace(target)
            installed.append(target)
        check_cancelled()
        if progress_callback:
            progress_callback({"completed": total, "total": total, "progress": 1.0, "message": "Portable bundle restored", "current": None})
    except BaseException as exc:
        rollback_errors = []
        for target in reversed(installed):
            try:
                staged = incoming / target.relative_to(root)
                staged.parent.mkdir(parents=True, exist_ok=True)
                target.replace(staged)
            except OSError as error:
                rollback_errors.append(str(error))
        for target, saved in reversed(backed_up):
            try:
                saved.replace(target)
            except OSError as error:
                rollback_errors.append(str(error))
        for parent in reversed(created_parents):
            try:
                parent.rmdir()
            except OSError:
                pass
        if rollback_errors:
            preserve_recovery = True
            raise RuntimeError(f"Bundle import failed; originals retained for recovery in {originals}: {rollback_errors}") from exc
        raise
    finally:
        if not preserve_recovery:
            shutil.rmtree(stage, ignore_errors=True)

    return {
        "bundle_path": _relative_to_project(root, bundle) if _is_inside(bundle, root) else str(bundle),
        "restored_files": restored_files,
        "skipped_files": skipped_files,
        "verified_files": verified_files,
        "restored_bytes": restored_bytes,
        "failed_files": failed_files,
        "cancelled": False,
    }


def _bundle_import_target(root: Path, relative_path: str) -> Path:
    """Reject traversal, Git metadata and linked destinations before staging."""
    relative = Path(relative_path)
    if (
        not relative.parts or relative.is_absolute() or relative.drive
        or any(part.casefold() in {"..", ".git"} or ":" in part for part in relative.parts)
        or relative.parts[0].startswith(".bundle-import-")
    ):
        raise ValueError(f"Invalid bundle destination: {relative_path}")
    target = root / relative
    if target == root or not _is_inside(target, root):
        raise ValueError(f"Refusing to restore path outside project root: {relative_path}")
    for path in (target, *target.parents):
        if path == root:
            break
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"Refusing linked bundle destination: {path}")
    return target.resolve()
