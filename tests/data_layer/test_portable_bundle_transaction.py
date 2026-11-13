from pathlib import Path
import zlib

import duckdb
import pytest

from data_layer.storage import portable_bundle as bundles


PART = "runtime_layer/data/market_parts/000001.parquet"
CATALOG = "runtime_layer/data/market_catalog.duckdb"


def make_bundle(root: Path, *, corrupt: bool = False, paths: list[str] | None = None) -> Path:
    path = root / "incoming.duckdb"
    with duckdb.connect(str(path)) as conn:
        bundles._init_bundle_db(conn)
        for rel in paths or [CATALOG, PART]:
            raw = ("new:" + rel).encode()
            conn.execute(
                "INSERT INTO bundle_items VALUES (?, 'data', 'market_part', ?, ?, 0, 'zlib', ?, ?)",
                [rel, len(raw), "corrupt" if corrupt and rel == PART else bundles.sha256_bytes(raw), len(zlib.compress(raw)), zlib.compress(raw)],
            )
    return path


def originals(root: Path) -> dict[str, bytes]:
    files = {CATALOG: b"old catalog", PART: b"old shard", "runtime_layer/data/market_parts/stale.parquet": b"stale shard"}
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return files


def assert_originals(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        assert (root / rel).read_bytes() == data
    assert not list(root.glob(".bundle-import-*"))


def test_corrupt_later_payload_preserves_all_existing_roots(tmp_path: Path) -> None:
    files = originals(tmp_path)
    bundle = make_bundle(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="verification failed"):
        bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle)
    assert_originals(tmp_path, files)


def test_cancellation_after_staging_preserves_originals(tmp_path: Path) -> None:
    files = originals(tmp_path)
    bundle = make_bundle(tmp_path)
    cancelled = False

    def progress(item):
        nonlocal cancelled
        if item["progress"] >= 0.9:
            cancelled = True

    with pytest.raises(InterruptedError):
        bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle, progress_callback=progress, cancel_check=lambda: cancelled)
    assert_originals(tmp_path, files)


def test_staging_write_failure_preserves_originals(tmp_path: Path, monkeypatch) -> None:
    files = originals(tmp_path)
    bundle = make_bundle(tmp_path)
    original_write = Path.write_bytes

    def disk_full(path, data):
        if "incoming" in path.parts and path.name == "000001.parquet":
            raise OSError("disk full during staging")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", disk_full)
    with pytest.raises(OSError, match="disk full"):
        bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle)
    assert_originals(tmp_path, files)


def test_publish_failure_rolls_back_previous_and_current_roots(tmp_path: Path, monkeypatch) -> None:
    files = originals(tmp_path)
    bundle = make_bundle(tmp_path)
    original_replace = Path.replace

    def fail_one_publish(path, target):
        if "incoming" in path.parts and path.name == "market_parts":
            raise OSError("locked target")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_one_publish)
    with pytest.raises(OSError, match="locked target"):
        bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle)
    assert_originals(tmp_path, files)


def test_cancellation_during_publish_rolls_back(tmp_path: Path) -> None:
    files = originals(tmp_path)
    bundle = make_bundle(tmp_path)

    def cancelled():
        catalog = tmp_path / CATALOG
        return catalog.exists() and catalog.read_bytes().startswith(b"new:")

    with pytest.raises(InterruptedError):
        bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle, cancel_check=cancelled)
    assert_originals(tmp_path, files)


def test_rollback_failure_retains_original_bytes_for_recovery(tmp_path: Path, monkeypatch) -> None:
    files = originals(tmp_path)
    bundle = make_bundle(tmp_path)
    original_replace = Path.replace

    def fail_publish_and_one_restore(path, target):
        if "incoming" in path.parts and path.name == "market_parts":
            raise OSError("publish failed")
        if "originals" in path.parts and path.name == "market_catalog.duckdb":
            raise OSError("restore destination locked")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_one_restore)
    with pytest.raises(RuntimeError, match="originals retained for recovery"):
        bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle)
    recovery_roots = list(tmp_path.glob(".bundle-import-*"))
    assert len(recovery_roots) == 1
    assert (recovery_roots[0] / "originals" / CATALOG).read_bytes() == files[CATALOG]
    assert (tmp_path / PART).read_bytes() == files[PART]


def test_success_replaces_exact_shard_tree_and_preserves_other_assets(tmp_path: Path) -> None:
    originals(tmp_path)
    unrelated = tmp_path / "runtime_layer/data/notes.txt"
    unrelated.write_bytes(b"keep")
    bundle = make_bundle(tmp_path)
    result = bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle)
    assert result["restored_files"] == result["verified_files"] == 2
    assert (tmp_path / PART).read_bytes() == ("new:" + PART).encode()
    assert not (tmp_path / "runtime_layer/data/market_parts/stale.parquet").exists()
    assert unrelated.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".bundle-import-*"))


def test_nonreplacement_keeps_existing_files(tmp_path: Path) -> None:
    files = originals(tmp_path)
    bundle = make_bundle(tmp_path)
    result = bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle, replace_existing=False)
    assert result["skipped_files"] == 2
    assert result["restored_files"] == 0
    assert_originals(tmp_path, files)


@pytest.mark.parametrize("relative", ["../outside.txt", ".git/config", ".GIT/config"])
def test_unsafe_destination_rejected_before_existing_data_changes(tmp_path: Path, relative: str) -> None:
    files = originals(tmp_path)
    bundle = make_bundle(tmp_path, paths=[relative])
    with pytest.raises(ValueError):
        bundles.import_portable_bundle(project_root=tmp_path, bundle_path=bundle)
    assert_originals(tmp_path, files)
