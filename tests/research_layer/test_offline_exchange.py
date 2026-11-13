from __future__ import annotations

import json
from pathlib import Path
import stat
import zipfile

import pytest

from research_layer.offline import (prepare_task_bundle, inspect_task_bundle, run_task_bundle,
                                    export_result_bundle, import_result_bundle, aggregate_results)


def config():
    return {"mode": "batch", "source_type": "demo", "strategy": "sma_trend",
            "parameter_grid": {"fast_window": [3, 5], "slow_window": [15, 20]}}


def test_two_offline_shards_resume_and_merge_all_candidates(tmp_path, monkeypatch):
    plan_path = tmp_path / "tasks.zip"
    plan = prepare_task_bundle(config(), plan_path, shard_count=2)
    assert len(plan["tasks"]) == 4
    assert inspect_task_bundle(plan_path)["plan_id"] == plan["plan_id"]
    one, two = tmp_path / "machine1", tmp_path / "machine2"
    assert run_task_bundle(plan_path, one, shard_index=0)["completed_tasks"] == 2
    assert run_task_bundle(plan_path, two, shard_index=1)["completed_tasks"] == 2
    from research_layer import runner
    def no_rerun(*args, **kwargs):
        raise AssertionError("Completed tasks were executed a second time")
    monkeypatch.setattr(runner, "run_experiment", no_rerun)
    assert run_task_bundle(plan_path, one, shard_index=0)["reused_tasks"] == 2
    first_results, second_results = tmp_path / "results1.zip", tmp_path / "results2.zip"
    export_result_bundle(one, first_results)
    export_result_bundle(two, second_results)
    collection = tmp_path / "collection"
    first = import_result_bundle(first_results, collection, plan_path)
    assert first["summary"]["status"] == "incomplete"
    assert len(first["summary"]["missing_task_ids"]) == 2
    assert import_result_bundle(first_results, collection, plan_path)["duplicates"] == 2
    second = import_result_bundle(second_results, collection, plan_path)
    assert second["summary"]["status"] == "complete"
    assert len(second["summary"]["trials"]) == 4
    assert len(second["summary"]["ranking"]) == 4
    assert (collection / "aggregate.csv").is_file()
    for path in (collection / "outcomes").glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["result"]["synthetic"] is True


def test_bad_package_checksums_traversal_and_symlinks_are_rejected(tmp_path):
    original = tmp_path / "tasks.zip"
    prepare_task_bundle(config(), original)
    for kind in ("corrupt", "traversal", "symlink"):
        target = tmp_path / f"{kind}.zip"
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(target, "w") as dest:
            for item in source.infolist():
                raw = source.read(item.filename)
                if kind == "corrupt" and item.filename == "input.csv":
                    raw += b"corrupt"
                dest.writestr(item, raw)
            if kind == "traversal":
                dest.writestr("../escape.txt", b"bad")
            if kind == "symlink":
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                dest.writestr(info, "outside")
        with pytest.raises(ValueError):
            inspect_task_bundle(target)
    assert not (tmp_path / "escape.txt").exists()


def test_duplicate_semantic_results_dedupe_and_conflict_is_rejected(tmp_path):
    from research_layer import offline
    plan_path = tmp_path / "plan.zip"
    plan = prepare_task_bundle({"mode": "single", "source_type": "demo", "strategy": "cash"}, plan_path)
    first, second = tmp_path / "a", tmp_path / "b"
    run_task_bundle(plan_path, first)
    run_task_bundle(plan_path, second)
    zip1, zip2 = tmp_path / "a.zip", tmp_path / "b.zip"
    export_result_bundle(first, zip1)
    export_result_bundle(second, zip2)
    collection = tmp_path / "collection"
    import_result_bundle(zip1, collection, plan_path)
    assert import_result_bundle(zip2, collection, plan_path)["duplicates"] == 1
    task_id = plan["tasks"][0]["task_id"]
    with zipfile.ZipFile(zip2) as z:
        manifest = json.loads(z.read("manifest.json"))
        members = {key: z.read(key) for key in manifest["files"]}
    key = f"outcomes/{task_id}.json"
    outcome = json.loads(members[key])
    outcome["result"]["strategy"]["metrics"]["final_nav"] += 1
    members[key] = offline._json(outcome)
    manifest.pop("bundle_id")
    manifest.pop("files")
    conflict = tmp_path / "conflict.zip"
    offline._write_bundle(conflict, manifest, members)
    before = (collection / "outcomes" / f"{task_id}.json").read_bytes()
    with pytest.raises(ValueError, match="Conflicting"):
        import_result_bundle(conflict, collection, plan_path)
    assert (collection / "outcomes" / f"{task_id}.json").read_bytes() == before


def test_failed_attempts_are_exported_and_retry_is_explicit(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.zip"
    prepare_task_bundle({"mode": "batch", "source_type": "demo", "strategy": "sma_trend",
                         "parameter_grid": {"fast_window": [3, 30], "slow_window": [10]}}, plan_path)
    workspace = tmp_path / "worker"
    first = run_task_bundle(plan_path, workspace)
    assert (first["completed_tasks"], first["failed_tasks"]) == (1, 1)
    second = run_task_bundle(plan_path, workspace)
    assert second["reused_tasks"] == 2
    third = run_task_bundle(plan_path, workspace, retry_failed=True)
    assert third["reused_tasks"] == 1
    assert third["failed_tasks"] == 1
    result_path = tmp_path / "results.zip"
    export_result_bundle(workspace, result_path)
    summary = import_result_bundle(result_path, tmp_path / "collection", plan_path)["summary"]
    assert summary["failed_tasks"] == 1 and summary["received_tasks"] == 2


def test_source_mismatch_and_modified_saved_task_are_rejected(tmp_path, monkeypatch):
    import research_layer.offline as offline
    plan_path = tmp_path / "plan.zip"
    prepare_task_bundle({"mode": "single", "source_type": "demo", "strategy": "cash"}, plan_path)
    with monkeypatch.context() as patch:
        patch.setattr(offline, "_current_implementation_hash", lambda: "different")
        with pytest.raises(ValueError, match="source differs"):
            run_task_bundle(plan_path, tmp_path / "wrong")
    workspace = tmp_path / "worker"
    run_task_bundle(plan_path, workspace)
    outcome = next(workspace.glob("tasks/*/attempts/*/outcome.json"))
    outcome.write_bytes(outcome.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        run_task_bundle(plan_path, workspace)


def test_cancelled_worker_reuses_completed_task_on_restart(tmp_path):
    plan_path = tmp_path / "plan.zip"
    prepare_task_bundle(config(), plan_path)
    state = {"cancel": False}
    def progress(update):
        state["cancel"] = update["completed"] >= 1
    workspace = tmp_path / "worker"
    with pytest.raises(InterruptedError):
        run_task_bundle(plan_path, workspace, progress_callback=progress, cancel_check=lambda: state["cancel"])
    resumed = run_task_bundle(plan_path, workspace)
    assert resumed["reused_tasks"] == 1
    assert resumed["completed_tasks"] == 4


def test_dead_worker_lock_is_preserved_and_recovered(tmp_path):
    plan_path = tmp_path / "plan.zip"
    plan = prepare_task_bundle({"mode": "single", "source_type": "demo", "strategy": "cash"}, plan_path)
    workspace = tmp_path / "worker"
    task_dir = workspace / "tasks" / plan["tasks"][0]["task_id"]
    task_dir.mkdir(parents=True)
    (task_dir / "running.lock").write_text(json.dumps({"pid": 999999999, "process_started": 0}), encoding="utf-8")
    assert run_task_bundle(plan_path, workspace)["completed_tasks"] == 1
    assert len(list(task_dir.glob("abandoned-lock-*.json"))) == 1


def test_collection_tampering_is_detected(tmp_path):
    plan_path = tmp_path / "plan.zip"
    prepare_task_bundle({"mode": "single", "source_type": "demo", "strategy": "cash"}, plan_path)
    run_task_bundle(plan_path, tmp_path / "worker")
    export_result_bundle(tmp_path / "worker", tmp_path / "results.zip")
    collection = tmp_path / "collection"
    import_result_bundle(tmp_path / "results.zip", collection, plan_path)
    path = next((collection / "outcomes").glob("*.json"))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        aggregate_results(collection, plan_path)


def test_walking_job_is_kept_as_one_complete_experiment_task(tmp_path):
    plan_path = tmp_path / "walk.zip"
    plan = prepare_task_bundle({**config(), "mode": "walk_forward"}, plan_path, shard_count=2)
    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["config"]["mode"] == "walk_forward"
    result = run_task_bundle(plan_path, tmp_path / "worker")
    assert result["completed_tasks"] == 1
    outcome = json.loads(next((tmp_path / "worker").glob("tasks/*/attempts/*/outcome.json")).read_text(encoding="utf-8"))
    assert outcome["result"]["walk_forward"]["holdout"]["status"] == "reserved"
