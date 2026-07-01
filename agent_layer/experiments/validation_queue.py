from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable
from uuid import uuid4


@dataclass(frozen=True)
class ValidationTask:
    task_id: str
    kind: str
    source_checkpoint: str
    checkpoint_path: str
    step: int
    updates: int
    days: int
    symbols: tuple[str, ...]
    device: str
    requested_at: str

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["symbols"] = list(self.symbols)
        return result

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ValidationTask":
        values = dict(payload)
        values["symbols"] = tuple(values.get("symbols") or ())
        return cls(**values)


class ValidationQueue:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.root = self.run_dir / "validations"
        self.tasks_dir = self.root / "tasks"
        self.results_dir = self.root / "results"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with self._queue_lock():
            if not self.status_path.exists():
                self._write_json(
                    self.status_path,
                    {
                        "status": "idle",
                        "pid": None,
                        "task_id": None,
                        "kind": None,
                        "progress": 0.0,
                        "completed": 0,
                        "total": 0,
                        "message": "No validation task",
                        "updated_at": _now(),
                        "last_result": None,
                        "error": None,
                    },
                )

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def pending_path(self) -> Path:
        return self.root / "pending.json"

    @property
    def active_path(self) -> Path:
        return self.root / "active.json"

    def create_task(
        self,
        *,
        kind: str,
        checkpoint: str | Path,
        step: int,
        updates: int,
        days: int,
        symbols: Iterable[str],
        device: str,
    ) -> ValidationTask:
        if kind not in {"quick", "final"}:
            raise ValueError("Validation task kind must be quick or final.")
        task_id = f"{kind}_{int(step):09d}_{uuid4().hex[:6]}"
        source = Path(checkpoint).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Validation checkpoint not found: {source}")
        snapshot = self.tasks_dir / f"{task_id}.pt"
        temporary = snapshot.with_suffix(".pt.tmp")
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(snapshot)
        return ValidationTask(
            task_id=task_id,
            kind=kind,
            source_checkpoint=str(source),
            checkpoint_path=str(snapshot),
            step=int(step),
            updates=int(updates),
            days=int(days),
            symbols=tuple(str(symbol).upper() for symbol in symbols),
            device=str(device),
            requested_at=_now(),
        )

    def submit(self, task: ValidationTask) -> None:
        with self._queue_lock():
            previous = None
            if self.pending_path.exists():
                previous = ValidationTask.from_payload(self._read_json(self.pending_path))
            self._write_json(self.pending_path, task.payload())
            if previous is not None:
                Path(previous.checkpoint_path).unlink(missing_ok=True)

    def claim(self) -> ValidationTask | None:
        with self._queue_lock():
            if not self.pending_path.exists():
                return None
            try:
                self.pending_path.replace(self.active_path)
            except FileNotFoundError:
                return None
            return ValidationTask.from_payload(self._read_json(self.active_path))

    def complete(self, task: ValidationTask) -> None:
        with self._queue_lock():
            if not self.active_path.exists():
                return
            try:
                active = ValidationTask.from_payload(self._read_json(self.active_path))
            except (OSError, ValueError, json.JSONDecodeError):
                self.active_path.unlink(missing_ok=True)
                return
            if active.task_id == task.task_id:
                self.active_path.unlink(missing_ok=True)

    def read_status(self) -> dict[str, Any]:
        with self._queue_lock():
            status = self._read_json(self.status_path)
            if self.pending_path.exists():
                try:
                    pending = self._read_json(self.pending_path)
                    status["pending_task_id"] = pending.get("task_id")
                    status["pending_kind"] = pending.get("kind")
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            return status

    def update_status(self, **updates: Any) -> dict[str, Any]:
        with self._queue_lock():
            status = self._read_json(self.status_path)
            status.update(updates)
            status["updated_at"] = _now()
            self._write_json(self.status_path, status)
            return status

    def write_result(self, task: ValidationTask, payload: dict[str, Any]) -> Path:
        path = self.results_dir / f"{task.task_id}.json"
        self._write_json(path, {"task": task.payload(), **payload})
        return path

    def cleanup_task_checkpoint(self, task: ValidationTask) -> None:
        Path(task.checkpoint_path).unlink(missing_ok=True)

    def list_results(self) -> list[dict[str, Any]]:
        result = []
        for path in self.results_dir.glob("*.json"):
            try:
                result.append(self._read_json(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(
            result,
            key=lambda item: item.get("completed_at", ""),
            reverse=True,
        )

    def ensure_worker(self) -> dict[str, Any]:
        status = self.read_status()
        pid = int(status.get("pid") or 0)
        if pid and _pid_alive(pid):
            return status
        stdout_path = self.root / "worker.stdout.log"
        stderr_path = self.root / "worker.stderr.log"
        command = [
            sys.executable,
            "-m",
            "agent_layer.cli.validator",
            "--run-dir",
            str(self.run_dir),
        ]
        kwargs: dict[str, Any] = {
            "cwd": str(Path(__file__).resolve().parents[2]),
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            )
        else:
            kwargs["start_new_session"] = True
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                **kwargs,
            )
        return self.update_status(
            status="starting",
            pid=process.pid,
            message="Independent validation worker launched",
            error=None,
        )

    def refresh_status(self, *, restart_pending: bool = True) -> dict[str, Any]:
        status = self.read_status()
        pid = int(status.get("pid") or 0)
        if pid and _pid_alive(pid):
            return status

        should_launch = False
        with self._queue_lock():
            if restart_pending and self.active_path.exists():
                if self.pending_path.exists():
                    try:
                        abandoned = ValidationTask.from_payload(
                            self._read_json(self.active_path)
                        )
                        Path(abandoned.checkpoint_path).unlink(missing_ok=True)
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                    self.active_path.unlink(missing_ok=True)
                else:
                    self.active_path.replace(self.pending_path)
            should_launch = bool(restart_pending and self.pending_path.exists())

        if should_launch:
            return self.ensure_worker()
        if status.get("status") in {"starting", "running"}:
            return self.update_status(
                status="interrupted",
                pid=None,
                message="Validation worker stopped before completing its task",
            )
        return status


    @contextmanager
    def _queue_lock(self):
        lock_path = self.root / ".validation.lock"
        deadline = time.monotonic() + 10.0
        descriptor = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, f"{os.getpid()} {_now()}".encode("utf-8"))
            except (FileExistsError, PermissionError):
                try:
                    if time.time() - lock_path.stat().st_mtime > 30.0:
                        try:
                            lock_path.unlink(missing_ok=True)
                            continue
                        except PermissionError:
                            pass
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for validation queue lock: {lock_path}")
                time.sleep(0.01)
        try:
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        deadline = time.monotonic() + 10.0
        delay = 0.025
        while True:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (PermissionError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(0.25, delay * 1.5)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        deadline = time.monotonic() + 10.0
        delay = 0.025
        while True:
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(0.25, delay * 1.5)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not process:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(
                    process, ctypes.byref(exit_code)
                ):
                    return False
                return exit_code.value == 259
            finally:
                ctypes.windll.kernel32.CloseHandle(process)
        except (AttributeError, OSError, ValueError):
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False
