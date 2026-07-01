from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from agent_layer.experiments.run_config import AgentRunConfig


DEFAULT_RUN_ROOT = Path("runtime_layer/runs/agent")
TERMINAL_STATUSES = {"completed", "failed", "stopped"}


class AgentRunStore:
    def __init__(self, root: str | Path = DEFAULT_RUN_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, config: AgentRunConfig) -> dict[str, Any]:
        config.validate()
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        run_dir = self.root / run_id
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
        self._write_json(run_dir / "config.json", config.payload())
        status = {
            "run_id": run_id,
            "run_name": config.run_name or run_id,
            "config_hash": config.config_hash,
            "status": "queued",
            "phase": "queued",
            "message": "Run created",
            "progress": 0.0,
            "steps": 0,
            "total_steps": config.total_steps,
            "updates": 0,
            "episodes": 0,
            "pid": None,
            "heartbeat": None,
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "metrics_seq": 0,
            "logs_seq": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "error": None,
        }
        self._write_json(run_dir / "status.json", status)
        self._write_json(run_dir / "control.json", {"action": None, "updated_at": _now()})
        return status

    def run_dir(self, run_id: str) -> Path:
        path = (self.root / str(run_id)).resolve()
        if path.parent != self.root.resolve():
            raise ValueError("Invalid Agent run id.")
        return path

    def read_config(self, run_id: str) -> dict[str, Any]:
        return self._read_json(self.run_dir(run_id) / "config.json")

    def read_status(self, run_id: str) -> dict[str, Any]:
        return self._read_json(self.run_dir(run_id) / "status.json")

    def update_status(self, run_id: str, **updates: Any) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        with self._status_lock(run_dir):
            status = self._read_json(run_dir / "status.json")
            status.update(updates)
            status["updated_at"] = _now()
            self._write_json(run_dir / "status.json", status)
            return status

    def list_runs(self) -> list[dict[str, Any]]:
        result = []
        for path in self.root.glob("run_*/status.json"):
            try:
                result.append(self._read_json(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(result, key=lambda item: item.get("created_at", ""), reverse=True)

    def append_metric(self, run_id: str, metric: dict[str, Any]) -> dict[str, Any]:
        return self._append_record(run_id, "metrics", metric)

    def append_log(self, run_id: str, message: str, *, level: str = "info") -> dict[str, Any]:
        return self._append_record(
            run_id,
            "logs",
            {"time": _now(), "level": str(level), "message": str(message)},
        )

    def read_records(
        self,
        run_id: str,
        kind: str,
        *,
        after: int = 0,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        if kind not in {"metrics", "logs"}:
            raise ValueError("Run records must be metrics or logs.")
        path = self.run_dir(run_id) / f"{kind}.jsonl"
        if not path.exists():
            return []
        current = int(self.read_status(run_id).get(f"{kind}_seq") or 0)
        gap = max(0, current - int(after))
        if after > 0 and gap <= int(limit):
            return [
                item for item in self._tail_json_records(path, max(1, gap + 1))
                if int(item.get("seq") or 0) > int(after)
            ][: int(limit)]
        result = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if int(item.get("seq") or 0) > int(after):
                    result.append(item)
                    if len(result) >= int(limit):
                        break
        return result

    @staticmethod
    def _tail_json_records(path: Path, count: int) -> list[dict[str, Any]]:
        chunk_size = 8192
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            data = b""
            while position > 0 and data.count(b"\n") <= count:
                read_size = min(chunk_size, position)
                position -= read_size
                handle.seek(position)
                data = handle.read(read_size) + data
        lines = [line for line in data.splitlines() if line.strip()]
        return [json.loads(line.decode("utf-8")) for line in lines[-count:]]

    def request(self, run_id: str, action: str) -> dict[str, Any]:
        if action not in {"pause", "resume", "checkpoint", "stop"}:
            raise ValueError("Unsupported Agent run action.")
        payload = {"action": action, "updated_at": _now()}
        self._write_json(self.run_dir(run_id) / "control.json", payload)
        return payload

    def read_control(self, run_id: str) -> dict[str, Any]:
        return self._read_json(self.run_dir(run_id) / "control.json")

    def clear_control(self, run_id: str) -> None:
        self._write_json(
            self.run_dir(run_id) / "control.json",
            {"action": None, "updated_at": _now()},
        )

    def _append_record(
        self,
        run_id: str,
        kind: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        with self._status_lock(run_dir):
            status = self._read_json(run_dir / "status.json")
            sequence_name = f"{kind}_seq"
            sequence = int(status.get(sequence_name) or 0) + 1
            item = {"seq": sequence, **record}
            path = run_dir / f"{kind}.jsonl"
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                handle.flush()
            status[sequence_name] = sequence
            status["updated_at"] = _now()
            self._write_json(run_dir / "status.json", status)
            return item

    @staticmethod
    @contextmanager
    def _status_lock(run_dir: Path):
        lock_path = run_dir / ".status.lock"
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
                    raise TimeoutError(f"Timed out waiting for Agent run status lock: {lock_path}")
                time.sleep(0.01)
        try:
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        deadline = time.monotonic() + 2.0
        while True:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (PermissionError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.025)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        deadline = time.monotonic() + 2.0
        while True:
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.025)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
