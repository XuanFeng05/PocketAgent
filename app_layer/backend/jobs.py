from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from threading import Lock
from typing import Any, Callable
from uuid import uuid4


class JobManager:
    """Small in-memory job manager for local dashboard tasks."""

    def __init__(self, *, max_workers: int = 2) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def create_job(self, job_type: str, *, title: str | None = None) -> str:
        now = datetime.now().isoformat(timespec="seconds")
        job_id = f"{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "type": job_type,
                "title": title or job_type,
                "status": "queued",
                "progress": 0.0,
                "message": "Queued",
                "current": None,
                "total": None,
                "completed": 0,
                "succeeded": 0,
                "failed": 0,
                "saved_rows": 0,
                "result": None,
                "error": None,
                "cancel_requested": False,
                "logs": [],
                "created_at": now,
                "updated_at": now,
            }
        return job_id

    def update_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(updates)
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")


    def add_log(self, job_id: str, message: str, *, level: str = "info") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            logs = job.setdefault("logs", [])
            logs.append({"time": now, "level": level, "message": str(message)})
            # Keep the in-memory dashboard lightweight during long downloads.
            if len(logs) > 500:
                del logs[:-500]
            job["updated_at"] = now

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job is not None else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in self._jobs.values()]

    def active_job(self, job_type: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            candidates = [
                item
                for item in self._jobs.values()
                if item.get("status") in {"queued", "running"}
                and (job_type is None or item.get("type") == job_type)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
            return deepcopy(candidates[0])

    def request_cancel_active(self, job_type: str | None = None) -> str | None:
        active = self.active_job(job_type)
        if not active:
            return None
        job_id = str(active["job_id"])
        return job_id if self.request_cancel(job_id) else None

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.get("status") in {"completed", "failed", "cancelled"}:
                return True
            job["cancel_requested"] = True
            job["message"] = "Cancel requested"
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")
            return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.get("cancel_requested"))

    def shutdown(self, *, wait: bool = True) -> None:
        for job in self.list_jobs():
            if job.get("status") in {"queued", "running"}:
                self.request_cancel(str(job["job_id"]))
                self.update_job(
                    str(job["job_id"]),
                    message="Server shutdown requested",
                )
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def submit(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        def run():
            current = self.get_job(job_id)
            if current and current.get("cancel_requested"):
                self.update_job(
                    job_id,
                    status="cancelled",
                    progress=0.0,
                    message="Cancelled before start",
                )
                return {"cancelled": True}
            self.update_job(job_id, status="running", message="Starting")
            return fn(*args, **kwargs)

        future = self._executor.submit(run)

        def _finalize(done_future):
            try:
                result = done_future.result()
            except InterruptedError as exc:
                self.update_job(
                    job_id,
                    status="cancelled",
                    error=None,
                    message=str(exc) or "Cancelled",
                )
                return
            except Exception as exc:
                self.update_job(
                    job_id,
                    status="failed",
                    error=str(exc),
                    message=f"Failed: {exc}",
                )
                return

            current = self.get_job(job_id)
            if current and current.get("status") not in {"failed", "cancelled"}:
                self.update_job(
                    job_id,
                    status="completed",
                    progress=1.0,
                    result=result,
                    message="Completed",
                )

        future.add_done_callback(_finalize)
