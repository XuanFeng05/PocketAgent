from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from time import sleep


def launch(run_dir: str | Path, *, processes: int, master_port: int) -> int:
    run_path = Path(run_dir).resolve()
    if processes < 2:
        raise ValueError("The distributed launcher requires at least two processes.")
    children: list[subprocess.Popen[bytes]] = []
    base_environment = dict(os.environ)
    base_environment.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(int(master_port)),
            "WORLD_SIZE": str(int(processes)),
            "LOCAL_WORLD_SIZE": str(int(processes)),
            "GROUP_RANK": "0",
            "GROUP_WORLD_SIZE": "1",
            "ROLE_WORLD_SIZE": str(int(processes)),
            "USE_LIBUV": "0" if os.name == "nt" else base_environment.get("USE_LIBUV", "1"),
        }
    )
    rendezvous_file: Path | None = None
    if os.name == "nt":
        rendezvous_file = run_path / f".ddp_store_{os.getpid()}_{master_port}"
        rendezvous_file.unlink(missing_ok=True)
        base_environment["POCKETAGENT_DDP_INIT_METHOD"] = rendezvous_file.as_uri()
    try:
        for rank in range(processes):
            environment = dict(base_environment)
            environment.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "ROLE_RANK": str(rank),
                }
            )
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        "-m",
                        "agent_layer.cli.worker",
                        "--run-dir",
                        str(run_path),
                    ],
                    cwd=str(Path.cwd()),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                )
            )
        while True:
            failures = [child.returncode for child in children if child.poll() not in {None, 0}]
            if failures:
                _record_launcher_failure(run_path, int(failures[0]))
                return int(failures[0])
            if all(child.poll() == 0 for child in children):
                return 0
            sleep(0.2)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
        if rendezvous_file is not None:
            rendezvous_file.unlink(missing_ok=True)


def _record_launcher_failure(run_path: Path, return_code: int) -> None:
    try:
        from agent_layer.experiments import AgentRunStore

        store = AgentRunStore(run_path.parent)
        status = store.read_status(run_path.name)
        if status.get("status") in {"completed", "failed", "paused", "stopped"}:
            return
        store.update_status(
            run_path.name,
            status="failed",
            phase="failed",
            message=f"Distributed worker rank exited with code {return_code}.",
            error=f"Distributed worker rank exited with code {return_code}.",
            pid=None,
        )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch one PocketAgent DDP process per device.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--processes", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    args = parser.parse_args()
    return launch(args.run_dir, processes=args.processes, master_port=args.master_port)


if __name__ == "__main__":
    raise SystemExit(main())
