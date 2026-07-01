from __future__ import annotations

import argparse
import multiprocessing as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PocketAgent local dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    return parser.parse_args()


def main() -> None:
    # Keep the multiprocessing spawn entrypoint lightweight on Windows.
    # Child processes re-import this module as __mp_main__; importing the
    # dashboard server at module import time would also import agent_controller
    # and torch/CUDA in every feature worker process.
    mp.freeze_support()
    from app_layer.backend.server import run_server

    args = parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
