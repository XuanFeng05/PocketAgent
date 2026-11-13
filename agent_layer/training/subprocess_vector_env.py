from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import traceback
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

from agent_layer.actions import PortfolioAction
from agent_layer.data import AgentTimelineLoader, open_agent_timeline_loader
from agent_layer.environment import ExecutionConfig, SingleSymbolTradingEnv


@dataclass(frozen=True)
class SubprocessEnvSpec:
    """Pickle-safe configuration for subprocess single-symbol env workers."""

    store_path: str
    universe: tuple[str, ...]
    frequencies: tuple[str, ...]
    episode_dates_by_symbol: dict[str, tuple[str, ...]]
    episode_days: int
    seed: int
    execution_config: dict[str, Any]
    reward_config: dict[str, Any]
    stream_chunk_size: int = 64


class SubprocessVectorEnv:
    """Run single-symbol trading environments in child processes.

    The learner/model stay in the parent process.  Child processes own only the
    environment, Agent cache reader, account, and reward/execution simulation.
    This avoids copying the CUDA model into every subprocess and keeps PPO
    updates centralized.
    """

    def __init__(self, spec: SubprocessEnvSpec, env_count: int) -> None:
        self.spec = spec
        self.env_count = max(1, int(env_count))
        self._closed = False
        self._ctx = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        try:
            for index in range(self.env_count):
                parent, child = self._ctx.Pipe(duplex=True)
                process = self._ctx.Process(
                    target=_worker_main,
                    args=(child, spec, int(index)),
                    daemon=True,
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
        except BaseException:
            self.close()
            raise

    def reset(self) -> tuple[list[Any], list[dict[str, Any]], dict[str, float]]:
        started = perf_counter()
        for connection in self._connections:
            connection.send(("reset", None))
        replies = [self._receive(index) for index in range(self.env_count)]
        observations = [reply["observation"] for reply in replies]
        infos = [dict(reply.get("info") or {}) for reply in replies]
        telemetry = _sum_worker_telemetry(replies)
        telemetry["initialized_env_seconds"] = perf_counter() - started
        return observations, infos, telemetry

    def step(self, actions: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
        if len(actions) != self.env_count:
            raise ValueError("Action count must match subprocess environment count.")
        for connection, (directions, sizes) in zip(self._connections, actions):
            connection.send(("step", (np.asarray(directions, dtype=np.int8), np.asarray(sizes, dtype=np.float32))))
        return [self._receive(index) for index in range(self.env_count)]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except Exception:
                pass
        for connection in self._connections:
            try:
                if connection.poll(3.0):
                    connection.recv()
            except Exception:
                pass
            finally:
                try:
                    connection.close()
                except Exception:
                    pass
        for process in self._processes:
            try:
                process.join(timeout=3.0)
            except Exception:
                pass
            if process.is_alive():
                try:
                    process.terminate()
                except Exception:
                    pass
        for process in self._processes:
            try:
                process.join(timeout=2.0)
            except Exception:
                pass

    def _receive(self, index: int) -> dict[str, Any]:
        connection = self._connections[index]
        if not connection.poll(600.0):
            process = self._processes[index]
            raise TimeoutError(
                f"Subprocess env {index} did not respond within 600 seconds "
                f"(pid={process.pid}, alive={process.is_alive()})."
            )
        try:
            reply = connection.recv()
        except (EOFError, OSError) as exc:
            process = self._processes[index]
            raise RuntimeError(
                f"Subprocess env {index} exited before replying "
                f"(pid={process.pid}, exitcode={process.exitcode})."
            ) from exc
        if not isinstance(reply, dict) or not reply.get("ok"):
            error = reply.get("error") if isinstance(reply, dict) else repr(reply)
            trace = reply.get("traceback") if isinstance(reply, dict) else ""
            raise RuntimeError(f"Subprocess env {index} failed: {error}\n{trace}")
        return reply

    def __enter__(self) -> "SubprocessVectorEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _worker_main(connection: Connection, spec: SubprocessEnvSpec, worker_index: int) -> None:
    runner: _SubprocessEnvRunner | None = None
    try:
        runner = _SubprocessEnvRunner(spec, worker_index)
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(_ok(runner.reset()))
            elif command == "step":
                directions, sizes = payload
                connection.send(_ok(runner.step(directions, sizes)))
            elif command == "close":
                runner.close()
                connection.send(_ok({"closed": True}))
                break
            else:
                raise ValueError(f"Unknown subprocess env command: {command}")
    except EOFError:
        pass
    except BaseException as exc:  # send full traceback to parent for debugging
        try:
            connection.send({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        if runner is not None:
            runner.close()
        try:
            connection.close()
        except Exception:
            pass


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, **payload}


class _SubprocessEnvRunner:
    def __init__(self, spec: SubprocessEnvSpec, worker_index: int) -> None:
        self.spec = spec
        self.worker_index = int(worker_index)
        self.rng = np.random.default_rng(int(spec.seed) + self.worker_index * 1_000_003)
        self.loader: AgentTimelineLoader | None = open_agent_timeline_loader(
            spec.store_path,
            universe=spec.universe,
            frequencies=spec.frequencies or None,
            validate_store=False,
            stream_chunk_size=max(1, int(spec.stream_chunk_size)),
        )
        self.symbol_loaders: dict[str, AgentTimelineLoader] = {}
        self.environment: SingleSymbolTradingEnv | None = None
        self.observation: Any = None
        self.episode_symbols = tuple(sorted(spec.episode_dates_by_symbol))
        if not self.episode_symbols:
            raise ValueError("Subprocess env spec contains no episode symbols.")
        self.episode_dates_by_symbol = {
            str(symbol).upper(): tuple(pd.Timestamp(value) for value in dates)
            for symbol, dates in spec.episode_dates_by_symbol.items()
        }

    def reset(self) -> dict[str, Any]:
        started = perf_counter()
        self._close_environment()
        self.environment = self._make_environment()
        self.observation, info = self.environment.reset()
        return {
            "observation": self.observation,
            "info": dict(info or {}),
            "reset_seconds": perf_counter() - started,
            "environment_seconds": 0.0,
            "data_load_seconds": 0.0,
            "data_chunks": 0,
            "data_steps": 0,
        }

    def step(self, directions: np.ndarray, sizes: np.ndarray) -> dict[str, Any]:
        if self.environment is None or self.observation is None:
            self.reset()
        assert self.environment is not None
        started = perf_counter()
        action = PortfolioAction(directions, sizes)
        next_observation, reward, terminated, truncated, info = self.environment.step(action)
        step_seconds = perf_counter() - started
        total_fees = float(self.environment.account.total_fees)
        done = bool(terminated or truncated)
        reset_seconds = 0.0
        if done:
            reset_started = perf_counter()
            self._close_environment()
            self.environment = self._make_environment()
            self.observation, _reset_info = self.environment.reset()
            reset_seconds = perf_counter() - reset_started
        else:
            if next_observation is None:
                raise RuntimeError("A non-terminal subprocess env step returned no observation.")
            self.observation = next_observation
        info_payload = dict(info or {})
        info_payload.setdefault("total_fees", total_fees)
        return {
            "observation": self.observation,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": done,
            "info": info_payload,
            "environment_seconds": step_seconds,
            "reset_seconds": reset_seconds,
            "data_load_seconds": 0.0,
            "data_chunks": 0,
            "data_steps": 0,
        }

    def close(self) -> None:
        self._close_environment()
        for loader in self.symbol_loaders.values():
            close = getattr(loader, "close", None)
            if callable(close):
                close()
        self.symbol_loaders.clear()
        if self.loader is not None:
            close = getattr(self.loader, "close", None)
            if callable(close):
                close()
            self.loader = None

    def _close_environment(self) -> None:
        if self.environment is not None:
            close = getattr(self.environment, "close", None)
            if callable(close):
                close()
            self.environment = None
        self.observation = None

    def _loader_for_symbol(self, symbol: str) -> AgentTimelineLoader:
        normalized = str(symbol).upper()
        loader = self.symbol_loaders.get(normalized)
        if loader is None:
            if self.loader is None:
                raise RuntimeError("Subprocess root loader is closed.")
            loader = self.loader.for_universe((normalized,))
            self.symbol_loaders[normalized] = loader
        return loader

    def _make_environment(self) -> SingleSymbolTradingEnv:
        last_error: Exception | None = None
        attempts = max(4, len(self.episode_symbols) * 2)
        days = max(1, int(self.spec.episode_days))
        for _ in range(attempts):
            symbol = str(self.rng.choice(self.episode_symbols)).upper()
            dates = self.episode_dates_by_symbol[symbol]
            if len(dates) < days:
                continue
            start_index = int(self.rng.integers(0, len(dates) - days + 1))
            try:
                return SingleSymbolTradingEnv(
                    self._loader_for_symbol(symbol),
                    start=dates[start_index],
                    end=dates[start_index + days - 1],
                    execution_config=ExecutionConfig(**self.spec.execution_config),
                    reward_scale=float(self.spec.reward_config.get("scale", 1.0)),
                    hurdle_rate_annual=float(self.spec.reward_config.get("hurdle_rate_annual", 0.0)),
                    drawdown_penalty=float(self.spec.reward_config.get("drawdown_penalty", 0.0)),
                    turnover_penalty=float(self.spec.reward_config.get("turnover_penalty", 0.0)),
                    invalid_action_penalty=float(self.spec.reward_config.get("invalid_action_penalty", 0.0)),
                    enforce_action_contract=True,
                )
            except Exception as exc:
                last_error = exc
        raise ValueError("Unable to create subprocess single-symbol episode.") from last_error


def _sum_worker_telemetry(replies: Iterable[dict[str, Any]]) -> dict[str, float]:
    result = {
        "reset_seconds": 0.0,
        "environment_seconds": 0.0,
        "data_load_seconds": 0.0,
        "data_chunks": 0.0,
        "data_steps": 0.0,
    }
    for reply in replies:
        for key in tuple(result):
            result[key] += float(reply.get(key) or 0.0)
    return result
