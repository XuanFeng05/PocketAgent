from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runtime_layer" / "runs" / "agent"
SPARK_CHARS = ".:-=+*#%@"


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, value: object, code: str) -> str:
        text = str(value)
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def cyan(self, value: object) -> str:
        return self.paint(value, "1;36")

    def green(self, value: object) -> str:
        return self.paint(value, "1;32")

    def yellow(self, value: object) -> str:
        return self.paint(value, "1;33")

    def red(self, value: object) -> str:
        return self.paint(value, "1;31")

    def dim(self, value: object) -> str:
        return self.paint(value, "2")


class ProcessSampler:
    """Low-overhead CPU/RSS sampling for one training process group."""

    def __init__(self) -> None:
        self._processes: dict[int, Any] = {}
        self._warmed = False
        self._psutil: Any | None = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except ImportError:
            pass

    def sample(self, pid: int, run_dir: Path) -> dict[str, float | None]:
        if self._psutil is None or pid <= 0:
            return {"capacity": None, "cores": None, "rss": None, "load": _load_average()}

        psutil = self._psutil
        try:
            root = psutil.Process(pid)
            parent = root.parent()
            if parent is not None and "agent_layer.cli.distributed_launcher" in " ".join(parent.cmdline()):
                root = parent
            discovered = [root, *root.children(recursive=True)]
            run_path = str(run_dir)
            for process in psutil.process_iter(["cmdline"]):
                command = " ".join(process.info.get("cmdline") or ())
                if run_path in command and any(
                    module in command
                    for module in (
                        "agent_layer.cli.worker",
                        "agent_layer.cli.validator",
                        "agent_layer.cli.distributed_launcher",
                    )
                ):
                    discovered.append(process)
        except (psutil.Error, OSError):
            return {"capacity": None, "cores": None, "rss": None, "load": _load_average()}

        active: dict[int, Any] = {}
        for process in discovered:
            try:
                active[process.pid] = self._processes.get(process.pid, process)
            except psutil.Error:
                continue
        self._processes = active

        cpu_percent = 0.0
        rss = 0.0
        sampled = 0
        for process in active.values():
            try:
                cpu_percent += float(process.cpu_percent(interval=None))
                rss += float(process.memory_info().rss)
                sampled += 1
            except psutil.Error:
                continue

        if not self._warmed:
            self._warmed = sampled > 0
            cores = None
            capacity = None
        else:
            cores = cpu_percent / 100.0
            capacity = 100.0 * cores / max(_effective_cpu_count(), 1.0)
        return {
            "capacity": capacity,
            "cores": cores,
            "rss": rss,
            "load": _load_average(),
        }


def _effective_cpu_count() -> float:
    visible = float(os.cpu_count() or 1)
    affinity = visible
    try:
        affinity = float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass

    quota = visible
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = cpu_max.read_text(encoding="ascii").split()[:2]
        if quota_text != "max":
            quota = float(quota_text) / float(period_text)
    except (OSError, ValueError, IndexError):
        try:
            quota_value = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="ascii"))
            period_value = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="ascii"))
            if quota_value > 0 and period_value > 0:
                quota = quota_value / period_value
        except (OSError, ValueError):
            pass
    return max(1.0, min(visible, affinity, quota))


def _load_average() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_recent_metrics(path: Path, limit: int = 80) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    payload = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        pass
    return list(rows)


def _latest_run(runs_root: Path) -> Path:
    candidates = [path for path in runs_root.glob("run_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No Agent runs found under {runs_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if str(_read_json(candidate / "status.json").get("status") or "").lower() in {
            "running",
            "starting",
            "stopping",
        }:
            return candidate
    return candidates[0]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    days, total = divmod(total, 86400)
    hours, total = divmod(total, 3600)
    minutes, secs = divmod(total, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


def _format_interval(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return _format_duration(seconds)


def _format_bytes(value: float | None) -> str:
    if value is None:
        return "--"
    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or suffix == "TiB":
            return f"{size:.1f} {suffix}"
        size /= 1024.0
    return "--"


def _format_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "--"
    number = _as_float(value, float("nan"))
    if not math.isfinite(number):
        return "--"
    if number != 0.0 and (abs(number) >= 10000 or abs(number) < 0.0001):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _sparkline(values: Iterable[float], width: int = 32) -> str:
    points = [value for value in values if math.isfinite(value)][-width:]
    if not points:
        return "-" * width
    low = min(points)
    high = max(points)
    if math.isclose(low, high):
        rendered = "=" * len(points)
    else:
        scale = (len(SPARK_CHARS) - 1) / (high - low)
        rendered = "".join(SPARK_CHARS[int((value - low) * scale)] for value in points)
    return rendered.rjust(width, " ")


def _trend(values: list[float]) -> tuple[str, float]:
    if len(values) < 4:
        return "FLAT", 0.0
    split = max(1, len(values) // 2)
    previous = sum(values[:split]) / split
    current = sum(values[split:]) / max(1, len(values) - split)
    denominator = max(abs(previous), 1e-9)
    change = max(-999.0, min(999.0, 100.0 * (current - previous) / denominator))
    if abs(change) < 0.1:
        return "FLAT", change
    return ("UP" if change > 0 else "DOWN"), change


def _heartbeat_age(status: dict[str, Any]) -> float | None:
    value = status.get("heartbeat") or status.get("updated_at")
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    except (psutil.Error, OSError):
        return False


def _progress_bar(progress: float, width: int) -> str:
    progress = max(0.0, min(1.0, progress))
    filled = int(progress * width)
    if filled >= width:
        return "[" + "=" * width + "]"
    marker = ">" if filled > 0 else ""
    return "[" + "=" * max(0, filled - 1) + marker + "." * (width - filled) + "]"


def _rolling_performance(updates: list[dict[str, Any]], window: int = 5) -> dict[str, float | None]:
    recent = updates[-window:]
    seconds = sum(
        max(0.0, _as_float(row.get("collect_seconds")))
        + max(0.0, _as_float(row.get("optimize_seconds")))
        for row in recent
    )
    samples = sum(max(0.0, _as_float(row.get("samples_per_update"))) for row in recent)
    speed = samples / seconds if seconds > 0 and samples > 0 else None
    return {
        "speed": speed,
        "collect": sum(_as_float(row.get("collect_seconds")) for row in recent) / len(recent) if recent else None,
        "optimize": sum(_as_float(row.get("optimize_seconds")) for row in recent) / len(recent) if recent else None,
        "samples": sum(_as_float(row.get("samples_per_update")) for row in recent) / len(recent) if recent else None,
    }


def _validation_summary(run_dir: Path) -> str:
    status = _read_json(run_dir / "validations" / "status.json")
    if str(status.get("status") or "").lower() in {"running", "starting"}:
        completed = _as_int(status.get("completed"))
        total = _as_int(status.get("total"))
        kind = status.get("kind") or "validation"
        return f"{kind} RUNNING  {completed:,}/{total:,}  {status.get('message') or ''}".rstrip()

    result_paths = sorted(
        (run_dir / "validations" / "results").glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in result_paths:
        payload = _read_json(path)
        if payload.get("status") != "completed":
            continue
        task = payload.get("task") or {}
        metrics = (payload.get("result") or {}).get("metrics") or {}
        return (
            f"step {_as_int(task.get('step')):,}  "
            f"return {_as_float(metrics.get('total_return')):+.2%}  "
            f"sharpe {_as_float(metrics.get('sharpe')):+.3f}  "
            f"drawdown {_as_float(metrics.get('maximum_drawdown')):.2%}"
        )
    return "not run yet"


def _checkpoint_summary(run_dir: Path) -> str:
    checkpoint_dir = run_dir / "checkpoints"
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    milestones = sorted(
        [*checkpoint_dir.glob("step_*.pt"), *checkpoint_dir.glob("milestone_step_*.pt")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    latest_text = f"latest.pt {'OK' if latest.exists() else 'MISSING'}"
    if latest.exists():
        latest_text += f" ({_format_bytes(latest.stat().st_size)})"
    if best.exists():
        return f"{latest_text}   best.pt OK"
    if milestones:
        return f"{latest_text}   milestone {milestones[0].name}"
    return latest_text


def _gpu_rows(enabled: bool) -> list[str]:
    if not enabled:
        return ["GPU    : disabled for this run"]
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2.0, check=True)
    except (OSError, subprocess.SubprocessError):
        return ["GPU    : CUDA run; nvidia-smi data unavailable"]
    rows = ["GPU      util   vram                 power     temp"]
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, utilization, used, total, power, temperature = fields
        rows.append(
            f"GPU {index:<3} {utilization:>4}%   {used:>7}/{total:<7} MiB   {power:>7} W   {temperature:>3} C"
        )
    return rows


def _status_text(status: dict[str, Any], palette: Palette, refresh: float) -> str:
    value = str(status.get("status") or "unknown").upper()
    age = _heartbeat_age(status)
    pid = _as_int(status.get("pid"))
    if (
        value in {"RUNNING", "STARTING"}
        and age is not None
        and age > max(30.0, refresh * 4.0)
        and not _pid_alive(pid)
    ):
        return palette.red("STALE")
    if value in {"RUNNING", "COMPLETED"}:
        return palette.green(value)
    if value in {"STARTING", "STOPPING", "PAUSED"}:
        return palette.yellow(value)
    if value in {"FAILED", "INTERRUPTED", "STALE"} or (value == "RUNNING" and pid <= 0):
        return palette.red(value)
    return value


def render_dashboard(
    run_dir: Path,
    sampler: ProcessSampler,
    palette: Palette,
    refresh: float,
) -> str:
    status = _read_json(run_dir / "status.json")
    config = _read_json(run_dir / "config.json")
    metrics = _read_recent_metrics(run_dir / "metrics.jsonl")
    updates = [
        row
        for row in metrics
        if row.get("kind") == "training" and row.get("phase") == "updated"
    ]
    latest = updates[-1] if updates else status
    performance = _rolling_performance(updates)

    width = max(96, min(120, shutil.get_terminal_size((108, 30)).columns))
    content_width = width - 2
    total_steps = max(0, _as_int(status.get("total_steps") or config.get("total_steps")))
    steps = max(0, _as_int(status.get("steps") or latest.get("steps")))
    progress = steps / total_steps if total_steps else _as_float(status.get("progress"))
    pid = _as_int(status.get("pid"))
    resources = sampler.sample(pid, run_dir)

    loss_values = [_as_float(row.get("loss")) for row in updates if row.get("loss") is not None]
    trend_name, trend_change = _trend(loss_values[-20:])
    trend_label = f"{trend_name} {trend_change:+.2f}%"
    if trend_name == "DOWN":
        trend_label = palette.green(trend_label)
    elif trend_name == "UP":
        trend_label = palette.yellow(trend_label)

    speed = performance["speed"]
    if speed is None or speed <= 0:
        speed = _as_float(status.get("steps_per_second")) or None
    eta = (total_steps - steps) / speed if speed and total_steps > steps else 0.0 if total_steps else None
    distributed = status.get("distributed") or {}
    world_size = _as_int(distributed.get("world_size") or status.get("world_size"), 1)
    device = str(distributed.get("device") or config.get("device") or "cpu")
    mode = f"DDP x{world_size}" if world_size > 1 else "single process"
    heartbeat = _heartbeat_age(status)

    lines = [
        palette.cyan("PocketAgent MAPPO") + "  |  LIVE TRAINING",
        "-" * width,
        f"Run    : {run_dir.name}   {mode} / {device.upper()}",
        (
            f"Status : {_status_text(status, palette, refresh)}   PID {pid or '--'}   "
            f"Phase {status.get('phase') or '--'}   Heartbeat "
            f"{heartbeat:.0f}s ago" if heartbeat is not None else
            f"Status : {_status_text(status, palette, refresh)}   PID {pid or '--'}   "
            f"Phase {status.get('phase') or '--'}   Heartbeat --"
        ),
        f"Step   : {steps:,} / {total_steps:,}   ({progress:.3%})   Update {_as_int(status.get('updates')):,}",
        "         " + _progress_bar(progress, max(20, min(70, content_width - 9))),
        "",
        (
            f"Loss   : {_format_metric(latest.get('loss'), 7):<13} trend {trend_label:<18} "
            f"recent {_sparkline(loss_values, min(32, content_width // 3))}"
        ),
        (
            f"PPO    : policy {_format_metric(latest.get('policy_loss')):<11} "
            f"value {_format_metric(latest.get('value_loss')):<11} "
            f"entropy {_format_metric(latest.get('entropy')):<10} "
            f"KL {_format_metric(latest.get('approximate_kl'), 7)}"
        ),
        (
            f"Speed  : {speed:.2f} steps/s   {1.0 / speed:.4f} s/step   rolling ETA {_format_duration(eta)}"
            if speed and speed > 0 else
            "Speed  : warming up; waiting for a complete collect + optimize cycle"
        ),
        (
            f"Batch  : {_as_int(performance['samples']):,}/update   "
            f"collect {_format_interval(performance['collect'])}   "
            f"optimize {_format_interval(performance['optimize'])}"
        ),
        (
            f"Action : buy {_as_int(latest.get('buy_actions')):,}   "
            f"hold {_as_int(latest.get('hold_actions')):,}   "
            f"sell {_as_int(latest.get('sell_actions')):,}   "
            f"executed {_as_int(latest.get('executed_orders')):,}   "
            f"blocked {_as_int(latest.get('blocked_orders')):,}"
        ),
        f"Valid  : {_validation_summary(run_dir)}",
        f"Save   : {_checkpoint_summary(run_dir)}",
        "",
    ]

    if resources["capacity"] is not None:
        lines.append(
            f"CPU    : {resources['capacity']:.1f}% capacity, {resources['cores']:.1f} cores, "
            f"load {resources['load']:.1f}, PocketAgent RSS {_format_bytes(resources['rss'])}"
        )
    else:
        system_cpu = status.get("cpu_percent")
        lines.append(
            f"CPU    : sampler warming up, system {_format_metric(system_cpu, 1)}%, "
            f"load {_format_metric(resources['load'], 1)}"
        )
    lines.extend(_gpu_rows(device.lower().startswith("cuda")))

    error = status.get("error")
    if error:
        lines.extend(["", palette.red(f"Error  : {error}")])
    lines.extend(
        [
            "",
            palette.dim(
                f"Refresh {refresh:g}s | Ctrl+C exits dashboard only; training keeps running."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live terminal dashboard for PocketAgent training.")
    parser.add_argument("--run-dir", type=Path, help="Agent run directory; defaults to the latest active run.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit.")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    if args.refresh < 1.0:
        parser.error("--refresh must be at least 1 second to keep monitoring overhead negligible.")

    try:
        run_dir = args.run_dir.resolve() if args.run_dir else _latest_run(args.runs_root.resolve())
    except (OSError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if not (run_dir / "status.json").exists():
        parser.error(f"Agent status file not found: {run_dir / 'status.json'}")

    interactive = bool(sys.stdout.isatty() and not args.once)
    palette = Palette(enabled=bool(interactive and not args.no_color))
    sampler = ProcessSampler()
    if interactive:
        sys.stdout.write("\x1b[2J\x1b[H\x1b[?25l")
        sys.stdout.flush()
    try:
        while True:
            frame = render_dashboard(run_dir, sampler, palette, args.refresh)
            if interactive:
                sys.stdout.write("\x1b[H" + frame + "\x1b[J")
                sys.stdout.flush()
            else:
                print(frame)
            if args.once:
                return 0
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        return 0
    finally:
        if interactive:
            sys.stdout.write("\x1b[?25h\n")
            sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
