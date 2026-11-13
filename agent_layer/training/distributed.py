from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass
class DistributedTrainingContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    backend: str | None = None
    cpu_threads: int = 1
    initialized: bool = False

    @classmethod
    def initialize(
        cls,
        *,
        strategy: str,
        requested_device: str,
        backend: str = "auto",
        cpu_threads_per_process: int = 0,
        timeout_seconds: int = 1800,
    ) -> "DistributedTrainingContext":
        world_size = int(os.environ.get("WORLD_SIZE") or 1)
        rank = int(os.environ.get("RANK") or 0)
        local_rank = int(os.environ.get("LOCAL_RANK") or 0)
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE") or world_size)
        if strategy == "off" and world_size > 1:
            raise RuntimeError("torchrun launched multiple ranks while distributed strategy is off.")
        if strategy == "ddp" and world_size <= 1:
            raise RuntimeError(
                "DDP requires torchrun with at least two processes. Start the run through "
                "the backend launcher or python -m torch.distributed.run."
            )

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA training was requested but CUDA is unavailable.")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} exceeds {torch.cuda.device_count()} visible CUDA devices."
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        cpu_count = max(1, int(os.cpu_count() or 1))
        cpu_threads = int(cpu_threads_per_process)
        if cpu_threads <= 0:
            cpu_threads = max(1, cpu_count // max(1, local_world_size))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
        except RuntimeError:
            pass

        initialized = False
        resolved_backend: str | None = None
        if world_size > 1:
            if not torch.distributed.is_available():
                raise RuntimeError("This PyTorch build does not include torch.distributed.")
            resolved_backend = backend
            if resolved_backend == "auto":
                resolved_backend = "nccl" if device.type == "cuda" else "gloo"
            if resolved_backend == "nccl" and not torch.distributed.is_nccl_available():
                raise RuntimeError(
                    "NCCL is unavailable. CUDA DDP requires a Linux PyTorch build with NCCL; "
                    "use Gloo only for CPU diagnostics."
                )
            if resolved_backend == "gloo" and not torch.distributed.is_gloo_available():
                raise RuntimeError("This PyTorch build does not include the Gloo backend.")
            torch.distributed.init_process_group(
                backend=resolved_backend,
                init_method=os.environ.get("POCKETAGENT_DDP_INIT_METHOD", "env://"),
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=int(timeout_seconds)),
            )
            initialized = True

        return cls(
            device=device,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            backend=resolved_backend,
            cpu_threads=cpu_threads,
            initialized=initialized,
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def wrap(self, module: nn.Module) -> nn.Module:
        if not self.enabled:
            return module
        kwargs: dict[str, Any] = {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
        }
        if self.device.type == "cuda":
            kwargs.update(device_ids=[self.local_rank], output_device=self.local_rank)
        return DistributedDataParallel(module, **kwargs)

    def all_reduce_float(self, value: float, *, reduction: str = "sum") -> float:
        if not self.enabled:
            return float(value)
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        operation = {
            "sum": torch.distributed.ReduceOp.SUM,
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }.get(reduction)
        if operation is None:
            raise ValueError(f"Unsupported distributed reduction: {reduction}")
        torch.distributed.all_reduce(tensor, op=operation)
        result = float(tensor.item())
        return result / self.world_size if reduction == "mean" else result

    def all_reduce_floats(
        self,
        values: list[float] | tuple[float, ...],
        *,
        reduction: str = "sum",
    ) -> list[float]:
        if not self.enabled:
            return [float(value) for value in values]
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        operation = {
            "sum": torch.distributed.ReduceOp.SUM,
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }.get(reduction)
        if operation is None:
            raise ValueError(f"Unsupported distributed reduction: {reduction}")
        torch.distributed.all_reduce(tensor, op=operation)
        return [float(value) for value in tensor.tolist()]

    def mean_float(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        return self.all_reduce_float(value, reduction="sum") / self.world_size

    def sum_int(self, value: int) -> int:
        return int(round(self.all_reduce_float(float(value), reduction="sum")))

    def any(self, value: bool) -> bool:
        return bool(self.all_reduce_float(float(bool(value)), reduction="max"))

    def all_gather_object(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        gathered: list[Any] = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def barrier(self) -> None:
        if self.enabled and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def close(self) -> None:
        if self.initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            self.initialized = False

    def payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "backend": self.backend,
            "device": str(self.device),
            "cpu_threads": self.cpu_threads,
        }
