"""Lazy bridge to legacy training; importing the dashboard never loads torch."""
from __future__ import annotations

from functools import partial
from importlib import import_module


class OptionalTrainingUnavailable(RuntimeError):
    pass


def _invoke(name: str, *args, **kwargs):
    try:
        controller = import_module("app_layer.backend.agent_controller")
    except ModuleNotFoundError as exc:
        if exc.name == "torch" or str(exc.name).startswith("torch."):
            raise OptionalTrainingUnavailable(
                "旧训练功能需要可选 PyTorch 环境。策略回测工作台无需安装训练依赖。"
            ) from exc
        raise
    return getattr(controller, name)(*args, **kwargs)


__all__ = [
    "agent_preflight", "agent_run_action", "agent_run_detail", "agent_run_preflight",
    "agent_run_records", "agent_spec_payload", "export_agent_model_to_evaluation_pool",
    "inspect_agent_cache_request", "list_agent_runs", "start_agent_cache_job",
    "start_agent_run", "start_agent_training_job",
]
for _name in __all__:
    globals()[_name] = partial(_invoke, _name)
