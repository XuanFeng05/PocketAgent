"""Neural-network runners are imported only when explicitly requested."""
from importlib import import_module

__all__ = [
    "EvaluationResult",
    "evaluate_policy",
    "SingleSymbolEvaluationSummary",
    "evaluate_checkpoint_by_symbol",
]


def __getattr__(name):
    if name in {"EvaluationResult", "evaluate_policy"}:
        return getattr(import_module("evaluation_layer.backtest.runner"), name)
    if name in {"SingleSymbolEvaluationSummary", "evaluate_checkpoint_by_symbol"}:
        return getattr(import_module("evaluation_layer.backtest.single_symbol_runner"), name)
    raise AttributeError(name)
