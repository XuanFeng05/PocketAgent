"""Compatibility exports; shared implementation lives in backtest_layer."""
from backtest_layer.execution import AshareExecutor, ExecutionConfig, ExecutionResult, TradeFill

__all__ = ["AshareExecutor", "ExecutionConfig", "ExecutionResult", "TradeFill"]
