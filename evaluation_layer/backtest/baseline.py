"""Executable legacy-timeline baselines without model or PyTorch imports."""
from typing import Any, Callable
import numpy as np
import pandas as pd
from evaluation_layer.metrics import portfolio_metrics
from evaluation_layer.backtest.valuation import execution_valuation


def buy_hold_baseline(loader: Any, *, symbol: str, start: str, end: str,
                      initial_nav: float, execution_config: Any = None,
                      stages: tuple[str, ...] | None = None,
                      cancel_check: Callable[[], bool] | None = None) -> dict[str, Any]:
    from agent_layer.actions import PortfolioAction
    from agent_layer.environment import ExecutionConfig, SingleSymbolTradingEnv
    config = execution_config or ExecutionConfig(initial_cash=initial_nav)
    if not np.isclose(config.initial_cash, initial_nav):
        raise ValueError("Baseline and candidate must use identical initial capital")
    selected = loader.for_universe((symbol,))
    env = SingleSymbolTradingEnv(selected, start=start, end=end, stages=stages,
                                 execution_config=config, enforce_action_contract=False)
    daily = {}
    fills = blocked = 0
    turnover = 0.0
    purchased = False
    try:
        observation, _ = env.reset()
        while observation is not None:
            if cancel_check and cancel_check():
                raise InterruptedError("Baseline evaluation cancellation requested")
            action = PortfolioAction(np.array([0 if purchased else 1]), np.array([0.0 if purchased else 1.0]))
            observation, _, terminated, truncated, info = env.step(action)
            execution = info["execution"]
            fills += len(execution.executed_fills)
            blocked += sum(fill.status == "blocked" for fill in execution.fills)
            turnover += execution.turnover_value
            purchased = purchased or bool(execution.executed_fills)
            snapshot = execution_valuation(info)
            daily[pd.Timestamp(snapshot["time"]).normalize()] = snapshot["nav"]
            if terminated or truncated:
                break
        metrics = portfolio_metrics([daily[d] for d in sorted(daily)], initial_nav=initial_nav,
                                    turnover_value=turnover, total_fees=env.account.total_fees)
        return {**metrics, "kind": "executable_buy_hold", "orders": fills,
                "blocked_orders": blocked, "purchased": purchased,
                "policy": "request maximum buy until first fill; then hold; no forced liquidation",
                "daily_nav": [{"date": str(d.date()), "nav": daily[d]} for d in sorted(daily)]}
    finally:
        env.close()
