from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from strategy_layer.base import AccountView, BarView, OrderIntent, Strategy
from strategy_layer.lightweight import ContextMomentumStrategy, ConstrainedRandomStrategy, RidgeForecastStrategy


@dataclass
class CashStrategy:
    name: str = "cash"

    def on_close(self, market: BarView, account: AccountView) -> None:
        return None


@dataclass
class BuyHoldStrategy:
    """Enter at the next open; retry blocked entries while still flat."""

    name: str = "buy_hold"

    def on_close(self, market: BarView, account: AccountView) -> OrderIntent | None:
        if account.total_shares == 0:
            return OrderIntent(1, reason="enter_and_hold; retry_while_flat")
        return None


@dataclass
class SmaTrendStrategy:
    fast_window: int = 5
    slow_window: int = 20
    name: str = "sma_trend"

    def __post_init__(self) -> None:
        if isinstance(self.fast_window, bool) or isinstance(self.slow_window, bool):
            raise ValueError("SMA windows must be integers.")
        if int(self.fast_window) != self.fast_window or int(self.slow_window) != self.slow_window:
            raise ValueError("SMA windows must be integers.")
        self.fast_window, self.slow_window = int(self.fast_window), int(self.slow_window)
        if not 1 <= self.fast_window < self.slow_window <= 2000:
            raise ValueError("Require 1 <= fast_window < slow_window <= 2000.")

    def on_close(self, market: BarView, account: AccountView) -> OrderIntent | None:
        if len(market.closes) < self.slow_window:
            return None
        fast = sum(market.closes[-self.fast_window:]) / self.fast_window
        slow = sum(market.closes[-self.slow_window:]) / self.slow_window
        if fast > slow and account.total_shares == 0:
            return OrderIntent(1, reason="fast_sma_above_slow")
        if fast <= slow and account.total_shares > 0:
            return OrderIntent(-1, reason="fast_sma_not_above_slow")
        return None


@dataclass
class BreakoutStrategy:
    lookback: int = 20
    exit_window: int = 10
    name: str = "breakout"

    def __post_init__(self) -> None:
        for field in ("lookback", "exit_window"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or not 1 <= value <= 2000:
                raise ValueError(f"{field} must be an integer from 1 to 2000")
            setattr(self, field, int(value))

    def on_close(self, market: BarView, account: AccountView) -> OrderIntent | None:
        if len(market.closes) <= max(self.lookback, self.exit_window):
            return None
        current = market.closes[-1]
        if account.total_shares == 0 and current > max(market.closes[-self.lookback-1:-1]):
            return OrderIntent(1, reason="close_above_prior_closes")
        if account.total_shares > 0 and current < min(market.closes[-self.exit_window-1:-1]):
            return OrderIntent(-1, reason="close_below_prior_exit_range")
        return None


@dataclass
class MeanReversionStrategy:
    window: int = 20
    entry_deviation: float = 0.05
    name: str = "mean_reversion"

    def __post_init__(self) -> None:
        if isinstance(self.window, bool) or not isinstance(self.window, (int, float)) or not math.isfinite(self.window) or int(self.window) != self.window or not 2 <= self.window <= 2000:
            raise ValueError("window must be an integer from 2 to 2000")
        if isinstance(self.entry_deviation, bool) or not isinstance(self.entry_deviation, (int, float)) or not math.isfinite(self.entry_deviation) or not 0 < self.entry_deviation < 1:
            raise ValueError("entry_deviation must be between 0 and 1")
        self.window = int(self.window)

    def on_close(self, market: BarView, account: AccountView) -> OrderIntent | None:
        if len(market.closes) < self.window:
            return None
        average = sum(market.closes[-self.window:]) / self.window
        current = market.closes[-1]
        if account.total_shares == 0 and current <= average * (1 - self.entry_deviation):
            return OrderIntent(1, reason="close_below_mean_band")
        if account.total_shares > 0 and current >= average:
            return OrderIntent(-1, reason="close_reverted_to_mean")
        return None


def strategy_catalog() -> list[dict[str, Any]]:
    return [
        {"id": "sma_trend", "name": "sma_trend", "label": "均线趋势", "parameters": {"fast_window": 5, "slow_window": 20},
         "parameter_grid": {"fast_window": [3, 5], "slow_window": [15, 20]},
         "description": "收盘快均线高于慢均线时持有，否则空仓；下一交易日开盘执行。"},
        {"id": "breakout", "name": "breakout", "label": "收盘突破", "parameters": {"lookback": 20, "exit_window": 10},
         "parameter_grid": {"lookback": [10, 20], "exit_window": [5, 10]},
         "description": "收盘突破此前窗口最高收盘价时入场，跌破退出窗口最低收盘价时离场。"},
        {"id": "mean_reversion", "name": "mean_reversion", "label": "均值回归", "parameters": {"window": 20, "entry_deviation": 0.05},
         "parameter_grid": {"window": [10, 20], "entry_deviation": [0.03, 0.05]},
         "description": "收盘偏离均线达到阈值时买入，回到均线时卖出；阈值 0.05 表示 5%。"},
        {"id": "context_momentum", "name": "context_momentum", "label": "市场上下文动量",
         "parameters": {"lookback": 5, "min_index_return": 0.0, "min_industry_return": 0.0, "require_industry": 1},
         "parameter_grid": {"lookback": [5, 10], "min_index_return": [-0.02, 0], "min_industry_return": [0], "require_industry": [1]},
         "requires_context": True,
         "description": "观察当时已知的指数和历史行业上下文，缺失时阻止新买入，仍允许已知趋势转弱后的退出。"},
        {"id": "ridge_forecast", "name": "ridge_forecast", "label": "轻量 Ridge 预测",
         "parameters": {"lags": 5, "train_window": 60, "min_samples": 20, "horizon": 1, "alpha": 1.0, "threshold": 0.001, "refit_interval": 5},
         "parameter_grid": {"lags": [3, 5], "train_window": [60], "min_samples": [20], "horizon": [1], "alpha": [1, 10], "threshold": [0.001], "refit_interval": [5]},
         "description": "仅用已经成熟的历史标签在线拟合 CPU 线性模型；预测收盘到收盘收益，实际委托仍在下一开盘执行。"},
        {"id": "constrained_random", "name": "constrained_random", "label": "受约束随机对照",
         "parameters": {"seed": 2026}, "parameter_grid": {"seed": [2026, 2027]},
         "description": "固定 seed，在信号时已知账户与市场约束下抽取持有或买卖；下一开盘由统一执行器最终判定。"},
        {"id": "buy_hold", "name": "buy_hold", "label": "买入持有", "parameters": {},
         "description": "首个可执行开盘买入；被拒绝且仍空仓时重试，持有至期末估值。"},
        {"id": "cash", "name": "cash", "label": "保持现金", "parameters": {},
         "description": "始终空仓，作为同资金、同期间对照。"},
    ]


def create_strategy(name: str, parameters: dict[str, Any] | None = None) -> Strategy:
    params = dict(parameters or {})
    classes = {"cash": CashStrategy, "buy_hold": BuyHoldStrategy, "sma_trend": SmaTrendStrategy,
               "breakout": BreakoutStrategy, "mean_reversion": MeanReversionStrategy,
               "context_momentum": ContextMomentumStrategy, "constrained_random": ConstrainedRandomStrategy,
               "ridge_forecast": RidgeForecastStrategy}
    if name not in classes:
        raise ValueError(f"Unknown strategy {name!r}. Choose one of {', '.join(classes)}.")
    allowed = {"sma_trend": {"fast_window", "slow_window"}, "breakout": {"lookback", "exit_window"},
               "mean_reversion": {"window", "entry_deviation"},
               "context_momentum": {"lookback", "min_index_return", "min_industry_return", "require_industry"},
               "constrained_random": {"seed"},
               "ridge_forecast": {"lags", "train_window", "min_samples", "horizon", "alpha", "threshold", "refit_interval"}}.get(name, set())
    if set(params) - allowed:
        raise ValueError(f"Unknown parameters for {name}: {sorted(set(params) - allowed)}")
    return classes[name](**params)
