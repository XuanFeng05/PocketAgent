from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from agent_layer.actions import PortfolioAction
from agent_layer.data import AgentMarketStep, AgentTimelineLoader, TimelineKey
from agent_layer.environment.execution import AshareExecutor, ExecutionConfig, ExecutionResult
from agent_layer.portfolio import PortfolioAccount
from agent_layer.rewards import net_asset_log_return


@dataclass(frozen=True)
class AgentObservation:
    market: AgentMarketStep
    runtime_state: np.ndarray


class AshareTradingEnv:
    def __init__(
        self,
        loader: AgentTimelineLoader,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stages: Iterable[str] | None = None,
        execution_config: ExecutionConfig | None = None,
        reward_scale: float = 1.0,
        hurdle_rate_annual: float = 0.0,
        drawdown_penalty: float = 0.0,
        turnover_penalty: float = 0.0,
        invalid_action_penalty: float = 0.0,
    ) -> None:
        self.loader = loader
        self.executor = AshareExecutor(execution_config)
        self.account = PortfolioAccount(
            loader.universe,
            initial_cash=self.executor.config.initial_cash,
        )
        self.keys: list[TimelineKey] = loader.timeline(start=start, end=end, stages=stages)
        if not self.keys:
            raise ValueError("The selected Agent timeline is empty.")
        self._index = 0
        self._market: AgentMarketStep | None = None
        self._market_stream: Iterator[AgentMarketStep] | None = None
        self.reward_scale = float(reward_scale)
        self.hurdle_rate_annual = float(hurdle_rate_annual)
        self.drawdown_penalty = float(drawdown_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self._peak_nav = self.account.initial_cash

    @property
    def current_market(self) -> AgentMarketStep:
        if self._market is None:
            raise RuntimeError("Environment must be reset before use.")
        return self._market

    def prefetched_markets(self) -> tuple[AgentMarketStep, ...]:
        peek = getattr(self._market_stream, "peek_buffer", None)
        return tuple(peek()) if callable(peek) else ()

    def reset(self) -> tuple[AgentObservation, dict[str, object]]:
        self._close_stream()
        self.account.reset()
        self._peak_nav = self.account.initial_cash
        self._index = 0
        stream_factory = getattr(self.loader, "stream_steps", None)
        self._market_stream = (
            stream_factory(self.keys)
            if callable(stream_factory)
            else iter(
                self.loader.load_step(key.decision_time, key.stage)
                for key in self.keys
            )
        )
        self._market = self._load_market(self.keys[0])
        observation = self._observation()
        return observation, self._info(None, reward=0.0)

    def step(
        self,
        action: PortfolioAction,
    ) -> tuple[AgentObservation | None, float, bool, bool, dict[str, object]]:
        market = self.current_market
        nav_before = max(self.account.net_asset_value, 1e-12)
        execution = self.executor.execute(self.account, market, action)
        terminated = self._index >= len(self.keys) - 1
        if terminated:
            nav_after = max(self.account.net_asset_value, 1e-12)
            reward = self._reward(nav_before, nav_after, execution, elapsed_days=0.0)
            info = self._info(execution, reward=reward)
            self._close_stream()
            return None, reward, True, False, info

        self.account.advance_holding_bars()
        self._index += 1
        self._market = self._load_market(self.keys[self._index])
        nav_after = max(self.account.net_asset_value, 1e-12)
        elapsed_days = max(
            0.0,
            (self._market.decision_time - market.decision_time).total_seconds() / 86400.0,
        )
        reward = self._reward(nav_before, nav_after, execution, elapsed_days=elapsed_days)
        return self._observation(), reward, False, False, self._info(execution, reward=reward)

    def _reward(
        self,
        nav_before: float,
        nav_after: float,
        execution: ExecutionResult,
        *,
        elapsed_days: float,
    ) -> float:
        previous_peak = max(self._peak_nav, nav_before)
        drawdown_before = max(0.0, 1.0 - nav_before / previous_peak)
        self._peak_nav = max(previous_peak, nav_after)
        drawdown_after = max(0.0, 1.0 - nav_after / self._peak_nav)
        drawdown_increase = max(0.0, drawdown_after - drawdown_before)
        turnover_ratio = execution.turnover_value / max(nav_before, 1e-12)
        blocked = sum(fill.status == "blocked" for fill in execution.fills)
        invalid_ratio = blocked / max(1, len(self.loader.universe))
        hurdle_log_return = np.log1p(self.hurdle_rate_annual) * max(0.0, elapsed_days) / 365.25
        return (
            self.reward_scale * (net_asset_log_return(nav_before, nav_after) - hurdle_log_return)
            - self.drawdown_penalty * drawdown_increase
            - self.turnover_penalty * turnover_ratio
            - self.invalid_action_penalty * invalid_ratio
        )

    def _load_market(self, key: TimelineKey) -> AgentMarketStep:
        if self._market_stream is None:
            raise RuntimeError("Agent market stream is not initialized.")
        market = next(self._market_stream)
        if market.decision_time != key.decision_time or market.stage != key.stage:
            raise RuntimeError("Agent market stream returned an out-of-order decision.")
        self.account.start_trading_day(market.decision_time.date())
        self.account.mark_to_market(market.symbols, market.execution_prices)
        return market

    def _close_stream(self) -> None:
        stream = self._market_stream
        close = getattr(stream, "close", None)
        if callable(close):
            close()
        self._market_stream = None

    def _observation(self) -> AgentObservation:
        market = self.current_market
        return AgentObservation(
            market=market,
            runtime_state=self.executor.runtime_state(self.account, market),
        )

    def _info(
        self,
        execution: ExecutionResult | None,
        *,
        reward: float,
    ) -> dict[str, object]:
        market = self.current_market
        return {
            "decision_time": market.decision_time,
            "stage": market.stage,
            "net_asset_value": self.account.net_asset_value,
            "cash": self.account.cash,
            "reward": reward,
            "execution": execution,
        }
