from __future__ import annotations

from typing import Iterable

import pandas as pd

from agent_layer.data import AgentTimelineLoader, SingleSymbolEpisodeBuffer, TimelineKey
from agent_layer.environment.execution import AshareExecutor, ExecutionConfig
from agent_layer.environment.trading_env import AgentObservation, AshareTradingEnv
from agent_layer.portfolio import PortfolioAccount


class SingleSymbolTradingEnv(AshareTradingEnv):
    """Trading environment for one stock episode.

    It keeps the legacy AshareTradingEnv step/reward/execution semantics, but
    enforces a single-symbol universe and uses SingleSymbolEpisodeBuffer when
    the loader is backed by Agent tensor cache.  In cache mode, reset/step no
    longer calls DuckDB/Parquet; market steps are read from the in-memory buffer.
    """

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
        universe = tuple(getattr(loader, "universe", ()) or ())
        if len(universe) != 1:
            raise ValueError(
                "SingleSymbolTradingEnv requires exactly one symbol in the loader universe."
            )
        self._episode_buffer: SingleSymbolEpisodeBuffer | None = None
        reader = getattr(loader, "reader", None)
        if reader is not None:
            self._episode_buffer = reader.episode_buffer(
                universe[0],
                start=start,
                end=end,
                stages=stages,
            )
            self.loader = loader
            self.executor = AshareExecutor(execution_config)
            self.account = PortfolioAccount(
                universe,
                initial_cash=self.executor.config.initial_cash,
            )
            self.keys: list[TimelineKey] = self._episode_buffer.keys()
            if not self.keys:
                raise ValueError("The selected single-symbol Agent episode is empty.")
            self._index = 0
            self._market = None
            self._market_stream = None
            self.reward_scale = float(reward_scale)
            self.hurdle_rate_annual = float(hurdle_rate_annual)
            self.drawdown_penalty = float(drawdown_penalty)
            self.turnover_penalty = float(turnover_penalty)
            self.invalid_action_penalty = float(invalid_action_penalty)
            self._peak_nav = self.account.initial_cash
            return
        super().__init__(
            loader,
            start=start,
            end=end,
            stages=stages,
            execution_config=execution_config,
            reward_scale=reward_scale,
            hurdle_rate_annual=hurdle_rate_annual,
            drawdown_penalty=drawdown_penalty,
            turnover_penalty=turnover_penalty,
            invalid_action_penalty=invalid_action_penalty,
        )

    def reset(self) -> tuple[AgentObservation, dict[str, object]]:
        if self._episode_buffer is None:
            return super().reset()
        self._close_stream()
        self.account.reset()
        self._peak_nav = self.account.initial_cash
        self._index = 0
        self._market = self._load_market(self.keys[0])
        return self._observation(), self._info(None, reward=0.0)

    def prefetched_markets(self) -> tuple[object, ...]:
        if self._episode_buffer is None:
            return super().prefetched_markets()
        end = min(len(self.keys), self._index + 33)
        return tuple(self._episode_buffer.market_step(i) for i in range(self._index + 1, end))

    def _load_market(self, key: TimelineKey):
        if self._episode_buffer is None:
            return super()._load_market(key)
        market = self._episode_buffer.market_step(self._index)
        if market.decision_time != key.decision_time or market.stage != key.stage:
            raise RuntimeError("Single-symbol episode buffer returned an out-of-order decision.")
        self.account.start_trading_day(market.decision_time.date())
        self.account.mark_to_market(market.symbols, market.execution_prices)
        return market
