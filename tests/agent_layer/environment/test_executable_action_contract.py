from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from agent_layer.actions import PortfolioAction
from agent_layer.data import AgentMarketStep
from agent_layer.data.cache_schema import REQUIRED_RUNTIME_CONTRACT_FIELDS
from agent_layer.environment import AshareExecutor, AshareTradingEnv, ExecutionConfig
from agent_layer.portfolio import PortfolioAccount


def _market(
    *,
    price: float,
    liquidity_amount: float,
    tradeable: bool = True,
    market_can_buy: bool = True,
    market_can_sell: bool = True,
) -> AgentMarketStep:
    return AgentMarketStep(
        decision_time=pd.Timestamp("2024-01-02 15:00:00"),
        stage="bar_close",
        symbols=("000001.SZ",),
        decision_ids=("decision",),
        active_mask=np.asarray([True]),
        execution_prices=np.asarray([price], dtype=np.float64),
        limit_reference_close=np.asarray([price], dtype=np.float64),
        limit_pct=np.asarray([0.1], dtype=np.float32),
        liquidity_volume=np.asarray([liquidity_amount / price], dtype=np.float64),
        liquidity_amount=np.asarray([liquidity_amount], dtype=np.float64),
        is_st=np.asarray([False]),
        market_can_buy=np.asarray([market_can_buy]),
        market_can_sell=np.asarray([market_can_sell]),
        is_tradeable=np.asarray([tradeable]),
        is_limit_up=np.asarray([False]),
        is_limit_down=np.asarray([False]),
        is_zero_volume=np.asarray([False]),
        market_sequences={"5min": np.zeros((1, 2, 1), dtype=np.float32)},
        sequence_masks={"5min": np.ones((1, 2), dtype=np.float32)},
        valid_ratios={"5min": np.ones(1, dtype=np.float32)},
        decision_context=np.zeros((1, 1), dtype=np.float32),
        feature_names={"5min": ("feature",)},
        decision_context_names=("context",),
        runtime_contract=REQUIRED_RUNTIME_CONTRACT_FIELDS,
        schema_hash="schema",
    )


def _runtime(executor: AshareExecutor, account: PortfolioAccount, market: AgentMarketStep) -> dict[str, float]:
    values = executor.runtime_state(account, market)[0]
    return dict(zip(market.runtime_contract, values, strict=True))


def test_advertised_buy_minimum_executes_across_account_boundaries() -> None:
    rng = random.Random(31)
    executor = AshareExecutor(ExecutionConfig())
    checked = 0

    for _ in range(1000):
        price = 10 ** rng.uniform(-0.2, 3.0)
        cash = 10 ** rng.uniform(2.5, 7.5)
        liquidity_amount = 10 ** rng.uniform(3.0, 10.0)
        market = _market(price=price, liquidity_amount=liquidity_amount)
        account = PortfolioAccount(market.symbols, initial_cash=cash)
        runtime = _runtime(executor, account, market)
        if runtime["can_buy_one_lot"] <= 0.5:
            continue

        checked += 1
        minimum = float(runtime["buy_one_lot_required_ratio"])
        size = np.float32(minimum + (1.0 - minimum) * 1e-6)
        result = executor.execute(
            account,
            market,
            PortfolioAction(np.asarray([1]), np.asarray([size])),
        )
        assert result.fills[0].status == "filled"
        assert result.fills[0].shares >= executor.config.lot_size

    assert checked > 500


def test_advertised_sell_minimum_executes_one_lot() -> None:
    executor = AshareExecutor(ExecutionConfig())
    market = _market(price=12.34, liquidity_amount=2_000_000.0)
    account = PortfolioAccount(market.symbols, initial_cash=100_000.0)
    position = account.position("000001.SZ")
    position.total_shares = 1_000
    position.sellable_shares = 1_000
    position.average_cost = 10.0
    position.last_price = 12.34
    runtime = _runtime(executor, account, market)

    result = executor.execute(
        account,
        market,
        PortfolioAction(
            np.asarray([-1]),
            np.asarray([runtime["sell_one_lot_required_ratio"]], dtype=np.float32),
        ),
    )

    assert runtime["can_sell_one_lot"] == 1.0
    assert result.fills[0].status == "filled"
    assert result.fills[0].shares >= executor.config.lot_size


def test_runtime_mask_uses_the_same_market_rejection_as_execution() -> None:
    executor = AshareExecutor(ExecutionConfig())
    market = _market(price=10.0, liquidity_amount=1_000_000.0, tradeable=False)
    account = PortfolioAccount(market.symbols, initial_cash=100_000.0)
    runtime = _runtime(executor, account, market)

    assert runtime["can_buy_one_lot"] == 0.0
    assert runtime["can_sell_one_lot"] == 0.0


def test_strict_contract_turns_a_blocked_policy_action_into_an_error() -> None:
    market = _market(price=10.0, liquidity_amount=1_000_000.0)
    account = PortfolioAccount(market.symbols, initial_cash=100_000.0)
    executor = AshareExecutor(ExecutionConfig(initial_cash=100_000.0))
    execution = executor.execute(
        account,
        market,
        PortfolioAction(np.asarray([1]), np.asarray([0.0])),
    )

    with pytest.raises(RuntimeError, match="action-mask/size-bound contract error"):
        AshareTradingEnv._assert_action_contract(
            PortfolioAction(np.asarray([1]), np.asarray([0.0])),
            execution,
        )


def test_observation_tensor_contract_requires_minimum_size_fields() -> None:
    pytest.importorskip("torch", reason="Tensor conversion requires the optional training dependencies.")
    from agent_layer.models import observation_state_to_tensors

    market = _market(price=10.0, liquidity_amount=1_000_000.0)
    account = PortfolioAccount(market.symbols, initial_cash=100_000.0)
    executor = AshareExecutor(ExecutionConfig(initial_cash=100_000.0))
    runtime = executor.runtime_state(account, market)

    class Observation:
        pass

    observation = Observation()
    observation.market = market
    observation.runtime_state = runtime
    tensors = observation_state_to_tensors(observation)

    assert tensors["can_buy"].item() is True
    assert tensors["buy_min_size"].item() > 0.0
    assert tensors["sell_min_size"].item() > 1.0

