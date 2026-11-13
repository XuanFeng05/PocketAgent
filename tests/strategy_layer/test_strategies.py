from datetime import datetime

import pytest

from strategy_layer.base import AccountView, BarView, OrderIntent
from strategy_layer.registry import create_strategy, strategy_catalog


def test_sma_uses_only_observed_closes_and_explicit_position_state():
    strategy = create_strategy("sma_trend", {"fast_window": 1, "slow_window": 2})
    flat = AccountView(10000, 10000, 0, 0)
    held = AccountView(0, 10000, 1000, 0)
    assert strategy.on_close(BarView("X", datetime(2025, 1, 2), (10,)), flat) is None
    assert strategy.on_close(BarView("X", datetime(2025, 1, 3), (10, 11)), flat).direction == 1
    assert strategy.on_close(BarView("X", datetime(2025, 1, 4), (10, 11, 9)), held).direction == -1


@pytest.mark.parametrize("name,parameters", [("unknown", {}), ("cash", {"fast_window": 2}), ("sma_trend", {"fast_window": 20, "slow_window": 5}), ("sma_trend", {"fast_window": 1.5})])
def test_invalid_strategy_config_is_rejected(name, parameters):
    with pytest.raises(ValueError): create_strategy(name, parameters)


def test_strategy_catalog_ids_are_stable():
    assert {item["id"] for item in strategy_catalog()} == {"cash", "buy_hold", "sma_trend", "breakout", "mean_reversion",
                                                           "context_momentum", "ridge_forecast", "constrained_random"}


def test_order_intent_has_explicit_valid_fraction():
    with pytest.raises(ValueError): OrderIntent(1, float("nan"))
