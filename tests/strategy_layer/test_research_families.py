from datetime import datetime

import pytest

from strategy_layer.base import AccountView, BarView
from strategy_layer.registry import create_strategy


def bar(prices):
    return BarView("DEMO", datetime(2025, 1, 10, 15), tuple(prices))


def account(shares=0):
    return AccountView(10000, 10000, shares, shares)


def test_breakout_excludes_current_close_from_reference_window():
    policy = create_strategy("breakout", {"lookback": 3, "exit_window": 2})
    assert policy.on_close(bar([10, 10, 10, 11]), account()).direction == 1
    assert policy.on_close(bar([10, 10, 10, 10]), account()) is None
    assert policy.on_close(bar([10, 10, 10, 9]), account(100)).direction == -1
    assert policy.on_close(bar([10, 11]), account()) is None


def test_mean_reversion_entry_and_exit_have_separate_thresholds():
    policy = create_strategy("mean_reversion", {"window": 3, "entry_deviation": 0.05})
    assert policy.on_close(bar([10, 10, 8]), account()).direction == 1
    assert policy.on_close(bar([10, 9, 10]), account(100)).direction == -1
    assert policy.on_close(bar([10, 10, 9.9]), account()) is None


@pytest.mark.parametrize("strategy,parameters", [
    ("breakout", {"lookback": 0}), ("breakout", {"exit_window": 1.2}),
    ("mean_reversion", {"window": float('nan')}), ("mean_reversion", {"entry_deviation": 0}),
    ("mean_reversion", {"entry_deviation": 1}), ("mean_reversion", {"entry_deviation": float('inf')}),
])
def test_invalid_research_parameters_fail_explicitly(strategy, parameters):
    with pytest.raises(ValueError):
        create_strategy(strategy, parameters)
