from datetime import datetime, timedelta

import numpy as np
import pytest

from strategy_layer.base import AccountView, BarView
from strategy_layer.registry import create_strategy


def market(values,context=None):
    return BarView("DEMO",datetime(2025,1,1)+timedelta(days=len(values)),tuple(values),context or {})


def flat():
    return AccountView(100000,100000,0,0)


def test_bar_context_is_read_only_and_backwards_compatible():
    bar = BarView("X",datetime(2025,1,1),(10,))
    assert not bar.context
    with pytest.raises(TypeError):bar.context["x"]=1


def test_context_missing_blocks_entries_but_not_observed_stock_exit():
    strategy = create_strategy("context_momentum",{"lookback":1})
    assert strategy.on_close(market([10,11]),flat()) is None
    assert strategy.diagnostics()["missing_context_entry_observations"] == 1
    visible={"context_index_available":True,"context_industry_available":True,
             "context_index_ret_5bar":.02,"context_industry_ret_5bar":.01}
    assert strategy.on_close(market([10,11],visible),flat()).direction == 1
    held=AccountView(0,1000,100,100)
    assert strategy.on_close(market([11,10]),held).direction == -1


def test_random_seed_is_reproducible_and_obeys_observed_account_masks():
    first=create_strategy("constrained_random",{"seed":3})
    second=create_strategy("constrained_random",{"seed":3})
    observed=market([10,10])
    assert [first.on_close(observed,flat()) for _ in range(30)] == [second.on_close(observed,flat()) for _ in range(30)]
    blocked=create_strategy("constrained_random",{"seed":3})
    assert all(blocked.on_close(market([10],{"can_buy_at_close":False}),flat()) is None for _ in range(20))
    locked=AccountView(0,1000,100,0)
    assert all(blocked.on_close(observed,locked) is None for _ in range(20))
    held=AccountView(0,1000,100,100)
    actions=[blocked.on_close(observed,held) for _ in range(30)]
    assert all(action is None or action.direction == -1 for action in actions)


def test_ridge_waits_for_matured_labels_and_fits_scaler_only_on_past_samples():
    strategy=create_strategy("ridge_forecast",{"lags":2,"horizon":3,"min_samples":5,"train_window":20,"refit_interval":1})
    closes=np.asarray([10,10.1,10.3,10.2,10.4,10.7,10.6,10.8,10.9,11.2])
    strategy.on_close(market(closes[:9]),flat())
    assert strategy.diagnostics()["fit_count"] == 0
    strategy.on_close(market(closes),flat())
    details=strategy.diagnostics()
    assert details["training_samples"] == 5
    assert details["last_fit_observations"] == 10
    returns=closes[1:]/closes[:-1]-1
    expected=np.asarray([returns[i-2:i] for i in range(2,7)])
    np.testing.assert_allclose(strategy._mean,expected.mean(axis=0))
    assert details["gpu_required"] is False
    assert "close" in details["label_definition"]


@pytest.mark.parametrize("name,params",[
    ("ridge_forecast",{"alpha":0}), ("ridge_forecast",{"horizon":0}),
    ("ridge_forecast",{"min_samples":100,"train_window":20}),
    ("ridge_forecast",{"lags":float('nan')}), ("constrained_random",{"seed":-1}),
    ("context_momentum",{"require_industry":2}),
])
def test_invalid_model_and_context_parameters_are_rejected(name,params):
    with pytest.raises(ValueError):create_strategy(name,params)
