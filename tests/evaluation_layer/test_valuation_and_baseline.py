from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from agent_layer.actions import PortfolioAction
from agent_layer.environment import ExecutionConfig, SingleSymbolTradingEnv
from evaluation_layer.backtest.baseline import buy_hold_baseline
from evaluation_layer.backtest.valuation import execution_valuation


def market(time, price, can_buy=True):
    return SimpleNamespace(
        decision_time=pd.Timestamp(time), stage="bar_close", symbols=("000001.SZ",),
        runtime_contract=(), execution_prices=np.array([price]),
        active_mask=np.array([True]), is_tradeable=np.array([True]),
        market_can_buy=np.array([can_buy]), market_can_sell=np.array([True]),
        is_st=np.array([False]), is_limit_up=np.array([False]), is_limit_down=np.array([False]),
        is_zero_volume=np.array([False]), liquidity_amount=np.array([1e9]),
        liquidity_volume=np.array([1e8]), limit_reference_close=np.array([price]), limit_pct=np.array([0.1]),
    )


class Loader:
    universe = ("000001.SZ",)

    def __init__(self, markets):
        self.markets = markets

    def timeline(self, **kwargs):
        return [SimpleNamespace(decision_time=m.decision_time, stage=m.stage) for m in self.markets]

    def stream_steps(self, keys):
        return iter(self.markets)

    def for_universe(self, symbols):
        assert tuple(symbols) == self.universe
        return self


def free_execution():
    return ExecutionConfig(initial_cash=10000, commission_rate=0, minimum_commission=0,
                           stamp_duty_rate=0, transfer_fee_rate=0, base_slippage_rate=0,
                           impact_coefficient=0)


def test_cross_day_snapshot_keeps_close_and_locked_lots_before_advance():
    loader = Loader([market("2024-01-02 15:00", 10), market("2024-01-03 09:30", 11),
                     market("2024-01-03 15:00", 10.5)])
    env = SingleSymbolTradingEnv(loader, execution_config=free_execution())
    env.reset()
    _, reward, _, _, info = env.step(PortfolioAction(np.array([1]), np.array([1.])))
    snapshot = execution_valuation(info)
    assert snapshot["nav"] == 10000
    assert snapshot["time"] == pd.Timestamp("2024-01-02 15:00")
    assert snapshot["accounts"]["000001.SZ"]["locked_shares"] == 1000
    assert snapshot["accounts"]["000001.SZ"]["sellable_shares"] == 0
    assert info["net_asset_value"] == 11000
    assert env.account.position("000001.SZ").sellable_shares == 1000
    assert reward == pytest.approx(np.log(1.1))  # RL next-state return is preserved.
    daily = {snapshot["time"].normalize(): snapshot["nav"]}
    for _ in range(2):
        _, _, _, _, info = env.step(PortfolioAction.hold(1))
        s = execution_valuation(info)
        daily[s["time"].normalize()] = s["nav"]
    assert list(daily.values()) == [10000, 10500]
    assert snapshot["accounts"]["000001.SZ"]["last_price"] == 10
    env.close()


def test_baseline_retries_rejected_buy_and_holds_with_real_execution_costs():
    loader = Loader([market("2024-01-02 15:00", 10, False), market("2024-01-03 15:00", 10),
                     market("2024-01-04 15:00", 11)])
    cfg = ExecutionConfig(initial_cash=10000)
    result = buy_hold_baseline(loader, symbol="000001.SZ", start="2024-01-02", end="2024-01-04",
                               initial_nav=10000, execution_config=cfg)
    assert result["orders"] == 1
    assert result["blocked_orders"] == 1
    assert result["total_fees"] > 0
    assert 0 < result["total_return"] < 0.1
    assert result["daily_nav"][0]["nav"] == 10000


def test_baseline_respects_position_cap_and_rejects_capital_mismatch():
    loader = Loader([market("2024-01-02 15:00", 10), market("2024-01-03 15:00", 11)])
    cfg = ExecutionConfig(initial_cash=10000, max_position_ratio=.2)
    result = buy_hold_baseline(loader, symbol="000001.SZ", start="2024-01-02", end="2024-01-03",
                               initial_nav=10000, execution_config=cfg)
    assert 0 < result["total_return"] < .02
    with pytest.raises(ValueError, match="initial capital"):
        buy_hold_baseline(loader, symbol="000001.SZ", start="2024-01-02", end="2024-01-03",
                          initial_nav=20000, execution_config=cfg)


def test_evaluation_package_and_baseline_import_without_torch():
    subprocess.run([sys.executable, "-c", "import evaluation_layer.backtest; import evaluation_layer.backtest.baseline; import sys; assert 'torch' not in sys.modules"], check=True)


def test_missing_execution_snapshot_is_not_silently_misdated():
    with pytest.raises(ValueError, match="snapshot"):
        execution_valuation({"net_asset_value": 11000})


def test_actual_policy_evaluation_records_the_execution_day_without_torch():
    # Execute the production evaluation function, substituting only tensor/model operations.
    import ast
    from dataclasses import dataclass
    from pathlib import Path
    from evaluation_layer.metrics import portfolio_metrics
    source = Path(__file__).parents[2] / "evaluation_layer/backtest/runner.py"
    nodes = ast.parse(source.read_text(encoding="utf-8")).body
    selected = [n for n in nodes if getattr(n, "name", "") in {"EvaluationResult", "evaluate_policy"}]
    for node in selected:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    ast.fix_missing_locations(module)

    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def detach(self): return self
        def cpu(self): return self
        def squeeze(self, axis): return Tensor(self.data.squeeze(axis))
        def numpy(self): return self.data
        def __getitem__(self, item): return Tensor(self.data[item])

    class Model:
        count = 0
        def to(self, device): pass
        def eval(self): pass
        def encode_market(self, **kwargs): return Tensor([[0.]])
        def act_from_market_state(self, **kwargs):
            buy = self.count == 0
            self.count += 1
            return SimpleNamespace(directions=Tensor([[int(buy)]]), sizes=Tensor([[float(buy)]]))

    namespace = dict(dataclass=dataclass, np=np, pd=pd, torch=SimpleNamespace(device=lambda x: x),
                     PortfolioAction=PortfolioAction, portfolio_metrics=portfolio_metrics,
                     execution_valuation=execution_valuation, market_steps_to_tensors=lambda *a, **kw: {},
                     observation_state_to_tensors=lambda *a, **kw: {})
    exec(compile(module, str(source), "exec"), namespace)
    env = SingleSymbolTradingEnv(Loader([market("2024-01-02 15:00", 10),
        market("2024-01-03 09:30", 11), market("2024-01-03 15:00", 10.5)]), execution_config=free_execution())
    result = namespace["evaluate_policy"](env, Model())
    assert list(result.daily_nav) == [{"date": "2024-01-02", "nav": 10000}, {"date": "2024-01-03", "nav": 10500}]
    assert result.metrics["total_return"] == pytest.approx(.05)
    env.close()
