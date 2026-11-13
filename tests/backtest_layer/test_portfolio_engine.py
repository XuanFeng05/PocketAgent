from __future__ import annotations

from copy import deepcopy
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from backtest_layer.account import PortfolioAccount
from backtest_layer.execution import AshareExecutor, ExecutionConfig
from backtest_layer.market import MarketSnapshot
from backtest_layer.portfolio import run_portfolio, run_portfolio_comparison
from backtest_layer.portfolio_data import demo_portfolio_bars, normalize_portfolio_bars, load_portfolio_bars
from backtest_layer.target_orders import execute_target_weights


def bars(prices=None, days=5):
    prices = prices or {"A": [10.] * days, "B": [20.] * days}
    records=[]
    for symbol, values in prices.items():
        for i, price in enumerate(values):
            if price is None:
                continue
            records.append({"symbol":symbol,"date":f"2025-01-{i+2:02d}","open":price,"high":price,"low":price,
                            "close":price,"volume":1_000_000,"amount":100_000_000,"is_st":False,
                            "is_member":True,"membership_known_at":f"2025-01-{i+2:02d} 00:00:00"})
    return pd.DataFrame(records)


def execution(**overrides):
    return ExecutionConfig(**{**dict(initial_cash=10_000, commission_rate=0, minimum_commission=0,
                                   stamp_duty_rate=0, transfer_fee_rate=0, base_slippage_rate=0,
                                   impact_coefficient=0, maximum_slippage_rate=0, auction_participation_rate=1), **overrides})


def settings(**overrides):
    return {**dict(strategy="equal_weight_hold", max_positions=2, max_weight=1, lookback=1, rebalance_every=1,
                   require_membership=True), **overrides}


def market(day="2025-01-03", symbols=("A","B"), prices=(10,10)):
    n=len(symbols)
    return MarketSnapshot(pd.Timestamp(day+" 09:30"), symbols, np.array(prices,dtype=float), np.full(n,100_000_000.),
                          np.array(prices,dtype=float),np.full(n,.1),np.ones(n,dtype=bool),np.ones(n,dtype=bool),
                          np.ones(n,dtype=bool),np.ones(n,dtype=bool))


def test_two_symbols_use_one_cash_ledger_and_fifo_reconciles():
    result=run_portfolio(bars(),settings(),execution())
    filled=[f for f in result["fills"] if f["status"]=="filled"]
    assert sum(f["gross_value"] for f in filled) <= 10_000
    assert sum(f["gross_value"] for f in filled) == 9_000 # 500 A + 200 B, lot rounding
    assert result["metrics"]["initial_nav"] == 10_000
    assert result["metrics"]["final_nav"] == 10_000
    assert min(row["cash"] for row in result["equity_curve"]) >= 0
    assert result["metrics"]["final_position_count"]==2
    f=result["fifo_report"]["summary"]
    assert f["realized_net_pnl"]+f["unrealized_net_pnl"] == pytest.approx(0)


def test_joint_buy_scaling_is_deterministic_and_fee_safe():
    frame=bars({"A":[10]*4,"B":[10]*4})
    cfg=execution(commission_rate=.001,minimum_commission=5)
    first=run_portfolio(frame,settings(),cfg)
    permuted=run_portfolio(frame.sample(frac=1,random_state=42),settings(),cfg)
    assert first==permuted
    assert first["metrics"]["total_fees"]>0
    assert all(f["shares"]%100==0 for f in first["fills"])
    assert min(r["cash"] for r in first["equity_curve"])>=-1e-8
    f=first["fifo_report"]["summary"]
    assert f["realized_net_pnl"]+f["unrealized_net_pnl"] == pytest.approx(first["metrics"]["final_nav"]-10_000)


def test_missing_bar_never_trades_or_uses_future_price_and_marks_stale():
    frame=bars({"A":[10,10,None,10,10],"B":[20]*5})
    r=run_portfolio(frame,settings(),execution())
    stale=[p for p in r["positions"] if p["symbol"]=="A" and p["date"]=="2025-01-04"][0]
    assert stale["is_stale"] is True and stale["mark_price"]==10
    assert r["metrics"]["stale_valuation_count"]==1
    assert not any(f["symbol"]=="A" and f["execution_time"].startswith("2025-01-04") for f in r["fills"])
    changed=frame.copy();changed.loc[(changed.symbol=='A')&(changed.date>='2025-01-05'),['open','high','low','close']]=19
    r2=run_portfolio(changed,settings(),execution())
    assert r["equity_curve"][:3]==r2["equity_curve"][:3]


def test_unseen_later_symbol_has_no_initial_price_or_allocation():
    r=run_portfolio(bars({"A":[10]*5,"B":[None,None,20,20,20]}),settings(),execution())
    assert not any(f["symbol"]=='B' for f in r["fills"])
    assert all(p["mark_price"]==0 for p in r["positions"] if p["symbol"]=='B' and p["date"]<'2025-01-04')


def test_momentum_rebalances_on_schedule_and_does_not_use_future_scores():
    frame=bars({"A":[10,10.1,10.2,10.3,10.4,10.5],"B":[20,20.1,20.2,20.3,20.4,20.5]})
    cfg=settings(strategy="momentum_top_k",max_positions=1,lookback=1,rebalance_every=2)
    r=run_portfolio(frame,cfg,execution())
    assert [s["signal_time"][:10] for s in r["signals"]]==['2025-01-02','2025-01-04','2025-01-06']
    assert r["signals"][1]["target_weights"]=={'A':1.0}
    changed=frame.copy();changed.loc[changed.date>='2025-01-06',['open','high','low','close']]*=1.1
    r2=run_portfolio(changed,cfg,execution())
    assert r["signals"][:2]==r2["signals"][:2]


def test_target_weight_adapter_respects_t_plus_one_and_sells_before_buys():
    cfg=execution(); ex=AshareExecutor(cfg); account=PortfolioAccount(('A','B'),initial_cash=10_000)
    first=execute_target_weights(account,market(),{'A':1},ex)
    assert account.position('A').total_shares==1000
    same=execute_target_weights(account,market(),{'B':1},ex)
    assert any(f["reason"]=='t_plus_one_locked' for f in same['fills'])
    assert account.position('B').total_shares==0
    next_day=execute_target_weights(account,market('2025-01-04'),{'B':1},ex)
    assert account.position('A').total_shares==0 and account.position('B').total_shares==1000
    assert [f['side'] for f in next_day['fills'] if f['status']=='filled']==['sell','buy']


@pytest.mark.parametrize('weights',[{'A':1,'B':1},{'A':-.1},{'X':.5},{'A':float('nan')},{'A':True}])
def test_invalid_target_weights_rejected(weights):
    with pytest.raises(ValueError):
        execute_target_weights(PortfolioAccount(('A','B'),initial_cash=10_000),market(),weights,AshareExecutor(execution()))


def test_historical_membership_is_as_of_and_static_pool_is_explicit():
    frame=bars()
    frame.loc[(frame.symbol=='A')&(frame.date=='2025-01-02'),'is_member']=False
    r=run_portfolio(frame,settings(),execution())
    assert r['signals'][0]['target_weights']=={'B':1.}
    static=frame.drop(columns=['is_member','membership_known_at'])
    with pytest.raises(ValueError,match='membership|is_member'):
        run_portfolio(static,settings(),execution())
    r=run_portfolio(static,settings(require_membership=False),execution())
    assert r['membership_sources']==['user_declared_static_pool']
    with pytest.raises(ValueError,match='after'):
        bad=frame.copy();bad.loc[0,'membership_known_at']='2030-01-01';normalize_portfolio_bars(bad)


def test_comparison_has_one_shared_initial_cash_per_report():
    r=run_portfolio_comparison(demo_portfolio_bars(days=45),settings(strategy='momentum_top_k',lookback=5),execution())
    assert set(r['baselines'])=={'cash','equal_weight_hold'}
    assert r['baselines']['cash']['metrics']['total_return']==0
    assert all(report['metrics']['initial_nav']==10_000 for report in [r['strategy'],*r['baselines'].values()])


def test_portfolio_csv_is_read_only_and_strictly_validated(tmp_path):
    p=tmp_path/'portfolio.csv';bars().to_csv(p,index=False)
    before=p.read_bytes();mtime=p.stat().st_mtime_ns
    frame=load_portfolio_bars(csv_path=p,require_membership=True)
    assert len(frame)==10
    assert before==p.read_bytes() and mtime==p.stat().st_mtime_ns
    with pytest.raises(ValueError,match='no data'):
        load_portfolio_bars(csv_path=p,symbols=['ABSENT'])


def test_portfolio_has_no_torch_dependency():
    code='''import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from backtest_layer.portfolio import run_portfolio_comparison
from backtest_layer.portfolio_data import demo_portfolio_bars
r=run_portfolio_comparison(demo_portfolio_bars(days=35))
assert r['strategy']['metrics']['initial_nav']==1000000
assert 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_cancel_portfolio_is_not_silently_successful():
    with pytest.raises(InterruptedError):
        run_portfolio(bars(),cancel_check=lambda:True)


def test_membership_effective_at_close_cannot_retroactively_block_open():
    frame=bars({"A":[10]*4,"B":[20]*4})
    second=(frame.symbol=='A')&(frame.date=='2025-01-03')
    frame.loc[second,'is_member']=False
    frame.loc[second,'membership_known_at']='2025-01-03 15:00:00'
    r=run_portfolio(frame,settings(),execution())
    assert any(f['symbol']=='A' and f['status']=='filled' for f in r['fills'])
    frame.loc[second,'membership_known_at']='2025-01-03 00:00:00'
    blocked=run_portfolio(frame,settings(),execution())
    first_day=[f for f in blocked['fills'] if f['symbol']=='A' and f['execution_time'].startswith('2025-01-03')]
    assert len(first_day)==1 and first_day[0]['reason']=='market_buy_blocked'


def test_weight_cap_retains_cash_and_hold_does_not_force_rebalance_price_drift():
    frame=bars({"A":[10,10,10.5,11],"B":[20]*4})
    r=run_portfolio(frame,settings(max_weight=.25),execution(max_position_ratio=.25))
    assert len(r['fills'])==2
    assert all(f['side']=='buy' for f in r['fills'])
    assert r['equity_curve'][-1]['cash']>=5_000


def test_rotation_nav_positions_and_fifo_reconcile_with_costs():
    frame=bars({'A':[10,10.2,10.4,10.6,10.8,10.5,10.3,10.1,10,10],
                'B':[20,20,20,20,20,20.2,20.5,21,21.5,22]})
    cfg=settings(strategy='momentum_top_k',max_positions=1,lookback=2,rebalance_every=2)
    r=run_portfolio(frame,cfg,execution(commission_rate=.001,minimum_commission=2,stamp_duty_rate=.0005))
    assert any(f['side']=='sell' and f['status']=='filled' for f in r['fills'])
    for equity in r['equity_curve']:
        positions=[p for p in r['positions'] if p['date']==equity['date']]
        assert equity['nav']==pytest.approx(equity['cash']+sum(p['market_value'] for p in positions))
        assert sum(p['weight'] for p in positions)<=1+1e-10
    fifo=r['fifo_report']['summary']
    assert fifo['fee_reconciliation_difference']==pytest.approx(0,abs=1e-8)
    assert fifo['realized_net_pnl']+fifo['unrealized_net_pnl']==pytest.approx(r['metrics']['final_nav']-r['metrics']['initial_nav'])


def test_static_membership_provenance_survives_repeated_normalization():
    frame=bars().drop(columns=['is_member','membership_known_at'])
    normalized=normalize_portfolio_bars(frame)
    normalized_again=normalize_portfolio_bars(normalized)
    assert set(normalized_again.membership_source)=={'user_declared_static_pool'}
    with pytest.raises(ValueError,match='Historical'):
        normalize_portfolio_bars(normalized,require_membership=True)
