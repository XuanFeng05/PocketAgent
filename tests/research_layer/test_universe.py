from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backtest_layer.portfolio_data import demo_portfolio_bars
from research_layer.universe import run_universe


def test_independent_symbol_accounts_and_summary_do_not_create_portfolio_curve():
    frame=demo_portfolio_bars(symbols=['Z','A','B'],days=40)
    r=run_universe(frame,'sma_trend',{'fast_window':3,'slow_window':10},execution_config={'initial_cash':10000})
    assert r['mode']=='universe' and r['status']=='completed'
    assert r['summary']['total_symbols']==3
    assert r['selection_scope']=='selected_symbol_not_aggregate'
    assert r['display_symbol']=='A'
    assert 'equity_curve' not in r['summary']
    assert all(row['result']['strategy']['metrics']['initial_nav']==10000 for row in r['symbol_results'])
    values=[row['metrics']['total_return'] for row in r['symbol_results']]
    assert r['summary']['mean_return']==pytest.approx(sum(values)/3)
    assert r['strategy']==r['symbol_results'][0]['result']['strategy']


def test_invalid_symbol_data_and_source_failures_are_all_recorded():
    frame=demo_portfolio_bars(symbols=['A','B'],days=30)
    frame.loc[frame.symbol=='B','close']=float('nan')
    failed_source=frame.iloc[:1].copy()
    failed_source['symbol']='MISSING'
    failed_source['source_error']='No historical ST facts'
    frame=pd.concat([frame,failed_source],ignore_index=True)
    events=[]
    r=run_universe(frame,progress_callback=events.append)
    assert r['status']=='partial'
    assert r['summary']['completed_symbols']==1 and r['summary']['failed_symbols']==2
    assert len(events)==3
    failures={row['symbol']:row['error'] for row in r['symbol_results'] if row['status']=='failed'}
    assert 'finite' in failures['B']
    assert 'ST facts' in failures['MISSING']
    assert r['display_symbol']=='A'


def test_all_failed_symbols_remain_visible_without_fake_winner():
    r=run_universe(demo_portfolio_bars(days=30),parameters={'fast_window':20,'slow_window':5})
    assert r['status']=='failed'
    assert r['summary']['failed_symbols']==3
    assert r['summary']['mean_return'] is None
    assert r['strategy'] is None and r['baselines']=={}


def test_different_coverage_is_explicit_and_input_is_not_mutated():
    frame=demo_portfolio_bars(symbols=['A','B'],days=40)
    frame=frame.loc[~((frame.symbol=='B')&(frame.datetime<frame.datetime.unique()[10]))].copy()
    before=frame.copy(deep=True)
    r=run_universe(frame)
    pd.testing.assert_frame_equal(frame,before)
    assert r['summary']['periods_aligned'] is False
    assert len({d['bars'] for d in r['summary']['return_distribution']})==2


@pytest.mark.parametrize('value',[0,101,1.5,True])
def test_invalid_symbol_budget_rejected(value):
    with pytest.raises(ValueError):run_universe(demo_portfolio_bars(days=4),max_symbols=value)


def test_budget_excess_is_not_silent_truncation():
    with pytest.raises(ValueError,match='no symbols were dropped'):
        run_universe(demo_portfolio_bars(days=4),max_symbols=2)


def test_cancel_delivers_completed_symbol_and_stops_before_next():
    done=[]
    with pytest.raises(InterruptedError):
        run_universe(demo_portfolio_bars(days=30),progress_callback=lambda e:done.append(e),cancel_check=lambda:len(done)>=1)
    assert len(done)==1 and done[0]['symbol_result']['status']=='completed'


def test_permuted_rows_yield_same_deterministic_symbol_comparison():
    frame=demo_portfolio_bars(days=35)
    first=run_universe(frame)
    second=run_universe(frame.sample(frac=1,random_state=7))
    assert first==second
