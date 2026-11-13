import json
from pathlib import Path

import pytest

from backtest_layer.data import demo_bars
from backtest_layer.engine import run_backtest
from research_layer.robustness import analyze_robustness,parameter_stability,selection_diagnostics


def test_cost_scenarios_and_fifo_concentration_use_real_reports(tmp_path):
    bars=demo_bars(days=100)
    report=run_backtest(bars,'sma_trend')
    result=analyze_robustness({'strategy':report},tmp_path,
        rerun=lambda execution:run_backtest(bars,'sma_trend',execution_config=execution),cost_multipliers=[1,2])
    assert len(result['cost_sensitivity'])==2
    assert result['cost_sensitivity'][0]['metrics']==report['metrics']
    assert result['yearly'][0]['return']==pytest.approx(report['metrics']['total_return'])
    assert result['trade_concentration']['unit']=='completed_round_trips'


def test_neighbor_stability_counts_only_adjacent_grid_values():
    trials=[{'candidate_id':str(i),'parameters':{'window':v},'status':'completed','metrics':{'total_return':v/100}} for i,v in enumerate([5,10,20])]
    result=parameter_stability({'trials':trials,'best_candidate_id':'0'})
    assert [x['candidate_id'] for x in result['neighbors']]==['1']


def test_diagnostics_keep_failed_attempt_count_and_use_saved_daily_returns(tmp_path):
    trials=[]
    for i,fast in enumerate([2,3,5]):
        report=run_backtest(demo_bars(),'sma_trend',{'fast_window':fast,'slow_window':20})
        path=tmp_path/str(i);path.mkdir();(path/'result.json').write_text(json.dumps(report),encoding='utf-8')
        trials.append({'candidate_id':str(i),'parameters':{'fast_window':fast},'status':'completed','artifact_path':str(i),'metrics':report['metrics']})
    trials.append({'candidate_id':'bad','parameters':{},'status':'failed','error':'invalid'})
    result=selection_diagnostics({'trials':trials,'best_candidate_id':'0'},tmp_path)
    assert result['status']=='computed'
    assert result['attempts']==4 and result['failed_attempts']==1
    assert result['dsr']['trials']==3
    assert 0<=result['pbo']['probability_of_backtest_overfitting']<=1
