from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from backtest_layer.data import demo_bars
from research_layer.screening import resolve_screen_settings,screen_candidates


def candidates(*params):
    return [{"candidate_id":f"c{i}","parameters":p} for i,p in enumerate(params)]


def simple(closes,opens=None):
    opens=opens or closes
    return pd.DataFrame([{'symbol':'X','date':f'2025-01-{i+1:02d}','open':op,'high':max(op,cl),'low':min(op,cl),
                          'close':cl,'volume':1000,'is_st':False} for i,(op,cl) in enumerate(zip(opens,closes))])


def test_labels_use_next_open_and_do_not_cross_training_cutoff():
    frame=simple([10,11,12,13,14,15,16,17],[10,10,10,10,10,10,10,10])
    r=screen_candidates(frame,'sma_trend',candidates({'fast_window':1,'slow_window':2}),{'train_bars':6,'forward_horizon':2})
    # Eligible signals t=1,2,3. t=4 would finish outside training and is excluded.
    assert r['trials'][0]['sample_indices']==[1,2,3]
    assert r['trials'][0]['metrics']['mean_forward_return']==pytest.approx(np.mean([.3,.4,.5]))
    assert r['trials'][0]['metrics']['hit_rate']==1
    assert max(s['exit_index'] for s in r['label_samples'])==5
    assert all(s['exit_time'][:10]<=r['training_range']['end'] for s in r['label_samples'])
    assert r['reserved_range']['evaluated'] is False
    assert 'equity_curve' not in r and 'nav' not in r


def test_reserved_tail_perturbation_does_not_change_screen_result():
    frame=demo_bars(days=100)
    params=candidates({'fast_window':3,'slow_window':15},{'fast_window':5,'slow_window':15})
    original=screen_candidates(frame,'sma_trend',params)
    changed=frame.copy()
    changed.loc[70:,['open','high','low','close']]*=3
    changed.loc[70:,'amount']*=7
    assert screen_candidates(changed,'sma_trend',params)==original


def test_window_cache_reuses_same_rolling_feature():
    r=screen_candidates(demo_bars(days=50),'mean_reversion',candidates({'window':10,'entry_deviation':.01},{'window':10,'entry_deviation':.05}),{'forward_horizon':2})
    assert r['cache_stats']['rolling_computations']==1
    assert r['cache_stats']['cache_hits']==1
    assert r['cache_stats']['keys']==['mean:10']


def test_breakout_excludes_current_close_and_exit_warmup_matches_rule():
    frame=simple([10,10,10,11,12,11,10,12,13,14])
    r=screen_candidates(frame,'breakout',candidates({'lookback':2,'exit_window':2}),{'train_bars':8,'forward_horizon':1})
    assert r['trials'][0]['sample_indices']==[3,4]
    assert r['trials'][0]['metrics']['signal_count']==2


def test_failure_and_zero_signal_candidates_remain_visible():
    r=screen_candidates(demo_bars(days=40),'mean_reversion',candidates({'window':5,'entry_deviation':.99},{'window':0,'entry_deviation':.05}),{'forward_horizon':2})
    assert len(r['trials'])==2 and r['status']=='partial'
    assert r['trials'][0]['metrics']['signal_count']==0
    assert r['trials'][0]['metrics']['mean_forward_return'] is None
    assert r['trials'][1]['status']=='failed'
    assert r['ranking']==[] and r['automatic_promotion'] is False


@pytest.mark.parametrize('settings',[{'forward_horizon':0},{'train_fraction':1},{'train_fraction':False},{'train_bars':1},
                                     {'max_candidates':65},{'unknown':1},[],{'forward_horizon':1.5}])
def test_invalid_screen_settings_rejected(settings):
    with pytest.raises(ValueError):resolve_screen_settings(settings)


def test_reserve_budget_and_label_length_guards():
    frame=demo_bars(days=20)
    params=candidates({'fast_window':1,'slow_window':2})
    with pytest.raises(ValueError,match='reserve'):screen_candidates(frame,'sma_trend',params,{'train_bars':20})
    with pytest.raises(ValueError,match='short'):screen_candidates(frame,'sma_trend',params,{'train_bars':3,'forward_horizon':3})
    with pytest.raises(ValueError,match='budget'):screen_candidates(frame,'sma_trend',params,{'max_bar_evaluations':2})
    with pytest.raises(InterruptedError):screen_candidates(frame,'sma_trend',params,cancel_check=lambda:True)


def test_screening_runs_when_torch_import_is_forbidden():
    script='''import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from backtest_layer.data import demo_bars
from research_layer.screening import screen_candidates
r=screen_candidates(demo_bars(days=40),'sma_trend',[{'candidate_id':'a','parameters':{'fast_window':3,'slow_window':10}}])
assert r['approximate'] and 'torch' not in sys.modules
'''
    r=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr
