from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from research_layer.cli import multiscale_search as m
from research_layer.multiscale_engine import MultiscaleConfig


def protocol():
    return json.loads(Path('config/research/multiscale_us500_v1.json').read_text())


def fake_run(window,config,**kwargs):
    p=window.profit;interest=.02
    metric=dict.fromkeys(m.SUM_FIELDS,0.)
    metric.update(total_return=p+interest,closed_trades=1,wins=int(p>0),losses=int(p<0),
        gross_profit_cash=max(p,0),gross_loss_cash=max(-p,0),cash_interest=interest,
        closed_cash_pnl=p,sum_net_trade_returns=p,sum_R_multiples=p/.01,
        sum_win_returns=max(p,0),sum_loss_returns=min(p,0),mean_close_exposure=.5)
    return {'nav':np.array([1.01,1+interest+p]),'exposure':np.array([.5,.5]),'metrics':metric,
            'diagnostics':{'entries':1,'blocked_sessions':0,'risk_budget_breaches_at_assumed_stop':0}}


def test_frozen_family_1540_unique_with_all_baselines():
    p=protocol();cs=m.family_configs(p);ids={m.candidate_id(c) for c in cs}
    assert len(cs)==len(ids)==1540
    assert sum('&' in c.daily_signal for c in cs)==1260
    assert all(m.candidate_id(MultiscaleConfig(**b)) in ids for b in p['selection']['baselines'])


def test_aggregate_independent_sleeves_keeps_pnl_and_interest_separate(monkeypatch):
    monkeypatch.setattr(m,'run_strategy',fake_run)
    windows={'A':SimpleNamespace(dates=['2024-01-02','2024-01-03'],profit=.04),
             'B':SimpleNamespace(dates=['2024-01-02','2024-01-03'],profit=-.02)}
    r,nav,exposure,stocks=m.aggregate_case(MultiscaleConfig(),list(windows),windows,np.zeros(2),np.zeros(2))
    assert np.isclose(nav[-1],1.03)
    assert r['trades']['win_rate']==.5
    assert r['trades']['profit_factor']==2
    assert np.isclose(r['trades']['realized_pnl_per_initial_capital'],.01)
    assert r['cash_interest_per_initial_capital']==.02
    assert np.isclose(r['mean_capital_weighted_exposure'],.5)
    assert stocks.shape==(2,len(m.STOCK_FIELDS))


def candidate(identifier,win,excess,pf,closed=1200,stocks=150):
    return {'id':identifier,'metrics':{'mean_annualized_excess_rf':excess,'maximum_drawdown':.05,'total_return':excess+.03},
            'trades':{'closed':closed,'win_rate':win,'profit_factor':pf},'stocks_with_closed_trades':stocks}


def test_training_high_win_cannot_select_loser_or_tiny_sample(tmp_path):
    rows=[candidate('high_hit_loss',.9,-.02,.8),candidate('tiny_lucky',1.,.5,10.,100,5),
          candidate('lower_hit_profit',.55,.05,1.2)]
    training={'candidates':rows};m.save(tmp_path/'train/results.json',training)
    frozen=m.freeze_selection(training,protocol(),tmp_path)
    assert frozen['selected']['high_win']=='lower_hit_profit'
    assert frozen['selected']['balanced_top5'][0]=='lower_hit_profit'
    assert frozen['eligible_candidate_count']==2
    before=m.digest(tmp_path/'frozen_selection.json')
    assert m.freeze_selection(training,protocol(),tmp_path)==frozen
    assert m.digest(tmp_path/'frozen_selection.json')==before
    training['candidates'][2]['metrics']['total_return']=.7;m.save(tmp_path/'train/results.json',training)
    with pytest.raises(ValueError,match='signature changed'):
        m.require_frozen(tmp_path)


def test_stage_resume_preserves_frozen_summary_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'run_strategy',fake_run)
    monkeypatch.setattr(m,'benchmark',lambda *a,**k:{'nav':np.array([1.,1.01]),'kind':'fixture','metrics':{}})
    p=protocol();w=SimpleNamespace(dates=['2022-01-03','2022-01-04'],profit=.04)
    context={'identity':{'fixture':'v1'},'symbols':['A'],'dates':{'train':w.dates},
             'windows':{'train':{'A':w}},'rates':{'train':np.zeros(2)}}
    cs=[MultiscaleConfig()]
    first=m.run_stage('train',p,cs,context,'configsha',tmp_path);sha=m.digest(tmp_path/'train/results.json')
    second=m.run_stage('train',p,cs,context,'configsha',tmp_path)
    assert first==second and sha==m.digest(tmp_path/'train/results.json')
    context['identity']['fixture']='v2'
    with pytest.raises(ValueError,match='identity mismatch'):
        m.run_stage('train',p,cs,context,'configsha',tmp_path)


def test_validation_rejects_rule_change_after_training(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'run_strategy',fake_run)
    monkeypatch.setattr(m,'benchmark',lambda *a,**k:{'nav':np.array([1.,1.01]),'kind':'fixture','metrics':{}})
    p=protocol();w=SimpleNamespace(dates=['2022-01-03','2022-01-04'],profit=.04)
    context={'identity':{'engine_sha256':'v1'},'symbols':['A'],'dates':{'train':w.dates},
             'windows':{'train':{'A':w}},'rates':{'train':np.zeros(2)}}
    m.run_stage('train',p,[MultiscaleConfig()],context,'configsha',tmp_path)
    context['identity']['engine_sha256']='v2'
    with pytest.raises(ValueError,match='Train/validation source'):
        m.run_stage('validation',p,[MultiscaleConfig()],context,'configsha',tmp_path)
    assert not (tmp_path/'validation/stage_identity.json').exists()
