from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from research_layer.cli.elder_search import configs,run_stage,select_candidates,digest,save
from research_layer.elder_engine import ElderConfig


def protocol():
    return json.loads(Path('config/research/elder_search_v1.json').read_text(encoding='utf-8'))


def test_family_is_frozen_and_both_baselines_are_present():
    p=protocol();family=configs(p)
    assert len(family)==72
    for b in p['baselines']:
        expected={k:v for k,v in b.items() if k!='id'}
        assert any(all(getattr(c,k)==v for k,v in expected.items()) for c in family)


def test_resuming_completed_stage_preserves_frozen_selection(tmp_path,monkeypatch):
    from research_layer.cli import elder_search as mod
    c=ElderConfig();dates=['2022-01-03','2022-01-04']
    result={'id':'hist_efi2_position_protect50','config':c.to_dict(),'dates':dates,'nav':np.array([1.,1.01]),
            'metrics':{'cagr':.1,'maximum_drawdown':.02,'total_return':.01},
            'trades':{'closed':300,'win_rate':.4,'profit_factor':1.2}}
    monkeypatch.setattr(mod,'summarize',lambda *a,**k:result)
    bars={'AAPL':pd.DataFrame({'datetime':dates,'open':[10.,10.],'close':[10.,10.1]})}
    first=run_stage('train',['2022-01-01','2023-12-31'],[c],['AAPL'],{},bars,'snapshot','config',tmp_path)
    one=select_candidates(first,protocol(),tmp_path);h=digest(tmp_path/'train/results.json')
    second=run_stage('train',['2022-01-01','2023-12-31'],[c],['AAPL'],{},bars,'snapshot','config',tmp_path)
    two=select_candidates(second,protocol(),tmp_path)
    assert digest(tmp_path/'train/results.json')==h
    assert one==two
    with pytest.raises(ValueError,match='identity mismatch'):
        run_stage('train',['2022-01-01','2023-12-31'],[c],['AAPL'],{},bars,'changed','config',tmp_path)


def test_high_win_selection_requires_profit_not_only_hit_rate(tmp_path):
    rows=[{'id':'high_hit_loser','metrics':{'cagr':-.2,'maximum_drawdown':.3,'total_return':-.2},'trades':{'closed':300,'win_rate':.9,'profit_factor':.7}},
          {'id':'lower_hit_profit','metrics':{'cagr':.1,'maximum_drawdown':.1,'total_return':.1},'trades':{'closed':300,'win_rate':.4,'profit_factor':1.2}}]
    training={'candidates':rows};save(tmp_path/'train/results.json',training)
    chosen=select_candidates(training,protocol(),tmp_path)
    assert chosen['selected']['high_win']=='lower_hit_profit'
    assert chosen['selected']['balanced_top5'][0]=='lower_hit_profit'


def test_cash_diagnostic_checks_frozen_scope_and_labels_financing(tmp_path,monkeypatch):
    from research_layer.cli import elder_search as mod
    frozen={'selected_ids':['case_a']};save(tmp_path/'frozen_selection.json',frozen)
    dates=['2024-01-02','2024-01-03']
    cash={'selection_sha256':digest(tmp_path/'frozen_selection.json'),'rows':[{'id':'case_a','dates':dates,'nav':[1.,1.01]}]}
    save(tmp_path/'selected_cash_interest.json',cash)
    monkeypatch.setattr(mod,'assess_family',lambda *a,**k:{'limitations':['Idle cash earns zero in the engine; old assumption.'],'candidate_count':1})
    rf=pd.DataFrame({'RF':[.0001,.0001]},index=dates)
    result=mod.assess_selected_cash(protocol(),tmp_path,frozen,rf,dates)
    assert result['candidate_count']==1
    assert 'not all72' in result['scope']
    assert all('Idle cash earns zero' not in s for s in result['limitations'])
    assert any('French daily RF proxy' in s for s in result['limitations'])
    cash['rows'][0]['id']='unfrozen_winner';save(tmp_path/'selected_cash_interest.json',cash)
    with pytest.raises(ValueError,match='exactly the frozen'):
        mod.assess_selected_cash(protocol(),tmp_path,frozen,rf,dates)


def completed_fixture(tmp_path,monkeypatch,validation=True):
    """Completed checkpoints, created before any attempted rule change."""
    from research_layer.cli import elder_search as mod
    p=protocol();p['output']=str(tmp_path)
    config_path=tmp_path/'config.json';save(config_path,p)
    config_sha=digest(config_path)
    save(tmp_path/'registration.json',{'config_sha256':config_sha})
    snapshot=tmp_path/'input_snapshot.json';save(snapshot,{'fixture':'frozen prices'})
    snapshot_sha=digest(snapshot)
    cs=[ElderConfig(),ElderConfig(exit_mode='either')]
    dates=['2022-01-03','2022-01-04','2024-01-02','2024-01-03']
    bars={'AAPL':pd.DataFrame({'datetime':dates,'open':[10.]*4,'close':[10.,10.1,10.,10.1]})}
    def fake_summary(config,symbols,features,start,end,**kwargs):
        selected=dates[:2] if start.startswith('2022') else dates[2:]
        return {'id':mod.candidate_id(config),'config':config.to_dict(),'dates':selected,
            'nav':np.array([1.,1.01]),'metrics':{'cagr':.1,'maximum_drawdown':.02,'total_return':.01},
            'trades':{'closed':300,'win_rate':.4,'profit_factor':1.2}}
    monkeypatch.setattr(mod,'summarize',fake_summary)
    train=run_stage('train',p['train'],cs,['AAPL'],{},bars,snapshot_sha,config_sha,tmp_path)
    select_candidates(train,p,tmp_path)
    if validation:
        run_stage('validation',p['validation']['period'],cs,['AAPL'],{},bars,snapshot_sha,config_sha,tmp_path)
    monkeypatch.setattr(mod,'configs',lambda _:cs)
    monkeypatch.setattr(mod,'fetch_french_rf',lambda _: {'status':'success'})
    monkeypatch.setattr(mod,'prepare',lambda *_: (['AAPL'],{},bars,snapshot_sha))
    monkeypatch.setattr(mod,'verify_inputs',lambda *_: (['AAPL'],tmp_path,snapshot))
    return mod,p,cs,bars,snapshot_sha,config_sha,config_path


@pytest.mark.parametrize('stage',['validate','costs','analyze'])
@pytest.mark.parametrize('source',['research_layer/elder_engine.py','research_layer/elder_analysis.py',
                                   'research_layer/cli/elder_search.py','research_layer/technical.py'])
def test_cli_rejects_source_change_after_training(tmp_path,monkeypatch,stage,source):
    mod,p,cs,bars,snapshot_sha,config_sha,config_path=completed_fixture(tmp_path,monkeypatch,stage!='validate')
    real_digest=mod.digest
    monkeypatch.setattr(mod,'digest',lambda path:'changed source' if Path(path).resolve()==Path(source).resolve() else real_digest(path))
    monkeypatch.setattr(mod,'summarize',lambda *a,**k:pytest.fail('Changed rules reached execution'))
    monkeypatch.setattr(mod,'assess_family',lambda *a,**k:pytest.fail('Changed rules reached inference'))
    with pytest.raises(ValueError,match='source/configuration/data identity mismatch'):
        mod.main(['--config',str(config_path),'--stage',stage])
    if stage=='validate':assert not (tmp_path/'validation').exists()


def test_direct_validation_rejects_changed_source_before_writing(tmp_path,monkeypatch):
    mod,p,cs,bars,snapshot_sha,config_sha,_=completed_fixture(tmp_path,monkeypatch,False)
    real_digest=mod.digest
    monkeypatch.setattr(mod,'digest',lambda path:'changed' if str(path)=='research_layer/elder_engine.py' else real_digest(path))
    with pytest.raises(ValueError,match='source/configuration/data identity mismatch'):
        run_stage('validation',p['validation']['period'],cs,['AAPL'],{},bars,snapshot_sha,config_sha,tmp_path)
    assert not (tmp_path/'validation').exists()


@pytest.mark.parametrize('changed',['snapshot','configuration'])
def test_cross_stage_rejects_changed_data_or_configuration(tmp_path,monkeypatch,changed):
    mod,p,cs,_,snapshot_sha,config_sha,_=completed_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='configuration/data/period identity mismatch'):
        mod.require_frozen_context(cs,'changed' if changed=='snapshot' else snapshot_sha,
            'changed' if changed=='configuration' else config_sha,tmp_path)


@pytest.mark.parametrize('stage',['costs','analyze'])
def test_secondary_stage_rejects_validation_checkpoint_from_other_rules(tmp_path,monkeypatch,stage):
    mod,p,cs,_,_,_,config_path=completed_fixture(tmp_path,monkeypatch)
    checkpoint=tmp_path/'validation'/(mod.candidate_id(cs[0])+'.json')
    old=mod.read(checkpoint);old['identity']['engine_sha256']='another validation engine';save(checkpoint,old)
    with pytest.raises(ValueError,match='Completed validation source/configuration/data identity mismatch'):
        mod.main(['--config',str(config_path),'--stage',stage])


def test_legacy_completed_context_checks_preserve_all_artifacts(tmp_path,monkeypatch):
    mod,p,cs,_,snapshot_sha,config_sha,_=completed_fixture(tmp_path,monkeypatch)
    for stage in ['train','validation']:
        for config in cs:
            path=tmp_path/stage/(mod.candidate_id(config)+'.json')
            checkpoint=mod.read(path)
            del checkpoint['identity']['runner_sha256']
            del checkpoint['identity']['normalizer_sha256']
            save(path,checkpoint)
    before={str(path):(digest(path),path.stat().st_mtime_ns) for path in tmp_path.rglob('*.json')}
    frozen,train,validation=mod.require_frozen_context(cs,snapshot_sha,config_sha,tmp_path,
        training_period=p['train'],validation_period=p['validation']['period'])
    assert frozen['train_results_sha256']==digest(tmp_path/'train/results.json')
    assert train['stage']=='train' and validation['stage']=='validation'
    assert before=={str(path):(digest(path),path.stat().st_mtime_ns) for path in tmp_path.rglob('*.json')}


def test_new_checkpoint_records_runner_and_actual_normalizer(tmp_path,monkeypatch):
    mod,_,cs,_,_,_,_=completed_fixture(tmp_path,monkeypatch)
    for stage in ['train','validation']:
        for config in cs:
            identity=mod.read(tmp_path/stage/(mod.candidate_id(config)+'.json'))['identity']
            assert identity['runner_sha256']==digest(mod.__file__)
            assert identity['normalizer_sha256']==digest('research_layer/technical.py')


def test_new_training_checkpoint_rejects_runner_change_on_resume(tmp_path,monkeypatch):
    mod,p,cs,bars,snapshot_sha,config_sha,_=completed_fixture(tmp_path,monkeypatch,False)
    real_digest=mod.digest
    monkeypatch.setattr(mod,'digest',lambda path:'changed runner' if Path(path).resolve()==Path(mod.__file__).resolve() else real_digest(path))
    with pytest.raises(ValueError,match='Checkpoint identity mismatch'):
        run_stage('train',p['train'],cs,['AAPL'],{},bars,snapshot_sha,config_sha,tmp_path)
