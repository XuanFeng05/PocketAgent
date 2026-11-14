"""US adaptation contracts: price provenance, sessions and same-day exit semantics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from download_layer.cli.download_us import normalize_chart, calendar_frame
from feature_layer.triple_screen import build_triple_features
import research_layer.triple_screen as engine
from research_layer.triple_runner import normalize_triple_config
from app_layer.backend.jobs import JobManager
from app_layer.backend.triple_controller import TripleDashboardService


def mock_features(frame, calendar, settings):
    out=frame.copy()
    out['weekly_hist']=1.;out['weekly_hist_previous']=.5;out['weekly_up']=True
    out['force_index_2']=[1.,-1.,1.];out['entry_ready']=[False,True,False]
    out['indicators_ready']=True;out['weekly_completed_count']=100
    out['session_close']=out.datetime+pd.Timedelta(hours=16 if settings.is_us else 15)
    out['weekly_available_at']=out.session_close
    return out


@pytest.mark.parametrize('opening,entry,ambiguous',[(11.,12.,True),(15.,15.,False)])
def test_us_entry_day_stop_exits_and_charges_both_sides(monkeypatch,opening,entry,ambiguous):
    monkeypatch.setattr(engine,'build_triple_features',mock_features)
    bars=pd.DataFrame({'symbol':'AAPL','datetime':pd.bdate_range('2025-01-06',periods=3),
        'open':[10.,10.,opening],'high':[11.,12.,17.],'low':[8.,9.,7.],'close':[10.,10.,13.],
        'volume':1000.,'freq':'daily','adjust':'pre'})
    result=engine.run_triple_screen(bars,None,{'version':'triple_screen_us_pre_v1','cost_bps':10})
    assert result['metrics']['final_nav']==pytest.approx(8/entry*.999**2)
    assert result['metrics']['final_exposure']==0
    assert result['metrics']['position_changes']==2
    assert result['trades'][0]['entry_date']==result['trades'][0]['exit_date']
    assert result['trades'][0]['ambiguous_intraday_order'] is ambiguous
    assert result['equity_curve'][0]['valuation_time'].endswith('T16:00:00')
    cn=engine.run_triple_screen(bars.assign(symbol='000001.SZ'),None,{'cost_bps':10})
    assert cn['metrics']['final_exposure']==1
    assert cn['equity_curve'][-1]['pending_exit_reason']=='entry_day_stop_t_plus_one'


def test_us_calendar_covers_special_closure_and_early_close():
    pytest.importorskip('exchange_calendars')
    calendar=calendar_frame('2024-11-18','2025-01-12')
    indexed=calendar.set_index('date')
    assert not indexed.loc['2025-01-09','is_trading_day']
    assert not indexed.loc['2024-11-28','is_trading_day']
    assert indexed.loc['2024-11-29','session_close'].endswith('13:00:00')
    days=calendar.loc[calendar.is_trading_day,'date']
    bars=pd.DataFrame({'symbol':'AAPL','datetime':days,'open':10.,'high':11.,'low':9.,'close':10.,'volume':1000.,'freq':'daily','adjust':'pre'})
    features=build_triple_features(bars,calendar,{'version':'triple_screen_us_pre_v1','warmup_weeks':2})
    black_friday=features.set_index('datetime').loc['2024-11-29']
    assert black_friday.weekly_available_at==pd.Timestamp('2024-11-29 13:00')
    with pytest.raises(ValueError,match='session_close'):
        build_triple_features(bars,calendar.drop(columns='session_close'),{'version':'triple_screen_us_pre_v1'})


def provider_fixture(volume=10000000.):
    raw={'chart':{'error':None,'result':[{'meta':{'symbol':'AAPL','currency':'USD','instrumentType':'EQUITY','exchangeName':'NMS','exchangeTimezoneName':'America/New_York','regularMarketPrice':20.},
          'timestamp':[int(pd.Timestamp('2022-01-03 14:30Z').timestamp())],
          'indicators':{'quote':[{'open':[9.],'high':[11.],'low':[8.],'close':[10.],'volume':[volume]}],'adjclose':[{'adjclose':[5.]}]}}]}}
    cal=pd.DataFrame({'date':[pd.Timestamp('2022-01-03')],'is_trading_day':[True]})
    return raw,cal


def test_us_adjustment_preserves_provider_and_volume():
    raw,cal=provider_fixture()
    bars,vendor,quality=normalize_chart(json.dumps(raw),'AAPL',cal)
    assert bars.iloc[0]['open']==4.5 and bars.iloc[0]['close']==5.
    assert vendor.iloc[0]['close']==10. and vendor.iloc[0]['adjustment_factor']==.5
    assert bars.iloc[0]['volume']==10000000.
    assert quality['price_basis']=='yahoo_split_dividend_adjusted_ohlc'


@pytest.mark.parametrize('damage',['missing_volume','penny','otc','missing_session'])
def test_provider_rejects_missing_data_and_unqualified_stocks(damage):
    raw,cal=provider_fixture(None if damage=='missing_volume' else 10000000.)
    meta=raw['chart']['result'][0]['meta']
    if damage=='penny':meta['regularMarketPrice']=2.
    if damage=='otc':meta['exchangeName']='PNK'
    if damage=='missing_session':cal.loc[1]=[pd.Timestamp('2022-01-04'),True]
    with pytest.raises(ValueError):normalize_chart(json.dumps(raw),'AAPL',cal)


def test_us_and_cn_symbol_namespaces_do_not_mix(tmp_path):
    base={'source_type':'csv','source_path':str(tmp_path/'data.csv'),'calendar_path':str(tmp_path/'calendar.csv'),'symbols':['AAPL'],'settings':{'version':'triple_screen_us_pre_v1'}}
    assert normalize_triple_config(base)['symbols']==['AAPL']
    with pytest.raises(ValueError):normalize_triple_config({**base,'settings':{}})
    with pytest.raises(ValueError):normalize_triple_config({**base,'symbols':['000001.SZ']})
    with pytest.raises(ValueError,match='calendar_path'):normalize_triple_config({**base,'source_type':'market','calendar_path':None})


@pytest.mark.parametrize('runner_status',['partial','failed'])
def test_dashboard_preserves_failed_task_status_and_report_error(tmp_path,monkeypatch,runner_status):
    import time
    # Import dependent modules before monkeypatching the shared runner. Otherwise
    # their module-level aliases can retain this fixture beyond monkeypatch undo.
    import research_layer.triple_validation
    import research_layer.triple_study
    data=tmp_path/'market'; marker=data/'market_parts/bucket=000/AAPL/daily_pre.parquet';marker.parent.mkdir(parents=True);marker.touch()
    jobs=JobManager();service=TripleDashboardService(jobs,data_root=data,output_root=tmp_path/'out',market='US')
    def runner(config,output,**kwargs):
        assert config['settings']['version']=='triple_screen_us_pre_v1'
        assert config['calendar_path']==str(data/'calendar.csv')
        path=Path(output)/'fixture';path.mkdir(parents=True)
        (path/'summary.json').write_text(json.dumps({'status':runner_status,'failed_tasks':1}))
        return {'run_id':'fixture','path':str(path),'status':runner_status}
    monkeypatch.setattr('research_layer.triple_runner.run_triple_experiment',runner)
    def report(_):raise RuntimeError('explicit report failure')
    monkeypatch.setattr('research_layer.triple_report.render_triple_report',report)
    try:
        job=service.start({'mode':'single','symbols':['AAPL']})
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and jobs.get_job(job['job_id'])['status'] in {'queued','running'}:time.sleep(.01)
        record=service.detail(job['job_id'])
        assert record['status']==runner_status==jobs.get_job(job['job_id'])['status']
        assert record['summary']['failed_tasks']==1
        assert 'explicit report failure' in record['report_warning']
    finally:jobs.shutdown()


@pytest.mark.parametrize('bad_time',['13:59:00','16:01:00','16:00:01','16:00:00.000001'])
def test_us_calendar_rejects_non_exact_session_close(bad_time):
    days=pd.date_range('2025-01-06','2025-01-19')
    cal=pd.DataFrame({'date':days,'is_trading_day':days.weekday<5})
    cal['session_close']=[f'{d.date()}T16:00:00' if d.weekday()<5 else None for d in days]
    cal.loc[0,'session_close']=f'2025-01-06T{bad_time}'
    dates=days[days.weekday<5]
    bars=pd.DataFrame({'symbol':'AAPL','datetime':dates,'open':10.,'high':11.,'low':9.,'close':10.,
                       'volume':1000.,'freq':'daily','adjust':'pre'})
    with pytest.raises(ValueError,match='session close'):
        build_triple_features(bars,cal,{'version':'triple_screen_us_pre_v1','warmup_weeks':2})
