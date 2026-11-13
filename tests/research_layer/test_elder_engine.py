"""Synthetic execution fixtures, never reported as market performance."""
import numpy as np
import pandas as pd
import pytest

from research_layer.elder_engine import ElderConfig, build_features, run_elder


def features(rows, force=None, hist=None):
    n=len(rows)
    f=pd.DataFrame(rows,columns=['open','high','low','close'])
    f['datetime']=pd.date_range('2024-01-01',periods=n,freq='B')
    f['symbol']='TEST';f['volume']=100000.;f['tradeable']=True;f['ready']=True
    f['force']=force if force is not None else [-1.]*n
    f['adr20']=2.;f['atr14']=2.
    f['weekly_hist']=hist if hist is not None else [2.]*n
    f['weekly_hist_previous']=1.
    f['weekly_ema13']=101.;f['weekly_ema13_previous']=100.
    f.attrs.update(force_span=2,warmup_weeks=100)
    return f


def run(f,**kwargs):
    return run_elder(f,ElderConfig(**kwargs),start='2024-01-02',end='2024-12-31',detailed=True)


def test_intraday_entry_stop_is_conservative_and_risk_includes_both_fees():
    f=features([(99,100,98,99),(99,102,97,101)])
    r=run(f,stop_mode='fixed')
    t=r['trades'][0]
    assert t['status']=='closed' and t['ambiguous_intraday_order']
    assert t['exit_reason']=='entry_day_stop_conservative'
    assert t['planned_risk_cash']==pytest.approx(.01)
    assert t['cash_pnl']==pytest.approx(-.01)
    assert r['nav'][-1]==pytest.approx(.99)
    assert r['metrics']['total_fees']==pytest.approx(t['entry_fee']+t['exit_fee'])
    assert r['diagnostics']['risk_budget_breaches_at_assumed_stop']==0


def test_open_entry_stop_has_known_order_and_cash_never_borrowed():
    f=features([(99,100,98,99),(103,105,97,101)])
    r=run(f,stop_mode='fixed',risk_fraction=1.)
    t=r['trades'][0]
    assert t['entry_phase']=='open' and not t['ambiguous_intraday_order']
    assert t['exit_reason']=='entry_day_stop'
    assert r['diagnostics']['cash_limited_entries']==1
    assert all(d['cash']>=0 for d in r['decisions'])
    assert t['entry_cash_outlay']==pytest.approx(1.)


def test_gap_stop_can_exceed_planned_risk_and_uses_open():
    f=features([(99,100,98,99),(101,103,100,102),(90,93,88,92)])
    r=run(f,stop_mode='fixed')
    t=r['trades'][0]
    assert t['exit_reason']=='gap_stop' and t['exit_price']==90
    assert t['cash_pnl'] < -t['planned_risk_cash']
    assert t['exit_price'] < t['initial_stop']


def test_trailing_stop_is_next_session_only_and_entry_day_high_is_excluded():
    f=features([(99,100,98,99),(100,130,99,110),(106,108,104,107)])
    r=run(f,stop_mode='protect50')
    t=r['trades'][0]
    assert t['entry_date']=='2024-01-02' and t['exit_date']=='2024-01-03'
    # Entry-day high 130 is not known to occur after the intraday entry.
    expected=100.01+.5*(110-100.01)
    assert r['decisions'][0]['stop_effective_next_session']==pytest.approx(expected)
    assert t['exit_price']==pytest.approx(expected)
    assert t['exit_reason']=='protective_stop'
    # Entry day's low was below the NEW stop, yet no retroactive exit occurred.
    assert f.low.iloc[1] < expected


def test_pending_setup_survives_force_positive_until_weekly_reversal():
    f=features([(99,100,98,99),(98,99,97,98),(98,101,97.5,100)],force=[-1,5,5])
    r=run(f,stop_mode='fixed')
    assert r['trades'][0]['entry_date']=='2024-01-03'
    assert r['trades'][0]['entry_price']==pytest.approx(99.01)
    assert r['diagnostics']['setups_preserved_after_force_reversal']==1
    f.loc[1,'weekly_hist']=0.
    r2=run(f,stop_mode='fixed')
    assert not r2['trades'] and r2['diagnostics']['setups_cancelled_weekly']==1


def test_equal_weekly_histogram_does_not_force_exit():
    f=features([(99,100,98,99),(101,104,100,103),(103,105,102,104),(104,106,103,105)],
               hist=[2,1,0,0])
    r=run(f,stop_mode='fixed')
    assert r['trades'][0]['exit_date']=='2024-01-04'
    assert r['trades'][0]['exit_reason']=='weekly_trend_reversed'


def test_zero_force_does_not_mean_positive_exit():
    f=features([(99,100,98,99),(101,104,100,103),(103,105,102,104),(104,106,103,105)],
               force=[-1,0,2,2])
    r=run(f,stop_mode='fixed',exit_mode='swing')
    assert r['trades'][0]['exit_date']=='2024-01-04'
    assert r['trades'][0]['exit_reason']=='force_positive'


def test_stop_never_relaxes_when_price_pulls_back():
    f=features([(99,100,98,99),(101,112,100,110),(111,115,109,114),(113,114,112,113),(113,115,112,114)])
    r=run(f,stop_mode='protect70')
    stops=[v for v in r['stop_path'] if v is not None]
    assert all(b>=a for a,b in zip(stops,stops[1:]))
    assert r['trades'][0]['status']=='open'


def market_fixture():
    calendar=pd.DataFrame({'date':pd.date_range('2023-01-02','2023-08-06')})
    calendar['is_trading_day']=calendar.date.dt.dayofweek<5
    calendar['session_close']=calendar.date+pd.Timedelta(hours=16)
    calendar.loc[~calendar.is_trading_day,'session_close']=pd.NaT
    dates=calendar.loc[calendar.is_trading_day,'date'].reset_index(drop=True)
    x=np.arange(len(dates));close=100+.12*x+3*np.sin(x/4)
    bars=pd.DataFrame({'datetime':dates,'symbol':'TEST','adjust':'pre','freq':'daily',
                       'open':close-.2,'high':close+1,'low':close-1,'close':close,'volume':100000.})
    return bars,calendar


def test_feature_and_execution_prefix_invariant_to_future_price_changes():
    bars,cal=market_fixture()
    f=build_features(bars,cal,warmup_weeks=2)
    altered=bars.copy();cutoff=bars.datetime.iloc[90]
    mask=altered.datetime>cutoff
    altered.loc[mask,['open','high','low','close']]*=3
    g=build_features(altered,cal,warmup_weeks=2)
    past=f.datetime<=cutoff
    pd.testing.assert_frame_equal(f.loc[past],g.loc[past])
    cfg=ElderConfig(warmup_weeks=2)
    a=run_elder(f,cfg,start='2023-02-01',end='2023-08-04')
    b=run_elder(g,cfg,start='2023-02-01',end='2023-08-04')
    before=np.asarray(a['dates'])<=str(cutoff.date())
    np.testing.assert_array_equal(a['nav'][before],b['nav'][before])


def test_weekly_features_only_publish_on_actual_last_session_close():
    bars,cal=market_fixture()
    # Synthetic early close followed by a closed Friday.
    th=pd.Timestamp('2023-03-16');fr=pd.Timestamp('2023-03-17')
    cal.loc[cal.date.eq(th),'session_close']=th+pd.Timedelta(hours=13)
    cal.loc[cal.date.eq(fr),['is_trading_day','session_close']]=[False,pd.NaT]
    bars=bars.loc[~bars.datetime.eq(fr)]
    f=build_features(bars,cal,warmup_weeks=2)
    wed=f.loc[f.datetime.eq('2023-03-15')].iloc[0]
    thu=f.loc[f.datetime.eq(th)].iloc[0]
    assert wed.weekly_available_at==pd.Timestamp('2023-03-10 16:00')
    assert thu.weekly_available_at==pd.Timestamp('2023-03-16 13:00')
    assert thu.weekly_completed_count==wed.weekly_completed_count+1


def test_missing_open_bar_is_not_treated_as_suspension():
    bars,cal=market_fixture()
    with pytest.raises(ValueError,match='Missing open-session'):
        build_features(bars.drop(index=40),cal,warmup_weeks=2)
    bars.loc[40,'volume']=0
    f=build_features(bars,cal,warmup_weeks=2)
    assert not f.loc[40,'tradeable']


@pytest.mark.parametrize('kwargs',[{'cost_bps':-1},{'cost_bps':-.5},{'risk_fraction':0},
                                    {'risk_fraction':1.1},{'entry_buffer_adjusted':0}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):ElderConfig(**kwargs)


def test_fee_adjusted_break_even_is_neither_win_nor_loss():
    f=features([(99,100,98,99),(101,101.1,100,101.06),(101.3,102,101.1,101.5)])
    f['adr20']=.05
    r=run(f,stop_mode='protect50')
    t=r['trades'][0]
    assert t['exit_price']==pytest.approx(101*1.001/.999)
    assert t['cash_pnl']==0 and t['net_return']==0 and t['r_multiple']==0
    assert r['metrics']['breakeven_count']==1
    assert r['metrics']['wins']==r['metrics']['losses']==0
    assert r['metrics']['win_rate']==0  # All closed trades remain in denominator.
    assert r['nav'][-1]==pytest.approx(1.)


def test_closed_pnl_plus_open_mark_reconciles_to_ledger_nav():
    f=features([(99,100,98,99),(101,103,100,102),(90,93,88,92),
                (92,94,91,93),(94,96,93,95)])
    r=run(f,stop_mode='fixed')
    closed=sum(t['cash_pnl'] for t in r['trades'] if t['status']=='closed')
    opened=sum(t['unrealized_cash_pnl'] for t in r['trades'] if t['status']=='open')
    assert len(r['trades'])>=2
    assert r['nav'][-1]-1==pytest.approx(closed+opened,abs=1e-12)
    assert r['metrics']['total_fees']==pytest.approx(sum(x['fee'] for x in r['fills']))


def test_cash_interest_credits_previous_close_cash_not_same_day_sale_proceeds():
    f=features([(99,100,98,99),(101,103,100,102),(90,93,88,92)])
    cfg=ElderConfig(stop_mode='fixed')
    r=run_elder(f,cfg,start='2024-01-02',end='2024-12-31',detailed=True,
                cash_daily_rates=np.array([.001,.002]))
    first_close_cash=r['decisions'][0]['cash']
    interest=.001+first_close_cash*.002
    assert r['metrics']['cash_interest']==pytest.approx(interest)
    assert r['nav'][-1]-1==pytest.approx(sum(t['cash_pnl'] for t in r['trades'])+interest)
    with pytest.raises(ValueError,match='align'):
        run_elder(f,cfg,start='2024-01-02',cash_daily_rates=[.001])
