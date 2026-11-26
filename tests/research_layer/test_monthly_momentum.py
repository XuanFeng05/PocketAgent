import numpy as np
import pandas as pd
import pytest
from research_layer.monthly_momentum import build_monthly_decisions,_selection,_scale


def sample():
    dates=pd.bdate_range('2020-01-01','2022-03-31');i=np.arange(len(dates))
    bars={}
    for k in range(12):
        c=20*np.exp(i*(.0005+k*.00005))
        bars[f'S{k:02}']=pd.DataFrame({'datetime':dates,'symbol':f'S{k:02}','open':c,'high':c*1.01,'low':c*.99,'close':c,'volume':1000.,'freq':'daily','adjust':'pre'})
    cd=pd.date_range('2020-01-01','2022-04-30');cal=pd.DataFrame({'date':cd,'is_trading_day':cd.weekday<5})
    return bars,cal


def test_month_schedule_and_known_ranking():
    bars,cal=sample();d=build_monthly_decisions(bars,cal,start='2022-01-01',end='2022-03-31',policy='momentum10')
    assert [r['signal_date'] for r in d]==['2021-12-31','2022-01-31','2022-02-28']
    assert d[0]['metadata']['ranked_symbols']==[f'S{k:02}' for k in range(11,-1,-1)]
    assert set(d[0]['weights'])=={f'S{k:02}' for k in range(2,12)}
    assert all(w==.1 for w in d[0]['weights'].values())


@pytest.mark.parametrize('policy',['momentum10','buffer10','breadth10','vol12_10','defensive10','defensive5','equal100','cash'])
def test_future_prices_do_not_change_past_decisions(policy):
    bars,cal=sample();original=build_monthly_decisions(bars,cal,start='2022-01-01',end='2022-03-31',policy=policy)
    changed={s:f.copy() for s,f in bars.items()}
    for f in changed.values():f.loc[f.datetime>='2022-02-01',['open','high','low','close']]*=.1
    newer=build_monthly_decisions(changed,cal,start='2022-01-01',end='2022-03-31',policy=policy)
    assert original[:2]==newer[:2]


def test_buffer_retains_second_rank_band_not_top_only():
    names=['A','B','C','D','E']
    assert _selection(names,['C','E'],2,True)==['C','A']
    assert _selection(names,['C','E'],2,False)==['A','B']


def test_partial_slots_leave_cash():
    bars,cal=sample();bars={k:v for k,v in list(bars.items())[:3]}
    d=build_monthly_decisions(bars,cal,start='2022-01-01',end='2022-03-31',policy='momentum10')
    assert sum(d[0]['weights'].values())==pytest.approx(.3)


def test_portfolio_vol_uses_joint_returns_not_average_vol():
    x=np.tile([-.02,.02],32)[:63];h=pd.DataFrame({'A':x,'B':-x})
    scale,vol=_scale(h,{'A':.5,'B':.5},.5,True,True)
    assert vol==0 and scale==1
    assert _scale(h,{'A':.5,'B':.5},.49,True,True)[0]==0
    scale,vol=_scale(h,{'A':1},1,True,False)
    assert 0<scale<1 and scale==pytest.approx(.12/vol)


def test_missing_open_session_rejected():
    bars,cal=sample();bars['S00']=bars['S00'].drop(index=100)
    with pytest.raises(ValueError,match='session'):build_monthly_decisions(bars,cal,start='2022-01-01',end='2022-03-31',policy='cash')


def test_holiday_month_end_and_final_unfinished_month():
    bars,cal=sample();holiday=pd.Timestamp('2021-12-31')
    cal.loc[cal.date==holiday,'is_trading_day']=False
    bars={s:f.loc[f.datetime!=holiday] for s,f in bars.items()}
    out=build_monthly_decisions(bars,cal,start='2022-01-01',end='2022-02-15',policy='cash')
    assert [d['signal_date'] for d in out]==['2021-12-30','2022-01-31']
