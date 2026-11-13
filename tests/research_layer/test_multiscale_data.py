"""Synthetic catalogue/unit fixtures only; never used as market research data."""
import pandas as pd
import pytest

from research_layer.multiscale_data import diagnose_frame, download_frozen, eligibility, parse_holdings


def catalogue(extra=''):
    header = 'Ticker,Name,Type,Sector,Asset Class,Market Value,Price,Location,Exchange,Currency\n'
    rows = [f'X{i},Company{i},EQUITY,Technology,Equity,{1000000-i},20,United States,NASDAQ,USD' for i in range(501)]
    return ('iShares Russell 1000 ETF\nFund Holdings as of,"Nov 11, 2026"\n' + header + '\n'.join(rows) + '\n' + extra).encode()


def test_official_catalogue_date_order_and_class_symbol_mapping():
    date, rows, excluded = parse_holdings(catalogue('BRK B,Berkshire,EQUITY,Financials,Equity,2000000,500,United States,NYSE,USD\n'))
    assert date == '2026-11-11'
    assert rows[0]['symbol'] == 'BRK-B'
    assert rows[0]['provider_ticker'] == 'BRK B'
    assert rows[1]['symbol'] == 'X0'
    assert not excluded


def test_catalogue_excludes_pennies_foreign_and_duplicate_with_reasons():
    extra = ('PEN,Penny,EQUITY,Technology,Equity,3000000,2,United States,NASDAQ,USD\n'
             'FOR,Foreign,EQUITY,Technology,Equity,3000000,30,Canada,NASDAQ,USD\n'
             'X0,Duplicate,EQUITY,Technology,Equity,3000000,30,United States,NASDAQ,USD\n')
    _, rows, excluded = parse_holdings(catalogue(extra))
    assert len(rows) == 501
    assert {r['reason'] for r in excluded} == {'holdings_date_price_below_10','non_US_USD_equity','duplicate_normalized_symbol'}
    with pytest.raises(ValueError,match='CSV header'):
        parse_holdings(b'<html>error page</html>')


def test_endpoint_screen_does_not_use_strategy_returns():
    config = {'minimum_price_USD':10,'minimum_last60_median_vendor_dollars':25000000}
    frame = pd.DataFrame({'vendor_close':[20.]*60,'volume':[2000000.]*60})
    good = eligibility(frame,config)
    assert good['eligible'] and good['last60_median_vendor_close_times_volume']==40000000
    bad = frame.copy();bad.loc[59,'vendor_close']=5
    assert eligibility(bad,config)['exclusion_reasons']==['requested_end_price_below_10']
    short = frame.head(59)
    assert 'fewer_than_60_daily_observations' in eligibility(short,config)['exclusion_reasons']


def test_frozen_retry_rejects_nonmembers_without_requesting_data(monkeypatch):
    original=[f'X{i}' for i in range(100)];whole=[f'X{i}' for i in range(500)]
    monkeypatch.setattr('research_layer.multiscale_data.load_symbols',lambda p,n:original if n==100 else whole)
    config={'universe_file':'unused500','preserve_universe_file':'unused100','fixed_universe_count':500}
    with pytest.raises(ValueError,match='unique members'):
        download_frozen(config,['FOREIGN'])
    with pytest.raises(ValueError,match='unique members'):
        download_frozen(config,['X1','X1'])


def test_quality_diagnostic_preserves_zeros_and_distinguishes_price_changes():
    frame=pd.DataFrame({'datetime':['2024-01-02','2024-01-03','2024-01-04'],
        'open':[10.,10.,9.],'high':[10.,10.,12.],'low':[10.,10.,8.],'close':[10.,10.,11.],
        'volume':[1000,0,0],'is_tradeable':[True,False,False]})
    original=frame.copy(deep=True)
    result,events=diagnose_frame('TEST',frame,{'symbol':'TEST'})
    assert result['zero_volume_count']==2 and result['zero_four_prices_equal']==1
    assert result['zero_with_intraday_range']==1 and result['zero_close_changed_from_prior']==1
    assert result['largest_abs_adjusted_close_return_all_requested_dates']['signed_return']==pytest.approx(.1)
    assert result['largest_abs_adjusted_open_gap_all_requested_dates']['signed_return']==pytest.approx(-.1)
    assert all(not e['is_tradeable'] for e in events)
    pd.testing.assert_frame_equal(frame,original)
