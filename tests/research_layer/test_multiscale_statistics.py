import math

import numpy as np
import pandas as pd
import pytest

from research_layer.multiscale_statistics import (
    assess_multiscale_family, block_family_test, circular_bootstrap_means,
)


@pytest.mark.parametrize('n,block',[(23,5),(20,5),(11,11),(11,1)])
def test_sparse_block_means_equal_literal_date_resampling(n,block):
    values=np.random.default_rng(72).normal(size=(n,4))
    B=137;seed=18;rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(B,math.ceil(n/block)))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(B,-1)[:,:n]
    expected=values[ix].mean(axis=1)
    actual=np.concatenate([chunk for _,chunk in circular_bootstrap_means(values,B,block,seed,batch_size=7)])
    np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-13)


def test_duplicate_candidates_share_draws_and_have_equal_maxmean_p():
    x=np.random.default_rng(1).normal(.001,.01,size=80)
    result=block_family_test(np.column_stack([x,x]),['a','b'],['a'],B=199,block=10,seed=4)
    a,b=result['candidate_results']
    assert a['positive_excess_one_sided_p']==b['positive_excess_one_sided_p']
    assert a['positive_excess_one_sided_p']==a['single_step_max_mean_adjusted_p']
    assert a['mean_annualized_excess_rf_individual_ci95'] is not None
    assert b['mean_annualized_excess_rf_individual_ci95'] is None
    assert a['holm_adjusted_p']>=a['positive_excess_one_sided_p']


def test_holm_resolution_is_not_hidden_for_large_families():
    x=np.random.default_rng(4).normal(size=(30,100))
    r=block_family_test(x,[str(i) for i in range(100)],B=199,block=5)
    assert r['minimum_attainable_first_holm_p']==.5
    assert not r['holm_can_reject_at_5pct']


def test_missing_rf_and_selection_outside_family_are_errors():
    dates=pd.bdate_range('2024-01-01',periods=80).strftime('%Y-%m-%d').tolist()
    nav=np.cumprod(1+np.random.default_rng(9).normal(0,.005,size=(80,2)),axis=0)
    with pytest.raises(ValueError,match='Missing RF'):
        assess_multiscale_family(nav,dates,pd.Series(0.,index=dates[:-1]),['a'],['a','b'],B=199,blocks=(10,))
    with pytest.raises(ValueError,match='Selected IDs'):
        assess_multiscale_family(nav,dates,np.zeros(80),['winner_not_registered'],['a','b'],B=199,blocks=(10,))


def test_all_blocks_cover_full_family_and_trial_counts_are_disclosed():
    dates=pd.bdate_range('2024-01-01',periods=80).strftime('%Y-%m-%d').tolist()
    nav=np.cumprod(1+np.random.default_rng(42).normal(.0003,.01,size=(80,3)),axis=0)
    r=assess_multiscale_family(nav,dates,np.full(80,.0002),['b'],['a','b','c'],B=199,blocks=(10,20),lifetime_trials_minimum=1612)
    assert [x['candidate_count'] for x in r['block_sensitivity']]==[3,3]
    assert r['lifetime_recorded_strategy_trials_lower_bound']==1612
    assert 'Prior-close idle cash' in r['cash_interest']
    assert [s['assumed_effective_independent_trials'] for s in r['selected_PSR_DSR_diagnostics'][0]['DSR_scenarios']]==[1612,5000]


def test_exact_rf_cash_is_not_a_positive_alpha_from_roundoff():
    dates=pd.bdate_range('2024-01-01',periods=80).strftime('%Y-%m-%d').tolist()
    rf=np.linspace(.0001,.0003,80)
    nav=np.cumprod(1+rf)
    averaged=np.tile(nav[:,None],(1,500)).mean(axis=1)[:,None]
    r=assess_multiscale_family(averaged,dates,rf,['cash'],['cash'],B=199,blocks=(10,))
    case=r['block_sensitivity'][0]['candidate_results'][0]
    assert case['mean_annualized_excess_rf']==0.
    assert case['positive_excess_one_sided_p']==1.
    assert r['block_sensitivity'][0]['family_centered_max_mean_test']['p_one_sided']==1.
