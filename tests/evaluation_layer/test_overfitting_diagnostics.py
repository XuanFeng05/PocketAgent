import numpy as np
import pytest

from evaluation_layer.metrics.overfitting import deflated_sharpe_ratio,probabilistic_sharpe_ratio,probability_of_backtest_overfitting


def test_dsr_uses_zero_null_not_observed_average_sharpe():
    one=deflated_sharpe_ratio(.4,[.4],100)
    assert one['selection_adjusted_benchmark_sharpe']==0
    assert one['deflated_sharpe_probability']==pytest.approx(probabilistic_sharpe_ratio(.4,0,100))
    a=deflated_sharpe_ratio(.4,[.1,.2,.3,.4],100)
    b=deflated_sharpe_ratio(.4,[1.1,1.2,1.3,1.4],100)
    assert a['selection_adjusted_benchmark_sharpe']==pytest.approx(b['selection_adjusted_benchmark_sharpe'])


def test_explicit_dependence_assumption_is_recorded():
    result=deflated_sharpe_ratio(.4,[.1,.2,.3,.4],100,independent_trials=2)
    assert result['trials']==4 and result['independent_trials_assumed']==2
    with pytest.raises(ValueError):deflated_sharpe_ratio(.4,[.1,float('nan')],100)


def test_cscv_rejects_nonfinite_and_unbounded_work():
    with pytest.raises(ValueError):probability_of_backtest_overfitting(np.array([[1,np.nan]]*10),partitions=4)
    with pytest.raises(ValueError):probability_of_backtest_overfitting(np.ones((30,3)),partitions=20)


def test_cscv_tie_rank_does_not_depend_on_column_order():
    rng=np.random.default_rng(42)
    a=rng.normal(.001,.01,40)
    matrix=np.column_stack([a,a,rng.normal(0,.01,40)])
    result=probability_of_backtest_overfitting(matrix,partitions=4)
    swapped=probability_of_backtest_overfitting(matrix[:,[1,0,2]],partitions=4)
    assert result==swapped
