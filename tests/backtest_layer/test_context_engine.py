import pandas as pd
import pytest

from backtest_layer.data import demo_bars
from backtest_layer.engine import run_backtest,run_comparison
from feature_layer.context import demo_context_sources,enrich_daily_bars


def test_context_strategy_runs_in_shared_executor_and_reports_coverage():
    bars=demo_bars(days=80)
    with pytest.raises(ValueError,match="point-in-time context"):
        run_backtest(bars,"context_momentum")
    enriched=enrich_daily_bars(bars,**demo_context_sources(bars))
    result=run_backtest(enriched,"context_momentum",{"min_index_return":-.1,"min_industry_return":-.1})
    assert result["context_coverage"]["benchmark_available_bars"]==len(bars)
    assert result["context_coverage"]["industry_available_bars"]==len(bars)
    assert result["fills"]
    assert result["fifo_report"]["summary"]["fee_reconciliation_difference"] == pytest.approx(0)


def test_ridge_future_changes_cannot_change_earlier_signals_or_fits():
    bars=demo_bars(days=100)
    params={"lags":3,"min_samples":10,"train_window":30,"refit_interval":2,"threshold":0}
    baseline=run_backtest(bars,"ridge_forecast",params)
    changed=bars.copy(deep=True)
    changed.loc[60:,["open","high","low","close"]]*=1.2
    mutated=run_backtest(changed,"ridge_forecast",params)
    boundary=pd.Timestamp(bars.iloc[60]["datetime"]).isoformat()
    fields=["signal_time","direction","reason","policy_details"]
    def prefix(result):
        return [{k:s[k] for k in fields} for s in result["signals"] if s["signal_time"] < boundary]
    assert prefix(baseline)==prefix(mutated)
    assert prefix(baseline)
    assert baseline["policy_details"]["fit_count"] > 0
    assert all(s["policy_details"]["last_fit_time"] <= s["signal_time"] for s in baseline["signals"])


def test_warmup_ridge_is_causal_and_random_baseline_is_optional():
    bars=demo_bars(days=80)
    result=run_comparison(bars.iloc[50:],"ridge_forecast",{"min_samples":10},warmup_bars=bars.iloc[:50],random_seed=7)
    assert set(result["baselines"]) == {"cash","buy_hold","constrained_random"}
    assert len(result["strategy"]["equity_curve"])==30
    assert result["strategy"]["policy_details"]["last_fit_observations"] <= 80
    again=run_comparison(bars.iloc[50:],"ridge_forecast",{"min_samples":10},warmup_bars=bars.iloc[:50],random_seed=7)
    assert result==again


def test_pre_enriched_input_cannot_claim_future_context_and_string_masks_are_not_truthy():
    bars=demo_bars(days=30)
    enriched=enrich_daily_bars(bars,**demo_context_sources(bars))
    masked=enriched.copy()
    masked['context_index_available']='false'
    result=run_backtest(masked,'context_momentum')
    assert result['fills']==[]
    assert result['context_coverage']['benchmark_available_bars']==0
    future=enriched.copy()
    future.loc[0,'context_decision_time']='2099-01-01T15:00:00'
    with pytest.raises(ValueError,match='future'):
        run_backtest(future,'context_momentum')
