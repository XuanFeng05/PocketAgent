"""Synthetic execution/accounting fixtures, never reported as real returns."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research_layer import multiscale_engine as mod
from research_layer.multiscale_engine import MultiscaleConfig, prepare_arrays, prepare_window, run_strategy


def features(n=25):
    return pd.DataFrame({"datetime":pd.bdate_range("2024-01-01",periods=n),"open":100.,"high":101.,
        "low":99.,"close":100.,"volume":100000.,"atr14":2.,"tradeable":True,"ready":True,
        "universe_trade_eligible":True,"signal__a":True,"signal__b":True,"weekly__up":True,"weekly__none":True})


def run(f, **kwargs):
    cfg=MultiscaleConfig(daily_signal="a",weekly_filter="up",**kwargs)
    return run_strategy(f,cfg,start="2024-01-02",end="2024-12-31",detailed=True)


def test_next_open_entry_and_predetermined_fifth_session_close():
    f=features(7)
    f.loc[1,["open","high","low","close"]]=[102.,104.,101.,103.]
    f.loc[5,["open","high","low","close"]]=[104.,106.,103.,105.]
    r=run(f,exit_mode="hold5")
    t=r["trades"][0]
    assert t["signal_close_date"]=="2024-01-01"
    assert t["entry_date"]=="2024-01-02" and t["entry_price"]==102.
    assert t["exit_date"]=="2024-01-08" and t["exit_price"]==105.
    assert t["exit_reason"]=="scheduled_close" and t["holding_bars"]==5
    # A fresh entry can only occur on the following session.
    assert r["trades"][1]["entry_date"]=="2024-01-09"


def test_initial_stop_risk_contains_both_fees_and_no_leverage():
    f=features(2);f.loc[1,"low"]=94.
    r=run(f,exit_mode="hold5")
    t=r["trades"][0]
    assert t["initial_stop"]==96.
    assert t["planned_risk_cash"]==pytest.approx(.01)
    assert t["cash_pnl"]==pytest.approx(-.01)
    assert r["nav"][-1]==pytest.approx(.99)
    assert t["exit_reason"]=="entry_day_stop"
    assert r["diagnostics"]["risk_budget_breaches_at_assumed_stop"]==0
    highrisk=run(features(2),risk_fraction=1.)
    assert highrisk["trades"][0]["entry_cash_outlay"]==pytest.approx(1.)
    assert highrisk["decisions"][0]["cash"]>=0.


def test_gap_stop_uses_actual_open_and_can_exceed_budget():
    f=features(3);f.loc[2,["open","high","low","close"]]=[90.,92.,89.,91.]
    t=run(f)["trades"][0]
    assert t["exit_reason"]=="gap_stop" and t["exit_price"]==90.
    assert t["cash_pnl"] < -t["planned_risk_cash"]


def test_trailing_update_applies_next_session_and_never_loosens():
    f=features(4);f["atr14"]=1.
    f.loc[1,["open","high","low","close"]]=[100.,110.,99.,108.]
    f.loc[2,["open","high","low","close"]]=[106.,107.,104.,106.]
    r=run(f,exit_mode="trail")
    t=r["trades"][0]
    assert t["exit_date"]=="2024-01-03" and t["exit_price"]==105.
    assert r["decisions"][0]["stop_effective_next_session"]==105.
    assert f.low.iloc[1] < 105.  # Not retroactively stopped on the entry session.
    assert t["mfe"]==pytest.approx(.1)  # Open entry precedes the whole entry-day high.


def test_exit_day_excursions_exclude_prices_after_intraday_exit():
    f=features(3)
    f.loc[2,["open","high","low","close"]]=[100.,140.,95.,130.]
    t=run(f)["trades"][0]
    assert t["exit_reason"]=="protective_stop"
    assert t["excursion_exit_day_censored"]
    assert t["mfe"]==pytest.approx(.01)  # Exit-day high 140 is not falsely credited.
    assert t["mae"]==pytest.approx(-.04)


def test_weekly_failure_exits_only_trailing_variant():
    f=features(7);f.loc[1:,"weekly__up"]=False
    t=run(f,exit_mode="trail")["trades"][0]
    assert t["exit_date"]=="2024-01-03" and t["exit_reason"]=="weekly_filter_invalid"
    t=run(f,exit_mode="hold5")["trades"][0]
    assert t["exit_date"]=="2024-01-08" and t["exit_reason"]=="scheduled_close"


def test_pending_market_entry_waits_for_tradeable_session():
    f=features(4);f.loc[1,"volume"]=0.;f.loc[1:,"signal__a"]=False
    r=run(f)
    assert r["trades"][0]["entry_date"]=="2024-01-03"
    assert r["trades"][0]["signal_close_date"]=="2024-01-01"
    assert r["diagnostics"]["blocked_sessions"]==1
    f.loc[1,"weekly__up"]=False
    assert not run(f)["trades"]


def test_price_eligibility_rechecked_before_delayed_entry():
    f=features(4);f.loc[1,"volume"]=0.;f.loc[1:,"universe_trade_eligible"]=False
    r=run(f)
    assert not r["trades"]
    assert r["diagnostics"]["setups_cancelled_price_eligibility"]==1


def test_suspended_fixed_horizon_exit_remains_pending():
    f=features(8);f.loc[5,"volume"]=0.
    f.loc[6,["open","high","low","close"]]=[105.,107.,104.,106.]
    t=run(f,exit_mode="hold5")["trades"][0]
    assert t["holding_bars"]==6
    assert t["exit_date"]=="2024-01-09" and t["exit_price"]==105.
    assert t["exit_reason"]=="scheduled_exit_after_suspension"


def test_suspended_stop_is_not_fictitiously_filled():
    f=features(4);f.loc[2,["volume","low"]]=[0.,95.]
    f.loc[3,["open","high","low","close"]]=[99.,101.,98.,100.]
    t=run(f)["trades"][0]
    assert t["exit_date"]=="2024-01-04" and t["exit_price"]==99.
    assert t["exit_reason"]=="stop_after_suspension"


def test_stops_take_precedence_on_scheduled_exit_day():
    f=features(6);f.loc[5,"low"]=95.
    t=run(f,exit_mode="hold5")["trades"][0]
    assert t["exit_reason"]=="protective_stop" and t["exit_price"]==96.


def test_prelisting_and_warmup_remain_cash_on_shared_calendar():
    f=features(15);dates=f.datetime.iloc[1:]
    f=f.iloc[6:].reset_index(drop=True);f.loc[:2,"ready"]=False
    w=prepare_window(f,start="2024-01-02",end="2024-12-31",evaluation_dates=dates)
    r=run_strategy(w,MultiscaleConfig(daily_signal="a",weekly_filter="up"),cash_daily_rates=np.full(len(dates),.001))
    assert r["dates"]==dates.dt.strftime("%Y-%m-%d").tolist()
    np.testing.assert_allclose(r["nav"][:8],1.001**np.arange(1,9))
    assert r["diagnostics"]["prelisting_cash_sessions"]==5
    assert r["trades"][0]["entry_date"]=="2024-01-15"


def test_missing_internal_or_trailing_bars_fail_instead_of_filling():
    f=features(10);dates=f.datetime.iloc[1:]
    for damaged in [f.drop(index=5),f.iloc[:-1]]:
        with pytest.raises(ValueError,match="Missing interior/trailing"):
            prepare_window(damaged,start="2024-01-02",end="2024-12-31",evaluation_dates=dates)


def test_two_signal_and_mask_and_readiness_use_previous_close():
    f=features(6);f["signal__b"]=False;f.loc[3,"signal__b"]=True
    cfg=MultiscaleConfig(daily_signal="a&b",weekly_filter="up")
    r=run_strategy(f,cfg,start="2024-01-02",end="2024-12-31")
    assert r["trades"][0]["entry_date"]=="2024-01-05"
    f.loc[3,"ready"]=False
    assert not run_strategy(f,cfg,start="2024-01-02",end="2024-12-31")["trades"]


def test_prefix_is_unchanged_by_future_prices_and_signals():
    f=features(30);g=f.copy();g.loc[15:,["open","high","low","close"]]*=2.
    g.loc[15:,"signal__a"]=False;g.loc[15:,"weekly__up"]=False
    a=run(f,exit_mode="trail");b=run(g,exit_mode="trail")
    np.testing.assert_array_equal(a["nav"][:14],b["nav"][:14])


def test_cash_interest_uses_prior_cash_and_reconciles_with_closed_and_open_pnl():
    f=features(12);cfg=MultiscaleConfig(daily_signal="a",weekly_filter="up",exit_mode="hold5")
    rates=np.full(11,.001)
    r=run_strategy(f,cfg,start="2024-01-02",end="2024-12-31",cash_daily_rates=rates,detailed=True)
    expected=.001+sum(d["cash"]*.001 for d in r["decisions"][:-1])
    assert r["metrics"]["cash_interest"]==pytest.approx(expected)
    pnl=sum(t["cash_pnl"] if t["status"]=="closed" else t["unrealized_cash_pnl"] for t in r["trades"])
    assert r["nav"][-1]-1==pytest.approx(pnl+expected)
    assert r["metrics"]["total_fees"]==pytest.approx(sum(t["entry_fee"]+t.get("exit_fee",0.) for t in r["trades"]))
    fast=run_strategy(f,cfg,start="2024-01-02",end="2024-12-31",cash_daily_rates=rates,include_trades=False)
    assert not fast["trades"] and fast["metrics"]==r["metrics"]
    np.testing.assert_array_equal(fast["nav"],r["nav"])


def test_fee_breakeven_is_not_a_floating_point_win():
    f=features(6);f.loc[5,["high","close"]]=[101.,100.*1.001/.999]
    t=run(f,exit_mode="hold5")
    assert t["trades"][0]["cash_pnl"]==0.
    assert t["metrics"]["wins"]==0 and t["metrics"]["breakeven_count"]==1


@pytest.mark.parametrize("exit_mode",["hold5","trail"])
def test_python_fallback_matches_compiled_path(monkeypatch,exit_mode):
    f=features(30);f.loc[[1,6],"volume"]=0.
    f.loc[8,["open","high","low","close"]]=[90.,92.,89.,91.]
    cfg=MultiscaleConfig(daily_signal="a",weekly_filter="up",exit_mode=exit_mode)
    kw={"start":"2024-01-02","end":"2024-12-31","detailed":True,"cash_daily_rates":np.full(29,.001)}
    a=run_strategy(f,cfg,**kw)
    monkeypatch.setattr(mod,"_kernel",mod._kernel_python)
    b=run_strategy(f,cfg,**kw)
    np.testing.assert_allclose(a["nav"],b["nav"],rtol=1e-14)
    np.testing.assert_array_equal(a["exposure"],b["exposure"])
    assert a["trades"]==b["trades"] and a["metrics"]==b["metrics"] and a["decisions"]==b["decisions"]


@pytest.mark.parametrize("changes",[{"cost_bps":-1.},{"risk_fraction":0.},{"initial_stop_atr":0.},{"exit_mode":"oracle"}])
def test_bad_parameters_rejected(changes):
    with pytest.raises(ValueError):MultiscaleConfig(**changes)
