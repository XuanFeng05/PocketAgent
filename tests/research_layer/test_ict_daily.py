"""Priced and causal contracts for the frozen daily ICT adaptation."""
import math
from copy import deepcopy
from datetime import datetime, time

import pandas as pd
import pytest

from research_layer.ict_daily import detect_ict_setups, run_ict_daily
import research_layer.ict_daily as engine


def sample(length=26):
    dates = pd.bdate_range("2020-01-01", periods=length)
    rows = [{"symbol":"000001.SZ","datetime":d,"open":10.1,"high":10.4,"low":9.8,"close":10.1,
             "volume":1000.,"adjust":"pre","freq":"daily"} for d in dates]
    frame = pd.DataFrame(rows)
    replace(frame,20,(10.05,10.2,9.5,9.95))
    replace(frame,21,(9.95,11.2,9.9,11.0))
    replace(frame,22,(11.0,11.3,10.6,11.1))
    for i in range(23,length):
        replace(frame,i,(10.9,11.1,10.7,10.8))
    natural = pd.date_range(dates[0], dates[-1])
    calendar = pd.DataFrame({"date":natural,"is_trading_day":natural.dayofweek < 5})
    return frame,calendar


def replace(frame,index,ohlc):
    frame.loc[index,["open","high","low","close"]] = ohlc


def run(frame,calendar,**kwargs):
    return run_ict_daily(frame,calendar,start=frame.loc[20,"datetime"],end=frame.iloc[-1]["datetime"],**kwargs)


def test_confirmed_pattern_prior_tr_and_no_entry_on_confirmation_day():
    frame,cal = sample(25)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    replace(frame,24,(10.6,12.3,10.4,12.0))
    result = run(frame,cal)
    assert len(result["setups"]) == 1
    setup = result["setups"][0]
    assert setup["sweep_index"] == 20 and setup["confirm_index"] == 22
    assert setup["fvg_lower"] == 10.2 and setup["fvg_upper"] == 10.6
    assert setup["limit"] == pytest.approx(10.4)
    assert setup["stop"] == 9.5
    assert setup["structure_line"] == 10.4
    # TR20 = .7; thirteen preceding TRs = .6, excluding displacement TR.
    assert setup["prior_tr_mean14"] == pytest.approx((13*.6+.7)/14)
    assert result["fills"][0]["date"] == frame.loc[23,"datetime"].date().isoformat()
    assert result["trades"][0]["exit_reason"] == "target"
    assert result["trades"][0]["target_price"] == pytest.approx(12.2)
    assert result["metrics"]["final_nav"] == pytest.approx(12.2/10.4*.999**2)
    assert result["metrics"]["winning_trade_ratio"] == 1
    assert result["metrics"]["position_changes"] == 2


def test_no_early_setup_from_large_second_candle():
    frame,cal = sample(24)
    shortened = run_ict_daily(frame,cal,start=frame.loc[20,"datetime"],end=frame.loc[21,"datetime"])
    assert shortened["setups"] == [] and shortened["fills"] == []
    assert shortened["metrics"]["final_nav"] == 1


def test_pre_start_sweep_cannot_become_post_start_setup():
    frame,cal=sample(25)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    assert len(detect_ict_setups(frame)[0]) == 1
    result=run_ict_daily(frame,cal,start=frame.loc[21,"datetime"],end=frame.iloc[-1]["datetime"])
    assert result["sweeps"] == [] and result["setups"] == [] and result["fills"] == []
    # At the first evaluation close, historical bars can still define a sweep.
    starting_on_sweep=run(frame,cal)
    assert starting_on_sweep["setups"][0]["sweep_date"] == starting_on_sweep["start"]


def test_order_expires_after_fifth_allowed_bar_and_never_fills_sixth():
    frame,cal = sample(30)
    replace(frame,28,(10.8,11,10.3,10.5))
    result = run(frame,cal)
    assert result["fills"] == []
    order = result["orders"][0]
    assert order["status"] == "expired" and order["expires_index"] == 27
    assert order["resolved_date"] == frame.loc[27,"datetime"].date().isoformat()


def test_fifth_allowed_bar_can_fill():
    frame,cal = sample(28)
    replace(frame,27,(10.8,11,10.3,10.5))
    result = run(frame,cal)
    assert result["orders"][0]["status"] == "filled"
    assert result["trades"][0]["entry_index"] == 27
    assert result["trades"][0]["status"] == "open"


def test_explicit_suspension_counts_toward_order_expiry():
    frame,cal = sample(29)
    frame.loc[23:27,"volume"] = 0
    replace(frame,28,(10.8,11,10.3,10.5))
    result = run(frame,cal)
    assert result["orders"][0]["status"] == "expired" and not result["fills"]


def test_same_day_limit_and_stop_is_a_filled_t_plus_one_loss_not_filtered_out():
    frame,cal = sample(25)
    replace(frame,23,(10.9,11.1,9.3,10.6))
    replace(frame,24,(9.2,9.8,9.1,9.7))
    result = run(frame,cal)
    trade = result["trades"][0]
    assert trade["adjusted_entry_price"] == pytest.approx(10.4)
    assert trade["entry_day_stop_touch"] is True
    assert trade["ambiguous_entry_stop_order"] is True
    assert trade["exit_reason"] == "entry_day_stop_t_plus_one"
    assert trade["adjusted_exit_price"] == 9.2
    assert trade["holding_bars"] == 2
    assert trade["net_return"] == pytest.approx(9.2/10.4*.999**2-1)
    assert result["metrics"]["winning_trade_ratio"] == 0


def test_opening_quote_below_stop_cancels_before_order_activation():
    frame,cal = sample(24)
    replace(frame,23,(9.4,10.8,9.3,10.5))
    result = run(frame,cal)
    assert not result["fills"]
    assert result["orders"][0]["reason"] == "opening_quote_at_or_below_stop"


def test_better_open_fill_reprices_r_and_entry_day_target_cannot_sell():
    frame,cal = sample(25)
    replace(frame,23,(10.0,11.2,9.9,10.8))
    replace(frame,24,(10.8,11.2,10.7,11.0))
    result = run(frame,cal)
    trade = result["trades"][0]
    assert trade["adjusted_entry_price"] == 10
    assert trade["initial_risk"] == .5 and trade["target_price"] == 11
    assert trade["entry_day_target_touch"] and not trade["entry_day_stop_touch"]
    assert trade["entry_date"] != trade["exit_date"]
    assert trade["adjusted_exit_price"] == 11
    assert result["metrics"]["final_nav"] == pytest.approx(1.1*.999**2)


def test_entry_day_both_stop_target_touches_latch_only_stop():
    frame,cal = sample(25)
    replace(frame,23,(10.0,11.2,9.3,10.8))
    replace(frame,24,(10.9,11.2,10.7,11.0))
    result = run(frame,cal)
    trade = result["trades"][0]
    assert trade["entry_day_stop_touch"] and trade["entry_day_target_touch"]
    assert trade["ambiguous_entry_stop_order"] is False
    assert trade["exit_reason"] == "entry_day_stop_t_plus_one"
    assert trade["adjusted_exit_price"] == 10.9


def test_intraday_entry_target_touch_is_order_unknown_not_claimed_executable_profit():
    frame,cal = sample(24)
    replace(frame,23,(10.9,12.4,10.3,10.6))
    result = run(frame,cal)
    trade = result["trades"][0]
    assert trade["entry_day_target_touch"] and trade["ambiguous_entry_target_order"]
    assert trade["status"] == "open"
    assert result["metrics"]["completed_trades"] == 0


def test_subsequent_both_touched_bar_uses_stop_first():
    frame,cal = sample(25)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    replace(frame,24,(10.6,12.3,9.4,11.0))
    result = run(frame,cal)
    trade = result["trades"][0]
    assert trade["ambiguous_stop_target_order"]
    assert trade["exit_reason"] == "initial_stop"
    assert trade["adjusted_exit_price"] == 9.5
    assert result["metrics"]["final_nav"] == pytest.approx(9.5/10.4*.999**2)


@pytest.mark.parametrize("opening,reason",[(9.2,"gap_stop"),(12.5,"gap_target")])
def test_opening_gap_has_known_priority_over_intraday_ohlc(opening,reason):
    frame,cal = sample(25)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    replace(frame,24,(opening,max(opening,12.5),min(opening,9.0),10.7))
    result = run(frame,cal)
    assert result["trades"][0]["exit_reason"] == reason
    assert result["trades"][0]["adjusted_exit_price"] == opening
    assert not result["trades"][0]["ambiguous_stop_target_order"]


def test_entry_stop_pending_survives_suspension_and_rebound():
    frame,cal = sample(27)
    replace(frame,23,(10.9,11.1,9.3,10.6))
    frame.loc[24:25,"volume"] = 0
    replace(frame,24,(10.8,11,10.7,10.9))
    replace(frame,25,(11,11.2,10.9,11.1))
    replace(frame,26,(11.2,11.3,11.0,11.1))
    result = run(frame,cal)
    assert result["trades"][0]["exit_reason"] == "entry_day_stop_t_plus_one"
    assert result["trades"][0]["adjusted_exit_price"] == 11.2
    assert result["trades"][0]["holding_bars"] == 4


def test_later_suspended_stop_latches_until_tradeable_open():
    frame,cal = sample(27)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    frame.loc[24:25,"volume"] = 0
    replace(frame,24,(10.6,10.8,9.3,10.5))
    replace(frame,26,(10.9,11.1,10.7,10.8))
    result = run(frame,cal)
    assert result["trades"][0]["exit_reason"] == "stop_on_non_tradeable_bar"
    assert result["trades"][0]["adjusted_exit_price"] == 10.9


def test_time_exit_is_after_twentieth_close_not_on_it():
    frame,cal = sample(44)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    result = run(frame,cal)
    trade = result["trades"][0]
    assert trade["exit_reason"] == "time_exit_after_20_bars"
    assert trade["exit_date"] == frame.loc[43,"datetime"].date().isoformat()
    at20 = next(r for r in result["equity_curve"] if r["date"] == frame.loc[42,"datetime"].date().isoformat())
    assert at20["exposure"] == 1 and at20["pending_exit_reason"] == "time_exit_after_20_bars"


def test_end_open_position_includes_only_entry_cost_and_no_future_sale():
    frame,cal = sample(25)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    result = run_ict_daily(frame,cal,start=frame.loc[20,"datetime"],end=frame.loc[23,"datetime"])
    assert result["metrics"]["completed_trades"] == 0 and result["metrics"]["open_trades"] == 1
    assert result["metrics"]["winning_trade_ratio"] is None
    assert len(result["fills"]) == 1
    assert result["metrics"]["final_nav"] == pytest.approx(10.6/10.4*.999)


def test_invalidated_sweep_can_be_replaced_by_current_new_sweep():
    frame,cal = sample(26)
    replace(frame,21,(10.05,10.25,9.4,10.05))
    replace(frame,22,(10.05,11.3,10.0,11.2))
    replace(frame,23,(11.2,11.5,10.7,11.4))
    setups,sweeps = detect_ict_setups(frame)
    assert sweeps[0]["status"] == "invalidated"
    assert setups[0]["sweep_index"] == 21 and setups[0]["stop"] == 9.4


@pytest.mark.parametrize("displacement_index,expected",[(25,1),(26,0)])
def test_final_displacement_window_and_confirmation_delay(displacement_index,expected):
    frame,cal = sample(29)
    for i in range(21,29):
        replace(frame,i,(10.1,10.4,9.8,10.1))
    replace(frame,displacement_index,(10.1,11.2,10.0,11))
    replace(frame,displacement_index+1,(11,11.3,10.6,11.1))
    setups,_ = detect_ict_setups(frame)
    assert len(setups) == expected
    if setups:
        assert setups[0]["confirm_index"] == 26


def test_sweep_low_retest_invalidates_before_fvg_confirmation():
    frame,cal = sample(24)
    replace(frame,21,(9.95,11.2,9.5,11.0))
    result = run(frame,cal)
    assert not result["setups"]
    assert result["sweeps"][0]["status"] == "invalidated"


def additional_setup(monkeypatch,frame,index):
    setups,sweeps = detect_ict_setups(frame)
    extra=deepcopy(setups[0])
    extra.update(setup_id="extra_setup",confirm_index=index,
                 confirm_date=frame.loc[index,"datetime"].date().isoformat(),
                 available_at=datetime.combine(frame.loc[index,"datetime"].date(),time(15)).isoformat())
    monkeypatch.setattr(engine,"detect_ict_setups",lambda frame,first_sweep_index=0:(deepcopy(setups+[extra]),deepcopy(sweeps)))


def test_existing_pending_order_is_not_replaced_by_new_setup(monkeypatch):
    frame,cal=sample(25)
    additional_setup(monkeypatch,frame,23)
    result=run(frame,cal)
    assert result["orders"][0]["status"] == "pending_at_end"
    assert result["orders"][1]["status"] == "ignored"
    assert result["orders"][1]["reason"] == "older_order_active"
    assert result["diagnostics"]["accepted_orders"] == 1
    assert result["diagnostics"]["ignored_orders"] == 1


def test_position_ignores_new_setup(monkeypatch):
    frame,cal=sample(25)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    additional_setup(monkeypatch,frame,24)
    result=run(frame,cal)
    assert result["orders"][1]["reason"] == "position_open"
    assert len(result["fills"]) == 1


def test_exit_day_setup_can_arm_but_cannot_reenter_same_day(monkeypatch):
    frame,cal=sample(26)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    replace(frame,24,(10.6,12.3,10.2,12.0))
    replace(frame,25,(10.8,11.0,10.3,10.6))
    additional_setup(monkeypatch,frame,24)
    result=run(frame,cal)
    assert len(result["fills"]) == 3
    assert result["fills"][1]["target_exposure"] == 0
    assert result["fills"][2]["date"] == frame.loc[25,"datetime"].date().isoformat()
    assert result["orders"][1]["status"] == "filled"


def test_future_mutation_does_not_change_prior_events_or_decisions():
    frame,cal = sample(65)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    replace(frame,24,(10.6,12.3,10.4,12.0))
    original = run(frame,cal)
    changed = frame.copy()
    changed.loc[40:,["open","high","low","close"]] *= 2
    perturbed = run(changed,cal)
    cutoff = frame.loc[39,"datetime"].date().isoformat()
    assert len(original["fills"]) >= 2
    for key in ("equity_curve","fills","decisions"):
        assert [r for r in original[key] if r["date"] <= cutoff] == [r for r in perturbed[key] if r["date"] <= cutoff]
    # Malformed future prices are excluded before normalization when end is fixed.
    changed.loc[40:,["open","high","low","close"]] = -1
    options = {"start":frame.loc[20,"datetime"],"end":frame.loc[39,"datetime"]}
    assert run_ict_daily(frame,cal,**options) == run_ict_daily(changed,cal,**options)


def test_trade_net_return_product_reconciles_nav_and_cost_scenarios_keep_fills():
    frame,cal = sample(28)
    replace(frame,23,(10.9,11.1,10.3,10.6))
    replace(frame,24,(10.6,12.3,10.4,12.0))
    gross,net = [run(frame,cal,cost_bps=c) for c in (0,20)]
    assert [(f["date"],f["adjusted_price"]) for f in gross["fills"]] == [(f["date"],f["adjusted_price"]) for f in net["fills"]]
    assert math.prod(1+t["net_return"] for t in net["trades"]) == pytest.approx(net["metrics"]["final_nav"])
    assert net["metrics"]["final_nav"] == pytest.approx(gross["metrics"]["final_nav"]*.998**2)


@pytest.mark.parametrize("fault",["missing_bar","closed_bar","missing_calendar","missing_volume","intraday"])
def test_input_contract_rejects_ambiguous_calendar_or_prices(fault):
    frame,cal = sample()
    if fault == "missing_bar":
        frame=frame.drop(index=10)
    elif fault == "closed_bar":
        cal.loc[cal["date"].eq(frame.loc[10,"datetime"]),"is_trading_day"]=False
    elif fault == "missing_calendar":
        cal=cal.drop(index=4)
    elif fault == "missing_volume":
        frame=frame.drop(columns="volume")
    else:
        frame.loc[10,"datetime"] = datetime.combine(frame.loc[10,"datetime"].date(),time(1))
    with pytest.raises(ValueError):
        run(frame,cal)


def test_no_setup_is_valid_cash_result_not_failure():
    frame,cal = sample()
    for i in range(20,len(frame)):
        replace(frame,i,(10.1,10.4,9.8,10.1))
    result = run(frame,cal)
    assert result["metrics"]["total_return"] == 0 and result["diagnostics"]["cash_only"]
    assert result["metrics"]["winning_trade_ratio"] is None
