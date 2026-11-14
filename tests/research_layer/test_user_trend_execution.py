"""Priced contracts for user-authored entry and intraday ATR management."""
from datetime import datetime, time, timedelta
import json

import numpy as np
import pandas as pd
import pytest

import research_layer.user_trend as engine


def bars(values):
    dates = pd.bdate_range("2025-01-06", periods=len(values))
    return pd.DataFrame([{"symbol":"000001.SZ", "datetime":day, "open":o, "high":h, "low":l,
                          "close":c, "volume":1000., "freq":"daily", "adjust":"pre"}
                         for day,(o,h,l,c) in zip(dates,values)])


def calendar(data):
    days = pd.date_range(data.datetime.iloc[0], data.datetime.iloc[-1])
    return pd.DataFrame({"date":days, "is_trading_day":days.dayofweek < 5})


def fixed_features(monkeypatch, *, setups=(0,), bands=None, efi=None, below=(), weekly_down=(),
                   atrs=None, efi13=1., slopes=None):
    bands, efi, atrs, slopes = bands or {}, efi or {}, atrs or {}, slopes or {}

    def build(frame, cal, *, atr_period):
        out = frame.copy()
        out["daily_ema13"] = [bands.get(i, (99.,101.) if i in setups else (49.,50.))[0] for i in range(len(out))]
        out["daily_ema26"] = [bands.get(i, (99.,101.) if i in setups else (49.,50.))[1] for i in range(len(out))]
        for i in below:
            out.loc[i,"daily_ema26"] = 101.
        out["atr"] = [atrs.get(i,4.) for i in range(len(out))]
        out["efi2"] = [efi.get(i,(1.,1.,1.))[0] for i in range(len(out))]
        out["efi2_previous"] = [efi.get(i,(1.,1.,1.))[1] for i in range(len(out))]
        out["efi2_previous2"] = [efi.get(i,(1.,1.,1.))[2] for i in range(len(out))]
        out["efi13"] = efi13
        out["weekly_ema13"] = 90.
        out["weekly_ema26"] = 100.  # No golden-order restriction is permitted.
        out["weekly_ema13_slope"] = [slopes.get(i,(-1.,-1.) if i in weekly_down else (1.,1.))[0] for i in range(len(out))]
        out["weekly_ema26_slope"] = [slopes.get(i,(-1.,-1.) if i in weekly_down else (1.,1.))[1] for i in range(len(out))]
        out["indicators_ready"] = True
        out["weekly_completed_count"] = 27
        out["daily_available_at"] = [datetime.combine(day.date(),time(15)) for day in out.datetime]
        out["weekly_available_at"] = [datetime.combine(day.date()-timedelta(days=7),time(15)) for day in out.datetime]
        return out
    monkeypatch.setattr(engine,"build_user_trend_features",build)


def run(data, **kwargs):
    return engine.run_user_trend(data, calendar(data), start=data.datetime.iloc[0], end=data.datetime.iloc[-1], **kwargs)


def quiet(n=4):
    return bars([(100,101,99.5,100)]*n)


def test_band_entry_is_or_not_an_extra_efi_filter_or_golden_cross(monkeypatch):
    fixed_features(monkeypatch, efi13=-10000)
    result = run(quiet())
    assert len(result["trades"]) == 1
    signal = result["signals"][0]
    assert signal["efi2"] > 0
    assert signal["weekly_ema13"] < signal["weekly_ema26"]
    assert signal["entry_reasons"] == ["ema_band_pullback"]
    assert result["trades"][0]["entry_date"] == "2025-01-07"


@pytest.mark.parametrize("values,expected", [((-1.,-4.,-8.),True),((-2.,-4.,-8.),False),
                                             ((0.,-4.,-8.),False),((-1.,-4.,-2.),False),
                                             ((-1.,0.,-8.),False)])
def test_efi_recovery_uses_correct_reciprocal_and_two_rising_days(monkeypatch, values, expected):
    data = bars([(100,101,97.5,100)] + [(100,101,99.5,100)]*3)
    fixed_features(monkeypatch, efi={0:values})
    result = run(data)
    assert bool(result["trades"]) is expected
    assert result["decisions"][0]["branch_a_band_touch"] is False
    if expected:
        assert result["signals"][0]["efi_distance_ratio"] == .25
        assert result["signals"][0]["entry_reasons"] == ["one_atr_efi_recovery"]


@pytest.mark.parametrize("low,expected", [(99,True),(101,True),(98.9,False)])
def test_minimum_itself_must_be_inside_band_not_merely_crossing_it(monkeypatch, low, expected):
    data = bars([(101,103,low,102)] + [(100,101,99.5,100)]*3)
    fixed_features(monkeypatch)
    assert bool(run(data)["trades"]) is expected


@pytest.mark.parametrize("slope", [(0.,1.),(1.,0.),(-1.,1.)])
def test_both_weekly_slopes_must_be_strictly_positive(monkeypatch,slope):
    fixed_features(monkeypatch,slopes={0:slope})
    assert run(quiet())["trades"] == []


def test_ordered_ema_band_and_efi13_are_not_extra_requirements(monkeypatch):
    data=quiet()
    fixed_features(monkeypatch,bands={0:(101.,99.)},efi13=-999)
    left=run(data)
    fixed_features(monkeypatch,bands={0:(99.,101.)},efi13=999)
    right=run(data)
    for key in ("fills","trades","equity_curve"):
        assert left[key] == right[key]


def test_ohlc_and_olhc_have_different_causal_trailing_outcomes(monkeypatch):
    fixed_features(monkeypatch)
    data=bars([(100,101,99.5,100),(100,103,99,103),(103,106,102,105.5)])
    high_first=run(data,intraday_path="ohlc")
    low_first=run(data,intraday_path="olhc")
    assert high_first["trades"][0]["adjusted_exit_price"] == 105
    assert high_first["trades"][0]["exit_reason"] == "trailing_stop"
    assert high_first["metrics"]["final_nav"] == pytest.approx(1.05*.999**2)
    assert low_first["trades"][0]["status"] == "open"
    assert low_first["trades"][0]["active_stop"] == 105
    assert low_first["metrics"]["final_nav"] == pytest.approx(1.055*.999)
    path=high_first["decisions"][-1]["risk_trace"]
    assert path[-1]["path_leg"] == "high_to_low"
    assert path[-1]["peak_at_trigger"] == 106


@pytest.mark.parametrize("path,reason", [("ohlc","trailing_stop_t_plus_one"),("olhc","hard_stop_t_plus_one")])
def test_entry_day_first_price_stop_is_latched_even_after_rebound(monkeypatch,path,reason):
    fixed_features(monkeypatch)
    data=bars([(100,101,99.5,100),(100,106,97,105.5),(107,108,106,107)])
    result=run(data,intraday_path=path)
    assert result["equity_curve"][1]["exposure"] == 1
    assert result["equity_curve"][1]["pending_exit_reason"] == reason
    assert result["fills"][-1]["date"] == "2025-01-08"
    assert result["fills"][-1]["adjusted_price"] == 107
    assert result["trades"][0]["exit_reason"] == reason
    assert result["metrics"]["final_nav"] == pytest.approx(1.07*.999**2)


def test_close_entry_cannot_use_earlier_high_low_or_ema_count(monkeypatch):
    fixed_features(monkeypatch,bands={0:(94.,96.)},below=(0,))
    data=bars([(100,106,95,100),(100,103,99,102),(102,103,101,102)])
    result=run(data,entry_mode="signal_close")
    first=result["decisions"][0]
    assert first["risk_trace"] == []
    assert first["risk_after"]["peak"] == 100
    assert not first["risk_after"]["trail_active"]
    assert first["below_ema26_count"] == 0
    assert result["trades"][0]["entry_day_price_stop_touch"] is False
    assert result["metrics"]["final_nav"] == pytest.approx(1.02*.999)


def test_trailing_activation_is_strict_and_atr_is_fixed(monkeypatch):
    fixed_features(monkeypatch,atrs={1:100.,2:200.})
    data=bars([(100,101,99.5,100),(100,104,99,103),(103,104,100,103)])
    result=run(data,intraday_path="olhc")
    t=result["trades"][0]
    assert t["atr_at_entry"] == 4
    assert t["hard_stop"] == 98
    assert t["peak"] == 104
    assert not t["trail_active"]
    assert t["status"] == "open"


def test_gap_below_previously_effective_trail_uses_open_not_line(monkeypatch):
    fixed_features(monkeypatch)
    data=bars([(100,101,99.5,100),(100,105,99.5,104.8),(103,104,102,103)])
    result=run(data,intraday_path="olhc")
    t=result["trades"][0]
    assert result["equity_curve"][1]["active_stop"] == 104
    assert result["equity_curve"][1]["pending_exit_reason"] is None
    assert t["adjusted_exit_price"] == 103
    assert t["exit_reason"] == "gap_trailing_stop"
    assert t["net_return"] == pytest.approx(1.03*.999**2-1)


def test_gap_hard_stop_uses_worse_open(monkeypatch):
    fixed_features(monkeypatch)
    data=bars([(100,101,99.5,100),(100,101,99,100),(96,97,95,96)])
    t=run(data)["trades"][0]
    assert t["exit_reason"] == "gap_hard_stop"
    assert t["adjusted_exit_price"] == 96


def test_nontradeable_stop_latches_through_rebound(monkeypatch):
    fixed_features(monkeypatch)
    data=bars([(100,101,99.5,100),(100,101,99,100),(100,101,97,99),
               (101,103,100,102),(103,104,100,101)])
    data.loc[[2,3],"volume"]=0
    result=run(data)
    assert result["equity_curve"][2]["pending_exit_reason"] == "hard_stop_non_tradeable"
    assert result["equity_curve"][3]["pending_exit_reason"] == "hard_stop_non_tradeable"
    t=result["trades"][0]
    assert t["exit_date"] == "2025-01-10" and t["adjusted_exit_price"] == 103
    assert t["net_return"] == pytest.approx(1.03*.999**2-1)


@pytest.mark.parametrize("mode",["next_open","signal_close"])
def test_three_closes_count_only_after_entry_and_exit_next_open(monkeypatch,mode):
    fixed_features(monkeypatch,below=(0,1,2,3))
    data=bars([(100,101,99.5,100)]*4+[(102,103,101,102)])
    result=run(data,entry_mode=mode)
    counts=[d["below_ema26_count"] for d in result["decisions"]]
    assert counts[1:4] == [1,2,3]
    assert counts[0] == (0 if mode=="signal_close" else None)
    assert result["decisions"][3]["pending_exit_reason"] == "three_closes_below_ema26"
    t=result["trades"][0]
    assert t["exit_date"] == "2025-01-10" and t["adjusted_exit_price"] == 102
    assert t["exit_reason"] == "three_closes_below_ema26"


def test_ema_streak_resets_on_a_recovered_close(monkeypatch):
    fixed_features(monkeypatch,below=(1,2,3,4,5))
    data=bars([(100,101,99.5,100),(100,101,99.5,100),(100,103,99.5,102),
               (102,103,99.5,100),(100,101,99.5,100),(100,101,99.5,100),(101,102,100,101)])
    result=run(data)
    assert [d["below_ema26_count"] for d in result["decisions"]][1:6] == [1,0,1,2,3]
    assert result["trades"][0]["exit_date"] == "2025-01-14"


def test_pending_ema_exit_is_not_cancelled_by_halt_rebound(monkeypatch):
    fixed_features(monkeypatch,below=(1,2,3))
    data=bars([(100,101,99.5,100)]*4+[(103,104,102,103),(104,105,103,104)])
    data.loc[4,"volume"]=0
    result=run(data)
    assert result["equity_curve"][4]["pending_exit_reason"] == "three_closes_below_ema26"
    assert result["trades"][0]["adjusted_exit_price"] == 104


def test_weekly_weakening_and_long_duration_are_not_exit_rules(monkeypatch):
    fixed_features(monkeypatch,weekly_down=range(2,35))
    result=run(quiet(35))
    assert result["trades"][0]["status"] == "open"
    assert result["trades"][0]["holding_bars"] == 34
    assert result["trades"][0]["pending_exit_reason"] is None


def test_next_open_candidate_expires_on_nontradeable_next_bar(monkeypatch):
    fixed_features(monkeypatch)
    data=quiet();data.loc[1,"volume"]=0
    result=run(data)
    assert result["fills"] == []
    assert result["signals"][0]["status"] == "expired"
    assert result["signals"][0]["reason"] == "next_bar_non_tradeable"


def test_no_prestart_signal_is_carried_into_new_cash_account(monkeypatch):
    fixed_features(monkeypatch)
    data=quiet()
    result=engine.run_user_trend(data,calendar(data),start=data.datetime.iloc[1],end=data.datetime.iloc[-1])
    assert result["fills"] == []


@pytest.mark.parametrize("mode",["next_open","signal_close"])
def test_exit_day_signal_is_ignored_not_reentered_or_armed(monkeypatch,mode):
    fixed_features(monkeypatch,setups=(0,2),bands={2:(96.,98.)})
    data=bars([(100,101,99.5,100),(100,101,99,100),(100,101,97,99),(100,101,99,100)])
    result=run(data,entry_mode=mode)
    assert len(result["trades"]) == 1
    assert result["signals"][-1]["reason"] == "exit_day"
    assert result["metrics"]["final_exposure"] == 0


def test_hand_calculated_close_entry_costs_and_stop(monkeypatch):
    fixed_features(monkeypatch)
    data=bars([(100,101,99.5,100),(100,101,97,99),(99,100,98.5,99)])
    result=run(data,entry_mode="signal_close",cost_bps=10)
    assert [f["adjusted_price"] for f in result["fills"]] == [100,98]
    assert result["metrics"]["final_nav"] == pytest.approx(.98*.999**2)
    assert result["trades"][0]["net_return"] == pytest.approx(.98*.999**2-1)
    assert result["trades"][0]["exit_reason"] == "hard_stop"
    json.dumps(result,allow_nan=False)


def test_zero_atr_does_not_allow_a_risk_distance_entry(monkeypatch):
    fixed_features(monkeypatch,atrs={0:0.})
    assert run(quiet())["trades"] == []


def test_last_close_entry_marks_open_without_invented_sale(monkeypatch):
    fixed_features(monkeypatch,setups=(3,))
    result=run(quiet(),entry_mode="signal_close")
    assert result["trades"][0]["status"] == "open"
    assert result["metrics"]["final_nav"] == .999
    assert len(result["fills"]) == 1


def test_last_next_open_signal_is_pending_not_filled(monkeypatch):
    fixed_features(monkeypatch,setups=(3,))
    result=run(quiet())
    assert result["signals"][0]["status"] == "pending_at_end"
    assert result["trades"] == []


def real_sample():
    dates=pd.bdate_range("2020-01-06",periods=420)
    ix=np.arange(len(dates));close=50+.035*ix+3*np.sin(ix/12)
    opening=np.r_[close[0],close[:-1]]
    data=pd.DataFrame({"symbol":"000001.SZ","datetime":dates,"open":opening,
                       "high":np.maximum(opening,close)+.6,"low":np.minimum(opening,close)-.6,
                       "close":close,"volume":1000.,"adjust":"pre","freq":"daily"})
    days=pd.date_range(dates[0],dates[-1].date()+timedelta(days=7))
    cal=pd.DataFrame({"date":days,"is_trading_day":days.dayofweek<5})
    return data,cal


@pytest.mark.parametrize("mode",["next_open","signal_close"])
@pytest.mark.parametrize("path",["ohlc","olhc"])
def test_real_features_do_not_use_future_prices(mode,path):
    data,cal=real_sample()
    args={"start":data.datetime.iloc[180],"end":data.datetime.iloc[-1],"entry_mode":mode,"intraday_path":path}
    original=engine.run_user_trend(data,cal,**args)
    changed=data.copy();changed.loc[320:,["open","high","low","close"]]*=2
    perturbed=engine.run_user_trend(changed,cal,**args)
    cutoff=data.datetime.iloc[319].date().isoformat()
    assert len([f for f in original["fills"] if f["date"]<=cutoff]) >= 2
    for key in ("fills","equity_curve","decisions","signals"):
        assert [r for r in original[key] if r["date"]<=cutoff] == [r for r in perturbed[key] if r["date"]<=cutoff]


def test_end_cutoff_excludes_invalid_future_prices():
    data,cal=real_sample();cutoff=data.datetime.iloc[319]
    original=engine.run_user_trend(data,cal,start=data.datetime.iloc[180],end=cutoff)
    changed=data.copy();changed.loc[320:,["open","high","low","close"]]=np.nan
    assert engine.run_user_trend(changed,cal,start=data.datetime.iloc[180],end=cutoff) == original


@pytest.mark.parametrize("bad",[{"entry_mode":"overnight_limit"},{"intraday_path":"choose_best"},{"cost_bps":-1},
                                 {"cost_bps":True},{"cost_bps":10000},{"atr_period":0}])
def test_invalid_configuration_is_rejected(bad):
    data,cal=real_sample()
    with pytest.raises(ValueError):
        engine.run_user_trend(data,cal,start=data.datetime.iloc[180],end=data.datetime.iloc[-1],**bad)
