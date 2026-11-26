import math

import pandas as pd
import pytest

from research_layer.technical_portfolio import run_technical_portfolio


PRIOR, MON, TUE, WED = "2021-12-31", "2022-01-04", "2022-01-05", "2022-01-06"


def calendar():
    dates = pd.date_range(PRIOR, WED)
    return pd.DataFrame({"date": dates, "is_trading_day": dates.isin(pd.to_datetime([PRIOR, MON, TUE, WED]))})


def stock(symbol, opens, closes=None, volumes=(10, 10, 10)):
    closes = opens if closes is None else closes
    return pd.DataFrame({"symbol": symbol, "datetime": [MON,TUE,WED], "open": opens,
                         "high": [max(o,c) for o,c in zip(opens,closes)],
                         "low": [min(o,c) for o,c in zip(opens,closes)],
                         "close": closes, "volume": volumes, "adjust": "pre", "freq": "daily"})


def decision(signal, weights):
    return {"signal_date": signal, "weights": weights}


def run(bars, decisions, fee=0, end=WED):
    return run_technical_portfolio(bars, calendar(), start="2022-01-01", end=end,
                                   decisions=decisions, cost_bps=fee)


def test_cash_account_allocates_once_and_reinvests_sale_proceeds():
    bars = {"A":stock("A",[1.,2.,2.]), "B":stock("B",[2.,2.,4.])}
    results = run(bars,[decision(PRIOR,{"A":.5,"B":.5}),decision(MON,{"B":1.})])
    book = results["equity_curve"]
    assert [r["nav"] for r in book] == pytest.approx([1.,1.5,3.])
    assert book[0]["holdings"]["A"]["units"] == pytest.approx(.5)
    assert book[0]["holdings"]["B"]["units"] == pytest.approx(.25)
    assert [f["side"] for f in results["fills"] if f["date"] == TUE] == ["sell","buy"]
    assert book[-1]["holdings"]["B"]["units"] == pytest.approx(.75)
    assert all(r["cash"] >= 0 and r["exposure"] <= 1 for r in book)


def test_fees_reconcile_open_budget_and_roundtrip_cost():
    bars = {"A":stock("A",[1.,1.,1.])}
    decisions = [decision(PRIOR,{"A":1.}),decision(MON,{})]
    r = run(bars,decisions,fee=10)
    q = 1/1.001
    assert r["fills"][0]["units"] == pytest.approx(q)
    assert r["equity_curve"][0]["nav"] == pytest.approx(q)
    assert r["metrics"]["final_nav"] == pytest.approx(q*.999)
    assert r["metrics"]["maximum_drawdown"] == pytest.approx(1-q*.999)
    assert r["metrics"]["total_fees"] == pytest.approx(q*.002)
    assert r["metrics"]["calendar_days"] == 6
    assert r["metrics"]["cagr_calendar"] == pytest.approx((q*.999)**(365.25/6)-1)
    assert run(bars,decisions)["metrics"]["final_nav"] == 1.


def test_locked_holding_scales_multiple_buys_and_does_not_retry():
    bars = {"A":stock("A",[1.]*3,volumes=[10,0,10]),
            "B":stock("B",[1.,2.,2.]), "C":stock("C",[1.,.5,.5])}
    r = run(bars,[decision(PRIOR,{"A":.6}),decision(MON,{"B":.5,"C":.5})])
    second = r["equity_curve"][1]
    assert second["holdings"]["A"]["market_value"] == pytest.approx(.6)
    assert second["holdings"]["B"]["market_value"] == pytest.approx(.2)
    assert second["holdings"]["C"]["market_value"] == pytest.approx(.2)
    assert r["rebalances"][1]["buy_scale"] == pytest.approx(.4)
    assert r["rebalances"][1]["blocked"] == [{"symbol":"A","side":"sell","reason":"suspension"}]
    assert second["cash"] == pytest.approx(0)
    assert all(f["date"] != WED for f in r["fills"])
    assert r["equity_curve"][-1]["holdings"] == second["holdings"]


def test_month_end_signal_ignores_signal_day_price_for_execution():
    bars = {"A":stock("A",[2.,100.,120.])}
    r = run(bars,[decision(MON,{"A":1.})])
    assert r["equity_curve"][0]["cash"] == 1.
    assert r["fills"][0]["date"] == TUE and r["fills"][0]["price"] == 100.
    assert r["fills"][0]["units"] == .01
    assert r["metrics"]["final_nav"] == pytest.approx(1.2)


def test_same_signal_cannot_execute_twice_or_at_stated_earlier_open():
    bars = {"A":stock("A",[1.]*3)}
    with pytest.raises(ValueError,match="unique exchange"):
        run(bars,[decision(MON,{"A":1}),decision(MON,{})])
    with pytest.raises(ValueError,match="after signal"):
        run(bars,[{**decision(MON,{"A":1}),"execution_date":MON}])
    with pytest.raises(ValueError,match="unique exchange"):
        run(bars,[decision("2022-01-02",{"A":1})])
    assert run(bars,[decision(WED,{"A":1})])["fills"] == []


@pytest.mark.parametrize("weights",[{"B":1},{"A":True},{"A":math.nan},{"A":math.inf},{"A":-.1},{"A":1.001}])
def test_reject_invalid_target(weights):
    with pytest.raises(ValueError):
        run({"A":stock("A",[1.]*3)},[decision(PRIOR,weights)])


def test_require_pre_and_full_sessions_and_respect_tradeability():
    data = stock("A",[1.]*3)
    with pytest.raises(ValueError,match="sessions"):
        run({"A":data.drop(1)},[])
    with pytest.raises(ValueError,match="adjust=pre"):
        run({"A":data.assign(adjust="none")},[])
    with pytest.raises(ValueError,match="available capital"):
        run({"A":data,"B":stock("B",[1.]*3)},[decision(PRIOR,{"A":.51,"B":.5})])
    r = run({"A":data.assign(is_tradeable=[False,True,True])},[decision(PRIOR,{"A":1.})])
    assert r["fills"] == [] and r["rebalances"][0]["blocked"][0]["side"] == "buy"


def test_future_price_perturbation_and_end_cutoff():
    data = stock("A",[3.,5.,4.])
    signals = [decision(PRIOR,{"A":1}),decision(TUE,{})]
    original = run({"A":data},signals,fee=10)
    changed = data.copy()
    changed.loc[2,["open","high","low","close"]] *= 1.75
    perturbed = run({"A":changed},signals,fee=10)
    assert original["equity_curve"][:2] == perturbed["equity_curve"][:2]
    assert original["rebalances"][0] == perturbed["rebalances"][0]
    assert original["fills"][-1]["price"] != perturbed["fills"][-1]["price"]
    changed.loc[2,"open"] = -1.
    prefix = run({"A":changed},signals,fee=10,end=TUE)
    assert prefix["equity_curve"] == original["equity_curve"][:2]
