import numpy as np
import pandas as pd
import pytest

from research_layer.multiscale_events import prepare_outcomes, run_event_study, bootstrap_event_cases


def frame(n=10):
    dates = pd.bdate_range("2024-01-01", periods=n)
    values = np.arange(100., 100.+n)
    return pd.DataFrame({"datetime": dates, "open": values, "high": values+2,
        "low": values-1, "close": values+1, "volume": 1000.,
        "ready": True, "tradeable": True, "universe_trade_eligible": True})


def simple_masks(f):
    ready = f.ready.to_numpy(dtype=bool)
    chosen = f.get("chosen", pd.Series(True, index=f.index)).to_numpy(dtype=bool)
    return {"daily": {"unconditional": ready, "chosen": chosen}}, {"weekly:none": ready}


def test_forward_returns_use_next_open_and_count_entry_as_first_session():
    f = frame()
    result = prepare_outcomes(f, "2024-01-02", "2024-01-12", horizons=(1, 3), cost_bps=10)
    a, b = result["outcomes"][1], result["outcomes"][3]
    assert result["scope"][0]  # A prior close can place the first-period order.
    assert result["entry_index"][0] == 1
    assert a["exit_index"][0] == 1
    assert b["exit_index"][0] == 3
    assert a["net_return"][0] == pytest.approx(102*.999/(101*1.001)-1)
    assert b["net_return"][0] == pytest.approx(104*.999/(101*1.001)-1)
    assert a["mfe"][0] == pytest.approx(103/101-1)
    assert a["mae"][0] == pytest.approx(100/101-1)


def test_period_boundaries_never_borrow_future_exits():
    f = frame()
    result = prepare_outcomes(f, "2024-01-03", "2024-01-05", horizons=(1, 3), cost_bps=0)
    assert result["scope"].tolist() == [False, True, True, True, True, False, False, False, False, False]
    h = result["outcomes"][3]
    assert h["valid"][1] and h["exit_index"][1] == 4
    assert h["reason"][2] == "insufficient_forward_window"
    assert h["reason"][4] == "no_entry_in_period"
    assert not h["valid"][2:].any()


def test_suspension_rows_remain_in_horizon_and_delays_are_disclosed():
    f = frame()
    f.loc[[1, 4], "tradeable"] = False
    result = prepare_outcomes(f, f.datetime.iloc[0], f.datetime.iloc[-1], horizons=(3,), cost_bps=0)
    h = result["outcomes"][3]
    assert result["entry_index"][0] == 2
    assert result["entry_delay_sessions"][0] == 1
    assert h["exit_index"][0] == 5  # Scheduled index4 is retained but untradeable.
    assert h["exit_delay_sessions"][0] == 1
    assert h["holding_sessions"][0] == 4
    assert h["reason"][1] == "untradeable_signal"
    assert h["net_return"][0] == pytest.approx(106/102-1)
    short = prepare_outcomes(f, f.datetime.iloc[0], f.datetime.iloc[4], horizons=(3,))
    assert short["outcomes"][3]["reason"][0] == "no_tradeable_exit_in_period"


def test_future_changes_outside_label_horizon_do_not_change_earlier_label():
    f = frame()
    before = prepare_outcomes(f, f.datetime.iloc[0], f.datetime.iloc[-1], horizons=(1,))
    f.loc[2:, ["open", "high", "low", "close"]] *= 20
    after = prepare_outcomes(f, f.datetime.iloc[0], f.datetime.iloc[-1], horizons=(1,))
    for key in ["net_return", "mfe", "mae"]:
        assert before["outcomes"][1][key][0] == after["outcomes"][1][key][0]


def test_price_gate_and_invalid_holding_bar_are_not_silently_repaired():
    f = frame()
    f.loc[0, "universe_trade_eligible"] = False
    f.loc[3, "low"] = np.nan
    result = prepare_outcomes(f, f.datetime.iloc[0], f.datetime.iloc[-1], horizons=(3,))
    h = result["outcomes"][3]
    assert h["reason"][0] == "price_ineligible_signal"
    assert h["reason"][1] == "invalid_price"
    assert not h["valid"][1]


def test_matched_control_uses_same_dates_and_pooled_event_counts():
    a, b = frame(8), frame(8)
    a["chosen"] = False; b["chosen"] = False
    a.loc[0, "chosen"] = True
    b.loc[1, "chosen"] = True
    b[["open", "high", "low", "close"]] = [100., 101., 89., 90.]
    result = run_event_study({"A": a, "B": b}, a.datetime.iloc[0], a.datetime.iloc[-1],
                             horizons=(1,), cost_bps=0, mask_builder=simple_masks,
                             include_date_aggregates=True)
    chosen = next(x for x in result["rows"] if x["signal"] == "chosen")
    expected = np.array([102/101-1, -.1])
    market = np.array([(102/101-1-.1)/2, (103/102-1-.1)/2])
    assert chosen["event_count"] == 2
    assert chosen["net_win_rate"] == .5
    assert chosen["mean_net_return"] == pytest.approx(expected.mean())
    assert chosen["mean_excess_vs_matched_universe"] == pytest.approx((expected-market).mean())
    assert chosen["win_lift_vs_matched_universe"] == 0.
    dates = result["date_aggregates"]
    summary = next(x for x in dates["cases"] if "|chosen|" in x["id"])
    assert sum(summary["event_count"]) == 2
    assert sum(summary["net_return_sum"]) == pytest.approx(expected.sum())
    assert all(x["raw_trigger_count"] == x["event_count"]+sum(x["exclusions"].values()) for x in result["rows"])


def test_bootstrap_synchronized_duplicates_and_sparse_hypotheses():
    count = [1, 2, 1, 0, 2, 1]*5
    effect = np.array([.01, -.01, .02, 0, -.02, .03]*5)
    cases = [{"id": name, "event_count": count, "conditional_excess_sum": effect.tolist()}
             for name in ["a", "b"]]
    cases.append({"id": "sparse", "event_count": [1]+[0]*29,
                  "conditional_excess_sum": [.2]+[0.]*29})
    dates = pd.bdate_range("2024-01-01", periods=30).strftime("%Y-%m-%d").tolist()
    result = bootstrap_event_cases({"dates": dates, "cases": cases}, selected_ids=["a", "b"],
                                   repetitions=100, block_sessions=3, seed=10)
    a, b, sparse = result["candidate_results"]
    assert a["positive_effect_one_sided_p"] == b["positive_effect_one_sided_p"]
    assert a["individual_ci95"] == b["individual_ci95"]
    assert a["mean_effect"] == pytest.approx(effect.sum()/sum(count))
    assert sparse["statistically_eligible"] is False
    assert sparse["holm_adjusted_p"] == 1.


def test_masks_must_agree_between_symbols():
    a, b = frame(8), frame(8)
    b["different"] = 1
    def inconsistent(f):
        scales, conditions = simple_masks(f)
        if "different" in f:
            scales["daily"]["extra"] = np.ones(len(f), dtype=bool)
        return scales, conditions
    with pytest.raises(ValueError, match="keys differ"):
        run_event_study({"A": a, "B": b}, a.datetime.iloc[0], a.datetime.iloc[-1],
                         horizons=(1,), mask_builder=inconsistent)


def test_nan_flags_cannot_silently_become_true():
    a = frame(8)
    a["tradeable"] = a.tradeable.astype(float)
    a.loc[0, "tradeable"] = np.nan
    with pytest.raises(ValueError, match="explicit boolean"):
        prepare_outcomes(a, a.datetime.iloc[0], a.datetime.iloc[-1], horizons=(1,))
    a["tradeable"] = True
    def bad_mask(f):
        scales, conditions = simple_masks(f)
        scales["daily"]["chosen"] = np.full(len(f), np.nan)
        return scales, conditions
    with pytest.raises(ValueError, match="explicit boolean"):
        run_event_study({"A": a}, a.datetime.iloc[0], a.datetime.iloc[-1], horizons=(1,), mask_builder=bad_mask)
