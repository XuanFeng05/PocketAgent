import pandas as pd
import pytest

from research_layer.technical import normalize_technical_bars, run_technical_backtest


def bars(closes, opens=None, start="2023-04-24"):
    opens = closes if opens is None else opens
    return pd.DataFrame({
        "symbol": "300750.SZ", "datetime": pd.date_range(start, periods=len(closes)),
        "open": opens, "high": [max(o, c) for o, c in zip(opens, closes)],
        "low": [min(o, c) for o, c in zip(opens, closes)], "close": closes,
        "volume": 1000, "amount": 10000, "adjust": "pre", "freq": "daily",
    })


def test_forward_adjusted_returns_avoid_raw_ex_rights_drop():
    # Local CATL adjusted prices around 2023-04-26. This checks return
    # accounting on adjusted prices, not the data provider's adjustment formula.
    frame = bars([198.9958953, 198.9958953, 209.7499010],
                 [198.9958953, 198.9958953, 209.7499010])
    result = run_technical_backtest(frame, "buy_hold")
    assert result["metrics"]["total_return"] == pytest.approx(209.7499010 / 198.9958953 - 1)
    assert result["price_basis"] == "pre"
    assert all("shares" not in event for event in result["fills"])
    assert result["fills"][0]["execution_time"] == "2023-04-25T09:30:00"


def test_next_open_separates_old_overnight_and_new_intraday_exposure():
    frame = bars([10, 8, 10, 9], [10, 8, 10, 12])
    result = run_technical_backtest(frame, "mean_reversion", {"window": 2, "entry_deviation": .1})
    # Buy signalled at 8, executed at 10: the 8->10 overnight gain is not earned.
    # Exit signalled at 10, executed at 12: overnight 10->12 is earned,
    # while the subsequent 12->9 intraday loss is avoided.
    assert [e["nav"] for e in result["equity_curve"]] == pytest.approx([1, 1, 1, 1.2])
    assert [e["exposure"] for e in result["equity_curve"]] == [0, 0, 1, 0]


def test_costs_apply_once_per_change_and_cash_is_flat():
    frame = bars([10, 8, 10, 9], [10, 8, 10, 12])
    result = run_technical_backtest(frame, "mean_reversion", {"window": 2, "entry_deviation": .1}, cost_bps=10)
    assert result["metrics"]["final_nav"] == pytest.approx(1.2 * .999 ** 2)
    assert result["metrics"]["position_changes"] == 2
    assert run_technical_backtest(frame, "cash", cost_bps=10)["metrics"]["final_nav"] == 1
    assert run_technical_backtest(frame, "buy_hold", cost_bps=10) == run_technical_backtest(frame, "buy_hold", {}, cost_bps=10)


def test_future_changes_do_not_change_earlier_signals_or_nav():
    original = bars([10, 8, 10, 9, 8, 12], [10, 8, 10, 12, 9, 11])
    changed = original.copy()
    changed.loc[4:, ["open", "high", "low", "close"]] *= 3
    left = run_technical_backtest(original, "mean_reversion", {"window": 2, "entry_deviation": .1})
    right = run_technical_backtest(changed, "mean_reversion", {"window": 2, "entry_deviation": .1})
    assert left["equity_curve"][:4] == right["equity_curve"][:4]
    assert [s for s in left["signals"] if s["signal_time"] < "2023-04-27"] == [s for s in right["signals"] if s["signal_time"] < "2023-04-27"]


def test_warmup_may_queue_first_open_without_earning_past_return():
    past = bars([10, 8], start="2023-04-23")
    frame = bars([11, 12], [10, 12], start="2023-04-25")
    result = run_technical_backtest(frame, "mean_reversion", {"window": 2, "entry_deviation": .1}, warmup_bars=past)
    assert result["equity_curve"][0]["nav"] == pytest.approx(1.1)
    assert result["fills"][0]["signal_time"] == "2023-04-24T15:00:00"
    assert result["warmup"]["bars"] == 2
    with pytest.raises(ValueError, match="strictly before"):
        run_technical_backtest(frame, warmup_bars=frame)


@pytest.mark.parametrize("field,value", [("volume", 0), ("is_tradeable", False)])
def test_non_tradeable_bar_blocks_change_and_flat_strategy_retries(field, value):
    frame = bars([10, 20, 30, 40])
    if field == "is_tradeable":
        frame[field] = True
    frame.loc[1, field] = value
    result = run_technical_backtest(frame, "buy_hold")
    assert result["fills"][0]["status"] == "blocked"
    assert result["fills"][1]["status"] == "applied"
    assert result["metrics"]["final_nav"] == pytest.approx(40 / 30)


def test_non_tradeable_exit_keeps_marking_existing_exposure():
    frame = bars([10, 8, 10, 9, 12], [10, 8, 10, 12, 9])
    frame.loc[3, "volume"] = 0
    result = run_technical_backtest(frame, "mean_reversion", {"window": 2, "entry_deviation": .1})
    assert result["fills"][1]["status"] == "blocked"
    assert result["equity_curve"][3]["exposure"] == 1
    assert result["equity_curve"][3]["nav"] == pytest.approx(.9)
    assert result["equity_curve"][4]["nav"] == pytest.approx(1.2)


@pytest.mark.parametrize("change", ["raw", "mixed", "duplicate", "nan", "negative_amount", "missing_adjust", "two_symbols"])
def test_invalid_input_rejected(change):
    frame = bars([10, 11, 12])
    if change == "raw": frame["adjust"] = "none"
    if change == "mixed": frame.loc[1, "adjust"] = "none"
    if change == "duplicate": frame.loc[1, "datetime"] = frame.loc[0, "datetime"]
    if change == "nan": frame.loc[1, "close"] = float("nan")
    if change == "negative_amount": frame.loc[1, "amount"] = -1
    if change == "missing_adjust": frame = frame.drop(columns="adjust")
    if change == "two_symbols": frame.loc[1, "symbol"] = "000001.SZ"
    with pytest.raises(ValueError):
        normalize_technical_bars(frame)


def test_rejects_nontechnical_strategy_and_invalid_cost():
    frame = bars([10, 11])
    with pytest.raises(ValueError, match="Unsupported"):
        run_technical_backtest(frame, "ridge_forecast")
    with pytest.raises(ValueError, match="cost_bps"):
        run_technical_backtest(frame, cost_bps=-1)
