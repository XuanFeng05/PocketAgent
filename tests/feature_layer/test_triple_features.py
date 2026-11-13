import numpy as np
import pandas as pd
import pytest

from feature_layer.triple_screen import build_triple_features
from strategy_layer.triple_screen import TripleScreenSettings


def calendar(start="2024-01-01", end="2024-01-28", holidays=()):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates,
                         "is_trading_day": (dates.dayofweek < 5) & ~dates.isin(pd.to_datetime(list(holidays)))})


def bars(cal, closes=None):
    days = cal.loc[cal["is_trading_day"], "date"].reset_index(drop=True)
    prices = np.asarray(closes if closes is not None else 10 + np.arange(len(days)) * 0.2, dtype=float)
    return pd.DataFrame({"symbol": "000001.SZ", "datetime": days, "open": prices,
                         "high": prices + .1, "low": prices - .1, "close": prices,
                         "volume": 1000., "freq": "daily", "adjust": "pre"})


def settings():
    return TripleScreenSettings(macd_fast=2, macd_slow=3, macd_signal=2, warmup_weeks=2)


def test_weekly_hist_is_not_visible_until_final_session_close():
    cal = calendar()
    result = build_triple_features(bars(cal), cal, settings()).set_index("datetime")
    assert result.loc["2024-01-04", "weekly_completed_count"] == 0
    assert pd.isna(result.loc["2024-01-04", "weekly_hist"])
    assert result.loc["2024-01-05", "weekly_available_at"] == pd.Timestamp("2024-01-05 15:00")
    assert result.loc["2024-01-09", "weekly_completed_count"] == 1
    assert not result.loc["2024-01-09", "indicators_ready"]
    assert result.loc["2024-01-12", "weekly_completed_count"] == 2
    assert result.loc["2024-01-12", "indicators_ready"]


def test_holiday_week_is_published_on_thursday_close():
    cal = calendar(holidays=["2024-01-12"])
    result = build_triple_features(bars(cal), cal, settings()).set_index("datetime")
    assert result.loc["2024-01-10", "weekly_completed_count"] == 1
    assert result.loc["2024-01-11", "weekly_completed_count"] == 2
    assert result.loc["2024-01-11", "weekly_available_at"] == pd.Timestamp("2024-01-11 15:00")
    assert result.loc["2024-01-15", "weekly_available_at"] == pd.Timestamp("2024-01-11 15:00")


def test_truncated_first_week_does_not_seed_and_last_week_does_not_publish():
    cal = calendar()
    source = bars(cal)
    source = source.loc[source["datetime"].between("2024-01-03", "2024-01-24")]
    result = build_triple_features(source, cal, settings()).set_index("datetime")
    assert result.loc["2024-01-05", "weekly_completed_count"] == 0
    assert result.loc["2024-01-12", "weekly_completed_count"] == 1
    assert result.loc["2024-01-12", "weekly_hist"] == 0
    assert result.loc["2024-01-24", "weekly_completed_count"] == 2
    assert result.loc["2024-01-24", "weekly_available_at"] == pd.Timestamp("2024-01-19 15:00")


def test_calendar_boundary_week_is_not_published_without_known_week_end():
    cal = calendar(end="2024-01-26")
    result = build_triple_features(bars(cal), cal, settings())
    assert result.iloc[-1]["weekly_completed_count"] == 3
    assert result.iloc[-1]["weekly_available_at"] == pd.Timestamp("2024-01-19 15:00")


def test_future_daily_prices_cannot_change_prior_features():
    cal = calendar()
    source = bars(cal)
    changed = source.copy()
    mask = changed["datetime"].gt("2024-01-16")
    changed.loc[mask, ["open", "high", "low", "close"]] *= 1.7
    first = build_triple_features(source, cal, settings())
    second = build_triple_features(changed, cal, settings())
    columns = ["force_index_2", "weekly_hist", "weekly_hist_previous", "weekly_available_at", "weekly_completed_count", "entry_ready"]
    pd.testing.assert_frame_equal(first.loc[~mask, columns], second.loc[~mask, columns])


def test_prefix_and_full_history_publish_same_features():
    cal = calendar()
    source = bars(cal)
    prefix = source.loc[source["datetime"].le("2024-01-16")]
    first = build_triple_features(source, cal, settings())
    second = build_triple_features(prefix, cal, settings())
    pd.testing.assert_frame_equal(first.iloc[:len(prefix)], second)


def test_macd_is_raw_histogram_not_price_normalized():
    cal = calendar(end="2024-01-14")
    source = bars(cal, [12.] * 5 + [18.] * 5)
    result = build_triple_features(source, cal, settings())
    # First weekly DIF and signal are zero. Second fast EMA=16, slow=15,
    # DIF=1, signal=2/3, histogram=1/3, not (1/3)/18.
    assert result.iloc[-1]["weekly_hist"] == pytest.approx(1/3)
    assert result.iloc[-1]["weekly_hist_previous"] == 0
    assert result.iloc[-1]["weekly_up"]


def test_force_index_uses_supplied_share_volume_and_seed_zero():
    cal = calendar(end="2024-01-07")
    source = bars(cal, [10, 11, 9, 10, 10])
    source["volume"] = [100, 200, 300, 400, 500]
    result = build_triple_features(source, cal, settings())
    assert result.iloc[0]["force_index_2"] == 0
    assert result.iloc[1]["force_index_2"] == pytest.approx(200 * 2/3)
    assert result.iloc[2]["force_index_2"] == pytest.approx((200 * 2/3) / 3 - 600 * 2/3)


def test_entry_ready_requires_negative_force_and_tradeable_row():
    cal = calendar(end="2024-01-21")
    source = bars(cal, [10.] * 5 + [15.] * 5 + [14, 13, 12, 11, 10])
    first = build_triple_features(source, cal, settings()).set_index("datetime")
    assert first.loc["2024-01-15", "entry_ready"]
    source["is_tradeable"] = True
    source.loc[source["datetime"].eq("2024-01-15"), "is_tradeable"] = False
    source.loc[source["datetime"].eq("2024-01-16"), "volume"] = 0
    result = build_triple_features(source, cal, settings()).set_index("datetime")
    assert not result.loc["2024-01-15", "entry_ready"]
    assert not result.loc["2024-01-16", "entry_ready"]


def test_default_warmup_requires_one_hundred_complete_weeks():
    cal = calendar(end="2025-12-07")
    source = bars(cal)
    result = build_triple_features(source, cal)
    ready = result.loc[result["indicators_ready"]]
    assert not ready.empty
    assert ready.iloc[0]["weekly_completed_count"] == 100
    assert not result.loc[result["weekly_completed_count"] < 100, "indicators_ready"].any()


def test_missing_open_session_is_not_treated_as_a_halt():
    cal = calendar()
    source = bars(cal).drop(index=2)
    with pytest.raises(ValueError, match="Missing open-session daily bars"):
        build_triple_features(source, cal, settings())


@pytest.mark.parametrize("damage", ["duplicate", "missing_day", "invalid_flag", "timezone", "intraday", "short_range"])
def test_bad_calendar_is_rejected(damage):
    cal = calendar()
    source = bars(cal)
    if damage == "duplicate": cal = pd.concat([cal, cal.iloc[[0]]])
    if damage == "missing_day": cal = cal.drop(index=6)
    if damage == "invalid_flag":
        cal["is_trading_day"] = cal["is_trading_day"].astype(object)
        cal.loc[0, "is_trading_day"] = "unknown"
    if damage == "timezone": cal["date"] = cal["date"].dt.tz_localize("UTC")
    if damage == "intraday": cal.loc[0, "date"] = pd.Timestamp("2024-01-01 01:00")
    if damage == "short_range": cal = cal.iloc[1:]
    with pytest.raises(ValueError): build_triple_features(source, cal, settings())


@pytest.mark.parametrize("damage", ["missing_volume", "nan_volume", "negative_volume", "raw", "mixed_adjust", "intraday"])
def test_bad_daily_input_is_rejected(damage):
    cal = calendar()
    source = bars(cal)
    if damage == "missing_volume": source = source.drop(columns="volume")
    if damage == "nan_volume": source.loc[0, "volume"] = np.nan
    if damage == "negative_volume": source.loc[0, "volume"] = -1
    if damage == "raw": source["adjust"] = "none"
    if damage == "mixed_adjust": source.loc[0, "adjust"] = "none"
    if damage == "intraday": source.loc[0, "datetime"] = pd.Timestamp("2024-01-01 01:00")
    with pytest.raises(ValueError): build_triple_features(source, cal, settings())


def test_bars_on_calendar_closed_day_are_rejected():
    cal = calendar()
    source = bars(cal)
    cal.loc[cal["date"].eq("2024-01-02"), "is_trading_day"] = False
    with pytest.raises(ValueError, match="calendar-closed"):
        build_triple_features(source, cal, settings())


def test_inputs_are_not_mutated():
    cal = calendar()
    source = bars(cal)
    before_bars, before_calendar = source.copy(deep=True), cal.copy(deep=True)
    build_triple_features(source, cal, settings())
    pd.testing.assert_frame_equal(source, before_bars)
    pd.testing.assert_frame_equal(cal, before_calendar)


def test_precomputed_features_cannot_override_causal_calculations():
    cal = calendar()
    source = bars(cal)
    correct = build_triple_features(source, cal, settings())
    forged = correct.copy()
    forged["weekly_hist"] = 999.0
    forged["entry_ready"] = True
    recomputed = build_triple_features(forged, cal, settings())
    pd.testing.assert_frame_equal(recomputed, correct)


@pytest.mark.parametrize("bar_unit", ["ns", "us"])
@pytest.mark.parametrize("calendar_unit", ["ns", "us"])
def test_market_and_calendar_date_resolutions_do_not_suppress_completed_weeks(bar_unit, calendar_unit):
    cal = calendar()
    source = bars(cal)
    expected = build_triple_features(source, cal, settings())
    source["datetime"] = source["datetime"].astype(f"datetime64[{bar_unit}]")
    cal["date"] = cal["date"].astype(f"datetime64[{calendar_unit}]")
    actual = build_triple_features(source, cal, settings())
    assert actual.iloc[-1]["weekly_completed_count"] == 4
    assert actual["indicators_ready"].any()
    pd.testing.assert_frame_equal(actual, expected)
