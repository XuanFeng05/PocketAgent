"""Numerical and information-time contracts for user trend indicators."""
import numpy as np
import pandas as pd
import pytest

from feature_layer.user_trend import DERIVED_COLUMNS, build_user_trend_features


def calendar(start="2024-01-01", end="2024-07-14", holidays=()):
    days = pd.date_range(start, end)
    return pd.DataFrame({"date": days, "is_trading_day": (days.dayofweek < 5) & ~days.isin(pd.to_datetime(list(holidays)))})


def bars(cal, closes=None):
    days = cal.loc[cal["is_trading_day"], "date"].reset_index(drop=True)
    prices = np.asarray(closes if closes is not None else 20 + np.arange(len(days)) * .03 + np.sin(np.arange(len(days)) / 4), dtype=float)
    return pd.DataFrame({"symbol": "000001.SZ", "datetime": days, "open": prices,
                         "high": prices + .2, "low": prices - .2, "close": prices,
                         "volume": 1000., "freq": "daily", "adjust": "pre"})


def priced_sample():
    cal = calendar(end="2024-01-07")
    data = bars(cal, [10, 13, 12, 17, 18])
    data["open"] = [10, 12, 13, 15, 17]
    data["high"] = [11, 14, 14, 18, 19]
    data["low"] = [9, 12, 11, 14, 17]
    data["volume"] = [100, 200, 300, 400, 500]
    return data, cal


def test_wilder_atr_has_explicit_arithmetic_seed_and_recursive_followup():
    data, cal = priced_sample()
    result = build_user_trend_features(data, cal, atr_period=3)
    assert list(result["true_range"]) == [2, 4, 3, 6, 2]
    assert result["atr"].iloc[:2].isna().all()
    assert result["atr"].iloc[2:].tolist() == pytest.approx([3, 4, 10/3])
    assert result["daily_ready"].tolist() == [False, False, True, True, True]
    assert not result["indicators_ready"].any()


def test_atr_period_one_and_unseeded_long_period_are_defined():
    data, cal = priced_sample()
    one = build_user_trend_features(data, cal, atr_period=1)
    assert one["atr"].tolist() == one["true_range"].tolist()
    long = build_user_trend_features(data, cal, atr_period=14)
    assert long["atr"].isna().all()


def test_daily_ema_and_efi_match_hand_calculation_and_preserve_raw_lags():
    data, cal = priced_sample()
    result = build_user_trend_features(data, cal, atr_period=3)
    assert result.loc[0, "daily_ema13"] == result.loc[0, "daily_ema26"] == 10
    assert result.loc[1, "daily_ema13"] == pytest.approx(10 + 3/7)
    assert result.loc[1, "daily_ema26"] == pytest.approx(10 + 6/27)
    assert result["efi_raw"].tolist() == [0, 600, -300, 2000, 500]
    assert result.loc[1, "efi2"] == pytest.approx(400)
    assert result.loc[2, "efi2"] == pytest.approx(-200/3)
    assert result.loc[1, "efi13"] == pytest.approx(600/7)
    assert result.loc[2, "efi13"] == pytest.approx((600/7)*6/7-300/7)
    assert result.loc[2, "efi2_previous"] == result.loc[1, "efi2"]
    assert result.loc[2, "efi2_previous2"] == 0
    assert result.loc[2, "efi2_slope"] == pytest.approx(-1400/3)
    assert result.loc[2, "efi2_slope_previous"] == 400
    assert result.loc[0, ["efi2_previous", "efi2_previous2", "efi2_slope"]].isna().all()


def test_weekly_ema_is_seeded_by_complete_closes_and_visible_on_friday_only():
    cal = calendar(end="2024-01-21")
    data = bars(cal, [10]*5+[17]*5+[8]*5)
    out = build_user_trend_features(data, cal).set_index("datetime")
    assert out.loc["2024-01-04", "weekly_completed_count"] == 0
    assert pd.isna(out.loc["2024-01-04", "weekly_ema13"])
    assert out.loc["2024-01-05", "weekly_ema13"] == 10
    assert out.loc["2024-01-11", "weekly_ema13"] == 10
    assert out.loc["2024-01-12", "weekly_ema13"] == pytest.approx(11)
    assert out.loc["2024-01-12", "weekly_ema26"] == pytest.approx(10+14/27)
    assert out.loc["2024-01-12", "weekly_ema13_slope"] == pytest.approx(1)
    assert out.loc["2024-01-12", "weekly_ema26_slope"] == pytest.approx(14/27)
    assert out.loc["2024-01-15", "weekly_ema13_slope"] == pytest.approx(1)
    assert out.loc["2024-01-15", "weekly_available_at"] == pd.Timestamp("2024-01-12 15:00")
    assert out.loc["2024-01-19", "weekly_ema13_slope"] < 0


def test_week_with_friday_holiday_is_known_at_thursday_close():
    cal = calendar(end="2024-01-21", holidays=["2024-01-12"])
    out = build_user_trend_features(bars(cal), cal).set_index("datetime")
    assert out.loc["2024-01-10", "weekly_completed_count"] == 1
    assert out.loc["2024-01-11", "weekly_completed_count"] == 2
    assert out.loc["2024-01-11", "weekly_available_at"] == pd.Timestamp("2024-01-11 15:00")
    assert out.loc["2024-01-15", "weekly_available_at"] == pd.Timestamp("2024-01-11 15:00")


def test_truncated_boundary_weeks_never_seed_or_publish():
    cal = calendar(end="2024-01-28")
    data = bars(cal)
    data = data.loc[data["datetime"].between("2024-01-03", "2024-01-24")]
    out = build_user_trend_features(data, cal).set_index("datetime")
    assert out.loc["2024-01-05", "weekly_completed_count"] == 0
    assert out.loc["2024-01-12", "weekly_completed_count"] == 1
    assert out.loc["2024-01-12", "weekly_ema13"] == out.loc["2024-01-12", "close"]
    assert out.loc["2024-01-24", "weekly_available_at"] == pd.Timestamp("2024-01-19 15:00")
    shortened = calendar(end="2024-01-26")
    out = build_user_trend_features(bars(shortened), shortened)
    assert out.iloc[-1]["weekly_available_at"] == pd.Timestamp("2024-01-19 15:00")


def test_ready_requires_27_complete_weeks_and_seeded_atr():
    cal = calendar()
    result = build_user_trend_features(bars(cal), cal)
    ready = result.loc[result["indicators_ready"]]
    assert not ready.empty and ready.iloc[0]["weekly_completed_count"] == 27
    assert not result.loc[result["weekly_completed_count"] < 27, "weekly_ready"].any()
    unseeded = build_user_trend_features(bars(cal), cal, atr_period=1000)
    assert unseeded["weekly_ready"].any() and not unseeded["indicators_ready"].any()


def test_future_price_and_volume_perturbation_does_not_change_prior_features():
    cal = calendar()
    data = bars(cal)
    changed = data.copy()
    future = changed["datetime"].gt("2024-07-02")
    changed.loc[future, ["open", "high", "low", "close"]] *= 1.6
    changed.loc[future, "volume"] *= 4
    left = build_user_trend_features(data, cal)
    right = build_user_trend_features(changed, cal)
    pd.testing.assert_frame_equal(left.loc[~future], right.loc[~future])


@pytest.mark.parametrize("cutoff", ["2024-01-02", "2024-01-04", "2024-01-05", "2024-07-02"])
def test_prefix_and_full_input_agree_at_each_known_close(cutoff):
    cal = calendar()
    data = bars(cal)
    cut = data.loc[data["datetime"].le(cutoff)]
    full = build_user_trend_features(data, cal)
    prefix = build_user_trend_features(cut, cal)
    pd.testing.assert_frame_equal(full.iloc[:len(cut)], prefix)


def test_zero_volume_is_retained_and_force_is_zero_before_smoothing():
    data, cal = priced_sample()
    data.loc[2, "volume"] = 0
    data["is_tradeable"] = True
    data.loc[2, "is_tradeable"] = False
    result = build_user_trend_features(data, cal, atr_period=3)
    assert len(result) == len(data)
    assert result.loc[2, "efi_raw"] == 0
    assert result.loc[2, "efi2"] == pytest.approx(400/3)
    assert not result.loc[2, "is_tradeable"]
    assert result.loc[2, "atr"] == 3


def test_owned_fields_are_recomputed_and_inputs_unchanged():
    cal = calendar()
    data = bars(cal)
    data_before, cal_before = data.copy(deep=True), cal.copy(deep=True)
    expected = build_user_trend_features(data, cal)
    forged = expected.copy()
    for field in DERIVED_COLUMNS:
        forged[field] = "forged"
    actual = build_user_trend_features(forged, cal)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(data, data_before)
    pd.testing.assert_frame_equal(cal, cal_before)
    assert actual["session_close"].eq(actual["daily_available_at"]).all()
    assert actual["session_close"].dt.hour.eq(15).all()


@pytest.mark.parametrize("period", [0, -1, True, 1.5, "14", np.nan, np.inf])
def test_invalid_atr_period_rejected(period):
    data, cal = priced_sample()
    with pytest.raises(ValueError, match="positive integer"):
        build_user_trend_features(data, cal, atr_period=period)


@pytest.mark.parametrize("damage", ["missing_volume", "negative_volume", "nan_volume", "infinite_volume", "raw", "intraday", "missing_session", "closed_session"])
def test_invalid_daily_input_rejected(damage):
    cal = calendar(end="2024-01-28")
    data = bars(cal)
    if damage == "missing_volume": data = data.drop(columns="volume")
    if damage == "negative_volume": data.loc[1, "volume"] = -1
    if damage == "nan_volume": data.loc[1, "volume"] = np.nan
    if damage == "infinite_volume": data.loc[1, "volume"] = np.inf
    if damage == "raw": data["adjust"] = "none"
    if damage == "intraday": data.loc[1, "datetime"] = pd.Timestamp("2024-01-02 01:00")
    if damage == "missing_session": data = data.drop(index=2)
    if damage == "closed_session": cal.loc[cal["date"].eq(data.loc[2, "datetime"]), "is_trading_day"] = False
    with pytest.raises(ValueError):
        build_user_trend_features(data, cal)


@pytest.mark.parametrize("damage", ["duplicate", "missing_natural_date", "invalid_flag", "timezone", "short"])
def test_invalid_calendar_rejected(damage):
    cal = calendar(end="2024-01-28")
    data = bars(cal)
    if damage == "duplicate": cal = pd.concat([cal, cal.iloc[[0]]])
    if damage == "missing_natural_date": cal = cal.drop(index=6)
    if damage == "invalid_flag":
        cal["is_trading_day"] = cal["is_trading_day"].astype(object)
        cal.loc[0, "is_trading_day"] = "unknown"
    if damage == "timezone": cal["date"] = cal["date"].dt.tz_localize("UTC")
    if damage == "short": cal = cal.iloc[1:]
    with pytest.raises(ValueError):
        build_user_trend_features(data, cal)


@pytest.mark.parametrize("bar_unit", ["ns", "us"])
@pytest.mark.parametrize("calendar_unit", ["ns", "us"])
def test_date_resolution_does_not_change_week_publication(bar_unit, calendar_unit):
    cal = calendar()
    data = bars(cal)
    expected = build_user_trend_features(data, cal)
    data["datetime"] = data["datetime"].astype(f"datetime64[{bar_unit}]")
    cal["date"] = cal["date"].astype(f"datetime64[{calendar_unit}]")
    actual = build_user_trend_features(data, cal)
    pd.testing.assert_frame_equal(actual, expected)


def test_force_overflow_is_rejected_instead_of_producing_signals():
    data, cal = priced_sample()
    data["volume"] = 1e308
    with pytest.raises(ValueError, match="Force Index overflow"):
        build_user_trend_features(data, cal)
