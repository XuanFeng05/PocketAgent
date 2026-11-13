"""Synthetic tests of feature timing and formula conventions, not market data."""
import numpy as np
import pandas as pd
import pytest

from research_layer.multiscale_signals import (
    SIGNAL_IDS, WEEKLY_FILTER_IDS, CONDITION_IDS, build_features,
    signal_masks, weekly_signal_masks, weekly_filter_masks, condition_masks,
    indicator_frame, rsi, wilder_mean,
)


def market_fixture(weeks=115):
    start = pd.Timestamp("2020-01-06")
    cal = pd.DataFrame({"date": pd.date_range(start, periods=weeks*7)})
    cal["is_trading_day"] = cal.date.dt.dayofweek.lt(5)
    cal["session_close"] = cal.date+np.timedelta64(16, "h")
    cal.loc[~cal.is_trading_day, "session_close"] = pd.NaT
    days = cal.loc[cal.is_trading_day, "date"].reset_index(drop=True)
    x = np.arange(len(days))
    close = 80+.04*x+5*np.sin(x/8)+2*np.sin(x/2)
    b = pd.DataFrame({"datetime": days, "symbol": "TEST", "open": close-.1,
                      "high": close+1, "low": close-1, "close": close,
                      "volume": 100000.+10000*np.sin(x/11), "adjust": "pre", "freq": "daily"})
    return b, cal


def test_wilder_seeds_at_sma_and_resets_after_missing():
    v = pd.Series([np.nan, 1., 2., 3., 6., np.nan, 4., 5., 6.])
    r = wilder_mean(v, 3)
    assert r.iloc[:3].isna().all()
    assert r.iloc[3] == pytest.approx(2.)
    assert r.iloc[4] == pytest.approx(10/3)
    assert r.iloc[5:8].isna().all()
    assert r.iloc[8] == pytest.approx(5.)


def test_rsi_extremes_and_flat_history_are_finite_and_neutral():
    rising = pd.Series(np.arange(1., 80.))
    assert rsi(rising, 14).iloc[:14].isna().all()
    assert (rsi(rising, 14).iloc[14:] == 100).all()
    assert (rsi(rising.iloc[::-1].reset_index(drop=True), 14).iloc[14:] == 0).all()
    assert (rsi(pd.Series([5.]*80), 14).iloc[14:] == 50).all()


def test_zero_range_indicators_do_not_generate_extreme_false_signal():
    b = pd.DataFrame({k: [50.]*220 for k in ["open", "high", "low", "close"]})
    b["volume"] = 10000.
    f = indicator_frame(b)
    assert f.stoch_fast14.iloc[-1] == 50
    assert f.williams_r14.iloc[-1] == -50
    assert f.adx14.iloc[-1] == 0
    assert not np.isinf(f.to_numpy()).any()


def test_donchian_excludes_current_high_and_stochastic_smoothing():
    b = pd.DataFrame({"open": np.arange(1., 51.), "high": np.arange(2., 52.),
                      "low": np.arange(.5, 50.5), "close": np.arange(1.5, 51.5), "volume": 1000.})
    f = indicator_frame(b)
    assert f.donchian_high20.iloc[20] == b.high.iloc[:20].max()
    assert f.donchian_high20.iloc[20] < b.high.iloc[20]
    assert f.stoch_k14_3.iloc[20] == pytest.approx(f.stoch_fast14.iloc[18:21].mean())
    assert f.stoch_d14_3_3.iloc[20] == pytest.approx(f.stoch_k14_3.iloc[18:21].mean())


def test_future_price_and_volume_changes_leave_past_features_unchanged():
    b, cal = market_fixture()
    f = build_features(b, cal)
    cutoff = b.datetime.iloc[530]
    changed = b.copy()
    future = changed.datetime.gt(cutoff)
    changed.loc[future, ["open", "high", "low", "close"]] *= 2
    changed.loc[future, "volume"] *= 10
    g = build_features(changed, cal)
    pd.testing.assert_frame_equal(f.loc[f.datetime.le(cutoff)], g.loc[g.datetime.le(cutoff)])


def test_exact100_complete_week_warmup_and_published_events():
    b, cal = market_fixture()
    f = build_features(b, cal)
    first = f.loc[f.ready_week100].iloc[0]
    assert first.weekly_completed_count == 100
    assert first.datetime == b.datetime.iloc[499]
    assert first.weekly_published
    assert not f.loc[f.datetime.lt(first.datetime), "ready_week100"].any()
    for mask in weekly_signal_masks(f).values():
        assert not np.any(mask & ~f.weekly_published.to_numpy())


def test_partial_ipo_week_does_not_count_and_no_future_history_required():
    b, cal = market_fixture()
    short = b.iloc[2:].reset_index(drop=True)  # IPO Wednesday, incomplete first week
    f = build_features(short, cal)
    assert f.loc[f.ready_week100].datetime.iloc[0] == b.datetime.iloc[504]
    cut = b.datetime.iloc[520]
    prefix = short.loc[short.datetime.le(cut)]
    g = build_features(prefix, cal)
    pd.testing.assert_frame_equal(f.loc[f.datetime.le(cut)].reset_index(drop=True), g)


def test_holiday_week_publishes_at_actual_early_close_not_friday():
    b, cal = market_fixture()
    th = b.datetime.iloc[503]  # Thursday of week101
    fr = b.datetime.iloc[504]
    cal.loc[cal.date.eq(fr), "is_trading_day"] = False
    # Keep the closed day's clock value deliberately: trading flag rules it out.
    cal.loc[cal.date.eq(th), "session_close"] = th+np.timedelta64(13, "h")
    b = b.loc[~b.datetime.eq(fr)].reset_index(drop=True)
    f = build_features(b, cal)
    row = f.loc[f.datetime.eq(th)].iloc[0]
    assert row.weekly_published
    assert row.weekly_available_at == th+np.timedelta64(13, "h")
    previous = f.loc[f.datetime.lt(th)].iloc[-1]
    assert previous.weekly_completed_count == row.weekly_completed_count-1


def test_missing_open_day_rejected_but_zero_volume_preserved():
    b, cal = market_fixture()
    with pytest.raises(ValueError, match="Missing open-session"):
        build_features(b.drop(index=520), cal)
    b.loc[520, "volume"] = 0
    f = build_features(b, cal)
    assert len(f) == len(b)
    assert not f.tradeable.iloc[520]


def test_unfinished_tail_week_cannot_publish():
    b, cal = market_fixture()
    prefix = b.iloc[:502]  # ends Tuesday of week101
    f = build_features(prefix, cal)
    assert f.weekly_completed_count.iloc[-1] == 100
    assert not f.weekly_published.iloc[-1]


def test_feature_contract_and_price_gate_use_known_vendor_close():
    b, cal = market_fixture()
    b["vendor_close"] = b.close
    b.loc[500:510, "vendor_close"] = 9.99
    f = build_features(b, cal)
    assert set(signal_masks(f)) == set(SIGNAL_IDS)
    assert set(weekly_filter_masks(f)) == set(WEEKLY_FILTER_IDS)
    assert set(condition_masks(f)) == set(CONDITION_IDS)
    assert isinstance(f.index, pd.RangeIndex)
    assert f.loc[500:510, "ready"].all()
    assert not f.loc[500:510, "universe_trade_eligible"].any()
    assert not f.loc[500:510, ["signal__"+s for s in SIGNAL_IDS]].to_numpy().any()
    assert not f.loc[500:510, ["wsignal__"+s for s in SIGNAL_IDS]].to_numpy().any()


def test_condition_reference_windows_exclude_signal_day():
    b, _ = market_fixture()
    f = indicator_frame(b)
    i = 220
    assert f.atr_pct_prior126_median.iloc[i] == pytest.approx(f.atr_pct.iloc[i-126:i].median())
    assert f.volume_prior20_mean.iloc[i] == pytest.approx(b.volume.iloc[i-20:i].mean())
