from __future__ import annotations

from hashlib import sha256
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from backtest_layer.data import demo_bars
from feature_layer.context import enrich_daily_bars, demo_context_sources


def sources(count=25):
    bars = demo_bars(symbol="000001.SZ", days=count)
    context = demo_context_sources(bars)
    return bars, context


def test_context_uses_publication_time_and_never_future_observations():
    bars, context = sources()
    delayed = context["benchmark"].copy()
    delayed.loc[5, "available_at"] = delayed.loc[9, "available_at"]
    before = enrich_daily_bars(bars, delayed, context["industry"], context["membership"])
    assert not bool(before.loc[5, "context_index_available"])
    assert pd.isna(before.loc[5, "context_index_ret_1bar"])
    # An old observation not yet published cannot affect a newer visible row.
    changed = delayed.copy()
    changed.loc[5, "close"] *= 20
    after = enrich_daily_bars(bars, changed, context["industry"], context["membership"])
    columns = [c for c in before if c.startswith("context_")]
    pd.testing.assert_frame_equal(before.loc[:8, columns], after.loc[:8, columns])
    later = context["industry"].copy()
    later.loc[15:, "close"] *= 2
    mutated = enrich_daily_bars(bars, delayed, later, context["membership"])
    pd.testing.assert_frame_equal(before.loc[:14, columns], mutated.loc[:14, columns])


def test_membership_is_historical_and_must_be_known_at_decision():
    bars, context = sources()
    industry = context["industry"].copy()
    industry["industry"] = "A"
    other = industry.copy()
    other["industry"] = "B"
    other["close"] = np.linspace(200, 100, len(other))
    days = pd.to_datetime(bars["datetime"])
    membership = pd.DataFrame([
        {"symbol":"000001.SZ","industry":"A","valid_from":days.iloc[0],"valid_to":days.iloc[10],"known_at":days.iloc[0]},
        {"symbol":"000001.SZ","industry":"B","valid_from":days.iloc[10],"valid_to":None,"known_at":days.iloc[12].to_pydatetime()+timedelta(hours=14)},
    ])
    result = enrich_daily_bars(bars, context["benchmark"], pd.concat([industry,other]), membership)
    assert result.loc[9,"context_industry"] == "A"
    assert result.loc[10,"context_industry"] is None
    assert not result.loc[11,"context_industry_available"]
    assert result.loc[12,"context_industry"] == "B"
    assert result.loc[12,"context_industry_ret_5bar"] < 0


def test_missing_context_is_explicit_and_strict_mode_rejects_it():
    bars, context = sources()
    truncated = context["benchmark"].iloc[:-1]
    result = enrich_daily_bars(bars, truncated)
    assert not result.iloc[-1]["context_index_available"]
    assert result.iloc[-1]["context_index_observed_date"] == str(pd.Timestamp(bars.iloc[-2]["datetime"]).date())
    assert pd.isna(result.iloc[-1]["context_index_ret_5bar"])
    with pytest.raises(ValueError, match="Incomplete point-in-time"):
        enrich_daily_bars(bars, truncated, require_complete=True)


def test_relative_returns_require_same_observed_dates_and_ma_is_twenty_bars():
    bars, context = sources()
    result = enrich_daily_bars(bars, **context)
    assert pd.isna(result.loc[18,"context_index_ma20bar_distance"])
    expected = context["benchmark"].loc[19,"close"] / context["benchmark"].loc[:19,"close"].mean()-1
    assert result.loc[19,"context_index_ma20bar_distance"] == pytest.approx(expected)
    stock = bars.drop(index=4).reset_index(drop=True)
    missing_stock = enrich_daily_bars(stock, context["benchmark"])
    assert pd.isna(missing_stock.loc[5,"context_stock_minus_index_ret_5bar"])


@pytest.mark.parametrize("case", ["duplicate", "early", "overlap", "missing_available", "timezone"])
def test_invalid_point_in_time_contracts_fail(case):
    bars, context = sources()
    benchmark = context["benchmark"].copy()
    member = context["membership"].copy()
    if case == "duplicate": benchmark = pd.concat([benchmark, benchmark.iloc[:1]])
    if case == "early": benchmark.loc[0,"available_at"] = benchmark.loc[0,"date"]
    if case == "overlap": member = pd.concat([member, member])
    if case == "missing_available": benchmark = benchmark.drop(columns="available_at")
    if case == "timezone": benchmark["available_at"] = pd.to_datetime(benchmark["available_at"]).dt.tz_localize("Asia/Shanghai")
    with pytest.raises(ValueError):
        enrich_daily_bars(bars, benchmark, context["industry"], member)


def test_csv_sources_are_read_only_and_multi_benchmark_requires_choice(tmp_path):
    bars, context = sources()
    original = bars.copy(deep=True)
    path = tmp_path/"benchmark.csv"
    context["benchmark"].to_csv(path,index=False)
    digest = sha256(path.read_bytes()).hexdigest()
    result = enrich_daily_bars(bars, path)
    assert result["context_index_available"].all()
    assert sha256(path.read_bytes()).hexdigest() == digest
    pd.testing.assert_frame_equal(bars,original)
    another = context["benchmark"].copy()
    another["symbol"] = "OTHER"
    with pytest.raises(ValueError,match="benchmark_symbol"):
        enrich_daily_bars(bars,pd.concat([context["benchmark"],another]))
