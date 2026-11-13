from __future__ import annotations

import pandas as pd
import pytest

from backtest_layer.engine import run_backtest, run_comparison
from backtest_layer.execution import ExecutionConfig


def frame():
    return pd.DataFrame([
        {"symbol": "000001.SZ", "date": f"2025-01-{i + 2:02d}",
         "open": opening, "close": close, "high": max(opening, close) + .1,
         "low": min(opening, close) - .1, "volume": 1_000_000, "amount": 10_000_000,
         "is_st": False, "limit_pct": .3}
        for i, (opening, close) in enumerate([(10, 10), (10, 11), (12, 12.5), (13, 13.5)])
    ])


def execution():
    return ExecutionConfig(initial_cash=10_000, commission_rate=0, minimum_commission=0,
                           stamp_duty_rate=0, transfer_fee_rate=0, base_slippage_rate=0,
                           impact_coefficient=0)


def test_warmup_generates_first_open_signal_without_trades_or_performance():
    bars = frame()
    result = run_backtest(bars.iloc[2:], "sma_trend", {"fast_window": 1, "slow_window": 2}, execution(),
                          warmup_bars=bars.iloc[:2])
    assert result["bars"] == 2
    assert result["start"] == "2025-01-04"
    assert result["metrics"]["initial_nav"] == 10_000
    assert result["metrics"]["return_observations"] == 2
    assert result["fills"][0]["signal_time"] == "2025-01-03T15:00:00"
    assert result["fills"][0]["execution_time"] == "2025-01-04T09:30:00"
    assert result["fills"][0]["liquidity_reference_date"] == "2025-01-03"
    assert result["fills"][0]["execution_price"] == 12
    assert result["equity_curve"][0]["nav"] == 10400
    assert result["fifo_report"]["open_lots"]
    assert result["warmup"] == {"bars": 2, "start": "2025-01-02", "end": "2025-01-03", "excluded_from_performance": True}
    without = run_backtest(bars.iloc[2:], "sma_trend", {"fast_window": 1, "slow_window": 2}, execution())
    assert without["fills"] == []


def test_one_warmup_bar_is_supported_and_comparison_is_consistent():
    bars = frame()
    result = run_comparison(bars.iloc[2:], "buy_hold", execution_config=execution(), warmup_bars=bars.iloc[1:2])
    assert result["strategy"] == result["baselines"]["buy_hold"]
    assert result["strategy"]["fills"][0]["execution_time"] == "2025-01-04T09:30:00"
    assert result["baselines"]["cash"]["metrics"]["total_return"] == 0


def test_one_bar_warmup_can_estimate_missing_optional_amount_independently():
    bars = frame().drop(columns="amount")
    result = run_backtest(bars.iloc[1:], "buy_hold", execution_config=execution(), warmup_bars=bars.iloc[:1])
    assert result["warmup"]["bars"] == 1
    assert result["fills"][0]["execution_time"] == "2025-01-03T09:30:00"
    assert result["fills"][0]["liquidity_reference_date"] == "2025-01-02"


@pytest.mark.parametrize("warmup_slice", [slice(2, 4), slice(3, 4)])
def test_warmup_cannot_include_evaluation_or_future_dates(warmup_slice):
    bars = frame()
    with pytest.raises(ValueError):
        run_backtest(bars.iloc[2:], "buy_hold", warmup_bars=bars.iloc[warmup_slice])


def test_warmup_symbol_mismatch_is_rejected():
    bars = frame()
    warmup = bars.iloc[:2].copy()
    warmup["symbol"] = "600000.SH"
    with pytest.raises(ValueError, match="same symbol"):
        run_backtest(bars.iloc[2:], warmup_bars=warmup)


def test_cancel_check_interrupts_without_mutating_input():
    bars = frame()
    before = bars.copy(deep=True)
    checks = []
    def cancelled():
        checks.append(True)
        return len(checks) > 2
    with pytest.raises(InterruptedError):
        run_backtest(bars, "buy_hold", cancel_check=cancelled)
    pd.testing.assert_frame_equal(bars, before)
