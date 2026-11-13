from __future__ import annotations

import copy
import json

import pytest

from evaluation_layer.reports.fifo import build_fifo_report


def fill(identifier, side, shares, price, *, day=1, fees=0, symbol="A", **extra):
    return {"fill_id": identifier, "symbol": symbol, "side": side,
            "shares": shares, "execution_price": price, "total_fees": fees,
            "execution_time": f"2025-01-{day:02d}T09:30:00", "status": "filled", **extra}


def test_partial_fifo_allocates_all_costs_and_preserves_open_inventory():
    fills = [fill("b1", "buy", 100, 10, fees=10), fill("b2", "buy", 200, 12, fees=20),
             fill("s1", "sell", 150, 15, day=2, fees=15), fill("s2", "sell", 50, 11, day=3, fees=5)]
    original = copy.deepcopy(fills)
    report = build_fifo_report(fills, final_prices={"A": 13}, as_of="2025-01-03T15:00:00")
    assert fills == original
    rows = report["closed_trades"]
    assert [row["shares"] for row in rows] == [100, 50, 50]
    assert [row["buy_fill_id"] for row in rows] == ["b1", "b2", "b2"]
    assert [row["net_pnl"] for row in rows] == [480, 140, -60]
    summary = report["summary"]
    assert summary["closed_fragment_count"] == 3
    assert summary["sell_order_count"] == 2
    assert summary["completed_round_trip_count"] == 0
    assert summary["realized_net_pnl"] == 560
    assert summary["fragment_win_rate"] == pytest.approx(2/3)
    assert summary["sell_order_win_rate"] == .5
    assert summary["round_trip_win_rate"] is None
    assert summary["total_fees"] == 50
    assert summary["allocated_fees"] == 40
    assert summary["open_buy_fees"] == 10
    assert summary["fee_reconciliation_difference"] == 0
    assert summary["profit_factor"] == pytest.approx(620/60)
    lot = report["open_lots"][0]
    assert lot["remaining_shares"] == 100
    assert lot["remaining_cost"] == 1210
    assert lot["unrealized_pnl"] == 90
    assert lot["market_value"] == 1300
    json.dumps(report, allow_nan=False)


def test_sell_order_grouping_and_flat_to_flat_round_trips_have_distinct_counts():
    report = build_fifo_report([
        fill("b1", "buy", 100, 10), fill("b2", "buy", 100, 12),
        fill("s1", "sell", 150, 11, day=2, order_id="sell-order"),
        fill("s2", "sell", 50, 11, day=2, order_id="sell-order"),
        fill("b3", "buy", 100, 9, day=3), fill("s3", "sell", 100, 10, day=4),
    ])
    assert report["summary"]["closed_fragment_count"] == 4
    assert report["summary"]["filled_sell_count"] == 3
    assert report["summary"]["sell_order_count"] == 2
    assert report["summary"]["completed_round_trip_count"] == 2
    assert [trip["net_pnl"] for trip in report["round_trips"]] == [0, 100]
    assert report["summary"]["round_trip_win_rate"] == .5
    assert not report["open_lots"]


def test_initial_lots_are_explicit_and_different_symbols_do_not_cross_match():
    initial = [{"lot_id": "old", "symbol": "A", "shares": 100, "entry_price": 10,
                "entry_time": "2024-12-01", "total_fees": 10}]
    report = build_fifo_report([fill("sale", "sell", 100, 12, fees=5),
                               fill("other", "buy", 100, 3, symbol="B")], initial_lots=initial)
    assert report["summary"]["realized_net_pnl"] == 185
    assert report["round_trips"][0]["includes_initial_position"] is True
    assert report["open_lots"][0]["symbol"] == "B"
    assert report["open_lots"][0]["unrealized_pnl"] is None
    assert report["summary"]["unrealized_net_pnl"] is None
    assert report["summary"]["initial_lot_fees"] == 10


def test_ordering_is_chronological_then_explicit_sequence_or_input_order():
    buy, sell = fill("b", "buy", 100, 10), fill("s", "sell", 100, 11, day=2)
    assert build_fifo_report([sell, buy])["summary"]["realized_net_pnl"] == 100
    same_day_sell = {**sell, "execution_time": buy["execution_time"]}
    with pytest.raises(ValueError, match="Unmatched sell"):
        build_fifo_report([same_day_sell, buy])
    report = build_fifo_report([{**same_day_sell, "sequence": 2}, {**buy, "sequence": 1}])
    assert report["summary"]["realized_net_pnl"] == 100


def test_no_trades_and_only_winners_have_null_undefined_rates():
    empty = build_fifo_report([])
    assert empty["summary"]["fragment_win_rate"] is None
    assert empty["summary"]["profit_factor"] is None
    winner = build_fifo_report([fill("b", "buy", 100, 10), fill("s", "sell", 100, 11, day=2)])
    assert winner["summary"]["fragment_win_rate"] == 1
    assert winner["summary"]["profit_factor"] is None
    assert winner["summary"]["profit_factor_status"] == "no_losing_fragments"
    json.dumps(winner, allow_nan=False)


def test_blocked_orders_are_reported_but_not_inventory_or_fees():
    report = build_fifo_report([{"fill_id": "reject", "symbol": "A", "status": "blocked", "shares": 0,
                                 "execution_price": 0, "total_fees": 0, "reason": "t_plus_one"}])
    assert report["summary"]["blocked_order_count"] == 1
    assert report["summary"]["filled_sell_count"] == 0
    assert report["rejected_orders"][0]["reason"] == "t_plus_one"
    assert report["summary"]["total_fees"] == 0


@pytest.mark.parametrize("change", [
    {"shares": 0}, {"shares": 1.5}, {"shares": True}, {"execution_price": float("nan")},
    {"execution_price": float("inf")}, {"execution_price": -1}, {"total_fees": -1},
    {"execution_time": None}, {"symbol": ""}, {"sequence": 1.2},
    {"direction": -1}, {"status": "unknown"}, {"commission": 1, "total_fees": 2},
])
def test_malformed_fill_fails_closed(change):
    with pytest.raises(ValueError):
        build_fifo_report([{**fill("b", "buy", 100, 10), **change}])


def test_duplicate_ids_overselling_and_invalid_initial_state_raise():
    buy = fill("same", "buy", 100, 10)
    with pytest.raises(ValueError, match="Duplicate"):
        build_fifo_report([buy, buy])
    with pytest.raises(ValueError, match="Unmatched sell"):
        build_fifo_report([buy, fill("sell", "sell", 101, 12, day=2)])
    with pytest.raises(ValueError, match="Initial lots"):
        build_fifo_report([buy], initial_lots=[{"symbol": "A", "shares": 100, "price": 10, "entry_time": "2026-01-01"}])
    with pytest.raises(ValueError, match="as_of"):
        build_fifo_report([buy], as_of="2024-01-01")
    with pytest.raises(ValueError, match="mix naive"):
        build_fifo_report([buy], as_of="2025-01-02T15:00:00+08:00")


def test_fractional_fee_allocations_reconcile_after_many_partial_sales():
    records = [fill("buy", "buy", 7, "10.123", fees="0.1")]
    records += [fill(f"s{i}", "sell", 1, "11.011", fees="0.03", day=i+2) for i in range(7)]
    report = build_fifo_report(records)
    assert report["summary"]["total_fees"] == pytest.approx(.31)
    assert report["summary"]["allocated_fees"] == pytest.approx(.31)
    assert report["summary"]["fee_reconciliation_difference"] == pytest.approx(0, abs=1e-15)
    assert report["summary"]["open_shares"] == 0


def test_report_integrates_actual_engine_fills_without_torch():
    from backtest_layer.data import demo_bars
    from backtest_layer.engine import run_backtest
    result = run_backtest(demo_bars())
    report = build_fifo_report(result["fills"], final_prices={result["symbol"]: result["equity_curve"][-1]["close"]},
                               as_of=result["equity_curve"][-1]["valuation_time"])
    assert report["summary"]["open_shares"] == result["metrics"]["open_shares"]
    assert report["summary"]["input_fill_fees"] == pytest.approx(result["metrics"]["total_fees"])
    assert report["summary"]["realized_net_pnl"] + report["summary"]["unrealized_net_pnl"] == pytest.approx(
        result["metrics"]["final_nav"] - result["metrics"]["initial_nav"], abs=1e-7)


def test_open_lot_output_can_seed_next_reporting_period_without_losing_fees():
    first = build_fifo_report([fill("b", "buy", 100, 10, fees=10)])
    second = build_fifo_report([fill("s", "sell", 100, 12, fees=5, day=2)], initial_lots=first["open_lots"])
    assert second["summary"]["initial_lot_fees"] == 10
    assert second["summary"]["realized_net_pnl"] == 185
    assert second["summary"]["fee_reconciliation_difference"] == 0
