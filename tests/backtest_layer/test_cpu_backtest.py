from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backtest_layer.account import PortfolioAccount
from backtest_layer.actions import PortfolioAction, TradeDirection
from backtest_layer.data import load_daily_bars, normalize_daily_bars
from backtest_layer.engine import run_backtest, run_comparison
from backtest_layer.execution import AshareExecutor, ExecutionConfig
from backtest_layer.market import MarketSnapshot


def config(**overrides):
    return ExecutionConfig(**dict(initial_cash=10_000.0, commission_rate=0.0, minimum_commission=0.0,
                                 stamp_duty_rate=0.0, transfer_fee_rate=0.0, base_slippage_rate=0.0,
                                 impact_coefficient=0.0, **overrides))


def bars(closes=(10.0, 10.5, 11.0), opens=(10.0, 11.0, 10.5)):
    return pd.DataFrame([{"symbol": "000001.SZ", "datetime": f"2025-01-{day:02d}",
                          "open": opening, "close": close, "high": max(opening, close) + 0.1,
                          "low": min(opening, close) - 0.1, "volume": 1_000_000, "amount": 10_000_000.0,
                          "is_st": False, "limit_pct": 0.2}
                         for day, opening, close in zip(range(2, 2 + len(closes)), opens, closes)])


def test_close_signal_executes_next_open_and_preserves_true_daily_nav():
    result = run_backtest(bars(), "buy_hold", execution_config=config())
    fill = result["fills"][0]
    assert fill["signal_time"] == "2025-01-02T15:00:00"
    assert fill["execution_time"] == "2025-01-03T09:30:00"
    assert fill["execution_price"] == 11.0
    assert fill["shares"] == 900
    assert result["equity_curve"][0]["nav"] == 10_000.0
    assert result["equity_curve"][1]["nav"] == 100 + 900 * 10.5
    assert result["equity_curve"][2]["nav"] == 100 + 900 * 11.0


def test_opening_capacity_does_not_read_current_day_final_amount():
    original = bars()
    original.loc[0, "amount"] = 50_000.0
    changed = original.copy()
    changed.loc[1, "amount"] = 5_000_000_000.0
    first = run_backtest(original, "buy_hold", execution_config=config())["fills"][0]
    second = run_backtest(changed, "buy_hold", execution_config=config())["fills"][0]
    assert first == second
    assert first["liquidity_reference_date"] == "2025-01-02"


def test_strategy_and_baselines_use_same_executor_and_costs():
    execution = replace(config(), commission_rate=0.0003, minimum_commission=5)
    result = run_comparison(bars(), "buy_hold", execution_config=execution)
    assert result["strategy"] == result["baselines"]["buy_hold"]
    assert result["baselines"]["buy_hold"]["metrics"]["total_fees"] >= 5
    assert result["baselines"]["cash"]["metrics"]["total_return"] == 0
    assert not result["baselines"]["cash"]["fills"]


def market(day=2):
    return MarketSnapshot(datetime(2025, 1, day, 9, 30), ("000001.SZ",), np.array([10.0]),
                          np.array([10_000_000.0]), np.array([10.0]), np.array([0.1]),
                          np.array([True]), np.array([True]), np.array([True]), np.array([True]))


def test_shared_executor_respects_t_plus_one_and_lots():
    executor = AshareExecutor(config())
    account = PortfolioAccount(("000001.SZ",), initial_cash=10_000)
    buy = executor.execute(account, market(), PortfolioAction(np.array([1]), np.array([1.0])))
    assert buy.executed_fills[0].shares == 1000
    assert account.position("000001.SZ").sellable_shares == 0
    blocked = executor.execute(account, market(), PortfolioAction(np.array([-1]), np.array([1.0])))
    assert blocked.fills[0].reason == "t_plus_one_locked"
    sell = executor.execute(account, market(3), PortfolioAction(np.array([-1]), np.array([1.0])))
    assert sell.executed_fills[0].shares == 1000
    assert account.net_asset_value == 10_000


def test_market_rejection_retries_and_never_buys_st():
    frame = bars()
    frame.loc[1, "is_st"] = True
    result = run_backtest(frame, "buy_hold", execution_config=config())
    assert result["fills"][0]["status"] == "blocked"
    assert result["fills"][0]["reason"] == "market_buy_blocked"
    assert result["fills"][1]["status"] == "filled"
    assert result["fills"][1]["execution_time"] == "2025-01-04T09:30:00"


def test_zero_volume_halt_and_limit_up_are_not_filled():
    frame = bars()
    frame.loc[1, "volume"] = 0
    frame.loc[2, ["open", "close", "high", "low"]] = [12.6, 12.6, 12.6, 12.6]
    result = run_backtest(frame, "buy_hold", execution_config=config())
    assert [fill["reason"] for fill in result["fills"]] == ["not_tradeable", "market_buy_blocked"]
    assert result["metrics"]["open_shares"] == 0


def test_legacy_imports_resolve_to_same_shared_classes():
    from agent_layer.actions.action import PortfolioAction as OldAction
    from agent_layer.portfolio.account import PortfolioAccount as OldAccount
    from agent_layer.environment.execution import AshareExecutor as OldExecutor
    assert OldAction is PortfolioAction
    assert OldAccount is PortfolioAccount
    assert OldExecutor is AshareExecutor


@pytest.mark.parametrize("change", ["duplicate", "missing_st", "adjusted", "nan", "missing_date", "corporate_action"])
def test_invalid_or_unsupported_input_is_rejected(change):
    frame = bars()
    if change == "duplicate": frame.loc[1, "datetime"] = frame.loc[0, "datetime"]
    if change == "missing_st": frame = frame.drop(columns="is_st")
    if change == "adjusted": frame["adjust"] = "pre"
    if change == "nan": frame.loc[0, "close"] = np.nan
    if change == "missing_date": frame.loc[0, "datetime"] = None
    if change == "corporate_action": frame["corporate_action"] = [False, True, False]
    with pytest.raises(ValueError): normalize_daily_bars(frame)


def test_csv_loading_filters_without_mutating_source(tmp_path):
    path = tmp_path / "daily.csv"
    bars().to_csv(path, index=False)
    original = path.read_bytes()
    result = load_daily_bars(csv_path=path, symbol="000001.SZ", start="2025-01-03")
    assert len(result) == 2
    assert path.read_bytes() == original


@pytest.mark.parametrize("bounds", [{}, {"start": "2025-01-01"}, {"end": "2025-01-10"}, {"start": "2025-01-01", "end": "2025-01-10"}])
def test_csv_missing_date_is_rejected_before_date_filtering(tmp_path, bounds):
    frame = bars()
    frame.loc[1, "datetime"] = None
    path = tmp_path / "missing_date.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing dates"):
        load_daily_bars(csv_path=path, symbol="000001.SZ", **bounds)


def test_market_storage_reader_merges_dated_st_without_writing_database(tmp_path):
    import duckdb
    path = tmp_path / "market.duckdb"
    frame = bars().drop(columns=["is_st", "limit_pct"])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["freq"], frame["adjust"], frame["source"], frame["pctChg"] = "daily", "none", "test", 0.0
    statuses = pd.DataFrame({"symbol": frame["symbol"], "date": frame["datetime"], "is_st": [False, True, False], "source": "test"})
    with duckdb.connect(str(path)) as connection:
        connection.register("input_bars", frame)
        connection.register("input_status", statuses)
        connection.execute("CREATE TABLE kline_bars AS SELECT * FROM input_bars")
        connection.execute("CREATE TABLE stock_status_daily AS SELECT * FROM input_status")
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    loaded = load_daily_bars(db_path=path, symbol="000001.SZ")
    assert loaded["is_st"].tolist() == [False, True, False]
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime


def test_invalid_execution_values_are_rejected():
    for values in [{"initial_cash": float("nan")}, {"commission_rate": float("inf")}, {"max_position_ratio": float("nan")}]:
        with pytest.raises(ValueError): ExecutionConfig(**values)
