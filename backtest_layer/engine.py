from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, time
from typing import Any
from collections.abc import Callable

import numpy as np

from backtest_layer.account import PortfolioAccount
from backtest_layer.actions import PortfolioAction
from backtest_layer.data import normalize_daily_bars
from backtest_layer.execution import AshareExecutor, ExecutionConfig
from backtest_layer.market import MarketSnapshot
from evaluation_layer.metrics.portfolio import portfolio_metrics
from feature_layer.rules import price_limit_pct_for_symbol
from strategy_layer.base import AccountView, BarView, OrderIntent, Strategy
from strategy_layer.registry import create_strategy


ASSUMPTIONS = [
    "Single-stock independent account; results are not a shared-cash portfolio.",
    "Signals use completed closes; orders execute at the next available daily open.",
    "Opening liquidity capacity uses the previous observed day's amount as a proxy, not today's final volume or amount; fills are estimates, not auction reconstruction.",
    "A zero-volume day conservatively rejects execution; daily high/low is never used to improve the fill price.",
    "Dated ST input is required. Explicit per-day limits override the existing project board-limit helper; IPO exceptions and historical rule regime changes require supplied limits.",
    "Unadjusted prices are used. Dividends/splits and other corporate actions are not modeled; use unaffected intervals. Explicit corporate_action rows are rejected.",
    "Open holdings are marked at the final close, not forcibly liquidated; cash has no interest.",
    "Order size is executable-capacity fraction, not target portfolio weight.",
]


def _execution_config(value: ExecutionConfig | dict[str, Any] | None) -> ExecutionConfig:
    if isinstance(value, ExecutionConfig):
        return value
    return ExecutionConfig(**dict(value or {}))


def _bar_context(row: dict[str, Any], config: ExecutionConfig) -> dict[str, Any]:
    context = {}
    for key, value in row.items():
        if not key.startswith("context_"):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        if key.endswith("_available"):
            if value is True or value == 1 or str(value).lower() == "true":
                value = True
            elif value is False or value == 0 or str(value).lower() == "false":
                value = False
            elif value is None:
                value = False
            else:
                raise ValueError(f"{key} must be an explicit availability flag")
        context[key] = value
    cutoff = datetime.combine(row["datetime"].date(), time(15))
    if context.get("context_decision_time"):
        claimed = datetime.fromisoformat(str(context["context_decision_time"]))
        if claimed.tzinfo is not None or claimed > cutoff:
            raise ValueError("Context observation belongs to a future or nonlocal decision time")
    for key in ("context_index_observed_date", "context_industry_observed_date"):
        if context.get(key) and datetime.fromisoformat(str(context[key])).date() > cutoff.date():
            raise ValueError("Context source date cannot exceed the strategy decision date")
    context.update({"lot_size": config.lot_size, "minimum_commission": config.minimum_commission,
                    "can_buy_at_close": bool(row.get("market_can_buy", True)) and not bool(row["is_st"]) and row["volume"] > 0,
                    "can_sell_at_close": bool(row.get("market_can_sell", True)) and row["volume"] > 0})
    return context


def _policy_details(policy: Strategy) -> dict:
    return dict(policy.diagnostics()) if callable(getattr(policy, "diagnostics", None)) else {}


def _opening_market(row: dict[str, Any], previous: dict[str, Any], symbol: str) -> MarketSnapshot:
    reference = float(row.get("limit_reference_close", previous["close"]))
    limit = float(row.get("limit_pct", price_limit_pct_for_symbol(symbol, is_st=bool(row["is_st"]))))
    opening = float(row["open"])
    # Rounded daily limits and opening status are execution facts, not model inputs.
    upper, lower = round(reference * (1 + limit), 2), round(reference * (1 - limit), 2)
    tradeable = bool(row.get("is_tradeable", True)) and float(row["volume"]) > 0
    can_buy = bool(row.get("market_can_buy", True)) and not bool(row["is_st"]) and opening < upper - 1e-8
    can_sell = bool(row.get("market_can_sell", True)) and opening > lower + 1e-8
    return MarketSnapshot(
        decision_time=datetime.combine(row["datetime"].date(), time(9, 30)),
        symbols=(symbol,), execution_prices=np.asarray([opening], dtype=float),
        liquidity_amount=np.asarray([float(previous["amount"])], dtype=float),
        limit_reference_close=np.asarray([reference], dtype=float), limit_pct=np.asarray([limit], dtype=float),
        active_mask=np.asarray([True]), is_tradeable=np.asarray([tradeable]),
        market_can_buy=np.asarray([can_buy]), market_can_sell=np.asarray([can_sell]),
    )


def run_backtest(
    bars: Any, strategy: str | Strategy = "sma_trend", parameters: dict[str, Any] | None = None,
    execution_config: ExecutionConfig | dict[str, Any] | None = None, symbol: str | None = None,
    *, warmup_bars: Any | None = None, cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run a deterministic close-signal / next-open single-stock backtest.

    The shared AshareExecutor is also re-exported at the historical Agent path.
    No torch, checkpoint, tensor cache or RL lifecycle is needed here.
    """
    frame = normalize_daily_bars(bars, symbol=symbol)
    selected_symbol = str(frame.iloc[0]["symbol"])
    config = _execution_config(execution_config)
    policy = create_strategy(strategy, parameters) if isinstance(strategy, str) else deepcopy(strategy)
    if getattr(policy, "requires_context", False) and not any(str(column).startswith("context_") for column in frame.columns):
        raise ValueError("This strategy requires point-in-time context; enrich bars from explicit benchmark/industry sources first.")
    account = PortfolioAccount((selected_symbol,), initial_cash=config.initial_cash)
    executor = AshareExecutor(config)
    rows = frame.to_dict("records")
    warmup_rows: list[dict[str, Any]] = []
    if warmup_bars is not None and len(warmup_bars):
        warmup = normalize_daily_bars(warmup_bars, symbol=symbol, min_rows=1)
        if str(warmup.iloc[0]["symbol"]) != selected_symbol:
            raise ValueError("Warmup and evaluation must use the same symbol.")
        if warmup["datetime"].max() >= frame.iloc[0]["datetime"]:
            raise ValueError("Warmup must end strictly before the evaluation interval.")
        warmup_rows = warmup.to_dict("records")
    equity: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    closes: list[float] = []
    pending: tuple[OrderIntent, dict[str, Any]] | None = None
    turnover = 0.0
    context_counts = {"benchmark": 0, "industry": 0}

    def check_cancelled() -> None:
        if cancel_check and cancel_check():
            raise InterruptedError("Backtest cancelled")

    # Warmup observes prior closes with a fresh cash account, without fills,
    # fees or NAV samples. Only its final close can queue a scored opening order.
    for warmup_index, row in enumerate(warmup_rows):
        check_cancelled()
        closes.append(float(row["close"]))
        close_time = datetime.combine(row["datetime"].date(), time(15, 0))
        intent = policy.on_close(BarView(selected_symbol, close_time, tuple(closes), _bar_context(row, config)),
                                AccountView(account.cash, account.net_asset_value, 0, 0))
        if intent is not None and not isinstance(intent, OrderIntent):
            raise TypeError("A strategy must return OrderIntent or None.")
        if warmup_index == len(warmup_rows) - 1 and intent is not None and intent.direction:
            signal = {"signal_id": "signal_000001", "signal_time": close_time.isoformat(),
                      "direction": intent.direction, "capacity_fraction": intent.capacity_fraction,
                      "quantity_semantics": "executable_capacity_fraction", "reason": intent.reason,
                      "status": "pending", "origin": "warmup", "policy_details": _policy_details(policy)}
            signals.append(signal)
            pending = intent, signal

    for index, row in enumerate(rows):
        check_cancelled()
        day = row["datetime"].date()
        account.start_trading_day(day)
        if index or warmup_rows:
            previous = rows[index - 1] if index else warmup_rows[-1]
            opening = _opening_market(row, previous, selected_symbol)
            account.mark_to_market(opening.symbols, opening.execution_prices)
            if pending is not None:
                intent, signal = pending
                result = executor.execute(account, opening, PortfolioAction(
                    np.asarray([intent.direction]), np.asarray([intent.capacity_fraction])))
                turnover += result.turnover_value
                signal["status"] = "attempted"
                signal["execution_time"] = opening.decision_time.isoformat()
                for fill in result.fills:
                    item = asdict(fill)
                    item.update({"fill_id": f"fill_{len(fills) + 1:06d}", "signal_id": signal["signal_id"],
                                 "direction": int(fill.direction), "side": "buy" if fill.direction > 0 else "sell",
                                 "signal_time": signal["signal_time"], "order_time": opening.decision_time.isoformat(),
                                 "execution_time": opening.decision_time.isoformat(), "total_fees": fill.total_fees,
                                 "liquidity_reference_date": previous["datetime"].date().isoformat()})
                    fills.append(item)
                pending = None
        account.mark_to_market((selected_symbol,), np.asarray([row["close"]], dtype=float))
        account.advance_holding_bars()
        close_time = datetime.combine(day, time(15, 0))
        position = account.position(selected_symbol)
        equity.append({"date": day.isoformat(), "time": close_time.isoformat(), "valuation_time": close_time.isoformat(),
                       "nav": account.net_asset_value, "cash": account.cash, "shares": position.total_shares,
                       "sellable_shares": position.sellable_shares, "close": float(row["close"]),
                       "fees_to_date": account.total_fees})
        closes.append(float(row["close"]))
        visible_context = _bar_context(row, config)
        context_counts["benchmark"] += bool(visible_context.get("context_index_available", False))
        context_counts["industry"] += bool(visible_context.get("context_industry_available", False))
        intent = policy.on_close(
            BarView(selected_symbol, close_time, tuple(closes), visible_context),
            AccountView(account.cash, account.net_asset_value, position.total_shares, position.sellable_shares),
        )
        if intent is not None and not isinstance(intent, OrderIntent):
            raise TypeError("A strategy must return OrderIntent or None.")
        if intent is not None and intent.direction != 0:
            signal = {"signal_id": f"signal_{len(signals) + 1:06d}", "signal_time": close_time.isoformat(),
                      "direction": intent.direction, "capacity_fraction": intent.capacity_fraction,
                      "quantity_semantics": "executable_capacity_fraction", "reason": intent.reason,
                      "status": "pending_end_of_data" if index == len(rows) - 1 else "pending",
                      "policy_details": _policy_details(policy)}
            signals.append(signal)
            pending = intent, signal

    metrics = portfolio_metrics([row["nav"] for row in equity], initial_nav=config.initial_cash,
                                turnover_value=turnover, total_fees=account.total_fees)
    executed = [fill for fill in fills if fill["status"] == "filled"]
    metrics.update({"executed_trades": len(executed), "blocked_orders": len(fills) - len(executed),
                    "total_orders": len(fills), "realized_pnl_average_cost": account.realized_pnl,
                    "open_shares": account.position(selected_symbol).total_shares})
    result = {"schema_version": 1, "symbol": selected_symbol, "strategy": policy.name,
            "parameters": dict(parameters or {}), "execution": asdict(config),
            "start": equity[0]["date"], "end": equity[-1]["date"], "bars": len(rows),
            "metrics": metrics, "equity_curve": equity, "fills": fills, "signals": signals,
            "assumptions": list(ASSUMPTIONS),
            "policy_details": _policy_details(policy),
            "context_coverage": {"enabled": any(str(column).startswith("context_") for column in frame.columns),
                                 "bars": len(rows),
                                 "benchmark_available_bars": context_counts["benchmark"],
                                 "industry_available_bars": context_counts["industry"]},
            "warmup": {"bars": len(warmup_rows),
                       "start": warmup_rows[0]["datetime"].date().isoformat() if warmup_rows else None,
                       "end": warmup_rows[-1]["datetime"].date().isoformat() if warmup_rows else None,
                       "excluded_from_performance": True}}
    from evaluation_layer.reports.fifo import build_fifo_report
    result["fifo_report"] = build_fifo_report(
        fills, final_prices={selected_symbol: float(rows[-1]["close"])},
        as_of=equity[-1]["valuation_time"],
    )
    if result["context_coverage"]["enabled"]:
        result["assumptions"].append("External context uses explicit close publication times and historical membership known_at; missing/current-date coverage is reported, never filled from future observations.")
    if policy.name == "ridge_forecast":
        result["assumptions"].append("Ridge adapts online using matured labels only; training-window scaling and coefficients never use unseen closes. Its close-to-close forecast is not directly attainable next-open trading return.")
    if policy.name == "constrained_random":
        result["assumptions"].append("Random action choices use a fixed seed and known closing account/market masks; the next opening executor can still reject changed conditions.")
    return result


def run_comparison(
    bars: Any, strategy: str = "sma_trend", parameters: dict[str, Any] | None = None,
    execution_config: ExecutionConfig | dict[str, Any] | None = None, symbol: str | None = None,
    *, warmup_bars: Any | None = None, cancel_check: Callable[[], bool] | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    config = _execution_config(execution_config)
    result = run_backtest(bars, strategy, parameters, config, symbol, warmup_bars=warmup_bars, cancel_check=cancel_check)
    baselines = {name: run_backtest(bars, name, execution_config=config, symbol=symbol,
                                  warmup_bars=warmup_bars, cancel_check=cancel_check) for name in ("cash", "buy_hold")}
    if random_seed is not None:
        baselines["constrained_random"] = run_backtest(bars, "constrained_random", {"seed": random_seed}, config, symbol,
                                                       warmup_bars=warmup_bars, cancel_check=cancel_check)
    return {"strategy": result, "baselines": baselines, "comparison": [
        {"strategy": name, **item["metrics"]}
        for name, item in [(result["strategy"], result), *baselines.items()]
    ], "excess_return_vs_buy_hold": result["metrics"]["total_return"] - baselines["buy_hold"]["metrics"]["total_return"]}
