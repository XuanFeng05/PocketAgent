"""A multi-symbol portfolio with one cash ledger and observable daily prices."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, time
from typing import Any, Callable

import numpy as np

from backtest_layer.account import PortfolioAccount
from backtest_layer.execution import AshareExecutor, ExecutionConfig
from backtest_layer.market import MarketSnapshot
from backtest_layer.portfolio_data import normalize_portfolio_bars
from backtest_layer.target_orders import execute_target_weights
from evaluation_layer.metrics.portfolio import portfolio_metrics
from evaluation_layer.reports.fifo import build_fifo_report
from feature_layer.rules import price_limit_pct_for_symbol
from strategy_layer.portfolio import resolve_portfolio_settings, select_target_weights


ASSUMPTIONS = [
    "This is a shared-cash multi-symbol account, not an average of independent stock returns.",
    "Target weights use observed closing histories; orders execute at the next observed portfolio date's open, sells before buys.",
    "Cash-constrained buys are scaled jointly by the shared executor, with deterministic symbol order for lot rounding and fees.",
    "Opening liquidity uses each symbol's previous available daily amount as a proxy, never the current day's final amount.",
    "A missing bar cannot trade. Existing shares retain the last known close and are explicitly marked stale; no future initial prices are backfilled.",
    "Membership facts are used only after membership_known_at; a user-declared static pool does not establish a survivorship-bias-free historical universe.",
    "Momentum lookback counts observed closes per symbol; rebalance_every counts union dates present in the selected data.",
    "Equal-weight hold freezes its initial visible eligible basket; blocked entries retry until first fill, then those shares are held without periodic rebalancing.",
    "Corporate actions and delisting settlements are not modeled. Historical exceptional price limits need explicit daily limits; stale marked values are not proof of realizable proceeds.",
    "Fees, lot constraints, T+1 and market rejection use the same executor as single-stock research. Final holdings are marked, not forcibly liquidated.",
]


def _market(day, symbols, current, previous, account, memberships) -> MarketSnapshot:
    values = {name: [] for name in ["prices", "liquidity", "reference", "limit", "active", "tradeable", "buy", "sell"]}
    opening_time = datetime.combine(day.date(), time(9, 30))
    for symbol in symbols:
        row, prior = current.get(symbol), previous.get(symbol)
        if row is None:
            price = account.position(symbol).last_price
            reference = price
            limit = price_limit_pct_for_symbol(symbol)
            active = tradeable = buy = sell = False
        else:
            price = float(row["open"])
            reference = float(row.get("limit_reference_close", prior["close"] if prior else price))
            limit = float(row.get("limit_pct", price_limit_pct_for_symbol(symbol, is_st=bool(row["is_st"]))))
            active = True
            tradeable = bool(row.get("is_tradeable", True)) and float(row["volume"]) > 0
            member = bool(row["is_member"]) if row["membership_known_at"] <= opening_time else memberships.get(symbol, False)
            buy = member and not bool(row["is_st"]) and bool(row.get("market_can_buy", True)) and price < round(reference * (1 + limit), 2) - 1e-8
            sell = bool(row.get("market_can_sell", True)) and price > round(reference * (1 - limit), 2) + 1e-8
        for name, value in [("prices", price), ("liquidity", float(prior["amount"]) if prior else 0.),
                            ("reference", reference), ("limit", limit), ("active", active),
                            ("tradeable", tradeable), ("buy", buy), ("sell", sell)]:
            values[name].append(value)
    return MarketSnapshot(opening_time, symbols, np.asarray(values["prices"], dtype=float),
                          np.asarray(values["liquidity"], dtype=float), np.asarray(values["reference"], dtype=float),
                          np.asarray(values["limit"], dtype=float), np.asarray(values["active"], dtype=bool),
                          np.asarray(values["tradeable"], dtype=bool), np.asarray(values["buy"], dtype=bool),
                          np.asarray(values["sell"], dtype=bool))


def run_portfolio(bars: Any, settings=None, execution_config=None,
                  cancel_check: Callable[[], bool] | None = None) -> dict:
    resolved = resolve_portfolio_settings(settings)
    frame = normalize_portfolio_bars(bars, require_membership=resolved.require_membership)
    execution = execution_config if isinstance(execution_config, ExecutionConfig) else ExecutionConfig(**dict(execution_config or {}))
    universe = tuple(sorted(frame["symbol"].unique()))
    account = PortfolioAccount(universe, initial_cash=execution.initial_cash)
    executor = AshareExecutor(execution)
    histories = {symbol: [] for symbol in universe}
    previous, memberships = {}, {}
    equity, fills, signals, positions = [], [], [], []
    pending = None
    hold_targets = None
    hold_entered = set()
    turnover = 0.
    stale_marks = 0

    for index, (day, daily_frame) in enumerate(frame.groupby("datetime", sort=True)):
        if cancel_check and cancel_check():
            raise InterruptedError("Portfolio backtest cancelled")
        current = {row["symbol"]: row for row in daily_frame.to_dict("records")}
        account.start_trading_day(day.date())
        opening = _market(day, universe, current, previous, account, memberships)
        account.mark_to_market(universe, opening.execution_prices)
        if pending is not None:
            signal = pending
            weights = dict(signal["target_weights"])
            if resolved.strategy == "equal_weight_hold" and hold_entered:
                # Freeze existing shares at their current opening values. Only
                # the not-yet-filled initial basket receives new cash orders.
                nav = account.net_asset_value
                held = {symbol: account.position(symbol).market_value / nav for symbol in hold_entered}
                unfilled = {symbol: value for symbol, value in weights.items() if symbol not in hold_entered}
                capacity = max(0., 1 - sum(held.values()))
                scale = min(1., capacity / sum(unfilled.values())) if unfilled else 0.
                weights = {**held, **{symbol: value * scale for symbol, value in unfilled.items()}}
            result = execute_target_weights(account, opening, weights, executor)
            turnover += result["turnover_value"]
            signal["status"] = "attempted"
            signal["execution_time"] = opening.decision_time.isoformat()
            signal["order_intents"] = result["intents"]
            for fill in result["fills"]:
                item = {**fill, "fill_id": f"portfolio_fill_{len(fills)+1:06d}",
                        "signal_id": signal["signal_id"], "order_id": f"{signal['signal_id']}:{fill['symbol']}:{fill['side']}",
                        "signal_time": signal["signal_time"], "order_time": opening.decision_time.isoformat(),
                        "execution_time": opening.decision_time.isoformat(),
                        "liquidity_reference_date": previous[fill["symbol"]]["datetime"].date().isoformat() if fill["symbol"] in previous else None}
                fills.append(item)
                if resolved.strategy == "equal_weight_hold" and item["side"] == "buy" and item["status"] == "filled":
                    hold_entered.add(item["symbol"])
            pending = None
        close_time = datetime.combine(day.date(), time(15))
        for symbol, row in current.items():
            account.mark_to_market((symbol,), np.asarray([row["close"]], dtype=float))
            histories[symbol].append(float(row["close"]))
            if row["membership_known_at"] <= close_time:
                memberships[symbol] = bool(row["is_member"])
            previous[symbol] = row
        account.advance_holding_bars()
        nav = account.net_asset_value
        stale = [symbol for symbol in universe if account.position(symbol).total_shares and symbol not in current]
        stale_marks += len(stale)
        equity.append({"date": day.date().isoformat(), "time": close_time.isoformat(), "valuation_time": close_time.isoformat(),
                       "nav": nav, "cash": account.cash, "position_value": nav-account.cash,
                       "position_count": sum(p.total_shares > 0 for p in account.positions.values()),
                       "fees_to_date": account.total_fees, "stale_positions": len(stale)})
        for symbol in universe:
            p = account.position(symbol)
            positions.append({"date": day.date().isoformat(), "symbol": symbol, "shares": p.total_shares,
                              "sellable_shares": p.sellable_shares, "locked_shares": p.locked_shares,
                              "mark_price": p.last_price, "market_value": p.market_value,
                              "weight": p.market_value/nav, "is_stale": symbol in stale,
                              "last_observed_date": previous[symbol]["datetime"].date().isoformat() if symbol in previous else None})
        eligible = [symbol for symbol, row in current.items() if memberships.get(symbol, False)
                    and not row["is_st"] and row.get("is_tradeable", True) and row["volume"] > 0]
        target = scores = None
        if resolved.strategy == "momentum_top_k" and index % resolved.rebalance_every == 0:
            target, scores = select_target_weights(histories, eligible, resolved)
        elif resolved.strategy == "equal_weight_hold":
            if hold_targets is None and eligible:
                hold_targets, scores = select_target_weights(histories, eligible, resolved)
            if hold_targets is not None and set(hold_targets) - hold_entered:
                target = dict(hold_targets)
        if target is not None:
            signal = {"signal_id": f"portfolio_signal_{len(signals)+1:06d}", "signal_time": close_time.isoformat(),
                      "target_weights": target, "scores": scores or {}, "eligible_symbols": sorted(eligible),
                      "quantity_semantics": "target_portfolio_weight", "status": "pending"}
            signals.append(signal)
            pending = signal
    if pending is not None:
        pending["status"] = "pending_end_of_data"
    metrics = portfolio_metrics([r["nav"] for r in equity], initial_nav=execution.initial_cash,
                                turnover_value=turnover, total_fees=account.total_fees)
    metrics.update({"executed_trades": sum(r["status"] == "filled" for r in fills),
                    "blocked_orders": sum(r["status"] == "blocked" for r in fills), "total_orders": len(fills),
                    "realized_pnl_average_cost": account.realized_pnl,
                    "open_shares": sum(p.total_shares for p in account.positions.values()),
                    "final_position_count": sum(p.total_shares > 0 for p in account.positions.values()),
                    "stale_valuation_count": stale_marks})
    fifo = build_fifo_report(fills, final_prices={s: p.last_price for s,p in account.positions.items() if p.total_shares}, as_of=equity[-1]["time"])
    sources = sorted(set(frame["membership_source"]))
    return {"schema_version": 1, "mode": "portfolio", "symbol": "PORTFOLIO", "symbols": list(universe),
            "strategy": resolved.strategy, "parameters": asdict(resolved), "execution": asdict(execution),
            "start": equity[0]["date"], "end": equity[-1]["date"], "bars": len(equity), "input_rows": len(frame),
            "metrics": metrics, "equity_curve": equity, "fills": fills, "signals": signals, "positions": positions,
            "fifo_report": fifo, "membership_sources": sources,
            "historical_membership_required": resolved.require_membership,
            "assumptions": list(ASSUMPTIONS)}


def run_portfolio_comparison(bars: Any, settings=None, execution_config=None,
                             cancel_check: Callable[[], bool] | None = None) -> dict:
    resolved = resolve_portfolio_settings(settings)
    frame = normalize_portfolio_bars(bars, require_membership=resolved.require_membership)
    result = run_portfolio(frame, resolved, execution_config, cancel_check)
    baselines = {name: run_portfolio(frame, replace(resolved, strategy=name), execution_config, cancel_check)
                 for name in ("cash", "equal_weight_hold")}
    return {"mode": "portfolio", "strategy": result, "baselines": baselines,
            "comparison": [{"strategy": name, **report["metrics"]} for name, report in [(result["strategy"], result), *baselines.items()]],
            "excess_return_vs_equal_weight_hold": result["metrics"]["total_return"] - baselines["equal_weight_hold"]["metrics"]["total_return"]}
