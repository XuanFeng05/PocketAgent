"""Shared cash and fractional adjusted-price units for dated target weights."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd

from evaluation_layer.metrics.portfolio import portfolio_metrics
from feature_layer.triple_screen import _local_dates, _normalize_calendar
from research_layer.technical import normalize_technical_bars


def _date(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError("Expected a local calendar date.")
    return day


def _price_panel(bars_by_symbol, days):
    if not isinstance(bars_by_symbol, Mapping) or not bars_by_symbol:
        raise ValueError("Provide at least one symbol.")
    if any(not isinstance(key, str) or not key or key != key.strip().upper() for key in bars_by_symbol):
        raise ValueError("Symbol keys must be nonempty uppercase identifiers.")
    symbols = sorted(bars_by_symbol)
    openings, closings, tradeable = [], [], []
    for symbol in symbols:
        frame = pd.DataFrame(bars_by_symbol[symbol]).copy()
        if "datetime" not in frame and "date" in frame:
            frame = frame.rename(columns={"date": "datetime"})
        if "datetime" not in frame or "volume" not in frame:
            raise ValueError(f"{symbol}: timestamps and volume required.")
        frame["datetime"] = _local_dates(frame["datetime"], "Portfolio prices")
        frame = frame.loc[frame["datetime"].between(days[0], days[-1])]
        frame = normalize_technical_bars(frame, min_rows=1)
        if not frame["symbol"].eq(symbol).all():
            raise ValueError(f"Unexpected symbol in bars for {symbol}.")
        if not pd.DatetimeIndex(frame["datetime"]).equals(days):
            raise ValueError(f"{symbol}: bars must cover exactly the evaluation sessions.")
        available = frame["volume"].gt(0)
        if "is_tradeable" in frame:
            available &= frame["is_tradeable"]
        openings.append(frame["open"].to_numpy(dtype=float))
        closings.append(frame["close"].to_numpy(dtype=float))
        tradeable.append(available.to_numpy(dtype=bool))
    return symbols, np.column_stack(openings), np.column_stack(closings), np.column_stack(tradeable)


def _schedule(decisions, sessions, days, symbols):
    seen, schedule = set(), {}
    for decision in decisions:
        signal = _date(decision["signal_date"])
        if signal in seen or signal not in sessions:
            raise ValueError("Each signal must use a unique exchange session.")
        seen.add(signal)
        weights = decision["weights"]
        if not isinstance(weights, Mapping) or set(weights).difference(symbols):
            raise ValueError("Target weights must be keyed by known symbols.")
        for weight in weights.values():
            if (isinstance(weight, bool) or not isinstance(weight, Real)
                    or not math.isfinite(weight) or not 0 <= weight <= 1):
                raise ValueError("Weights must be finite, long-only fractions.")
        if math.fsum(weights.values()) > 1 + 1e-12:
            raise ValueError("Weights exceed available capital.")
        successor = sessions.get_loc(signal) + 1
        if successor == len(sessions):
            continue
        execution = sessions[successor]
        if "execution_date" in decision and _date(decision["execution_date"]) != execution:
            raise ValueError("Execution must use the first session after signal close.")
        if days[0] <= execution <= days[-1]:
            schedule[execution] = (signal.date().isoformat(), dict(weights))
    return schedule


def _mark(units, prices, cash):
    values = units * prices
    invested = float(values.sum())
    nav = cash + invested
    if not math.isfinite(nav) or nav <= 0 or cash < 0 or (units < 0).any():
        raise ArithmeticError("Invalid portfolio value or negative position.")
    return nav, invested / nav, values


def _rebalance(day, signal_date, weights, symbols, prices, available, units, cash, rate, last_buy):
    pre_nav, _, _ = _mark(units, prices, cash)
    targets = pre_nav / (1 + rate) * np.array([weights.get(s, 0.0) for s in symbols])
    changes = targets - units * prices
    tolerance = 1e-13 * pre_nav
    fills, blocked = [], []

    def book(i, new_units):
        nonlocal cash
        signed_units = float(new_units - units[i])
        signed_value = signed_units * float(prices[i])
        fee = abs(signed_value) * rate
        cash_before = cash
        cash -= signed_value + fee
        if cash < -tolerance:
            raise ArithmeticError("Buy cost exceeds shared cash.")
        cash = max(cash, 0.0)  # Proportional allocation can leave a negative floating-point residual.
        units[i] = new_units
        side = "buy" if signed_units > 0 else "sell"
        if side == "buy":
            last_buy[i] = day
        fills.append({"date": day, "signal_date": signal_date, "symbol": symbols[i], "side": side,
                      "price": float(prices[i]), "units": abs(signed_units), "delta_units": signed_units,
                      "notional": abs(signed_value), "fee": fee, "cash_before": cash_before, "cash_after": cash})

    for i in np.flatnonzero(changes < -tolerance):
        reason = "suspension" if not available[i] else "T+1" if last_buy[i] == day else None
        if reason:
            blocked.append({"symbol": symbols[i], "side": "sell", "reason": reason})
        else:
            book(i, targets[i] / prices[i])
    requested = np.flatnonzero(changes > tolerance)
    buyers = [i for i in requested if available[i]]
    blocked.extend({"symbol": symbols[i], "side": "buy", "reason": "suspension"}
                   for i in requested if not available[i])
    total_needed = math.fsum(float(changes[i]) * (1 + rate) for i in buyers)
    scale = min(1., cash / total_needed) if total_needed else 1.
    for i in buyers:
        amount = float(changes[i]) * scale
        if amount > tolerance:
            book(i, units[i] + amount / prices[i])
    post_nav, _, post_values = _mark(units, prices, cash)
    fees = math.fsum(f["fee"] for f in fills)
    if not math.isclose(post_nav + fees, pre_nav, rel_tol=1e-11):
        raise ArithmeticError("Rebalance does not conserve capital after fees.")
    rebalance = {"signal_date": signal_date, "execution_date": day, "pre_nav": pre_nav,
                 "post_nav": post_nav, "target_weights": weights, "cash_after": cash,
                 "weights_after": {s: float(v/post_nav) for s,v in zip(symbols,post_values) if v > 0},
                 "buy_scale": scale, "blocked": blocked, "fills_count": len(fills), "fees": fees,
                 "turnover": math.fsum(f["notional"] for f in fills) / pre_nav}
    return cash, fills, rebalance


def run_technical_portfolio(bars_by_symbol, calendar, *, start, end, decisions, cost_bps=10):
    """Price prior-close decisions at the next session open using a single budget."""
    start, end = _date(start), _date(end)
    if start > end:
        raise ValueError("Start must precede or equal end.")
    if (isinstance(cost_bps, bool) or not isinstance(cost_bps, Real)
            or not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000):
        raise ValueError("Invalid one-sided fee in basis points.")
    cal = _normalize_calendar(calendar, start, end)
    sessions = pd.DatetimeIndex(cal.loc[cal["is_trading_day"], "date"])
    days = sessions[(sessions >= start) & (sessions <= end)]
    if days.empty:
        raise ValueError("No exchange sessions in evaluation window.")
    symbols, opening, closing, available = _price_panel(bars_by_symbol, days)
    schedule = _schedule(decisions, sessions, days, symbols)
    equity, fills, rebalances = [], [], []
    units = np.zeros(len(symbols), dtype=float)
    last_buy = [None] * len(symbols)
    cash = previous_nav = 1.0
    for i, date in enumerate(days):
        day = date.date().isoformat()
        if date in schedule:
            signal, weights = schedule[date]
            cash, executed, record = _rebalance(day, signal, weights, symbols, opening[i], available[i],
                                                units, cash, cost_bps/10_000, last_buy)
            fills.extend(executed)
            rebalances.append(record)
        nav, exposure, values = _mark(units, closing[i], cash)
        holdings = {symbols[j]: {"units": float(units[j]), "market_value": float(values[j]),
                                 "weight": float(values[j]/nav)} for j in np.flatnonzero(units)}
        equity.append({"date": day, "nav": nav, "cash": cash, "exposure": exposure,
                       "holdings": holdings, "daily_return": nav/previous_nav - 1})
        previous_nav = nav
    fees = math.fsum(f["fee"] for f in fills)
    notional = math.fsum(f["notional"] for f in fills)
    metrics = portfolio_metrics([e["nav"] for e in equity], initial_nav=1., turnover_value=notional,
                                total_fees=fees)
    number_days = int((end - start).days + 1)
    cagr = previous_nav ** (365.25/number_days) - 1
    counts = Counter(f["side"] for f in fills)
    metrics.update(cagr_calendar=cagr, calendar_days=number_days,
                   mean_exposure=float(np.mean([e["exposure"] for e in equity])),
                   max_exposure=max(e["exposure"] for e in equity),
                   max_positions=max(len(e["holdings"]) for e in equity),
                   rebalance_count=len(rebalances), active_rebalance_days=sum(bool(r["fills_count"]) for r in rebalances),
                   fills_count=len(fills), buy_count=counts["buy"], sell_count=counts["sell"],
                   traded_notional=notional, turnover=sum(r["turnover"] for r in rebalances))
    return {"mode": "shared_cash_technical_portfolio", "price_basis": "pre", "initial_cash": 1.,
            "requested_start": start.date().isoformat(), "requested_end": end.date().isoformat(),
            "start": equity[0]["date"], "end": equity[-1]["date"], "cost_bps": float(cost_bps),
            "equity_curve": equity, "fills": fills, "rebalances": rebalances, "metrics": metrics,
            "assumptions": ["Calendar CAGR uses inclusive requested days divided by 365.25.",
                            "One shared cash account and fractional forward-adjusted units.",
                            "Daily volume and tradeability gate execution as a suspension proxy.",
                            "Execution uses next-open prices both to size and fill target quantities.",
                            "Opening auction queues, capacity, lots and full taxes are outside the model."]}
