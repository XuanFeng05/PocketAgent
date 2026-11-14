"""Frozen, ICT-inspired daily sweep/structure/FVG research; not all of ICT.

This is a CN long/cash technical NAV on pre-adjusted daily bars. Order activation
uses the observed opening quote, not an overnight resting auction order. Daily
OHLC does not reveal event order: ambiguous stops are handled conservatively.
"""
from __future__ import annotations

from collections import Counter
from datetime import time
import math
from statistics import mean

import numpy as np
import pandas as pd

from evaluation_layer.metrics.portfolio import portfolio_metrics
from research_layer.technical import normalize_technical_bars


VERSION = "ict_daily_sweep_fvg_pre_v1"
TOLERANCE = 1e-12
SETTINGS = {"version": VERSION, "sweep_lookback": 20, "structure_lookback": 5,
            "displacement_after_sweep": 5, "prior_tr_mean_period": 14,
            "entry_window_bars": 5, "maximum_holding_bars": 20,
            "entry_fraction_of_gap": .5, "price_basis": "pre", "market": "CN"}
ASSUMPTIONS = [
    "This frozen daily adaptation is one explicit hypothesis, not a test of every ICT model or an original intraday/kill-zone model.",
    "Pre-adjusted single-symbol long/cash technical NAV starts at one; no real shares, cash dividends, portfolio allocation or leverage are modeled.",
    "The earliest active sweep is retained independently of portfolio state. The first fully qualified three-bar FVG retires that sweep; no right-confirmed swing pivots are used.",
    "Signals become available only at the third bar's 15:00 close. Orders may activate at opening quotes on the next five calendar-trading bars, with explicit suspension bars counting toward expiry.",
    "An observed open at or below the setup stop cancels before activation; an open between stop and limit gives the opening price. This assumes zero-latency quote-conditioned execution, not an overnight auction order.",
    "A tradeable bar opening above the limit and later touching both limit and stop enters at the limit. Its low cannot retrospectively cancel the entry.",
    "CN T+1 forbids entry-day selling: a stop touch latches exit at the next tradeable open, while an entry-day target touch is only recorded. An intraday-entry target touch may have preceded the fill and is separately marked as order-unknown.",
    "After entry day, pending exits take priority, then opening gaps. If both intraday stop and target are touched, stop executes first and ambiguity is recorded.",
    "A stop touch on a non-tradeable holding bar remains latched. At the twentieth holding close, counting entry as bar one, a time exit is queued for the next tradeable open.",
    "One-sided proportional NAV cost is charged at each executed entry or exit. Spread, taxes, lots, price-limit queues and auction capacity are not reconstructed.",
    "The final open position is marked to the last evaluation close without a fabricated sale. Cash earns zero interest. Different exit policies may change later entry opportunities.",
]


def _date(value, name):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError(f"{name} must be a valid local date at midnight")
    return stamp


def _flag(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError("Calendar requires explicit boolean or 0/1 trading flags")


def _prepare(bars, calendar, end):
    original = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    if "datetime" not in original and "date" in original:
        original = original.rename(columns={"date": "datetime"})
    if "datetime" not in original:
        raise ValueError("Daily bars require datetime")
    dates = pd.to_datetime(original["datetime"], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Daily bars require valid local dates at midnight")
    original["datetime"] = dates.astype("datetime64[ns]")
    # Cap before any feature/price validation: future prices cannot affect a run.
    original = original.loc[original["datetime"] <= end].copy()
    frame = normalize_technical_bars(original)
    if "volume" not in frame:
        raise ValueError("ICT daily research requires explicit volume")
    cal = calendar.copy(deep=True) if isinstance(calendar, pd.DataFrame) else pd.DataFrame(calendar)
    if cal.empty or not {"date", "is_trading_day"}.issubset(cal.columns):
        raise ValueError("Calendar requires date and is_trading_day")
    cal_dates = pd.to_datetime(cal["date"], errors="raise")
    if cal_dates.isna().any() or cal_dates.dt.tz is not None or not cal_dates.eq(cal_dates.dt.normalize()).all():
        raise ValueError("Calendar requires local dates at midnight")
    cal["date"] = cal_dates.astype("datetime64[ns]")
    first = frame.iloc[0]["datetime"]
    cal = cal.loc[cal["date"].between(first, end)].copy()
    if cal["date"].duplicated().any():
        raise ValueError("Duplicate calendar dates")
    cal["is_trading_day"] = cal["is_trading_day"].map(_flag)
    cal = cal.sort_values("date").reset_index(drop=True)
    expected_natural = pd.date_range(first, end)
    if not pd.DatetimeIndex(cal["date"]).equals(expected_natural):
        raise ValueError("Calendar must cover all natural dates including closed days")
    sessions = pd.DatetimeIndex(cal.loc[cal["is_trading_day"], "date"])
    actual = pd.DatetimeIndex(frame["datetime"].astype("datetime64[ns]"))
    if len(actual.difference(sessions)):
        raise ValueError("Daily bars fall on calendar-closed dates")
    if len(sessions.difference(actual)):
        raise ValueError("Missing open-session bars; suspension must be explicit")
    return frame


def _tradeable(row):
    return bool(row.get("is_tradeable", True)) and float(row["volume"]) > 0


def detect_ict_setups(frame, *, first_sweep_index=0):
    """Causal sequential detector; input is normalized daily bars, not raw data."""
    rows = frame.to_dict("records")
    active, setups, sweeps = None, [], []
    true_ranges = []
    for i, row in enumerate(rows):
        prior_close = rows[i-1]["close"] if i else row["close"]
        tr = max(row["high"]-row["low"], abs(row["high"]-prior_close), abs(row["low"]-prior_close))
        true_ranges.append(float(tr))
        if active is not None:
            s = active["index"]
            if row["low"] <= active["stop"]:
                active.update(status="invalidated", resolved_date=row["datetime"].date().isoformat())
                active = None
            elif i > s+6:
                active.update(status="expired", resolved_date=row["datetime"].date().isoformat())
                active = None
            elif s+2 <= i <= s+6:
                d = i-1
                a, b, c = rows[d-1], rows[d], row
                prior_tr = mean(true_ranges[d-14:d]) if d >= 14 else None
                displacement = (prior_tr is not None and b["close"] > b["open"]
                                and b["close"]-b["open"] >= prior_tr
                                and b["close"] > active["structure_line"])
                gap = (c["low"] > a["high"] and b["low"] <= a["high"] and b["high"] >= c["low"])
                if displacement and gap and all(_tradeable(x) for x in (a,b,c)):
                    setup = {"setup_id": f"setup_{len(setups)+1:06d}", "confirm_index": i,
                             "confirm_date": c["datetime"].date().isoformat(),
                             "available_at": pd.Timestamp.combine(c["datetime"].date(), time(15)).isoformat(),
                             "sweep_id": active["sweep_id"], "sweep_index": s,
                             "sweep_date": active["date"], "swept_level": active["swept_level"],
                             "structure_line": active["structure_line"], "stop": active["stop"],
                             "displacement_date": b["datetime"].date().isoformat(),
                             "prior_tr_mean14": prior_tr, "displacement_body": float(b["close"]-b["open"]),
                             "fvg_lower": float(a["high"]), "fvg_upper": float(c["low"]),
                             "limit": float((a["high"]+c["low"])/2)}
                    setups.append(setup)
                    active.update(status="confirmed", resolved_date=setup["confirm_date"], setup_id=setup["setup_id"])
                    active = None
        if active is None and i >= max(20,first_sweep_index) and _tradeable(row):
            swept = min(r["low"] for r in rows[i-20:i])
            if row["low"] < swept and row["close"] > swept:
                active = {"sweep_id": f"sweep_{len(sweeps)+1:06d}", "index": i,
                          "date": row["datetime"].date().isoformat(), "swept_level": float(swept),
                          "structure_line": float(max(r["high"] for r in rows[i-5:i])),
                          "stop": float(row["low"]), "status": "waiting"}
                sweeps.append(active)
    return setups, sweeps


def run_ict_daily(bars, calendar, *, start, end, cost_bps=10, target_r=2):
    start_date, end_date = _date(start, "start"), _date(end, "end")
    if start_date > end_date:
        raise ValueError("start must not follow end")
    for value, name in ((cost_bps,"cost_bps"),(target_r,"target_r")):
        if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= cost_bps < 10000 or target_r <= 0:
        raise ValueError("Require 0 <= cost_bps < 10000 and target_r > 0")
    frame = _prepare(bars, calendar, end_date)
    rows = frame.to_dict("records")
    indices = [i for i,r in enumerate(rows) if r["datetime"] >= start_date]
    if not indices:
        raise ValueError("No evaluation bars")
    first = indices[0]
    setups, sweeps = detect_ict_setups(frame,first_sweep_index=first)
    setups = [s for s in setups if rows[s["confirm_index"]]["datetime"] >= start_date]
    setup_by_index = {s["confirm_index"]:s for s in setups}
    nav, exposure, fees, turnover = 1.0, 0, 0.0, 0.0
    position, pending_order, pending_exit = None, None, None
    orders, fills, trades, equity, decisions = [], [], [], [], []
    cost = float(cost_bps)/10000

    def fill(day, target, price, phase, reason, setup_id):
        nonlocal nav, exposure, fees, turnover
        before = nav
        fee = before*cost
        nav -= fee
        fees += fee
        turnover += before
        event = {"event_id":f"fill_{len(fills)+1:06d}", "date":day, "phase":phase,
                 "status":"applied", "from_exposure":exposure,"target_exposure":target,
                 "adjusted_price":float(price),"reason":reason,"setup_id":setup_id,
                 "nav_before_cost":before,"nav_after_cost":nav,"cost_nav_units":fee}
        fills.append(event)
        exposure = target
        return event

    for i in indices:
        row = rows[i]
        day = row["datetime"].date().isoformat()
        opening, high, low, close = [float(row[k]) for k in ("open","high","low","close")]
        tradeable = _tradeable(row)
        before_nav, before_fees, old_exposure = nav, fees, exposure
        decision = {"date":day,"tradeable":tradeable,"opening_exposure":old_exposure,
                    "entry_day_stop_touch":False,"entry_day_target_touch":False,
                    "ambiguous_entry_stop_order":False,"ambiguous_entry_target_order":False,
                    "ambiguous_stop_target_order":False}
        if old_exposure:
            stop, target = position["initial_stop"], position["target_price"]
            price, phase, reason = None, None, None
            if tradeable:
                if pending_exit:
                    price, phase, reason = opening, "open", pending_exit
                elif opening <= stop:
                    price, phase, reason = opening, "open", "gap_stop"
                elif opening >= target:
                    price, phase, reason = opening, "open", "gap_target"
                elif low <= stop:
                    price, phase, reason = stop, "intraday", "initial_stop"
                    if high >= target:
                        decision["ambiguous_stop_target_order"] = True
                        position["ambiguous_stop_target_order"] = True
                elif high >= target:
                    price, phase, reason = target, "intraday", "target"
            elif low <= stop and pending_exit is None:
                pending_exit = "stop_on_non_tradeable_bar"
            prior_close = float(rows[i-1]["close"])
            if price is not None:
                nav *= price/prior_close
                event = fill(day,0,price,phase,reason,position["setup_id"])
                position.update(status="closed",exit_date=day,exit_phase=phase,exit_reason=reason,
                                adjusted_exit_price=float(price),exit_nav=nav,
                                net_return=nav/position["entry_nav_before_cost"]-1,
                                price_return=float(price)/position["adjusted_entry_price"]-1,
                                realized_price_r=(float(price)-position["adjusted_entry_price"])/position["initial_risk"],
                                holding_bars=i-position["entry_index"]+1,exit_event_id=event["event_id"])
                trades.append(position)
                position, pending_exit = None, None
            else:
                nav *= close/prior_close
        elif pending_order is not None:
            if i > pending_order["expires_index"]:
                pending_order.update(status="expired",resolved_date=day)
                pending_order = None
            elif tradeable:
                stop, limit = pending_order["stop"], pending_order["limit"]
                if opening <= stop:
                    pending_order.update(status="cancelled",reason="opening_quote_at_or_below_stop",resolved_date=day)
                    pending_order = None
                else:
                    price = opening if opening <= limit else limit if low <= limit else None
                    if price is not None:
                        risk = price-stop
                        if risk <= 0:
                            raise ValueError("Nonpositive entry risk")
                        phase = "open" if opening <= limit else "intraday"
                        entry_nav = nav
                        event = fill(day,1,price,phase,"fvg_midpoint",pending_order["setup_id"])
                        position = {"trade_id":f"trade_{len(trades)+1:06d}","status":"open",
                                    "setup_id":pending_order["setup_id"],"entry_date":day,"entry_index":i,
                                    "entry_phase":phase,"adjusted_entry_price":float(price),"initial_stop":stop,
                                    "initial_risk":risk,"target_price":float(price+target_r*risk),
                                    "entry_nav_before_cost":entry_nav,"entry_event_id":event["event_id"],
                                    "entry_day_stop_touch":False,"entry_day_target_touch":False,
                                    "ambiguous_entry_stop_order":False,"ambiguous_entry_target_order":False,
                                    "ambiguous_stop_target_order":False}
                        pending_order.update(status="filled",resolved_date=day,fill_price=float(price),trade_id=position["trade_id"])
                        pending_order = None
                        nav *= close/price
                        if low <= stop:
                            pending_exit = "entry_day_stop_t_plus_one"
                            decision["entry_day_stop_touch"] = position["entry_day_stop_touch"] = True
                            if phase == "intraday":
                                decision["ambiguous_entry_stop_order"] = position["ambiguous_entry_stop_order"] = True
                        if high >= position["target_price"]:
                            decision["entry_day_target_touch"] = position["entry_day_target_touch"] = True
                            if phase == "intraday":
                                decision["ambiguous_entry_target_order"] = position["ambiguous_entry_target_order"] = True
            elif low <= pending_order["stop"]:
                pending_order.update(status="cancelled",reason="non_tradeable_bar_at_or_below_stop",resolved_date=day)
                pending_order = None
        # Expiry occurs after the last allowed bar has had its fill opportunity.
        if pending_order is not None and i >= pending_order["expires_index"]:
            pending_order.update(status="expired",resolved_date=day)
            pending_order = None
        if position is not None and i-position["entry_index"]+1 >= SETTINGS["maximum_holding_bars"] and pending_exit is None:
            pending_exit = "time_exit_after_20_bars"
        if i in setup_by_index:
            setup = setup_by_index[i]
            order = {"order_id":f"order_{len(orders)+1:06d}","setup_id":setup["setup_id"],
                     "created_date":day,"available_at":setup["available_at"],
                     "limit":setup["limit"],"stop":setup["stop"],"first_active_index":i+1,
                     "expires_index":i+SETTINGS["entry_window_bars"],"status":"pending"}
            if exposure:
                order.update(status="ignored",reason="position_open")
            elif pending_order is not None:
                order.update(status="ignored",reason="older_order_active")
            else:
                pending_order = order
            orders.append(order)
        decision.update(closing_exposure=exposure,pending_exit_reason=pending_exit,
                        active_order_id=pending_order["order_id"] if pending_order else None)
        decisions.append(decision)
        equity.append({"date":day,"valuation_time":pd.Timestamp.combine(row["datetime"].date(),time(15)).isoformat(),
                       "nav":nav,"daily_return":nav/before_nav-1,"exposure":exposure,
                       "overnight_exposure":old_exposure,"adjusted_close":close,
                       "cost_nav_units":fees-before_fees,"cumulative_cost_nav_units":fees,
                       "initial_stop":position["initial_stop"] if position else None,
                       "target_price":position["target_price"] if position else None,
                       "pending_exit_reason":pending_exit})
    if position is not None:
        position.update(as_of=equity[-1]["date"],marked_nav=nav,
                        net_return=nav/position["entry_nav_before_cost"]-1,
                        price_return=rows[indices[-1]]["close"]/position["adjusted_entry_price"]-1,
                        holding_bars=indices[-1]-position["entry_index"]+1,pending_exit_reason=pending_exit)
        trades.append(position)
    if pending_order is not None:
        pending_order.update(status="pending_at_end",as_of=equity[-1]["date"])
    closed = [t for t in trades if t["status"] == "closed"]
    wins = [t["net_return"] for t in closed if t["net_return"] > TOLERANCE]
    losses = [t["net_return"] for t in closed if t["net_return"] < -TOLERANCE]
    flat = len(closed)-len(wins)-len(losses)
    metrics = portfolio_metrics([e["nav"] for e in equity],initial_nav=1.0,turnover_value=turnover,total_fees=fees)
    metrics["total_cost_nav_units"] = metrics.pop("total_fees")
    metrics.update(completed_trades=len(closed),open_trades=sum(t["status"]=="open" for t in trades),
                   entries=len(trades),winning_trades=len(wins),losing_trades=len(losses),flat_trades=flat,
                   winning_trade_ratio=len(wins)/len(closed) if closed else None,
                   average_win=mean(wins) if wins else None,average_loss=mean(losses) if losses else None,
                   payoff_ratio=mean(wins)/abs(mean(losses)) if wins and losses else None,
                   mean_closed_trade_return=mean(t["net_return"] for t in closed) if closed else None,
                   profit_factor=sum(wins)/abs(sum(losses)) if losses else None,
                   position_changes=len(fills),final_exposure=exposure,
                   invested_close_ratio=mean(e["exposure"] for e in equity),
                   entry_day_stop_count=sum(t["entry_day_stop_touch"] for t in trades),
                   entry_day_target_touch_count=sum(t["entry_day_target_touch"] for t in trades),
                   ambiguous_entry_stop_count=sum(t["ambiguous_entry_stop_order"] for t in trades),
                   ambiguous_entry_target_count=sum(t["ambiguous_entry_target_order"] for t in trades),
                   ambiguous_stop_target_count=sum(t["ambiguous_stop_target_order"] for t in trades))
    eval_sweeps = [s for s in sweeps if s["date"] >= start_date.date().isoformat()]
    diagnostics = {"sweeps":len(eval_sweeps),"confirmed_setups":len(setups),"orders":len(orders),
                   "accepted_orders":sum(o["status"] != "ignored" for o in orders),
                   "ignored_orders":sum(o["status"] == "ignored" for o in orders),
                   "order_status_counts":dict(Counter(o["status"] for o in orders)),
                   "closed_exit_reasons":dict(Counter(t["exit_reason"] for t in closed)),
                   "cash_only":not trades,"closed_win_tolerance":TOLERANCE,
                   "profit_factor_basis":"sum of positive closed-trade percentage returns divided by absolute sum of negative returns; not currency PnL"}
    return {"schema_version":1,"mode":"ict_daily","strategy":VERSION,
            "symbol":str(frame.iloc[0]["symbol"]),"price_basis":"pre",
            "return_basis":"forward_adjusted_long_cash_index",
            "requested_start":start_date.date().isoformat(),"requested_end":end_date.date().isoformat(),
            "start":equity[0]["date"],"end":equity[-1]["date"],"bars":len(equity),
            "settings":{**SETTINGS,"cost_bps":float(cost_bps),"target_r":float(target_r)},
            "metrics":metrics,"equity_curve":equity,"trades":trades,"fills":fills,
            "setups":setups,"orders":orders,"sweeps":eval_sweeps,"decisions":decisions,
            "diagnostics":diagnostics,"assumptions":ASSUMPTIONS.copy()}
