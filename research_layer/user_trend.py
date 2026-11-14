"""User-authored EMA/EFI/ATR trend rules with explicit daily path scenarios.

OHLC and OLHC are hypothetical intraday paths, not proven performance bounds.
The same-close mode is a closing-fill approximation, not an auction replay.
"""
from __future__ import annotations

from collections import Counter
from datetime import time
import math
from statistics import mean

import numpy as np
import pandas as pd

from evaluation_layer.metrics.portfolio import portfolio_metrics
from feature_layer.triple_screen import _local_dates, _normalize_calendar
from research_layer.technical import normalize_technical_bars


VERSION = "user_trend_ema_efi_atr_pre_v1"
TOLERANCE = 1e-12
ENTRY_COMBINATIONS = frozenset({"OR", "AND_EXPANDED", "AND_BAND"})
SETTINGS = {"version": VERSION, "ema_periods": [13,26], "efi_periods": [2,13],
            "activation_atr": 1.0, "trailing_drawdown_atr": .25, "hard_stop_atr": .5,
            "ema_exit_closes": 3, "ema_count_scope": "observed_closes_after_entry",
            "entry_combination": "OR", "atr_anchor": "entry_signal"}
ASSUMPTIONS = [
    "User-authored long/cash trend hypothesis on pre-adjusted daily bars; each stock has independent NAV1, not a shared cash or real-share portfolio.",
    "Completed weekly EMA13 and EMA26 must both rise for entry. Their relative order is unrestricted, and weekly weakening never forces an exit.",
    "Entry is the OR of a daily-low touch inside the EMA band and a one-ATR-expanded band touch with negative, twice-rising EFI2 and abs(today)/abs(yesterday)<0.5. EFI13 is informational only.",
    "ATR is Wilder-smoothed and fixed at the entry's known signal ATR. Hard stop is entry minus0.5ATR; trailing activates strictly above entry plus1ATR, then follows held peak minus0.25ATR without loosening.",
    "next_open uses a completed close signal only at the following trading-calendar bar's open; a non-tradeable next bar expires the candidate. No pre-start candidate is inherited.",
    "signal_close uses final close-time indicators and assumes a fill at that same final close. This closing-fill approximation does not reconstruct order submission, auction knowledge or execution priority.",
    "Intraday O-H-L-C and O-L-H-C are separate monotonic-segment scenarios. Other intraday oscillations are unknown; these scenarios are not guaranteed upper/lower bounds.",
    "Entry-day sales are prohibited under the declared CN T+1 model. The first price-stop hit latches to the next tradeable open, even after a rebound. Queued EMA exits also survive suspensions.",
    "The three-close EMA26 count starts after entry: open entries count that day's later close; same-close entries first count the following close. A third below-EMA close queues the next tradeable open.",
    "A same-close entry ignores all earlier high/low history of its entry bar. No same-day re-entry or exit-day candidate is allowed. There is no fixed target, time exit, weekly exit or averaging down.",
    "Non-tradeable rows do not fill orders. Their declared OHLC path may conservatively latch a stop; all supplied daily rows remain in indicator and EMA-count history.",
    "One-sided proportional NAV costs apply to each fill. Price-limit queues, auctions, lots, spreads and full taxes are not reconstructed. Cash has zero interest; final holdings remain marked, not force-sold.",
]


def build_user_trend_features(bars, calendar, *, atr_period=14):
    from feature_layer.user_trend import build_user_trend_features as build
    return build(bars, calendar, atr_period=atr_period)


def _date(value, label):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError(f"{label} must be a local date at midnight")
    return stamp


def _truth(value, label):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an explicit boolean")
    return bool(value)


def _tradeable(row):
    return _truth(row.get("is_tradeable", True), "is_tradeable") and float(row["volume"]) > 0


def _prepare(bars, calendar, end):
    source = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    if "datetime" not in source and "date" in source:
        source = source.rename(columns={"date": "datetime"})
    if "datetime" not in source:
        raise ValueError("Daily input requires datetime")
    source["datetime"] = _local_dates(source["datetime"], "Daily bars")
    # Censor prices before validation/indicators; later bars cannot affect a run.
    source = source.loc[source["datetime"] <= end].copy()
    source = normalize_technical_bars(source)
    if "volume" not in source:
        raise ValueError("User trend research requires explicit share volume")
    cal = _normalize_calendar(calendar, source.iloc[0]["datetime"], end)
    sessions = pd.DatetimeIndex(cal.loc[cal["date"].between(source.iloc[0]["datetime"], end) & cal["is_trading_day"], "date"])
    actual = pd.DatetimeIndex(source["datetime"])
    if len(actual.difference(sessions)) or len(sessions.difference(actual)):
        raise ValueError("Daily bars must match every open session; suspension must be explicit")
    return source, cal


def _finite_or_none(value):
    return float(value) if value is not None and pd.notna(value) and math.isfinite(float(value)) else None


def _observe(row, *, entry_combination="OR"):
    ready = _truth(row["indicators_ready"], "indicators_ready")
    names = ("daily_ema13", "daily_ema26", "atr", "efi2", "efi13", "efi2_previous", "efi2_previous2",
             "weekly_ema13", "weekly_ema26", "weekly_ema13_slope", "weekly_ema26_slope")
    out = {name: _finite_or_none(row.get(name)) for name in names}
    daily_time = pd.Timestamp(row["daily_available_at"])
    expected_time = pd.Timestamp.combine(pd.Timestamp(row["datetime"]).date(), time(15))
    if pd.isna(daily_time) or daily_time != expected_time:
        raise ValueError("Daily feature time must be its completed15:00 close")
    known = row.get("weekly_available_at")
    if ready:
        if any(out[name] is None for name in names):
            raise ValueError("Ready indicators must be finite")
        known = pd.Timestamp(known)
        if pd.isna(known) or known.tzinfo is not None or known > daily_time:
            raise ValueError("Weekly features are not yet available")
    out.update(indicators_ready=ready, daily_available_at=daily_time.isoformat(),
               weekly_available_at=None if pd.isna(known) else pd.Timestamp(known).isoformat(),
               weekly_completed_count=int(row["weekly_completed_count"]))
    low = float(row["low"])
    lower = min(out["daily_ema13"], out["daily_ema26"]) if out["daily_ema13"] is not None and out["daily_ema26"] is not None else None
    upper = max(out["daily_ema13"], out["daily_ema26"]) if lower is not None else None
    atr = out["atr"]
    ratio = abs(out["efi2"])/abs(out["efi2_previous"]) if out["efi2"] is not None and out["efi2_previous"] not in (None, 0) else None
    ratio = ratio if ratio is not None and math.isfinite(ratio) else None
    branch_a = ready and lower <= low <= upper
    efi_recovery = (ready and out["efi2"] < 0 and out["efi2"] > out["efi2_previous"] > out["efi2_previous2"]
                    and ratio is not None and ratio < .5)
    branch_b = ready and atr > 0 and lower-atr <= low <= upper and efi_recovery
    weekly_up = ready and out["weekly_ema13_slope"] > 0 and out["weekly_ema26_slope"] > 0
    entry_rule = {"OR": branch_a or branch_b,
                  "AND_EXPANDED": branch_b,
                  "AND_BAND": branch_a and branch_b}[entry_combination]
    out.update(band_lower=lower, band_upper=upper,
               atr_band_lower=lower-atr if lower is not None and atr is not None else None,
               efi_distance_ratio=ratio, efi_recovery=bool(efi_recovery),
               branch_a_band_touch=bool(branch_a), branch_b_atr_efi=bool(branch_b), weekly_up=bool(weekly_up),
               entry_ready=bool(ready and atr > 0 and weekly_up and _tradeable(row) and entry_rule))
    return out


def _risk_snapshot(position):
    return {key: position[key] for key in ("peak", "trail_active", "trailing_stop", "hard_stop", "active_stop")}


def _raise_peak(position, price):
    position["peak"] = max(position["peak"], float(price))
    if position["peak"] > position["activation_price"]:
        position["trail_active"] = True
    if position["trail_active"]:
        position["trailing_stop"] = position["peak"] - .25*position["atr_at_entry"]
    position["active_stop"] = max(position["hard_stop"], position["trailing_stop"] if position["trail_active"] else position["hard_stop"])


def _walk_path(position, row, *, path, can_sell, blocked_suffix, pending_exit):
    """Visit opening quote then monotonic legs; update peak only when observed."""
    day = row["datetime"].date().isoformat()
    order = ("open", "high", "low", "close") if path == "ohlc" else ("open", "low", "high", "close")
    trace = []
    points = [(name, float(row[name])) for name in order]

    def hit(price, phase, leg):
        kind = "trailing_stop" if position["trail_active"] else "hard_stop"
        event = {"trigger_date": day, "trigger_phase": phase, "path_leg": leg,
                 "trigger_price": float(price), "stop_kind": kind,
                 "active_stop_at_trigger": position["active_stop"], "peak_at_trigger": position["peak"]}
        trace.append({"event": "stop_touch", **event})
        if can_sell:
            return {"price": float(price), "phase": phase, "reason": "gap_"+kind if phase == "open" else kind, **event}
        position["blocked_price_stop_count"] += 1
        if blocked_suffix == "t_plus_one":
            position["entry_day_price_stop_touch"] = True
        return {"reason": kind+"_"+blocked_suffix, **event}

    opening = points[0][1]
    if pending_exit is None and opening <= position["active_stop"]:
        event = hit(opening, "open", "opening_gap")
        if can_sell:
            return event, pending_exit, trace
        pending_exit = event
    before = _risk_snapshot(position)
    _raise_peak(position, opening)
    trace.append({"event": "observe", "point": "open", "price": opening, "before": before, "after": _risk_snapshot(position)})
    previous_name, previous_price = points[0]
    for name, price in points[1:]:
        if pending_exit is None and price < previous_price and price <= position["active_stop"]:
            event = hit(position["active_stop"], "intraday", previous_name+"_to_"+name)
            if can_sell:
                return event, pending_exit, trace
            pending_exit = event
        before = _risk_snapshot(position)
        _raise_peak(position, price)
        trace.append({"event": "observe", "point": name, "price": price, "before": before, "after": _risk_snapshot(position)})
        previous_name, previous_price = name, price
    return None, pending_exit, trace


def run_user_trend(bars, calendar, *, start, end, entry_mode="next_open", intraday_path="ohlc", cost_bps=10, atr_period=14,
                   entry_combination="OR"):
    if not isinstance(entry_combination, str) or entry_combination not in ENTRY_COMBINATIONS:
        raise ValueError("entry_combination must be OR, AND_EXPANDED or AND_BAND")
    if entry_mode not in {"next_open", "signal_close"}:
        raise ValueError("entry_mode must be next_open or signal_close")
    if intraday_path not in {"ohlc", "olhc"}:
        raise ValueError("intraday_path must be ohlc or olhc")
    if isinstance(cost_bps, bool) or not isinstance(cost_bps, (int, float)) or not math.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be finite in [0,10000)")
    start_date, end_date = _date(start, "start"), _date(end, "end")
    if start_date > end_date:
        raise ValueError("start must not exceed end")
    source, cal = _prepare(bars, calendar, end_date)
    features = build_user_trend_features(source, cal, atr_period=atr_period)
    if len(features) != len(source) or not features["datetime"].reset_index(drop=True).equals(source["datetime"].reset_index(drop=True)):
        raise ValueError("Feature dates/order must preserve the normalized source")
    rows = features.to_dict("records")
    observations = [_observe(row, entry_combination=entry_combination) for row in rows]
    indices = [i for i,row in enumerate(rows) if start_date <= row["datetime"] <= end_date]
    if not indices:
        raise ValueError("No daily bars in evaluation range")
    nav, exposure, costs, turnover = 1.0, 0, 0.0, 0.0
    position, pending_exit, candidate = None, None, None
    fills, trades, signals, equity, decisions = [], [], [], [], []
    rate = float(cost_bps)/10000

    def fill(day, target, price, phase, reason, signal_id):
        nonlocal nav, exposure, costs, turnover
        before = nav
        fee = before*rate
        nav -= fee
        costs += fee
        turnover += before
        event = {"event_id": f"fill_{len(fills)+1:06d}", "date": day, "phase": phase, "status": "applied",
                 "from_exposure": exposure, "target_exposure": target, "adjusted_price": float(price),
                 "reason": reason, "signal_id": signal_id, "nav_before_cost": before,
                 "nav_after_cost": nav, "cost_nav_units": fee}
        fills.append(event)
        exposure = target
        return event

    def enter(signal, row, index, phase):
        nonlocal position
        day = row["datetime"].date().isoformat()
        price = float(row["open"] if phase == "open" else row["close"])
        atr = float(signal["atr"])
        before = nav
        event = fill(day, 1, price, phase, "user_trend_entry", signal["signal_id"])
        position = {"trade_id": f"trade_{len(trades)+1:06d}", "status": "open", "signal_id": signal["signal_id"],
                    "entry_signal_date": signal["date"], "entry_signal_available_at": signal["daily_available_at"],
                    "entry_date": day, "entry_index": index, "entry_phase": phase,
                    "adjusted_entry_price": price, "entry_nav_before_cost": before,
                    "entry_event_id": event["event_id"], "entry_reasons": signal["entry_reasons"].copy(),
                    "atr_at_entry": atr, "hard_stop": price-.5*atr, "initial_stop": price-.5*atr,
                    "activation_price": price+atr, "peak": price, "trail_active": False,
                    "trailing_stop": None, "active_stop": price-.5*atr, "below_ema26_count": 0,
                    "entry_day_price_stop_touch": False, "blocked_price_stop_count": 0}
        signal.update(status="filled", fill_date=day, fill_phase=phase, fill_price=price, trade_id=position["trade_id"])

    def exit_position(row, index, event, reference_price):
        nonlocal nav, position, pending_exit
        day = row["datetime"].date().isoformat()
        price = float(event["price"])
        nav *= price/reference_price
        applied = fill(day, 0, price, event["phase"], event["reason"], position["signal_id"])
        position.update(status="closed", exit_date=day, exit_phase=event["phase"], exit_reason=event["reason"],
                        adjusted_exit_price=price, exit_nav=nav, net_return=nav/position["entry_nav_before_cost"]-1,
                        price_return=price/position["adjusted_entry_price"]-1,
                        holding_bars=index-position["entry_index"]+1, exit_event_id=applied["event_id"],
                        exit_trigger={key:value for key,value in event.items() if key not in {"price","phase"}})
        trades.append(position)
        position, pending_exit = None, None

    for i in indices:
        row, observed = rows[i], observations[i]
        day = row["datetime"].date().isoformat()
        opening, close = float(row["open"]), float(row["close"])
        tradeable = _tradeable(row)
        before_nav, before_costs, old_exposure = nav, costs, exposure
        exited_today, entered_open = False, False
        trace = []
        decision = {"date": day, **observed, "opening_exposure": old_exposure, "tradeable": tradeable,
                    "intraday_path": intraday_path, "risk_before": _risk_snapshot(position) if position else None}
        if old_exposure:
            previous_close = float(rows[i-1]["close"])
            if pending_exit is not None and tradeable:
                _raise_peak(position, opening)
                event = {**pending_exit, "price": opening, "phase": "open"}
                trace.append({"event": "execute_pending", **event})
                exit_position(row, i, event, previous_close)
                exited_today = True
            else:
                event, pending_exit, trace = _walk_path(position, row, path=intraday_path, can_sell=tradeable,
                                                       blocked_suffix="non_tradeable", pending_exit=pending_exit)
                if event is not None:
                    exit_position(row, i, event, previous_close)
                    exited_today = True
                else:
                    nav *= close/previous_close
        elif candidate is not None:
            if i != candidate["active_index"]:
                raise ValueError("Opening candidate escaped its single next-bar window")
            if tradeable:
                enter(candidate, row, i, "open")
                entered_open = True
                _, pending_exit, trace = _walk_path(position, row, path=intraday_path, can_sell=False,
                                                    blocked_suffix="t_plus_one", pending_exit=None)
                nav *= close/position["adjusted_entry_price"]
            else:
                candidate.update(status="expired", resolved_date=day, reason="next_bar_non_tradeable")
            candidate = None
        # Existing/open-entry holdings observe today's close; a new close-entry does not.
        if position is not None and (old_exposure or entered_open):
            position["below_ema26_count"] = position["below_ema26_count"]+1 if close < observed["daily_ema26"] else 0
            if position["below_ema26_count"] >= 3 and pending_exit is None:
                pending_exit = {"reason": "three_closes_below_ema26", "trigger_date": day,
                                "trigger_phase": "close", "trigger_price": close, "ema26_at_trigger": observed["daily_ema26"]}
                trace.append({"event": "queue_ema_exit", **pending_exit})
        if observed["entry_ready"]:
            reasons = [label for label,key in (("ema_band_pullback","branch_a_band_touch"),("one_atr_efi_recovery","branch_b_atr_efi")) if observed[key]]
            signal = {"signal_id": f"signal_{len(signals)+1:06d}", "date": day, "index": i,
                      **observed, "entry_reasons": reasons, "status": "pending"}
            if exited_today:
                signal.update(status="ignored", reason="exit_day")
            elif exposure:
                signal.update(status="ignored", reason="position_open")
            elif entry_mode == "signal_close":
                enter(signal, row, i, "close")
            else:
                signal["active_index"] = i+1
                candidate = signal
            signals.append(signal)
        decision.update(closing_exposure=exposure, risk_after=_risk_snapshot(position) if position else None,
                        below_ema26_count=position["below_ema26_count"] if position else None,
                        pending_exit_reason=pending_exit["reason"] if pending_exit else None,
                        pending_exit=dict(pending_exit) if pending_exit else None, risk_trace=trace)
        decisions.append(decision)
        equity.append({"date": day, "valuation_time": observed["daily_available_at"], "nav": nav,
                       "daily_return": nav/before_nav-1, "exposure": exposure, "overnight_exposure": old_exposure,
                       "adjusted_close": close, "cost_nav_units": costs-before_costs, "cumulative_cost_nav_units": costs,
                       "active_stop": position["active_stop"] if position else None,
                       "peak": position["peak"] if position else None,
                       "trail_active": position["trail_active"] if position else False,
                       "pending_exit_reason": pending_exit["reason"] if pending_exit else None})
    if position is not None:
        position.update(as_of=equity[-1]["date"], marked_nav=nav, net_return=nav/position["entry_nav_before_cost"]-1,
                        price_return=float(rows[indices[-1]]["close"])/position["adjusted_entry_price"]-1,
                        holding_bars=indices[-1]-position["entry_index"]+1,
                        pending_exit_reason=pending_exit["reason"] if pending_exit else None)
        trades.append(position)
    if candidate is not None:
        candidate.update(status="pending_at_end", as_of=equity[-1]["date"])
    closed = [t for t in trades if t["status"] == "closed"]
    wins = [t["net_return"] for t in closed if t["net_return"] > TOLERANCE]
    losses = [t["net_return"] for t in closed if t["net_return"] < -TOLERANCE]
    metrics = portfolio_metrics([e["nav"] for e in equity], initial_nav=1.0, turnover_value=turnover, total_fees=costs)
    metrics["total_cost_nav_units"] = metrics.pop("total_fees")
    metrics.update(entries=len(trades), completed_trades=len(closed), open_trades=len(trades)-len(closed),
                   winning_trades=len(wins), losing_trades=len(losses), flat_trades=len(closed)-len(wins)-len(losses),
                   winning_trade_ratio=len(wins)/len(closed) if closed else None,
                   average_win=mean(wins) if wins else None, average_loss=mean(losses) if losses else None,
                   payoff_ratio=mean(wins)/abs(mean(losses)) if wins and losses else None,
                   mean_closed_trade_return=mean(t["net_return"] for t in closed) if closed else None,
                   final_exposure=exposure, position_changes=len(fills),
                   invested_close_ratio=mean(e["exposure"] for e in equity),
                   trailing_activated_trades=sum(t["trail_active"] for t in trades),
                   entry_day_price_stop_count=sum(t["entry_day_price_stop_touch"] for t in trades))
    settings = {**SETTINGS, "entry_mode": entry_mode, "intraday_path": intraday_path,
                "cost_bps": float(cost_bps), "atr_period": int(atr_period),
                "entry_combination": entry_combination,
                "ema_periods": SETTINGS["ema_periods"].copy(), "efi_periods": SETTINGS["efi_periods"].copy()}
    diagnostics = {"signals": len(signals), "accepted_signals": sum(s["status"] != "ignored" for s in signals),
                   "signal_status_counts": dict(Counter(s["status"] for s in signals)),
                   "ignored_reasons": dict(Counter(s["reason"] for s in signals if s["status"] == "ignored")),
                   "closed_exit_reasons": dict(Counter(t["exit_reason"] for t in closed)),
                   "cash_only": not trades, "path_scope": "scenario, not upper/lower bound", "win_tolerance": TOLERANCE}
    assumptions = ASSUMPTIONS.copy()
    if entry_combination == "AND_EXPANDED":
        assumptions[2] = "Entry requires the daily low inside the EMA band or no more than one ATR below its lower edge, AND negative, twice-rising EFI2 with abs(today)/abs(yesterday)<0.5. This AND_EXPANDED policy requires EFI on every entry and preserves the expanded price range. EFI13 is informational only."
    elif entry_combination == "AND_BAND":
        assumptions[2] = "Entry requires the daily low inside the EMA13/EMA26 band, AND negative, twice-rising EFI2 with abs(today)/abs(yesterday)<0.5. This AND_BAND policy excludes prices below the EMA band, including the one-ATR extension. EFI13 is informational only."
    return {"schema_version": 1, "mode": "user_trend", "strategy": VERSION, "symbol": str(source.iloc[0]["symbol"]),
            "price_basis": "pre", "return_basis": "forward_adjusted_long_cash_index",
            "requested_start": start_date.date().isoformat(), "requested_end": end_date.date().isoformat(),
            "start": equity[0]["date"], "end": equity[-1]["date"], "bars": len(equity),
            "settings": settings, "metrics": metrics, "equity_curve": equity, "trades": trades, "fills": fills,
            "signals": signals, "decisions": decisions, "diagnostics": diagnostics, "assumptions": assumptions}
