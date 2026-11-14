"""Causal long/cash Triple Screen research on forward-adjusted daily bars."""
from __future__ import annotations

import math
from datetime import datetime, time
from typing import Any

import numpy as np
import pandas as pd

from evaluation_layer.metrics.portfolio import portfolio_metrics
from research_layer.technical import normalize_technical_bars
from strategy_layer.triple_screen import resolve_triple_settings


RESEARCH_VARIANTS = {
    "full": (),
    "drop_weekly": ("weekly_direction_and_exit",),
    "drop_weekly_entry": ("weekly_direction",),
    "drop_weekly_exit": ("weekly_exit",),
    "weekly_exit_loss_only": ("weekly_exit_when_price_profitable",),
    "drop_pullback": ("daily_force_pullback",),
    "drop_trigger": ("prior_high_breakout",),
}


ASSUMPTIONS = [
    "Forward-adjusted, single-symbol technical return index; initial NAV is 1.0, exposure is long or cash, with no real share ledger.",
    "Only the previous completed daily bar and completed weekly features can authorize today's candidate; current high/low only determine trigger events.",
    "A candidate expires at the end of its next trading session; another candidate requires another qualifying daily close.",
    "A strict break above the prior high enters at that high, unless the open already exceeds it, in which case entry is at the open.",
    "Initial stop is the lower of the setup-day low and preceding daily low, fixed for the position; weekly-filter invalidation exits at the next tradable open.",
    "T+1 forbids an exit on the entry date. An entry-day stop touch is conservatively treated as after entry, marked ambiguous, and forces exit at the next tradable open even after a rebound.",
    "Pending exits persist through zero-volume or explicitly non-tradeable bars; no same-day re-entry follows an exit.",
    "Intraday OHLC triggers have a date and phase only; their exact intraday timestamp and path are unknown.",
    "Cost bps is a one-sided proportional NAV friction at every applied exposure change; it is not a reconstruction of taxes, lot sizes, liquidity or limit-lock execution.",
    "Cash has zero interest; final long exposure remains marked at the last adjusted close and is not forcibly liquidated.",
]


def build_triple_features(bars, calendar, settings):
    # Deferred import keeps the implementation boundary explicit and testable.
    from feature_layer.triple_screen import build_triple_features as build
    return build(bars, calendar, settings)


def _date(value: Any, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise ValueError(f"{name} must be a local date without a time or timezone.")
    return parsed


def _truth(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Feature {name} must be an explicit boolean.")
    return bool(value)


def _feature_values(row: dict) -> dict:
    ready = _truth(row["indicators_ready"], "indicators_ready")
    weekly_up = _truth(row["weekly_up"], "weekly_up")
    entry_ready = _truth(row["entry_ready"], "entry_ready")
    out = {"indicators_ready": ready, "weekly_up": weekly_up, "entry_ready": entry_ready}
    for field in ("force_index_2", "weekly_hist", "weekly_hist_previous"):
        raw = row[field]
        value = float(raw) if pd.notna(raw) else None
        if ready and (value is None or not math.isfinite(value)):
            raise ValueError(f"Ready feature {field} is missing or nonfinite.")
        out[field] = value if value is not None and math.isfinite(value) else None
    known = row["weekly_available_at"]
    if ready:
        known = pd.Timestamp(known)
        cutoff = pd.Timestamp(row.get("session_close", datetime.combine(pd.Timestamp(row["datetime"]).date(), time(15))))
        if pd.isna(known) or known.tzinfo is not None or known > cutoff:
            raise ValueError("Weekly features are unavailable at the decision close.")
        out["weekly_available_at"] = known.isoformat()
        if weekly_up != (out["weekly_hist"] > out["weekly_hist_previous"]):
            raise ValueError("weekly_up is inconsistent with raw MACD histogram slope.")
        if entry_ready and (not weekly_up or out["force_index_2"] >= 0):
            raise ValueError("entry_ready is inconsistent with the first two filters.")
    else:
        if weekly_up or entry_ready:
            # A not-yet-warmed histogram may have an up slope, but entry_ready
            # must still be false. weekly_up itself may be an informational flag.
            if entry_ready:
                raise ValueError("An unready feature row cannot authorize entry.")
        out["weekly_available_at"] = None if pd.isna(known) else pd.Timestamp(known).isoformat()
    out["weekly_completed_count"] = int(row["weekly_completed_count"])
    return out


def _tradeable(row: dict) -> bool:
    flag = row.get("is_tradeable", True)
    return _truth(flag, "is_tradeable") and float(row["volume"]) > 0


def run_triple_screen(bars, calendar, settings=None, *, start=None, end=None, variant="full") -> dict:
    """Evaluate v1 or an explicitly labelled exploratory rule variation.

    All variants retain identical feature calculation, readiness, market-specific
    entry-day stops, costs and initial-stop accounting. Disabled rules never
    overwrite observed features.
    """
    if not isinstance(variant, str) or variant not in RESEARCH_VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(RESEARCH_VARIANTS)}.")
    disabled_filters = list(RESEARCH_VARIANTS[variant])
    resolved = resolve_triple_settings(settings)
    requested_start, requested_end = _date(start, "start"), _date(end, "end")
    if requested_start is not None and requested_end is not None and requested_start > requested_end:
        raise ValueError("start must not be after end.")
    source = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    if "datetime" not in source and "date" in source:
        source = source.rename(columns={"date": "datetime"})
    if requested_end is not None and "datetime" in source:
        dates = pd.to_datetime(source["datetime"], errors="raise")
        if dates.isna().any() or dates.dt.tz is not None:
            raise ValueError("Every input bar needs a valid local date.")
        source = source.loc[dates <= requested_end].copy()
    source = normalize_technical_bars(source)
    features = build_triple_features(source, calendar, resolved)
    source_dates = source["datetime"].astype("datetime64[ns]").reset_index(drop=True)
    feature_dates = features["datetime"].astype("datetime64[ns]").reset_index(drop=True)
    if len(features) != len(source) or not feature_dates.equals(source_dates):
        raise ValueError("Feature rows must preserve normalized daily dates and order.")
    rows = features.to_dict("records")
    annotations = [_feature_values(row) for row in rows]
    eligible = [index for index in range(2, len(rows))
                if (requested_start is None or rows[index]["datetime"] >= requested_start)
                and annotations[index - 1]["indicators_ready"]]
    if not eligible:
        raise ValueError("No evaluable bars after weekly warmup and requested start.")
    first = eligible[0]
    nav, exposure = 1.0, 0
    costs, turnover = 0.0, 0.0
    stop = None
    pending_exit = None
    open_trade = None
    equity, fills, signals, trades, decisions = [], [], [], [], []
    cost_rate = float(resolved.cost_bps) / 10000.0

    def switch(day, target, price, phase, reason, *, signal_id=None):
        nonlocal costs, turnover, nav, exposure
        before = nav
        fee = before * cost_rate
        turnover += before
        costs += fee
        nav -= fee
        event = {"event_id": f"change_{len(fills)+1:06d}", "event_type": "technical_position_change",
                 "date": day, "phase": phase, "status": "applied", "from_exposure": exposure,
                 "target_exposure": target, "adjusted_price": price, "reason": reason,
                 "signal_id": signal_id, "cost_nav_units": fee,
                 "nav_before_cost": before, "nav_after_cost": nav}
        fills.append(event)
        exposure = target
        return event

    for index in range(first, len(rows)):
        row, previous, before_previous = rows[index], rows[index-1], rows[index-2]
        observed = annotations[index-1]
        if not observed["indicators_ready"]:
            raise ValueError("Indicators became unavailable after evaluation began.")
        day = row["datetime"].date().isoformat()
        decision_time = pd.Timestamp(previous.get("session_close", datetime.combine(previous["datetime"].date(), time(15)))).isoformat()
        opening, high, low, close = (float(row[name]) for name in ("open", "high", "low", "close"))
        last_close = float(previous["close"])
        old_exposure = exposure
        before_nav = nav
        today_cost_before = costs
        tradeable = _tradeable(row)
        previous_tradeable = _tradeable(previous)
        if variant in {"drop_weekly", "drop_weekly_entry"}:
            entry_rule_passed = observed["force_index_2"] < 0 and previous_tradeable
        elif variant == "drop_pullback":
            entry_rule_passed = observed["weekly_up"] and previous_tradeable
        else:
            # Preserve the original entry gate exactly for the full baseline
            # and when only the weekly exit or breakout trigger is removed.
            entry_rule_passed = observed["entry_ready"] and previous_tradeable
        day_decision = {"date": day, "decision_time": decision_time, **observed,
                        "research_variant": variant, "disabled_filters": disabled_filters,
                        "entry_rule_passed": entry_rule_passed,
                        "entry_line": float(previous["high"]),
                        "candidate_stop": min(float(previous["low"]), float(before_previous["low"])),
                        "opening_exposure": old_exposure, "tradeable": tradeable,
                        "ambiguous_intraday_order": False}

        if old_exposure:
            if variant not in {"drop_weekly", "drop_weekly_exit"} and not observed["weekly_up"] and pending_exit is None:
                # The conditional rule uses only the completed prior close.
                # A pending exit is already latched and never reconsidered here.
                if variant != "weekly_exit_loss_only" or last_close <= float(open_trade["adjusted_entry_price"]):
                    pending_exit = "weekly_filter_invalidated"
            exit_price, phase, reason = None, None, None
            if tradeable:
                if pending_exit is not None:
                    exit_price, phase, reason = opening, "open", pending_exit
                elif opening <= stop:
                    exit_price, phase, reason = opening, "open", "gap_stop"
                elif low <= stop:
                    exit_price, phase, reason = stop, "intraday", "initial_stop"
            elif pending_exit is not None or low <= stop:
                if pending_exit is None:
                    pending_exit = "stop_on_non_tradeable_bar"
                fills.append({"event_id": f"change_{len(fills)+1:06d}",
                              "event_type": "technical_position_change", "date": day, "phase": "unavailable",
                              "status": "blocked", "from_exposure": 1, "target_exposure": 0,
                              "adjusted_price": None, "reason": "non_tradeable_bar",
                              "pending_exit_reason": pending_exit, "cost_nav_units": 0.0})
            if exit_price is not None:
                # The old holding earns only through its exit, never the rest
                # of today's close-to-close move.
                nav *= exit_price / last_close
                event = switch(day, 0, exit_price, phase, reason)
                open_trade.update(status="closed", exit_date=day, exit_phase=phase,
                                  adjusted_exit_price=exit_price, exit_reason=reason,
                                  exit_nav=nav, net_return=nav/open_trade["entry_nav_before_cost"]-1,
                                  holding_bars=index-open_trade["entry_index"]+1,
                                  exit_event_id=event["event_id"])
                trades.append(open_trade)
                open_trade, stop, pending_exit = None, None, None
            else:
                nav *= close / last_close
        elif entry_rule_passed:
            trigger = float(previous["high"])
            candidate_stop = min(float(previous["low"]), float(before_previous["low"]))
            signal = {"signal_id": f"setup_{len(signals)+1:06d}", "decision_time": decision_time,
                      "valid_date": day, "entry_line": trigger, "initial_stop": candidate_stop,
                      "research_variant": variant, "disabled_filters": disabled_filters,
                      "entry_rule_passed": entry_rule_passed,
                      "trigger_required": variant != "drop_trigger",
                      "observed_breakout": high > trigger,
                      **observed, "status": "expired_no_breakout"}
            signals.append(signal)
            candidate_entry = opening if variant == "drop_trigger" else trigger
            if candidate_stop >= candidate_entry:
                # Flat/one-price candles are valid market observations. Their
                # zero-distance candidate is not a valid setup, but does not
                # invalidate all earlier/later research on this stock.
                signal.update(status="rejected_nonpositive_stop_distance",
                              rejection_reason="Initial stop must be below the entry price." if variant == "drop_trigger" else "Initial stop must be below the entry line.")
                day_decision["candidate_rejection_reason"] = "rejected_nonpositive_stop_distance"
            elif not tradeable:
                signal["status"] = "expired_non_tradeable"
            elif variant == "drop_trigger" or high > trigger:
                entry_price = opening if variant == "drop_trigger" or opening > trigger else trigger
                phase = "open" if variant == "drop_trigger" or opening > trigger else "intraday"
                entry_nav = nav
                reason = "next_open_without_breakout_filter" if variant == "drop_trigger" else "prior_high_breakout"
                event = switch(day, 1, entry_price, phase, reason, signal_id=signal["signal_id"])
                stop = candidate_stop
                signal.update(status="entered", event_id=event["event_id"])
                open_trade = {"trade_id": f"trade_{len(trades)+1:06d}", "status": "open",
                              "signal_id": signal["signal_id"], "entry_date": day, "entry_phase": phase,
                              "entry_index": index, "adjusted_entry_price": entry_price,
                              "initial_stop": stop, "entry_nav_before_cost": entry_nav,
                              "entry_event_id": event["event_id"], "ambiguous_intraday_order": False}
                nav *= close / entry_price
                if low <= stop:
                    # Daily OHLC cannot prove path order; choose the declared
                    # conservative T+1 outcome, persist it through suspensions.
                    ambiguous = not resolved.is_us or phase == "intraday"
                    signal["ambiguous_intraday_order"] = ambiguous
                    open_trade["ambiguous_intraday_order"] = ambiguous
                    day_decision["ambiguous_intraday_order"] = ambiguous
                    event["ambiguous_intraday_order"] = ambiguous
                    if resolved.is_us:
                        # US research permits a same-session exit. For an
                        # intraday breakout, assume the adverse stop touch came
                        # after entry; daily OHLC cannot establish the order.
                        nav *= stop / close
                        exit_event = switch(day, 0, stop, "intraday", "entry_day_stop_conservative")
                        open_trade.update(status="closed", exit_date=day, exit_phase="intraday",
                                          adjusted_exit_price=stop, exit_reason="entry_day_stop_conservative",
                                          exit_nav=nav, net_return=nav/open_trade["entry_nav_before_cost"]-1,
                                          holding_bars=1, exit_event_id=exit_event["event_id"])
                        trades.append(open_trade)
                        open_trade, stop, pending_exit = None, None, None
                    else:
                        pending_exit = "entry_day_stop_t_plus_one"
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError("Technical NAV became nonpositive or nonfinite.")
        day_decision.update(closing_exposure=exposure, pending_exit_reason=pending_exit)
        decisions.append(day_decision)
        equity.append({"date": day, "valuation_time": pd.Timestamp(row.get("session_close", f"{day}T15:00:00")).isoformat(), "nav": nav,
                       "daily_return": nav/before_nav-1, "exposure": exposure,
                       "overnight_exposure": old_exposure, "adjusted_close": close,
                       "cost_nav_units": costs-today_cost_before, "cumulative_cost_nav_units": costs,
                       "initial_stop": stop, "pending_exit_reason": pending_exit,
                       "ambiguous_intraday_order": day_decision["ambiguous_intraday_order"]})
    if open_trade is not None:
        open_trade.update(as_of=equity[-1]["date"], marked_nav=nav,
                          net_return=nav/open_trade["entry_nav_before_cost"]-1,
                          holding_bars=len(rows)-open_trade["entry_index"], pending_exit_reason=pending_exit)
        trades.append(open_trade)
    metrics = portfolio_metrics([item["nav"] for item in equity], initial_nav=1.0,
                                turnover_value=turnover, total_fees=costs)
    metrics["total_cost_nav_units"] = metrics.pop("total_fees")
    closed = [trade for trade in trades if trade["status"] == "closed"]
    metrics.update(position_changes=sum(fill["status"] == "applied" for fill in fills),
                   blocked_changes=sum(fill["status"] == "blocked" for fill in fills),
                   completed_trades=len(closed), open_trades=int(open_trade is not None),
                   final_exposure=exposure, winning_trade_ratio=sum(trade["net_return"] > 0 for trade in closed)/len(closed) if closed else None,
                   ambiguous_entry_count=sum(trade["ambiguous_intraday_order"] for trade in trades),
                   invested_close_ratio=sum(item["exposure"] for item in equity)/len(equity))
    return {"schema_version": 1, "mode": "triple_screen", "strategy": "triple_screen",
            "research_variant": variant, "disabled_filters": disabled_filters,
            "symbol": str(source.iloc[0]["symbol"]), "price_basis": "pre",
            "return_basis": "forward_adjusted_long_cash_index", "cost_units": "normalized_nav",
            "settings": resolved.to_dict(), "parameters": resolved.to_dict(),
            "requested_start": requested_start.date().isoformat() if requested_start is not None else None,
            "requested_end": requested_end.date().isoformat() if requested_end is not None else None,
            "start": equity[0]["date"], "end": equity[-1]["date"], "bars": len(equity),
            "warmup": {"bars": first, "completed_weeks": annotations[first-1]["weekly_completed_count"],
                       "excluded_from_performance": True},
            "metrics": metrics, "equity_curve": equity, "fills": fills,
            "signals": signals, "trades": trades, "decisions": decisions,
            "assumptions": _variant_assumptions(variant, is_us=resolved.is_us),
            **({"market": "US", "timezone": "America/New_York", "price_adjustment": "yahoo_split_dividend_adjusted_ohlc"} if resolved.is_us else {})}


def _variant_assumptions(variant: str, *, is_us=False) -> list[str]:
    assumptions = list(ASSUMPTIONS)
    if is_us:
        assumptions[5] = "US version permits entry-day stops; an intraday entry/stop ambiguity assumes entry first and exit at the stop on that same session. No A-share T+1 sell lock is applied."
        assumptions.append("US technical NAV only: brokerage cash settlement, margin requirements and account trading restrictions are not simulated. Yahoo vendor OHLC is adjusted by AdjClose/Close; volume remains provider-reported shares.")
    if variant == "drop_weekly":
        assumptions[4] = "Initial stop remains fixed; weekly direction neither filters new entries nor forces exits in drop_weekly."
        assumptions.append("drop_weekly removes both the weekly entry filter and weekly-invalidated exit, while preserving the same completed-week warmup and observed indicator fields.")
    elif variant == "drop_weekly_entry":
        assumptions.append("drop_weekly_entry removes only the weekly entry filter; negative daily EFI, prior-high breakout, the fixed initial stop and weekly-invalidated exit remain active. Completed-week warmup and observed indicator fields are unchanged.")
    elif variant == "drop_weekly_exit":
        assumptions[4] = "Initial stop remains fixed; weekly-filter invalidation does not force an exit in drop_weekly_exit."
        assumptions.append("drop_weekly_exit removes only the weekly-invalidated exit; the original weekly direction and negative daily EFI entry filters and prior-high breakout remain active. Completed-week warmup and observed indicator fields are unchanged.")
    elif variant == "weekly_exit_loss_only":
        assumptions[4] = "Initial stop remains fixed; while weekly direction is invalid, a new weekly exit is latched only when the completed previous daily close is at or below this trade's adjusted entry price, then executes at the next tradable open."
        assumptions.append("weekly_exit_loss_only retains full entry rules and checks the condition each day, without requiring a newly completed week. The comparison is price profit before costs, includes equality, and never uses today's open/high/low/close. Any pending weekly, T+1 or stop exit remains latched through rebounds and non-tradeable bars.")
    elif variant == "drop_pullback":
        assumptions.append("drop_pullback removes only the negative daily EFI requirement; the weekly direction filter, breakout trigger and weekly-invalidated exit remain active.")
    elif variant == "drop_trigger":
        assumptions[3] = "drop_trigger enters at the next tradable candidate session's open without requiring a prior-high breakout; a nonpositive stop distance at that open rejects the candidate."
        assumptions.append("The candidate still expires after one session and requires the original weekly direction and negative daily EFI; it is not held indefinitely through a suspension.")
    if variant == "weekly_exit_loss_only":
        assumptions.append("This post-hoc exploratory hypothesis was motivated by already-inspected event outcomes; it is not an untouched validation result or a replacement for frozen full v1.")
    elif variant != "full":
        assumptions.append("This labelled filter ablation is an exploratory comparison, not a replacement for the frozen full v1 strategy.")
    return assumptions
