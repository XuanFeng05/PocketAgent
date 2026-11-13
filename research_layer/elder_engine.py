"""Independent, explicitly mechanized Elder long-only research engine.

This is not a claim that Elder specified one fully mechanical system. Figure
43-3 of the 1993 book motivates prior-bar initial stops; ADR20 activation and
next-session updates are declared daily-data approximations. Existing project
rules and market-specific application flows are deliberately untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from research_layer.technical import normalize_technical_bars


@dataclass(frozen=True)
class ElderConfig:
    weekly_filter: str = "hist"
    force_span: int = 2
    exit_mode: str = "position"
    stop_mode: str = "protect50"
    risk_fraction: float = .01
    cost_bps: float = 10.0
    warmup_weeks: int = 100
    entry_buffer_adjusted: float = .01
    version: str = "elder_daily_mechanized_us_v1"

    def __post_init__(self):
        choices = {"weekly_filter": {"hist", "ema", "impulse"},
                   "exit_mode": {"position", "swing", "either"},
                   "stop_mode": {"fixed", "protect50", "protect70", "atr3"}}
        for field, allowed in choices.items():
            if getattr(self, field) not in allowed:
                raise ValueError(f"{field} must be one of {sorted(allowed)}")
        if self.version != "elder_daily_mechanized_us_v1":
            raise ValueError("Unsupported rule version")
        for field in ("force_span", "warmup_weeks"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"{field} must be an integer >=2")
        for field, lo, hi in [("risk_fraction", 0, 1), ("cost_bps", -1, 10000),
                              ("entry_buffer_adjusted", 0, float("inf"))]:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite numeric")
            if not lo < value <= hi or (field == "cost_bps" and not 0 <= value < 10000):
                raise ValueError(f"Invalid {field}")

    def to_dict(self):
        return asdict(self)


def _local_dates(values, name):
    dates = pd.to_datetime(values, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError(f"{name} must contain naive local midnight dates")
    return dates.astype("datetime64[ns]")


def _flag(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError("Explicit calendar trading flags are required")


def build_features(bars: Any, calendar: Any, force_span: int = 2,
                   warmup_weeks: int = 100) -> pd.DataFrame:
    """Build completed-week features available at each NY local session close.

    Calendar must contain every natural date and a naive local session_close
    for each open date, including early closes. Boundary weeks are discarded.
    Derived columns are rebuilt. Input adjustment_factor is metadata, not a
    purported reconstruction of historical nominal-price ticks.
    """
    config = ElderConfig(force_span=force_span, warmup_weeks=warmup_weeks)
    original = pd.DataFrame(bars).copy(deep=True)
    date_key = "datetime" if "datetime" in original else "date"
    original[date_key] = _local_dates(original[date_key], "Bars")
    frame = normalize_technical_bars(original)
    if "volume" not in frame:
        raise ValueError("Explicit share volume required")
    cal = pd.DataFrame(calendar).copy(deep=True)
    if not {"date", "is_trading_day", "session_close"}.issubset(cal):
        raise ValueError("Calendar needs date, is_trading_day and NY local session_close")
    cal["date"] = _local_dates(cal["date"], "Calendar")
    if cal["date"].duplicated().any():
        raise ValueError("Duplicate calendar dates")
    cal = cal.sort_values("date").reset_index(drop=True)
    if not pd.DatetimeIndex(cal["date"]).equals(pd.date_range(cal.date.iloc[0], cal.date.iloc[-1], freq="D")):
        raise ValueError("Calendar must cover every natural date")
    if cal.date.iloc[0] > frame.datetime.iloc[0] or cal.date.iloc[-1] < frame.datetime.iloc[-1]:
        raise ValueError("Calendar does not cover bars")
    cal["is_trading_day"] = cal["is_trading_day"].map(_flag)
    cal["session_close"] = pd.to_datetime(cal["session_close"], errors="coerce")
    if cal.session_close.dt.tz is not None:
        raise ValueError("session_close must be NY local naive time")
    sessions = cal.loc[cal.is_trading_day]
    if sessions.session_close.isna().any() or not sessions.session_close.dt.normalize().eq(sessions.date).all():
        raise ValueError("Every open date needs its own NY local session close")
    actual = pd.DatetimeIndex(frame.datetime)
    expected = pd.DatetimeIndex(sessions.date)
    if len(actual.difference(expected)):
        raise ValueError("Bars occur on closed dates")
    in_range = expected[(expected >= actual[0]) & (expected <= actual[-1])]
    missing = in_range.difference(actual)
    if len(missing):
        raise ValueError(f"Missing open-session bar: {missing[0].date()}; do not infer suspension")
    # Preserve a documented zero-volume observation as untradeable.
    if "is_tradeable" in frame:
        tradeable = frame.is_tradeable.map(_flag) & frame.volume.gt(0)
    else:
        tradeable = frame.volume.gt(0)
    frame["tradeable"] = tradeable
    close_by_date = sessions.set_index("date").session_close
    frame["session_close"] = frame.datetime.map(close_by_date)
    force = frame.close.diff().fillna(0) * frame.volume
    frame["force"] = force.ewm(span=config.force_span, adjust=False).mean()
    frame["adr20"] = (frame.high - frame.low).rolling(20, min_periods=20).mean()
    tr = pd.concat([frame.high-frame.low, (frame.high-frame.close.shift()).abs(),
                    (frame.low-frame.close.shift()).abs()], axis=1).max(axis=1)
    frame["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    weekly_rows = []
    cal_period = cal.date.dt.to_period("W-SUN")
    for period, group in frame.groupby(frame.datetime.dt.to_period("W-SUN"), sort=True):
        if period.start_time < cal.date.iloc[0] or period.end_time.normalize() > cal.date.iloc[-1]:
            continue
        required = cal.loc[cal_period.eq(period) & cal.is_trading_day]
        if required.empty or not pd.DatetimeIndex(group.datetime).equals(pd.DatetimeIndex(required.date)):
            continue
        weekly_rows.append({"weekly_available_at": required.session_close.iloc[-1],
                            "weekly_close": float(group.close.iloc[-1])})
    # Remove user-supplied derived fields before publication.
    derived = ["weekly_available_at", "weekly_close", "weekly_hist", "weekly_hist_previous",
               "weekly_ema13", "weekly_ema13_previous", "weekly_completed_count", "ready"]
    frame = frame.drop(columns=derived, errors="ignore")
    if weekly_rows:
        weekly = pd.DataFrame(weekly_rows)
        wc = weekly.weekly_close
        dif = wc.ewm(span=12, adjust=False).mean() - wc.ewm(span=26, adjust=False).mean()
        weekly["weekly_hist"] = dif-dif.ewm(span=9, adjust=False).mean()
        weekly["weekly_hist_previous"] = weekly.weekly_hist.shift()
        weekly["weekly_ema13"] = wc.ewm(span=13, adjust=False).mean()
        weekly["weekly_ema13_previous"] = weekly.weekly_ema13.shift()
        weekly["weekly_completed_count"] = np.arange(1, len(weekly)+1)
        frame = pd.merge_asof(frame, weekly.drop(columns="weekly_close"),
                              left_on="session_close", right_on="weekly_available_at", direction="backward")
    else:
        frame["weekly_available_at"] = pd.NaT
        for field in derived[2:-1]:
            frame[field] = np.nan
    frame["weekly_completed_count"] = frame.weekly_completed_count.fillna(0).astype(int)
    frame["ready"] = (frame.weekly_completed_count.ge(config.warmup_weeks)
                       & frame.weekly_hist_previous.notna()
                       & frame.weekly_ema13_previous.notna()
                       & frame.adr20.gt(0) & frame.atr14.gt(0))
    frame.attrs.update(force_span=force_span, warmup_weeks=warmup_weeks,
                       feature_version="elder_completed_week_features_v1")
    return frame.reset_index(drop=True)


prepare_features = build_features


def run_elder(features: pd.DataFrame, config: ElderConfig | dict | None = None,
              start="2022-01-01", end="2025-12-31", detailed: bool = False,
              cash_daily_rates=None) -> dict:
    """Simulate one independent risk-sized, unlevered adjusted-price sleeve.

    Use build_features output. Orders use only previous-close observations.
    Stops raised at today's close take effect next session; no retroactive stop
    application is allowed. Daily intraday-entry stop ordering is conservative.
    Optional cash_daily_rates is a finite 1D array aligned to RETURNED dates
    (not the complete feature history). Each session credits previous-close
    cash before opening orders; this is a financing sensitivity proxy only.
    """
    cfg = config if isinstance(config, ElderConfig) else ElderConfig(**(config or {}))
    if features.attrs.get("force_span", cfg.force_span) != cfg.force_span:
        raise ValueError("Feature force_span/config mismatch")
    if features.attrs.get("warmup_weeks", cfg.warmup_weeks) != cfg.warmup_weeks:
        raise ValueError("Feature warmup/config mismatch")
    f = features
    required = {"datetime", "open", "high", "low", "close", "tradeable", "ready", "force", "adr20", "atr14",
                "weekly_hist", "weekly_hist_previous", "weekly_ema13", "weekly_ema13_previous"}
    if required-set(f):
        raise ValueError(f"Missing feature fields: {sorted(required-set(f))}")
    dates = pd.to_datetime(f.datetime).dt.strftime("%Y-%m-%d").to_numpy()
    selected = np.flatnonzero((dates >= str(start)[:10]) & (dates <= str(end)[:10]))
    selected = selected[(selected > 0) & f.ready.shift(1, fill_value=False).to_numpy()[selected]]
    if len(selected) == 0:
        raise ValueError("No evaluable sessions after warmup")
    if not np.all(np.diff(selected) == 1):
        raise ValueError("Readiness failed inside evaluation window")
    rates = np.zeros(len(selected)) if cash_daily_rates is None else np.asarray(cash_daily_rates, dtype=float)
    if rates.shape != (len(selected),) or not np.isfinite(rates).all() or (rates <= -1).any():
        raise ValueError("cash_daily_rates must align to selected evaluation dates and be finite > -1")
    numeric = {name: f[name].to_numpy(dtype=float) for name in required
               if name not in {"datetime", "ready", "tradeable"}}
    for name, values in numeric.items():
        if not np.isfinite(values[selected]).all():
            raise ValueError(f"Nonfinite evaluated {name}")
    o,h,l,c = (numeric[name] for name in ["open", "high", "low", "close"])
    tradeable = f.tradeable.to_numpy(dtype=bool)
    hist_diff = numeric["weekly_hist"]-numeric["weekly_hist_previous"]
    ema_diff = numeric["weekly_ema13"]-numeric["weekly_ema13_previous"]
    if cfg.weekly_filter == "hist":
        permit, reverse = hist_diff > 0, hist_diff < 0
    elif cfg.weekly_filter == "ema":
        permit, reverse = ema_diff > 0, ema_diff < 0
    else:
        reverse = (hist_diff < 0) & (ema_diff < 0)
        permit = ~reverse
    fee_rate = cfg.cost_bps/10000
    cash, units = 1., 0.
    armed = False
    entry_line = None
    stop = None
    initial_stop = None
    peak = None
    protected = False
    pending_exit = None
    active = None
    trades, fills, decisions = [], [], []
    navs, exposures, stop_path = [], [], []
    counts = {"ambiguous_entries": 0, "blocked_sessions": 0, "setups_created": 0,
              "setups_cancelled_weekly": 0, "setups_preserved_after_force_reversal": 0,
              "entries": 0, "exits": 0, "cash_limited_entries": 0,
              "risk_budget_breaches_at_assumed_stop": 0}
    fees = 0.
    cash_interest = 0.
    previous_nav = 1.

    def exit_trade(i, price, reason, phase):
        nonlocal cash, units, stop, active, armed, entry_line, fees, pending_exit
        gross = units * price
        fee = gross*fee_rate
        cash += gross-fee
        fees += fee
        pnl = gross-fee-active["entry_cash_outlay"]
        if abs(pnl) <= 1e-12*max(1., active["entry_cash_outlay"]):
            pnl = 0.  # Fee-adjusted break-even must not become a floating-point win.
        active.update(status="closed", exit_date=dates[i], exit_price=float(price), exit_reason=reason,
                      exit_phase=phase, exit_fee=fee, cash_pnl=pnl,
                      net_return=pnl/active["entry_cash_outlay"],
                      r_multiple=pnl/active["planned_risk_cash"],
                      holding_bars=int(i-active.pop("entry_index")+1))
        trades.append(active)
        if detailed:
            fills.append({"date": dates[i], "phase": phase, "side": "sell", "price": float(price),
                          "units": units, "fee": fee, "reason": reason})
        units, stop, active, pending_exit = 0., None, None, None
        armed, entry_line = False, None
        counts["exits"] += 1

    for k, i in enumerate(selected):
        credit = cash*rates[k]
        cash += credit
        cash_interest += credit
        j = i-1
        current_stop = stop
        entered = False
        action = "hold_cash" if not units else "hold_position"
        if units:
            weekly_exit = bool(reverse[j]) and cfg.exit_mode in {"position", "either"}
            force_exit = numeric["force"][j] > 0 and cfg.exit_mode in {"swing", "either"}
            if pending_exit is None:
                if weekly_exit:
                    pending_exit = "weekly_trend_reversed"
                elif force_exit:
                    pending_exit = "force_positive"
            if not tradeable[i]:
                counts["blocked_sessions"] += 1
                # A crossed stop is latched until a tradeable session, never erased.
                if l[i] <= stop and pending_exit is None:
                    pending_exit = "stop_on_untradeable_session"
            elif pending_exit is not None:
                action = pending_exit
                exit_trade(i, o[i], action, "open")
            elif o[i] <= stop:
                action = "gap_stop"
                exit_trade(i, o[i], action, "open")
            elif l[i] <= stop:
                action = "protective_stop"
                exit_trade(i, stop, action, "intraday")
        else:
            if reverse[j] and armed:
                armed, entry_line = False, None
                counts["setups_cancelled_weekly"] += 1
            if not armed and permit[j] and numeric["force"][j] < 0 and tradeable[j]:
                armed = True
                entry_line = h[j]+cfg.entry_buffer_adjusted
                counts["setups_created"] += 1
            elif armed:
                entry_line = min(entry_line, h[j]+cfg.entry_buffer_adjusted)
                if numeric["force"][j] >= 0:
                    counts["setups_preserved_after_force_reversal"] += 1
            if armed:
                if not tradeable[i]:
                    counts["blocked_sessions"] += 1
                elif h[i] >= entry_line:
                    price = max(o[i], entry_line)
                    candidate_stop = l[j]-cfg.entry_buffer_adjusted
                    if candidate_stop > 0 and candidate_stop < price:
                        phase = "open" if o[i] >= entry_line else "intraday"
                        desired_risk = cash*cfg.risk_fraction
                        loss_per_unit = price*(1+fee_rate)-candidate_stop*(1-fee_rate)
                        risk_units = desired_risk/loss_per_unit
                        available_units = cash/(price*(1+fee_rate))
                        units = min(risk_units, available_units)
                        counts["cash_limited_entries"] += int(risk_units>available_units)
                        entry_fee = units*price*fee_rate
                        outlay = units*price+entry_fee
                        planned_risk = units*loss_per_unit
                        if planned_risk > desired_risk+1e-12:
                            counts["risk_budget_breaches_at_assumed_stop"] += 1
                        cash -= outlay
                        if abs(cash) < 1e-14:
                            cash = 0.
                        fees += entry_fee
                        stop = initial_stop = candidate_stop
                        peak = max(price,c[i])  # No assumption that entry-day high follows an intraday entry.
                        protected = False
                        active = {"status": "open", "entry_date": dates[i], "entry_index": int(i),
                                  "entry_phase": phase, "entry_price": float(price), "initial_stop": float(stop),
                                  "units": units, "entry_fee": entry_fee, "entry_cash_outlay": outlay,
                                  "risk_budget_cash": desired_risk, "planned_risk_cash": planned_risk,
                                  "frozen_adr20": float(numeric["adr20"][j]), "ambiguous_intraday_order": False}
                        counts["entries"] += 1
                        armed, entry_line, entered = False, None, True
                        action = "entry"
                        if detailed:
                            fills.append({"date": dates[i], "phase": phase, "side": "buy", "price": float(price),
                                          "units": units, "fee": entry_fee, "reason": "trailing_buy_stop"})
                        if l[i] <= stop:
                            active["ambiguous_intraday_order"] = phase == "intraday"
                            counts["ambiguous_entries"] += int(phase == "intraday")
                            action = "entry_day_stop_conservative" if phase == "intraday" else "entry_day_stop"
                            exit_trade(i, stop, action, "intraday")
        if units and tradeable[i]:
            if not entered:
                peak = max(peak,h[i])
            if cfg.stop_mode in {"protect50", "protect70"}:
                # Declared mechanization: close gain reaches one pre-entry ADR20.
                protected = protected or c[i]-active["entry_price"] >= active["frozen_adr20"]
                if protected:
                    fraction = .5 if cfg.stop_mode == "protect50" else .7
                    break_even = active["entry_price"]*(1+fee_rate)/(1-fee_rate)
                    candidate = max(break_even, active["entry_price"]+fraction*(peak-active["entry_price"]))
                    stop = max(stop,candidate)
            elif cfg.stop_mode == "atr3":
                stop = max(stop,c[i]-3*numeric["atr14"][i])
            assert stop >= initial_stop
        nav = cash+units*c[i]
        if not math.isfinite(nav) or nav <= 0 or cash < -1e-12 or units < 0:
            raise ValueError("Invalid ledger state")
        navs.append(nav)
        exposures.append(units*c[i]/nav)
        stop_path.append(float(stop) if stop is not None else None)
        if detailed:
            decisions.append({"date": dates[i], "observed_close_date": dates[j],
                              "weekly_entry_permitted": bool(permit[j]), "weekly_reversed": bool(reverse[j]),
                              "force": float(numeric["force"][j]), "action": action,
                              "stop_effective_today": current_stop,
                              "stop_effective_next_session": stop, "armed_for_next_check": armed,
                              "cash": cash, "units": units, "nav": nav,
                              "daily_return": nav/previous_nav-1})
        previous_nav = nav
    if active is not None:
        i = selected[-1]
        active.update(as_of=dates[i], marked_position_value=units*c[i],
                      unrealized_cash_pnl=units*c[i]-active["entry_cash_outlay"],
                      holding_bars=int(i-active.pop("entry_index")+1))
        trades.append(active)
    nav = np.asarray(navs)
    closed = [t for t in trades if t["status"] == "closed"]
    pnl = np.array([t["cash_pnl"] for t in closed])
    peak_nav = np.maximum.accumulate(np.r_[1.,nav])[1:]
    gross_profit = float(pnl[pnl>0].sum())
    gross_loss = float(-pnl[pnl<0].sum())
    metrics = {"total_return": float(nav[-1]-1), "maximum_drawdown": float((1-nav/peak_nav).max()),
               "closed_trades": len(closed), "open_trades": int(active is not None),
               "wins": int((pnl>0).sum()), "losses": int((pnl<0).sum()),
               "breakeven_count": int((pnl==0).sum()),
               "win_rate": float(np.mean(pnl>0)) if len(pnl) else None,
               "gross_profit_cash": gross_profit,"gross_loss_cash": gross_loss,
               "profit_factor": gross_profit/gross_loss if gross_loss else None,
               "total_fees": float(fees),"cash_interest": float(cash_interest),"mean_close_exposure": float(np.mean(exposures))}
    return {"config":cfg.to_dict(),"dates": dates[selected].tolist(),"nav":nav,
            "exposure":np.asarray(exposures),"stop_path":stop_path if detailed else [],
            "trades":trades,"fills":fills,"decisions":decisions,"metrics":metrics,
            "diagnostics":{**counts,
                "cash_interest":float(cash_interest),"cash_rate_supplied":cash_daily_rates is not None,"market":"US","long_only":True,
                "price_basis":"adjusted-price fractional research units, not nominal historical share ledger",
                "buffer_basis":"0.01 adjusted USD default; not a reconstructed exchange tick",
                "neutral_weekly_slope":"no new hist/ema setup, no forced exit or cancellation",
                "initial_stop_approximation":"previous completed daily low minus adjusted-price buffer; entry-day partial low unavailable",
                "dynamic_stop_timing":"close-confirmed activation/update effective next session only",
                "entry_day_peak":"max(entry_price, close), excludes unknown-before-entry intraday high",
                "risk_limit":"planned stop fill including proportional fees; gaps can exceed planned risk",
                "cash_policy":("no leverage, no cross-symbol transfer; prior-close cash earns supplied daily proxy before opening orders"
                               if cash_daily_rates is not None else "no leverage, no cross-symbol capital transfer, idle cash earns zero")}}
