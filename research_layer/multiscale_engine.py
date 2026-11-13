"""Causal daily execution for registered multi-timeframe research combinations.

This generic next-open engine is distinct from Elder's trailing buy-stop rules.
The optional Numba kernel and the Python fallback execute the same arithmetic.
Prepared windows should be reused across a large candidate grid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    njit = None


@dataclass(frozen=True)
class MultiscaleConfig:
    daily_signal: str = "efi2_pullback"
    weekly_filter: str = "hist_rising"
    exit_mode: str = "hold10"
    risk_fraction: float = .01
    cost_bps: float = 10.
    initial_stop_atr: float = 2.
    trailing_stop_atr: float = 3.
    version: str = "multiscale_next_open_us_v1"

    def __post_init__(self):
        if self.exit_mode not in {"hold5", "hold10", "hold20", "trail"}:
            raise ValueError("exit_mode must be hold5, hold10, hold20 or trail")
        if self.version != "multiscale_next_open_us_v1":
            raise ValueError("Unsupported execution version")
        for key in ("daily_signal", "weekly_filter"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty identifier")
        for key in ("risk_fraction", "cost_bps", "initial_stop_atr", "trailing_stop_atr"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite numeric")
        if not 0 < self.risk_fraction <= 1 or not 0 <= self.cost_bps < 10000:
            raise ValueError("Invalid risk_fraction or cost_bps")
        if self.initial_stop_atr <= 0 or self.trailing_stop_atr <= 0:
            raise ValueError("ATR stop multiples must be positive")

    def to_dict(self):
        return asdict(self)


Config = MultiscaleConfig


@dataclass
class PreparedFeatures:
    dates: np.ndarray
    numeric: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]
    signals: dict[str, np.ndarray]
    weekly: dict[str, np.ndarray]


@dataclass
class PreparedWindow:
    dates: np.ndarray
    observed_dates: np.ndarray
    numeric: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]
    signals: dict[str, np.ndarray]
    weekly: dict[str, np.ndarray]
    combined_signals: dict[str, np.ndarray] = field(default_factory=dict)
    dates_list: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.dates_list = self.dates.tolist()

    def signal(self, identifier):
        if identifier in self.signals:
            return self.signals[identifier]
        if identifier not in self.combined_signals:
            parts = identifier.split("&")
            if len(parts) < 2 or len(parts) != len(set(parts)) or any(p not in self.signals for p in parts):
                raise ValueError(f"Unknown or invalid daily signal: {identifier}")
            self.combined_signals[identifier] = np.logical_and.reduce([self.signals[p] for p in parts])
        return self.combined_signals[identifier]


def _boolean(values, name):
    s = pd.Series(values)
    if s.isna().any() or not s.isin([True, False, 0, 1]).all():
        raise ValueError(f"Explicit boolean/0/1 values required: {name}")
    return np.ascontiguousarray(s.to_numpy(dtype=np.bool_))


def prepare_arrays(features: pd.DataFrame | PreparedFeatures) -> PreparedFeatures:
    if isinstance(features, PreparedFeatures):
        return features
    required = {"datetime", "open", "high", "low", "close", "atr14", "tradeable", "ready"}
    if required - set(features):
        raise ValueError(f"Missing feature columns: {sorted(required-set(features))}")
    date = pd.to_datetime(features.datetime)
    if date.empty or date.isna().any() or date.dt.tz is not None or not date.eq(date.dt.normalize()).all():
        raise ValueError("Dates must be nonempty naive local midnight observations")
    if date.duplicated().any() or not date.is_monotonic_increasing:
        raise ValueError("Feature dates must be strictly increasing")
    values = {k: np.ascontiguousarray(features[k].to_numpy(dtype=np.float64))
              for k in ["open", "high", "low", "close", "atr14"]}
    o, h, l, c = [values[k] for k in ["open", "high", "low", "close"]]
    if any(not np.isfinite(values[k]).all() for k in ["open", "high", "low", "close"]):
        raise ValueError("Observed OHLC cannot be missing")
    if (l <= 0).any() or (h < np.maximum(o, c)).any() or (l > np.minimum(o, c)).any():
        raise ValueError("Invalid observed OHLC")
    flags = {k: _boolean(features[k], k) for k in ["tradeable", "ready"]}
    flags["universe_trade_eligible"] = (_boolean(features.universe_trade_eligible,"universe_trade_eligible")
                                          if "universe_trade_eligible" in features else np.ones(len(date),dtype=np.bool_))
    if "volume" in features:
        volume = features.volume.to_numpy(dtype=float)
        if not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError("Volume must be finite nonnegative shares")
        flags["tradeable"] &= volume > 0
    if np.any(flags["ready"] & (~np.isfinite(values["atr14"]) | (values["atr14"] <= 0))):
        raise ValueError("Ready observations require positive finite ATR")
    signals = {k[len("signal__"):]: _boolean(features[k], k) for k in features if k.startswith("signal__")}
    weekly = {k[len("weekly__"):]: _boolean(features[k], k) for k in features if k.startswith("weekly__")}
    if not signals or not weekly:
        raise ValueError("Need signal__ and weekly__ feature masks")
    return PreparedFeatures(date.dt.strftime("%Y-%m-%d").to_numpy(dtype="U10"), values, flags, signals, weekly)


def prepare_window(features: pd.DataFrame | PreparedFeatures, start="2022-01-01", end="2025-12-31",
                   evaluation_dates=None) -> PreparedWindow:
    """Align to requested sessions, retaining cash before IPO/warmup.

    A supplied exchange session grid permits leading pre-listing cash dates.
    Missing interior or trailing observations are rejected rather than imputed.
    """
    f = prepare_arrays(features)
    if evaluation_dates is None:
        dates = f.dates[(f.dates >= str(start)[:10]) & (f.dates <= str(end)[:10])]
    else:
        dates = pd.to_datetime(pd.Series(evaluation_dates)).dt.strftime("%Y-%m-%d").to_numpy(dtype="U10")
        if len(dates) == 0 or len(np.unique(dates)) != len(dates) or np.any(dates[1:] <= dates[:-1]):
            raise ValueError("evaluation_dates must be nonempty strictly increasing sessions")
        if dates[0] < str(start)[:10] or dates[-1] > str(end)[:10]:
            raise ValueError("evaluation_dates lie outside requested start/end")
    if not len(dates):
        raise ValueError("No evaluation dates; provide the common exchange calendar for pre-listing cash")
    idx = np.searchsorted(f.dates, dates)
    leading = dates < f.dates[0]
    safe = np.minimum(idx, len(f.dates)-1)
    observed = (~leading) & (idx < len(f.dates)) & (f.dates[safe] == dates)
    if np.any(~observed & ~leading):
        raise ValueError(f"Missing interior/trailing observation: {dates[np.flatnonzero(~observed & ~leading)[0]]}")
    previous = safe - 1
    have_previous = observed & (previous >= 0)
    prev_safe = np.maximum(previous, 0)
    numeric = {k: np.where(observed, v[safe], 0.) for k, v in f.numeric.items() if k != "atr14"}
    numeric["atr14"] = np.where(observed, f.numeric["atr14"][safe], 0.)
    numeric["prior_atr14"] = np.where(have_previous, f.numeric["atr14"][prev_safe], 0.)
    flags = {"tradeable": observed & f.flags["tradeable"][safe],
             "prior_ready": have_previous & f.flags["ready"][prev_safe],
             "prior_tradeable": have_previous & f.flags["tradeable"][prev_safe],
             "prior_universe_trade_eligible": have_previous & f.flags["universe_trade_eligible"][prev_safe],
             "observed": observed}
    signals = {key: have_previous & val[prev_safe] for key, val in f.signals.items()}
    weekly = {key: have_previous & val[prev_safe] for key, val in f.weekly.items()}
    observed_dates = np.where(have_previous, f.dates[prev_safe], "")
    return PreparedWindow(dates, observed_dates, numeric, flags, signals, weekly)


# Exit reason ids: 1 gap stop,2 protective stop,3 entry-day stop,
# 4 weekly filter invalid,5 scheduled close,6 scheduled exit delayed by suspension,
# 7 stop observed while untradeable. Exit phases:1 open,2 intraday,3 close.
REASONS = {1: "gap_stop", 2: "protective_stop", 3: "entry_day_stop", 4: "weekly_filter_invalid",
           5: "scheduled_close", 6: "scheduled_exit_after_suspension", 7: "stop_after_suspension"}
PHASES = {1: "open", 2: "intraday", 3: "close"}


def _kernel_python(o, h, l, c, atr, prior_atr, tradeable, prior_ready, prior_tradeable, prior_eligible,
                   signal, permit, rates, horizon, risk_fraction, fee, initial_atr, trail_atr,
                   retain_trades, detailed):
    n = len(o)
    nav = np.empty(n); exposure = np.empty(n)
    ledger = np.zeros((n, 24)) if retain_trades else np.empty((0, 24))
    trace = np.zeros((n, 7)) if detailed else np.empty((0, 7))
    stats = np.zeros(19); diagnostic = np.zeros(8)
    cash, units, stop, initial_stop = 1., 0., 0., 0.
    entry, entry_i, setup_i, entry_atr, outlay, entry_fee, risk_budget, planned_risk = 0., 0, 0, 0., 0., 0., 0., 0.
    peak, trough = 0., 0.
    pending_entry, pending_exit, trade_count = False, 0, 0
    # stats: closed,wins,losses,breakeven,grossprofit,grossloss,fees,interest,
    # sumreturn,sumR,sumwinreturn,sumlossreturn,unrealized,open_count,holding_sum,
    # sum_mae,sum_mfe,planned_risk_breaches,closed_cash_pnl.
    for k in range(n):
        credit = cash * rates[k]; cash += credit; stats[7] += credit
        old_stop = stop
        entered = False
        action = 0
        if units == 0.:
            if pending_entry and (not permit[k] or not prior_eligible[k]):
                pending_entry = False
                if not permit[k]: diagnostic[2] += 1
                else: diagnostic[7] += 1
            if not pending_entry and prior_ready[k] and prior_tradeable[k] and prior_eligible[k] and signal[k] and permit[k]:
                pending_entry = True; setup_i = k; diagnostic[1] += 1
            if pending_entry:
                if not tradeable[k]:
                    diagnostic[3] += 1
                elif not prior_ready[k] or not math.isfinite(prior_atr[k]) or prior_atr[k] <= 0.:
                    pending_entry = False; diagnostic[5] += 1
                else:
                    candidate_stop = o[k] - initial_atr * prior_atr[k]
                    pending_entry = False
                    if candidate_stop <= 0. or candidate_stop >= o[k]:
                        diagnostic[5] += 1
                    else:
                        entry, entry_i, entry_atr = o[k], k, prior_atr[k]
                        risk_budget = cash * risk_fraction
                        loss_per_unit = entry*(1.+fee) - candidate_stop*(1.-fee)
                        risk_units = risk_budget / loss_per_unit
                        cash_units = cash / (entry*(1.+fee))
                        units = min(risk_units, cash_units)
                        diagnostic[4] += 1. if risk_units > cash_units else 0.
                        entry_fee = units*entry*fee
                        outlay = units*entry+entry_fee
                        planned_risk = units*loss_per_unit
                        stats[17] += 1. if planned_risk > risk_budget+1e-12 else 0.
                        cash -= outlay
                        if abs(cash) < 1e-14: cash = 0.
                        stats[6] += entry_fee
                        stop = candidate_stop; initial_stop = stop
                        peak = entry; trough = entry
                        entered = True; action = 1; diagnostic[0] += 1
        elif horizon == 0 and not permit[k] and pending_exit == 0:
            pending_exit = 4
        if units > 0.:
            holding = k-entry_i+1
            reason, phase, price = 0, 0, 0.
            if not tradeable[k]:
                diagnostic[3] += 1
                if l[k] <= stop and pending_exit == 0: pending_exit = 7
                if horizon > 0 and holding >= horizon and pending_exit == 0: pending_exit = 6
            else:
                if o[k] <= stop:
                    reason, phase, price = 1, 1, o[k]
                elif pending_exit:
                    reason, phase, price = pending_exit, 1, o[k]
                elif l[k] <= stop:
                    reason, phase, price = (3 if entered else 2), 2, stop
                elif horizon > 0 and holding >= horizon:
                    reason, phase, price = 5, 3, c[k]
                if reason:
                    # Full exit-day H/L would include prices after a stop/open exit.
                    if phase == 3:
                        peak = max(peak, h[k]); trough = min(trough, l[k])
                    else:
                        peak = max(peak, o[k], price); trough = min(trough, o[k], price)
                    exit_fee = units*price*fee
                    proceeds = units*price-exit_fee
                    cash += proceeds; stats[6] += exit_fee
                    pnl = proceeds-outlay
                    if abs(pnl) <= 1e-12*max(1., outlay): pnl = 0.
                    ret = pnl/outlay; rm = pnl/planned_risk
                    mae = min(0., trough/entry-1.); mfe = max(0., peak/entry-1.)
                    stats[0] += 1; stats[8] += ret; stats[9] += rm; stats[14] += holding
                    stats[15] += mae; stats[16] += mfe; stats[18] += pnl
                    if pnl > 0.:
                        stats[1] += 1; stats[4] += pnl; stats[10] += ret
                    elif pnl < 0.:
                        stats[2] += 1; stats[5] -= pnl; stats[11] += ret
                    else: stats[3] += 1
                    if retain_trades:
                        ledger[trade_count] = np.array([1., entry_i, k, entry, price, initial_stop, units,
                            entry_fee, exit_fee, outlay, risk_budget, planned_risk, pnl, ret, rm, holding,
                            mae, mfe, reason, phase, entry_atr, setup_i, 1. if phase!=3 else 0., stop])
                    trade_count += 1
                    units = 0.; stop = 0.; pending_exit = 0; action = 10+reason
                else:
                    peak = max(peak, h[k]); trough = min(trough, l[k])
                    if horizon == 0 and math.isfinite(atr[k]) and atr[k] > 0.:
                        stop = max(stop, c[k]-trail_atr*atr[k])
        nav[k] = cash+units*c[k]
        if not math.isfinite(nav[k]) or nav[k] <= 0. or cash < -1e-12 or units < 0.:
            raise ValueError("Invalid execution ledger")
        exposure[k] = units*c[k]/nav[k]
        if detailed: trace[k] = np.array([cash,units,nav[k],old_stop,stop,action,1. if pending_entry else 0.])
    if units > 0.:
        stats[12] = units*c[-1]-outlay; stats[13] = 1.
        if retain_trades:
            ledger[trade_count] = np.array([0.,entry_i,n-1,entry,c[-1],initial_stop,units,entry_fee,0.,outlay,
                risk_budget,planned_risk,stats[12],stats[12]/outlay,stats[12]/planned_risk,n-entry_i,
                min(0.,trough/entry-1.),max(0.,peak/entry-1.),0.,0.,entry_atr,setup_i,0.,stop])
        trade_count += 1
    diagnostic[6] = 1. if pending_entry else 0.
    return nav, exposure, ledger[:trade_count] if retain_trades else ledger, trace, stats, diagnostic


_kernel = njit(cache=True, nogil=True)(_kernel_python) if njit is not None else _kernel_python


def run_strategy(features: pd.DataFrame | PreparedFeatures | PreparedWindow,
                 config: MultiscaleConfig | dict | None = None,
                 start="2022-01-01", end="2025-12-31", cash_daily_rates=None,
                 detailed: bool = False, include_trades: bool = True, evaluation_dates=None) -> dict[str, Any]:
    cfg = config if isinstance(config, MultiscaleConfig) else MultiscaleConfig(**(config or {}))
    window = features if isinstance(features, PreparedWindow) else prepare_window(features,start,end,evaluation_dates)
    if cfg.weekly_filter not in window.weekly:
        raise ValueError(f"Unknown weekly filter: {cfg.weekly_filter}")
    n = len(window.dates)
    rates = np.zeros(n) if cash_daily_rates is None else np.ascontiguousarray(cash_daily_rates,dtype=np.float64)
    if rates.shape != (n,) or not np.isfinite(rates).all() or (rates <= -1.).any():
        raise ValueError("cash_daily_rates must align with returned dates and be finite >-1")
    horizon = 0 if cfg.exit_mode == "trail" else int(cfg.exit_mode[4:])
    numeric, flags = window.numeric, window.flags
    nav, exposure, ledger, trace, s, d = _kernel(
        *[numeric[k] for k in ["open","high","low","close","atr14","prior_atr14"]],
        *[flags[k] for k in ["tradeable","prior_ready","prior_tradeable","prior_universe_trade_eligible"]],
        window.signal(cfg.daily_signal), window.weekly[cfg.weekly_filter], rates, horizon,
        cfg.risk_fraction,cfg.cost_bps/10000.,cfg.initial_stop_atr,cfg.trailing_stop_atr,
        include_trades or detailed,detailed)
    dates = window.dates
    trades = []
    for t in ledger:
        closed = bool(t[0]); ei, xi, si = int(t[1]), int(t[2]), int(t[21])
        row = {"status":"closed" if closed else "open", "entry_date":str(dates[ei]),
               "signal_close_date":str(window.observed_dates[si]),"entry_phase":"open","entry_price":t[3],
               "initial_stop":t[5],"units":t[6],"entry_fee":t[7],"entry_cash_outlay":t[9],
               "risk_budget_cash":t[10],"planned_risk_cash":t[11],"holding_bars":int(t[15]),
               "mae":t[16],"mfe":t[17],"entry_atr":t[20],
               "excursion_exit_day_censored":bool(t[22]),"last_protective_stop":t[23]}
        if closed:
            row.update(exit_date=str(dates[xi]),exit_price=t[4],exit_fee=t[8],cash_pnl=t[12],
                       net_return=t[13],r_multiple=t[14],exit_reason=REASONS[int(t[18])],exit_phase=PHASES[int(t[19])])
        else:
            row.update(as_of=str(dates[xi]),marked_position_value=t[6]*t[4],unrealized_cash_pnl=t[12])
        trades.append(row)
    nc = int(s[0]); nw = int(s[1]); nl = int(s[2])
    peak_nav = np.maximum.accumulate(np.r_[1.,nav])[1:]
    metrics = {"total_return":float(nav[-1]-1.),"maximum_drawdown":float(np.max(1.-nav/peak_nav)),
        "closed_trades":nc,"open_trades":int(s[13]),"wins":nw,"losses":nl,"breakeven_count":int(s[3]),
        "win_rate":nw/nc if nc else None,"nonloss_rate":(nc-nl)/nc if nc else None,
        "gross_profit_cash":float(s[4]),"gross_loss_cash":float(s[5]),"profit_factor":float(s[4]/s[5]) if s[5] else None,
        "total_fees":float(s[6]),"cash_interest":float(s[7]),"mean_net_trade_return":float(s[8]/nc) if nc else None,
        "mean_R_multiple":float(s[9]/nc) if nc else None,"average_win_return":float(s[10]/nw) if nw else None,
        "average_loss_return":float(s[11]/nl) if nl else None,"unrealized_cash_pnl":float(s[12]),
        "closed_cash_pnl":float(s[18]),"mean_holding_bars":float(s[14]/nc) if nc else None,
        "mean_mae":float(s[15]/nc) if nc else None,"mean_mfe":float(s[16]/nc) if nc else None,
        "sum_net_trade_returns":float(s[8]),"sum_R_multiples":float(s[9]),
        "sum_win_returns":float(s[10]),"sum_loss_returns":float(s[11]),
        "sum_holding_bars":float(s[14]),"sum_mae":float(s[15]),"sum_mfe":float(s[16]),
        "mean_close_exposure":float(np.mean(exposure))}
    if not np.isclose(nav[-1]-1.,s[18]+s[12]+s[7],rtol=1e-10,atol=1e-12):
        raise ValueError("Closed/open PnL and cash interest do not reconcile")
    decisions = []
    if detailed:
        for k, t in enumerate(trace):
            code = int(t[5]);action = "entry" if code == 1 else (REASONS[code-10] if code >= 10 else "hold")
            decisions.append({"date":str(dates[k]),"observed_close_date":str(window.observed_dates[k]),
                "cash":t[0],"units":t[1],"nav":t[2],"stop_effective_today":t[3] or None,
                "stop_effective_next_session":t[4] or None,"action":action,"entry_pending":bool(t[6])})
    return {"config":cfg.to_dict(),"dates":window.dates_list,"nav":nav,"exposure":exposure,
        "trades":trades,"metrics":metrics,"decisions":decisions,
        "stop_path":[x or None for x in trace[:,4]] if detailed else [],
        "diagnostics":{"entries":int(d[0]),"setups_created":int(d[1]),"setups_cancelled_weekly":int(d[2]),
          "blocked_sessions":int(d[3]),"cash_limited_entries":int(d[4]),"invalid_stop_entries_rejected":int(d[5]),
          "pending_entry_at_end":bool(d[6]),"risk_budget_breaches_at_assumed_stop":int(s[17]),
          "setups_cancelled_price_eligibility":int(d[7]),
          "prelisting_cash_sessions":int((~flags['observed']).sum()),"eligible_sessions":int(flags['prior_ready'].sum()),
          "compiled_kernel":njit is not None,"market":"US","long_only":True,
          "initial_stop":"entry open minus multiple of previous completed session ATR; risk includes both fees",
          "entry":"prior-close condition, next tradeable open; pending entries cancelled if weekly permit fails",
          "fixed_holding":"entry session counts one; scheduled close, stop takes precedence; suspended expiry exits at next tradeable open",
          "trailing":"close minus ATR multiple effective next session, never loosened; weekly filter failure exits next tradeable open",
          "excursions":"exit-day H/L excluded for open/stop exits; their known-path MAE/MFE are censored",
          "cash_policy":"independent sleeve, no leverage; prior-close cash receives optional supplied daily RF proxy"}}


run_multiscale = run_strategy
