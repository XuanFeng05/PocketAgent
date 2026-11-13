"""Forward-adjusted price research with a normalized long/cash return index.

This module intentionally has no real share ledger. It reuses the rule strategies,
but their account view's 0/1 share fields represent a position state only.
"""
from __future__ import annotations

from datetime import datetime, time
from dataclasses import asdict
import math
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from evaluation_layer.metrics.portfolio import portfolio_metrics
from strategy_layer.base import AccountView, BarView
from strategy_layer.registry import create_strategy


SUPPORTED_STRATEGIES = frozenset({"sma_trend", "breakout", "mean_reversion", "cash", "buy_hold"})
RETURN_BASIS = "forward_adjusted_long_cash_index"
ASSUMPTIONS = [
    "Forward-adjusted OHLC prices determine signals and normalized price returns; initial NAV is 1.0.",
    "This is a long/cash technical return index, not a real-share account or a reconstruction of dividend cash flows.",
    "Close signals change exposure at the next observed bar's open. Old exposure earns the overnight return; new exposure earns the intraday return.",
    "One-sided cost_bps reduces NAV proportionally at each actual exposure change; default zero means gross technical returns.",
    "Zero-volume or explicitly non-tradeable bars block exposure changes. No price-limit, lot-size, auction-capacity, tax, or real-order execution model is applied.",
    "Cash earns no interest. Final exposure is marked at the final adjusted close and is not forcibly closed.",
    "The input adjustment version must be consistent across evaluation and warmup; the supplied price history is not independently authenticated here.",
]


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return str(value).strip().lower() in {"true", "1"}
    raise ValueError(f"{field} must contain explicit boolean or 0/1 values.")


def normalize_technical_bars(bars: Any, symbol: str | None = None, min_rows: int = 2) -> pd.DataFrame:
    """Validate one daily pre-adjusted series without relabelling raw prices."""
    if isinstance(min_rows, bool) or not isinstance(min_rows, Integral) or min_rows < 1:
        raise ValueError("min_rows must be a positive integer.")
    frame = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    if "datetime" not in frame and "date" in frame:
        frame = frame.rename(columns={"date": "datetime"})
    required = {"datetime", "open", "high", "low", "close", "adjust"}
    if required - set(frame.columns):
        raise ValueError(f"Technical input missing columns: {', '.join(sorted(required - set(frame.columns)))}")
    if "symbol" not in frame:
        if not symbol:
            raise ValueError("Technical input requires a symbol column or explicit symbol.")
        frame["symbol"] = symbol
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    if frame["symbol"].isin(["", "NONE", "NAN", "<NA>"]).any():
        raise ValueError("Every bar requires a nonempty symbol.")
    if symbol:
        frame = frame.loc[frame["symbol"].eq(str(symbol).upper().strip())].copy()
    if len(frame) < min_rows:
        raise ValueError(f"At least {min_rows} daily bars are required.")
    if frame["symbol"].nunique() != 1:
        raise ValueError("Technical research uses one independent symbol at a time.")
    if not frame["adjust"].astype(str).eq("pre").all():
        raise ValueError("Technical research requires explicit forward-adjusted prices (adjust=pre).")
    if "freq" in frame and not frame["freq"].astype(str).eq("daily").all():
        raise ValueError("Technical research requires daily bars.")
    dates = pd.to_datetime(frame["datetime"], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Every bar requires a valid local date without timezone offsets.")
    frame["datetime"] = dates.dt.normalize()
    if frame["datetime"].duplicated().any():
        raise ValueError("Duplicate symbol/date rows are not allowed.")
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        if field not in frame:
            continue
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(frame[field].to_numpy(dtype=float)).all():
            raise ValueError(f"{field} must contain finite values.")
        if ((frame[field] <= 0) if field in {"open", "high", "low", "close"} else (frame[field] < 0)).any():
            raise ValueError("Prices must be positive; volume and amount must be nonnegative.")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any() or (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("OHLC bounds are inconsistent.")
    if "is_tradeable" in frame:
        frame["is_tradeable"] = frame["is_tradeable"].map(lambda value: _boolean(value, "is_tradeable"))
    return frame.sort_values("datetime", kind="stable").reset_index(drop=True)


def run_technical_backtest(
    bars: Any, strategy: str = "sma_trend", parameters: dict | None = None,
    *, cost_bps: float = 0.0, warmup_bars: Any | None = None,
    session_closes: dict | None = None,
) -> dict:
    """Apply existing technical rules to a fully invested/cash return index."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported technical strategy: {strategy!r}")
    if isinstance(cost_bps, bool) or not isinstance(cost_bps, (int, float)) or not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be finite and between 0 inclusive and 10000 exclusive.")
    frame = normalize_technical_bars(bars)
    symbol = str(frame.iloc[0]["symbol"])
    policy = create_strategy(strategy, parameters)
    warmup = None
    if warmup_bars is not None and len(warmup_bars):
        warmup = normalize_technical_bars(warmup_bars, min_rows=1)
        if str(warmup.iloc[0]["symbol"]) != symbol:
            raise ValueError("Warmup and evaluation must use the same symbol.")
        if warmup.iloc[-1]["datetime"] >= frame.iloc[0]["datetime"]:
            raise ValueError("Warmup must end strictly before evaluation.")

    nav, exposure, costs, turnover = 1.0, 0, 0.0, 0.0
    closes: list[float] = []
    equity, fills, signals = [], [], []
    pending = None
    previous_close = None
    cost_rate = float(cost_bps) / 10_000

    def observe(row, *, origin="evaluation"):
        closes.append(float(row["close"]))
        timestamp = pd.Timestamp(session_closes[row["datetime"].date().isoformat()]).to_pydatetime() if session_closes is not None else datetime.combine(row["datetime"].date(), time(15))
        # The registry only observes whether a position exists. No adjusted
        # price is converted to a fabricated number of real shares.
        intent = policy.on_close(BarView(symbol, timestamp, tuple(closes)),
                                 AccountView(nav if not exposure else 0.0, nav, exposure, exposure))
        if intent is None or intent.direction == 0:
            return None
        target = 1 if intent.direction > 0 else 0
        if target == exposure:
            return None
        return {"signal_time": timestamp.isoformat(), "target_exposure": target,
                "reason": intent.reason, "origin": origin}

    if warmup is not None:
        for row in warmup.to_dict("records"):
            pending = observe(row, origin="warmup")
            previous_close = float(row["close"])
        if pending is not None:
            pending.update(signal_id="signal_000001", status="pending")
            signals.append(pending)

    for row in frame.to_dict("records"):
        day = row["datetime"].date()
        opening, close = float(row["open"]), float(row["close"])
        old_exposure = exposure
        overnight_return = opening / previous_close - 1.0 if previous_close is not None else 0.0
        nav *= 1.0 + old_exposure * overnight_return
        opening_nav = nav
        cost = 0.0
        if pending is not None:
            tradeable = bool(row.get("is_tradeable", True)) and float(row.get("volume", 1.0)) > 0
            event = {"event_type": "technical_position_change", "signal_id": pending["signal_id"],
                     "signal_time": pending["signal_time"],
                     "execution_time": datetime.combine(day, time(9, 30)).isoformat(),
                     "from_exposure": exposure, "target_exposure": pending["target_exposure"],
                     "adjusted_open": opening, "status": "applied" if tradeable else "blocked",
                     "reason": None if tradeable else "non_tradeable_bar", "cost_nav_units": 0.0}
            if tradeable:
                turnover += nav
                cost = nav * cost_rate
                nav *= 1.0 - cost_rate
                costs += cost
                exposure = pending["target_exposure"]
                event["cost_nav_units"] = cost
            fills.append(event)
            pending["status"] = event["status"]
            pending["execution_time"] = event["execution_time"]
            pending = None
        intraday_return = close / opening - 1.0
        nav *= 1.0 + exposure * intraday_return
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError("Technical NAV became nonfinite or nonpositive.")
        equity.append({"date": day.isoformat(), "time": pd.Timestamp(session_closes[day.isoformat()]).isoformat() if session_closes is not None else datetime.combine(day, time(15)).isoformat(),
                       "nav": nav, "exposure": exposure, "overnight_exposure": old_exposure,
                       "adjusted_close": close, "overnight_return": overnight_return,
                       "intraday_return": intraday_return, "opening_nav_before_cost": opening_nav,
                       "cost_nav_units": cost, "cumulative_cost_nav_units": costs})
        pending = observe(row)
        if pending is not None:
            pending.update(signal_id=f"signal_{len(signals)+1:06d}", status="pending")
            signals.append(pending)
        previous_close = close
    if pending is not None:
        pending["status"] = "pending_end_of_data"
    metrics = portfolio_metrics([item["nav"] for item in equity], initial_nav=1.0,
                                turnover_value=turnover, total_fees=costs)
    metrics["total_cost_nav_units"] = metrics.pop("total_fees")
    metrics.update(position_changes=sum(item["status"] == "applied" for item in fills),
                   blocked_changes=sum(item["status"] == "blocked" for item in fills),
                   final_exposure=exposure,
                   invested_close_ratio=sum(item["exposure"] for item in equity) / len(equity))
    return {"schema_version": 1, "mode": "technical", "symbol": symbol, "strategy": strategy,
            "parameters": {key: value for key, value in asdict(policy).items() if key != "name"},
            "price_basis": "pre", "return_basis": RETURN_BASIS,
            "cost_bps": float(cost_bps), "cost_units": "normalized_nav", "start": equity[0]["date"],
            "end": equity[-1]["date"], "bars": len(frame), "metrics": metrics,
            "equity_curve": equity, "fills": fills, "signals": signals,
            "warmup": {"bars": 0 if warmup is None else len(warmup), "excluded_from_performance": True},
            "assumptions": list(ASSUMPTIONS)}
