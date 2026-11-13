"""Daily point-in-time benchmark and industry context.

Inputs use exchange-local naive timestamps. A daily close cannot be published
before 15:00 on its date. Membership intervals are [valid_from, valid_to), and
known_at controls when an assignment may be used. Revisions need a separate
versioned dataset: duplicate observations and overlapping assignments fail.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, time
from typing import Any

import numpy as np
import pandas as pd


def _close_time(day: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day.date(), time(15)))


def _frame(value: Any, name: str) -> pd.DataFrame:
    if isinstance(value, (str, Path)):
        return pd.read_csv(value, dtype={"symbol": str, "industry": str})
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a DataFrame, records, or CSV path") from exc


def _timestamps(values, name: str, *, nullable: bool = False) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} needs valid exchange-local timestamps") from exc
    if not nullable and parsed.isna().any():
        raise ValueError(f"{name} cannot be missing")
    if parsed.dt.tz is not None:
        raise ValueError(f"{name} must be exchange-local timestamps without timezone offsets")
    return parsed


def _prices(value: Any, key: str, name: str) -> dict[str, pd.DataFrame]:
    frame = _frame(value, name)
    if frame.empty:
        return {}
    if "date" not in frame and "datetime" in frame:
        frame = frame.rename(columns={"datetime": "date"})
    required = {key, "date", "available_at", "close"}
    if not required <= set(frame):
        raise ValueError(f"{name} missing columns: {sorted(required-set(frame))}")
    if "freq" in frame and not frame["freq"].astype(str).eq("daily").all():
        raise ValueError(f"{name} requires daily observations")
    if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}.{key} cannot be missing")
    frame[key] = frame[key].astype(str).str.strip().str.upper()
    frame["date"] = _timestamps(frame["date"], f"{name}.date").dt.normalize()
    frame["available_at"] = _timestamps(frame["available_at"], f"{name}.available_at")
    if (frame["available_at"] < frame["date"].map(_close_time)).any():
        raise ValueError(f"{name} daily close cannot be available before its 15:00 close")
    if frame.duplicated([key, "date"]).any():
        raise ValueError(f"{name} has duplicate {key}/date observations")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if not np.isfinite(frame["close"]).all() or (frame["close"] <= 0).any():
        raise ValueError(f"{name}.close must be finite and positive")
    if "amount" in frame:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="raise")
        if not np.isfinite(frame["amount"]).all() or (frame["amount"] < 0).any():
            raise ValueError(f"{name}.amount must be finite and nonnegative")
    return {str(group): rows.sort_values("date").reset_index(drop=True)
            for group, rows in frame.groupby(key, sort=True)}


def _memberships(value: Any) -> dict[str, pd.DataFrame]:
    frame = _frame(value, "membership")
    if frame.empty:
        return {}
    required = {"symbol", "industry", "valid_from", "valid_to", "known_at"}
    if not required <= set(frame):
        raise ValueError(f"membership missing columns: {sorted(required-set(frame))}")
    for key in ("symbol", "industry"):
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"membership.{key} cannot be missing")
        frame[key] = frame[key].astype(str).str.strip().str.upper()
    frame["valid_from"] = _timestamps(frame["valid_from"], "membership.valid_from").dt.normalize()
    frame["valid_to"] = _timestamps(frame["valid_to"], "membership.valid_to", nullable=True).dt.normalize().fillna(pd.Timestamp("2262-04-11"))
    frame["known_at"] = _timestamps(frame["known_at"], "membership.known_at")
    if (frame["valid_from"] >= frame["valid_to"]).any():
        raise ValueError("membership valid_from must precede exclusive valid_to")
    result = {}
    for symbol, rows in frame.groupby("symbol", sort=True):
        rows = rows.sort_values("valid_from").reset_index(drop=True)
        if len(rows) > 1 and (rows["valid_from"].iloc[1:].to_numpy() < rows["valid_to"].iloc[:-1].to_numpy()).any():
            raise ValueError(f"Overlapping historical industry memberships for {symbol}")
        result[str(symbol)] = rows
    return result


def _snapshot(frame: pd.DataFrame | None, day: pd.Timestamp) -> dict:
    empty = {"available": False, "observed_date": None, "ret_1bar": None,
             "ret_5bar": None, "ma20bar_distance": None, "amount_ratio20bar": None,
             "previous_1bar_date": None, "previous_5bar_date": None}
    if frame is None or frame.empty:
        return empty
    cutoff = _close_time(day)
    visible = frame.loc[(frame["date"] <= day) & (frame["available_at"] <= cutoff)]
    if visible.empty:
        return empty
    latest = visible.iloc[-1]
    empty["observed_date"] = latest["date"].date().isoformat()
    if latest["date"] != day:
        return empty  # A stale observation is explicit, not a current-market feature.
    result = {**empty, "available": True}
    for lag in (1, 5):
        if len(visible) > lag:
            previous = visible.iloc[-lag-1]
            result[f"ret_{lag}bar"] = float(latest["close"] / previous["close"] - 1)
            result[f"previous_{lag}bar_date"] = previous["date"].date().isoformat()
    if len(visible) >= 20:
        history = visible.iloc[-20:]
        result["ma20bar_distance"] = float(latest["close"] / history["close"].mean() - 1)
        if "amount" in history and history["amount"].mean() > 0:
            result["amount_ratio20bar"] = float(latest["amount"] / history["amount"].mean())
    return result


def enrich_daily_bars(
    bars: Any, benchmark: Any = None, industry: Any = None, membership: Any = None,
    *, benchmark_symbol: str | None = None, require_complete: bool = False,
) -> pd.DataFrame:
    """Add causal context_* columns without changing or writing the input.

    Returns named ``1bar``/``5bar`` and ``ma20bar`` refer to observed daily
    bars, not calendar days. Excess returns are only present when the stock
    and reference actually share the same start and end dates.
    """
    frame = _frame(bars, "bars")
    if "datetime" not in frame and "date" in frame:
        frame = frame.rename(columns={"date": "datetime"})
    if not {"datetime", "symbol", "close"} <= set(frame) or frame.empty:
        raise ValueError("Context needs nonempty bars with datetime/date, symbol and close")
    if not isinstance(require_complete, bool):
        raise ValueError("require_complete must be boolean")
    frame["datetime"] = _timestamps(frame["datetime"], "bars.datetime").dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if frame["symbol"].isin(["", "NAN", "NONE"]).any() or frame.duplicated(["symbol", "datetime"]).any():
        raise ValueError("Bars require nonempty symbols and unique symbol/dates")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if not np.isfinite(frame["close"]).all() or (frame["close"] <= 0).any():
        raise ValueError("bars.close must be finite and positive")
    index_series = _prices(benchmark, "symbol", "benchmark")
    industry_series = _prices(industry, "industry", "industry")
    assignments = _memberships(membership)
    if bool(industry_series) != bool(assignments):
        raise ValueError("Industry context requires both industry prices and historical membership")
    if benchmark_symbol is None:
        if len(index_series) > 1:
            raise ValueError("Choose benchmark_symbol when more than one benchmark is supplied")
        benchmark_symbol = next(iter(index_series), None)
    else:
        benchmark_symbol = str(benchmark_symbol).strip().upper()
        if benchmark_symbol not in index_series:
            raise ValueError("Requested benchmark_symbol has no input observations")
    if not index_series and not industry_series:
        raise ValueError("Provide benchmark and/or industry context sources")
    # Cache reference snapshots across stock symbols on the same date.
    cache: dict[tuple[str, str, pd.Timestamp], dict] = {}
    output = []
    for symbol, stock in frame.groupby("symbol", sort=False):
        stock = stock.sort_values("datetime")
        stock_rows = stock.to_dict("records")
        for i, row in enumerate(stock_rows):
            day = row["datetime"]
            decision = _close_time(day)
            selected_industry = None
            member = assignments.get(symbol)
            if member is not None:
                eligible = member.loc[(member["valid_from"] <= day) & (day < member["valid_to"]) & (member["known_at"] <= decision)]
                if not eligible.empty:
                    selected_industry = str(eligible.iloc[0]["industry"])
            contexts = {}
            for prefix, identifier, sources in (("index", benchmark_symbol, index_series), ("industry", selected_industry, industry_series)):
                cache_key = (prefix, str(identifier), day)
                if cache_key not in cache:
                    cache[cache_key] = _snapshot(sources.get(identifier), day)
                contexts[prefix] = cache[cache_key]
            enriched = {**row, "context_decision_time": decision.isoformat(),
                        "context_benchmark_symbol": benchmark_symbol, "context_industry": selected_industry}
            for prefix, context in contexts.items():
                enriched[f"context_{prefix}_available"] = context["available"]
                enriched[f"context_{prefix}_observed_date"] = context["observed_date"]
                for feature in ("ret_1bar", "ret_5bar", "ma20bar_distance", "amount_ratio20bar"):
                    enriched[f"context_{prefix}_{feature}"] = context[feature]
                for lag in (1, 5):
                    relative = None
                    if i >= lag and context[f"ret_{lag}bar"] is not None:
                        prior = stock_rows[i-lag]
                        if prior["datetime"].date().isoformat() == context[f"previous_{lag}bar_date"]:
                            relative = float(row["close"] / prior["close"] - 1 - context[f"ret_{lag}bar"])
                    enriched[f"context_stock_minus_{prefix}_ret_{lag}bar"] = relative
            if require_complete and ((index_series and not contexts["index"]["available"]) or
                                     (industry_series and not contexts["industry"]["available"])):
                raise ValueError(f"Incomplete point-in-time context for {symbol} on {day.date()}")
            output.append(enriched)
    return pd.DataFrame(output).sort_values(["datetime", "symbol"], kind="stable").reset_index(drop=True)


def demo_context_sources(bars: Any) -> dict[str, pd.DataFrame]:
    """Explicitly synthetic context for demonstrations, never inferred real data."""
    frame = _frame(bars, "bars")
    dates = pd.to_datetime(frame["datetime"] if "datetime" in frame else frame["date"]).dt.normalize().drop_duplicates().sort_values()
    index_rows, industry_rows, members = [], [], []
    for i, day in enumerate(dates):
        index_rows.append({"symbol": "DEMO_INDEX", "date": day, "available_at": _close_time(day),
                           "close": 100 + .2*i + 3*np.sin(i/10), "amount": 100_000_000 + i*1000})
        industry_rows.append({"industry": "DEMO_INDUSTRY", "date": day, "available_at": _close_time(day),
                              "close": 100 + .25*i + 4*np.sin(i/11)})
    for symbol in frame["symbol"].astype(str).unique():
        members.append({"symbol": symbol, "industry": "DEMO_INDUSTRY", "valid_from": dates.iloc[0],
                        "valid_to": None, "known_at": dates.iloc[0]})
    return {"benchmark": pd.DataFrame(index_rows), "industry": pd.DataFrame(industry_rows), "membership": pd.DataFrame(members)}
