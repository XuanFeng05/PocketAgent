"""Forward-event study for single and conditional technical signals.

These are overlapping hypothetical events, not a portfolio or independent
trades. Signals are observed at a session close, entered at the next available
tradable session open, and exited at the close of a predeclared session horizon.
No stop or target is imposed: this isolates horizon response from exit rules.
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd


HORIZONS = (1, 3, 5, 10, 20, 40)
REASONS = ("valid", "not_ready", "price_ineligible_signal", "untradeable_signal", "no_entry_in_period",
           "insufficient_forward_window", "no_tradeable_exit_in_period", "invalid_price")


def _dates(values):
    result = pd.DatetimeIndex(pd.to_datetime(values))
    if result.tz is not None or not result.equals(result.normalize()):
        raise ValueError("Session dates must be timezone-naive local midnights")
    if not result.is_monotonic_increasing or result.has_duplicates:
        raise ValueError("Session dates must be sorted and unique")
    return result


def _boolean(values, length, name):
    array = np.asarray(values)
    if array.shape != (length,):
        raise ValueError(f"{name} must contain one flag per feature row")
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(bool, copy=False)
    if np.issubdtype(array.dtype, np.number) and np.isfinite(array).all() and np.isin(array, [0, 1]).all():
        return array.astype(bool)
    raise ValueError(f"{name} must contain explicit boolean or finite0/1 flags")


def prepare_outcomes(features: pd.DataFrame, start, end, horizons=HORIZONS,
                     cost_bps: float = 10.) -> dict:
    """Return per-signal-date forward labels, with exclusive exclusion reasons.

    Every supplied row remains a session, including valid zero-volume suspended
    observations. A suspended next session delays entry; a suspended scheduled
    exit delays exit. Neither delay may cross the evaluation period boundary.
    Missing market sessions must already have been rejected by feature creation.
    """
    f = features
    required = {"datetime", "open", "high", "low", "close", "ready", "tradeable"}
    if required-set(f):
        raise ValueError(f"Missing event fields: {sorted(required-set(f))}")
    dates = _dates(f.datetime)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError("Evaluation start exceeds end")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be finite and in [0,10000)")
    horizons = tuple(horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(
            isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1 for h in horizons):
        raise ValueError("Horizons must be unique positive integer session counts")
    n = len(f)
    if n == 0:
        raise ValueError("No feature rows")
    inside = np.asarray((dates >= start) & (dates <= end))
    indices = np.flatnonzero(inside)
    scope = inside.copy()
    # Previous close may place an order on the first evaluation session.
    if len(indices) and indices[0] > 0:
        scope[indices[0]-1] = True
    ready = _boolean(f.ready, n, "ready")
    price_eligible = _boolean(f.universe_trade_eligible, n, "universe_trade_eligible") if "universe_trade_eligible" in f else np.ones(n, dtype=bool)
    tradeable = _boolean(f.tradeable, n, "tradeable")
    o, high, low, close = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    price_ok = (np.isfinite(o) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
                & (low > 0) & (low <= np.minimum(o, close))
                & (high >= np.maximum(o, close)) & (high >= low))
    next_tradable = np.full(n+1, n, dtype=np.int32)
    for i in range(n-1, -1, -1):
        next_tradable[i] = i if tradeable[i] else next_tradable[i+1]
    signal_index = np.arange(n, dtype=np.int32)
    entry = next_tradable[signal_index+1]
    safe_entry = np.minimum(entry, n-1)
    entry_in_period = (entry < n) & inside[safe_entry]
    fee = cost_bps/10000
    result = {"dates": dates, "scope": scope, "horizons": horizons,
              "entry_index": entry, "entry_delay_sessions": np.maximum(0, entry-signal_index-1),
              "outcomes": {}}
    for horizon in horizons:
        planned_exit = entry+horizon-1
        safe_planned = np.minimum(planned_exit, n)
        exit_index = next_tradable[safe_planned]
        safe_exit = np.minimum(exit_index, n-1)
        planned_in_period = (planned_exit < n) & inside[np.minimum(planned_exit, n-1)]
        exit_in_period = (exit_index < n) & inside[safe_exit]
        reason = np.full(n, "valid", dtype=object)
        reason[~ready] = "not_ready"
        reason[ready & ~price_eligible] = "price_ineligible_signal"
        reason[ready & price_eligible & ~tradeable] = "untradeable_signal"
        candidate = ready & price_eligible & tradeable
        reason[candidate & ~entry_in_period] = "no_entry_in_period"
        candidate &= entry_in_period
        reason[candidate & ~planned_in_period] = "insufficient_forward_window"
        candidate &= planned_in_period
        reason[candidate & ~exit_in_period] = "no_tradeable_exit_in_period"
        candidate &= exit_in_period
        # Every observed holding bar must be valid; a bad bar is not repaired by
        # deleting it, and final-window extrema never leak into an earlier label.
        bad_prefix = np.r_[0, np.cumsum(~price_ok)]
        holding_bad = (bad_prefix[safe_exit+1]-bad_prefix[safe_entry]) > 0
        invalid = candidate & (holding_bad | ~price_ok)
        reason[invalid] = "invalid_price"
        valid = scope & (reason == "valid")
        returns = np.full(n, np.nan)
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        holding = np.zeros(n, dtype=np.int32)
        idx = np.flatnonzero(valid)
        returns[idx] = close[exit_index[idx]]*(1-fee)/(o[entry[idx]]*(1+fee))-1
        returns[idx[np.abs(returns[idx]) < 1e-12]] = 0.
        # Adjusted-price excursions are pre-fee diagnostics, not executable PnL.
        # Full entry-day OHLC is available here because every entry is at open.
        for j in idx:
            a, b = entry[j], exit_index[j]+1
            mfe[j] = high[a:b].max()/o[a]-1
            mae[j] = low[a:b].min()/o[a]-1
            holding[j] = b-a
        result["outcomes"][int(horizon)] = {
            "valid": valid, "reason": reason, "net_return": returns,
            "mfe": mfe, "mae": mae, "exit_index": exit_index,
            "exit_delay_sessions": np.maximum(0, exit_index-planned_exit),
            "holding_sessions": holding,
        }
    return result


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else None


def _divided_sum(values, valid, axis=0):
    numerator = np.where(valid, values, 0.).sum(axis=axis)
    denominator = valid.sum(axis=axis)
    result = np.full(np.shape(numerator), np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _summary(returns, mae, mfe):
    count = len(returns)
    if not count:
        return {k: None for k in ("net_win_rate", "net_nonloss_rate", "mean_net_return",
                "median_net_return", "average_win", "average_loss", "payoff_ratio",
                "profit_factor", "return_q05", "expected_shortfall_05", "mean_mfe",
                "mean_mae", "worst_mae", "best_mfe")}
    wins, losses = returns[returns > 0], returns[returns < 0]
    q05 = float(np.quantile(returns, .05))
    avgwin, avgloss = _safe_mean(wins), _safe_mean(losses)
    return {"net_win_rate": len(wins)/count, "net_nonloss_rate": np.mean(returns >= 0).item(),
            "mean_net_return": float(returns.mean()), "median_net_return": float(np.median(returns)),
            "average_win": avgwin, "average_loss": avgloss,
            "payoff_ratio": avgwin/-avgloss if avgwin is not None and avgloss is not None else None,
            "profit_factor": float(wins.sum()/-losses.sum()) if len(losses) else None,
            "return_q05": q05, "expected_shortfall_05": float(returns[returns <= q05].mean()),
            "mean_mfe": _safe_mean(mfe), "mean_mae": _safe_mean(mae),
            "worst_mae": float(np.min(mae)), "best_mfe": float(np.max(mfe))}


def _default_masks(features, pairs, include_weekly, include_regimes):
    from research_layer.multiscale_signals import (
        signal_masks, weekly_signal_masks, weekly_filter_masks, condition_masks)
    daily = signal_masks(features)
    if pairs:
        originals = list(daily)
        daily = {**daily, **{f"{a}&{b}": daily[a] & daily[b] for a, b in combinations(originals, 2)}}
    eligible = features.ready.to_numpy(dtype=bool) & features.universe_trade_eligible.to_numpy(dtype=bool)
    scales = {"daily": {"unconditional": eligible, **daily}}
    if include_weekly:
        scales["weekly"] = {"unconditional": eligible
                             & features.weekly_published.to_numpy(dtype=bool),
                             **weekly_signal_masks(features)}
    conditions = {"weekly:"+k: v for k, v in weekly_filter_masks(features).items()}
    if include_regimes:
        conditions.update({"regime:"+k: v for k, v in condition_masks(features).items() if k != "none"})
    return scales, conditions


def run_event_study(features_by_symbol: Mapping[str, pd.DataFrame], start, end,
                    horizons=HORIZONS, cost_bps=10., pair_signals=False,
                    include_weekly_signals=True, include_regimes=True,
                    include_date_aggregates=False, mask_builder=None) -> dict:
    """Pool fixed-universe events with same-date conditional control outcomes.

    `mask_builder(frame)` may supply `(scales, conditions)` for an explicitly
    registered subset or tests. Each is a dictionary of length-frame booleans;
    scales maps e.g. daily/weekly to named signal dictionaries. Keys must agree
    between all symbols. Default technical masks already include readiness.
    No automatic choice or ranking of indicators is made by this function.
    """
    if not features_by_symbol:
        raise ValueError("A nonempty fixed symbol universe is required")
    symbols = list(features_by_symbol)
    prepared, masks, eligibility = {}, {}, []
    date_set = set()
    expected_keys = None
    for symbol, frame in features_by_symbol.items():
        out = prepare_outcomes(frame, start, end, horizons, cost_bps)
        prepared[symbol] = out
        built = mask_builder(frame) if mask_builder else _default_masks(
            frame, pair_signals, include_weekly_signals, include_regimes)
        scales, conditions = built
        keys = ({scale: tuple(signals) for scale, signals in scales.items()}, tuple(conditions))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"Signal/condition keys differ for {symbol}")
        scales = {scale: {name: _boolean(values, len(frame), f"{symbol}/{scale}/{name}")
                          for name, values in signals.items()} for scale, signals in scales.items()}
        conditions = {name: _boolean(values, len(frame), f"{symbol}/{name}") for name, values in conditions.items()}
        masks[symbol] = (scales, conditions)
        date_set.update(out["dates"][out["scope"]])
        ready = frame.ready.to_numpy(dtype=bool)
        price_eligible = frame.universe_trade_eligible.to_numpy(dtype=bool) if "universe_trade_eligible" in frame else np.ones(len(frame), dtype=bool)
        ready_dates = out["dates"][ready]
        eligibility.append({"symbol": symbol, "feature_rows": len(frame),
            "first_ready_signal_date": ready_dates[0].strftime("%Y-%m-%d") if len(ready_dates) else None,
            "eligible_signal_sessions_in_scope": int((ready & out["scope"]).sum()),
            "warmup_excluded_sessions_in_scope": int((~ready & out["scope"]).sum()),
            "historical_price_excluded_sessions_in_scope": int((ready & ~price_eligible & out["scope"]).sum()),
            "no_evaluable_events": not any(x["valid"].any() for x in out["outcomes"].values())})
    dates = pd.DatetimeIndex(sorted(date_set))
    if not len(dates):
        raise ValueError("No signal dates intersect the evaluation period")
    shape = (len(symbols), len(dates))
    position = {s: dates.get_indexer(prepared[s]["dates"][prepared[s]["scope"]]) for s in symbols}
    scope_panel = np.zeros(shape, dtype=bool)
    ready_panel = np.zeros(shape, dtype=bool)
    signal_panels = {scale: {k: np.zeros(shape, dtype=bool) for k in ids}
                     for scale, ids in expected_keys[0].items()}
    condition_panels = {k: np.zeros(shape, dtype=bool) for k in expected_keys[1]}
    panels = {h: {"net_return": np.full(shape, np.nan), "mfe": np.full(shape, np.nan),
                  "mae": np.full(shape, np.nan), "valid": np.zeros(shape, dtype=bool),
                  "reason": np.full(shape, "outside_symbol_scope", dtype=object),
                  "entry_delay": np.zeros(shape, dtype=np.int32),
                  "exit_delay": np.zeros(shape, dtype=np.int32),
                  "holding_sessions": np.zeros(shape, dtype=np.int32)} for h in horizons}
    for si, symbol in enumerate(symbols):
        out = prepared[symbol]; idx = position[symbol]; chosen = out["scope"]
        scope_panel[si, idx] = True
        ready_panel[si, idx] = features_by_symbol[symbol].ready.to_numpy(dtype=bool)[chosen]
        for scale, signals in masks[symbol][0].items():
            for name, values in signals.items():
                signal_panels[scale][name][si, idx] = np.asarray(values, dtype=bool)[chosen]
        for name, values in masks[symbol][1].items():
            condition_panels[name][si, idx] = np.asarray(values, dtype=bool)[chosen]
        for horizon in horizons:
            source = out["outcomes"][horizon]
            for name in ("net_return", "mfe", "mae", "valid", "reason", "holding_sessions"):
                panels[horizon][name][si, idx] = source[name][chosen]
            panels[horizon]["entry_delay"][si, idx] = out["entry_delay_sessions"][chosen]
            panels[horizon]["exit_delay"][si, idx] = source["exit_delay_sessions"][chosen]
    rows, date_rows = [], []
    for horizon in horizons:
        panel = panels[horizon]
        returns = panel["net_return"]
        universe = panel["valid"]
        universe_return = _divided_sum(returns, universe)
        universe_win = _divided_sum(returns > 0, universe)
        reason_masks = {reason: panel["reason"] == reason for reason in REASONS if reason != "valid"}
        for condition, condition_mask in condition_panels.items():
            control = universe & condition_mask
            conditional_return = _divided_sum(returns, control)
            conditional_win = _divided_sum(returns > 0, control)
            for scale, signals in signal_panels.items():
                for signal, signal_mask in signals.items():
                    triggered = signal_mask & condition_mask & scope_panel
                    valid = triggered & universe
                    count = int(valid.sum())
                    ci = np.nonzero(valid)[1]
                    observed = returns[valid]
                    case_id = f"{scale}|{signal}|{condition}|h{horizon}"
                    row = {"id": case_id, "scale": scale, "signal": signal,
                           "condition": condition, "horizon_sessions": int(horizon),
                           "raw_trigger_count": int(triggered.sum()), "event_count": count,
                           "unique_signal_dates": int(np.any(valid, axis=0).sum()),
                           "symbols_with_events": int(np.any(valid, axis=1).sum()),
                           "exclusions": {reason: int((triggered & mask).sum()) for reason, mask in reason_masks.items()},
                           **_summary(observed, panel["mae"][valid], panel["mfe"][valid]),
                           "matched_universe_mean_return": _safe_mean(universe_return[ci]),
                           "matched_universe_win_rate": _safe_mean(universe_win[ci]),
                           "matched_condition_mean_return": _safe_mean(conditional_return[ci]),
                           "matched_condition_win_rate": _safe_mean(conditional_win[ci]),
                           "mean_excess_vs_matched_universe": _safe_mean(observed-universe_return[ci]),
                           "mean_excess_vs_matched_condition": _safe_mean(observed-conditional_return[ci]),
                           "win_lift_vs_matched_universe": _safe_mean((observed > 0)-universe_win[ci]),
                           "win_lift_vs_matched_condition": _safe_mean((observed > 0)-conditional_win[ci]),
                           "delayed_entry_events": int((valid & (panel["entry_delay"] > 0)).sum()),
                           "delayed_exit_events": int((valid & (panel["exit_delay"] > 0)).sum()),
                           "mean_actual_holding_sessions": _safe_mean(panel["holding_sessions"][valid])}
                    assert row["raw_trigger_count"] == count+sum(row["exclusions"].values())
                    rows.append(row)
                    if include_date_aggregates:
                        date_rows.append({"id": case_id,
                            "event_count": valid.sum(axis=0).tolist(),
                            "net_return_sum": np.where(valid, returns, 0.).sum(axis=0).tolist(),
                            "universe_excess_sum": np.where(valid, returns-universe_return, 0.).sum(axis=0).tolist(),
                            "conditional_excess_sum": np.where(valid, returns-conditional_return, 0.).sum(axis=0).tolist(),
                            "net_win_sum": np.where(valid, returns > 0, 0.).sum(axis=0).tolist(),
                            "conditional_win_lift_sum": np.where(valid, (returns > 0)-conditional_win, 0.).sum(axis=0).tolist()})
    result = {"period": [str(start)[:10], str(end)[:10]], "symbol_count": len(symbols),
              "case_count": len(rows), "horizons_sessions": list(horizons),
              "cost_bps_per_side": cost_bps, "rows": rows, "universe_eligibility": eligibility,
              "methodology": [
                  "Signal at close; next tradable session open; scheduled horizon close, entry session counted as1.",
                  "Observed suspension rows remain in session counts; untradeable signal dates are excluded; entry/exit may delay without crossing period boundary.",
                  "Previous-period final close may initiate first-period-session entry; no event exits beyond evaluation end.",
                  "Net event returns include proportional entry and exit fees; excursions are adjusted-price, pre-fee diagnostics.",
                  "Events overlap within and across stocks. Counts are not independent trials, and this output is not a tradable portfolio or annualized strategy return.",
                  "Matched control is the equal-weight eligible stock pool on the same signal date and planned horizon, with the same fees; conditional control also satisfies the stated condition.",
                  "The event stock remains in its same-date control pool; return differences are descriptive, not factor-adjusted alpha or causal effects.",
                  "Technical signal masks include readiness; a raw trigger count is after those masks, while per-stock eligibility discloses warmup exclusions.",
                  "Signals and unconditional controls share the prior-close vendor_close>=10 universe_trade_eligible gate; vendor prices may already include split adjustment, so this is not a historical nominal-price guarantee.",
                  "Weekly signals fire only on newly completed-week publication dates; the same numeric lookback then covers weeks rather than days.",
                  "These periods are previously observed historical data; moving an indicator to weekly bars does not by itself prove a longer useful lifetime.",
              ]}
    if include_date_aggregates:
        result["date_aggregates"] = {"dates": dates.strftime("%Y-%m-%d").tolist(), "cases": date_rows}
    return result


def bootstrap_event_cases(date_aggregates: dict, selected_ids=(), metric="conditional_excess_sum",
                          repetitions=3000, block_sessions=20, seed=20261113) -> dict:
    """Synchronized date-block ratio bootstrap, preserving event overlap/clusters.

    The statistic is pooled event sum / pooled event count, not the mean of
    individual-date means. Every hypothesis passed here belongs to the adjusted
    testing family; selected_ids only controls which confidence intervals print.
    """
    from research_layer.elder_analysis import holm_adjust
    cases = date_aggregates["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or not set(selected_ids).issubset(ids):
        raise ValueError("Case IDs must be unique and selected IDs must belong")
    counts = np.asarray([case["event_count"] for case in cases], dtype=float).T
    sums = np.asarray([case[metric] for case in cases], dtype=float).T
    if counts.shape != sums.shape or counts.ndim != 2 or counts.shape[0] != len(date_aggregates["dates"]):
        raise ValueError("Date aggregate arrays must align")
    if not np.isfinite(counts).all() or not np.isfinite(sums).all() or (counts < 0).any():
        raise ValueError("Aggregate counts/sums must be finite with nonnegative counts")
    if repetitions < 99 or block_sessions < 1 or block_sessions > len(counts):
        raise ValueError("Invalid bootstrap repetition/block settings")
    total = counts.sum(axis=0)
    eligible = (total > 0) & ((counts > 0).sum(axis=0) >= block_sessions)
    point = np.divide(sums.sum(axis=0), total, out=np.full(len(ids), np.nan), where=total > 0)
    rng = np.random.default_rng(seed)
    n = len(counts); blocks = (n+block_sessions-1)//block_sessions
    estimates = np.full((repetitions, len(ids)), np.nan)
    for i in range(repetitions):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None]+np.arange(block_sessions)) % n).ravel()[:n]
        weights = np.bincount(indices, minlength=n)
        den = weights@counts
        estimates[i] = np.divide(weights@sums, den, out=np.full(len(ids), np.nan), where=den > 0)
    good = eligible & np.isfinite(estimates).all(axis=0)
    p = np.ones(len(ids))
    p[good] = (1+np.sum(estimates[:, good]-point[good] >= point[good], axis=0))/(repetitions+1)
    holm = holm_adjust(p)
    selected = set(selected_ids)
    rows = []
    for j, identifier in enumerate(ids):
        row = {"id": identifier, "events": int(total[j]), "statistically_eligible": bool(good[j]),
               "mean_effect": float(point[j]) if np.isfinite(point[j]) else None,
               "positive_effect_one_sided_p": float(p[j]), "holm_adjusted_p": float(holm[j])}
        if identifier in selected and good[j]:
            row["individual_ci95"] = np.quantile(estimates[:, j], [.025, .975]).tolist()
        rows.append(row)
    family_p = None
    if good.any():
        family_p = float((1+np.sum((estimates[:, good]-point[good]).max(axis=1) >= point[good].max()))/(repetitions+1))
    return {"metric": metric, "candidate_count": len(ids), "statistically_eligible_cases": int(good.sum()),
            "bootstrap_repetitions": repetitions, "block_sessions": block_sessions, "seed": seed,
            "candidate_results": rows, "centered_max_mean_family_p": family_p,
            "limitations": ["Retrospective event diagnostic, not portfolio alpha or a causal test.",
                            "Synchronized circular date blocks preserve same-date cross-stock clustering and approximate serial dependence; a fixed block length cannot guarantee all long-horizon dependence is captured.",
                            "Unequal event counts use a ratio statistic; rare cases with fewer active dates than the block size or empty resamples receive no positive finding.",
                            "Holm includes every supplied case, including sparse cases with p=1. Selected intervals are individual, not simultaneous confidence bounds.",
                            "No winner may be reselected using these validation results."]}
