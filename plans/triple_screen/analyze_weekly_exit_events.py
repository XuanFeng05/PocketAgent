"""Offline event diagnostics for actual weekly-filter exits, not a new strategy.

Example: python plans/triple_screen/analyze_weekly_exit_events.py
  --study development=runtime_layer/reports/.../study_ID
  --study validation=runtime_layer/reports/.../study_ID
  --output runtime_layer/reports/annual_weekly_events/events.json

All horizons count the actual exit date as bar 1. Future outcomes are explicitly
ex post diagnostics. Grouping features use only the preceding close and the
recorded decision available before the exit open. No thresholds are optimized.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import pandas as pd


DESIGN = {
    "scope": "Exploratory diagnostics at actual full-strategy weekly exits; no parameter selection or deployable strategy claim.",
    "selection": "Only closed trades with exit_reason=weekly_filter_invalidated and exit_phase=open.",
    "horizons": [5, 10, 20],
    "clock": "Calendar trading-session bars, including explicit non-tradeable bars; exit date is bar 1.",
    "features": "Execution previous-close unrealized price return, recorded weekly histogram sign, trailing 20-session return, SMA20 distance, known holding bars <=5 versus >5.",
    "cutoff": "Never use a bar later than each source result's evaluation end. Strict complete-20-bar results are the primary comparison. Incomplete calendar windows are separately retained; an already executed stop is resolved, whereas an open end mark is unresolved for the intended 20-bar outcome.",
    "counterfactual": "At actual exit open retain the same fixed initial stop for at most 20 bars. Gap <=stop exits at open; low <=stop exits at stop. A non-tradeable stop touch latches until next tradable open. No reentry, weekly exit or trailing stop.",
    "cost_basis": "Forward price returns relative to actual exit price, not portfolio NAV. When both paths eventually sell, equal proportional outgoing cost cancels in their relative value. Horizon close marks are not forced executable fills and do not imply liquidation fees.",
    "priority": "Core pending weekly exit takes precedence over same-open gap stop. Such overlaps are separately identified; stop-only incremental return is exactly zero.",
    "limits": "Events overlap and cluster by stock; results are descriptive, not independent observations, significance tests, or sums of returns. The chosen events come from the full strategy; this is not a replay from original entry.",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified(path, expected):
    digest = sha(path)
    if digest != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return digest


def tradeable(row):
    flag = row.get("is_tradeable", True)
    if isinstance(flag, str):
        if flag.lower() not in {"true", "false"}:
            raise ValueError("Invalid explicit is_tradeable flag")
        flag = flag.lower() == "true"
    if flag not in (True, False):
        raise ValueError("Invalid explicit is_tradeable flag")
    return bool(flag) and float(row["volume"]) > 0


def stopped_forward(rows, exit_price, stop, horizon=20):
    """Hold an already-owned position; return a price diagnostic, not a fill ledger."""
    if not rows or horizon < 1 or exit_price <= 0 or stop <= 0:
        raise ValueError("Invalid forward-path inputs")
    selected = rows[:horizon]
    pending = False
    for count, row in enumerate(selected, 1):
        opening, low = float(row["open"]), float(row["low"])
        if tradeable(row):
            if pending:
                price, reason = opening, "deferred_stop_open"
            elif opening <= stop:
                price, reason = opening, "gap_stop"
            elif low <= stop:
                price, reason = stop, "initial_stop"
            else:
                continue
            return {"return": price / exit_price - 1, "price": price,
                    "date": row["date"], "bars_to_outcome": count,
                    "reason": reason, "closed": True, "pending_stop": False,
                    "complete_horizon": len(rows) >= horizon, "resolved_outcome": True}
        elif low <= stop:
            pending = True
    last = selected[-1]
    return {"return": float(last["close"]) / exit_price - 1,
            "price": float(last["close"]), "date": last["date"],
            "bars_to_outcome": len(selected), "reason": "horizon_mark" if len(rows) >= horizon else "evaluation_end_mark",
            "closed": False, "pending_stop": pending, "complete_horizon": len(rows) >= horizon,
            "resolved_outcome": len(rows) >= horizon}


def known_features(rows, exit_index, trade, decision):
    if exit_index < 21:
        raise ValueError("Insufficient pre-event bars")
    previous = rows[exit_index - 1]
    close = float(previous["close"])
    prior20 = float(rows[exit_index - 21]["close"])
    sma20 = mean(float(r["close"]) for r in rows[exit_index - 20:exit_index])
    entry_index = next(i for i in range(exit_index) if rows[i]["date"] == trade["entry_date"])
    observed_time = pd.Timestamp(decision["decision_time"])
    if observed_time.date().isoformat() != previous["date"] or observed_time >= pd.Timestamp(rows[exit_index]["date"]):
        raise ValueError("Decision is not available before the event open")
    if pd.Timestamp(decision["weekly_available_at"]) > observed_time:
        raise ValueError("Weekly feature was not available")
    output = {"feature_date": previous["date"], "decision_time": decision["decision_time"],
              "previous_close": close, "unrealized_price_return": close / trade["adjusted_entry_price"] - 1,
              "weekly_hist": float(decision["weekly_hist"]), "weekly_up": bool(decision["weekly_up"]),
              "weekly_available_at": decision["weekly_available_at"],
              "momentum20": close / prior20 - 1, "sma20_distance": close / sma20 - 1,
              "known_holding_bars": exit_index - entry_index}
    output["groups"] = {name: "negative" if output[name] < 0 else "nonnegative"
                        for name in ("unrealized_price_return", "weekly_hist", "momentum20", "sma20_distance")}
    output["groups"]["holding_bars"] = "up_to_5" if output["known_holding_bars"] <= 5 else "above_5"
    return output


def load_events(label, root):
    root = root.resolve()
    manifest = read(root / "artifacts.json")["sha256"]
    case_file = root / "case_results" / "primary.json"
    verified(case_file, manifest["case_results/primary.json"])
    case = read(case_file)
    if case["status"] != "completed" or case["variant"] != "full":
        raise ValueError("Event research requires completed full primary")
    child = root / "cases" / "primary" / case["child_run_id"]
    child_manifest = child / "artifacts.json"
    verified(child_manifest, manifest[child_manifest.relative_to(root).as_posix()])
    child_hashes = read(child_manifest)["sha256"]
    verified(child / "config.json", child_hashes["config.json"])
    calendar_sha = verified(child / "calendar.csv", child_hashes["calendar.csv"])
    config = read(child / "config.json")
    if config["configuration"]["settings"]["version"] != "triple_screen_pre_v1":
        raise ValueError("This event diagnostic is specifically CN")
    events, source_files = [], {}
    for outcome in case["outcomes"]:
        if outcome["status"] != "completed":
            raise ValueError("Incomplete task")
        input_rel = f"inputs/{outcome['symbol'].replace('.', '_')}.csv"
        data_path, result_path = child / input_rel, child / outcome["artifact_path"]
        input_sha = verified(data_path, child_hashes[input_rel])
        result_sha = verified(result_path, child_hashes[outcome["artifact_path"]])
        if input_sha != outcome["data_sha256"] or calendar_sha != outcome["calendar_sha256"]:
            raise ValueError("Outcome input identity mismatch")
        source_files[input_rel] = input_sha
        source_files[outcome["artifact_path"]] = result_sha
        strategy = read(result_path)["strategy"]
        if strategy["price_basis"] != "pre" or strategy["research_variant"] != "full":
            raise ValueError("Wrong source price basis or variant")
        frame = pd.read_csv(data_path, dtype={"symbol": str})
        frame["date"] = pd.to_datetime(frame["datetime"]).dt.date.astype(str)
        # Input snapshots can extend beyond the evaluation end: explicitly cut.
        frame = frame.loc[frame["date"] <= strategy["end"]]
        if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
            raise ValueError("Unordered source dates")
        rows = frame.to_dict("records")
        indices = {r["date"]: i for i, r in enumerate(rows)}
        decisions = {r["date"]: r for r in strategy["decisions"]}
        for trade in strategy["trades"]:
            if trade.get("exit_reason") != "weekly_filter_invalidated":
                continue
            if trade["status"] != "closed" or trade["exit_phase"] != "open":
                raise ValueError("Unexpected weekly exit contract")
            date = trade["exit_date"]
            index = indices[date]
            price, stop = float(trade["adjusted_exit_price"]), float(trade["initial_stop"])
            if price != float(rows[index]["open"]) or not tradeable(rows[index]):
                raise ValueError("Weekly exit does not match tradable open")
            features = known_features(rows, index, trade, decisions[date])
            pending_days = [d for d, value in decisions.items()
                            if trade["entry_date"] <= d < date and value.get("pending_exit_reason") == "weekly_filter_invalidated"]
            forward = rows[index:]
            horizons = {}
            for h in DESIGN["horizons"]:
                horizons[str(h)] = {"complete": len(forward) >= h,
                    "date": forward[h - 1]["date"] if len(forward) >= h else None,
                    "return": float(forward[h - 1]["close"]) / price - 1 if len(forward) >= h else None}
            mechanical = stopped_forward(forward, price, stop)
            overlap = price <= stop
            if overlap and (mechanical["return"] != 0 or mechanical["bars_to_outcome"] != 1):
                raise ValueError("A same-open gap stop must be redundant with weekly exit")
            events.append({"study": label, "symbol": outcome["symbol"], "year": int(date[:4]),
                           "trade_id": trade["trade_id"], "entry_date": trade["entry_date"], "exit_date": date,
                           "evaluation_start": strategy["start"], "evaluation_end": strategy["end"],
                           "exit_price": price, "initial_stop": stop,
                           "same_open_gap_stop_overlap": overlap,
                           "weekly_pending_since": min(pending_days) if pending_days else date,
                           "deferred_weekly_exit": bool(pending_days),
                           "features": features, "raw_forward": horizons, "stop_only20": mechanical,
                           "data_sha256": input_sha, "calendar_sha256": outcome["calendar_sha256"], "result_sha256": result_sha})
    return events, {"study": label, "path": str(root), "primary_case_sha256": sha(case_file),
                    "child_manifest_sha256": sha(child_manifest),
                    "implementation_sha256": config["implementation_sha256"],
                    "calendar_sha256": calendar_sha,
                    "source_files_verified": len(source_files), "source_sha256": source_files}


def moments(values):
    return {"count": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "exit_helped_ratio": mean(v < 0 for v in values) if values else None,
            "continued_holding_helped_ratio": mean(v > 0 for v in values) if values else None,
            "equal_ratio": mean(v == 0 for v in values) if values else None}


def summarize(events):
    result = {"events": len(events), "symbols": len({e["symbol"] for e in events}),
              "same_open_overlap_count": sum(e["same_open_gap_stop_overlap"] for e in events),
              "deferred_weekly_exit_count": sum(e["deferred_weekly_exit"] for e in events),
              "raw_forward": {str(h): moments([e["raw_forward"][str(h)]["return"] for e in events if e["raw_forward"][str(h)]["complete"]]) for h in DESIGN["horizons"]}}
    complete = [e for e in events if e["stop_only20"]["complete_horizon"]]
    nonoverlap = [e for e in complete if not e["same_open_gap_stop_overlap"]]
    result["stop_only20_complete"] = moments([e["stop_only20"]["return"] for e in complete])
    result["stop_only20_complete_nonoverlap"] = moments([e["stop_only20"]["return"] for e in nonoverlap])
    incomplete = [e for e in events if not e["stop_only20"]["complete_horizon"]]
    result["stop_only20_incomplete_calendar_horizon"] = moments([e["stop_only20"]["return"] for e in incomplete])
    result["stop_only20_incomplete_but_closed"] = moments([e["stop_only20"]["return"] for e in incomplete if e["stop_only20"]["closed"]])
    result["stop_only20_unresolved_end_marks"] = moments([e["stop_only20"]["return"] for e in incomplete if not e["stop_only20"]["closed"]])
    result["stop_only20_resolved"] = moments([e["stop_only20"]["return"] for e in events if e["stop_only20"]["resolved_outcome"]])
    result["stop_only20_complete_outcomes"] = dict(Counter(e["stop_only20"]["reason"] for e in complete))
    by_stock = defaultdict(list)
    for event in complete:
        by_stock[event["symbol"]].append(event["stop_only20"]["return"])
    result["stock_mean_stop_only20_complete"] = moments([mean(values) for values in by_stock.values()])
    return result


def self_check():
    def bar(day, opening=10., low=9., close=11., volume=1):
        return {"date": f"2025-01-{day:02}", "open": opening, "low": low, "close": close, "volume": volume}
    checks = []
    result = stopped_forward([bar(1, opening=7, low=6)], 7, 8)
    assert result["return"] == 0 and result["reason"] == "gap_stop"
    assert result["resolved_outcome"] and not result["complete_horizon"]
    checks.append("same-open-gap-stop-is-exactly-zero")
    result = stopped_forward([bar(1, low=7)], 10, 8)
    assert math.isclose(result["return"], -.2) and result["reason"] == "initial_stop"
    checks.append("same-day-low-stop-after-retaining-existing-position")
    result = stopped_forward([bar(1, low=7, volume=0), bar(2, opening=12, low=11, close=13)], 10, 8)
    assert math.isclose(result["return"], .2) and result["reason"] == "deferred_stop_open"
    checks.append("nontradable-stop-latches-even-after-price-rebound")
    result = stopped_forward([bar(1), bar(2, close=20)], 10, 8, horizon=1)
    assert math.isclose(result["return"], .1) and result["date"] == "2025-01-01"
    checks.append("horizon-ignores-later-price")
    result = stopped_forward([bar(1)], 10, 8)
    assert not result["complete_horizon"] and result["reason"] == "evaluation_end_mark"
    assert not result["resolved_outcome"]
    checks.append("evaluation-end-censoring-is-explicit")
    dates = pd.bdate_range("2025-01-01", periods=30)
    rows = [{**bar(1), "date": d.date().isoformat()} for d in dates]
    trade = {"entry_date": rows[20]["date"], "adjusted_entry_price": 10}
    decision = {"decision_time": rows[24]["date"]+"T15:00:00", "weekly_available_at": rows[23]["date"]+"T15:00:00", "weekly_hist": -.1, "weekly_up": False}
    before = known_features(rows, 25, trade, decision)
    rows[25]["close"], rows[26]["close"] = 9999, .01
    assert known_features(rows, 25, trade, decision) == before
    checks.append("grouping-features-ignore-exit-day-and-future-prices")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", default=[], help="Label=study-directory; may repeat")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    checks = self_check()
    if args.self_check and not args.study:
        print(json.dumps({"passed": len(checks), "checks": checks}, indent=2))
        return
    if not args.study or args.output is None:
        parser.error("Provide at least one --study and --output")
    events, sources = [], []
    labels = set()
    for raw in args.study:
        label, path = raw.split("=", 1)
        if label in labels:
            raise ValueError("Duplicate study label")
        labels.add(label)
        rows, source = load_events(label, Path(path))
        events.extend(rows)
        sources.append(source)
    # Do not silently count the same stock/date twice when source windows overlap.
    identities = [(e["symbol"], e["exit_date"]) for e in events]
    if len(identities) != len(set(identities)):
        raise ValueError("Overlapping sources duplicate actual exit events; analyze separately")
    groups = []
    for year in sorted({e["year"] for e in events}):
        annual = [e for e in events if e["year"] == year]
        groups.append({"year": year, "feature": "all", "group": "all", **summarize(annual)})
        for feature in ("unrealized_price_return", "weekly_hist", "momentum20", "sma20_distance", "holding_bars"):
            for group in sorted({e["features"]["groups"][feature] for e in annual}):
                selected = [e for e in annual if e["features"]["groups"][feature] == group]
                groups.append({"year": year, "feature": feature, "group": group, **summarize(selected)})
    result = {"design": DESIGN, "script_sha256": sha(Path(__file__)), "self_checks": checks,
              "sources": sources, "overall": summarize(events), "annual_groups": groups, "events": events}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output), "events": len(events), "overall": result["overall"],
                      "annual_groups": [g for g in groups if g["feature"] == "all"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
