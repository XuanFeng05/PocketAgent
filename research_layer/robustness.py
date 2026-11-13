"""Traceable research diagnostics; none of these statistics accepts a strategy."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np

from evaluation_layer.metrics.overfitting import deflated_sharpe_ratio, probability_of_backtest_overfitting


def daily_returns(report: dict) -> tuple[list[str], np.ndarray]:
    rows = report.get("equity_curve", [])
    initial = float(report["metrics"]["initial_nav"])
    dates = [str(row.get("date", row.get("time", "")))[:10] for row in rows]
    nav = np.asarray([initial, *[float(row["nav"]) for row in rows]])
    if not rows or len(set(dates)) != len(dates) or not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError("Diagnostics require unique daily dates and positive finite NAV")
    return dates, nav[1:] / nav[:-1] - 1


def _yearly(report: dict) -> list[dict]:
    dates, returns = daily_returns(report)
    groups = defaultdict(list)
    for day, value in zip(dates, returns):
        groups[day[:4]].append(float(value))
    result = []
    for year, values in sorted(groups.items()):
        curve = np.cumprod([1.0, *[1 + value for value in values]])
        result.append({"year": year, "observations": len(values),
                       "return": float(curve[-1] - 1),
                       "maximum_drawdown": float(-(curve / np.maximum.accumulate(curve) - 1).min())})
    return result


def _trade_concentration(report: dict) -> dict:
    fifo = report.get("fifo_report") or {}
    trades = fifo.get("round_trips") or fifo.get("closed_trades") or []
    values = [float(item["net_pnl"]) for item in trades]
    winners = sorted((value for value in values if value > 0), reverse=True)
    positive = sum(winners)
    return {"unit": "completed_round_trips" if fifo.get("round_trips") else "fifo_fragments",
            "count": len(values), "gross_winning_pnl": positive,
            "largest_winner_share": winners[0] / positive if positive else None,
            "top_three_winner_share": sum(winners[:3]) / positive if positive else None,
            "net_pnl_excluding_largest_winner": sum(values) - (winners[0] if winners else 0),
            "open_positions_excluded": True}


def _candidate_report(root: Path, trial: dict) -> dict:
    path = (root / trial["artifact_path"] / "result.json").resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Candidate report escapes the experiment directory")
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_stability(batch: dict) -> dict:
    trials = batch.get("trials", [])
    best = next((t for t in trials if t["candidate_id"] == batch.get("best_candidate_id")), None)
    if best is None:
        return {"status": "unavailable", "reason": "No successful selected candidate"}
    params = best["parameters"]
    neighbors = []
    for candidate in trials:
        if candidate.get("status") != "completed" or candidate is best:
            continue
        other = candidate["parameters"]
        keys = set(params) | set(other)
        difference = [key for key in keys if params.get(key) != other.get(key)]
        if len(difference) != 1:
            continue
        key = difference[0]
        values = {json.dumps(t["parameters"].get(key), sort_keys=True) for t in trials}
        decoded = [json.loads(value) for value in values]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in decoded):
            ordered = sorted(decoded)
            if abs(ordered.index(params.get(key)) - ordered.index(other.get(key))) != 1:
                continue
        neighbors.append({"candidate_id": candidate["candidate_id"], "parameters": other,
                          "total_return": candidate["metrics"]["total_return"]})
    returns = [item["total_return"] for item in neighbors]
    return {"status": "computed", "scope": "adjacent_grid_values_on_one_parameter_in_sample",
            "selected_candidate_id": best["candidate_id"], "neighbors": neighbors,
            "neighbor_count": len(neighbors), "median_return": median(returns) if returns else None,
            "minimum_return": min(returns) if returns else None,
            "positive_fraction": sum(v > 0 for v in returns) / len(returns) if returns else None}


def selection_diagnostics(batch: dict, run_dir: str | Path, *, partitions: int = 4) -> dict:
    trials = batch.get("trials", [])
    root = Path(run_dir)
    completed = [trial for trial in trials if trial.get("status") == "completed"]
    result = {"attempts": len(trials), "successful_attempts": len(completed),
              "failed_attempts": len(trials) - len(completed), "scope": "this_recorded_batch_only",
              "status": "unavailable", "assumptions": [
                  "DSR uses unannualized daily Sharpe and the zero-true-Sharpe null.",
                  "Raw eligible candidate count is treated as independent trials; dependency can invalidate that assumption.",
                  "Other experiments and discarded human ideas are not known; do not interpret this as complete search-adjusted evidence.",
                  "CSCV is a selection diagnostic, not chronological walk-forward validation or an acceptance gate.",
              ]}
    if len(completed) < 2:
        return {**result, "reason": "At least two successful candidates are required"}
    dates, columns, ids = None, [], []
    omitted = []
    for trial in completed:
        current_dates, returns = daily_returns(_candidate_report(root, trial))
        if dates is not None and current_dates != dates:
            return {**result, "reason": "Candidate daily periods differ; no silent alignment or dropping"}
        dates = current_dates
        if len(returns) < max(20, 2 * partitions) or float(returns.std(ddof=1)) <= 1e-15:
            omitted.append(trial["candidate_id"])
            continue
        columns.append(returns)
        ids.append(trial["candidate_id"])
    result.update(omitted_degenerate_or_short_candidates=omitted, candidate_ids=ids,
                  observations=len(dates or []))
    selected = batch.get("best_candidate_id")
    if len(columns) < 2 or selected not in ids:
        return {**result, "reason": "Insufficient nonconstant return series or selected candidate is degenerate"}
    values = np.column_stack(columns)
    sharpes = values.mean(axis=0) / values.std(axis=0, ddof=1)
    chosen = values[:, ids.index(selected)]
    centered = chosen - chosen.mean()
    variance = float(np.mean(centered ** 2))
    skew = float(np.mean(centered ** 3) / variance ** 1.5)
    kurtosis = float(np.mean(centered ** 4) / variance ** 2)
    dsr = deflated_sharpe_ratio(float(sharpes[ids.index(selected)]), sharpes, len(chosen), skewness=skew, kurtosis=kurtosis)
    pbo = probability_of_backtest_overfitting(values, partitions=partitions)
    correlations = np.corrcoef(values, rowvar=False)
    off_diagonal = correlations[np.triu_indices(values.shape[1], k=1)]
    return {**result, "status": "computed", "dsr": dsr, "pbo": pbo,
            "average_pairwise_correlation": float(off_diagonal.mean()), "partitions": partitions,
            "selected_candidate_id": selected, "daily_sharpes": dict(zip(ids, sharpes.tolist()))}


def analyze_robustness(result: dict, run_dir: str | Path, *,
                       rerun: Callable[[dict], dict] | None = None,
                       cost_multipliers: list[float] | None = None,
                       partitions: int = 4, cancel_check=None) -> dict:
    report = result.get("strategy")
    if not report:
        return {"status": "unavailable", "reason": "No successful report"}
    output = {"status": "computed", "report_scope": result.get("selection_scope", "single_configuration"),
              "yearly": _yearly(report), "trade_concentration": _trade_concentration(report),
              "cost_sensitivity": [], "selection_diagnostics": {"status": "unavailable", "reason": "Requires a recorded parameter batch"}}
    if result.get("batch"):
        output["parameter_stability"] = parameter_stability(result["batch"])
        output["selection_diagnostics"] = selection_diagnostics(result["batch"], run_dir, partitions=partitions)
    for value in cost_multipliers or []:
        if cancel_check and cancel_check():
            raise InterruptedError("Cancelled during cost sensitivity")
        multiplier = float(value)
        if not math.isfinite(multiplier) or not .25 <= multiplier <= 4:
            raise ValueError("Cost multipliers must be finite and in [0.25,4]")
        execution = dict(report["execution"])
        for key in ("commission_rate", "minimum_commission", "base_slippage_rate", "impact_coefficient", "maximum_slippage_rate"):
            execution[key] *= multiplier
        if multiplier == 1:
            scenario = report
        elif rerun is None:
            continue
        else:
            scenario = rerun(execution)
        output["cost_sensitivity"].append({"multiplier": multiplier, "metrics": scenario["metrics"]})
    output["assumptions"] = ["Yearly returns compound daily net returns within the actual recorded dates.",
                              "Cost stress changes commissions and slippage, not tax rates or the original recorded configuration.",
                              "A cost change can change lot sizes and trades; returns need not be monotonic.",
                              "Trade concentration excludes still-open positions; units are explicitly recorded."]
    return output
