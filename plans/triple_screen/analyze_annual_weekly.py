"""Audit annual reset studies and slice a continuous four-case path, offline."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import pandas as pd

from analyze_weekly_components import read, verify_manifest, child_path


CASES = ("primary", "entry_off", "exit_off", "both_off")
SIGN_TOLERANCE = 1e-12  # NAV return units; classify numerical ties, never change a return.


def drawdown(values, initial):
    peak, worst = initial, 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, 1-value/peak)
    return worst


def slice_curve(curve, year):
    prefix = str(year)
    prior = [x for x in curve if x["date"][:4] < prefix]
    rows = [x for x in curve if x["date"].startswith(prefix)]
    if not rows:
        raise ValueError(f"Missing year {year}")
    initial = prior[-1]["nav"] if prior else 1.0
    prior_peak = max([1.0, *[x["nav"] for x in prior]])
    historical_peak = prior_peak
    worst_historical_underwater = 0.0
    for row in rows:
        historical_peak = max(historical_peak, row["nav"])
        worst_historical_underwater = max(worst_historical_underwater, 1-row["nav"]/historical_peak)
    return {"start": rows[0]["date"], "end": rows[-1]["date"], "initial_nav": initial,
            "total_return": rows[-1]["nav"]/initial-1,
            "maximum_drawdown": drawdown([r["nav"] for r in rows], initial),
            "worst_underwater_from_historical_peak": worst_historical_underwater,
            "invested_close_ratio": mean(r["exposure"] for r in rows),
            "carry_in": prior[-1]["exposure"] if prior else 0}


def self_test():
    curve = [{"date": "2022-12-30", "nav": 1.2, "exposure": 1},
             {"date": "2023-01-03", "nav": 1.08, "exposure": 1},
             {"date": "2023-12-29", "nav": 1.26, "exposure": 0},
             {"date": "2024-01-02", "nav": 1.134, "exposure": 0}]
    a, b, c = [slice_curve(curve, y) for y in (2022, 2023, 2024)]
    assert math.isclose(b["total_return"], .05)
    assert math.isclose(b["maximum_drawdown"], .1)
    assert math.isclose(c["maximum_drawdown"], .1)
    assert math.isclose(math.prod(1+x["total_return"] for x in (a,b,c)), 1.134)
    assert a["carry_in"] == 0 and b["carry_in"] == 1 and c["carry_in"] == 0
    declining = [{"date": "2022-12-30", "nav": .7, "exposure": 0},
                 {"date": "2023-01-03", "nav": .77, "exposure": 0}]
    segment = slice_curve(declining, 2023)
    assert segment["maximum_drawdown"] == 0
    assert math.isclose(segment["worst_underwater_from_historical_peak"], .23)


def load_study(root):
    root = root.resolve()
    summary = read(root/"summary.json")
    if summary["status"] != "completed" or summary["completed_tasks"] != 400 or summary["failed_tasks"]:
        raise ValueError(f"Expected complete 400-task study: {root}")
    verified = verify_manifest(root/"artifacts.json")
    cases = {name: read(root/"case_results"/f"{name}.json") for name in CASES}
    maps = {name: {r["symbol"]: r for r in c["outcomes"]} for name, c in cases.items()}
    for name, rows in maps.items():
        if len(rows) != 100 or len(cases[name]["outcomes"]) != 100 or rows.keys() != maps["primary"].keys():
            raise ValueError("Unexpected symbol coverage")
        for symbol, row in rows.items():
            base = maps["primary"][symbol]
            for key in ("cost_bps", "status", "start", "end", "data_sha256", "calendar_sha256"):
                if row[key] != base[key]:
                    raise ValueError(f"Pairing mismatch {name}/{symbol}/{key}")
            if row["cost_bps"] != 10 or row["status"] != "completed":
                raise ValueError("Unexpected cost or failed row")
    protocol = read(root/"protocol.json")
    return root, cases, maps, {"run_id": root.name, "path": str(root), "verified_artifact_entries": verified,
                             "protocol_sha256": summary["protocol_sha256"],
                             "implementation_sha256": protocol["implementation_sha256"]}


def context_before_start(root, case, row):
    path = child_path(root, case)/"inputs"/(row["symbol"].replace(".", "_")+".csv")
    frame = pd.read_csv(path)
    prior = frame.loc[pd.to_datetime(frame["datetime"]) < pd.Timestamp(row["start"])]
    closes = prior["close"].astype(float)
    if len(closes) < 127:
        raise ValueError("Insufficient prior bars for fixed 126-bar context")
    return {"as_of": str(prior.iloc[-1]["datetime"])[:10],
            "prior_126_return": float(closes.iloc[-1]/closes.iloc[-127]-1),
            "above_prior_sma126": bool(closes.iloc[-1] > closes.iloc[-126:].mean())}


def stats(rows):
    metrics = [r["metrics"] for r in rows]
    returns = [m["total_return"] for m in metrics]
    return {"count": len(rows), "mean_return": mean(returns), "median_return": median(returns),
            "positive_return_ratio": mean(x > 0 for x in returns),
            "mean_maximum_drawdown": mean(m["maximum_drawdown"] for m in metrics),
            "mean_buy_hold_return": mean(r["buy_hold_return"] for r in rows),
            "median_buy_hold_return": median(r["buy_hold_return"] for r in rows),
            "beat_buy_hold_ratio": mean(r["metrics"]["total_return"] > r["buy_hold_return"] for r in rows),
            "mean_buy_hold_maximum_drawdown": mean(r["buy_hold_maximum_drawdown"] for r in rows),
            "mean_invested_close_ratio": mean(m["invested_close_ratio"] for m in metrics),
            "mean_closed_trades": mean(m["completed_trades"] for m in metrics),
            "mean_position_changes": mean(m["position_changes"] for m in metrics),
            "carry_in_count": sum(r.get("carry_in", 0) for r in rows),
            "mean_worst_underwater_from_historical_peak": mean(r["worst_underwater_from_historical_peak"] for r in rows)
                if "worst_underwater_from_historical_peak" in rows[0] else None}


def paired_effect(on, off, symbols):
    pairs = [{"symbol": s, "return_gain": on[s]["metrics"]["total_return"]-off[s]["metrics"]["total_return"],
              "drawdown_reduction": off[s]["metrics"]["maximum_drawdown"]-on[s]["metrics"]["maximum_drawdown"],
              "buy_hold_return": on[s]["buy_hold_return"]} for s in sorted(symbols)]
    gains = [p["return_gain"] for p in pairs]
    reductions = [p["drawdown_reduction"] for p in pairs]
    ordered = sorted(gains)
    trim = int(len(gains)*.1)
    trimmed = ordered[trim:-trim] if trim else ordered
    quadrants = Counter("return_up_risk_down" if p["return_gain"] > SIGN_TOLERANCE and p["drawdown_reduction"] > SIGN_TOLERANCE
                        else "return_not_up_risk_down" if p["return_gain"] <= SIGN_TOLERANCE and p["drawdown_reduction"] > SIGN_TOLERANCE
                        else "return_up_risk_not_down" if p["return_gain"] > SIGN_TOLERANCE
                        else "neither_improved" for p in pairs)
    return {"count": len(pairs), "mean_return_gain": mean(gains), "median_return_gain": median(gains),
            "trimmed_10pct_mean_return_gain": mean(trimmed), "positive_gain_ratio": mean(g > SIGN_TOLERANCE for g in gains),
            "return_tie_count": sum(abs(g) <= SIGN_TOLERANCE for g in gains),
            "drawdown_tie_count": sum(abs(d) <= SIGN_TOLERANCE for d in reductions),
            "sign_tolerance": SIGN_TOLERANCE,
            "mean_drawdown_reduction": mean(reductions), "median_drawdown_reduction": median(reductions),
            "drawdown_reduced_ratio": mean(d > SIGN_TOLERANCE for d in reductions),
            "quadrants": dict(quadrants), "pairs": pairs}


def groups(maps, contexts):
    output = []
    for start, end in sorted({(r["start"], r["end"]) for r in maps["primary"].values()}):
        symbols = {s for s,r in maps["primary"].items() if (r["start"], r["end"]) == (start,end)}
        group = {"start": start, "end": end, "symbols": sorted(symbols),
                 "cases": {c: stats([maps[c][s] for s in sorted(symbols)]) for c in CASES},
                 "weekly_exit_with_entry_filter": paired_effect(maps["primary"], maps["exit_off"], symbols),
                 "weekly_exit_without_entry_filter": paired_effect(maps["entry_off"], maps["both_off"], symbols)}
        group["interaction_mean_return"] = (group["weekly_exit_with_entry_filter"]["mean_return_gain"]
                                             -group["weekly_exit_without_entry_filter"]["mean_return_gain"])
        conditional = {}
        predicates = {
            "posthoc_same_year_buy_hold_negative": lambda s: maps["primary"][s]["buy_hold_return"] < 0,
            "posthoc_same_year_buy_hold_nonnegative": lambda s: maps["primary"][s]["buy_hold_return"] >= 0,
            "prior_126_momentum_negative": lambda s: contexts[s]["prior_126_return"] < 0,
            "prior_126_momentum_nonnegative": lambda s: contexts[s]["prior_126_return"] >= 0,
            "prior_close_above_sma126": lambda s: contexts[s]["above_prior_sma126"],
            "prior_close_not_above_sma126": lambda s: not contexts[s]["above_prior_sma126"],
        }
        for label, predicate in predicates.items():
            subset = {s for s in symbols if predicate(s)}
            if subset:
                conditional[label] = paired_effect(maps["primary"], maps["exit_off"], subset)
        group["conditions"] = conditional
        output.append(group)
    return output


def annual(root):
    root, cases, maps, evidence = load_study(root)
    contexts = {s: context_before_start(root, cases["primary"], r) for s,r in maps["primary"].items()}
    for name, case in cases.items():
        for symbol, row in maps[name].items():
            result = read(child_path(root, case)/row["artifact_path"])
            row["buy_hold_maximum_drawdown"] = result["baselines"]["buy_hold"]["metrics"]["maximum_drawdown"]
    return {**evidence, "groups": groups(maps, contexts), "prior_context": contexts}, maps


def continuous(root):
    root, cases, maps, evidence = load_study(root)
    years = (2022,2023,2024,2025)
    by_year = {str(y): {c: {} for c in CASES} for y in years}
    contexts = {str(y): {} for y in years}
    max_compounding_error = 0.0
    for name, case in cases.items():
        for symbol, row in maps[name].items():
            result = read(child_path(root, case)/row["artifact_path"])
            strategy, baseline = result["strategy"], result["baselines"]["buy_hold"]
            factors = []
            for year in years:
                cut = slice_curve(strategy["equity_curve"], year)
                bh = slice_curve(baseline["equity_curve"], year)
                factors.append(1+cut["total_return"])
                metrics = {k:cut[k] for k in ("total_return", "maximum_drawdown", "invested_close_ratio")}
                metrics["completed_trades"] = sum(t["status"] == "closed" and t["exit_date"].startswith(str(year)) for t in strategy["trades"])
                metrics["position_changes"] = sum(f["status"] == "applied" and f["date"].startswith(str(year)) for f in strategy["fills"])
                sliced = {"symbol":symbol, "start":cut["start"], "end":cut["end"], "metrics":metrics,
                          "carry_in":cut["carry_in"], "initial_nav":cut["initial_nav"],
                          "worst_underwater_from_historical_peak":cut["worst_underwater_from_historical_peak"],
                          "buy_hold_return":bh["total_return"], "buy_hold_maximum_drawdown":bh["maximum_drawdown"]}
                by_year[str(year)][name][symbol] = sliced
                if name == "primary":
                    contexts[str(year)][symbol] = context_before_start(root, case, sliced)
            error = abs(math.prod(factors)-strategy["metrics"]["final_nav"])
            max_compounding_error = max(max_compounding_error,error)
            if error > 1e-10:
                raise ValueError("Continuous annual factors do not reconcile")
    return {**evidence, "max_compounding_error":max_compounding_error,
            "years": {y:{"groups":groups(m,contexts[y])} for y,m in by_year.items()}}, by_year


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual", action="append", required=True, help="YEAR=study-directory; repeat for 2022-2025")
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    self_test()
    annuals, maps, annual_paths = {}, {}, {}
    for item in args.annual:
        year,path = item.split("=",1)
        if year in annuals:
            raise ValueError("Duplicate annual label")
        annual_paths[year] = Path(path)
        annuals[year], maps[year] = annual(Path(path))
        if any(not row["start"].startswith(year+"-") or not row["end"].startswith(year+"-")
               for row in maps[year]["primary"].values()):
            raise ValueError(f"Study dates do not belong to supplied year {year}")
    if set(annuals) != {"2022","2023","2024","2025"}:
        raise ValueError("Expected exactly four years")
    carried, continuous_maps = continuous(args.continuous)
    identities = {v["implementation_sha256"] for v in annuals.values()} | {carried["implementation_sha256"]}
    if len(identities) != 1:
        raise ValueError("Study source identities differ")
    # End-truncated inputs have different SHA values. Require the complete
    # annual input (including warmup) to equal the continuous input prefix.
    continuous_primary = read(args.continuous/"case_results"/"primary.json")
    prefix_checks = 0
    for year, root in annual_paths.items():
        original = read(root/"case_results"/"primary.json")
        continuous_rows = {r["symbol"]:r for r in continuous_primary["outcomes"]}
        for row in original["outcomes"]:
            if row["calendar_sha256"] != continuous_rows[row["symbol"]]["calendar_sha256"]:
                raise ValueError("Calendar version differs across studies")
            relative = Path("inputs")/(row["symbol"].replace(".","_")+".csv")
            short = pd.read_csv(child_path(root,original)/relative,dtype=str,keep_default_na=False)
            long = pd.read_csv(child_path(args.continuous,continuous_primary)/relative,dtype=str,keep_default_na=False)
            prefix = long.loc[long["datetime"] <= short["datetime"].iloc[-1]].reset_index(drop=True)
            if not short.equals(prefix):
                raise ValueError(f"Input prefix changed: {year}/{row['symbol']}")
            prefix_checks += 1
    comparisons = {}
    for year, m in maps.items():
        comparisons[year] = {}
        for c in CASES:
            same = [s for s in m[c] if (m[c][s]["start"],m[c][s]["end"]) == (continuous_maps[year][c][s]["start"],continuous_maps[year][c][s]["end"])]
            period_groups = []
            for start,end in sorted({(m[c][s]["start"],m[c][s]["end"]) for s in same}):
                subset = [s for s in same if (m[c][s]["start"],m[c][s]["end"]) == (start,end)]
                deltas = [continuous_maps[year][c][s]["metrics"]["total_return"]-m[c][s]["metrics"]["total_return"] for s in subset]
                period_groups.append({"start":start,"end":end,"paired_count":len(subset),
                                      "mean_continuous_minus_reset":mean(deltas),
                                      "median_continuous_minus_reset":median(deltas),
                                      "different_return_count":sum(abs(x)>1e-12 for x in deltas)})
            comparisons[year][c] = {"by_effective_period":period_groups,"excluded_symbols":sorted(set(m[c])-set(same))}
    balanced = set.intersection(*[set(max(v["groups"],key=lambda g:len(g["symbols"]))["symbols"]) for v in annuals.values()])
    stability = {}
    for label, data in (("annual_reset",maps),("continuous",continuous_maps)):
        details = []
        for symbol in sorted(balanced):
            gains = [data[y]["primary"][symbol]["metrics"]["total_return"]-data[y]["exit_off"][symbol]["metrics"]["total_return"] for y in sorted(data)]
            details.append({"symbol":symbol,"gains":gains,"positive_years":sum(g>SIGN_TOLERANCE for g in gains),
                            "tie_years":sum(abs(g)<=SIGN_TOLERANCE for g in gains)})
        stability[label] = {"balanced_symbols":len(balanced),"positive_years_histogram":dict(Counter(x["positive_years"] for x in details)),"details":details}
    output = {"scope":"Fixed retrospective CN100, adjust=pre, one-sided 10bps; separate technical NAVs; no untouched test set.",
              "annual_reset":annuals,"continuous":carried,"continuous_minus_reset":comparisons,"four_year_stability":stability,
              "annual_input_prefix_checks":prefix_checks,
              "source_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"years":sorted(annuals),"source_identity":list(identities)[0],
                      "balanced_symbols":len(balanced),"max_compounding_error":carried["max_compounding_error"]}))


if __name__ == "__main__":
    main()
