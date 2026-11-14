"""Describe one posthoc exit hypothesis and verify unchanged control results."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_weekly_components import read, verify_manifest, child_path
from analyze_annual_weekly import stats, paired_effect, slice_curve


CASES = ("primary", "exit_off", "loss_exit")


def summarize(maps):
    groups = []
    for start,end in sorted({(r["start"],r["end"]) for r in maps["primary"].values()}):
        symbols = {s for s,r in maps["primary"].items() if (r["start"],r["end"]) == (start,end)}
        groups.append({"start":start,"end":end,"symbols":sorted(symbols),
                       "cases":{name:stats([maps[name][s] for s in sorted(symbols)]) for name in CASES},
                       "loss_vs_primary":paired_effect(maps["loss_exit"],maps["primary"],symbols),
                       "loss_vs_exit_off":paired_effect(maps["loss_exit"],maps["exit_off"],symbols)})
    return groups


def analyze(root, original, *, continuous=False):
    summary = read(root/"summary.json")
    if summary["status"] != "completed" or summary["completed_tasks"] != 300 or summary["failed_tasks"]:
        raise ValueError("Expected complete 300-task hypothesis study")
    verified = verify_manifest(root/"artifacts.json")
    verified_original = verify_manifest(original/"artifacts.json")
    cases = {name:read(root/"case_results"/f"{name}.json") for name in CASES}
    maps = {name:{r["symbol"]:r for r in case["outcomes"]} for name,case in cases.items()}
    original_cases = {name:read(original/"case_results"/f"{name}.json") for name in ("primary","exit_off")}
    old_maps = {name:{r["symbol"]:r for r in case["outcomes"]} for name,case in original_cases.items()}
    exact_controls = 0
    max_compounding_error = 0.0
    years = {str(y):{name:{} for name in CASES} for y in (2022,2023,2024,2025)} if continuous else {}
    variants = {"primary":"full","exit_off":"drop_weekly_exit","loss_exit":"weekly_exit_loss_only"}
    for name, case in cases.items():
        if len(case["outcomes"]) != 100 or len(maps[name]) != 100 or maps[name].keys() != maps["primary"].keys():
            raise ValueError("Unexpected symbol coverage")
        if case["variant"] != variants[name]:
            raise ValueError("Unexpected research variant")
        for symbol,row in maps[name].items():
            base = maps["primary"][symbol]
            for key in ("cost_bps","status","start","end","data_sha256","calendar_sha256"):
                if row[key] != base[key]:
                    raise ValueError(f"Invalid pair: {name}/{symbol}/{key}")
            if row["status"] != "completed" or row["cost_bps"] != 10:
                raise ValueError("Unexpected status or cost")
            result = read(child_path(root,case)/row["artifact_path"])
            identity = result["strategy"]
            if (identity["research_variant"] != variants[name] or identity["symbol"] != symbol
                    or identity["settings"]["cost_bps"] != 10):
                raise ValueError("Result strategy identity differs from case")
            if name in original_cases:
                old = old_maps[name][symbol]
                for key in ("start","end","metrics","buy_hold_return","data_sha256","calendar_sha256"):
                    if old[key] != row[key]:
                        raise ValueError(f"Control changed: {name}/{symbol}/{key}")
                before = read(child_path(original,original_cases[name])/old["artifact_path"])
                if result != before:
                    raise ValueError(f"Entire control result changed: {name}/{symbol}")
                exact_controls += 1
            row["buy_hold_maximum_drawdown"] = result["baselines"]["buy_hold"]["metrics"]["maximum_drawdown"]
            if continuous:
                strategy = result["strategy"]
                factors = []
                for year, year_maps in years.items():
                    cut = slice_curve(strategy["equity_curve"], int(year))
                    factors.append(1+cut["total_return"])
                    bh = slice_curve(result["baselines"]["buy_hold"]["equity_curve"],int(year))
                    metrics = {k:cut[k] for k in ("total_return","maximum_drawdown","invested_close_ratio")}
                    metrics["completed_trades"] = sum(t["status"] == "closed" and t["exit_date"].startswith(year) for t in strategy["trades"])
                    metrics["position_changes"] = sum(f["status"] == "applied" and f["date"].startswith(year) for f in strategy["fills"])
                    year_maps[name][symbol] = {"symbol":symbol,"start":cut["start"],"end":cut["end"],"metrics":metrics,
                                              "buy_hold_return":bh["total_return"],"buy_hold_maximum_drawdown":bh["maximum_drawdown"],
                                              "carry_in":cut["carry_in"],
                                              "worst_underwater_from_historical_peak":cut["worst_underwater_from_historical_peak"]}
                error = abs(math.prod(factors)-strategy["metrics"]["final_nav"])
                max_compounding_error = max(max_compounding_error,error)
                if error > 1e-10:
                    raise ValueError("Hypothesis annual factors do not reconcile")
    return {"path":str(root.resolve()),"run_id":root.name,"verified_artifact_entries":verified,
            "verified_original_artifact_entries":verified_original,"max_compounding_error":max_compounding_error,
            "implementation_sha256":read(root/"protocol.json")["implementation_sha256"],
            "controls_exact_equal":exact_controls,"groups":summarize(maps),
            "years":{year:{"groups":summarize(m)} for year,m in years.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair",action="append",required=True,help="YEAR_OR_continuous=NEW_STUDY=ORIGINAL_STUDY")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    windows = {}
    for item in args.pair:
        label,new,old = item.split("=",2)
        if label in windows:
            raise ValueError("Duplicate window")
        windows[label] = analyze(Path(new),Path(old),continuous=label=="continuous")
        if label != "continuous" and any(not g["start"].startswith(label+"-") or not g["end"].startswith(label+"-")
                                          for g in windows[label]["groups"]):
            raise ValueError("Supplied year does not match study dates")
    if set(windows) != {"2022","2023","2024","2025","continuous"}:
        raise ValueError("Expected four annual windows and continuous path")
    if len({v["implementation_sha256"] for v in windows.values()}) != 1:
        raise ValueError("Hypothesis windows used different implementations")
    output = {"scope":"Single posthoc natural-zero-threshold hypothesis chosen after annual and event results; no untouched holdout or deployment recommendation.",
              "rule":"Retain full entry and fixed stop. Start a weekly-invalidated exit only if prior close <= adjusted entry price; existing pending exits remain latched.",
              "windows":windows,"controls_exact_equal":sum(v["controls_exact_equal"] for v in windows.values()),
              "source_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"controls_exact_equal":output["controls_exact_equal"]}))


if __name__ == "__main__":
    main()
