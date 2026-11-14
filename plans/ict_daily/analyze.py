"""Independently summarize verified ICT daily artifacts; no strategy selection."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


EPS=1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def trade_stats(trades):
    closed=[t for t in trades if t["status"]=="closed"]
    wins=[t["net_return"] for t in closed if t["net_return"]>EPS]
    losses=[t["net_return"] for t in closed if t["net_return"] < -EPS]
    return {"entries":len(trades),"closed":len(closed),"open":len(trades)-len(closed),
            "wins":len(wins),"losses":len(losses),"ties":len(closed)-len(wins)-len(losses),
            "win_rate":len(wins)/len(closed) if closed else None,
            "average_win":mean(wins) if wins else None,"average_loss_abs":abs(mean(losses)) if losses else None,
            "payoff_ratio":mean(wins)/abs(mean(losses)) if wins and losses else None,
            "mean_closed_return":mean(t["net_return"] for t in closed) if closed else None,
            "median_closed_return":median(t["net_return"] for t in closed) if closed else None,
            "mean_holding_bars":mean(t["holding_bars"] for t in closed) if closed else None,
            "mean_realized_price_r":mean(t["realized_price_r"] for t in closed) if closed else None,
            "exit_reasons":dict(Counter(t["exit_reason"] for t in closed)),
            "entry_day_stop_touches":sum(t["entry_day_stop_touch"] for t in trades),
            "entry_day_target_touches":sum(t["entry_day_target_touch"] for t in trades),
            "ambiguous_entry_stop":sum(t["ambiguous_entry_stop_order"] for t in trades),
            "ambiguous_entry_target":sum(t["ambiguous_entry_target_order"] for t in trades),
            "ambiguous_stop_target":sum(t["ambiguous_stop_target_order"] for t in trades),
            "any_order_flag_union":sum(any(t[k] for k in ("ambiguous_entry_stop_order","ambiguous_entry_target_order","ambiguous_stop_target_order")) for t in trades)}


def cut(curve,year):
    current=[r for r in curve if r["date"].startswith(str(year))]
    previous=[r for r in curve if r["date"][:4]<str(year)]
    initial=previous[-1]["nav"] if previous else 1.0
    peak,dd=initial,0.0
    for r in current:
        peak=max(peak,r["nav"])
        dd=max(dd,1-r["nav"]/peak)
    return {"start":current[0]["date"],"end":current[-1]["date"],"total_return":current[-1]["nav"]/initial-1,
            "maximum_drawdown":dd,"invested_close_ratio":mean(r["exposure"] for r in current)}


def distribution(rows):
    if len({(r["start"],r["end"]) for r in rows})!=1:
        raise ValueError("Do not combine unequal evaluation periods")
    return {"count":len(rows),"start":rows[0]["start"],"end":rows[0]["end"],
            "mean_return":mean(r["total_return"] for r in rows),"median_return":median(r["total_return"] for r in rows),
            "positive_return_ratio":mean(r["total_return"]>EPS for r in rows),
            "mean_maximum_drawdown":mean(r["maximum_drawdown"] for r in rows),
            "mean_invested_close_ratio":mean(r["invested_close_ratio"] for r in rows),
            "mean_buy_hold_return":mean(r["buy_hold_return"] for r in rows),
            "median_buy_hold_return":median(r["buy_hold_return"] for r in rows),
            "beat_buy_hold_ratio":mean(r["total_return"]-r["buy_hold_return"]>EPS for r in rows)}


def analyze(root):
    manifest=read(root/"artifacts.json")["sha256"]
    for relative,digest in manifest.items():
        file=(root/relative).resolve()
        if not file.is_relative_to(root.resolve()) or sha(file)!=digest:
            raise ValueError(f"Invalid artifact: {relative}")
    freeze=read(root/"freeze.json")
    for item in [freeze["calendar"],*freeze["inputs"].values()]:
        if sha(Path(item["path"]))!=item["sha256"]:
            raise ValueError("Source input changed")
    summary=read(root/"summary.json")
    if summary["status"]!="completed" or summary["completed_tasks"]!=400 or summary["failed_tasks"]:
        raise ValueError("Expected all400 frozen tasks completed")
    config=freeze["configuration"]
    expected={(c["id"],s) for c in config["cases"] for s in config["symbols"]}
    actual=[(r["case_id"],r["symbol"]) for r in summary["outcomes"]]
    if len(actual)!=400 or len(set(actual))!=400 or set(actual)!=expected:
        raise ValueError("Task identities do not reconcile")
    cases={}
    cost_paths={}
    max_trade_error=max_nav_error=max_annual_error=0.0
    for case in config["cases"]:
        rows=[r for r in summary["outcomes"] if r["case_id"]==case["id"]]
        trades,assets=[],[]
        years={str(y):{"trades":[],"assets":[]} for y in (2022,2023,2024,2025)}
        diagnostic=Counter()
        order_status=Counter()
        symbols_with_entries=symbols_with_closed=0
        symbol_win_rates=[]
        for row in rows:
            path=(root/row["artifact_path"]).resolve()
            if row["artifact_path"] not in manifest or not path.is_relative_to(root.resolve()):
                raise ValueError("Result is outside the verified manifest")
            if row["input_sha256"]!=freeze["inputs"][row["symbol"]]["sha256"] or row["calendar_sha256"]!=freeze["calendar"]["sha256"]:
                raise ValueError("Task input identities differ from freeze")
            result=read(path)
            strategy=result["strategy"]
            if (strategy["symbol"]!=row["symbol"] or strategy["settings"]["target_r"]!=case["target_r"]
                    or strategy["settings"]["cost_bps"]!=case["cost_bps"] or strategy["metrics"]!=row["metrics"]
                    or strategy["strategy"]!=config["model"] or strategy["price_basis"]!="pre"
                    or strategy["requested_start"]!=config["start"] or strategy["requested_end"]!=config["end"]
                    or strategy["start"]!=row["start"] or strategy["end"]!=row["end"]):
                raise ValueError("Result identity differs from task")
            for baseline in result["baselines"].values():
                if (baseline["start"],baseline["end"],baseline["price_basis"],baseline["cost_bps"])!=(strategy["start"],strategy["end"],"pre",case["cost_bps"]):
                    raise ValueError("Baseline identity differs from strategy")
            if any(abs(v["nav"]-1)>EPS or v["exposure"]!=0 for v in result["baselines"]["cash"]["equity_curve"]):
                raise ValueError("Cash baseline is not zero-interest cash")
            cost=case["cost_bps"]/10000
            compounded=[]
            for t in strategy["trades"]:
                terminal=t["adjusted_exit_price"] if t["status"]=="closed" else strategy["equity_curve"][-1]["adjusted_close"]
                expected_return=terminal/t["adjusted_entry_price"]*(1-cost)**(2 if t["status"]=="closed" else 1)-1
                error=abs(expected_return-t["net_return"])
                max_trade_error=max(max_trade_error,error)
                if error>1e-10:
                    raise ValueError("Trade price/cost return does not reconcile")
                if t["status"]=="closed" and t["exit_date"]<=t["entry_date"]:
                    raise ValueError("Entry-day sale violates declared CN model")
                compounded.append(1+t["net_return"])
                item={**t,"symbol":row["symbol"]}
                trades.append(item)
                if t["status"]=="closed":
                    years[t["exit_date"][:4]]["trades"].append(item)
            error=abs(math.prod(compounded)-strategy["metrics"]["final_nav"])
            max_nav_error=max(max_nav_error,error)
            if error>1e-10:
                raise ValueError("Trade factors do not reproduce NAV")
            current=trade_stats(strategy["trades"])
            if current["wins"]!=strategy["metrics"]["winning_trades"] or current["closed"]!=strategy["metrics"]["completed_trades"]:
                raise ValueError("Engine win count differs from independent calculation")
            symbols_with_entries+=bool(current["entries"])
            symbols_with_closed+=bool(current["closed"])
            if current["closed"]:
                symbol_win_rates.append(current["win_rate"])
            for key in ("sweeps","confirmed_setups","accepted_orders","ignored_orders"):
                diagnostic[key]+=strategy["diagnostics"][key]
            order_status.update(strategy["diagnostics"]["order_status_counts"])
            assets.append({"symbol":row["symbol"],"start":strategy["start"],"end":strategy["end"],
                           **{k:strategy["metrics"][k] for k in ("total_return","maximum_drawdown","invested_close_ratio")},
                           "buy_hold_return":result["baselines"]["buy_hold"]["metrics"]["total_return"],"trade_stats":current})
            annual_factors=[]
            for year in years:
                sliced=cut(strategy["equity_curve"],year)
                bh=cut(result["baselines"]["buy_hold"]["equity_curve"],year)
                annual_factors.append(1+sliced["total_return"])
                years[year]["assets"].append({"symbol":row["symbol"],**sliced,"buy_hold_return":bh["total_return"]})
            error=abs(math.prod(annual_factors)-strategy["metrics"]["final_nav"])
            max_annual_error=max(max_annual_error,error)
            if error>1e-10:
                raise ValueError("Calendar annual factors do not reproduce NAV")
            if case["target_r"]==2:
                price_path=[{k:f[k] for k in ("date","phase","status","from_exposure","target_exposure","adjusted_price","reason","setup_id")} for f in strategy["fills"]]
                if row["symbol"] in cost_paths and price_path!=cost_paths[row["symbol"]]:
                    raise ValueError("Costs unexpectedly changed the2R execution path")
                cost_paths[row["symbol"]]=price_path
        if diagnostic["accepted_orders"]!=sum(v for k,v in order_status.items() if k!="ignored"):
            raise ValueError("Order funnel does not reconcile")
        stats=trade_stats(trades)
        if order_status["filled"]!=stats["entries"]:
            raise ValueError("Filled orders differ from entries")
        nonambiguous=[t for t in trades if not(t["ambiguous_entry_stop_order"] or t["ambiguous_stop_target_order"])]
        cases[case["id"]]={"settings":case,"trade_stats":stats,"funnel":dict(diagnostic),"order_status":dict(order_status),
                           "distribution":distribution(assets),
                           "distribution_among_symbols_with_entries":distribution([a for a in assets if a["trade_stats"]["entries"]]) if symbols_with_entries else None,
                           "symbols_with_entries":symbols_with_entries,
                           "symbols_with_closed":symbols_with_closed,"symbols_with_no_entries":100-symbols_with_entries,
                           "median_symbol_win_rate_among_traded":median(symbol_win_rates) if symbol_win_rates else None,
                           "trades_excluding_conservative_stop_flags":trade_stats(nonambiguous),
                           "ambiguity_subset_note":"Posthoc diagnostic only, not an executable strategy. Entry-day stop flags mark the conservative handling group; they do not prove uncertain ordering for a resting limit path. Entry-target flags do not imply realizable same-day sales. Only use the all-trades win rate as primary.",
                           "years":{y:{"closed_trade_stats_by_exit_year":trade_stats(v["trades"]),
                                        "new_entries_by_entry_year":sum(t["entry_date"].startswith(y) for t in trades),
                                        "trade_scope":"Win rate uses trades CLOSED in this year, not new entries; open trades excluded.",
                                        "calendar_nav_distribution":distribution(v["assets"])} for y,v in years.items()},
                           "assets":assets,"trades":trades}
    return {"run_id":root.name,"path":str(root.resolve()),"scope":freeze["scope"],"cases":cases,
            "verified_artifact_references":len(manifest),"verified_input_files":len(freeze["inputs"])+1,
            "same_execution_2r_costs_symbols":len(cost_paths),
            "max_trade_return_reconstruction_error":max_trade_error,"max_nav_product_error":max_nav_error,
            "max_annual_reconciliation_error":max_annual_error,"script_sha256":sha(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.run.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"cases":{k:v["trade_stats"] for k,v in result["cases"].items()}},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
