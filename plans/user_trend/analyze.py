"""Audit and compare the user's EMA/EFI/ATR path and execution scenarios."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,median

EPS=1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


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
            "exit_reasons":dict(Counter(t["exit_reason"] for t in closed)),
            "entry_sources":dict(Counter("both" if len(t["entry_reasons"])==2 else "A_only" if t["entry_reasons"]==["ema_band_pullback"] else "B_only" for t in trades)),
            "trailing_activated_trades":sum(t["trail_active"] for t in trades),
            "entry_day_stop_lock_trades":sum(t["entry_day_price_stop_touch"] for t in trades)}


def cut(curve,year):
    rows=[r for r in curve if r["date"].startswith(year)]
    before=[r for r in curve if r["date"][:4]<year]
    initial=before[-1]["nav"] if before else 1.0
    peak,dd=initial,0.
    for row in rows:
        peak=max(peak,row["nav"]);dd=max(dd,1-row["nav"]/peak)
    return {"start":rows[0]["date"],"end":rows[-1]["date"],"total_return":rows[-1]["nav"]/initial-1,
            "maximum_drawdown":dd,"invested_close_ratio":mean(r["exposure"] for r in rows),
            "carry_in":before[-1]["exposure"] if before else 0}


def distribution(rows):
    if len({(r["start"],r["end"]) for r in rows})!=1:
        raise ValueError("Unequal periods must not share a distribution")
    returns=sorted(r["total_return"] for r in rows);n=len(rows);trim=int(n*.1)
    return {"count":n,"start":rows[0]["start"],"end":rows[0]["end"],
            "mean_return":mean(returns),"median_return":median(returns),
            "trimmed10_mean_return":mean(returns[trim:-trim] if trim else returns),
            "positive_return_ratio":mean(r>EPS for r in returns),
            "mean_maximum_drawdown":mean(r["maximum_drawdown"] for r in rows),
            "mean_invested_close_ratio":mean(r["invested_close_ratio"] for r in rows),
            "mean_buy_hold_return":mean(r["buy_hold_return"] for r in rows),
            "median_buy_hold_return":median(r["buy_hold_return"] for r in rows),
            "beat_buy_hold_ratio":mean(r["total_return"]-r["buy_hold_return"]>EPS for r in rows),
            "carry_in_count":sum(r.get("carry_in",0) for r in rows)}


def pair_effect(left,right):
    a={r["symbol"]:r for r in left};b={r["symbol"]:r for r in right}
    if a.keys()!=b.keys():raise ValueError("Unmatched symbols")
    pairs=[]
    for symbol in sorted(a):
        if (a[symbol]["start"],a[symbol]["end"])!=(b[symbol]["start"],b[symbol]["end"]):raise ValueError("Unmatched dates")
        pairs.append({"symbol":symbol,"return_delta":a[symbol]["total_return"]-b[symbol]["total_return"],
                      "drawdown_delta":a[symbol]["maximum_drawdown"]-b[symbol]["maximum_drawdown"]})
    return {"count":len(pairs),"mean_return_delta":mean(p["return_delta"] for p in pairs),
            "median_return_delta":median(p["return_delta"] for p in pairs),
            "positive_delta_count":sum(p["return_delta"]>EPS for p in pairs),
            "return_tie_count":sum(abs(p["return_delta"])<=EPS for p in pairs),
            "mean_drawdown_delta":mean(p["drawdown_delta"] for p in pairs),"pairs":pairs}


def analyze(root):
    manifest=read(root/"artifacts.json")["sha256"]
    for rel,digest in manifest.items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError(f"Invalid artifact {rel}")
    freeze=read(root/"freeze.json");config=freeze["configuration"]
    for item in [freeze["calendar"],*freeze["inputs"].values()]:
        if sha(Path(item["path"]))!=item["sha256"]:raise ValueError("Source changed")
    summary=read(root/"summary.json")
    if summary["status"]!="completed" or summary["completed_tasks"]!=800 or summary["failed_tasks"]:raise ValueError("Expected800complete tasks")
    identities=[(r["case_id"],r["symbol"]) for r in summary["outcomes"]]
    expected={(c["id"],s) for c in config["cases"] for s in config["symbols"]}
    if len(identities)!=800 or len(set(identities))!=800 or set(identities)!=expected:raise ValueError("Task identities differ")
    cases={};cost_paths={};cost_checks=0
    max_trade_error=max_nav_error=max_annual_error=0.
    for case in config["cases"]:
        rows=[r for r in summary["outcomes"] if r["case_id"]==case["id"]]
        trades,assets=[],[]
        years={str(y):{"trades":[],"assets":[]} for y in (2022,2023,2024,2025)}
        diagnostic,signal_status,ignored=Counter(),Counter(),Counter()
        for row in rows:
            p=(root/row["artifact_path"]).resolve()
            if row["artifact_path"] not in manifest or not p.is_relative_to(root.resolve()):raise ValueError("Result outside manifest")
            if row["input_sha256"]!=freeze["inputs"][row["symbol"]]["sha256"] or row["calendar_sha256"]!=freeze["calendar"]["sha256"]:raise ValueError("Task input differs")
            result=read(p);strategy=result["strategy"]
            if strategy["strategy"]!=config["model"] or strategy["symbol"]!=row["symbol"] or strategy["price_basis"]!="pre":raise ValueError("Model identity differs")
            if strategy["metrics"]!=row["metrics"] or strategy["requested_start"]!=config["start"] or strategy["requested_end"]!=config["end"]:raise ValueError("Result metadata differs")
            for key in ("entry_mode","intraday_path","cost_bps"):
                if strategy["settings"][key]!=case[key]:raise ValueError("Case settings differ")
            if strategy["settings"]["atr_period"]!=config["atr_period"]:raise ValueError("ATRperiod differs")
            for baseline in result["baselines"].values():
                if (baseline["start"],baseline["end"],baseline["price_basis"],baseline["cost_bps"])!=(strategy["start"],strategy["end"],"pre",case["cost_bps"]):raise ValueError("Baseline differs")
            if any(abs(e["nav"]-1)>EPS for e in result["baselines"]["cash"]["equity_curve"]):raise ValueError("Cash changed")
            factor=1.;cost=case["cost_bps"]/10000
            for t in strategy["trades"]:
                price=t["adjusted_exit_price"] if t["status"]=="closed" else strategy["equity_curve"][-1]["adjusted_close"]
                expected_return=price/t["adjusted_entry_price"]*(1-cost)**(2 if t["status"]=="closed" else 1)-1
                error=abs(expected_return-t["net_return"]);max_trade_error=max(max_trade_error,error)
                if error>1e-10:raise ValueError("Trade return does not reconcile")
                if t["status"]=="closed" and t["exit_date"]<=t["entry_date"]:raise ValueError("Same-day sale")
                if abs(t["initial_stop"]-(t["adjusted_entry_price"]-.5*t["atr_at_entry"]))>1e-9:raise ValueError("Hardstop differs")
                factor*=1+t["net_return"]
                item={**t,"symbol":row["symbol"]};trades.append(item)
                if t["status"]=="closed":years[t["exit_date"][:4]]["trades"].append(item)
            error=abs(factor-strategy["metrics"]["final_nav"]);max_nav_error=max(max_nav_error,error)
            if error>1e-9:raise ValueError("NAV does not reconcile")
            stats=trade_stats(strategy["trades"])
            if (stats["wins"],stats["closed"])!=(strategy["metrics"]["winning_trades"],strategy["metrics"]["completed_trades"]):raise ValueError("Win count differs")
            assets.append({"symbol":row["symbol"],"start":strategy["start"],"end":strategy["end"],
                           **{k:strategy["metrics"][k] for k in ("total_return","maximum_drawdown","invested_close_ratio")},
                           "buy_hold_return":result["baselines"]["buy_hold"]["metrics"]["total_return"],"trade_stats":stats})
            for key in ("signals","accepted_signals"):diagnostic[key]+=strategy["diagnostics"][key]
            signal_status.update(strategy["diagnostics"]["signal_status_counts"]);ignored.update(strategy["diagnostics"]["ignored_reasons"])
            annual_factor=1.
            for year in years:
                c=cut(strategy["equity_curve"],year);bh=cut(result["baselines"]["buy_hold"]["equity_curve"],year)
                annual_factor*=1+c["total_return"]
                years[year]["assets"].append({"symbol":row["symbol"],**c,"buy_hold_return":bh["total_return"]})
            error=abs(annual_factor-strategy["metrics"]["final_nav"]);max_annual_error=max(max_annual_error,error)
            if error>1e-9:raise ValueError("Annual factors differ")
            if case["entry_mode"]=="next_open":
                key=(case["intraday_path"],row["symbol"])
                path=[{k:f[k] for k in ("date","phase","from_exposure","target_exposure","adjusted_price","reason","signal_id")} for f in strategy["fills"]]
                if key in cost_paths:
                    if path!=cost_paths[key]:raise ValueError("Costs changed execution")
                    cost_checks+=1
                cost_paths[key]=path
        stats=trade_stats(trades)
        if signal_status["filled"]!=stats["entries"]:raise ValueError("Signals and entries differ")
        cases[case["id"]]={"settings":case,"trade_stats":stats,"distribution":distribution(assets),
                           "symbols_with_no_entries":sum(a["trade_stats"]["entries"]==0 for a in assets),
                           "signals":dict(diagnostic),"signal_status":dict(signal_status),"ignored_reasons":dict(ignored),
                           "years":{y:{"trade_stats_by_exit_year":trade_stats(v["trades"]),
                                        "new_entries_by_entry_year":sum(t["entry_date"].startswith(y) for t in trades),
                                        "distribution":distribution(v["assets"])} for y,v in years.items()},"assets":assets,"trades":trades}
    comparisons={}
    for mode in ("next_open","signal_close"):
        left=f"{mode}_ohlc_10bps";right=f"{mode}_olhc_10bps"
        comparisons[mode+"_ohlc_minus_olhc"]={"left":left,"right":right,**pair_effect(cases[left]["assets"],cases[right]["assets"])}
    for path in ("ohlc","olhc"):
        left=f"signal_close_{path}_10bps";right=f"next_open_{path}_10bps"
        comparisons[path+"_close_minus_open"]={"left":left,"right":right,**pair_effect(cases[left]["assets"],cases[right]["assets"])}
    return {"run_id":root.name,"path":str(root.resolve()),"scope":freeze["scope"],"cases":cases,"paired_comparisons":comparisons,
            "verified_artifacts":len(manifest),"input_files":101,"cost_same_path_checks":cost_checks,
            "max_trade_reconstruction_error":max_trade_error,"max_nav_product_error":max_nav_error,
            "max_annual_reconciliation_error":max_annual_error,"script_sha256":sha(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();result=analyze(args.run.resolve());args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"cases":{k:{"trades":v["trade_stats"],"distribution":v["distribution"]} for k,v in result["cases"].items()}},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
