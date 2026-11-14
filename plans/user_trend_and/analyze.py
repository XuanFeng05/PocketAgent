"""Compare both requested AND policies with exact OR controls and the ICT baseline."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,median
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from plans.user_trend.analyze import distribution,pair_effect,cut

EPS=1e-12


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def verified_run(root,tasks):
    root=root.resolve();manifest=read(root/"artifacts.json")["sha256"]
    for rel,digest in manifest.items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or sha(p)!=digest:raise ValueError(f"Artifact changed: {p}")
    summary=read(root/"summary.json");freeze=read(root/"freeze.json")
    if summary["status"]!="completed" or summary["total_tasks"]!=tasks or summary["failed_tasks"]:raise ValueError("Incomplete run")
    identities=[(r["case_id"],r["symbol"]) for r in summary["outcomes"]]
    expected={(c["id"],s) for c in freeze["configuration"]["cases"] for s in freeze["configuration"]["symbols"]}
    if len(identities)!=tasks or len(set(identities))!=tasks or set(identities)!=expected:raise ValueError("Task coverage differs")
    for item in [freeze["calendar"],*freeze["inputs"].values()]:
        if sha(Path(item["path"]))!=item["sha256"]:raise ValueError("Source input changed")
    rows={(r["case_id"],r["symbol"]):r for r in summary["outcomes"]}
    return {"root":root,"manifest":manifest,"summary":summary,"freeze":freeze,"rows":rows}


def result_of(run,row):
    p=(run["root"]/row["artifact_path"]).resolve()
    if row["artifact_path"] not in run["manifest"] or not p.is_relative_to(run["root"]):raise ValueError("Unpinned result")
    if row["input_sha256"]!=run["freeze"]["inputs"][row["symbol"]]["sha256"] or row["calendar_sha256"]!=run["freeze"]["calendar"]["sha256"]:raise ValueError("Task input identity differs")
    result=read(p);s=result["strategy"]
    if s["symbol"]!=row["symbol"] or s["metrics"]!=row["metrics"] or (s["start"],s["end"])!=(row["start"],row["end"]):raise ValueError("Result metadata differs")
    config=run["freeze"]["configuration"]
    if (s["requested_start"],s["requested_end"])!=(config["start"],config["end"]):raise ValueError("Requested dates differ")
    if "atr_period" in config and s["settings"]["atr_period"]!=config["atr_period"]:raise ValueError("ATRperiod differs")
    for b in result["baselines"].values():
        if (b["symbol"],b["start"],b["end"],b["price_basis"],b["cost_bps"])!=(s["symbol"],s["start"],s["end"],"pre",row["cost_bps"]):raise ValueError("Baseline identity differs")
    return result


def stats(trades):
    closed=[t for t in trades if t["status"]=="closed"]
    wins=[t["net_return"] for t in closed if t["net_return"]>EPS];losses=[t["net_return"] for t in closed if t["net_return"] < -EPS]
    return {"entries":len(trades),"closed":len(closed),"open":len(trades)-len(closed),"wins":len(wins),"losses":len(losses),
            "ties":len(closed)-len(wins)-len(losses),"win_rate":len(wins)/len(closed) if closed else None,
            "average_win":mean(wins) if wins else None,"average_loss_abs":abs(mean(losses)) if losses else None,
            "payoff_ratio":mean(wins)/abs(mean(losses)) if wins and losses else None,
            "mean_closed_return":mean(t["net_return"] for t in closed) if closed else None,
            "median_closed_return":median(t["net_return"] for t in closed) if closed else None,
            "mean_holding_bars":mean(t["holding_bars"] for t in closed) if closed else None,
            "exit_reasons":dict(Counter(t["exit_reason"] for t in closed))}


def reprice_curve(curve,fills,old_bps,new_bps):
    by_day=Counter(f.get("date",f.get("execution_time","")[:10]) for f in fills if f.get("status")=="applied")
    ratio=(1-new_bps/10000)/(1-old_bps/10000);n=0;rows=[]
    for row in curve:
        n+=by_day[row["date"]]
        rows.append({"date":row["date"],"nav":row["nav"]*ratio**n,"exposure":row["exposure"]})
    if n!=sum(by_day.values()):raise ValueError("Fill date outside curve")
    return rows


def reprice_trades(strategy,bps):
    ratio=1-bps/10000;out=[]
    for t in strategy["trades"]:
        price=t["adjusted_exit_price"] if t["status"]=="closed" else strategy["equity_curve"][-1]["adjusted_close"]
        out.append({**t,"net_return":price/t["adjusted_entry_price"]*ratio**(2 if t["status"]=="closed" else 1)-1})
    return out


def asset(symbol,strategy,bh,curve=None,bh_curve=None):
    curve=curve if curve is not None else strategy["equity_curve"]
    bh_curve=bh_curve if bh_curve is not None else bh["equity_curve"]
    peak=1.;dd=0.
    for row in curve:peak=max(peak,row["nav"]);dd=max(dd,1-row["nav"]/peak)
    return {"symbol":symbol,"start":curve[0]["date"],"end":curve[-1]["date"],"total_return":curve[-1]["nav"]-1,
            "maximum_drawdown":dd,"invested_close_ratio":mean(r["exposure"] for r in curve),"buy_hold_return":bh_curve[-1]["nav"]-1}


def self_test():
    curve=[{"date":"2025-01-02","nav":.999,"exposure":1},{"date":"2025-01-03","nav":1.1*.999**2,"exposure":0}]
    fills=[{"date":"2025-01-02","status":"applied"},{"date":"2025-01-03","status":"applied"}]
    zero=reprice_curve(curve,fills,10,0);high=reprice_curve(curve,fills,10,20)
    assert abs(zero[-1]["nav"]-1.1)<1e-12 and abs(high[-1]["nav"]-1.1*.998**2)<1e-12
    assert abs(zero[0]["nav"]-1)<1e-12


def analyze(new_root,old_root,ict_root):
    self_test();new=verified_run(new_root,1200);old=verified_run(old_root,800);ict=verified_run(ict_root,400)
    symbols=new["freeze"]["configuration"]["symbols"]
    for other in (old,ict):
        if set(other["freeze"]["inputs"])!=set(symbols):raise ValueError("Universe differs")
        for symbol in symbols:
            if other["freeze"]["inputs"][symbol]["sha256"]!=new["freeze"]["inputs"][symbol]["sha256"]:raise ValueError("Price snapshots differ")
        if other["freeze"]["calendar"]["sha256"]!=new["freeze"]["calendar"]["sha256"]:raise ValueError("Calendar differs")
    books={};dates={};controls=0;cost_checks=0;cost_max_error=0.;nav_max_error=0.
    for case in new["freeze"]["configuration"]["cases"]:
        assets=[];trades=[];signals=Counter();years={str(y):{"assets":[],"trades":[]} for y in (2022,2023,2024,2025)}
        repriced={str(c):{"assets":[],"trades":[]} for c in (0,20)};inactive=[]
        for symbol in symbols:
            row=new["rows"][(case["id"],symbol)];result=result_of(new,row);s=result["strategy"];bh=result["baselines"]["buy_hold"]
            for key in ("entry_combination","entry_mode","intraday_path","cost_bps"):
                if s["settings"][key]!=case[key] or row[key]!=case[key]:raise ValueError("Case metadata differs")
            if s["price_basis"]!="pre" or s["strategy"]!=new["freeze"]["configuration"]["model"]:raise ValueError("Wrong model/basis")
            if case["entry_combination"]=="OR":
                old_id=f'{case["entry_mode"]}_{case["intraday_path"]}_10bps'
                before=result_of(old,old["rows"][(old_id,symbol)])
                if result!=before:raise ValueError(f"Original OR changed: {old_id}/{symbol}")
                controls+=1
            dt=(case["entry_mode"],case["intraday_path"],symbol)
            dates.setdefault(dt,{})[case["entry_combination"]]={x["date"] for x in s["signals"]}
            for signal in s["signals"]:
                if case["entry_combination"]=="AND_EXPANDED" and not signal["branch_b_atr_efi"]:raise ValueError("Expanded gate wrong")
                if case["entry_combination"]=="AND_BAND" and not(signal["branch_a_band_touch"] and signal["branch_b_atr_efi"]):raise ValueError("Strict gate wrong")
            if not s["trades"]:inactive.append(symbol)
            t=[{**v,"symbol":symbol} for v in s["trades"]];trades.extend(t)
            price_t=reprice_trades(s,10)
            if any(abs(x["net_return"]-y["net_return"])>1e-10 for x,y in zip(t,price_t)):raise ValueError("Trade accounting differs")
            error=abs(math.prod(1+x["net_return"] for x in t)-s["metrics"]["final_nav"]);nav_max_error=max(nav_max_error,error)
            if error>1e-9:raise ValueError("NAV factors differ")
            assets.append(asset(symbol,s,bh));signals.update(s["diagnostics"]["signal_status_counts"])
            factors=[]
            for year in years:
                sliced=cut(s["equity_curve"],year);b=cut(bh["equity_curve"],year);factors.append(1+sliced["total_return"])
                years[year]["assets"].append({"symbol":symbol,**sliced,"buy_hold_return":b["total_return"]})
                years[year]["trades"].extend(v for v in t if v["status"]=="closed" and v["exit_date"].startswith(year))
            if abs(math.prod(factors)-s["metrics"]["final_nav"])>1e-9:raise ValueError("Annual factors differ")
            for bps in (0,20):
                curve=reprice_curve(s["equity_curve"],s["fills"],10,bps)
                bh_new=reprice_curve(bh["equity_curve"],bh["fills"],10,bps)
                repriced[str(bps)]["assets"].append(asset(symbol,s,bh,curve,bh_new))
                rtrades=[{**v,"symbol":symbol} for v in reprice_trades(s,bps)];repriced[str(bps)]["trades"].extend(rtrades)
                if case["entry_combination"]=="OR" and case["entry_mode"]=="next_open":
                    old_id=f'next_open_{case["intraday_path"]}_{bps}bps';ref=result_of(old,old["rows"][(old_id,symbol)])["strategy"]
                    if len(curve)!=len(ref["equity_curve"]) or len(rtrades)!=len(ref["trades"]):raise ValueError("Repricing shape mismatch")
                    if any((a["date"],a["exposure"])!=(b["date"],b["exposure"]) for a,b in zip(curve,ref["equity_curve"])):raise ValueError("Repricing dates/exposures differ")
                    for a,b in zip(rtrades,ref["trades"]):
                        if any(a.get(k)!=b.get(k) for k in ("entry_date","exit_date","status","adjusted_entry_price","adjusted_exit_price","exit_reason")):raise ValueError("Repricing trade identity differs")
                    errors=[abs(a["nav"]-b["nav"]) for a,b in zip(curve,ref["equity_curve"])]
                    errors.extend(abs(a["net_return"]-b["net_return"]) for a,b in zip(rtrades,ref["trades"]))
                    error=max(errors);cost_max_error=max(cost_max_error,error)
                    if error>1e-9:raise ValueError("Cost replay failed saved-result comparison")
                    cost_checks+=1
        books[case["id"]]={"settings":case,"trade_stats":stats(trades),"distribution":distribution(assets),
                           "signals_status":dict(signals),"no_trade_symbols":inactive,
                           "entry_sources":dict(Counter("both" if len(t["entry_reasons"])==2 else "A_only" if t["entry_reasons"]==["ema_band_pullback"] else "B_only" for t in trades)),
                           "years":{y:{"trade_stats_by_exit_year":stats(v["trades"]),"distribution":distribution(v["assets"])} for y,v in years.items()},
                           "repriced_costs":{c:{"distribution":distribution(v["assets"]),"trade_stats":stats(v["trades"])} for c,v in repriced.items()},
                           "assets":assets,"trades":trades}
    for key,sets in dates.items():
        if not sets["AND_BAND"]<=sets["AND_EXPANDED"]<=sets["OR"]:raise ValueError(f"Signal subset violated: {key}")
    if controls!=400 or cost_checks!=400 or len(dates)!=400:raise ValueError("Validation coverage incomplete")
    ict_assets=[];ict_trades=[];ict_active=set()
    for symbol in symbols:
        r=result_of(ict,ict["rows"][("primary_2r_10bps",symbol)]);s=r["strategy"]
        if s["settings"]["cost_bps"]!=10 or s["settings"]["target_r"]!=2:raise ValueError("Wrong ICT baseline")
        ict_assets.append(asset(symbol,s,r["baselines"]["buy_hold"]))
        ict_trades.extend({**t,"symbol":symbol} for t in s["trades"])
        if s["trades"]:ict_active.add(symbol)
    comparisons={}
    for case_id,book in books.items():
        case=book["settings"];or_id=f'or_{case["entry_mode"]}_{case["intraday_path"]}_10bps'
        comparisons[case_id]={"vs_ict":pair_effect(book["assets"],ict_assets),
                              "vs_or":pair_effect(book["assets"],books[or_id]["assets"]),
                              "common_ict_active_distribution":distribution([r for r in book["assets"] if r["symbol"] in ict_active])}
    return {"scope":"Same100pre snapshots2022-2025. Whole-model comparison, not an ICT-universal test or isolated buy-signal effect. AND is userrequested after OR outcomes; allretrospective.",
            "run_id":new_root.name,"books":books,"comparisons":comparisons,
            "ict":{"distribution":distribution(ict_assets),"trade_stats":stats(ict_trades),"active_symbols":sorted(ict_active),
                   "active_distribution":distribution([r for r in ict_assets if r["symbol"] in ict_active]),"assets":ict_assets},
            "validation":{"exact_or_controls":controls,"signal_subset_stock_scenarios":len(dates),"old_cost_curve_comparisons":cost_checks,
                          "max_cost_replay_error":cost_max_error,"max_trade_factor_error":nav_max_error,
                          "artifact_references":{k:len(v["manifest"]) for k,v in [("new",new),("old_or",old),("ict",ict)]}},
            "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/"plans/user_trend/analyze.py"]}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);p.add_argument("--old-or",type=Path,required=True);p.add_argument("--ict",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();out=analyze(args.run.resolve(),args.old_or.resolve(),args.ict.resolve());args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"validation":out["validation"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
