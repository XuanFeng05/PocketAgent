"""Independent account metrics and target checks for the frozen monthly study."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import pandas as pd


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def curve_metrics(curve,start,end,initial=1.):
    days=(pd.Timestamp(end)-pd.Timestamp(start)).days+1
    if days<=0 or not curve:raise ValueError("Empty/invalid period")
    peak=initial;dd=0.;peak_date=start;trough_date=start;dd_peak=start;peak_at_dd=initial;trough_nav=initial
    for row in curve:
        if row["nav"]>peak:peak=row["nav"];peak_date=row["date"]
        decline=1-row["nav"]/peak
        if decline>dd:dd=decline;trough_date=row["date"];dd_peak=peak_date;peak_at_dd=peak;trough_nav=row["nav"]
    final=curve[-1]["nav"];cagr=(final/initial)**(365.25/days)-1
    return {"start":start,"end":end,"calendar_days":days,"initial_nav":initial,"final_nav":final,
            "total_return":final/initial-1,"cagr_calendar":cagr,"maximum_drawdown":dd,
            "drawdown_peak_date":dd_peak,"drawdown_trough_date":trough_date,"peak_nav":peak_at_dd,"trough_nav":trough_nav}


def slices(curve):
    by_year={};by_month={};previous=1.
    for year in sorted({r["date"][:4] for r in curve}):
        rows=[r for r in curve if r["date"].startswith(year)]
        by_year[year]=curve_metrics(rows,year+"-01-01",year+"-12-31",previous);previous=rows[-1]["nav"]
    previous=1.
    for month in sorted({r["date"][:7] for r in curve}):
        rows=[r for r in curve if r["date"].startswith(month)]
        by_month[month]=rows[-1]["nav"]/previous-1;previous=rows[-1]["nav"]
    if abs(math.prod(1+x["total_return"] for x in by_year.values())-curve[-1]["nav"])>1e-10:raise ValueError("Annual compounding differs")
    return by_year,by_month


def analyze(root):
    manifest=read(root/"artifacts.json")["sha256"]
    for rel,digest in manifest.items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError("Unverified artifact")
    freeze=read(root/"freeze.json");c=freeze["configuration"];summary=read(root/"summary.json")
    if summary["status"]!="completed" or summary["failed"] or summary["total"]!=24:raise ValueError("Incomplete study")
    expected={(p,cost) for p in c["policies"] for cost in c["cost_bps"]}
    actual=[(r["policy"],r["cost_bps"]) for r in summary["outcomes"]]
    if len(actual)!=24 or set(actual)!=expected:raise ValueError("Task identities differ")
    data={}
    for symbol,item in freeze["inputs"].items():
        if sha(Path(item["path"]))!=item["sha256"]:raise ValueError("Price input changed")
        f=pd.read_csv(item["path"]);data[symbol]={str(r["datetime"])[:10]:r for r in f.to_dict("records")}
    if sha(Path(freeze["calendar"]["path"]))!=freeze["calendar"]["sha256"]:raise ValueError("Calendar changed")
    outcomes=[];max_cash_error=max_nav_error=0.
    for row in summary["outcomes"]:
        if row["artifact_path"] not in manifest:raise ValueError("Unpinned result")
        result=read(root/row["artifact_path"]);curve=result["equity_curve"]
        if result["metrics"]!=row["metrics"]:raise ValueError("Metric identity differs")
        if (result["requested_start"],result["requested_end"],result["price_basis"],result["cost_bps"])!=(c["start"],c["end"],"pre",row["cost_bps"]):raise ValueError("Account identity differs")
        computed=curve_metrics(curve,c["start"],c["end"])
        years,months=slices(curve)
        fills=result["fills"];bydate=defaultdict(list)
        for fill in fills:bydate[fill["date"]].append(fill)
        cash=1.;units=defaultdict(float);fees=0.;symbols=set();traded_notional=0.;last_buy={}
        for day in curve:
            date=day["date"]
            for f in bydate.get(date,[]):
                symbol=f["symbol"];q=float(f["units"]);price=float(f["price"])
                if abs(price-float(data[symbol][date]["open"]))>1e-8:raise ValueError("Fill is not observed open")
                if float(data[symbol][date]["volume"])<=0:raise ValueError("Nontradeable fill")
                fee=q*price*row["cost_bps"]/10000
                if abs(f["fee"]-fee)>1e-10:raise ValueError("Fee mismatch")
                if f["side"]=="buy":cash-=q*price+fee;units[symbol]+=q;last_buy[symbol]=date
                elif f["side"]=="sell":
                    if last_buy.get(symbol)==date:raise ValueError("T+1 violation")
                    cash+=q*price-fee;units[symbol]-=q
                else:raise ValueError("Unknown fill side")
                if cash < -1e-10 or units[symbol]<-1e-10:raise ValueError("Leverage/short in ledger")
                fees+=fee;traded_notional+=q*price;symbols.add(symbol)
            marked=sum(q*float(data[symbol][date]["close"]) for symbol,q in units.items())
            nav=cash+marked;max_nav_error=max(max_nav_error,abs(nav-day["nav"]));max_cash_error=max(max_cash_error,abs(cash-day["cash"]))
            if abs(nav-day["nav"])>1e-9 or abs(cash-day["cash"])>1e-9:raise ValueError("Shared cash NAV mismatch")
            if marked/nav>1+1e-10 or nav<=0:raise ValueError("Invalid exposure/NAV")
            for symbol,h in day["holdings"].items():
                if abs(h["units"]-units[symbol])>1e-10 or abs(h["market_value"]-units[symbol]*float(data[symbol][date]["close"]))>1e-9:raise ValueError("Holding ledger differs")
        for key in ("total_return","maximum_drawdown","cagr_calendar"):
            if key in result["metrics"] and abs(result["metrics"][key]-computed[key])>1e-10:raise ValueError(f"Engine metric differs: {key}")
        win= computed["cagr_calendar"]>c["objective"]["account_cagr_min_exclusive"] and computed["maximum_drawdown"]<=c["objective"]["close_nav_max_drawdown_max"]
        outcomes.append({"id":row["id"],"policy":row["policy"],"cost_bps":row["cost_bps"],**computed,
                         "target_met":win,"years":years,"monthly_returns":months,"worst_month":min(months.items(),key=lambda x:x[1]),
                         "mean_exposure":mean(d["exposure"] for d in curve),"fills":len(fills),
                         "buy_fills":sum(f["side"]=="buy" for f in fills),"sell_fills":sum(f["side"]=="sell" for f in fills),
                         "trade_dates":len(bydate),"symbols_traded":len(symbols),"sum_fees_nav_units":fees,"gross_traded_notional_nav_units":traded_notional,
                         "gross_turnover":sum(v["turnover"] for v in result["rebalances"]),
                         "max_single_stock_weight":max((h["weight"] for d in curve for h in d["holdings"].values()),default=0.),
                         "max_positions":max(len(d["holdings"]) for d in curve),
                         "blocked_actions":sum(len(v["blocked"]) for v in result["rebalances"]),
                         "rebalance_count":len(result.get("rebalances",[])),"artifact_path":row["artifact_path"],
                         "early_period":curve_metrics([d for d in curve if d["date"]<"2025-01-01"],"2022-01-01","2024-12-31"),
                         "last_year_return":years["2025"]["total_return"]})
    return {"run_id":root.name,"objective":c["objective"],"outcomes":outcomes,"passed_cases":[r["id"] for r in outcomes if r["target_met"]],
            "validation":{"artifact_count":len(manifest),"input_count":len(data)+1,"ledger_cash_max_error":max_cash_error,"ledger_nav_max_error":max_nav_error},
            "scope":freeze["scope"],"script_sha256":sha(Path(__file__))}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    out=analyze(a.run.resolve());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(a.output),"passed":out["passed_cases"],"validation":out["validation"]},indent=2))
