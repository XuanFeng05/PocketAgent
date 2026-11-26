"""Independent reconstruction of monthly selections using frozen input CSVs."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def audit(run):
    freeze=json.loads((run/"freeze.json").read_text(encoding="utf-8"));config=freeze["configuration"]
    frames={}
    for symbol,item in freeze["inputs"].items():
        f=pd.read_csv(item["path"]);f["datetime"]=pd.to_datetime(f["datetime"]);frames[symbol]=f.set_index("datetime").sort_index()
    cal=pd.read_csv(freeze["calendar"]["path"]);cal["date"]=pd.to_datetime(cal["date"])
    sessions=pd.DatetimeIndex(cal.loc[cal.is_trading_day.astype(str).str.lower().isin(["true","1"]),"date"])
    ends=pd.Series(sessions,index=sessions).groupby(sessions.to_period("M")).max()
    expected_dates=[]
    for date in ends:
        later=sessions[sessions>date]
        if len(later) and pd.Timestamp(config["start"])<=later[0]<=pd.Timestamp(config["end"]):expected_dates.append(date.date().isoformat())
    reports={};maximum_error=0.
    for policy in config["policies"]:
        decisions=json.loads((run/"decisions"/(policy+".json")).read_text(encoding="utf-8"))
        if [d["signal_date"] for d in decisions]!=expected_dates:raise ValueError("Policy decision schedule differs")
        previous=[];checks=0
        for decision in decisions:
            date=pd.Timestamp(decision["signal_date"]);ranked=[];histories={};valid=above=0;tradeable=[]
            for symbol,f in frames.items():
                sub=f.loc[:date]
                if sub.empty or sub.index[-1]!=date:continue
                closes=sub.close.to_numpy(float);row=sub.iloc[-1]
                flag=row.get("is_tradeable",True)
                flag=flag if isinstance(flag,(bool,np.bool_)) else str(flag).lower() in ("true","1")
                if row.volume>0 and flag:tradeable.append(symbol)
                if len(closes)<127:continue
                valid+=1;over=closes[-1]>float(np.mean(closes[-126:]));above+=over
                r63=closes[-1]/closes[-64]-1;r126=closes[-1]/closes[-127]-1
                if over and r63>0 and r126>0 and row.volume>0 and flag:
                    ranked.append((symbol,.5*(math.log1p(r63)+math.log1p(r126))))
                    histories[symbol]=pd.Series(closes[-63:]/closes[-64:-1]-1,index=sub.index[-63:])
            ranked.sort(key=lambda x:(-x[1],x[0]));names=[s for s,_ in ranked];K=5 if policy=="defensive5" else 10
            if policy=="cash":weights={}
            elif policy=="equal100":weights={s:1/len(frames) for s in sorted(tradeable)}
            else:
                retained=[] if policy=="momentum10" else [s for s in previous if s in names[:2*K]]
                selected=retained+[s for s in names if s not in retained][:K-len(retained)]
                previous=selected[:]
                scale=1.
                if policy in ("vol12_10","defensive10","defensive5") and selected:
                    history=pd.concat([histories[s] for s in selected],axis=1)
                    if history.isna().any().any() or len(history)!=63:raise ValueError("Risk observations do not align")
                    portfolio_returns=history.sum(axis=1)/K
                    sigma=float(portfolio_returns.std(ddof=1)*math.sqrt(252))
                    scale=min(1.,.12/sigma) if sigma>0 else 1.
                if policy in ("breadth10","defensive10","defensive5") and (above/valid if valid else 0)<.5:scale=0.
                weights={s:scale/K for s in selected if scale>0}
            actual={s:float(w) for s,w in decision["weights"].items() if abs(w)>1e-12}
            if set(actual)!=set(weights):raise ValueError(f"Selection mismatch {policy}/{date.date()}: {set(actual)^set(weights)}")
            error=max((abs(actual[s]-weights[s]) for s in weights),default=0.);maximum_error=max(maximum_error,error)
            if error>1e-10:raise ValueError(f"Weight mismatch {policy}/{date.date()}: {error}")
            checks+=1
        reports[policy]={"decisions":checks}
    return {"policies":reports,"total_decisions":sum(r["decisions"] for r in reports.values()),"maximum_weight_error":maximum_error,
            "method":"Direct CSV-to-target reconstruction; no imports of policy or account engine."}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();result=audit(a.run)
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2))
