"""Frozen monthly shared-cash feasibility study. Offline existing snapshots only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from research_layer.technical_portfolio import run_technical_portfolio
from research_layer.monthly_momentum import build_monthly_decisions


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(p)
def checked(p,h):
    actual=sha(p)
    if actual!=h:raise ValueError(f"Hash mismatch: {p}")
    return actual
def implementation():
    paths=["research_layer/technical_portfolio.py","research_layer/monthly_momentum.py","research_layer/technical.py",
           "feature_layer/triple_screen.py","evaluation_layer/metrics/portfolio.py","plans/target20/run_study.py"]
    return {p:sha(ROOT/p) for p in paths}


def run(config_path,output):
    c=read(config_path)
    if len(c["symbols"])!=100 or len(set(c["symbols"]))!=100 or len(c["policies"])*len(c["cost_bps"])!=c["total_runs"]:raise ValueError("Frozen budget/universe differs")
    source=ROOT/c["snapshot_study"];parent=read(source/"artifacts.json")["sha256"]
    cf=source/"case_results"/(c["snapshot_case"]+".json");checked(cf,parent[cf.relative_to(source).as_posix()]);case=read(cf)
    child=source/"cases"/case["id"]/case["child_run_id"];mf=child/"artifacts.json";checked(mf,parent[mf.relative_to(source).as_posix()])
    manifest=read(mf)["sha256"];calfile=child/"calendar.csv";calhash=checked(calfile,manifest["calendar.csv"]);calendar=pd.read_csv(calfile)
    source_rows={r["symbol"]:r for r in case["outcomes"]}
    if set(source_rows)!=set(c["symbols"]):raise ValueError("Snapshot symbols differ")
    inputs={};bars={}
    for symbol in c["symbols"]:
        rel="inputs/"+symbol.replace(".","_")+".csv";p=child/rel;h=checked(p,manifest[rel])
        if h!=source_rows[symbol]["data_sha256"] or calhash!=source_rows[symbol]["calendar_sha256"]:raise ValueError("Snapshot row identity differs")
        inputs[symbol]={"path":str(p.resolve()),"sha256":h};bars[symbol]=pd.read_csv(p)
    runid=datetime.now(timezone.utc).strftime("portfolio_%Y%m%d_%H%M%S_")+uuid4().hex[:8];dest=output/runid;dest.mkdir(parents=True,exist_ok=False)
    identity=implementation();confighash=sha(config_path)
    freeze={"configuration":c,"config_sha256":confighash,"registered_at_utc":datetime.now(timezone.utc).isoformat(),
            "implementation":identity,"inputs":inputs,"calendar":{"path":str(calfile.resolve()),"sha256":calhash},
            "source_case_sha256":sha(cf),"source_manifest_sha256":sha(mf),"python":platform.python_version(),
            "scope":"One shared cash account; pre-adjusted synthetic units. Retrospective fixed100, no untouched holdout or brokerage reconstruction."}
    write(dest/"freeze.json",freeze);artifacts={"freeze.json":sha(dest/"freeze.json")};outcomes=[]
    print(json.dumps({"event":"registered","path":str(dest),"runs":c["total_runs"]}),flush=True)
    for policy in c["policies"]:
        if implementation()!=identity or sha(config_path)!=confighash:raise ValueError("Source changed")
        decisions=build_monthly_decisions(bars,calendar,start=c["start"],end=c["end"],policy=policy)
        df=dest/"decisions"/(policy+".json");write(df,decisions);artifacts[df.relative_to(dest).as_posix()]=sha(df)
        for cost in c["cost_bps"]:
            name=f"{policy}_{cost}bps";p=dest/"cases"/(name+".json")
            row={"id":name,"policy":policy,"cost_bps":cost}
            try:
                result=run_technical_portfolio(bars,calendar,start=c["start"],end=c["end"],decisions=decisions,cost_bps=cost)
                write(p,result);row.update(status="completed",metrics=result["metrics"],artifact_path=p.relative_to(dest).as_posix())
            except Exception as exc:
                row.update(status="failed",error=f"{type(exc).__name__}: {exc}");write(p,row)
            artifacts[p.relative_to(dest).as_posix()]=sha(p);outcomes.append(row)
            write(dest/"status.json",{"status":"running","recorded":len(outcomes),"total":c["total_runs"],"failed":sum(x["status"]!="completed" for x in outcomes)})
            print(json.dumps({"event":"case","id":name,"status":row["status"],"recorded":len(outcomes)}),flush=True)
    if implementation()!=identity or sha(config_path)!=confighash:raise ValueError("Source/config changed during run")
    for item in [freeze["calendar"],*inputs.values()]:checked(Path(item["path"]),item["sha256"])
    failed=sum(x["status"]!="completed" for x in outcomes)
    summary={"run_id":runid,"status":"completed" if not failed else "partial","total":len(outcomes),"failed":failed,"outcomes":outcomes}
    write(dest/"summary.json",summary);artifacts["summary.json"]=sha(dest/"summary.json");write(dest/"artifacts.json",{"sha256":artifacts})
    write(dest/"status.json",{k:summary[k] for k in ("run_id","status","total","failed")})
    print(json.dumps({"event":"finished","path":str(dest),"failed":failed}),flush=True)
    if failed:raise RuntimeError(f"{failed} cases failed; evidence preserved")
    return dest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=ROOT/"plans/target20/config.json");p.add_argument("--output",type=Path,default=ROOT/"runtime_layer/reports/target20")
    a=p.parse_args();run(a.config.resolve(),a.output.resolve())
