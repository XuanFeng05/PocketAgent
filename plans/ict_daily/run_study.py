"""Run the frozen daily ICT adaptation on verified, existing local snapshots."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd

from research_layer.ict_daily import run_ict_daily, VERSION, SETTINGS
from research_layer.technical import run_technical_backtest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def checked(path,expected):
    value=sha(path)
    if value!=expected:
        raise ValueError(f"Hash mismatch: {path}")
    return value


def implementation():
    paths=["research_layer/ict_daily.py","research_layer/technical.py","evaluation_layer/metrics/portfolio.py",
           "strategy_layer/registry.py","strategy_layer/base.py","strategy_layer/lightweight.py",
           "plans/ict_daily/run_study.py"]
    return {p:sha(ROOT/p) for p in paths}


def run(config_path,output):
    config=read(config_path)
    if config["model"]!=VERSION or len(config["symbols"])!=100 or len(set(config["symbols"]))!=100:
        raise ValueError("Expected explicit frozen CN100 model")
    cases=config["cases"]
    if len({c["id"] for c in cases})!=len(cases) or len(cases)*100!=config["max_tasks"]:
        raise ValueError("Case identities/task budget do not reconcile")
    source=(ROOT/config["snapshot_study"]).resolve()
    parent=read(source/"artifacts.json")["sha256"]
    case_file=source/"case_results"/(config["snapshot_case"]+".json")
    checked(case_file,parent[case_file.relative_to(source).as_posix()])
    source_case=read(case_file)
    child=source/"cases"/source_case["id"]/source_case["child_run_id"]
    child_manifest=child/"artifacts.json"
    checked(child_manifest,parent[child_manifest.relative_to(source).as_posix()])
    child_hashes=read(child_manifest)["sha256"]
    source_rows={r["symbol"]:r for r in source_case["outcomes"]}
    if set(source_rows)!=set(config["symbols"]):
        raise ValueError("Snapshot universe differs")
    calendar_file=child/"calendar.csv"
    calendar_hash=checked(calendar_file,child_hashes["calendar.csv"])
    calendar=pd.read_csv(calendar_file)
    inputs={}
    frames={}
    for symbol in config["symbols"]:
        relative="inputs/"+symbol.replace(".","_")+".csv"
        file=child/relative
        digest=checked(file,child_hashes[relative])
        if digest!=source_rows[symbol]["data_sha256"] or calendar_hash!=source_rows[symbol]["calendar_sha256"]:
            raise ValueError("Snapshot case and manifest differ")
        inputs[symbol]={"path":str(file),"sha256":digest}
        frames[symbol]=pd.read_csv(file)
    run_id=datetime.now(timezone.utc).strftime("ict_%Y%m%d_%H%M%S_")+uuid4().hex[:8]
    destination=output/run_id
    destination.mkdir(parents=True,exist_ok=False)
    frozen_implementation=implementation()
    freeze={"configuration":config,"config_sha256":sha(config_path),"registered_at_utc":datetime.now(timezone.utc).isoformat(),
            "settings":SETTINGS,"implementation":frozen_implementation,"inputs":inputs,
            "calendar":{"path":str(calendar_file),"sha256":calendar_hash},
            "source_case_sha256":sha(case_file),"source_child_manifest_sha256":sha(child_manifest),
            "environment":{"python":platform.python_version(),"pandas":pd.__version__},
            "scope":"Retrospective fixed-universe daily adaptation; separate technical NAVs; no untouched holdout."}
    write(destination/"freeze.json",freeze)
    outcomes=[]
    artifacts={"freeze.json":sha(destination/"freeze.json")}
    print(json.dumps({"event":"registered","run_id":run_id,"path":str(destination),"tasks":config["max_tasks"]}),flush=True)
    for case in cases:
        for symbol in config["symbols"]:
            if implementation()!=frozen_implementation or sha(config_path)!=freeze["config_sha256"]:
                raise ValueError("Implementation/config changed after registration")
            record={"case_id":case["id"],"symbol":symbol,"target_r":case["target_r"],"cost_bps":case["cost_bps"],
                    "input_sha256":inputs[symbol]["sha256"],"calendar_sha256":calendar_hash}
            result_path=destination/"cases"/case["id"]/(symbol.replace(".","_")+".json")
            try:
                bars=frames[symbol]
                strategy=run_ict_daily(bars,calendar,start=config["start"],end=config["end"],
                                       target_r=case["target_r"],cost_bps=case["cost_bps"])
                dates=pd.to_datetime(bars["datetime"])
                evaluation=bars.loc[dates.between(strategy["start"],strategy["end"])].copy()
                warmup=bars.loc[dates.lt(strategy["start"])].copy()
                baselines={name:run_technical_backtest(evaluation,name,cost_bps=case["cost_bps"],warmup_bars=warmup)
                           for name in ("cash","buy_hold")}
                result={"strategy":strategy,"baselines":baselines}
                write(result_path,result)
                record.update(status="completed",start=strategy["start"],end=strategy["end"],metrics=strategy["metrics"],
                              diagnostics=strategy["diagnostics"],buy_hold_return=baselines["buy_hold"]["metrics"]["total_return"],
                              artifact_path=result_path.relative_to(destination).as_posix())
            except Exception as exc:
                record.update(status="failed",error=f"{type(exc).__name__}: {exc}")
                write(result_path,{"error":record})
            artifacts[result_path.relative_to(destination).as_posix()]=sha(result_path)
            outcomes.append(record)
            write(destination/"status.json",{"status":"running","recorded_tasks":len(outcomes),"total_tasks":config["max_tasks"],
                                            "failed_tasks":sum(r["status"]!="completed" for r in outcomes),"current_case":case["id"],"current_symbol":symbol})
            if len(outcomes)%25==0:
                print(json.dumps({"event":"progress","completed":len(outcomes),"total":config["max_tasks"],"failures":sum(r["status"]!="completed" for r in outcomes)}),flush=True)
    if implementation()!=frozen_implementation:
        raise ValueError("Implementation changed during run")
    for item in inputs.values():
        checked(Path(item["path"]),item["sha256"])
    checked(calendar_file,calendar_hash)
    failures=sum(r["status"]!="completed" for r in outcomes)
    summary={"run_id":run_id,"status":"completed" if failures==0 else "partial","total_tasks":len(outcomes),
             "completed_tasks":len(outcomes)-failures,"failed_tasks":failures,"outcomes":outcomes}
    write(destination/"summary.json",summary)
    artifacts["summary.json"]=sha(destination/"summary.json")
    write(destination/"artifacts.json",{"sha256":artifacts,"external_inputs":"freeze.json pins verified original snapshot files; inputs were not modified or downloaded."})
    write(destination/"status.json",{k:summary[k] for k in ("run_id","status","total_tasks","completed_tasks","failed_tasks")})
    print(json.dumps({"event":"completed","path":str(destination),"tasks":len(outcomes),"failed":failures}),flush=True)
    if failures:
        raise RuntimeError(f"{failures} tasks failed; outputs retained")
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"plans/ict_daily/config.json")
    parser.add_argument("--output",type=Path,default=ROOT/"runtime_layer/reports/ict_daily")
    args=parser.parse_args()
    run(args.config.resolve(),args.output.resolve())


if __name__=="__main__":
    main()
