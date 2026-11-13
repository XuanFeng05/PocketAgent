"""Durable local-only jobs for the 100-stock Triple Screen workbench."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from threading import RLock
import time
from uuid import uuid4
from weakref import WeakValueDictionary

import psutil
from app_layer.backend.jobs import JobManager
from strategy_layer.triple_screen import resolve_triple_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "runtime_layer" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime_layer" / "reports" / "triple_dashboard"
VALID_MODES = {"single", "batch", "annual", "study"}
ACTIVE = {"queued", "running", "cancelling"}
TERMINAL = {"completed", "partial", "failed", "cancelled", "interrupted"}
_SESSIONS = WeakValueDictionary()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    deadline = time.monotonic() + 2
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def local_symbols(data_root=DEFAULT_DATA_ROOT, *, market="CN"):
    pattern = r"[A-Z]{1,5}(?:-[A-Z])?" if market == "US" else r"\d{6}_(SH|SZ|BJ)"
    return sorted({p.parent.name.replace("_", ".").upper()
                   for p in (Path(data_root) / "market_parts").glob("bucket=*/*/daily_pre.parquet")
                   if re.fullmatch(pattern, p.parent.name.upper())})


class TripleDashboardService:
    def __init__(self, jobs: JobManager, *, output_root=None, data_root=None, market="CN"):
        if market not in {"CN", "US"}:
            raise ValueError("Unsupported dashboard market")
        self.market = market
        self.api_prefix = "/api/triple-us" if market == "US" else "/api/triple"
        self.default_dates = ("2022-01-03", "2025-12-31") if market == "US" else ("2022-01-04", "2024-12-31")
        self.jobs = jobs
        self.data_root = Path(data_root or (PROJECT_ROOT / "runtime_layer/data_us" if market == "US" else DEFAULT_DATA_ROOT)).resolve()
        self.output_root = Path(output_root or (PROJECT_ROOT / "runtime_layer/reports/triple_dashboard_us" if market == "US" else DEFAULT_OUTPUT)).resolve()
        self.requests_root = self.output_root / "requests"
        self._lock = RLock()
        self.session = uuid4().hex
        self.owner = {"pid": os.getpid(), "process_started": psutil.Process().create_time(), "session": self.session}
        _SESSIONS[self.session] = self

    def spec(self):
        symbols = local_symbols(self.data_root, market=self.market)
        return {"mode": "triple_screen", "data_root": str(self.data_root), "price_basis": "pre",
                "market": self.market, "price_adjustment": "Yahoo拆股/分红调整OHLC" if self.market == "US" else "BaoStock前复权",
                "symbols": symbols, "symbol_count": len(symbols), "default_start": self.default_dates[0],
                "default_end": self.default_dates[1], "default_costs_bps": [10], "default_mode": "batch",
                "settings_defaults": resolve_triple_settings({"version": "triple_screen_us_pre_v1"} if self.market == "US" else None).to_dict(),
                "default_cases": _read(PROJECT_ROOT / "plans/triple_screen/m6_study_cases.json"),
                "modes": [{"id": k, "label": v} for k,v in [("single","单股回测"),("batch","股票批量回测"),("annual","年度验证"),("study","参数与滤网对照")]],
                "limitations": ["仅使用本地前复权行情，不下载。", "标准化独立账户，非共享资金组合。", "取消在一个股票/成本任务结束后生效，保留已有结果。"]}

    def _path(self, job_id):
        if not isinstance(job_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,140}",job_id):
            raise ValueError("Invalid Triple Screen job identifier")
        return self.requests_root / f"{job_id}.json"

    def _update(self, job_id, **updates):
        with self._lock:
            record = _read(self._path(job_id))
            record.update(updates, updated_at=_now())
            _write(self._path(job_id), record)
            return record

    def _normalize(self, payload):
        if not isinstance(payload,dict):
            raise ValueError("请求必须是对象")
        mode = payload.get("mode", "batch")
        if mode not in VALID_MODES:
            raise ValueError("未知研究方式")
        symbols = payload.get("symbols")
        if not isinstance(symbols,list) or not symbols or any(not isinstance(s,str) for s in symbols):
            raise ValueError("请选择至少一只本地股票")
        symbols = [s.strip().upper() for s in symbols]
        if len(symbols)>100 or len(set(symbols))!=len(symbols):
            raise ValueError("股票最多100只且不能重复")
        if set(symbols)-set(local_symbols(self.data_root, market=self.market)):
            raise ValueError("股票不在本地前复权清单")
        if mode=="single" and len(symbols)!=1:
            raise ValueError("单股模式只能选择一只股票")
        dates = {}
        for field,default in zip(("start", "end"), self.default_dates):
            value=payload.get(field,default)
            if not isinstance(value,str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}",value):
                raise ValueError("日期需为YYYY-MM-DD")
            dates[field]=datetime.strptime(value,"%Y-%m-%d").date().isoformat()
        if dates["start"]>dates["end"]:
            raise ValueError("开始日期不能晚于结束日期")
        costs=payload.get("costs_bps",[10])
        if not isinstance(costs,list) or not 1<=len(costs)<=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<10000 for x in costs) or len(set(costs))!=len(costs):
            raise ValueError("成本需为1到4个不重复的非负有限bps值")
        settings=payload.get("settings",{})
        if not isinstance(settings,dict) or set(settings)-{"macd_fast","macd_slow","macd_signal","force_span","warmup_weeks"}:
            raise ValueError("规则参数字段无效")
        resolved=resolve_triple_settings({**settings, "version": "triple_screen_us_pre_v1"} if self.market == "US" else settings).to_dict()
        config={"mode":mode,"source_type":"market","source_path":str(self.data_root),"adjust":"pre",
                "symbols":symbols,**dates,"settings":resolved,"cost_scenarios_bps":costs,"research_scope":"exploratory"}
        if self.market == "US":
            config["calendar_path"] = str(self.data_root / "calendar.csv")
        if mode=="annual":
            windows=payload.get("windows")
            if windows is None:
                windows=[{"label":str(y),"start":max(dates["start"],f"{y}-01-01"),"end":min(dates["end"],f"{y}-12-31")}
                         for y in range(int(dates["start"][:4]),int(dates["end"][:4])+1)]
            config["windows"]=windows
        if mode=="study":
            config["cases"]=payload.get("cases",self.spec()["default_cases"])
            config["max_tasks"]=2000
        self._preflight(config)
        return config

    def _child_config(self, config):
        child={k:deepcopy(v) for k,v in config.items() if k not in {"windows","cases"}}
        child["mode"]="triple_screen"
        if config["mode"]=="annual":
            child.pop("start");child.pop("end")
            child["research_scope"]="exploratory_validation"
        if config["mode"]=="study":
            child["research_scope"]="exploratory_study"
        return child

    def _preflight(self, config):
        from research_layer.triple_runner import normalize_triple_config
        from research_layer.triple_validation import normalize_validation_protocol
        from research_layer.triple_study import normalize_study_protocol
        child=self._child_config(config)
        if config["mode"]=="annual":
            protocol=normalize_validation_protocol(child,config["windows"])
            for w in protocol["windows"]:
                if w["start"]<config["start"] or w["end"]>config["end"]:
                    raise ValueError("年度窗口需位于所选起止日期内")
            factor=len(protocol["windows"])
        elif config["mode"]=="study":
            protocol=normalize_study_protocol(child,config["cases"])
            factor=len(protocol["cases"])
        else:
            normalize_triple_config(child);factor=1
        total=len(config["symbols"])*len(config["cost_scenarios_bps"])*factor
        if total>2000:
            raise ValueError("本次超过2000任务预算，请缩小范围")
        return total

    def preview(self,payload):
        config=self._normalize(payload)
        return {"config":config,"total_tasks":self._preflight(config),"price_basis":"pre"}

    def start(self,payload):
        config=self._normalize(payload)
        with self._lock:
            if any(r["status"] in ACTIVE for r in self.list_runs()):
                raise ValueError("已有三重滤网任务，请先等待完成或取消")
            job_id=self.jobs.create_job("triple_screen",title=f"三重滤网 · {config['mode']}")
            total=self._preflight(config)
            _write(self._path(job_id),{"job_id":job_id,"status":"queued","created_at":_now(),"updated_at":_now(),
                 "config":config,"owner":self.owner,"result_path":None,"progress":{"completed":0,"total":total,"message":"等待运行"}})
            self.jobs.submit(job_id,self._run,job_id)
        return {"job_id":job_id,"status":"queued","mode":config["mode"],"symbol_count":len(config["symbols"])}

    def _run(self,job_id):
        from research_layer.triple_runner import run_triple_experiment
        from research_layer.triple_validation import run_triple_validation
        from research_layer.triple_study import run_triple_study
        record=self._update(job_id,status="running")
        config=record["config"];mode=config["mode"]
        def cancelled():
            return self.jobs.is_cancel_requested(job_id)
        def progress(event):
            updates={"progress":{"completed":event.get("completed",0),"total":event.get("total",record["progress"]["total"]),
                                  "message":event.get("message") or event.get("event", "运行中")}}
            if event.get("event")=="run_started":
                updates.update(result_path=str(event["path"]),run_id=event["run_id"])
            self._update(job_id,**updates)
        try:
            if cancelled():raise InterruptedError("开始前已取消")
            root=self.output_root / "runs" / job_id
            kwargs={"progress_callback":progress,"cancel_check":cancelled}
            if mode in {"single","batch"}:
                result=run_triple_experiment(self._child_config(config),root,**kwargs)
            elif mode=="annual":
                result=run_triple_validation(self._child_config(config),config["windows"],root,**kwargs)
            else:
                result=run_triple_study(self._child_config(config),config["cases"],root,**kwargs)
            final=result["status"]
            if final not in {"completed","partial","failed"}:raise ValueError("未知底层任务终态")
            warning=None
            if mode in {"single","batch"}:
                try:
                    from research_layer.triple_report import render_triple_report
                    render_triple_report(result["path"])
                except Exception as exc:
                    warning=f"结果已保存，但HTML报告未生成：{exc}"
            self._update(job_id,status=final,run_id=result["run_id"],result_path=str(result["path"]),report_warning=warning)
            self.jobs.update_job(job_id,status=final,progress=1.0,message=final)
            return {"status":final,"path":str(result["path"])}
        except InterruptedError as exc:
            self._update(job_id,status="cancelled",error=None,message=str(exc))
            raise
        except Exception as exc:
            self._update(job_id,status="failed",error=str(exc))
            raise

    def _reconcile(self,record):
        if record.get("status") not in ACTIVE:return record
        owner=record.get("owner") or {}
        service=_SESSIONS.get(owner.get("session"))
        if service:
            job=service.jobs.get_job(record["job_id"])
            if job and job["status"] in {"queued","running"}:return record
            fresh=_read(self._path(record["job_id"]))
            if fresh.get("status") not in ACTIVE:return fresh
            if job and job["status"] in {"cancelled","failed"}:
                return self._update(record["job_id"],status=job["status"],error=job.get("error"))
        elif owner.get("pid") and owner.get("pid")!=os.getpid():
            try:
                p=psutil.Process(int(owner["pid"]))
                if abs(p.create_time()-float(owner.get("process_started",-1)))<.01:return record
            except psutil.AccessDenied:return record
            except (psutil.NoSuchProcess,ValueError,TypeError):pass
        # Worker disappeared: prefer an actual terminal disk result if available.
        result_path=record.get("result_path")
        if result_path:
            run=Path(result_path).resolve()
            if run.is_relative_to(self.output_root) and (run/"status.json").exists():
                state=_read(run/"status.json").get("status")
                if state in TERMINAL:
                    return self._update(record["job_id"],status=state)
        return self._update(record["job_id"],status="interrupted",error="所属进程已结束；保存的记录与已完成结果保留")

    def list_runs(self):
        with self._lock:
            rows=[]
            for p in self.requests_root.glob("*.json"):
                rows.append(self._reconcile(_read(p)))
            return sorted(rows,key=lambda r:r.get("created_at",""),reverse=True)

    def _result_root(self,record):
        if not record.get("result_path"):return None
        root=Path(record["result_path"]).resolve()
        if not root.is_relative_to(self.output_root):raise ValueError("结果路径超出三重滤网输出目录")
        return root

    def _report_targets(self,record,summary):
        root=self._result_root(record)
        if root is None:return []
        mode=record["config"]["mode"]
        if mode in {"single","batch"}:
            return [("逐股完整报告",root)] if (root/"summary.json").exists() else []
        entries=summary.get("windows",[]) if mode=="annual" else summary.get("cases",[])
        links=[]
        for e in entries:
            raw=e.get("path") if mode=="annual" else e.get("child_path")
            if not raw:continue
            child=Path(raw).resolve()
            if not child.is_relative_to(root):raise ValueError("子报告路径不合法")
            if (child/"summary.json").exists():links.append((str(e.get("label") or e.get("id")),child))
        return links

    def detail(self,job_id):
        with self._lock:
            record=self._reconcile(_read(self._path(job_id)))
        root=self._result_root(record)
        if root and record["status"] in TERMINAL:
            file=root/"summary.json"
            if not file.exists():file=root/"partial_result.json"
            if file.exists():record["summary"]=_read(file)
            elif (root/"progress.json").exists():record["summary"]=_read(root/"progress.json")
            targets=self._report_targets(record,record.get("summary",{}))
            record["report_links"]=[{"label":label,"url":f"{self.api_prefix}/runs/{job_id}/reports/{i}"} for i,(label,_) in enumerate(targets)]
            if targets:record["report_url"]=record["report_links"][0]["url"]
        return record

    def report(self,job_id,index=0):
        with self._lock:
            record=self.detail(job_id)
            targets=self._report_targets(record,record.get("summary",{}))
            if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(targets):
                raise FileNotFoundError("该报告不存在")
            from research_layer.triple_report import render_triple_report
            report=targets[index][1]/"report.html"
            # Run artifacts are immutable; generate a missing derived view once.
            if not report.exists():report=render_triple_report(targets[index][1])
            return report

    def cancel(self,job_id):
        with self._lock:
            record=self._reconcile(_read(self._path(job_id)))
            if record["status"] not in ACTIVE:return {"job_id":job_id,"status":record["status"]}
            service=_SESSIONS.get(record.get("owner",{}).get("session"))
            if service is None or not service.jobs.request_cancel(job_id):
                raise ValueError("任务属于另一运行进程，请在原实例取消")
            self._update(job_id,status="cancelling")
        return {"job_id":job_id,"status":"cancelling"}
