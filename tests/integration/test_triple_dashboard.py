from __future__ import annotations
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import pytest
from app_layer.backend.server import DashboardServer, RequestHandler, ThreadingHTTPServer

@pytest.fixture
def market(tmp_path):
    from data_layer.storage.partitioned_storage import save_kline_to_market_shard, save_trade_calendar_to_market_shard
    root=tmp_path/"market"
    dates=pd.bdate_range("2020-01-01","2021-06-30")
    for i,symbol in enumerate(["000001.SZ","000002.SZ","600036.SH"]):
        x=np.arange(len(dates));close=10+i+.015*x+.6*np.sin(x/3)
        opening=np.r_[close[0],close[:-1]]
        frame=pd.DataFrame({"symbol":symbol,"datetime":dates,"open":opening,"close":close,
                            "high":np.maximum(opening,close)+.15,"low":np.minimum(opening,close)-.15,
                            "volume":1e6,"amount":1e7,"freq":"daily","adjust":"pre","source":"synthetic_test","pctChg":0.})
        save_kline_to_market_shard(frame,root)
    dates=pd.date_range("2019-12-30","2021-07-04")
    save_trade_calendar_to_market_shard(pd.DataFrame({"date":dates,"is_trading_day":dates.weekday<5,"exchange":"CN","source":"synthetic_test"}),root)
    return root

@pytest.fixture
def dashboard(tmp_path,market):
    state=DashboardServer(research_root=tmp_path/"research",triple_data_root=market)
    server=ThreadingHTTPServer(("127.0.0.1",0),RequestHandler);server.app_state=state
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f"http://127.0.0.1:{server.server_port}",state
    server.shutdown();server.server_close();thread.join(timeout=5);state.jobs.shutdown()

def request(base,path,payload=None):
    req=Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=10) as response:return json.load(response)

def wait_run(base,job_id):
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        record=request(base,f"/api/triple/runs/{job_id}")["data"]
        if record["status"] in {"completed","partial","failed","cancelled","interrupted"}:return record
        time.sleep(.02)
    raise AssertionError("No terminal task status")

def payload(mode="batch"):
    return {"mode":mode,"symbols":["000001.SZ"],"start":"2020-02-03","end":"2021-06-30","costs_bps":[0,10],
            "settings":{"macd_fast":2,"macd_slow":3,"macd_signal":2,"warmup_weeks":2}}

@pytest.mark.parametrize("mode",["single","batch","annual","study"])
def test_all_modes_finish_with_progress_history_and_report(dashboard,mode):
    base,state=dashboard
    with urlopen(base+"/triple.html") as r:assert "三重滤网" in r.read().decode()
    spec=request(base,"/api/triple/spec")["data"]
    assert spec["symbol_count"]==3 and len(spec["default_cases"])==10
    body=payload(mode)
    if mode=="batch":body["symbols"]=["000001.SZ","000002.SZ"]
    if mode=="annual":body["windows"]=[{"label":"2020","start":"2020-02-03","end":"2020-12-31"},{"label":"2021","start":"2021-01-01","end":"2021-06-30"}]
    if mode=="study":body["cases"]=[{"id":"primary","label":"完整","settings":{},"variant":"full"},{"id":"no_pullback","label":"移除回调","settings":{},"variant":"drop_pullback"}]
    plan=request(base,"/api/triple/preview",body)["data"]
    job=request(base,"/api/triple/start",body)["data"]
    record=wait_run(base,job["job_id"])
    assert record["status"]=="completed",record
    assert record["progress"]["completed"]==plan["total_tasks"]
    assert record["progress"]["total"]==plan["total_tasks"]
    assert record["summary"]["failed_tasks"]==0
    assert record["report_links"]
    for link in record["report_links"]:
        with urlopen(base+link["url"]) as r:assert "三重滤网" in r.read().decode()
    assert request(base,"/api/triple/runs")["data"][0]["status"]=="completed"
    second=DashboardServer(research_root=state.research.output_root,triple_data_root=state.triple.data_root)
    try: assert second.triple.detail(job["job_id"])["summary"]==record["summary"]
    finally:second.jobs.shutdown()

@pytest.mark.parametrize("override",[{"symbols":[]},{"symbols":["999999.SZ"]},{"mode":"bad"},{"settings":[]},{"costs_bps":[True]},{"costs_bps":[float('nan')]},{"mode":"single","symbols":["000001.SZ","000002.SZ"]},{"mode":"annual","windows":[{"label":"bad","start":"2019-01-01","end":"2020-02-05"}]}])
def test_invalid_requests_are_rejected_before_job_creation(dashboard,override):
    base,state=dashboard
    with pytest.raises(HTTPError) as exc:request(base,"/api/triple/start",{**payload(),**override})
    assert exc.value.code==400
    assert not state.triple.list_runs()

def test_report_cannot_escape_output_directory(dashboard,tmp_path):
    from app_layer.backend.triple_controller import _write
    base,state=dashboard
    outside=tmp_path/"outside";outside.mkdir();(outside/"report.html").write_text("private")
    _write(state.triple._path("evil"),{"job_id":"evil","status":"completed","result_path":str(outside),"config":{"mode":"single"}})
    with pytest.raises(HTTPError) as exc:request(base,"/api/triple/runs/evil/report")
    assert exc.value.code==400


def test_running_cancel_persists_completed_progress(dashboard,monkeypatch):
    from app_layer.backend.triple_controller import _write
    base,state=dashboard
    began=threading.Event();release=threading.Event()
    def controlled(config,output_root,*,progress_callback,cancel_check):
        root=Path(output_root)/"controlled";root.mkdir(parents=True)
        progress_callback({"event":"run_started","run_id":"controlled","path":str(root),"completed":0,"total":2})
        _write(root/"progress.json",{"outcomes":[{"status":"completed"}]})
        progress_callback({"event":"task_completed","completed":1,"total":2})
        began.set();assert release.wait(5)
        assert cancel_check()
        _write(root/"status.json",{"status":"cancelled"})
        raise InterruptedError("after current step")
    monkeypatch.setattr("research_layer.triple_runner.run_triple_experiment",controlled)
    job=request(base,"/api/triple/start",payload())["data"]["job_id"]
    assert began.wait(5)
    try:
        assert request(base,f"/api/triple/runs/{job}")["data"]["progress"]["completed"]==1
        with pytest.raises(HTTPError):request(base,"/api/triple/start",payload())
        assert request(base,f"/api/triple/runs/{job}/cancel",{})["data"]["status"]=="cancelling"
    finally:release.set()
    record=wait_run(base,job)
    assert record["status"]=="cancelled"
    assert record["summary"]["outcomes"][0]["status"]=="completed"
    assert request(base,f"/api/triple/runs/{job}/cancel",{})["data"]["status"]=="cancelled"


@pytest.mark.parametrize("terminal",["partial","failed"])
def test_child_terminal_state_is_preserved(dashboard,monkeypatch,terminal):
    from app_layer.backend.triple_controller import _write
    base,state=dashboard
    def worker(config,output_root,**kwargs):
        root=Path(output_root)/"terminal";root.mkdir(parents=True)
        _write(root/"summary.json",{"status":terminal,"failed_tasks":1})
        return {"status":terminal,"run_id":"terminal","path":str(root)}
    monkeypatch.setattr("research_layer.triple_runner.run_triple_experiment",worker)
    job=request(base,"/api/triple/start",payload())["data"]["job_id"]
    record=wait_run(base,job)
    assert record["status"]==terminal
    assert record["summary"]["failed_tasks"]==1
    assert record["report_warning"]
    assert state.jobs.get_job(job)["status"]==terminal


def test_orphaned_request_is_interrupted_not_permanently_running(dashboard):
    from app_layer.backend.triple_controller import _write
    import os
    base,state=dashboard
    _write(state.triple._path("orphan"),{"job_id":"orphan","status":"running","owner":{"pid":os.getpid(),"session":"ended-session"},"config":{"mode":"single"},"result_path":None})
    record=request(base,"/api/triple/runs/orphan")["data"]
    assert record["status"]=="interrupted"
    assert request(base,"/api/triple/runs")["data"][0]["status"]=="interrupted"


def test_cancel_before_worker_starts(dashboard):
    base,state=dashboard
    occupied=[]
    release=threading.Event()
    def hold():
        release.wait(5)
    for _ in range(2):
        job=state.jobs.create_job("test_blocker");state.jobs.submit(job,hold);occupied.append(job)
    try:
        job=request(base,"/api/triple/start",payload())["data"]["job_id"]
        request(base,f"/api/triple/runs/{job}/cancel",{})
    finally:release.set()
    record=wait_run(base,job)
    assert record["status"]=="cancelled"
    assert record["result_path"] is None
