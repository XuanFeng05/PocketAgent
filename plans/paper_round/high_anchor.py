"""Test a previously specified 52-week-high ranking on frozen monthly account signals."""
import argparse
from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plans.target20.run_study import read, write, sha
from plans.target20_followup.cash_diagnostic import metrics
from plans.target20_followup.rank_study import load_snapshot
from plans.target20_followup.broad_sleeve import alternatives
from plans.paper_round.recovery import recovery_choices
from research_layer.monthly_momentum import build_monthly_decisions
from research_layer.technical_portfolio import run_technical_portfolio


def price_panels(frames):
    indexed = {s: f.set_index(pd.to_datetime(f.datetime)) for s, f in frames.items()}
    panels = {key: pd.DataFrame({s: f[key] for s, f in indexed.items()}).sort_index()
              for key in ("close", "high", "open", "volume")}
    return panels


def anchor_orders(original, panels, use_own_vol):
    close, high = panels["close"], panels["high"]
    first = {s: high[s].first_valid_index() for s in high}
    previous = []
    orders = []
    for d in original:
        date = pd.Timestamp(d["signal_date"])
        lower = pd.Timestamp(date.date()-timedelta(weeks=52))
        meta = d["metadata"]
        k = meta["slots"]
        eligible = meta["ranked_symbols"]
        if any(first[s] > lower for s in eligible):
            raise ValueError("Incomplete 52-week history in frozen comparison")
        window = high.loc[(high.index > lower) & (high.index <= date), eligible]
        if window.isna().any().any():
            raise ValueError("Missing observation in high-price window")
        score = close.loc[date, eligible] / window.max()
        if not (score.gt(0) & score.le(1+1e-10)).all():
            raise ValueError("Invalid near-high score")
        ranked = sorted(eligible, key=lambda s: (-score[s],s))
        keep = [s for s in previous if s in ranked[:2*k]] if k==5 else []
        selected = keep + [s for s in ranked if s not in keep][:k-len(keep)]
        previous = selected
        scale = meta["risk_scale"]
        if k==5 and use_own_vol and selected:
            history = close.loc[:date,selected].tail(64)
            if len(history)!=64 or history.isna().any().any():
                raise ValueError("Return window incomplete")
            basket = history.pct_change(fill_method=None).iloc[1:].sum(axis=1)/k
            sigma = float(basket.std(ddof=1)*np.sqrt(252))
            scale = min(1.,.12/sigma) if sigma>0 else 1.
            if meta["breadth"] < .5:
                scale = 0.
        weights = {s:scale/k for s in selected if scale>0}
        if not use_own_vol and abs(sum(weights.values())-sum(d["weights"].values()))>1e-12:
            raise ValueError("Target exposure differs from matched control")
        orders.append({"signal_date":d["signal_date"], "weights":weights,
                       "metadata":{"ranked":ranked, "score":score.to_dict(), "selected":selected}})
    return orders


def rank_diagnostic(original, anchor, panels, sessions, end):
    rows = []
    omitted = []
    openings = [sessions[sessions.get_loc(pd.Timestamp(d["signal_date"]))+1] for d in original]
    for i, (d, h) in enumerate(zip(original,anchor,strict=True)):
        start = openings[i]
        stop = openings[i+1] if i+1<len(openings) else sessions[sessions<=pd.Timestamp(end)][-1]
        terminal_prices = panels["open"] if i+1<len(openings) else panels["close"]
        eligible = d["metadata"]["ranked_symbols"]
        available = [s for s in eligible if panels["volume"].at[start,s]>0 and panels["volume"].at[stop,s]>0]
        if len(available)<10:
            omitted.append(d["signal_date"])
            continue
        stock_returns = terminal_prices.loc[stop,available] / panels["open"].loc[start,available] - 1
        y = stock_returns.rank().to_numpy()
        x = pd.Series(h["metadata"]["score"])[available].rank().to_numpy()
        old = pd.Series(d["metadata"]["scores"])[available].rank().to_numpy()
        controls = np.column_stack([np.ones(len(old)), old])
        def residual(v):
            return v - controls@np.linalg.lstsq(controls,v,rcond=None)[0]
        def correlation(a,b):
            if min(np.std(a),np.std(b))<1e-10:
                return None
            return float(np.corrcoef(a,b)[0,1])
        rows.append({"signal_date":d["signal_date"], "start":str(start.date()), "stop":str(stop.date()),
                     "available":len(available), "endpoint_exclusions":len(eligible)-len(available),
                     "high_ic":correlation(x,y), "original_ic":correlation(old,y),
                     "partial_high_ic":correlation(residual(x),residual(y))})
    return {"months":rows, "fewer_than_ten":omitted,
            "means":{key:float(np.mean([r[key] for r in rows if r[key] is not None]))
                     for key in ("high_ic","original_ic","partial_high_ic")}}


def risk_stats(account, config):
    m = metrics(account,config)
    m["pass_both"] = m["cagr_calendar"]>.20 and m["maximum_drawdown"]<=.15
    m["max_positions"] = account["metrics"]["max_positions"]
    return m


def joint_orders(original, equal, high):
    """Keep both predeclared branches intact, including their respective risk budgets."""
    _, broad, _ = alternatives(original,equal)
    recovery, states = recovery_choices(original,broad)
    return [{"signal_date":d["signal_date"],
             "weights":dict(r["weights"] if st["weak"] else h["weights"])}
            for d,r,h,st in zip(original,recovery,high,states,strict=True)]


def study(run):
    manifest, config, frames, calendar = load_snapshot(run)
    panels = price_panels(frames)
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_trading_day,"date"]))
    period = {key:config[key] for key in ("start","end")}
    report = {"base_run":run.name, "source_sha256":sha(Path(__file__)), "families":{}, "prefix_checks":0,
              "source_manifest_sha256":sha(run/"artifacts.json")}
    for family in ("momentum10","defensive5"):
        decisions = build_monthly_decisions(frames,calendar,policy=family,**period)
        rel = f"decisions/{family}.json"
        if sha(run/rel)!=manifest[rel] or read(run/rel)!=decisions:
            raise ValueError("Baseline decision mismatch")
        variants = {"original":decisions, "matched":anchor_orders(decisions,panels,False)}
        if family=="defensive5":
            variants["own_vol"] = anchor_orders(decisions,panels,True)
        for variant,orders in variants.items():
            if variant=="original":
                continue
            for i in (0,11,23,35,47):
                prefix_date = decisions[i]["signal_date"]
                truncated = {k:f.loc[:prefix_date] for k,f in panels.items()}
                if anchor_orders(decisions[:i+1],truncated,variant=="own_vol")!=orders[:i+1]:
                    raise ValueError("Future observations affect earlier choices")
                report["prefix_checks"]+=1
        if family=="defensive5":
            rel="decisions/equal100.json"
            if sha(run/rel)!=manifest[rel]:
                raise ValueError("Archived broad universe differs")
            variants["joint_recovery"] = joint_orders(decisions,read(run/rel),variants["own_vol"])
        cases = {}
        for cost in (0,10,20):
            cases[str(cost)] = {}
            for variant,orders in variants.items():
                outcome = run_technical_portfolio(frames,calendar,decisions=orders,cost_bps=cost,**period)
                if variant=="original":
                    rel=f"cases/{family}_{cost}bps.json"
                    if sha(run/rel)!=manifest[rel] or read(run/rel)!=outcome:
                        raise ValueError("Archived control differs")
                cases[str(cost)][variant] = risk_stats(outcome,config)
        report["families"][family] = {"by_cost":cases,
            "mean_selection_overlap":float(np.mean([len(set(d["metadata"]["selected"])&set(h["metadata"]["selected"]))/
                                      max(1,len(d["metadata"]["selected"])) for d,h in zip(decisions,variants["matched"])])),
            "decisions":variants["matched"]}
        if family=="momentum10":
            report["diagnostic"]=rank_diagnostic(decisions,variants["matched"],panels,sessions,config["end"])
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=study(args.run)
    write(args.output,result)
    print({k:{n:round(m["cagr_calendar"]*100,2) for n,m in v["by_cost"]["10"].items()} for k,v in result["families"].items()})
