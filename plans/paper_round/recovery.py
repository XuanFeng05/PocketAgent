"""Predeclared breadth recovery gate with account and matched-count timing controls."""
import argparse
from pathlib import Path
import sys
from statistics import mean, median

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plans.target20.run_study import checked, read, sha, write
from plans.target20_followup.broad_sleeve import alternatives, cash_audit
from plans.target20_followup.cash_diagnostic import metrics
from plans.target20_followup.rank_study import simulate
from research_layer.technical_portfolio import run_technical_portfolio, _price_panel


def recovery_choices(original, broad):
    prior = None
    decisions = []
    states = []
    for source, broad_target in zip(original, broad, strict=True):
        breadth = float(source["metadata"]["breadth"])
        weak = breadth < .5
        improving = weak and prior is not None and breadth > prior
        weights = broad_target["weights"] if improving else source["weights"]
        decisions.append({"signal_date": source["signal_date"], "weights": dict(weights)})
        states.append({"signal": source["signal_date"], "breadth": breadth, "prior": prior,
                       "weak": weak, "improving": improving, "weight": sum(weights.values())})
        prior = breadth
    return decisions, states


def group_returns(records):
    returns = [r["potential_gross"] for r in records]
    return {"count": len(records), "average": mean(returns) if records else None,
            "median": median(returns) if records else None,
            "positive": sum(v > 1e-12 for v in returns), "negative": sum(v < -1e-12 for v in returns),
            "geometric_selected_windows": float(np.prod(1 + np.array(returns)) - 1) if records else None,
            "end_suspension_marks":sum(bool(r["endpoint_marks"]) for r in records)}


def timing_distribution(orders, states, broad, symbols, panel, dates, execution_indices, actual):
    """Conditional randomization preserving trigger counts in each execution year."""
    samples = 256
    generator = np.random.default_rng(20261126)
    weights = np.array([[d["weights"].get(s, 0.) for s in symbols] for d in broad])
    mask = np.ones((len(orders), samples + 1), dtype=bool)
    by_year = {}
    for i, state in enumerate(states):
        if state["weak"]:
            mask[i, :] = False
            mask[i, 0] = state["improving"]
            by_year.setdefault(dates[execution_indices[i]].year, []).append(i)
    for group in by_year.values():
        n = sum(states[i]["improving"] for i in group)
        for col in range(1, samples + 1):
            mask[generator.choice(group, size=n, replace=False), col] = True
        if not np.all(mask[group].sum(axis=0) == n):
            raise ValueError("Timing control changed yearly number of weak-month entries")
    target = weights[:, None, :] * mask[:, :, None]
    execution = {day: i for i, day in enumerate(execution_indices)}
    nav, exposure, fills = simulate(target, execution, *panel, .001)
    np.testing.assert_allclose(nav[:,0], [r["nav"] for r in actual["equity_curve"]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(exposure[:,0], [r["exposure"] for r in actual["equity_curve"]], rtol=0, atol=1e-12)
    if fills[0] != len(actual["fills"]):
        raise ValueError("Randomization control replay differs")
    cagr = nav[-1] ** .25 - 1
    peaks = np.maximum.accumulate(np.vstack([np.ones(samples+1), nav]), axis=0)[1:]
    drawdown = (1-nav/peaks).max(axis=0)
    return {"count":samples,"seed":20261126,"cagr_at_or_below_actual":int((cagr[1:]<=cagr[0]+1e-12).sum()),
            "cagr_p10_median_p90":np.quantile(cagr[1:], [.1,.5,.9]).tolist(),
            "drawdown_p10_median_p90":np.quantile(drawdown[1:], [.1,.5,.9]).tolist(),
            "average_exposure_p10_median_p90":np.quantile(exposure[:,1:].mean(axis=0),[.1,.5,.9]).tolist(),
            "actual_exposure":float(exposure[:,0].mean()), "actual_cagr":float(cagr[0]),
            "counts_by_execution_year":{str(y):{"opportunity_months":len(g),"selected_months":int(mask[g,0].sum())} for y,g in by_year.items()},
            "scope":"Timing diagnostic labels use the realized per-year number of triggers. The 256 alternative schedules are not live strategies."}


def study(run, broad_path, cash_path):
    manifest = read(run/"artifacts.json")["sha256"]
    def pinned(relative):
        path=run/relative
        checked(path, manifest[relative])
        return read(path)
    freeze = pinned("freeze.json")
    for name, expected in freeze["implementation"].items():
        checked(ROOT/name, expected)
    frames, tables, breadth_data = {}, {}, {}
    for symbol, item in freeze["inputs"].items():
        path = Path(item["path"])
        checked(path,item["sha256"])
        f = pd.read_csv(path)
        frames[symbol] = f
        close = f.set_index("datetime")["close"]
        breadth_data[symbol] = (close > close.rolling(126).mean()).where(close.shift(126).notna())
        tables[symbol] = {str(r.datetime)[:10]:(r.open,r.volume,r.close) for r in f.itertuples(index=False)}
    calendar_path = Path(freeze["calendar"]["path"])
    checked(calendar_path, freeze["calendar"]["sha256"])
    calendar = pd.read_csv(calendar_path)
    config = freeze["configuration"]
    original = pinned("decisions/defensive5.json")
    _, broad, _ = alternatives(original, pinned("decisions/equal100.json"))
    proposed, states = recovery_choices(original,broad)
    if states[0]["weak"] or sum(s["weak"] for s in states)!=30:
        raise ValueError("Archived breadth choices differ")
    breadth = pd.DataFrame(breadth_data).mean(axis=1)
    for i, state in enumerate(states):
        if abs(breadth.at[state["signal"]]-state["breadth"])>1e-12:
            raise ValueError("Archived breadth differs from close-only recomputation")
        prefix,_ = recovery_choices(original[:i+1],broad[:i+1])
        if prefix[-1]!=proposed[i]: raise ValueError("Future decisions modified target")
    previous = read(broad_path)
    if previous["source_artifact_manifest_sha256"]!=sha(run/"artifacts.json"):
        raise ValueError("Inconsistent archived account")
    cash_report = read(cash_path)
    if cash_report["validation"]["baseline_manifest_sha256"]!=sha(run/"artifacts.json"):
        raise ValueError("Inconsistent gross windows")
    windows = cash_report["windows"]
    for state,w,b in zip(states,windows,broad,strict=True):
        if (state["signal"],state["weak"])!=(w["signal_date"],w["risk_off"]):
            raise ValueError("Inconsistent shadow windows")
        ret=0.; blocked=[]
        for symbol,weight in b["weights"].items():
            start_open,volume,_=tables[symbol][w["start_open"]]
            if volume<=0:continue
            o,v,c=tables[symbol][w["end"]]
            last = o if w["end_phase"]=="open" else c
            if v<=0:blocked.append(symbol)
            ret += weight*(last/start_open - 1)
        state.update(start_open=w["start_open"],end=w["end"],potential_gross=ret,
                     intended_broad_weight=sum(b["weights"].values()),endpoint_marks=blocked)
    # Diagnostic only. This multiplier is calibrated on full-period target
    # exposures and therefore cannot serve as a historically tradable rule.
    planned = sum(s["weight"] for s in states if s["weak"])
    available = sum(s["intended_broad_weight"] for s in states if s["weak"])
    multiplier = planned / available
    uniform = []
    for state, broad_order, original_order in zip(states,broad,original,strict=True):
        targets = ({s:multiplier*v for s,v in broad_order["weights"].items()}
                   if state["weak"] else original_order["weights"].copy())
        uniform.append({"signal_date":state["signal"], "weights":targets})
    uniform_notional = sum(sum(d["weights"].values()) for d,s in zip(uniform,states) if s["weak"])
    if abs(uniform_notional-planned)>1e-10:
        raise ValueError("Budget-matched control has unequal planned stock allocation")
    result_by_cost = {}
    error = 0.
    for cost in (0,10,20):
        summaries = {}
        for label,decisions in (("cash",original),("broad",broad),("recovery",proposed),
                                ("retrospective_uniform_budget",uniform)):
            account = run_technical_portfolio(frames,calendar,start=config["start"],end=config["end"],decisions=decisions,cost_bps=cost)
            m = metrics(account,config)
            if label=="cash" and account!=pinned(f"cases/defensive5_{cost}bps.json"):
                raise ValueError("Defensive account failed reproduction")
            if label=="broad":
                recorded = previous["by_cost"][str(cost)]["broad_basket"]
                if any(v!=recorded[k] for k,v in m.items()):raise ValueError("Broad account mismatch")
            error=max(error,cash_audit(account,tables,cost))
            m["worst_month"] = min((r for r in _months(account)),key=lambda p:p[1])
            m["target_hit"] = m["cagr_calendar"]>.2 and m["maximum_drawdown"]<=.15
            summaries[label]=m
            if cost==10 and label=="recovery": actual_account=account
        result_by_cost[str(cost)]=summaries
    dates=pd.DatetimeIndex(pd.to_datetime([r["date"] for r in actual_account["equity_curve"]])).as_unit("ns")
    syms,op,cl,active = _price_panel(frames,dates)
    indices = [dates.get_loc(pd.Timestamp(s["start_open"])) for s in states]
    timing = timing_distribution(proposed,states,broad,syms,(op,cl,active),dates,indices,actual_account)
    return {"cases":result_by_cost, "states":states, "weak_improving":group_returns([s for s in states if s["improving"]]),
            "weak_flat_or_falling":group_returns([s for s in states if s["weak"] and not s["improving"]]),
            "matched_count_comparison":timing, "max_cash_or_nav_error":error,"signal_prefix_checks":len(states),
            "retrospective_uniform_multiplier":multiplier, "summed_weak_month_target_exposures":planned,
            "reproduced_original_controls":3,"reproduced_broad_metrics":3,"script_sha256":sha(Path(__file__)),
            "source_manifest_sha256":sha(run/"artifacts.json"), "prior_broad_sha256":sha(broad_path),
            "prior_cash_sha256":sha(cash_path)}


def _months(account):
    ends={}
    for d in account["equity_curve"]:ends[d["date"][:7]]=d["nav"]
    previous=1.
    for month, nav in ends.items():
        yield (month,nav/previous-1)
        previous=nav


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--broad",type=Path,required=True)
    parser.add_argument("--cash",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=study(args.run,args.broad,args.cash)
    write(args.output,result)
    print({k:result["cases"]["10"][k]["cagr_calendar"] for k in ("cash","broad","recovery")})
