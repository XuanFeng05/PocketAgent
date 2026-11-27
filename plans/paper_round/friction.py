"""Monthly partial-adjustment scenarios registered in friction.md before the run."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plans.target20.analyze import curve_metrics, slices
from plans.target20.run_study import checked, read, sha, write
from plans.target20_followup.broad_sleeve import alternatives, cash_audit
from research_layer.technical_portfolio import _price_panel, _rebalance, run_technical_portfolio


def adjusted_order(reference, held, style):
    delta = reference - held
    if style == "full":
        return reference.copy()
    if style == "symmetric_half":
        return held + .5 * delta
    if style == "fast_reductions":
        return np.where(delta < 0, reference, held + .5 * delta)
    if style == "zero":
        return held.copy()
    raise ValueError(f"Unknown policy {style}")


def self_check():
    reference = np.array([0., .4, .2, .2])
    held = np.array([.2, .1, 0., .3])
    expected = {
        "full": reference, "zero": held, "symmetric_half": [.1, .25, .1, .25],
        "fast_reductions": [0., .25, .1, .2]
    }
    for mode, desired in expected.items():
        np.testing.assert_allclose(adjusted_order(reference, held, mode), desired, rtol=0, atol=1e-15)


def simulate(panel, days, schedule, style, bps):
    names, opens, closes, available = panel
    n = len(names)
    units, cash, last_buy = np.zeros(n), 1., [None] * n
    rate = bps / 10000
    fills, rebalances, curve = [], [], []
    for i, date in enumerate(days):
        day = date.date().isoformat()
        if day in schedule:
            decision = schedule[day]
            held = units * opens[i]
            before = cash + float(held.sum())
            original = decision["weights"]
            original_weights = np.array([original.get(s, 0.) for s in names])
            desired = adjusted_order(original_weights * before / (1 + rate), held, style)
            # The engine reserves proportional fees when translating weights into dollars.
            weights = original if style == "full" else {
                s: float(amount*(1+rate)/before) for s, amount in zip(names,desired) if amount>0}
            cash, trades, record = _rebalance(day, decision["signal_date"], weights, names, opens[i],
                available[i], units, cash, rate, last_buy)
            fills.extend(trades)
            marked = units * opens[i]
            post_nav = cash + float(marked.sum())
            record["intended_weights"] = original
            record["gross_target"] = float(original_weights.sum())
            record["achieved_exposure"] = float(marked.sum()/post_nav)
            record["off_list_weight"] = float(marked[original_weights == 0].sum()/post_nav)
            rebalances.append(record)
        marked = units * closes[i]
        nav = cash + float(marked.sum())
        curve.append({"date":day, "nav":nav, "cash":cash, "exposure":float(marked.sum()/nav),
                      "largest_weight":float(marked.max()/nav)})
    return {"fills":fills, "equity_curve":curve, "rebalances":rebalances}


def summarize(account, config):
    curve, fills, records = (account[k] for k in ("equity_curve","fills","rebalances"))
    result = curve_metrics(curve, config["start"], config["end"])
    years, months = slices(curve)
    gaps = [r["achieved_exposure"]-r["gross_target"] for r in records]
    trades_by_day = fills_by_date(fills)
    result.update(
        annual={y:v["total_return"] for y,v in years.items()},
        worst_month=min(months.items(), key=lambda x:x[1]),
        fills=len(fills), dates_traded=len({f["date"] for f in fills}),
        buys=sum(f["side"]=="buy" for f in fills), sells=sum(f["side"]=="sell" for f in fills),
        turnover=sum(r["turnover"] for r in records), fees=sum(f["fee"] for f in fills),
        mean_exposure=float(np.mean([d["exposure"] for d in curve])),
        max_name_weight=max(d["largest_weight"] for d in curve),
        mean_absolute_exposure_deviation=float(np.mean(np.abs(gaps))),
        max_target_exposure_overshoot=float(max(0.,max(gaps))),
        max_off_list_weight=max(r["off_list_weight"] for r in records),
        mean_off_list_weight=float(np.mean([r["off_list_weight"] for r in records])),
        blocked=sum(len(r["blocked"]) for r in records),
        orders_below_10bps_of_account=sum(f["notional"]<r["pre_nav"]*.001
            for r in records for f in trades_by_day.get(r["execution_date"],[])))
    return result


def fills_by_date(fills):
    groups={}
    for f in fills:
        groups.setdefault(f["date"],[]).append(f)
    return groups


def study(run, broad_report):
    self_check()
    manifest = read(run/"artifacts.json")["sha256"]
    def pinned(rel):
        path=run/rel
        checked(path,manifest[rel])
        return read(path)
    freeze=pinned("freeze.json")
    for path,expected in freeze["implementation"].items():
        checked(ROOT/path,expected)
    cfg=freeze["configuration"]
    frames, tables = {}, {}
    for symbol,item in freeze["inputs"].items():
        path=Path(item["path"])
        checked(path,item["sha256"])
        frame=pd.read_csv(path)
        frames[symbol]=frame
        tables[symbol]={str(r.datetime)[:10]:(r.open,r.volume,r.close) for r in frame.itertuples(index=False)}
    calendar_path=Path(freeze["calendar"]["path"])
    checked(calendar_path,freeze["calendar"]["sha256"])
    calendar=pd.read_csv(calendar_path)
    sessions=pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_trading_day,"date"])).as_unit("ns")
    days=sessions[(sessions>=cfg["start"]) & (sessions<=cfg["end"])]
    panel=_price_panel(frames,days)
    _, decisions, _ = alternatives(pinned("decisions/defensive5.json"),pinned("decisions/equal100.json"))
    schedule={sessions[sessions.get_loc(pd.Timestamp(d["signal_date"]))+1].date().isoformat():d for d in decisions}
    baseline=read(broad_report)
    if baseline["source_artifact_manifest_sha256"]!=sha(run/"artifacts.json") or len(schedule)!=48:
        raise ValueError("Archived source mismatch")
    zero=simulate(panel,days,schedule,"zero",10)
    assert not zero["fills"] and all(r["nav"]==1 for r in zero["equity_curve"])
    out={"source_sha256":sha(Path(__file__)), "source_run":run.name,
         "frozen_input_manifest_sha256":sha(run/"artifacts.json"),"by_cost":{},"checks":{}}
    for bps in (0,10,20):
        out["by_cost"][str(bps)]={}
        ref=run_technical_portfolio(frames,calendar,start=cfg["start"],end=cfg["end"],decisions=decisions,cost_bps=bps)
        for style in ("full","symmetric_half","fast_reductions"):
            account=simulate(panel,days,schedule,style,bps)
            row=summarize(account,cfg)
            row["target_met"] = row["cagr_calendar"]>.2 and row["maximum_drawdown"]<=.15
            out["by_cost"][str(bps)][style]=row
            out["checks"][f"{style}_{bps}_ledger_error"]=cash_audit(account,tables,bps)
            if style=="full":
                if ref["fills"]!=account["fills"]:
                    raise ValueError("Reference trade stream differs")
                np.testing.assert_allclose([d["nav"] for d in ref["equity_curve"]],
                    [d["nav"] for d in account["equity_curve"]],atol=1e-12,rtol=0)
                old=baseline["by_cost"][str(bps)]["broad_basket"]
                for key in ("cagr_calendar","maximum_drawdown","turnover","mean_exposure"):
                    np.testing.assert_allclose(row[key],old[key],rtol=0,atol=1e-12)
            if bps==10:
                out.setdefault("daily",{})[style]=account["equity_curve"]
                out.setdefault("monthly",{})[style]=account["rebalances"]
    # This post-outcome diagnostic imports only the gross target at each
    # decision's execution open from the contemporaneous fast-reductions account.
    matched_schedule={}
    for record in out["monthly"]["fast_reductions"]:
        day=record["execution_date"]
        original=schedule[day]
        budget=sum(record["target_weights"].values())
        original_budget=sum(original["weights"].values())
        pro_rata={s:w*budget/original_budget for s,w in original["weights"].items()} if original_budget else {}
        if not np.isclose(sum(pro_rata.values()),budget,atol=1e-12,rtol=0):
            raise ValueError("Unmatched stock budget")
        matched_schedule[day]={"signal_date":original["signal_date"],"weights":pro_rata}
    matched=simulate(panel,days,matched_schedule,"full",10)
    out["posthoc_matched_10bps"]=summarize(matched,cfg)
    out["checks"]["posthoc_budget_ledger_error"]=cash_audit(matched,tables,10)
    out["checks"]["zero_investment_exact"]=True
    out["checks"]["full_reference_fills_and_curves_verified_costs"]=[0,10,20]
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--broad",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    result=study(a.run,a.broad)
    write(a.output,result)
    for bps, items in result["by_cost"].items():
        for name, v in items.items():
            print(bps,name,v["cagr_calendar"],v["maximum_drawdown"],v["fills"],v["turnover"])
