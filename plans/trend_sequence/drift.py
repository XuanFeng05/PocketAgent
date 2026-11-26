"""Monthly broad portfolio: preserve retained positions and transact only residual capital.

The same signal-date list and total stock budget drive every variant. The
original account sizes trades at the following session open. Here the opening
mark sizes amendments, too. All variants use the same proportional fees and
shared cash and fractional adjusted-price units.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plans.target20.analyze import curve_metrics, slices
from plans.target20.run_study import checked, read, sha, write
from plans.target20_followup.broad_sleeve import alternatives, cash_audit
from research_layer.technical_portfolio import _price_panel, _rebalance


def drift_amounts(reference, held, tolerance):
    """Meet the aggregate target with minimum two-way trading of retained names.

    New names start at the usual allocation. Remove nonmembers. For retained
    names move only in the required aggregate direction, largest deviation
    first, with lexical tie-breaking inherited from the sorted panel.
    """
    retained = (reference > 0) & (held > 0)
    target = np.where(retained, held, reference)
    difference = float(np.sum(reference) - np.sum(target))
    if abs(difference) > tolerance:
        sign = np.sign(difference)
        remaining = abs(difference)
        capacities = np.where(retained, np.maximum(sign*(reference-held), 0), 0)
        for index in np.argsort(-capacities, kind="stable"):
            amount = min(capacities[index], remaining)
            target[index] += sign * amount
            remaining -= amount
            if remaining <= tolerance:
                break
        if remaining > 10*tolerance:
            raise ArithmeticError("Cannot fund aggregate target from eligible positions")
    np.testing.assert_allclose(target.sum(), reference.sum(), rtol=0, atol=10*tolerance)
    if np.any(target < -tolerance):
        raise ArithmeticError("Negative target")
    return target


def sanity_checks():
    close = np.testing.assert_allclose
    close(drift_amounts(np.array([.4,.4,0]), np.array([.5,.3,0]), 1e-12), [.5,.3,0])
    # A 0.2 budget reduction needs one sale, not two sales and a buy.
    close(drift_amounts(np.array([.3,.3,0]), np.array([.55,.25,0]), 1e-12), [.35,.25,0])
    # Removed name funds a new, equal-sized slot, while a retained winner drifts.
    close(drift_amounts(np.array([.3,.3,0,.3]),np.array([.35,.25,.3,0]),1e-12), [.35,.25,0,.3])


def simulate(panel, days, schedule, start, end, bps, mode):
    symbols, opening, closing, tradeable = panel
    units = np.zeros(len(symbols))
    last_buy = [None]*len(symbols)
    cash, rate = 1., bps/10000
    prior_plan = None
    fills, curve, rebalances = [], [], []
    labels = Counter()
    fills_by_plan_change = Counter()
    decisions_by_change = Counter()
    for i, date in enumerate(days):
        day = date.date().isoformat()
        if day in schedule:
            decision = schedule[day]
            planned = decision["weights"]
            plan_id = (frozenset(planned), round(sum(planned.values()), 12))
            changed = plan_id != prior_plan
            if prior_plan is None:
                cause = "initial_allocation"
            elif plan_id[0] != prior_plan[0]:
                cause = "membership_changed"
            elif changed:
                cause = "budget_only"
            else:
                cause = "unchanged"
            decisions_by_change[cause] += 1
            prior_plan = plan_id
            prices = opening[i]
            held = units * prices
            pre_nav = cash + held.sum()
            weights = planned
            if mode != "equalize" and (mode == "budget_matched" or changed):
                reference = np.array([planned.get(s,0.) for s in symbols])*pre_nav/(1+rate)
                amounts = drift_amounts(reference,held,pre_nav*1e-13)
                weights = {s:float(a*(1+rate)/pre_nav) for s,a in zip(symbols,amounts) if a>0}
            if mode != "change_only" or changed:
                cash, trades, summary = _rebalance(day,decision["signal_date"],weights,symbols,
                                                   prices,tradeable[i],units,cash,rate,last_buy)
                fills.extend(trades)
                fills_by_plan_change[cause] += len(trades)
                summary["original_budget"] = sum(planned.values())
                rebalances.append(summary)
                for trade in trades:
                    index = symbols.index(trade["symbol"])
                    reason = ("liquidate" if trade["symbol"] not in planned else
                              "open_position" if held[index] == 0 else "resize_retained")
                    labels[reason] += 1
        invested = units*closing[i]
        nav = float(cash+invested.sum())
        curve.append({"date":day, "nav":nav, "cash":cash, "exposure":float(invested.sum()/nav),
                      "max_name_weight":float(invested.max()/nav)})
    summary = curve_metrics(curve,start,end)
    years, months = slices(curve)
    return {"equity_curve":curve,"fills":fills,"rebalances":rebalances,"summary":{
        **summary, "years":{y:v["total_return"] for y,v in years.items()},"worst_month":min(months.items(),key=lambda item:item[1]),
        "mean_exposure":float(np.mean([r["exposure"] for r in curve])),
        "max_name_weight":max(r["max_name_weight"] for r in curve),
        "fills":len(fills),"dates_traded":len({f["date"] for f in fills}),
        "fills_by_purpose":dict(labels),"turnover":sum(r["turnover"] for r in rebalances),
        "decisions_by_plan_change":dict(decisions_by_change), "fills_by_plan_change":dict(fills_by_plan_change),
        "fees":sum(f["fee"] for f in fills),"blocked":sum(len(r["blocked"]) for r in rebalances),
        "unchanged_plan_skips":len(schedule)-len(rebalances)}}


def study(run, broad_file):
    sanity_checks()
    hashes = read(run/"artifacts.json")["sha256"]
    def pinned(rel):
        path = run/rel
        checked(path,hashes[rel])
        return read(path)
    freeze = pinned("freeze.json")
    for path, expected in freeze["implementation"].items():
        checked(ROOT/path,expected)
    frames, ledger_data = {}, {}
    for symbol, meta in freeze["inputs"].items():
        path = Path(meta["path"])
        checked(path,meta["sha256"])
        frame = pd.read_csv(path)
        frames[symbol] = frame
        ledger_data[symbol] = {str(r.datetime)[:10]:(r.open,r.volume,r.close) for r in frame.itertuples(index=False)}
    calendar = Path(freeze["calendar"]["path"])
    checked(calendar, freeze["calendar"]["sha256"])
    cal = pd.read_csv(calendar)
    sessions = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_trading_day, "date"])).as_unit("ns")
    start, end = (freeze["configuration"][k] for k in ("start","end"))
    days = sessions[(sessions>=start)&(sessions<=end)]
    panel = _price_panel(frames,days)
    _, decisions, _ = alternatives(pinned("decisions/defensive5.json"),pinned("decisions/equal100.json"))
    schedule = {sessions[sessions.get_loc(pd.Timestamp(d["signal_date"]))+1].date().isoformat():d for d in decisions}
    if len(schedule)!=48:
        raise ValueError("Unexpected target schedule")
    original = read(broad_file)
    if original["source_artifact_manifest_sha256"] != sha(run/"artifacts.json"):
        raise ValueError("Wrong comparison account")
    report = {"cases":{},"checks":{"reference_checks":0},"script_sha256":sha(Path(__file__)),
              "prior_broad_sha256":sha(broad_file)}
    for bps in (0,10,20):
        rows = {}
        for mode in ("equalize","budget_matched","change_only"):
            account = simulate(panel,days,schedule,start,end,bps,mode)
            rows[mode] = account["summary"]
            report["checks"][f"{mode}_{bps}_ledger_error"] = cash_audit(account,ledger_data,bps)
            if mode=="equalize":
                old = original["by_cost"][str(bps)]["broad_basket"]
                for new_key,old_key in (("cagr_calendar","cagr_calendar"),("maximum_drawdown","maximum_drawdown"),
                                        ("turnover","turnover"),("fills","fills_count"),("fees","total_fees")):
                    np.testing.assert_allclose(rows[mode][new_key],old[old_key],rtol=0,atol=1e-10)
                report["checks"]["reference_checks"] += 1
        report["cases"][str(bps)] = rows
    return report


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--broad",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    report = study(args.run,args.broad)
    write(args.output,report)
    print(report["cases"]["10"])
