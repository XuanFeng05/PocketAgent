"""Matched EFI event diagnostics and previous-low exits on fixed daily snapshots."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

import numpy as np
import pandas as pd


FEE = .001
EPS = 1e-12
PATHS = ("ohlc", "olhc")
POLICIES = ("original", "previous_low")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Snapshot mismatch: {path}")
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(source):
    manifest = read(source / "artifacts.json")["sha256"]
    frozen = read(checked(source / "freeze.json", manifest["freeze.json"]))
    cfg = frozen["configuration"]
    cal = pd.read_csv(checked(Path(frozen["calendar"]["path"]), frozen["calendar"]["sha256"]))
    sessions = cal.date[cal.is_trading_day.astype(str).str.lower().isin(["true", "1"])]
    dates = sessions[sessions.between(cfg["start"], cfg["end"])].tolist()
    data, originals = {}, {}
    above = {}
    for symbol, input_spec in sorted(frozen["inputs"].items()):
        frame = pd.read_csv(checked(Path(input_spec["path"]), input_spec["sha256"])).set_index("datetime").sort_index()
        if not (frame.symbol.eq(symbol).all() and frame.adjust.eq("pre").all() and frame.freq.eq("daily").all()):
            raise ValueError("Snapshot identity mismatch")
        above[symbol] = (frame.close > frame.close.rolling(126, min_periods=126).mean()).loc[dates].to_numpy()
        prior_lows = frame.low.shift(1).loc[dates].to_numpy(float)
        daily = frame.loc[dates]
        arrays = {key:daily[key].to_numpy(float) for key in ("open","high","low","close","volume")}
        tradeable = arrays["volume"] > 0
        if "is_tradeable" in daily:
            tradeable &= daily.is_tradeable.astype(str).str.lower().isin(["true","1"]).to_numpy()
        for path in PATHS:
            filename = f"cases/and_band_next_open_{path}_10bps/{symbol.replace('.', '_')}.json"
            strategy = read(checked(source / filename, manifest[filename]))["strategy"]
            assert strategy["settings"]["entry_combination"] == "AND_BAND"
            assert [d["date"] for d in strategy["decisions"]] == dates
            originals[(symbol,path)] = {
                "trades": strategy["trades"], "nav":[d["nav"] for d in strategy["equity_curve"]]}
            if path == "ohlc":
                dec = pd.DataFrame(strategy["decisions"]).set_index("date")
                efi = dec.efi2
                valid = dec.indicators_ready & dec.weekly_up & dec.branch_a_band_touch & tradeable & dec.atr.gt(0)
                passes = (efi.lt(0) & efi.gt(dec.efi2_previous) & dec.efi2_previous.gt(dec.efi2_previous2)
                          & efi.abs().lt(.5 * dec.efi2_previous.abs()))
                assert ((valid & passes) == dec.entry_ready).all()
                assert (pd.to_datetime(dec.weekly_available_at) <= pd.to_datetime(dec.daily_available_at)).all()
                data[symbol] = {**arrays, "tradeable":tradeable, "prior_low":prior_lows, "atr":dec.atr.to_numpy(),
                                "ema":dec.daily_ema26.to_numpy(), "valid":valid.to_numpy(bool), "pass":passes.to_numpy(bool)}
    breadth = np.mean(list(above.values()), axis=0)
    return data, dates, originals, breadth


def event_sample(data, dates, breadth):
    rows = []
    skipped = Counter()
    for symbol, f in data.items():
        for signal in np.flatnonzero(f["valid"]):
            tag = "pass" if f["pass"][signal] else "fail"
            entry = int(signal) + 1
            if entry + 19 >= len(dates):
                skipped[tag + "_no_full20"] += 1
                continue
            if not f["tradeable"][entry]:
                skipped[tag + "_blocked_open"] += 1
                continue
            rows.append({"symbol":symbol, "pass":tag=="pass", "signal":int(signal), "date":dates[signal],
                         "year":dates[signal][:4], "vol":f["atr"][signal] / f["close"][signal],
                         "breadth":breadth[signal]})
    benchmark = {}
    for i in {r["signal"] for r in rows}:
        entry = i+1
        for h in (5,20):
            last = i+h
            benchmark[i,h] = mean((f["close"][last]/f["open"][entry]*(1-FEE)**2-1) if f["tradeable"][entry] else 0
                                  for f in data.values())
    for r in rows:
        i=r["signal"]; f=data[r["symbol"]]; price=f["open"][i+1]
        for h in (5,20):
            value=f["close"][i+h]/price*(1-FEE)**2-1
            r[f"return{h}"]=value
            r[f"excess{h}"]=value-benchmark[i,h]
            r[f"mae{h}"]=max(0., 1-min(f["low"][i+1:i+h+1])/price)
            r[f"blocked_mark{h}"]=not f["tradeable"][i+h]
    return rows, dict(skipped)


def paired(pairs):
    if not pairs:
        return {"n":0}
    result={"n":len(pairs),"symbols":len({a["symbol"] for a,b in pairs}),
            "mean_gap_sessions":mean(a["signal"]-b["signal"] for a,b in pairs),
            "mean_abs_vol_difference":mean(abs(a["vol"]-b["vol"]) for a,b in pairs),
            "mean_abs_breadth_difference":mean(abs(a["breadth"]-b["breadth"]) for a,b in pairs)}
    for h in (5,20):
        for measure in ("return","excess","mae"):
            key=f"{measure}{h}"
            changes=[a[key]-b[key] for a,b in pairs]
            result[key]={"positive_mean":mean(a[key] for a,b in pairs), "control_mean":mean(b[key] for a,b in pairs),
                         "mean_difference":mean(changes), "median_difference":median(changes),
                         "positive_difference_fraction":mean(bool(g>EPS) for g in changes)}
        result[f"h{h}_blocked_marks"]=sum(int(a[f"blocked_mark{h}"])+int(b[f"blocked_mark{h}"]) for a,b in pairs)
    return result


def build_pairs(rows, disjoint=False):
    groups=defaultdict(list)
    for row in rows:
        groups[row["symbol"],row["year"]].append(row)
    pairs=[]
    occupancy=defaultdict(set)
    def window(row):
        return set(range(row["signal"]+1,row["signal"]+21))
    for (symbol,year),group in sorted(groups.items()):
        unused=[r for r in group if not r["pass"]]
        for a in sorted((r for r in group if r["pass"]),key=lambda r:r["signal"]):
            pool=[]
            for b in unused:
                gap=a["signal"]-b["signal"]
                dv=abs(math.log(a["vol"]/b["vol"]))
                db=abs(a["breadth"]-b["breadth"])
                if 1<=gap<=63 and dv<=math.log(1.5) and db<=.15+EPS:
                    if disjoint and (gap<20 or occupancy[symbol] & (window(a)|window(b))):
                        continue
                    pool.append((gap/63+dv/math.log(1.5)+db/.15,gap,b))
            if pool:
                b=min(pool,key=lambda t:(t[0],t[1]))[-1]
                unused.remove(b);pairs.append((a,b))
                if disjoint:occupancy[symbol].update(window(a)|window(b))
    return pairs


def match(rows):
    pairs=build_pairs(rows)
    occupancy=defaultdict(set);separate=[]
    for a,b in sorted(pairs,key=lambda p:(p[0]["signal"],p[0]["symbol"])):
        occupied=occupancy[a["symbol"]]
        windows=[set(range(r["signal"]+1,r["signal"]+21)) for r in (a,b)]
        if not (windows[0]&windows[1] or occupied&(windows[0]|windows[1])):
            separate.append((a,b));occupied.update(windows[0]|windows[1])
    second=build_pairs(rows,disjoint=True)
    return {"all":paired(pairs),"nonoverlap":paired(separate),"disjoint_rematch":paired(second),
            "disjoint_rematch_by_year":{year:paired([(a,b) for a,b in second if a["year"]==year]) for year in sorted({r["year"] for r in rows})},
            "disjoint_rematch_pairs":[[a["symbol"],a["date"],b["date"]] for a,b in second],
            "by_year":{year:paired([(a,b) for a,b in pairs if a["year"]==year]) for year in sorted({r["year"] for r in rows})},
            "sample_counts":dict(Counter("positive" if r["pass"] else "control" for r in rows)),
            "pairs":[{"positive":a,"control":b} for a,b in pairs]}


def isolated(f, dates, entry, path, policy):
    """Evaluate an independent position, with the first exit instruction latched."""
    opening=f["open"][entry];atr=f["atr"][entry-1]
    hard=opening-.5*atr;activation=opening+atr
    peak=opening;low_seen=opening;stop=hard;pending=None;below=0;activated=False
    active_before_exit_instruction=False

    def terminal(index,price,reason,phase):
        status="open" if reason=="end_mark" else "closed"
        net=price/opening*(1-FEE)**(1 if status=="open" else 2)-1
        return {"entry_signal_date":dates[entry-1],"entry_date":dates[entry],"entry_index":entry,
                "entry_phase":"open","adjusted_entry_price":float(opening),"atr":float(atr),
                "exit_date":dates[index] if status=="closed" else None,"exit_index":index,
                "exit_phase":phase,"adjusted_exit_price":float(price),"exit_reason":reason,"status":status,
                "net_return":float(net),"holding_bars":int(index-entry+1),
                "activated_before_first_exit":bool(active_before_exit_instruction),
                "peak_price":float(max(peak,price)),"giveback":float((max(peak,price)-price)/opening),
                "adverse":float(max(0.,1-min(low_seen,price)/opening))}
    for i in range(entry,len(dates)):
        op,hi,lo,cl=(f[k][i] for k in ("open","high","low","close"))
        can_sell=i>entry and f["tradeable"][i]
        if pending and can_sell:return terminal(i,op,pending,"open")
        for k,price in enumerate((op,hi,lo,cl) if path=="ohlc" else (op,lo,hi,cl)):
            if not pending and price<=stop and (k==0 or price<previous):
                kind="trailing_stop" if policy=="original" and activated else "hard_stop"
                active_before_exit_instruction=activated
                if can_sell:return terminal(i,op if k==0 else stop,("gap_" if k==0 else "")+kind,"open" if k==0 else "intraday")
                pending=kind+("_t_plus_one" if i==entry else "_non_tradeable")
            peak=max(peak,price);low_seen=min(low_seen,price)
            if peak>activation:activated=True
            if policy=="original" and activated:stop=max(hard,peak-.25*atr)
            previous=price
        below=below+1 if cl<f["ema"][i] else 0
        if not pending:
            if below>=3:pending="three_closes_below_ema26"
            elif policy=="previous_low" and activated and cl<f["prior_low"][i]:pending="activated_close_below_previous_low"
            if pending:active_before_exit_instruction=activated
    return terminal(len(dates)-1,f["close"][-1],"end_mark","mark")


def summarize(trades):
    closed=[t for t in trades if t["status"]=="closed"]
    if not closed:return {"trades":len(trades),"closed":0}
    vals=[t["net_return"] for t in closed]
    return {"trades":len(trades),"closed":len(closed),"open":len(trades)-len(closed),"wins":sum(x>EPS for x in vals),
            "win_rate":mean(x>EPS for x in vals),"mean_net":mean(vals),"median_net":median(vals),
            "mean_giveback":mean(t["giveback"] for t in closed),"mean_holding_bars":mean(t["holding_bars"] for t in closed),
            "never_activated_before_exit":sum(not t["activated_before_first_exit"] for t in closed),
            "mean_adverse":mean(t["adverse"] for t in closed),"exit_reasons":dict(Counter(t["exit_reason"] for t in closed))}


def simulate(data,dates,library,capacity,*,per_notional=False):
    # Eligibility checks here read only signal day; unwind already-held positions at their simulated fill.
    cash=1.;holdings={};last_exit={};trades=[];curve=[];planned=[]
    starts=defaultdict(list)
    for symbol,f in data.items():
        for i in np.flatnonzero(f["valid"] & f["pass"]):
            if i+1<len(dates):starts[int(i+1)].append(symbol)
    for i,day in enumerate(dates):
        initial=cash+sum(q*data[s]["open"][i] for s,(q,t) in holdings.items())
        def sell(phase):
            nonlocal cash
            for s,(q,t) in list(holdings.items()):
                if t["status"]=="closed" and t["exit_index"]==i and t["exit_phase"]==phase:
                    cash+=q*t["adjusted_exit_price"]*(1-FEE)
                    del holdings[s];last_exit[s]=i
        sell("open")
        for s in sorted(planned):
            if len(holdings)>=capacity or s in holdings or last_exit.get(s)==i or not data[s]["tradeable"][i]:continue
            budget=min(initial/capacity,cash)
            if budget<1e-12:continue
            t=library[s,i]
            # Archived single-symbol engine multiplies NAV by (1-fee) on entry.
            # The shared account instead pays fee on the actual traded notional.
            q=(budget/(1+FEE) if per_notional else budget*(1-FEE))/t["adjusted_entry_price"]
            cash-=budget
            t=dict(t)
            if per_notional:
                cash_multiplier=1-FEE if t["status"]=="closed" else 1.
                t["net_return"]=float(q*t["adjusted_exit_price"]*cash_multiplier/budget-1)
            holdings[s]=(q,t);trades.append({**t,"symbol":s,"units":float(q),"entry_budget":float(budget)})
        sell("intraday")
        val=sum(q*data[s]["close"][i] for s,(q,t) in holdings.items())
        nav=cash+val
        if cash< -1e-10 or nav<=0 or len(holdings)>capacity:raise ArithmeticError("Invalid ledger")
        curve.append({"date":day,"nav":float(nav),"cash":float(cash),"invested":float(val/nav),"positions":len(holdings),
                      "units":{s:float(q) for s,(q,t) in holdings.items()}})
        planned=[s for s in starts.get(i+1,[]) if s not in holdings and last_exit.get(s)!=i]
    return trades,curve


def reconcile_account(trades, curve, data):
    actions=defaultdict(list);units=defaultdict(float);cash=1.;worst_error=0.
    for t in trades:
        symbol=t["symbol"];qty=t["units"];entry=t["entry_index"];sale=t["exit_index"]
        if t["entry_phase"]!="open" or t["adjusted_entry_price"]!=data[symbol]["open"][entry]:raise ValueError("Wrong entry")
        actions[entry].append((1,symbol,qty,t["adjusted_entry_price"]))
        if t["status"]=="closed":
            if sale<=entry:raise ValueError("T+1 failure")
            actions[sale].append((0 if t["exit_phase"]=="open" else 2,symbol,-qty,t["adjusted_exit_price"]))
    for i,mark in enumerate(curve):
        for _,symbol,qty,price in sorted(actions[i]):
            units[symbol]+=qty
            notional=qty*price
            cash-=notional+abs(notional)*FEE
            if cash< -1e-10 or units[symbol]< -1e-10:raise ValueError("Negative inventory/cash")
        nav=cash+sum(q*data[s]["close"][i] for s,q in units.items())
        worst_error=max(worst_error,abs(nav-mark["nav"]),abs(cash-mark["cash"]))
    if worst_error>1e-10:raise ValueError("Shared cash ledger mismatch")
    return worst_error


def account_summary(trades,curve):
    nav=curve[-1]["nav"];peak=1.;dd=0.;annual={};last=1.
    for r in curve:peak=max(peak,r["nav"]);dd=max(dd,1-r["nav"]/peak)
    for y in sorted({d["date"][:4] for d in curve}):
        rows=[d for d in curve if d["date"][:4]==y];annual[y]=rows[-1]["nav"]/last-1;last=rows[-1]["nav"]
    return {"cagr":nav**.25-1,"total":nav-1,"mdd":dd,"fills":len(trades)+sum(t["status"]=="closed" for t in trades),
            "exposure":mean(d["invested"] for d in curve),"annual":annual,**summarize(trades)}


def exits(data, dates, originals):
    report={}
    for path in PATHS:
        scalar_return={};summaries={};raw={};paired_trades={};fixed={}
        for policy in POLICIES:
            lib={}
            for symbol,f in data.items():
                for s in np.flatnonzero(f["pass"] & f["valid"]):
                    if s+1<len(dates) and f["tradeable"][s+1]:
                        lib[symbol,int(s+1)]=isolated(f,dates,int(s+1),path,policy)
            all_trades=[];returns=[];fixed_trades=[]
            for symbol in data:
                trades,curve=simulate({symbol:data[symbol]},dates,lib,1)
                old=originals[symbol,path]
                if policy=="original":
                    np.testing.assert_allclose([d["nav"] for d in curve],old["nav"],atol=2e-12,rtol=0)
                    for t,u in zip(trades,old["trades"],strict=True):
                        for k in ("entry_date","exit_date","status","exit_reason","exit_phase"):
                            assert t[k]==u.get(k),(symbol,k,t,u)
                        assert abs(t["net_return"]-u["net_return"])<2e-12
                for t in old["trades"]:
                    fixed_trades.append({"symbol":symbol,**lib[symbol,dates.index(t["entry_date"])]})
                returns.append(curve[-1]["nav"]-1);all_trades.extend(trades)
            shared,curve=simulate(data,dates,lib,5,per_notional=True)
            summaries[policy]=account_summary(shared,curve)
            summaries[policy]["ledger_reconstruction_error"]=reconcile_account(shared,curve,data)
            scalar_return[policy]={"mean_total":mean(returns),"median_total":median(returns),**summarize(all_trades)}
            paired_trades[policy]=fixed_trades
            fixed[policy]=summarize(fixed_trades)
            raw[policy]={"shared_trades":shared,"shared_curve":curve,"individual_trades":all_trades,"fixed_trades":fixed_trades}
        pairs=list(zip(paired_trades["original"],paired_trades["previous_low"],strict=True))
        assert all((x["symbol"],x["entry_date"])==(z["symbol"],z["entry_date"]) for x,z in pairs)
        both_closed=[(x,y) for x,y in pairs if x["status"]==y["status"]=="closed"]
        diff=[y["net_return"]-x["net_return"] for x,y in both_closed]
        changed=[(x,y) for x,y in both_closed if abs(x["net_return"]-y["net_return"])>EPS]
        previous_losers=[(x,y) for x,y in both_closed if x["net_return"] < -EPS]
        previous_winners=[(x,y) for x,y in both_closed if x["net_return"] > EPS]
        fixed["paired"]={"both_closed":len(diff),"changed":len(changed),"better":sum(d>EPS for d in diff),
                         "worse":sum(d< -EPS for d in diff),"ties":sum(abs(d)<=EPS for d in diff),
                         "gain_mean":mean(diff),"gain_median":median(diff),"losers_to_winners":sum(y["net_return"]>EPS for x,y in previous_losers),
                         "winners_to_losers":sum(y["net_return"]< -EPS for x,y in previous_winners),
                         "changed_without_original_activation":sum(not x["activated_before_first_exit"] for x,y in changed),
                         "top_five_improvements":sorted(changed,key=lambda p:p[1]["net_return"]-p[0]["net_return"],reverse=True)[:5]}
        report[path]={"fixed":fixed,"individual_sequential":scalar_return,"shared":summaries,"records":raw}
    return report


def self_check():
    d=["2025-01-02","2025-01-03","2025-01-06","2025-01-07","2025-01-08"]
    f={"open":[100,100,105,104,102], "high":[101,103,108,107,103],
       "low":[99,99,103,102,100],"close":[100,102,107,102.5,101],
       "tradeable":[True]*5, "atr":[4.]*5, "prior_low":[98,99,99,103,102], "ema":[95.]*5}
    original=isolated(f,d,1,"ohlc","original")
    replacement=isolated(f,d,1,"ohlc","previous_low")
    assert original["exit_date"]==d[2] and original["adjusted_exit_price"]==107
    assert replacement["exit_date"]==d[4] and replacement["adjusted_exit_price"]==102
    assert replacement["exit_reason"]=="activated_close_below_previous_low"
    # Later prices are invisible to exits already booked.
    altered={key:values[:] for key,values in f.items()}
    for key in ("open","high","low","close"):
        altered[key][-1]*=3.
    assert original==isolated(altered,d,1,"ohlc","original")
    # Hit the entry-day hard stop before activation. Both exits stay latched through rebound.
    altered["low"][1]=97
    for policy in POLICIES:
        trade=isolated(altered,d,1,"olhc",policy)
        assert trade["exit_reason"]=="hard_stop_t_plus_one" and trade["adjusted_exit_price"]==105
        assert not trade["activated_before_first_exit"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    self_check()
    data,dates,originals,breadth=snapshot(args.run)
    events,skipped=event_sample(data,dates,breadth)
    entry=match(events);entry["excluded"]=skipped
    output={"run":args.run.name,"script_sha256":digest(Path(__file__)),"entry":entry,"exit":exits(data,dates,originals)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,allow_nan=False,default=lambda v:v.item()),encoding="utf-8")
    print(json.dumps({"file":str(args.output),"pairs":entry["all"]["n"],
                      "old_strict_reproduced":sum(v["fixed"]["original"]["closed"] for v in output["exit"].values())}))


if __name__=="__main__":main()
