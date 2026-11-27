"""Evaluate five locked policies over the following five local months, carrying positions."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plans.paper_round.high_anchor import anchor_orders, joint_orders, price_panels
from plans.paper_round.recovery import recovery_choices
from plans.target20.analyze import curve_metrics
from plans.target20.run_study import read, sha, write
from plans.target20_followup.broad_sleeve import alternatives
from plans.target20_followup.rank_study import load_snapshot
from research_layer.monthly_momentum import build_monthly_decisions
from research_layer.technical_portfolio import run_technical_portfolio


def extend_frames(frames, old_calendar, local, original_end, end):
    local_calendar = pd.read_parquet(local / "_global/trade_calendar.parquet")
    def calendar_flags(calendar):
        values = calendar.set_index(pd.to_datetime(calendar.date)).sort_index().is_trading_day
        return values.astype(str).str.lower().map({"true":True, "false":False, "1":True, "0":False})

    old_flags, current_flags = calendar_flags(old_calendar), calendar_flags(local_calendar)
    pd.testing.assert_series_equal(old_flags.loc[:end],current_flags.loc[:end],check_index_type=False)
    expected = current_flags.loc[original_end:end].iloc[1:]
    expected_dates = expected[expected].index
    output, hashes = {}, {}
    largest_price_error = 0.
    prefix_rows = 0
    for symbol, archived in frames.items():
        files = list(local.glob(f"bucket=*/{symbol.replace('.', '_')}/daily_pre.parquet"))
        if len(files)!=1:
            raise ValueError(f"Expected a unique local price file for {symbol}")
        hashes[symbol] = sha(files[0])
        prices = pd.read_parquet(files[0])
        prices = prices.set_index(pd.to_datetime(prices.datetime)).sort_index().loc[:end].copy()
        if prices.index.has_duplicates or not (prices.symbol.eq(symbol) & prices.freq.eq("daily") & prices.adjust.eq("pre") & prices.source.eq("baostock")).all():
            raise ValueError(f"Invalid daily data for {symbol}")
        old = archived.set_index(pd.to_datetime(archived.datetime)).sort_index()
        check = prices.loc[old.index[0]:original_end]
        pd.testing.assert_index_equal(check.index,old.index,exact=False)
        columns = ["open","high","low","close"]
        error = np.abs(check[columns].to_numpy()-old[columns].to_numpy()).max()
        largest_price_error = max(largest_price_error,float(error))
        np.testing.assert_allclose(check[columns],old[columns],atol=1e-8,rtol=1e-8)
        np.testing.assert_array_equal(check.volume,old.volume)
        prefix_rows += len(old)
        suffix = prices[prices.index>original_end].copy()
        if list(suffix.index)!=list(expected_dates):
            raise ValueError(f"Session coverage differs for {symbol}")
        if not np.isfinite(suffix[columns+["volume"]].to_numpy()).all() or (suffix[columns] <= 0).any().any() or (suffix.volume < 0).any():
            raise ValueError(f"Invalid prices in extension for {symbol}")
        if (suffix.low > suffix[["open","close"]].min(axis=1)).any() or (suffix.high < suffix[["open","close"]].max(axis=1)).any():
            raise ValueError(f"OHLC bounds violated for {symbol}")
        suffix["datetime"] = suffix.index.strftime("%Y-%m-%d")
        suffix = suffix.rename(columns={"amount":"source_amount_unused"}).reset_index(drop=True)
        output[symbol] = pd.concat([archived,suffix],ignore_index=True)
    return output, {"overlap_rows":prefix_rows,"max_price_difference":largest_price_error,
                    "new_sessions_per_stock":len(expected_dates),"local_file_sha256":hashes,
                    "first":str(expected_dates[0].date()),"last":str(expected_dates[-1].date())}


def targets(frames, calendar, start, end):
    defensive = build_monthly_decisions(frames,calendar,start=start,end=end,policy="defensive5")
    equal = build_monthly_decisions(frames,calendar,start=start,end=end,policy="equal100")
    _, broad, _ = alternatives(defensive,equal)
    recovery, _ = recovery_choices(defensive,broad)
    high = anchor_orders(defensive,price_panels(frames),True)
    return {"defensive":defensive,"broad":broad,"recovery":recovery,"high":high,
            "joint":joint_orders(defensive,equal,high)}


def study(run, local, recovery_report, high_report):
    _, cfg, old, cal = load_snapshot(run)
    end="2026-05-31"
    frames, checks = extend_frames(old,cal,local,cfg["end"],end)
    original = targets(old,cal,cfg["start"],cfg["end"])
    extended = targets(frames,cal,cfg["start"],end)
    anchor = read(high_report)["families"]["defensive5"]["by_cost"]["10"]
    recovered = read(recovery_report)["cases"]["10"]
    frozen = {"defensive":recovered["cash"],"broad":recovered["broad"],"recovery":recovered["recovery"],
              "high":anchor["own_vol"],"joint":anchor["joint_recovery"]}
    snapshots = {}
    for name, new_orders in extended.items():
        old_orders=original[name]
        if new_orders[:len(old_orders)]!=old_orders or len(old_orders)!=48 or len(new_orders)!=53:
            raise ValueError("Old targets changed or extension incomplete")
        ref=run_technical_portfolio(old,cal,start=cfg["start"],end=cfg["end"],decisions=old_orders,cost_bps=10)
        for key in ("maximum_drawdown","cagr_calendar","final_nav"):
            np.testing.assert_allclose(ref["metrics"][key],frozen[name][key],rtol=0,atol=1e-12)
        account=run_technical_portfolio(frames,cal,start=cfg["start"],end=end,decisions=new_orders,cost_bps=10)
        if account["equity_curve"][:len(ref["equity_curve"])]!=ref["equity_curve"]:
            raise ValueError("Account history changed")
        fresh=account["equity_curve"][len(ref["equity_curve"]):]
        if len(fresh)!=checks["new_sessions_per_stock"]:
            raise ValueError("Extension cut mismatch")
        stats=curve_metrics(fresh,"2026-01-01",end,ref["metrics"]["final_nav"])
        # Report this short interval as a holding-period return, not annualized.
        snapshots[name]={"previous":ref["metrics"],"new_period_return":stats["total_return"],
                         "new_period_drawdown":stats["maximum_drawdown"],
                         "carry_positions":len(ref["equity_curve"][-1]["holdings"]),
                         "new_period_mean_exposure":float(np.mean([r["exposure"] for r in fresh])),
                         "new_period_fills":sum(f["date"]>cfg["end"] for f in account["fills"]),
                         "full_cagr":account["metrics"]["cagr_calendar"], "full_drawdown":account["metrics"]["maximum_drawdown"],
                         "new_period_months":{m:len([r for r in fresh if r["date"].startswith(m)]) for m in sorted({r["date"][:7] for r in fresh})},
                         "extension_marks":fresh}
    return {"asof_end":end,"checks":checks,"results":snapshots,"script_sha256":sha(Path(__file__)),
            "input_reports":{str(p):sha(p) for p in (high_report,recovery_report)}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--local",type=Path,default=ROOT/"runtime_layer/data/market_parts")
    p.add_argument("--recovery",type=Path,required=True)
    p.add_argument("--high",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=study(args.run,args.local,args.recovery,args.high)
    write(args.output,result)
    print({k:(v["new_period_return"],v["new_period_drawdown"]) for k,v in result["results"].items()})
