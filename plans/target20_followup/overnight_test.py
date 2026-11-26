"""One post-hoc experiment, registered before computing its outcomes.

After seeing the original portfolios' overnight losses, retain eligible names
whose 63-session sum of log(open / previous close) is nonnegative. Everything
else follows archived momentum10 and defensive5. The breadth denominator stays
the same; the defensive risk estimate is recomputed on the actual new basket.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_layer.monthly_momentum import build_monthly_decisions
from research_layer.technical_portfolio import run_technical_portfolio


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected source version: {path}")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def overnight_feature(history, as_of):
    """Include this signal day's known open, always paired with prior close."""
    sample = history.loc[:as_of].tail(64)
    if len(sample) != 64 or sample.index[-1] != as_of:
        raise ValueError("Missing price history for a previously eligible symbol")
    previous = sample.close.to_numpy()[:-1]
    opens = sample.open.to_numpy()[1:]
    closes = sample.close.to_numpy()[1:]
    overnight = math.fsum(math.log(op / last) for op, last in zip(opens, previous))
    intraday = math.fsum(math.log(close / op) for op, close in zip(opens, closes))
    if abs(overnight + intraday - math.log(closes[-1] / previous[0])) > 1e-11:
        raise ValueError("Price-return decomposition is inconsistent")
    return overnight


def filtered_decisions(original, prices, policy):
    decisions = []
    previous = []
    slots = 5 if policy == "defensive5" else 10
    for signal in original:
        as_of = pd.Timestamp(signal["signal_date"])
        meta = signal["metadata"]
        values = {s: overnight_feature(prices[s], as_of) for s in meta["ranked_symbols"]}
        survivors = [s for s in meta["ranked_symbols"] if values[s] >= 0]
        retained = [s for s in previous if s in survivors[:2*slots]] if policy == "defensive5" else []
        selected = retained + [s for s in survivors if s not in retained][:slots - len(retained)]
        previous = selected
        scale = 1.
        if policy == "defensive5":
            if selected:
                closes = pd.concat([prices[s].loc[:as_of].close.tail(64) for s in selected], axis=1)
                if len(closes) != 64 or closes.isna().any().any():
                    raise ValueError("Unsynchronized volatility window")
                basket = closes.pct_change(fill_method=None).iloc[1:].sum(axis=1) / slots
                sigma = stdev(basket.to_list()) * math.sqrt(252)
                if sigma > 0:
                    scale = min(1., .12 / sigma)
            if meta["breadth"] < .5:
                scale = 0.
        decisions.append({"signal_date":signal["signal_date"],
                          "weights":{s:scale / slots for s in selected if scale > 0},
                          "metadata":{"original_candidates":len(values), "filtered_candidates":len(survivors),
                                      "original_breadth":meta["breadth"], "overnight_sums":values, "selected":selected}})
    return decisions


def summarize(result):
    curve = result["equity_curve"]
    year_ends = {d["date"][:4]: d["nav"] for d in curve}
    annual = {}
    previous = peak = 1.
    drawdown = 0.
    for year, final in year_ends.items():
        annual[year] = final / previous - 1.
        previous = final
    for d in curve:
        peak = max(peak, d["nav"])
        drawdown = max(drawdown, 1. - d["nav"]/peak)
    years = ((pd.Timestamp(result["requested_end"])-pd.Timestamp(result["requested_start"])).days+1)/365.25
    cagr = previous**(1/years) - 1.
    if abs(cagr-result["metrics"]["cagr_calendar"])>1e-11 or abs(drawdown-result["metrics"]["maximum_drawdown"])>1e-11:
        raise ValueError("Metrics differ from curve")
    return {"cagr":cagr, "drawdown":drawdown, "annual":annual, "days_traded":len({f["date"] for f in result["fills"]}),
            "fills":len(result["fills"]), "mean_exposure":mean(row["exposure"] for row in curve),
            "target_met":cagr > .2 and drawdown <= .15}


def run(root):
    manifest = read(root/"artifacts.json")["sha256"]
    verified(root/"freeze.json", manifest["freeze.json"])
    freeze = read(root/"freeze.json")
    for relative, expected in freeze["implementation"].items():
        verified(ROOT/relative, expected)
    bars = {}
    prices = {}
    for symbol, item in freeze["inputs"].items():
        source = Path(item["path"])
        verified(source, item["sha256"])
        frame = pd.read_csv(source, parse_dates=["datetime"])
        if not frame.symbol.eq(symbol).all() or not frame.adjust.eq("pre").all() or not frame.freq.eq("daily").all():
            raise ValueError("Unexpected price source or adjustment")
        bars[symbol] = frame
        prices[symbol] = frame.set_index("datetime", verify_integrity=True).sort_index()
    verified(Path(freeze["calendar"]["path"]), freeze["calendar"]["sha256"])
    calendar = pd.read_csv(freeze["calendar"]["path"])
    dates = {k:freeze["configuration"][k] for k in ["start","end"]}
    output = {"prior_run":root.name, "rule":"sum of 63 known log(open / previous close) >= 0", "policies":{},
              "source_sha256":digest(Path(__file__)), "prior_freeze_sha256":manifest["freeze.json"], "exact_controls":0}
    for policy in ["momentum10", "defensive5"]:
        # This validates the full calendar, OHLC, price basis and dated eligibility.
        original = build_monthly_decisions(bars, calendar, policy=policy, **dates)
        reference_file = f"decisions/{policy}.json"
        verified(root/reference_file, manifest[reference_file])
        if original != read(root/reference_file):
            raise ValueError("Decision baseline changed")
        filtered = filtered_decisions(original, prices, policy)
        out = {"cases":{}, "decisions":filtered}
        for cost in (0,10,20):
            base = run_technical_portfolio(bars, calendar, decisions=original, cost_bps=cost, **dates)
            name = f"cases/{policy}_{cost}bps.json"
            verified(root/name, manifest[name])
            if base != read(root/name):
                raise ValueError("Archived portfolio drift")
            output["exact_controls"] += 1
            variant = run_technical_portfolio(bars, calendar, decisions=filtered, cost_bps=cost, **dates)
            out["cases"][str(cost)] = {"baseline":summarize(base), "screened":summarize(variant)}
        output["policies"][policy] = out
    return output


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    results = run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2)+"\n",encoding="utf-8")
    temporary.replace(args.output)
