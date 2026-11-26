"""Reconcile stock-level P&L to three archived shared-cash portfolio ledgers."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


POLICIES = ("momentum10", "buffer10", "defensive5", "equal100")
TOLERANCE = 1e-10


def verified(path, expected):
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Hash mismatch: {path}")


def price_inputs(freeze):
    prices = {}
    for symbol, identity in freeze["inputs"].items():
        path = Path(identity["path"])
        verified(path, identity["sha256"])
        with path.open(encoding="utf-8", newline="") as stream:
            rows = {}
            for bar in csv.DictReader(stream):
                if bar["symbol"] != symbol or bar["adjust"] != "pre" or bar["freq"] != "daily":
                    raise ValueError(f"Unexpected input identity for {symbol}")
                date = bar["datetime"][:10]
                if date in rows:
                    raise ValueError(f"Duplicate session: {symbol}/{date}")
                rows[date] = (float(bar["open"]), float(bar["close"]))
            prices[symbol] = rows
    return prices


def drawdown_period(curve):
    peak, peak_idx = curve[0]["nav"], 0
    worst, start, end = 0.0, 0, 0
    for idx, row in enumerate(curve[1:], start=1):
        if row["nav"] > peak:
            peak, peak_idx = row["nav"], idx
        drawdown = 1 - row["nav"] / peak
        if drawdown > worst:
            worst, start, end = drawdown, peak_idx, idx
    return start, end, worst


def add_price_effects(result, prices):
    """Independently rebuild daily P&L from overnight/open-to-close exposure."""
    fills_by_date = defaultdict(list)
    for fill in result["fills"]:
        fills_by_date[fill["date"]].append(fill)
    curve = [{"date": result["requested_start"], "nav": result["initial_cash"], "holdings": {}}] + result["equity_curve"]
    effects = [{}]
    for previous, current in zip(curve, curve[1:]):
        day = current["date"]
        owned = previous["holdings"]
        closing = current["holdings"]
        daily = {}
        fees = defaultdict(float)
        for fill in fills_by_date.get(day, []):
            fees[fill["symbol"]] += fill["fee"]
            if abs(fill["price"] - prices[fill["symbol"]][day][0]) > TOLERANCE:
                raise ValueError(f"Fill price differs from frozen opening quote on {day}")
        for symbol in set(owned) | set(closing) | set(fees):
            opening, close = prices[symbol][day]
            old_units = owned.get(symbol, {}).get("units", 0.0)
            new_units = closing.get(symbol, {}).get("units", 0.0)
            overnight = old_units * (opening - prices[symbol][previous["date"]][1]) if old_units else 0.0
            intraday = new_units * (close - opening)
            daily[symbol] = (overnight, intraday, fees[symbol])
        net = math.fsum(overnight + intraday - fees_paid for overnight, intraday, fees_paid in daily.values())
        if abs(net - (current["nav"] - previous["nav"])) > TOLERANCE:
            raise ValueError(f"Daily close-to-open-to-close validation failed on {day}")
        effects.append(daily)
    return curve, effects


def period(curve, effects, fills, start_idx, end_idx):
    start, end = curve[start_idx], curve[end_idx]
    # Start is the already-marked close (or the original cash inception). Exclude its fills.
    components = defaultdict(lambda: defaultdict(float))
    for symbol, position in start["holdings"].items():
        components[symbol]["start_mark"] = position["market_value"]
    for symbol, position in end["holdings"].items():
        components[symbol]["end_mark"] = position["market_value"]
    for fill in fills:
        if start["date"] < fill["date"] <= end["date"]:
            parts = components[fill["symbol"]]
            parts[fill["side"]] += fill["notional"]
            parts["fees"] += fill["fee"]
    for daily in effects[start_idx + 1:end_idx + 1]:
        for symbol, (overnight, intraday, _) in daily.items():
            components[symbol]["overnight"] += overnight
            components[symbol]["intraday"] += intraday
    records = []
    for symbol, parts in components.items():
        gross = parts["end_mark"] - parts["start_mark"] + parts["sell"] - parts["buy"]
        if abs(gross - parts["overnight"] - parts["intraday"]) > TOLERANCE:
            raise ValueError(f"Per-stock price vs cash-flow attribution mismatch: {symbol}")
        records.append({"symbol":symbol, "net_contribution":(gross-parts["fees"])/start["nav"],
                        "gross_contribution":gross/start["nav"], "fees_over_start_nav":parts["fees"]/start["nav"],
                        "overnight_contribution":parts["overnight"]/start["nav"],
                        "intraday_contribution":parts["intraday"]/start["nav"]})
    records.sort(key=lambda r:r["net_contribution"])
    losses = [r for r in records if r["net_contribution"] < -TOLERANCE]
    winners = [r for r in records if r["net_contribution"] > TOLERANCE]
    positive = math.fsum(r["net_contribution"] for r in winners)
    negative = -math.fsum(r["net_contribution"] for r in losses)
    account_return = end["nav"]/start["nav"]-1
    reconciliation = math.fsum(r["net_contribution"] for r in records)-account_return
    if abs(reconciliation) > TOLERANCE:
        raise ValueError(f"Period attribution mismatch: {reconciliation}")
    return {"begin_mark":start["date"], "end_mark":end["date"], "start_nav":start["nav"], "end_nav":end["nav"],
            "account_return":account_return, "attribution_error":reconciliation,
            "stock_count":len(records), "losers":len(losses), "winners":len(winners),
            "ties":len(records)-len(losses)-len(winners), "sum_positive_net_contributions":positive,
            "sum_negative_net_contributions":-negative, "five_largest_loss_share_of_all_stock_losses":
                -math.fsum(r["net_contribution"] for r in losses[:5])/negative if negative else None,
            "five_largest_losses":losses[:5], "five_largest_gains":winners[-5:][::-1],
            "fee_contribution":-math.fsum(r["fees_over_start_nav"] for r in records),
            "overnight_contribution":math.fsum(r["overnight_contribution"] for r in records),
            "intraday_contribution":math.fsum(r["intraday_contribution"] for r in records),
            "per_stock":records}


def audit(root):
    manifest = json.loads((root/"artifacts.json").read_text(encoding="utf-8"))["sha256"]
    for name in ["freeze.json", "summary.json", *(f"cases/{p}_10bps.json" for p in POLICIES)]:
        verified(root/name, manifest[name])
    freeze = json.loads((root/"freeze.json").read_text(encoding="utf-8"))
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if summary["status"]!="completed" or summary["failed"]!=0 or summary["total"]!=24:
        raise ValueError("Prior run incomplete")
    prices = price_inputs(freeze)
    answer = {"run":root.name, "input_csv_sha_count":len(prices), "policies":{}}
    for policy in POLICIES:
        name = f"cases/{policy}_10bps.json"
        result = json.loads((root/name).read_text(encoding="utf-8"))
        row = next(r for r in summary["outcomes"] if r["policy"]==policy and r["cost_bps"]==10)
        if row["artifact_path"]!=name or result["metrics"]!=row["metrics"] or result["cost_bps"]!=10:
            raise ValueError(f"Identity mismatch: {name}")
        curve, effects = add_price_effects(result, prices)
        peak, trough, drawdown = drawdown_period(curve)
        if abs(drawdown-result["metrics"]["maximum_drawdown"]) > TOLERANCE:
            raise ValueError("Drawdown mismatch")
        periods = {"full": (0,len(curve)-1), "worst_drawdown": (peak,trough)}
        for year in sorted({row["date"][:4] for row in curve[1:]}):
            indices = [i for i, row in enumerate(curve) if i and row["date"].startswith(year)]
            periods[year] = (indices[0]-1, indices[-1])
        answer["policies"][policy] = {label:period(curve,effects,result["fills"],a,b) for label,(a,b) in periods.items()}
    return answer


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); report=audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
