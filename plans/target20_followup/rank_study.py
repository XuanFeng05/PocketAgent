"""Matched monthly stock-ranking study on frozen PocketAgent account decisions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_layer.technical_portfolio import run_technical_portfolio

SEED = 20261126
N_RANDOM = 256
RULES = ["original", "skip_last_month", "low_volatility"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected hash for {path}")
    return path


def load_snapshot(run):
    manifest = read(run / "artifacts.json")["sha256"]
    freeze = read(verified(run / "freeze.json", manifest["freeze.json"]))
    frames = {s: pd.read_csv(verified(Path(v["path"]), v["sha256"]))
              for s, v in sorted(freeze["inputs"].items())}
    calendar = pd.read_csv(verified(Path(freeze["calendar"]["path"]), freeze["calendar"]["sha256"]))
    return manifest, freeze["configuration"], frames, calendar


def make_targets(decisions, close, defensive):
    """Hold entry universe, number of slots, signal dates and target exposures fixed."""
    symbols = list(close.columns)
    ids = {s: i for i, s in enumerate(symbols)}
    slots = 5 if defensive else 10
    generator = np.random.Generator(np.random.PCG64(SEED))
    permutations = [generator.permutation(len(symbols)).tolist() for _ in range(N_RANDOM)]
    labels = RULES + [f"neutral_{i:03}" for i in range(N_RANDOM)]
    if defensive:
        labels += ["skip_last_month_own_vol", "low_volatility_own_vol"]
    targets = np.zeros((len(decisions), len(labels), len(symbols)))
    retained = [[] for _ in labels]
    for month, decision in enumerate(decisions):
        date = decision["signal_date"]
        history = close.loc[:date]
        trailing_returns = history.tail(64).pct_change(fill_method=None).iloc[1:]
        eligible = [ids[s] for s in decision["metadata"]["ranked_symbols"]]
        eligible_set = set(eligible)
        if eligible and trailing_returns.iloc[:, eligible].isna().any().any():
            raise ValueError("Incomplete risk history")
        lagged = np.log(history.iloc[-22] / history.iloc[-64]) / 2 + np.log(history.iloc[-22] / history.iloc[-127]) / 2
        vol = trailing_returns.std(ddof=1).to_numpy()
        by_skip = sorted(eligible, key=lambda i: (-lagged.iloc[i], symbols[i]))
        by_vol = sorted(eligible, key=lambda i: (vol[i], symbols[i]))
        ranks = [eligible, by_skip, by_vol]
        ranks.extend([i for i in order if i in eligible_set] for order in permutations)
        if defensive:
            ranks += [by_skip, by_vol]
        original_scale = decision["metadata"]["risk_scale"]
        for j, (label, ranked) in enumerate(zip(labels, ranks, strict=True)):
            keep = [i for i in retained[j] if i in ranked[:2*slots]] if defensive else []
            chosen = keep + [i for i in ranked if i not in keep][:slots-len(keep)]
            retained[j] = chosen
            scale = original_scale
            if label.endswith("_own_vol") and chosen:
                basket = trailing_returns.iloc[:, chosen].sum(axis=1) / slots
                sigma = basket.std(ddof=1) * np.sqrt(252)
                scale = min(1.0, .12 / sigma) if sigma > 0 else 1.0
                if decision["metadata"]["breadth"] < .5:
                    scale = 0.0
            targets[month, j, chosen] = scale / slots
        expected = [decision["weights"].get(s, 0.) for s in symbols]
        np.testing.assert_allclose(targets[month, 0], expected, atol=1e-12, rtol=0)
        same_budget = targets[month, :len(RULES)+N_RANDOM].sum(axis=1)
        np.testing.assert_allclose(same_budget, sum(expected), atol=1e-12, rtol=0)
    return labels, targets


def simulate(targets, execution, opens, closes, available, cost):
    """Independent batched shared-account execution, next-open monthly rebalance."""
    count, n_symbols = targets.shape[1:]
    cash = np.ones(count)
    units = np.zeros((count, n_symbols))
    fill_count = np.zeros(count, dtype=int)
    nav = np.empty((len(closes), count))
    exposure = np.empty_like(nav)
    for day, (opening, closing, tradeable) in enumerate(zip(opens, closes, available, strict=True)):
        if day in execution:
            held = units * opening
            starting_nav = cash + held.sum(axis=1)
            tolerance = starting_nav[:, None] * 1e-13
            difference = targets[execution[day]] * (starting_nav[:, None] / (1+cost)) - held
            sales = np.where(tradeable & (difference < -tolerance), -difference, 0.)
            cash += sales.sum(axis=1) * (1-cost)
            buys = np.where(tradeable & (difference > tolerance), difference, 0.)
            requested = buys.sum(axis=1) * (1+cost)
            scale = np.divide(cash, requested, out=np.ones_like(cash), where=requested>0)
            buys *= np.minimum(1., scale[:,None])
            buys = np.where(buys > tolerance, buys, 0.)
            cash -= buys.sum(axis=1) * (1+cost)
            units += (buys-sales)/opening
            if np.min(cash) < -1e-11 or np.min(units) < -1e-11:
                raise ValueError("Unfunded account or short position")
            cash = np.maximum(cash, 0.)
            units = np.maximum(units, 0.)
            fill_count += ((buys+sales)>0).sum(axis=1)
        invested = (units*closing).sum(axis=1)
        nav[day] = invested+cash
        exposure[day] = invested/nav[day]
    return nav, exposure, fill_count


def metrics(nav, exposure, fills, dates, year_count):
    result = []
    for curve, risk, count in zip(nav.T, exposure.T, fills, strict=True):
        peaks = np.maximum.accumulate(np.r_[1.,curve])[1:]
        annual = {}
        previous = 1.
        for year in sorted(set(dates.year)):
            last = curve[dates.year == year][-1]
            annual[str(year)] = last/previous-1
            previous = last
        result.append({"cagr":float(curve[-1]**(1/year_count)-1),
                       "drawdown":float(np.max(1-curve/peaks)), "exposure":float(risk.mean()),
                       "fills":int(count), "annual":annual})
    return result


def describe(pool, observed):
    values = np.array([r["cagr"] for r in pool])
    return {"count":len(pool), "cagr_p10_median_p90":np.quantile(values,[.1,.5,.9]).tolist(),
            "drawdown_median":float(np.median([r["drawdown"] for r in pool])),
            "fills_median":float(np.median([r["fills"] for r in pool])),
            "at_or_below_original":int(np.sum(values<=observed["cagr"]+1e-12)),
            "by_year":{year:{"original":value,"median":float(np.median([r["annual"][year] for r in pool])),
                             "at_or_below_original":int(sum(r["annual"][year]<=value+1e-12 for r in pool))}
                       for year,value in observed["annual"].items()}}


def omission_stress(decisions, symbols, execution, opens, closes, available, dates, years):
    """Replace each name in turn; keep the observed gross-risk scale and regime fixed."""
    indices = {symbol: i for i, symbol in enumerate(symbols)}
    targets = np.zeros((len(decisions), len(symbols), len(symbols)))
    for column, omitted in enumerate(symbols):
        selected = []
        for month, decision in enumerate(decisions):
            meta = decision["metadata"]
            ranked = [s for s in meta["ranked_symbols"] if s != omitted]
            keep = [s for s in selected if s in ranked[:10]]
            selected = keep + [s for s in ranked if s not in keep][:5-len(keep)]
            targets[month, column, [indices[s] for s in selected]] = meta["risk_scale"] / 5
    curves, exposures, counts = simulate(targets, execution, opens, closes, available, .001)
    return targets, curves, dict(zip(symbols, metrics(curves, exposures, counts, dates, years), strict=True))


def study(run):
    manifest, config, frames, calendar = load_snapshot(run)
    symbols = sorted(frames)
    panels = {k:pd.DataFrame({s:f.set_index("datetime")[k] for s,f in frames.items()}).sort_index()
              for k in ("open","close","volume")}
    dates = pd.DatetimeIndex(pd.to_datetime(panels["close"].loc[config["start"]:config["end"]].index))
    def panel(key):
        return panels[key].loc[config["start"]:config["end"],symbols].to_numpy(float)
    opens, closes, volume = [panel(k) for k in ("open","close","volume")]
    available = volume>0
    for col,symbol in enumerate(symbols):
        if "is_tradeable" in frames[symbol]:
            flags = frames[symbol].set_index("datetime").loc[config["start"]:config["end"],"is_tradeable"]
            available[:,col] &= flags.astype(str).str.lower().isin(["true","1"]).to_numpy()
    if not np.isfinite(opens).all() or not np.isfinite(closes).all():
        raise ValueError("Incomplete execution data")
    years = ((pd.Timestamp(config["end"])-pd.Timestamp(config["start"])).days+1)/365.25
    result = {"seed":SEED, "source_run":run.name, "families":{}, "checks":{}, "rule_outcomes":{},
              "source_manifest_sha256":digest(run / "artifacts.json"), "script_sha256":digest(Path(__file__))}
    for family in ("momentum10","defensive5"):
        rel=f"decisions/{family}.json"
        decisions=read(verified(run/rel,manifest[rel]))
        labels, targets = make_targets(decisions, panels["close"][symbols], family=="defensive5")
        execution = {int(dates.searchsorted(d["signal_date"],side="right")):i for i,d in enumerate(decisions)}
        if len(execution)!=48 or max(execution)>=len(dates):
            raise ValueError("Execution date mismatch")
        if family == "defensive5":
            omit_targets, omit_curves, result["omitted_once_at_10bps"] = omission_stress(
                decisions, symbols, execution, opens, closes, available, dates, years)
            j = symbols.index("300308.SZ")
            verify = [{"signal_date":d["signal_date"],
                       "weights":{symbols[k]:float(w) for k,w in enumerate(omit_targets[i,j]) if w>0}}
                      for i,d in enumerate(decisions)]
            ref = run_technical_portfolio(frames,calendar,start=config["start"],end=config["end"],decisions=verify,cost_bps=10)
            direct_nav = np.array([r["nav"] for r in ref["equity_curve"]])
            np.testing.assert_allclose(direct_nav,omit_curves[:,j],atol=1e-11,rtol=0)
            result["checks"]["omit_300308"] = float(np.max(np.abs(direct_nav-omit_curves[:,j])))
        cases={}
        for bps in (0,10,20):
            nav, exp, fills = simulate(targets,execution,opens,closes,available,bps/10000)
            books = metrics(nav,exp,fills,dates,years)
            old_rel=f"cases/{family}_{bps}bps.json"
            old=read(verified(run/old_rel,manifest[old_rel]))
            legacy_curve=np.array([r["nav"] for r in old["equity_curve"]])
            np.testing.assert_allclose(nav[:,0],legacy_curve,atol=1e-11,rtol=0)
            if fills[0]!=len(old["fills"]):raise ValueError("Baseline fill count changed")
            cases[str(bps)]={"rankings":{label:b for label,b in zip(labels,books) if not label.startswith("neutral_")},
                             "neutral_distribution":describe(books[3:3+N_RANDOM],books[0]),
                             "neutral_accounts":books[3:3+N_RANDOM]}
            if bps==10:
                # Cross-check both standalone alternatives and a placebo against the original engine.
                for j in [1,2,3]+([len(labels)-2,len(labels)-1] if family=="defensive5" else []):
                    orders=[{"signal_date":d["signal_date"],"weights":{symbols[k]:float(w) for k,w in enumerate(targets[i,j]) if w>0}}
                            for i,d in enumerate(decisions)]
                    direct=run_technical_portfolio(frames,calendar,start=config["start"],end=config["end"],decisions=orders,cost_bps=10)
                    direct_curve=[r["nav"] for r in direct["equity_curve"]]
                    np.testing.assert_allclose(direct_curve,nav[:,j],atol=1e-11,rtol=0)
                    if len(direct["fills"])!=fills[j]:raise ValueError("Direct fill count mismatch")
                    result["checks"][family+"/"+labels[j]] = float(np.max(np.abs(direct_curve-nav[:,j])))
            result["rule_outcomes"][family+"/"+str(bps)]=[label for i,label in enumerate(labels)
                if not label.startswith("neutral_") and books[i]["cagr"]>.20 and books[i]["drawdown"]<=.15]
        result["families"][family]=cases
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    report=study(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"target_hits":report["rule_outcomes"],"direct_checks":report["checks"]},indent=2))
