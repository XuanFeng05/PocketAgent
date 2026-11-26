"""Posthoc broad-basket control, recorded before its account outcomes were computed.

The previously observed risk-off group contained 30 execution months. The gross
equal-100 shadow basket outperformed the original five planned stocks in this
group. Use this observation once: on original risk-on dates copy the complete
defensive5 targets; on risk-off dates allocate the planned ungated five-stock
notional equally across the signal-day equal100 candidates. No ETF or index
fills. The five-stock volatility scale is retained to match target notional.
Compare with holding cash or retaining those five stocks at 0/10/20 bps.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd

from plans.target20.run_study import read, sha, checked, write
from plans.target20_followup.cash_diagnostic import metrics
from research_layer.technical_portfolio import run_technical_portfolio


def alternatives(five_decisions, broad_decisions):
    if [d['signal_date'] for d in five_decisions] != [d['signal_date'] for d in broad_decisions]:
        raise ValueError('Different monthly signal dates')
    five, broad, targets = [], [], []
    for original, benchmark in zip(five_decisions, broad_decisions, strict=True):
        date, info = original['signal_date'], original['metadata']
        off = info['breadth'] < .5
        if off:
            if original['weights']:
                raise ValueError('Expected zero original target in a closed month')
            sigma = info['estimated_portfolio_volatility']
            scale = min(1., .12/sigma) if sigma > 0 else 1.
            # Match the frozen control's slot-weight then scaling operation bit-for-bit.
            small = {s: (1 / info['slots']) * scale for s in info['selected']}
            notional = math.fsum(small.values())
            valid = [s for s,w in benchmark['weights'].items() if w > 0]
            wide = {s: notional / len(valid) for s in valid} if valid and notional else {}
            if valid and not math.isclose(math.fsum(wide.values()),notional,abs_tol=1e-12):
                raise ValueError('Unequal target notional')
        else:
            small = wide = original['weights'].copy()
        five.append({'signal_date': date, 'weights': small})
        broad.append({'signal_date': date, 'weights': wide})
        targets.append({'signal_date': date, 'risk_off': off, 'five_names': list(small),
                        'broad_names': list(wide), 'five_total': math.fsum(small.values()),
                        'broad_total': math.fsum(wide.values())})
    return five, broad, targets


def cash_audit(result, tables, bps):
    active = {}
    cash = 1.
    by_day = {}
    for fill in result['fills']:
        by_day.setdefault(fill['date'], []).append(fill)
    max_error = 0.
    for mark in result['equity_curve']:
        date = mark['date']
        for fill in by_day.get(date,[]):
            symbol, amount = fill['symbol'], fill['delta_units']
            price, volume = tables[symbol][date][:2]
            if price != fill['price'] or volume <= 0:
                raise ValueError('Unavailable fill or price mismatch')
            value = price * amount
            cash -= value + abs(value) * bps/10000
            active[symbol] = active.get(symbol, 0.) + amount
            if abs(active[symbol]) < 1e-14:
                del active[symbol]
        if cash < -1e-10 or any(q < -1e-10 for q in active.values()):
            raise ValueError('Unfunded account')
        nav = cash + math.fsum(q*tables[s][date][2] for s,q in active.items())
        max_error = max(max_error, abs(cash-mark['cash']), abs(nav-mark['nav']))
    if max_error > 1e-10:
        raise ValueError('Account reconstruction differs')
    return max_error


def run_study(run_root, cash_results):
    manifest = read(run_root/'artifacts.json')['sha256']
    def pinned(relative):
        file = run_root/relative
        checked(file, manifest[relative])
        return read(file)
    freeze = pinned('freeze.json')
    for path, digest in freeze['implementation'].items():
        checked(ROOT/path, digest)
    frames, tables = {}, {}
    for symbol, item in freeze['inputs'].items():
        path = Path(item['path']); checked(path, item['sha256'])
        frame = pd.read_csv(path); frames[symbol] = frame
        tables[symbol] = {str(row.datetime)[:10]: (float(row.open),float(row.volume),float(row.close))
                          for row in frame.itertuples(index=False)}
    cp = Path(freeze['calendar']['path']); checked(cp, freeze['calendar']['sha256'])
    calendar = pd.read_csv(cp); config = freeze['configuration']
    original = pinned('decisions/defensive5.json')
    five, broad, targets = alternatives(original, pinned('decisions/equal100.json'))
    if sum(t['risk_off'] for t in targets) != 30 or len(targets) != 48:
        raise ValueError('Unexpected target coverage')
    results = {}; max_error = 0.; baseline_exact = 0
    for cost in [0,10,20]:
        comparisons = {}
        for label, decisions in [('original_cash',original),('keep_five',five),('broad_basket',broad)]:
            account = run_technical_portfolio(frames,calendar,start=config['start'],end=config['end'],decisions=decisions,cost_bps=cost)
            if label == 'original_cash':
                if account != pinned(f'cases/defensive5_{cost}bps.json'):
                    raise ValueError('Baseline account changed')
                baseline_exact += 1
            m = metrics(account,config)
            if label == 'keep_five' and cost == 10:
                previous = read(cash_results)['without_gate']
                if m != previous:
                    raise ValueError('Ungated matched control changed')
            max_error = max(max_error, cash_audit(account,tables,cost))
            if any(r['target_weights'] != d['weights'] for r,d in zip(account['rebalances'],decisions,strict=True)):
                raise ValueError('Decision mismatch')
            m.update(buys=account['metrics']['buy_count'],sells=account['metrics']['sell_count'],
                     max_positions=account['metrics']['max_positions'],
                     blocked=dict(Counter(x['reason'] for r in account['rebalances'] for x in r['blocked'])),
                     total_fees=account['metrics']['total_fees'])
            comparisons[label] = m
        results[str(cost)] = comparisons
    return {'study':__doc__.strip(),'run':run_root.name,'source_sha256':sha(Path(__file__)),
            'source_artifact_manifest_sha256':sha(run_root/'artifacts.json'),'targets':targets,'by_cost':results,
            'checks':{'monthly_decisions':len(targets),'off_windows':30,'baseline_exact_matches':baseline_exact,
                      'prior_ungated_control_matched':True,'max_cash_or_nav_error':max_error},
            'interpretation':'Ex ante target capital is matched to the ungated five-stock control and shares its volatility scale. Realized exposure, volatility, turnover, fees and concentration differ with portfolio constituents. Broad holdings require separate individual stock orders; suspension blocks and residual cash remain in the account.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--cash-analysis',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run_study(args.run,args.cash_analysis);write(args.output,report)
    print(json.dumps({'checks':report['checks'],'by_cost':report['by_cost']},indent=2))
