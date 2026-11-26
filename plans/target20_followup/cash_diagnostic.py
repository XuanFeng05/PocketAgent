"""Audit the existing defensive5 breadth gate over fixed open-to-open windows."""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plans.target20.analyze import curve_metrics, slices
from plans.target20.run_study import checked, read, sha, write
from research_layer.technical_portfolio import run_technical_portfolio


def tradable(row):
    flag = str(row.get('is_tradeable', True)).strip().lower() in {'true', '1'}
    return flag and row['volume'] > 0


def shadow(tables, weights, start, stop, terminal):
    """Independent unit-capital monthly mark. A blocked sale is explicitly labelled."""
    blocked_entry, blocked_exit = [], []
    earnings = []
    for symbol, weight in weights.items():
        day0, day1 = tables[symbol].loc[start], tables[symbol].loc[stop]
        if not tradable(day0):
            blocked_entry.append(symbol)
            continue  # Keep the slot cash, no rerank.
        if not terminal and not tradable(day1):
            blocked_exit.append(symbol)  # A reported quote mark, no executable liquidation assumed.
        price1 = day1['close'] if terminal else day1['open']
        earnings.append(weight * (price1 / day0['open'] - 1))
    return {'return': math.fsum(earnings), 'unfilled_entry_symbols': blocked_entry,
            'nontradable_open_end_marks': blocked_exit}


def statistics(values):
    if not values:
        return {'n': 0}
    gains = [x for x in values if x > 1e-12]
    losses = [x for x in values if x < -1e-12]
    return {'n': len(values), 'mean': mean(values), 'median': median(values),
            'positive_count': len(gains), 'negative_count': len(losses),
            'positive_rate': len(gains) / len(values),
            'positive_sum_equal_month_notional': math.fsum(gains),
            'absolute_negative_sum_equal_month_notional': -math.fsum(losses),
            'compounded_only_selected_windows': math.prod(1 + x for x in values) - 1}


def window_group(windows):
    output = {}
    for name in ('basket', 'vol_scaled_basket', 'equal100'):
        returns = [w[name]['return'] for w in windows]
        clean = [w[name]['return'] for w in windows if not w[name]['nontradable_open_end_marks']]
        output[name] = statistics(returns)
        output[name]['without_nontradable_end_marks'] = statistics(clean)
    output['selection_minus_equal100'] = statistics([w['basket']['return'] - w['equal100']['return'] for w in windows])
    # The difference series is descriptive. Compounding differences has no account interpretation.
    output['selection_minus_equal100'].pop('compounded_only_selected_windows', None)
    return output


def ledger_check(account, tables, cost):
    cash, units = 1., {s: 0. for s in tables}
    fills_by_date = {}
    for fill in account['fills']:
        fills_by_date.setdefault(fill['date'], []).append(fill)
    max_error = 0.
    for row in account['equity_curve']:
        date = row['date']
        for f in fills_by_date.get(date, []):
            symbol = f['symbol']
            bar = tables[symbol].loc[date]
            if not tradable(bar) or not math.isclose(float(bar['open']), f['price']):
                raise ValueError('Invalid price or suspended fill')
            delta = f['units'] if f['side'] == 'buy' else -f['units']
            notional = delta * float(bar['open'])
            cash -= notional + abs(notional) * cost / 10000
            units[symbol] += delta
            if cash < -1e-10 or units[symbol] < -1e-10:
                raise ValueError('Invalid cash or short exposure')
        marked = math.fsum(q * tables[s].at[date, 'close'] for s, q in units.items() if q)
        max_error = max(max_error, abs(cash - row['cash']), abs(cash + marked - row['nav']))
    if max_error > 1e-10:
        raise ValueError(f'Account ledger discrepancy {max_error}')
    return max_error


def metrics(account, config):
    curve = account['equity_curve']
    primary = curve_metrics(curve, config['start'], config['end'])
    annual, _ = slices(curve)
    for field in ('cagr_calendar', 'maximum_drawdown', 'final_nav'):
        if abs(primary[field] - account['metrics'][field]) > 1e-10:
            raise ValueError(f'Account metric mismatch {field}')
    return {**primary, 'annual_returns': {y: m['total_return'] for y, m in annual.items()},
            **{k: account['metrics'][k] for k in ('turnover', 'mean_exposure', 'fills_count', 'active_rebalance_days')}}


def audit(run):
    hashes = read(run / 'artifacts.json')['sha256']
    def pinned(path):
        checked(run / path, hashes[path]); return read(run / path)
    freeze = pinned('freeze.json')
    for path, digest in freeze['implementation'].items():
        checked(ROOT / path, digest)
    config = freeze['configuration']
    calendar_path = Path(freeze['calendar']['path'])
    checked(calendar_path, freeze['calendar']['sha256'])
    calendar = pd.read_csv(calendar_path)
    sessions = calendar.loc[calendar.is_trading_day.astype(str).str.lower().isin(['true', '1']), 'date'].tolist()
    frames, tables = {}, {}
    for s, item in freeze['inputs'].items():
        path = Path(item['path']); checked(path, item['sha256'])
        frames[s] = pd.read_csv(path)
        tables[s] = frames[s].set_index(frames[s]['datetime'].str[:10], verify_integrity=True)
    baseline = pinned('cases/defensive5_10bps.json')
    decisions = pinned('decisions/defensive5.json')
    equal_decisions = pinned('decisions/equal100.json')
    equal_by_date = {d['signal_date']: d['weights'] for d in equal_decisions}
    kwargs = {'start': config['start'], 'end': config['end'], 'cost_bps': 10}
    if run_technical_portfolio(frames, calendar, decisions=decisions, **kwargs) != baseline:
        raise ValueError('Could not reproduce the saved baseline exactly')
    intervals, alternative = [], []
    for i, d in enumerate(decisions):
        info = d['metadata']; signal = d['signal_date']; original = baseline['rebalances'][i]
        start_date = original['execution_date']
        if original['signal_date'] != signal or sessions[sessions.index(signal) + 1] != start_date:
            raise ValueError('Signal/open timing mismatch')
        final = i + 1 == len(decisions)
        stop_date = baseline['end'] if final else baseline['rebalances'][i + 1]['execution_date']
        if stop_date < start_date or (not final and start_date >= stop_date):
            raise ValueError('Non-increasing window dates')
        base_weights = {s: 1 / info['slots'] for s in info['selected']}
        scale = min(1., .12 / info['estimated_portfolio_volatility']) if info['estimated_portfolio_volatility'] > 0 else 1.
        revised_weights = {s: w * scale for s, w in base_weights.items()}
        off = info['breadth'] < .5
        if not off and revised_weights != d['weights'] or off and d['weights']:
            raise ValueError('Would alter more than the breadth gate')
        alternative.append({'signal_date': signal, 'weights': revised_weights})
        original_basket = shadow(tables, base_weights, start_date, stop_date, final)
        benchmark = shadow(tables, equal_by_date[signal], start_date, stop_date, final)
        intervals.append({'signal_date': signal, 'start_open': start_date, 'end': stop_date,
                          'end_phase': 'close' if final else 'open', 'year': start_date[:4], 'breadth': info['breadth'],
                          'risk_off': bool(off), 'vol_scale': scale, 'planned_names': info['selected'],
                          'basket': original_basket, 'vol_scaled_basket': {**original_basket, 'return': scale * original_basket['return']},
                          'equal100': benchmark})
    control = run_technical_portfolio(frames, calendar, decisions=alternative, **kwargs)
    if any(b['signal_date'] >= b['execution_date'] for b in control['rebalances']):
        raise ValueError('Lookahead in account')
    if len(intervals) != 48 or len(control['rebalances']) != 48:
        raise ValueError('Incomplete monthly coverage')
    off = [w for w in intervals if w['risk_off']]
    ranked = sorted(off, key=lambda w: w['vol_scaled_basket']['return'])
    by_gate = {key: window_group([w for w in intervals if w['risk_off'] == value])
               for key, value in [('risk_off', True), ('risk_on', False)]}
    annual = {year: {key: window_group([w for w in intervals if w['risk_off'] == value and w['year'] == year])
                     for key, value in [('risk_off', True), ('risk_on', False)]}
              for year in sorted({w['year'] for w in intervals})}
    def extremes(rows):
        return [{'start_open': r['start_open'], 'end': r['end'], 'equal100_return': r['equal100']['return'],
                 'unscaled_return': r['basket']['return'], 'vol_scaled_return':r['vol_scaled_basket']['return']} for r in rows]
    return {'method': 'First observed open after each original month-end signal through following execution open; last window ends on final close. Independent gross fixed-slot normalized windows. Unfilled entry slots stay cash. Suspended terminal quotes are valuation marks, never fills. Group products are conditional window composites; grouped sums denote equal unit notional per month. Benchmark uses original equal100 weights. Year group assigned by opening month, so December includes the following January opening gap except at final cutoff.',
            'one_change': 'Reuse all planned defensive5 names and estimated vol, including during risk-off, remove only breadth switch. Full counterfactual account uses same synthetic-unit execution, blocked positions, T+1 and 10bps costs.',
            'by_gate': by_gate, 'annual_by_gate': annual, 'windows': intervals,
            'top_five_helpful_cash': extremes(ranked[:5]), 'top_five_missed_cash': extremes(ranked[-5:][::-1]),
            'with_gate': metrics(baseline, config), 'without_gate': metrics(control, config),
            'validation': {'baseline_exact_reproduction': True, 'ledger_error_with_gate': ledger_check(baseline,tables,10),
                           'ledger_error_without_gate': ledger_check(control,tables,10), 'num_windows':len(intervals),
                           'utc_run_id':run.name, 'baseline_manifest_sha256':sha(run/'artifacts.json'), 'script_sha256':sha(Path(__file__))}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.run.resolve()); write(args.output.resolve(), report)
    print(json.dumps({k:report[k] for k in ('by_gate','with_gate','without_gate','validation')}, indent=2))
