"""Registered US500 signal/condition research with frozen chronological selection.

All generated observations, caches and results remain under runtime_layer.
The current survivor universe and previously viewed history are not pristine OOS.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from research_layer.elder_analysis import fetch_french_rf, metrics
from research_layer.multiscale_engine import MultiscaleConfig, prepare_arrays, prepare_window, run_strategy
from research_layer.multiscale_events import run_event_study
from research_layer.multiscale_signals import build_features
from research_layer.multiscale_statistics import assess_multiscale_family


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def candidate_id(config):
    return f'{config.weekly_filter}__{config.daily_signal.replace("&", "_AND_")}__{config.exit_mode}'


def family_configs(protocol):
    signals = protocol['daily_signals']
    entries = list(signals) + ['&'.join(pair) for pair in combinations(signals, 2)]
    fixed = {k: protocol['fixed'][k] for k in ['risk_fraction', 'cost_bps', 'initial_stop_atr', 'trailing_stop_atr']}
    configs = [MultiscaleConfig(daily_signal=s, weekly_filter=w, exit_mode=e, **fixed)
               for s, w, e in product(entries, protocol['weekly_filters'], protocol['exits'])]
    if len(configs) != protocol['candidate_count'] or len({candidate_id(c) for c in configs}) != len(configs):
        raise ValueError('Registered family count or unique identities differ')
    return configs


def ensure_registration(protocol, config_path, out):
    identity = digest(config_path)
    path = out/'registration.json'
    if path.exists():
        if read(path)['config_sha256'] != identity: raise ValueError('Configuration changed after registration')
    else:
        save(path, {'registered_utc': now(), 'config_sha256': identity,
                    'candidate_count': protocol['candidate_count'], 'new_strategy_returns_observed': False,
                    'prior_results_known': 'Previous US100 results and previously viewed2022-2025 history'})
    return identity


def verify_inputs(protocol, out):
    symbols = Path(protocol['universe']).read_text(encoding="utf-8").split()
    original = Path(protocol['original_universe']).read_text(encoding="utf-8").split()
    if len(symbols) != protocol['universe_count'] or len(set(symbols)) != len(symbols):
        raise ValueError('Expected exactly500 unique frozen symbols')
    if len(original) != 100 or symbols[:100] != original: raise ValueError('Original100 membership/order changed')
    root = Path(protocol['data_root']); calendar = root/'calendar.csv'; files = {}
    for symbol in symbols:
        mp = root/'metadata'/f'{symbol}.json'; m = read(mp)
        if m.get('status') != 'success' or m.get('start') != '2020-01-01' or m.get('end') != '2026-07-12':
            raise ValueError(f'Unexpected request/status: {symbol}')
        if m.get('source') != 'yahoo_finance_chart': raise ValueError(f'Different price provider: {symbol}')
        daily = root/'daily'/f'{symbol}.csv'; raw = root/'raw'/f'{symbol}.json'
        if digest(daily) != m['daily_sha256'] or digest(raw) != m['raw_sha256']:
            raise ValueError(f'Price snapshot hash mismatch: {symbol}')
        files[symbol] = {'daily_sha256': m['daily_sha256'], 'raw_sha256': m['raw_sha256'],
                         'metadata_sha256': digest(mp), 'download_finished_utc': m['download_finished_utc']}
    snapshot = {'market': 'US', 'symbols': symbols, 'universe_sha256': digest(protocol['universe']),
                'original_universe_sha256': digest(protocol['original_universe']),
                'calendar_sha256': digest(calendar), 'files': files,
                'vintage_note': 'Original100 reused earlier snapshot; new400 retain actual later download vintage.'}
    path = out/'input_snapshot.json'
    if path.exists() and read(path) != snapshot: raise ValueError('Frozen input snapshot changed; choose a new research output')
    if not path.exists(): save(path, snapshot)
    return symbols, root, snapshot, digest(path)


def ensure_factors(out):
    target = out/'factors'; previous = Path('runtime_layer/reports/elder_search_v1/factors')
    if not (target/'manifest.json').exists() and (previous/'manifest.json').exists():
        target.mkdir(parents=True, exist_ok=True)
        for name in ['official_ff5_daily.zip', 'ff5_daily.csv', 'manifest.json']:
            if not (previous/name).exists(): break
        else:
            for name in ['official_ff5_daily.zip', 'ff5_daily.csv', 'manifest.json']:
                shutil.copyfile(previous/name, target/name)
    status = fetch_french_rf(target)
    if status['status'] != 'success': raise ValueError(f'Official daily RF unavailable: {status}')
    return pd.read_csv(target/'ff5_daily.csv', index_col=0), digest(target/'ff5_daily.csv')


def _feature_task(task):
    symbol, daily_path, calendar_path, cache_path, identity, warmup = task
    cache_path = Path(cache_path); meta_path = cache_path.with_suffix('.json')
    if cache_path.exists() and meta_path.exists() and read(meta_path) == identity:
        return symbol, str(cache_path), True
    frame = build_features(pd.read_csv(daily_path), pd.read_csv(calendar_path), warmup)
    temporary = cache_path.with_suffix('.partial'); frame.to_pickle(temporary); temporary.replace(cache_path)
    save(meta_path, identity)
    return symbol, str(cache_path), False


def prepare_context(protocol, out, workers=4):
    symbols, root, snapshot, snapshot_sha = verify_inputs(protocol, out)
    factors, factor_sha = ensure_factors(out)
    calendar = pd.read_csv(root/'calendar.csv')
    sessions = calendar.loc[calendar.is_trading_day.astype(str).str.lower().eq('true'), 'date'].astype(str).tolist()
    dates = {label: [s for s in sessions if period[0] <= s <= period[1]]
             for label, period in [('train', protocol['train']), ('validation', protocol['validation'])]}
    rates = {label: factors.loc[d, 'RF'].to_numpy(dtype=float) for label, d in dates.items()}
    if any(not np.isfinite(r).all() for r in rates.values()): raise ValueError('Missing/nonfinite aligned RF')
    feature_root = out/'features'; feature_root.mkdir(parents=True, exist_ok=True)
    feature_sha = digest('research_layer/multiscale_signals.py')
    validator_sha = digest('research_layer/elder_engine.py')
    tasks = [(s, str(root/'daily'/f'{s}.csv'), str(root/'calendar.csv'), str(feature_root/f'{s}.pkl'),
              {'signals_sha256': feature_sha, 'daily_sha256': snapshot['files'][s]['daily_sha256'],
               'calendar_sha256': snapshot['calendar_sha256'], 'legacy_validator_sha256':validator_sha,
               'warmup_weeks': protocol['fixed']['warmup_weeks']},
              protocol['fixed']['warmup_weeks']) for s in symbols]
    features = {}; reused = 0; started = time.perf_counter()
    if workers == 1:
        completed = map(_feature_task, tasks)
        for i, (s, p, old) in enumerate(completed, 1):
            features[s] = pd.read_pickle(p); reused += old
            if i % 25 == 0: print(json.dumps({'phase': 'features', 'completed': i, 'total': len(symbols), 'reused': reused}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (s, p, old) in enumerate(pool.map(_feature_task, tasks), 1):
                features[s] = pd.read_pickle(p); reused += old
                if i % 25 == 0: print(json.dumps({'phase': 'features', 'completed': i, 'total': len(symbols), 'reused': reused}), flush=True)
    windows = {label: {} for label in dates}; eligibility = []
    for symbol in symbols:
        f = features[symbol]; arrays = prepare_arrays(f)
        for label in dates:
            windows[label][symbol] = prepare_window(arrays, *protocol[label], evaluation_dates=dates[label])
        ready_dates = f.loc[f.ready.astype(bool), 'datetime'].astype(str).str[:10].tolist()
        eligibility.append({'symbol': symbol, 'first_bar': str(f.datetime.iloc[0])[:10],
                            'first_ready_signal_close': ready_dates[0] if ready_dates else None,
                            **{label+'_eligible_sessions': int(windows[label][symbol].flags['prior_ready'].sum()) for label in dates}})
    save(out/'feature_preparation.json', {'finished_utc': now(), 'symbols': len(symbols), 'reused': reused,
         'elapsed_seconds': time.perf_counter()-started, 'signals_sha256': feature_sha, 'eligibility': eligibility})
    identity = {'input_snapshot_sha256': snapshot_sha, 'factor_csv_sha256': factor_sha,
                'signals_sha256': feature_sha, 'engine_sha256': digest('research_layer/multiscale_engine.py'),
                'legacy_validator_sha256':validator_sha,
                'metric_source_sha256':digest('research_layer/elder_analysis.py'),
                'runner_sha256': digest(__file__)}
    return {'symbols': symbols, 'features': features, 'windows': windows, 'dates': dates,
            'rates': rates, 'factors': factors, 'identity': identity}


STOCK_FIELDS = ['total_return', 'closed_trades', 'wins', 'losses', 'breakeven_count', 'gross_profit_cash',
                'gross_loss_cash', 'total_fees', 'cash_interest', 'closed_cash_pnl', 'unrealized_cash_pnl',
                'mean_close_exposure', 'open_trades']
SUM_FIELDS = ['closed_trades', 'wins', 'losses', 'breakeven_count', 'gross_profit_cash', 'gross_loss_cash',
              'total_fees', 'cash_interest', 'closed_cash_pnl', 'unrealized_cash_pnl', 'open_trades',
              'sum_net_trade_returns', 'sum_R_multiples', 'sum_win_returns', 'sum_loss_returns',
              'sum_holding_bars', 'sum_mae', 'sum_mfe']


def aggregate_case(config, symbols, windows, rates, report_rf, failure_path=None):
    n = len(next(iter(windows.values())).dates); nav_sum = np.zeros(n); value_sum = np.zeros(n)
    group_nav = np.zeros((2,n))
    sums = dict.fromkeys(SUM_FIELDS, 0.); stocks = np.empty((len(symbols), len(STOCK_FIELDS)))
    diagnostics = {'entries': 0, 'blocked_sessions': 0, 'risk_budget_breaches_at_assumed_stop': 0}
    for i, symbol in enumerate(symbols):
        try:
            r = run_strategy(windows[symbol], config, cash_daily_rates=rates, include_trades=False)
            nav_sum += r['nav']; value_sum += r['nav']*r['exposure']; m = r['metrics']
            group_nav[0 if i < 100 else 1] += r['nav']
            stocks[i] = [m[k] for k in STOCK_FIELDS]
            for key in SUM_FIELDS: sums[key] += m[key]
            for key in diagnostics: diagnostics[key] += r['diagnostics'][key]
        except Exception as exc:
            if failure_path is not None:
                save(failure_path, {'at_utc': now(), 'candidate': candidate_id(config), 'symbol': symbol,
                                    'type': type(exc).__name__, 'error': str(exc), 'completed_stocks_before_failure': i})
            raise
    nav = nav_sum/len(symbols); exposure = np.divide(value_sum, nav_sum)
    closed = int(sums['closed_trades']); wins = int(sums['wins']); losses = int(sums['losses'])
    count = len(symbols)
    ledger_total = (sums['closed_cash_pnl']+sums['unrealized_cash_pnl']+sums['cash_interest'])/count
    if not np.isclose(nav[-1]-1, ledger_total, atol=1e-11): raise ValueError('Aggregate ledger does not reconcile with NAV')
    trades = {'closed': closed, 'open': int(sums['open_trades']), 'wins': wins, 'losses': losses,
              'breakeven': int(sums['breakeven_count']), 'win_rate': wins/closed if closed else None,
              'nonloss_rate': (closed-losses)/closed if closed else None,
              'profit_factor': sums['gross_profit_cash']/sums['gross_loss_cash'] if sums['gross_loss_cash'] else None,
              'gross_profit_cash': sums['gross_profit_cash'], 'gross_loss_cash': sums['gross_loss_cash'],
              'mean_net_trade_return': sums['sum_net_trade_returns']/closed if closed else None,
              'mean_R_multiple': sums['sum_R_multiples']/closed if closed else None,
              'average_win_return': sums['sum_win_returns']/wins if wins else None,
              'average_loss_return': sums['sum_loss_returns']/losses if losses else None,
              'mean_holding_bars': sums['sum_holding_bars']/closed if closed else None,
              'mean_mae': sums['sum_mae']/closed if closed else None,
              'mean_mfe': sums['sum_mfe']/closed if closed else None,
              'realized_pnl_per_initial_capital': sums['closed_cash_pnl']/count,
              'unrealized_pnl_per_initial_capital': sums['unrealized_cash_pnl']/count,
              'total_fees_per_initial_capital': sums['total_fees']/count}
    result = {'id': candidate_id(config), 'config': config.to_dict(), 'metrics': metrics(nav, report_rf),
              'trades': trades, 'cash_interest_per_initial_capital': sums['cash_interest']/count,
              'mean_capital_weighted_exposure': float(exposure.mean()),
              'stocks_with_closed_trades': int((stocks[:,1] > 0).sum()),
              'profitable_stocks': int((stocks[:,0] > 0).sum()),
              'median_stock_return': float(np.median(stocks[:,0])), 'diagnostics': diagnostics}
    result['universe_groups']={}
    for name,idx,lo,hi in [('original100',0,0,min(100,count)),('added400',1,100,count)]:
        if hi <= lo:continue
        group=stocks[lo:hi];number=hi-lo;gp=group[:,5].sum();gl=group[:,6].sum();closed_group=group[:,1].sum()
        result['universe_groups'][name]={'symbols':number,'metrics':metrics(group_nav[idx]/number,report_rf),
            'closed_trades':int(closed_group),'net_win_rate':float(group[:,2].sum()/closed_group) if closed_group else None,
            'profit_factor':float(gp/gl) if gl else None,
            'cash_interest_per_initial_capital':float(group[:,8].mean())}
    return result, nav, exposure, stocks


def benchmark(windows, symbols, rates, cost_bps):
    n = len(rates); total = np.zeros(n); cost = cost_bps/10000.
    for symbol in symbols:
        w = windows[symbol]; cash = 1.; units = 0.; nav = np.zeros(n)
        for i in range(n):
            cash *= 1+rates[i]
            if units == 0 and w.flags['tradeable'][i]:
                units = cash/(w.numeric['open'][i]*(1+cost)); cash = 0.
            nav[i] = cash+units*w.numeric['close'][i]
        total += nav
    nav = total/len(symbols)
    return {'kind': 'Same500 equal initial full-allocation buy-and-hold; enter first observed tradeable open, RF cash before listing; no100-week warmup; entry fee, no forced final sale',
            'nav': nav, 'metrics': metrics(nav, rates),
            'RF_cash_metrics': metrics(np.cumprod(1+rates), rates)}


def run_stage(label, protocol, configs, context, config_sha, out):
    stage = out/label; stage.mkdir(parents=True, exist_ok=True)
    identity = {**context['identity'], 'config_sha256': config_sha, 'period': protocol[label],
                'symbols': context['symbols'], 'cash_policy': 'French daily RF prior-close cash proxy'}
    if label=='validation':
        training_identity=read(out/'train/results.json')['identity']
        if {k:v for k,v in training_identity.items() if k!='period'}!={k:v for k,v in identity.items() if k!='period'}:
            raise ValueError('Train/validation source, configuration or input identity mismatch')
    marker = stage/'stage_identity.json'
    if marker.exists():
        registered = read(marker)
        if registered['identity'] != identity: raise ValueError(f'Stage identity mismatch: {label}')
    else:
        registered = {'started_utc': now(), 'identity': identity}; save(marker, registered)
    rows = []; navs = []; dates = context['dates'][label]; rates = context['rates'][label]
    for i, config in enumerate(configs, 1):
        case = candidate_id(config); path = stage/(case+'.json'); npz = stage/(case+'.npz')
        if path.exists():
            prior = read(path)
            if prior['config'] != config.to_dict() or not npz.exists() or prior['array_sha256'] != digest(npz):
                raise ValueError(f'Candidate checkpoint identity/hash mismatch: {case}')
            result = prior['result']
            with np.load(npz) as arrays: nav = arrays['nav'].copy()
        else:
            result, nav, exposure, stocks = aggregate_case(config, context['symbols'], context['windows'][label],
                rates, rates, stage/'failure.json')
            save_npz(npz, nav=nav, exposure=exposure, stock_metrics=stocks,
                     stock_fields=np.asarray(STOCK_FIELDS), symbols=np.asarray(context['symbols']))
            save(path, {'config': config.to_dict(), 'array_sha256': digest(npz), 'result': result})
        rows.append(result); navs.append(nav)
        if i % 20 == 0 or i == len(configs):
            print(json.dumps({'phase': label, 'candidates': i, 'total_candidates': len(configs),
                              'completed_stock_tasks': i*len(context['symbols'])}), flush=True)
    baseline = benchmark(context['windows'][label], context['symbols'], rates, protocol['fixed']['cost_bps'])
    arrays_path = stage/'nav_matrix.npz'
    if not arrays_path.exists():
        save_npz(arrays_path, nav=np.column_stack(navs), dates=np.asarray(dates), ids=np.asarray([r['id'] for r in rows]),
                 benchmark_nav=baseline['nav'], rf=rates)
    else:
        with np.load(arrays_path) as prior:
            if not np.array_equal(prior['nav'], np.column_stack(navs)): raise ValueError('Stage NAV matrix changed on resume')
    summary = {'stage': label, 'started_utc': registered['started_utc'], 'finished_utc': now(),
               'period': protocol[label], 'dates': dates, 'identity': identity,
               'completed_tasks': len(configs)*len(context['symbols']), 'failed_tasks': 0,
               'candidate_count': len(configs), 'symbols': len(context['symbols']), 'candidates': rows,
               'benchmark': {k:v for k,v in baseline.items() if k != 'nav'},
               'nav_matrix_sha256': digest(arrays_path)}
    path = stage/'results.json'
    if path.exists():
        prior = read(path)
        if any(prior[k] != plain(summary[k]) for k in summary if k != 'finished_utc'):
            raise ValueError('Stage summary changed on resume')
        return prior
    save(path, summary)
    return summary


def freeze_selection(training, protocol, out):
    s = protocol['selection']; rows = training['candidates']
    eligible = [r for r in rows if r['trades']['closed'] >= s['minimum_closed_trades']
                and r['stocks_with_closed_trades'] >= s['minimum_stocks_with_closed_trades']]
    def score(row): return row['metrics']['mean_annualized_excess_rf']/max(row['metrics']['maximum_drawdown'], .01)
    def pf(row): return row['trades']['profit_factor'] or 0.
    balanced = sorted(eligible, key=lambda r: (-score(r), -pf(r), r['id']))
    viable = [r for r in eligible if r['metrics']['mean_annualized_excess_rf'] > 0 and pf(r) > 1]
    high_win = sorted(viable, key=lambda r: (-r['trades']['win_rate'], -pf(r), r['id']))
    high_return = sorted(eligible, key=lambda r: (-r['metrics']['total_return'], -pf(r), r['id']))
    baseline_ids = [candidate_id(MultiscaleConfig(**b)) for b in s['baselines']]
    chosen = {'balanced_top5': [r['id'] for r in balanced[:s['balanced_top_count']]],
              'high_win': high_win[0]['id'] if high_win else None,
              'high_return': high_return[0]['id'] if high_return else None, 'baselines': baseline_ids}
    selected = sorted(set(chosen['balanced_top5']+baseline_ids+[v for k,v in chosen.items() if k in ['high_win','high_return'] and v]))
    frozen = {'selected_utc': now(), 'train_results_sha256': digest(out/'train/results.json'),
              'eligible_candidate_count': len(eligible), 'positive_excess_and_profit_factor_candidate_count': len(viable),
              'information': s['information'], 'selected': chosen, 'selected_ids': selected,
              'ranking': [{'id':r['id'], 'score':score(r)} for r in balanced],
              'not_untouched': 'Previously viewed history and current survivor universe; ranking is retrospective.'}
    path = out/'frozen_selection.json'
    if path.exists():
        prior = read(path)
        if any(prior[k] != plain(frozen[k]) for k in frozen if k != 'selected_utc'):
            raise ValueError('Frozen selection changed; cannot overwrite after validation')
        return prior
    save(path, frozen)
    return frozen


def require_frozen(out):
    frozen = read(out/'frozen_selection.json')
    if digest(out/'train/results.json') != frozen['train_results_sha256']: raise ValueError('Training signature changed after selection')
    return frozen


def require_matching_training_context(out,context):
    path=out/'train/results.json'
    if not path.exists():return
    identity=read(path)['identity']
    if any(identity.get(k)!=v for k,v in context['identity'].items()) or identity['symbols']!=context['symbols']:
        raise ValueError('Current source/data context differs from completed training')


def run_events(protocol, context, out, labels=('train','validation')):
    for label in labels:
        if label=='validation':require_frozen(out)
        result = run_event_study(context['features'], *protocol[label],
            horizons=tuple(protocol['events']['holding_sessions']), cost_bps=protocol['events']['cost_bps'],
            pair_signals=protocol['events']['pair_signals'], include_weekly_signals=protocol['events']['include_weekly_signals'],
            include_regimes=protocol['events']['include_regimes'], include_date_aggregates=False)
        result['input_identity'] = context['identity']; result['event_source_sha256'] = digest('research_layer/multiscale_events.py')
        save(out/f'{label}_events.json', result)
        print(json.dumps({'phase': 'events', 'period': label, 'cases': result['case_count'], 'symbols': result['symbol_count']}), flush=True)


def selected_sensitivities(protocol, configs, context, out):
    require_matching_training_context(out,context)
    frozen = require_frozen(out); validation = read(out/'validation/results.json')
    base = {r['id']:r for r in validation['candidates']}; chosen = [c for c in configs if candidate_id(c) in frozen['selected_ids']]
    rows = []; rates = context['rates']['validation']; window = context['windows']['validation']
    for config in chosen:
        case = candidate_id(config)
        for cost in protocol['selected_sensitivities']['cost_bps']:
            result = base[case] if cost == protocol['fixed']['cost_bps'] else aggregate_case(
                replace(config, cost_bps=cost), context['symbols'], window, rates, rates, out/'sensitivity_failure.json')[0]
            rows.append({'id':case, 'cost_bps':cost, 'cash_interest':True, 'result':result})
        if protocol['selected_sensitivities']['zero_cash_interest']:
            result = aggregate_case(config, context['symbols'], window, np.zeros_like(rates), rates, out/'sensitivity_failure.json')[0]
            rows.append({'id':case, 'cost_bps':config.cost_bps, 'cash_interest':False, 'result':result})
        print(json.dumps({'phase':'selected_sensitivities','id':case}),flush=True)
    save(out/'selected_sensitivities.json', {'selection_sha256':digest(out/'frozen_selection.json'), 'rows':rows,
          'completed_stock_tasks': len(chosen)*(len(protocol['selected_sensitivities']['cost_bps'])-1+int(protocol['selected_sensitivities']['zero_cash_interest']))*len(context['symbols']),
          'reused_base_stock_tasks':len(chosen)*len(context['symbols']), 'failed_stock_tasks':0})


def analyze(protocol, out):
    frozen = require_frozen(out); train = read(out/'train/results.json'); validation = read(out/'validation/results.json')
    if {k:v for k,v in train['identity'].items() if k!='period'}!={k:v for k,v in validation['identity'].items() if k!='period'}:
        raise ValueError('Train/validation source, configuration or input identity mismatch')
    if not train['finished_utc'] <= frozen['selected_utc'] <= validation['started_utc']:
        raise ValueError('Frozen selection must precede validation')
    path = out/'validation/nav_matrix.npz'
    if digest(path) != validation['nav_matrix_sha256']: raise ValueError('Validation matrix hash changed')
    with np.load(path) as matrix:
        nav=matrix['nav'].copy();dates=matrix['dates'].tolist();ids=matrix['ids'].tolist()
    factors, rf_sha = ensure_factors(out)
    if rf_sha != validation['identity']['factor_csv_sha256']: raise ValueError('RF snapshot changed')
    s=protocol['statistics']
    stats=assess_multiscale_family(nav,dates,factors['RF'],frozen['selected_ids'],ids,
          B=s['bootstrap_repetitions'],blocks=tuple(s['block_sessions']),seed=s['seed'],
          lifetime_trials_minimum=protocol['lifetime_trials_minimum'],dsr_trial_scenarios=tuple(s['dsr_trial_scenarios']),cash_interest=True)
    stats['statistics_source_sha256']=digest('research_layer/multiscale_statistics.py');save(out/'family_statistics.json',stats)
    training={r['id']:r for r in train['candidates']};rows=[]
    for r in validation['candidates']:
        rows.append({'id':r['id'],'config':r['config'],'selected_from_training':r['id'] in frozen['selected_ids'],
             'train':training[r['id']],'validation':r})
    result={'selection':frozen,'train_period':protocol['train'],'validation_period':protocol['validation'],
          'candidate_count':len(rows),'stock_count':validation['symbols'],'candidates':rows,
          'completed_core_stock_tasks':train['completed_tasks']+validation['completed_tasks'],'failed_core_stock_tasks':0,
          'train_benchmark':train['benchmark'],'validation_benchmark':validation['benchmark'],
          'not_untouched':'Retrospective chronological analysis on viewed history, current survivor universe and endpoint screening; no prospective proof.'}
    save(out/'research_comparison.json',result)
    flat=[{'id':r['id'],'selected_from_training':r['selected_from_training'],
           **{f'{stage}_{k}':v for stage in ['train','validation'] for k,v in r[stage]['metrics'].items()},
           **{f'{stage}_trade_{k}':v for stage in ['train','validation'] for k,v in r[stage]['trades'].items()}}
          for r in rows]
    pd.DataFrame(flat).to_csv(out/'candidate_comparison.csv',index=False)
    print(json.dumps({'phase':'statistics','candidate_count':len(rows),'status':'completed'}),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/research/multiscale_us500_v1.json')
    parser.add_argument('--stage',choices=['prepare','train','validate','events','costs','analyze','all'],default='all')
    parser.add_argument('--workers',type=int,default=4,help='Parallel feature preparation workers; strategy cases retain deterministic order.')
    args=parser.parse_args(argv)
    if args.workers < 1 or args.workers > 8:parser.error('--workers must be1..8')
    protocol=read(args.config);out=Path(protocol['output']);out.mkdir(parents=True,exist_ok=True)
    config_sha=ensure_registration(protocol,args.config,out);configs=family_configs(protocol)
    if args.stage=='analyze':analyze(protocol,out);return 0
    context=prepare_context(protocol,out,args.workers)
    require_matching_training_context(out,context)
    if args.stage=='prepare':return 0
    if args.stage=='all':run_events(protocol,context,out,labels=('train',))
    if args.stage in ['train','all']:
        training=run_stage('train',protocol,configs,context,config_sha,out);freeze_selection(training,protocol,out)
    elif args.stage in ['validate','costs']:
        require_frozen(out)
    if args.stage in ['validate','all']:
        if args.stage=='all':run_events(protocol,context,out,labels=('validation',))
        require_frozen(out);run_stage('validation',protocol,configs,context,config_sha,out)
    if args.stage=='events':
        require_frozen(out);run_events(protocol,context,out)
    if args.stage in ['costs','all']:selected_sensitivities(protocol,configs,context,out)
    if args.stage=='all':analyze(protocol,out)
    print(json.dumps({'status':'completed','stage':args.stage,'output':str(out)}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
