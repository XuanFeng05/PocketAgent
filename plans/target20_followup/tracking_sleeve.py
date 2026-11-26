"""Five-stock tracking feasibility test registered after the 100-stock account was examined.

Freeze: use the broad replacement's exact planned total allocation in each of the
30 defensive months. At each signal close, use 63 completed daily returns for all
then-eligible broad basket stocks. The fit target is their equally weighted daily
return, rebalanced daily *only as a fitting series*. Greedily add the name that
minimizes the in-window mean squared difference between the target and a basket
equal-weighted across the names chosen so far. Ties resolve by symbol. Rebuild
each month, select five slots, keep unfilled slots in cash. Daily returns used in
the fit are point-in-time; realized performance uses monthly share holdings and
next-open fills. In other months reuse the original defensive5 target. Existing
0/10/20bps assumptions, no extra windows, rank weights or switches to optimize.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from plans.target20.run_study import checked, read, sha, write
from plans.target20.analyze import slices
from plans.target20_followup.broad_sleeve import alternatives, cash_audit
from plans.target20_followup.cash_diagnostic import metrics, shadow
from research_layer.technical_portfolio import run_technical_portfolio


def selection(closes, date, eligible):
    """Use the same historical rows and fit target for every selection stage."""
    names = sorted(eligible)
    if not names:
        return [], None
    cutoff = closes.loc[:date, names].tail(64)
    if len(cutoff) != 64 or cutoff.index[-1] != date or cutoff.isna().any().any():
        raise ValueError('Incomplete synchronized fitting history')
    history = cutoff.to_numpy(dtype=float)
    returns = history[1:] / history[:-1] - 1
    if not np.isfinite(returns).all():
        raise ValueError('Nonfinite fitting return')
    target = returns.mean(axis=1)
    total = np.zeros(63); indices = []; residual = None
    for _ in range(min(5,len(names))):
        errors = (returns + total[:,None]) / (len(indices)+1) - target[:,None]
        mse = np.mean(errors**2,axis=0)
        mse[indices] = np.inf
        chosen = int(np.argmin(mse))
        indices.append(chosen)
        total += returns[:,chosen]
        residual = float(math.sqrt(mse[chosen]))
    return [names[i] for i in indices], residual


def test_selection():
    dates = pd.date_range('2020-01-01',periods=72).strftime('%Y-%m-%d')
    # Exact ties under identical prices force deterministic lexical order.
    panel = pd.DataFrame(1.,index=dates,columns=list('GFADECB'))
    chosen, rmse = selection(panel, dates[63], list(panel.columns))
    assert chosen == list('ABCDE') and rmse == 0
    panel.loc[dates[64]:,'G'] = np.arange(1.,9.)
    assert selection(panel, dates[63], list(panel.columns)) == (chosen, rmse)


def study(run, broad_report):
    test_selection()
    manifest = read(run/'artifacts.json')['sha256']
    def pinned(rel):
        file = run/rel; checked(file,manifest[rel]); return read(file)
    freeze = pinned('freeze.json'); c = freeze['configuration']
    for src,h in freeze['implementation'].items(): checked(ROOT/src,h)
    prior = read(broad_report)
    if prior['source_artifact_manifest_sha256'] != sha(run/'artifacts.json'):
        raise ValueError('Different parent account')
    frames, tables, ledger_prices = {}, {}, {}
    for symbol, item in sorted(freeze['inputs'].items()):
        file = Path(item['path']); checked(file,item['sha256'])
        f = pd.read_csv(file); frames[symbol] = f; tables[symbol] = f.set_index(f.datetime.str[:10],verify_integrity=True)
        ledger_prices[symbol] = {str(r.datetime)[:10]:(r.open,r.volume,r.close) for r in f.itertuples(index=False)}
    closes = pd.DataFrame({s:t.close for s,t in tables.items()}).sort_index()
    cp = Path(freeze['calendar']['path']);checked(cp, freeze['calendar']['sha256']);cal = pd.read_csv(cp)
    original = pinned('decisions/defensive5.json'); equal = pinned('decisions/equal100.json')
    old_base = pinned('cases/defensive5_10bps.json')
    kwargs = dict(start=c['start'],end=c['end'])
    if run_technical_portfolio(frames,cal,decisions=original,cost_bps=10,**kwargs) != old_base:
        raise ValueError('Original defensive account reproduction failed')
    _, broad, info = alternatives(original,equal)
    orders, diagnostics = [], []
    prefix_cases = 0
    for i,(o, b, item) in enumerate(zip(original,broad,info,strict=True)):
        date = item['signal_date']; off = item['risk_off']
        if off:
            names = equal[i]['weights'].keys()
            chosen,fit_error = selection(closes,date,names)
            if selection(closes.loc[:date],date,names) != (chosen, fit_error):
                raise ValueError('Future observations influence selected names')
            prefix_cases += 1
            weights = {s:item['broad_total']/5 for s in chosen if item['broad_total']>0}
            if len(chosen) == 5 and not math.isclose(math.fsum(weights.values()),item['broad_total'],abs_tol=1e-12):
                raise ValueError('Mismatched broad exposure')
        else:
            weights = dict(o['weights']); chosen = list(weights); fit_error = None
            if weights != b['weights']:raise ValueError('Risk-on targets differ')
        orders.append({'signal_date':date,'weights':weights})
        start = old_base['rebalances'][i]['execution_date']; terminal = i+1==len(info)
        end = old_base['end'] if terminal else old_base['rebalances'][i+1]['execution_date']
        # These normalized gross windows assess realized sleeve approximation at the common target budget.
        small_period = shadow(tables,weights,start,end,terminal)
        broad_period = shadow(tables,b['weights'],start,end,terminal)
        diagnostics.append({'signal_date':date,'risk_off':off,'start_open':start,'end':end,'end_is_terminal_close':terminal,
                            'weights':weights,'fit_daily_unscaled_rmse':fit_error,'target_budget':item['broad_total'],
                            'small':small_period,'broad':broad_period,'gross_monthly_difference':small_period['return']-broad_period['return']})
    differences = [m['gross_monthly_difference'] for m in diagnostics if m['risk_off']]
    results = {}; max_error = 0.; max_filled = 0
    for cost in (0,10,20):
        output = run_technical_portfolio(frames,cal,decisions=orders,cost_bps=cost,**kwargs)
        comparator = run_technical_portfolio(frames,cal,decisions=broad,cost_bps=cost,**kwargs)
        max_error = max(max_error, cash_audit(output,ledger_prices,cost),cash_audit(comparator,ledger_prices,cost))
        if len(output['rebalances']) != 48:
            raise ValueError('Missing scheduled rebalances')
        base_metrics = metrics(comparator,c)
        if any(v != prior['by_cost'][str(cost)]['broad_basket'][k] for k,v in base_metrics.items()):
            raise ValueError('Broad sleeve reproduction differs')
        m = metrics(output,c)
        _, months = slices(output['equity_curve'])
        _, broad_months = slices(comparator['equity_curve'])
        actual_differences = [months[k]-broad_months[k] for k in months]
        max_filled = max(max_filled, output['metrics']['max_positions'])
        m.update(worst_month=min(months.items(),key=lambda v:v[1]),max_positions=output['metrics']['max_positions'],
                 buys=output['metrics']['buy_count'],sells=output['metrics']['sell_count'],blocked=sum(len(r['blocked']) for r in output['rebalances']),
                 net_monthly_return_difference_vs_broad_rmse=math.sqrt(mean(v*v for v in actual_differences)),
                 monthly_returns=months,total_fee=output['metrics']['total_fees'])
        results[str(cost)] = {**prior['by_cost'][str(cost)],'tracking5':m}
    return {'design':__doc__.strip(),'source_run':run.name,'by_cost':results,'windows':diagnostics,
            'execution_window_tracking':{'window':'30 risk-off, from open to next open, last to final close, gross at matched target allocation. Nontradeable endpoint marks separately identified. Positive differences favor tracking5.',
                                         'count':len(differences),'mean_difference':mean(differences),
                                         'rmse':math.sqrt(mean(v*v for v in differences)),
                                         'mean_absolute_difference':mean(abs(v) for v in differences),
                                         'min_difference':min(differences),'max_difference':max(differences)},
            'checks':{'parent_original_exact':True,'broad_metrics_exact_costs':[0,10,20],'real_prefix_checks':prefix_cases,'max_ledger_error':max_error,'max_actual_positions_seen':max_filled},
            'prior_file_sha256':sha(broad_report),'script_sha256':sha(Path(__file__))}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--broad',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();report=study(args.run,args.broad);write(args.output,report)
    print(json.dumps({'checks':report['checks'],'window_tracking':report['execution_window_tracking'],
                      'tracking5':{k:r['tracking5'] for k,r in report['by_cost'].items()}},indent=2))
