"""Reproduce conservative source-quality and eligibility-matched benchmarks.

Read the completed registered US500 experiment; retain its selected IDs and500
capital denominator. Replace only predeclared suspect-symbol accounts withRF
cash, never synthesize or repair quotes. All generated artifacts remain inside
the main protocol's runtime output. These are additional descriptive/data
handling sensitivities, not a replacement for primary results or pristine OOS.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from research_layer.multiscale_engine import MultiscaleConfig,prepare_arrays,prepare_window,run_strategy
from research_layer.elder_analysis import metrics
from research_layer.multiscale_statistics import assess_multiscale_family


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b):
    if not np.isclose(a,b,rtol=1e-10,atol=1e-12):raise ValueError(f"Recomputed stock/ledger differs: {a} vs {b}")


def run_cash_substitution(ROOT,protocol,symbols,replaced,frozen,quality_config_sha):
    selected=frozen['selected_ids']
    read=lambda name:json.loads((ROOT/name).read_text(encoding="utf-8"))
    feature={s:prepare_arrays(pd.read_pickle(ROOT/'features'/f'{s}.pkl')) for s in replaced}
    rows=[];max_stock_reproduction_error=0.;period_results={};quality_matrix={}
    for label in ['train','validation']:
        stage=read(f'{label}/results.json');base={r['id']:r for r in stage['candidates']}
        with np.load(ROOT/label/'nav_matrix.npz') as a:
            matrix=a['nav'].copy();dates=a['dates'].tolist();ids=a['ids'].tolist();rf=a['rf'].copy();benchmark_nav=a['benchmark_nav'].copy()
        cash_nav=np.cumprod(1+rf)
        windows={s:prepare_window(feature[s],*protocol[label],evaluation_dates=dates) for s in replaced}
        adjusted_matrix=np.empty_like(matrix);phase_rows=[]
        for candidate_number,identifier in enumerate(ids):
            original=base[identifier];config=MultiscaleConfig(**original['config'])
            with np.load(ROOT/label/(identifier+'.npz')) as a:
                stock=a['stock_metrics'].copy();fields=a['stock_fields'].tolist();array_symbols=a['symbols'].tolist()
            removed=[];nav=matrix[:,ids.index(identifier)].copy()
            for s in replaced:
                r=run_strategy(windows[s],config,cash_daily_rates=rf,include_trades=False);m=r['metrics'];removed.append(m)
                old=stock[array_symbols.index(s)]
                for j,key in enumerate(fields):
                    error=abs(float(old[j])-m[key]);max_stock_reproduction_error=max(max_stock_reproduction_error,error)
                    close(old[j],m[key])
                nav+=(cash_nav-r['nav'])/500
            t=copy.deepcopy(original['trades']);old_t=original['trades']
            for key,mkey in [('closed','closed_trades'),('open','open_trades'),('wins','wins'),('losses','losses'),('breakeven','breakeven_count')]:
                t[key]-=sum(m[mkey] for m in removed)
            for key in ['gross_profit_cash','gross_loss_cash']:
                t[key]-=sum(m[key] for m in removed)
                if abs(t[key])<1e-12:t[key]=0.
            t['win_rate']=t['wins']/t['closed'] if t['closed'] else None
            t['nonloss_rate']=(t['closed']-t['losses'])/t['closed'] if t['closed'] else None
            t['profit_factor']=t['gross_profit_cash']/t['gross_loss_cash'] if t['gross_loss_cash'] else None
            for key,mkey in [('realized_pnl_per_initial_capital','closed_cash_pnl'),('unrealized_pnl_per_initial_capital','unrealized_cash_pnl'),('total_fees_per_initial_capital','total_fees')]:
                t[key]-=sum(m[mkey] for m in removed)/500
            for key,mkey,den in [('mean_net_trade_return','sum_net_trade_returns','closed'),('mean_R_multiple','sum_R_multiples','closed'),
                                 ('average_win_return','sum_win_returns','wins'),('average_loss_return','sum_loss_returns','losses'),
                                 ('mean_holding_bars','sum_holding_bars','closed'),('mean_mae','sum_mae','closed'),('mean_mfe','sum_mfe','closed')]:
                total=(old_t[key] or 0)*old_t[den]-sum(m[mkey] for m in removed)
                t[key]=total/t[den] if t[den] else None
            interest=original['cash_interest_per_initial_capital']-sum(m['cash_interest'] for m in removed)/500+len(replaced)*(cash_nav[-1]-1)/500
            close(nav[-1]-1,t['realized_pnl_per_initial_capital']+t['unrealized_pnl_per_initial_capital']+interest)
            row={'stage':label,'id':identifier,'selected_from_original_training':identifier in selected,'metrics':metrics(nav,rf),'trades':t,'cash_interest_per_initial_capital':float(interest),
                         'original_total_return':original['metrics']['total_return'],
                         'total_return_change':float(nav[-1]-1-original['metrics']['total_return']),
                         'removed_stock_metrics':dict(zip(replaced,removed)) if identifier in selected else None}
            phase_rows.append(row);adjusted_matrix[:,candidate_number]=nav
            if identifier in selected:rows.append(row)
            if (candidate_number+1)%200==0:print(json.dumps({'quality_stage':label,'candidates':candidate_number+1,'total':len(ids)}),flush=True)
        # Apply the same three cash sleeves to the original full-allocation benchmark.
        for s in replaced:
            w=windows[s];cash=1.;units=0.;single=[]
            for k in range(len(dates)):
                cash*=1+rf[k]
                if units==0. and w.flags['tradeable'][k]:
                    units=cash/(w.numeric['open'][k]*(1+protocol['fixed']['cost_bps']/10000.));cash=0.
                single.append(cash+units*w.numeric['close'][k])
            benchmark_nav+=(cash_nav-np.asarray(single))/500
        target=ROOT/f'quality_{label}';target.mkdir(exist_ok=True)
        with (target/'nav_matrix.npz').open('wb') as f:
            np.savez_compressed(f,dates=np.asarray(dates),ids=np.asarray(ids),nav=adjusted_matrix,rf=rf,benchmark_nav=benchmark_nav)
        phase={'period':protocol[label],'candidate_count':1540,'symbols':500,'RF_cash_replaced_symbols':replaced,
           'original_selection_reused':True,'candidates':phase_rows,'quality_buy_hold_metrics':metrics(benchmark_nav,rf),
           'primary_results_sha256':sha(ROOT/label/'results.json'),'matrix_sha256':sha(target/'nav_matrix.npz')}
        (target/'results.json').write_text(json.dumps(phase,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
        period_results[label]={k:v for k,v in phase.items() if k!='candidates'}
        quality_matrix[label]=(adjusted_matrix,dates,rf,ids)

    nav,dates,rf,ids=quality_matrix['validation'];s=protocol['statistics']
    statistics=assess_multiscale_family(nav,dates,rf,selected,ids,B=s['bootstrap_repetitions'],blocks=tuple(s['block_sessions']),
       seed=s['seed'],lifetime_trials_minimum=protocol['lifetime_trials_minimum'],dsr_trial_scenarios=tuple(s['dsr_trial_scenarios']),cash_interest=True)
    statistics['quality_sensitivity_scope']='Same1540 parameter sets;SW/FERG/TLN cash sleeves retained in500 denominator; original selected IDs fixed. This is an additional data-handling scenario, not corrected historical quotes.'
    statistics['limitations'].append('Do not choose the favorable main/quality scenario after results. Additional data-handling and financing sensitivity trials are not all included in the1612 strategy-parameter lower bound.')
    (ROOT/'quality_family_statistics.json').write_text(json.dumps(statistics,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")

    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed',
       'selection_sha256':sha(ROOT/'frozen_selection.json'),'replaced_with_RF_cash':replaced,'denominator':500,
       'selected_candidate_count':len(selected),'full_candidate_count_per_period':1540,'evaluated_periods':['train','validation'],'selected_rows':rows,
       'periods':period_results,'quality_config_sha256':quality_config_sha,
       'max_removed_stock_metric_reproduction_error':max_stock_reproduction_error,
       'scope':'All1540 existing parameter sets in both periods;500 mean adjusted by subtracting exactSW/FERG/TLN sleeve NAVs and adding three RF cash sleeves. No deletion, synthetic prices, new selection or modification of the primary-family results.',
       'interpretation':'Cash replacement is a conservative availability sensitivity, not repaired historical prices or proof that every remaining source-data lineage is correct. Main/quality comparisons must be reported together, not cherry-picked.'}
    (ROOT/'quality_sensitivity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
    return {'quality_sensitivity':'completed','selected':len(selected),'full_cases':3080,'reproduction_error':max_stock_reproduction_error}


def run_eligibility_benchmarks(ROOT,p,symbols,replaced):
    fees=p['fixed']['cost_bps']/10000.
    features={s:pd.read_pickle(ROOT/'features'/f'{s}.pkl') for s in symbols}
    rows={};quality_rows={};never_ready=[]
    for s,f in features.items():
        ix=(f.datetime>=pd.Timestamp('2022-01-01'))&(f.datetime<=pd.Timestamp('2025-12-31'))
        if not f.loc[ix,'ready'].any():never_ready.append(s)
    for label in ['train','validation']:
        with np.load(ROOT/label/'nav_matrix.npz') as m:
            dates=m['dates'].tolist();rf=m['rf'].copy()
        navs=[];details=[];eligibility_at_first=eligibility_at_last=0
        for s in symbols:
            f=features[s];fd=f.datetime.dt.strftime('%Y-%m-%d').to_numpy();n=len(f)
            o,h,l,c=[f[k].to_numpy(dtype=float) for k in ['open','high','low','close']]
            ready=f.ready.to_numpy(dtype=bool);tradeable=f.tradeable.to_numpy(dtype=bool)
            price_eligible=f.universe_trade_eligible.to_numpy(dtype=bool)
            cash=1.;units=0.;interest=0.;exposure=[];nav=[];entry_date=None
            first_ready=None
            for j,date in enumerate(dates):
                credit=cash*rf[j];cash+=credit;interest+=credit
                k=int(np.searchsorted(fd,date))
                observed=k<n and fd[k]==date
                before_listing=date<fd[0]
                assert observed or before_listing,(s,date)
                prior=k-1
                prior_ready=observed and prior>=0 and ready[prior]
                if prior_ready and first_ready is None:first_ready=date
                if j==0:eligibility_at_first+=int(prior_ready)
                if j==len(dates)-1:eligibility_at_last+=int(prior_ready)
                entry_eligible=prior_ready and tradeable[prior] and price_eligible[prior]
                if units==0. and entry_eligible and tradeable[k]:
                    units=cash/(o[k]*(1+fees));cash=0.;entry_date=date
                value=units*c[k] if observed else 0.
                nav.append(cash+value);exposure.append(value/(cash+value))
            nav=np.asarray(nav);navs.append(nav)
            details.append({'symbol':s,'first_eligible_execution_session':first_ready,'entry_date':entry_date,
               'total_return':float(nav[-1]-1),'cash_interest':interest,'mean_exposure':float(np.mean(exposure))})
        nav=np.mean(navs,axis=0);ret=nav/np.r_[1.,nav[:-1]]-1.;peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
        quality_nav=nav.copy();cash_nav=np.cumprod(1+rf)
        for symbol in replaced:quality_nav+=(cash_nav-np.asarray(navs[symbols.index(symbol)]))/len(symbols)
        quality_rows[label]=metrics(quality_nav,rf)
        with (ROOT/f'{label}_eligibility_matched_quality_buy_hold.npz').open('wb') as f:
            np.savez_compressed(f,dates=np.asarray(dates),nav=quality_nav,rf=rf)
        rows[label]={'period':p[label],'symbols':500,'qualified_at_first_execution_session':eligibility_at_first,
           'qualified_at_last_execution_session':eligibility_at_last,'entered_stocks':sum(x['entry_date'] is not None for x in details),
           'cash_only_stocks':sum(x['entry_date'] is None for x in details),'total_return':float(nav[-1]-1),
           'cagr':math.expm1(math.log(nav[-1])*252/len(nav)),'maximum_drawdown':float(np.max(1-nav/peak)),
           'mean_annualized_excess_rf':float(np.mean(ret-rf)*252),'stock_details':details}
        with (ROOT/f'{label}_eligibility_matched_buy_hold.npz').open('wb') as f:
            np.savez_compressed(f,dates=np.asarray(dates),nav=nav,rf=rf)
    out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'purpose':'Post-registration descriptive benchmark alignment only; never used in candidate selection, ranking or family significance tests.',
       'method':'500 equal initial sleeves. Prior-close100-complete-week readiness, known vendor>=10 proxy and prior/current tradeability permit the first full-allocation entry at the next open. Thereafter hold with no signal exits.10bps entry cost, no forced final sale. Prior-close cash receives the same daily RF proxy.',
       'does_not_match':'Risk-sized strategy exposures, weekly direction filters, signal timings and stop-loss behavior are not matched. This is not a factor-alpha benchmark.',
       'unqualified_through_2025':never_ready,'unqualified_count':len(never_ready),'periods':rows}
    (ROOT/'eligibility_matched_buy_hold.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
    (ROOT/'eligibility_matched_quality_buy_hold.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'replaced_with_RF_cash':replaced,'denominator':len(symbols),'periods':quality_rows,'method':'Same eligibility-matched benchmark with suspect sleeves replaced byRF cash; no candidate reselection.'},indent=2)+'\n', encoding="utf-8")
    return {'eligibility_benchmark':'completed','never_ready':never_ready}


def validate_completed_study(protocol,root,replaced):
    symbols=Path(protocol['universe']).read_text(encoding="utf-8").split()
    if len(symbols)!=500 or len(set(symbols))!=500:raise ValueError('Exactly500 unique frozen symbols required')
    if not replaced or len(replaced)!=len(set(replaced)) or not set(replaced).issubset(symbols):
        raise ValueError('Quality replacement symbols must be unique members of the frozen universe')
    train=json.loads((root/'train/results.json').read_text(encoding="utf-8"))
    validation=json.loads((root/'validation/results.json').read_text(encoding="utf-8"))
    frozen=json.loads((root/'frozen_selection.json').read_text(encoding="utf-8"))
    if frozen['train_results_sha256']!=sha(root/'train/results.json'):raise ValueError('Frozen training signature differs')
    if {k:v for k,v in train['identity'].items() if k!='period'}!={k:v for k,v in validation['identity'].items() if k!='period'}:
        raise ValueError('Train/validation rule or data identities differ')
    if train['identity']['engine_sha256']!=sha('research_layer/multiscale_engine.py'):
        raise ValueError('Current execution engine differs from completed primary run')
    if train['identity']['signals_sha256']!=sha('research_layer/multiscale_signals.py'):
        raise ValueError('Current features differ from completed primary run')
    for label,stage in [('train',train),('validation',validation)]:
        if stage['candidate_count']!=1540 or stage['symbols']!=500 or stage['failed_tasks']!=0:
            raise ValueError('Complete primary1540x500 stage required')
        if stage['nav_matrix_sha256']!=sha(root/label/'nav_matrix.npz'):
            raise ValueError('Primary NAV matrix hash differs')
    return symbols,frozen


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/research/us500_quality_sensitivity.json')
    parser.add_argument('--stage',choices=['all','quality','benchmarks'],default='all')
    args=parser.parse_args(argv)
    q=json.loads(Path(args.config).read_text(encoding="utf-8"))
    p=json.loads(Path(q['strategy_config']).read_text(encoding="utf-8"));root=Path(p['output'])
    symbols,frozen=validate_completed_study(p,root,q['symbols'])
    if sha(q['strategy_config'])!=json.loads((root/'registration.json').read_text(encoding="utf-8"))['config_sha256']:
        raise ValueError('Primary protocol differs from registered study')
    marker=root/'quality_method_config.json';identity={'quality_config_sha256':sha(args.config),'configuration':q}
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))!=identity:raise ValueError('Quality method configuration differs; choose a new run version')
    if not marker.exists():marker.write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
    if args.stage in ['all','quality']:
        print(json.dumps(run_cash_substitution(root,p,symbols,q['symbols'],frozen,sha(args.config))),flush=True)
    if args.stage in ['all','benchmarks']:
        print(json.dumps(run_eligibility_benchmarks(root,p,symbols,q['symbols'])),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
