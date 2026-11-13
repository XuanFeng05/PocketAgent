"""Registered finite-family Elder research; frozen chronological selection.

All generated prices, NAVs and results stay in runtime_layer. This CLI is
independent of the existing A-share strategy and Dashboard job protocol.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from research_layer.elder_engine import ElderConfig, build_features, run_elder
from research_layer.elder_analysis import metrics, fetch_french_rf, assess_family


def now(): return datetime.now(timezone.utc).isoformat()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v, dict): return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    if isinstance(v, np.ndarray): return plain(v.tolist())
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
    return v

def save(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def candidate_id(c):return f'{c.weekly_filter}_efi{c.force_span}_{c.exit_mode}_{c.stop_mode}'

def configs(protocol):
    keys=list(protocol['grid'])
    result=[ElderConfig(**protocol['fixed'],**dict(zip(keys,vs))) for vs in product(*(protocol['grid'][k] for k in keys))]
    assert len(result)==protocol['candidate_count']==72
    assert len({candidate_id(c) for c in result})==len(result)
    return result

def verify_inputs(protocol,out):
    root=Path(protocol['data_root']);symbols=Path(protocol['universe']).read_text(encoding='utf-8').split()
    assert len(symbols)==100 and len(set(symbols))==100
    calendar=root/'calendar.csv';assert calendar.exists()
    files={}; rows={}
    for symbol in symbols:
        p=root/'daily'/f'{symbol}.csv';mp=root/'metadata'/f'{symbol}.json'
        if not p.exists() or not mp.exists():raise ValueError(f'Missing restored data for {symbol}')
        m=read(mp)
        if m['status']!='success' or m['start']!='2020-01-01' or m['end']!='2026-07-12':raise ValueError(f'Unverified request for {symbol}')
        sha=digest(p)
        if sha!=m['daily_sha256']:raise ValueError(f'Price hash changed for {symbol}')
        raw=root/'raw'/f'{symbol}.json'
        if digest(raw)!=m['raw_sha256']:raise ValueError(f'Raw response hash changed for {symbol}')
        files[symbol]={'daily_sha256':sha,'raw_sha256':m['raw_sha256'],'metadata_sha256':digest(mp)}
        rows[symbol]=m['quality']['rows']
    snapshot={'market':'US','universe':symbols,'universe_sha256':digest(protocol['universe']),
              'calendar_sha256':digest(calendar),'files':files,'rows':rows,'old_snapshot_restored':False}
    p=out/'input_snapshot.json'
    if p.exists() and read(p)!=snapshot:raise ValueError('Research input snapshot changed; use a new output/version')
    if not p.exists():save(p,snapshot)
    return symbols,root,p

def prepare(protocol,out):
    symbols,root,snapshot=verify_inputs(protocol,out)
    calendar=pd.read_csv(root/'calendar.csv')
    features={};bars={};base_dates=None
    for i,s in enumerate(symbols):
        b=pd.read_csv(root/'daily'/f'{s}.csv');bars[s]=b
        for span in protocol['grid']['force_span']:
            features[(s,span)]=build_features(b,calendar,span,protocol['fixed']['warmup_weeks'])
        if (i+1)%20==0:print(json.dumps({'phase':'features','complete':i+1,'total':100}),flush=True)
    return symbols,features,bars,digest(snapshot)

def summarize(config,symbols,features,start,end,cash_rates=None):
    navs=[];exposures=[];dates=None;stock=[];closed=[];open_count=0;fees=0.;interest=0.;ambiguous=0
    for symbol in symbols:
        kw={} if cash_rates is None else {'cash_daily_rates':cash_rates}
        r=run_elder(features[(symbol,config.force_span)],config,start,end,False,**kw)
        if dates is None:dates=r['dates']
        if dates!=r['dates']:raise ValueError('Nonaligned stock evaluation dates; cannot silently aggregate')
        navs.append(r['nav']);exposures.append(r['exposure'])
        stock.append({'symbol':symbol,**r['metrics']})
        closed.extend({**t,'symbol':symbol} for t in r['trades'] if t['status']=='closed')
        open_count+=r['metrics']['open_trades'];fees+=r['metrics']['total_fees']
        interest+=r['metrics'].get('cash_interest',0.)
        ambiguous+=r['diagnostics']['ambiguous_entries']
    n=np.asarray(navs).T;ex=np.asarray(exposures).T;nav=n.mean(axis=1)
    win=[t for t in closed if t['cash_pnl']>0];loss=[t for t in closed if t['cash_pnl']<0]
    gp=sum(t['cash_pnl'] for t in win);gl=-sum(t['cash_pnl'] for t in loss)
    returns=np.array([t['net_return'] for t in closed])
    pnl=np.array([t['cash_pnl'] for t in closed]);nclose=len(closed)
    result={'id':candidate_id(config),'config':config.to_dict(),'dates':dates,'nav':nav,
            'metrics':metrics(nav),'trades':{'closed':nclose,'open':open_count,'wins':len(win),'losses':len(loss),
             'breakeven':nclose-len(win)-len(loss),'win_rate':len(win)/nclose if nclose else None,
             'nonloss_rate':(nclose-len(loss))/nclose if nclose else None,
             'profit_factor':gp/gl if gl else None,'gross_profit_cash':gp,'gross_loss_cash':gl,
             'mean_net_trade_return':float(returns.mean()) if nclose else None,
             'average_win_return':float(np.mean([t['net_return'] for t in win])) if win else None,
             'average_loss_return':float(np.mean([t['net_return'] for t in loss])) if loss else None,
             'mean_R_multiple':float(np.mean([t['r_multiple'] for t in closed])) if nclose else None,
             'realized_pnl_per_initial_capital':float(pnl.sum()/len(symbols)),
             'ambiguous_entries':ambiguous,'total_fees_per_initial_capital':fees/len(symbols)},
            'mean_capital_weighted_exposure':float(np.mean(np.sum(n*ex,axis=1)/n.sum(axis=1))),
            'cash_interest_per_initial_capital':interest/len(symbols),
            'stock_metrics':stock,'profitable_stocks':sum(s['total_return']>0 for s in stock),
            'median_stock_return':float(np.median([s['total_return'] for s in stock]))}
    assert np.isclose(nav[-1]-1,np.mean([s['total_return'] for s in stock]))
    return result

def _source_identity():
    return {'engine_sha256':digest('research_layer/elder_engine.py'),
            'analysis_sha256':digest('research_layer/elder_analysis.py'),
            'runner_sha256':digest(__file__),
            'normalizer_sha256':digest('research_layer/technical.py')}


def _identity_matches(recorded,current):
    # Legacy checkpoints predate the runner/normalizer fields. Verify exactly
    # their recorded schema, never add present-day hashes to old signatures.
    legacy=set(current)-{'runner_sha256','normalizer_sha256'}
    return (set(recorded) in (legacy,set(current))
            and all(recorded[key]==current[key] for key in recorded))


def _verify_completed_stage(label,cs,snapshot_sha,config_sha,out,period=None):
    """Read legacy checkpoint identities without rewriting frozen artifacts."""
    summary=read(out/label/'results.json')
    if (summary['stage']!=label or summary['snapshot_sha256']!=snapshot_sha
            or summary['config_sha256']!=config_sha or summary['failed_tasks']!=0
            or (period is not None and summary['period']!=period)):
        raise ValueError(f'Completed {label} configuration/data/period identity mismatch')
    expected_ids=[candidate_id(c) for c in cs]
    if [r['id'] for r in summary['candidates']]!=expected_ids:
        raise ValueError(f'Completed {label} candidate family differs')
    source=_source_identity()
    for config,result in zip(cs,summary['candidates']):
        path=out/label/(candidate_id(config)+'.json');checkpoint=read(path)
        expected={'snapshot_sha256':snapshot_sha,'config_sha256':config_sha,
                  **source,'period':summary['period'],'settings':config.to_dict()}
        if not _identity_matches(checkpoint['identity'],expected):
            raise ValueError(f'Completed {label} source/configuration/data identity mismatch: {path}')
        if checkpoint['result']!=result or result['config']!=config.to_dict():
            raise ValueError(f'Completed {label} checkpoint differs from summary: {path}')
    return summary


def require_frozen_context(cs,snapshot_sha,config_sha,out,*,training_period=None,validation_period=None):
    """Reject changed rules/data before validation, sensitivities or analysis.

    Check every candidate against the frozen summary and current context.
    New checkpoints include the CLI and technical normalizer in addition to
    the engine and metric source. Older artifacts lack those two hashes, so
    checking them does not retroactively certify unrecorded dependencies.
    """
    frozen=read(out/'frozen_selection.json')
    if digest(out/'train/results.json')!=frozen['train_results_sha256']:
        raise ValueError('Frozen training signature changed')
    train=_verify_completed_stage('train',cs,snapshot_sha,config_sha,out,training_period)
    if train['finished_utc']>frozen['selected_utc']:
        raise ValueError('Frozen selection must follow completed training')
    ids={candidate_id(c) for c in cs}
    if len(set(frozen['selected_ids']))!=len(frozen['selected_ids']) or not set(frozen['selected_ids']).issubset(ids):
        raise ValueError('Frozen selected IDs differ from completed candidate family')
    validation=None
    if validation_period is not None:
        validation=_verify_completed_stage('validation',cs,snapshot_sha,config_sha,out,validation_period)
        if frozen['selected_utc']>validation['started_utc']:
            raise ValueError('Frozen selection must precede validation')
    return frozen,train,validation


def run_stage(label,period,cs,symbols,features,bars,snapshot_sha,config_sha,out):
    if label=='validation':
        require_frozen_context(cs,snapshot_sha,config_sha,out)
    stage=out/label;stage.mkdir(parents=True,exist_ok=True);results=[];started=now()
    source=_source_identity()
    for i,c in enumerate(cs):
        path=stage/(candidate_id(c)+'.json');identity={'snapshot_sha256':snapshot_sha,'config_sha256':config_sha,
               **source,'period':period,'settings':c.to_dict()}
        if path.exists():
            saved=read(path)
            if not _identity_matches(saved['identity'],identity):raise ValueError(f'Checkpoint identity mismatch: {path}')
            result=saved['result']
        else:
            try:result=summarize(c,symbols,features,*period)
            except Exception as exc:
                save(stage/'failure.json',{'case':candidate_id(c),'error':str(exc),'type':type(exc).__name__,'at':now()})
                raise
            save(path,{'identity':identity,'result':result})
        results.append(result)
        print(json.dumps({'phase':label,'cases':i+1,'total_cases':len(cs),'completed_stock_tasks':(i+1)*len(symbols)},ensure_ascii=False),flush=True)
    dates=results[0]['dates'];bh=[];cost=cs[0].cost_bps/10000
    for s in symbols:
        b=bars[s].set_index('datetime').loc[dates]
        bh.append((b.close/(float(b.open.iloc[0])*(1+cost))).to_numpy())
    benchmark_nav=np.asarray(bh).mean(axis=0)
    final={'stage':label,'started_utc':started,'finished_utc':now(),'period':period,'dates':dates,
           'completed_tasks':len(cs)*len(symbols),'failed_tasks':0,'snapshot_sha256':snapshot_sha,
           'config_sha256':config_sha,'aggregation':'Equal initial segregated sleeves; no rebalancing/shared cash; idle cash interest0.',
           'benchmark':{'kind':'same100 full-allocation buy-and-hold; entry fee included; unsold at end','nav':benchmark_nav,'metrics':metrics(benchmark_nav)},
           'candidates':results}
    summary_path=stage/'results.json'
    if summary_path.exists():
        existing=read(summary_path)
        for key in ['stage','period','dates','completed_tasks','failed_tasks','snapshot_sha256','config_sha256','benchmark','candidates']:
            if existing[key]!=plain(final[key]):raise ValueError(f'Stage summary changed during resume: {label} {key}')
        return existing  # Preserve timestamps and frozen training hash on resume.
    save(summary_path,final)
    return final

def select_candidates(training,protocol,out):
    rows=training['candidates'];eligible=[r for r in rows if r['trades']['closed']>=protocol['selection']['minimum_closed_trades']]
    def balanced(r):return r['metrics']['cagr']/max(r['metrics']['maximum_drawdown'],.01)
    ranking=sorted(eligible,key=lambda r:(-balanced(r),-(r['trades']['profit_factor'] or 0),r['id']))
    viable=[r for r in eligible if r['metrics']['total_return']>0 and (r['trades']['profit_factor'] or 0)>1]
    win=sorted(viable,key=lambda r:(-r['trades']['win_rate'],-r['metrics']['total_return'],r['id']))
    high_return=sorted(eligible,key=lambda r:(-r['metrics']['total_return'],r['id']))
    baseline_ids={b['id']:candidate_id(ElderConfig(**protocol['fixed'],**{k:v for k,v in b.items() if k!='id'})) for b in protocol['baselines']}
    chosen={'balanced_top5':[r['id'] for r in ranking[:5]],'high_win':win[0]['id'] if win else None,
            'high_return':high_return[0]['id'] if high_return else None,'baselines':baseline_ids}
    ids=sorted(set(chosen['balanced_top5']+list(baseline_ids.values())+[v for k,v in chosen.items() if k in ['high_win','high_return'] and v]))
    pareto=[]
    for a in eligible:
        av=(a['metrics']['cagr'],a['trades']['win_rate'],-a['metrics']['maximum_drawdown'])
        dominated=False
        for b in eligible:
            bv=(b['metrics']['cagr'],b['trades']['win_rate'],-b['metrics']['maximum_drawdown'])
            if all(x>=y for x,y in zip(bv,av)) and any(x>y for x,y in zip(bv,av)):dominated=True;break
        if not dominated:pareto.append(a['id'])
    frozen={'selected_utc':now(),'train_results_sha256':digest(out/'train/results.json'),
            'selection_information':'2022-2023 only; all years previously viewed, not pristine holdout',
            'selected':chosen,'selected_ids':ids,'training_pareto_ids':pareto,
            'ranking':[{'id':r['id'],'training_calmar_floor1pct':balanced(r)} for r in ranking]}
    p=out/'frozen_selection.json'
    if p.exists():
        old=read(p)
        if any(old[k]!=frozen[k] for k in ['train_results_sha256','selected','selected_ids','training_pareto_ids','ranking']):raise ValueError('Refusing to overwrite prior selection with changed result')
        return old
    save(p,frozen);return frozen

def assess_selected_cash(protocol,out,frozen,factors,dates):
    path=out/'selected_cash_interest.json'
    if not path.exists():return None
    sensitivity=read(path)
    if sensitivity['selection_sha256']!=digest(out/'frozen_selection.json'):
        raise ValueError('Cash sensitivity selection signature changed')
    rows=sensitivity['rows'];ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(frozen['selected_ids']):
        raise ValueError('Cash sensitivity must contain exactly the frozen choices')
    if any(r['dates']!=dates for r in rows):raise ValueError('Cash sensitivity dates differ')
    result=assess_family(np.column_stack([r['nav'] for r in rows]),dates,factors['RF'],ids,ids,
        B=protocol['validation']['bootstrap_repetitions'],block=protocol['validation']['bootstrap_block_sessions'],seed=protocol['validation']['seed'])
    result['scope']=f'{len(ids)} training-frozen choices with registered cash-interest sensitivity; not all72 financed variants, and not new selection.'
    result['limitations']=[s for s in result['limitations'] if not s.startswith('Idle cash earns zero')]
    result['limitations'].append('Prior-close idle cash earns the French daily RF proxy; this is a financing approximation, not an actual broker cash-sweep quote or factor alpha.')
    result['limitations'].append('PSR/DSR trial-count scenarios remain72/200; selected-case Sharpe dispersion is conditional and cannot represent the full financed search family.')
    save(out/'selected_cash_family_statistics.json',result)
    return result


def analyze_family(protocol,out):
    _,_,snapshot=verify_inputs(protocol,out)
    frozen,train,validation=require_frozen_context(configs(protocol),digest(snapshot),
        read(out/'registration.json')['config_sha256'],out,training_period=protocol['train'],
        validation_period=protocol['validation']['period'])
    factors=pd.read_csv(out/'factors/ff5_daily.csv',index_col=0)
    ids=[r['id'] for r in validation['candidates']];dates=validation['dates']
    matrix=np.column_stack([r['nav'] for r in validation['candidates']])
    assessment=assess_family(matrix,dates,factors['RF'],frozen['selected_ids'],ids,
          B=protocol['validation']['bootstrap_repetitions'],block=protocol['validation']['bootstrap_block_sessions'],seed=protocol['validation']['seed'])
    save(out/'family_statistics.json',assessment)
    assess_selected_cash(protocol,out,frozen,factors,dates)
    records=[];train_by={r['id']:r for r in train['candidates']}
    for r in validation['candidates']:
        t=train_by[r['id']]
        records.append({'id':r['id'],'selected_from_training':r['id'] in frozen['selected_ids'],
          'train_metrics':metrics(t['nav'],factors.loc[t['dates'],'RF'].to_numpy()),'train_trades':t['trades'],
          'validation_metrics':metrics(r['nav'],factors.loc[dates,'RF'].to_numpy()),'validation_trades':r['trades'],
          'validation_profitable_stocks':r['profitable_stocks'],'validation_exposure':r['mean_capital_weighted_exposure']})
    result={'selection':frozen,'candidates':records,'train_period':train['dates'][::len(train['dates'])-1],
       'validation_period':dates[::len(dates)-1],
       'validation_buy_hold':metrics(validation['benchmark']['nav'],factors.loc[dates,'RF'].to_numpy()),
       'validation_RF':metrics(np.cumprod(1+factors.loc[dates,'RF'].to_numpy())),
       'not_untouched':'Retrospective chronological comparison on previously viewed years, not prospective proof',
       'cash_interest':'Core family zero; selected sensitivity in selected_cash_interest.json',
       'completed_core_stock_tasks':train['completed_tasks']+validation['completed_tasks'],'failed_core_stock_tasks':0}
    save(out/'research_comparison.json',result)
    flat=[]
    for r in records:
        flat.append({'id':r['id'],'selected_from_training':r['selected_from_training'],
                     **{f'train_{k}':v for k,v in r['train_metrics'].items()},
                     **{f'validation_{k}':v for k,v in r['validation_metrics'].items()},
                     **{f'train_trade_{k}':v for k,v in r['train_trades'].items()},
                     **{f'validation_trade_{k}':v for k,v in r['validation_trades'].items()}})
    pd.DataFrame(flat).to_csv(out/'candidate_comparison.csv',index=False)
    print(json.dumps({'phase':'statistics','status':'completed','candidates':len(records)}),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/research/elder_search_v1.json')
    parser.add_argument('--stage',choices=['train','validate','costs','analyze','all'],default='all')
    args=parser.parse_args(argv);protocol=read(args.config);out=Path(protocol['output']);out.mkdir(parents=True,exist_ok=True)
    config_sha=digest(args.config)
    registered=read(out/'registration.json') if (out/'registration.json').exists() else None
    if registered is None:save(out/'registration.json',{'registered_utc':now(),'config_sha256':config_sha,'candidate_count':72,'returns_observed_yet':False})
    elif registered['config_sha256']!=config_sha:raise ValueError('Configuration differs from preregistration')
    rf_status=fetch_french_rf(out/'factors')
    if rf_status['status']!='success':raise ValueError(f'Official RF retrieval failed: {rf_status}')
    if args.stage=='analyze':
        analyze_family(protocol,out);return 0
    cs=configs(protocol);symbols,features,bars,snapshot=prepare(protocol,out)
    if args.stage in ['train','all']:
        train=run_stage('train',protocol['train'],cs,symbols,features,bars,snapshot,config_sha,out)
        frozen=select_candidates(train,protocol,out)
    else:
        frozen,_,_=require_frozen_context(cs,snapshot,config_sha,out,training_period=protocol['train'])
    if args.stage in ['validate','all']:
        run_stage('validation',protocol['validation']['period'],cs,symbols,features,bars,snapshot,config_sha,out)
    if args.stage in ['costs','all']:
        frozen,_,_=require_frozen_context(cs,snapshot,config_sha,out,training_period=protocol['train'],
            validation_period=protocol['validation']['period'])
        selected=[c for c in cs if candidate_id(c) in frozen['selected_ids']]
        if not (out/'validation/results.json').exists():raise ValueError('Validation needed before selected sensitivity analysis')
        base={r['id']:r for r in read(out/'validation/results.json')['candidates']};rows=[]
        for c in selected:
            for cost in protocol['validation']['selected_cost_scenarios_bps']:
                if cost==protocol['fixed']['cost_bps']:r=base[candidate_id(c)]
                else:r=summarize(replace(c,cost_bps=cost),symbols,features,*protocol['validation']['period'])
                rows.append({'id':candidate_id(c),'cost_bps':cost,'metrics':r['metrics'],'trades':r['trades']})
            print(json.dumps({'phase':'selected_costs','candidate':candidate_id(c)}),flush=True)
        save(out/'selected_costs.json',{'selection_sha256':digest(out/'frozen_selection.json'),'rows':rows})
        factors=pd.read_csv(out/'factors/ff5_daily.csv',index_col=0)
        cash_rows=[]
        for c in selected:
            dates=base[candidate_id(c)]['dates']; rates=factors.loc[dates,'RF'].to_numpy()
            r=summarize(c,symbols,features,*protocol['validation']['period'],cash_rates=rates)
            r['metrics']=metrics(r['nav'],rates)
            cash_rows.append(r)
            print(json.dumps({'phase':'selected_cash_interest','candidate':candidate_id(c)}),flush=True)
        save(out/'selected_cash_interest.json',{'assumption':'Daily RF proxy credited to prior close cash before orders; not broker sweep returns','selection_sha256':digest(out/'frozen_selection.json'),'rows':cash_rows})
    if args.stage=='all':analyze_family(protocol,out)
    print(json.dumps({'status':'completed','stage':args.stage,'output':str(out)}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
