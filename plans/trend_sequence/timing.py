"""Wait for strict EMA/EFI confirmation on new monthly slots, using one cash account."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from statistics import mean, median
import math
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from feature_layer.user_trend import build_user_trend_features
from plans.target20.run_study import checked, read, sha, write
from plans.target20.analyze import curve_metrics, slices
from research_layer.technical_portfolio import run_technical_portfolio, _rebalance, _price_panel


def pinned(run, relative):
    path = run / relative
    checked(path, read(run / 'artifacts.json')['sha256'][relative])
    return read(path)


def signals_for(frame, calendar):
    f = build_user_trend_features(frame, calendar)
    a, b = f.daily_ema13, f.daily_ema26
    band = f.low.between(np.minimum(a, b), np.maximum(a, b))
    gate = (f.indicators_ready & (f.weekly_ema13_slope > 0) & (f.weekly_ema26_slope > 0)
            & (f.volume > 0) & (f.atr > 0) & band & (f.efi2 < 0)
            & (f.efi2 > f.efi2_previous) & (f.efi2_previous > f.efi2_previous2)
            & (f.efi2.abs() < .5 * f.efi2_previous.abs()))
    if 'is_tradeable' in f:
        gate &= f.is_tradeable
    known = pd.to_datetime(f.loc[gate, 'weekly_available_at'])
    if (known > f.loc[gate, 'daily_available_at']).any():
        raise ValueError('Unpublished weekly feature used')
    return set(f.datetime.loc[gate].dt.strftime('%Y-%m-%d'))


def summary(curve, fills, config):
    out = curve_metrics(curve, config['start'], config['end'])
    annual, _ = slices(curve)
    return {**out, 'annual':{y:m['total_return'] for y,m in annual.items()},
            'fills':len(fills), 'execution_days':len({f['date'] for f in fills}),
            'turnover':math.fsum(f['notional']/f['day_open_nav'] for f in fills),
            'exposure':mean(row['exposure'] for row in curve)}


def compare_cash_ledger(curve, fills, symbols, closes, opens, dates):
    """Rebuild portfolio independently using signed fill notionals and source daily marks."""
    cash = 1.
    holdings = {}
    day_fills = {}
    for f in fills:
        day_fills.setdefault(f['date'], []).append(f)
    error = 0.
    for i, (day, row) in enumerate(zip(dates,curve,strict=True)):
        for f in day_fills.get(day, []):
            j = symbols.index(f['symbol'])
            if f['signal_date'] >= day or f['price'] != opens[i,j]:
                raise ValueError('Fill used unavailable information or incorrect quote')
            cash -= f['delta_units']*f['price'] + f['fee']
            holdings[j] = holdings.get(j,0.) + f['delta_units']
            if cash < -1e-10 or holdings[j] < -1e-10:
                raise ValueError('Cash or holdings unfunded')
        mark = math.fsum(q*closes[i,j] for j,q in holdings.items())
        error = max(error,abs(cash-row['cash']),abs(cash+mark-row['nav']))
    if error > 1e-9:
        raise ValueError('Cash reconstruction failed')
    return error


def simulate(decisions, schedule, dates, symbols, opens, closes, tradable,
             signals, bps, *, all_pass=False):
    """Wrap the monthly allocator, reserving the executable budget for each new position."""
    rate = bps/10000
    units = np.zeros(len(symbols))
    idx = {s:j for j,s in enumerate(symbols)}
    last_buy = [None]*len(symbols)
    cash = 1.
    slots, fills, curve, marks = [], [], [], []
    live_slots = []
    weights = {}
    blocked = []
    cap_count = 0

    def record(symbol, delta, price, signal_date, prev_nav):
        nonlocal cash
        notional = abs(delta)*price
        fee = notional*rate
        cash -= delta*price + fee
        j = idx[symbol]
        if delta < 0 and last_buy[j] == day:
            raise ValueError('T+1 sale')
        units[j] += delta
        if cash < -1e-10 or units[j] < -1e-10:
            raise ValueError('Unfunded transaction')
        # Match the reference allocator's tolerance for machine-scale rounding.
        cash = max(0.,cash)
        units[j] = max(0.,units[j])
        if delta > 0:
            last_buy[j] = day
        fills.append({'date':day,'symbol':symbol,'signal_date':signal_date,'delta_units':float(delta),
                      'price':float(price),'fee':float(fee),'notional':float(notional),'day_open_nav':prev_nav})

    def finish_slots(day, prices, availability, terminal=False):
        for e in live_slots:
            j = idx[e['symbol']]
            if e['status']=='waiting':
                e['status']='unfilled_at_end' if terminal else 'expired'
            e['end_date'] = day
            e['end_is_valuation_only'] = bool(not availability[j])
            e['end_phase'] = 'close' if terminal else 'open'
            e['immediate_gross'] = float(prices[j]/e['origin_open'] - 1)
            e['deferred_gross_on_reserved_budget'] = (
                (float(prices[j])/e['entry_price']-1)*e['used_fraction']
                if e['status']=='filled' else 0.)
        live_slots.clear()

    for i, day in enumerate(dates):
        opening, closing, available = opens[i], closes[i], tradable[i]
        opening_nav = float(cash + units@opening)
        if day in schedule:
            finish_slots(day, opening, available)
            decision = decisions[schedule[day]]
            weights = decision['weights']
            before_units = units.copy()
            # Run the original sells-first, pro-rata budget allocation on a shadow ledger.
            # A new stock that has to wait puts exactly its allocated cash back in reserve.
            _, orders, reb = _rebalance(day, decision['signal_date'], weights, symbols, opening,
                                       available, before_units.copy(), cash, rate, last_buy.copy())
            blocked.extend(dict(b,date=day) for b in reb['blocked'])
            for order in orders:
                symbol = order['symbol']
                if order['side']=='buy' and before_units[idx[symbol]]==0.:
                    e = {'origin':day,'origin_signal':decision['signal_date'],'symbol':symbol,
                         'origin_open':order['price'],'budget':order['notional'],'status':'waiting'}
                    slots.append(e)
                    live_slots.append(e)
                    if not all_pass and decision['signal_date'] not in signals[symbol]:
                        continue
                    e.update(status='filled',entry_date=day,signal_date=decision['signal_date'],
                             entry_price=order['price'],used_fraction=1.)
                record(symbol, order['delta_units'], order['price'], decision['signal_date'], opening_nav)
        elif i > 0:
            trigger = dates[i-1]
            ready = []
            for e in live_slots:
                if e['status']!='waiting' or trigger not in signals[e['symbol']]:
                    continue
                j=idx[e['symbol']]
                e['first_strict_signal']=trigger
                if not available[j]:
                    e['status']='blocked_first_open'
                    continue
                amount=min(e['budget'], weights[e['symbol']]*opening_nav/(1+rate))
                ready.append((e,j,amount))
            # A moved portfolio may have less risk capacity than at the month's opening.
            room=max(0.,sum(weights.values())*opening_nav/(1+rate)-float(units@opening))
            total_requested=math.fsum(n for _,_,n in ready)
            scale=min(1.,cash/(1+rate)/total_requested,room/total_requested) if total_requested>0 else 0.
            for e,j,amount in ready:
                used=amount*scale
                if used < 1e-13*opening_nav:
                    e['status']='risk_budget_exhausted'
                    continue
                if used < e['budget']*(1-1e-10):
                    cap_count+=1
                record(e['symbol'],used/opening[j],opening[j],trigger,opening_nav)
                e.update(status='filled',entry_date=day,signal_date=trigger,
                         entry_price=float(opening[j]),used_fraction=used/e['budget'])
        invested=float(units@closing)
        nav=cash+invested
        if not 0 <= invested/nav <= 1+1e-11:
            raise ValueError('Invalid exposure')
        curve.append({'date':day,'cash':cash,'nav':nav,'exposure':invested/nav})
        marks.append(units.copy())
        reserves=math.fsum(e['budget']*(1+rate) for e in live_slots if e['status']=='waiting')
        if reserves>cash+1e-10:
            raise ValueError('Promises exceed shared cash')
    finish_slots(dates[-1], closes[-1], tradable[-1], True)
    return curve, fills, marks, slots, blocked, cap_count


def control_check(baseline, curve, fills, marks, symbols):
    if len(fills)!=len(baseline['fills']) or len(curve)!=len(baseline['equity_curve']):
        raise ValueError('Unconditional reference counts differ')
    error=0.
    for date, orig, units in zip(curve,baseline['equity_curve'],marks,strict=True):
        if date['date']!=orig['date']:
            raise ValueError('Date mismatch')
        error=max(error,abs(date['cash']-orig['cash']),abs(date['nav']-orig['nav']),
                  max(abs(q-orig['holdings'].get(s,{}).get('units',0.)) for q,s in zip(units,symbols)))
    for actual, ref in zip(fills,baseline['fills'],strict=True):
        if any(ref[k]!=actual[k] for k in ('date','symbol','price','signal_date')):
            raise ValueError('Unconditional fill identity differs')
        error=max(error,abs(actual['delta_units']-ref['delta_units']))
    if error > 1e-9:
        raise ValueError(f'Wait engine fails unconditional controls {error}')
    return error


def descriptive(slots):
    groups = {}
    for key, sample in [('all',slots),('filled',[e for e in slots if e['status']=='filled']),
                        ('unfilled',[e for e in slots if e['status']!='filled'])]:
        group={'n':len(sample)}
        if sample:
            returns=[e['immediate_gross'] for e in sample]
            group.update(positive_immediate=sum(v>1e-12 for v in returns), negative_immediate=sum(v < -1e-12 for v in returns),
                         mean_immediate=mean(returns),median_immediate=median(returns),
                         mean_deferred=mean(e['deferred_gross_on_reserved_budget'] for e in sample),
                         nontradeable_endpoint_marks=sum(e['end_is_valuation_only'] for e in sample))
            if key=='filled':
                group.update(mean_entry_price_vs_immediate=mean(e['entry_price']/e['origin_open']-1 for e in sample),
                             cheaper_than_immediate=sum(e['entry_price']<e['origin_open']-1e-12 for e in sample))
        groups[key]=group
    return groups


def run(monthly, strict):
    freeze=pinned(monthly,'freeze.json');cfg=freeze['configuration'];old=pinned(strict,'freeze.json')
    inputs=freeze['inputs'];frames={}; signals={}; verified=0
    cp=Path(freeze['calendar']['path']);checked(cp,freeze['calendar']['sha256']);cal=pd.read_csv(cp)
    if old['calendar']['sha256']!=freeze['calendar']['sha256']:
        raise ValueError('Different calendars')
    for symbol,meta in sorted(inputs.items()):
        if old['inputs'][symbol]['sha256']!=meta['sha256']:
            raise ValueError('Different price basis')
        p=Path(meta['path']);checked(p,meta['sha256'])
        f=pd.read_csv(p);frames[symbol]=f
        valid=signals_for(f,cal);signals[symbol]=valid
        prior=pinned(strict,f'cases/and_band_next_open_ohlc_10bps/{symbol.replace(".","_")}.json')
        archive_dates={d['date'] for d in prior['strategy']['signals']}
        if {day for day in valid if cfg['start']<=day<=cfg['end']}!=archive_dates:
            raise ValueError(f'Raw strict AND daily dates failed {symbol}')
        verified+=len(archive_dates)
    sessions=pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_trading_day.astype(str).str.lower().isin(['true','1']),'date']))
    days=sessions[(sessions>=cfg['start'])&(sessions<=cfg['end'])]
    symbols, opens, closes, allowed=_price_panel(frames,days)
    dates=list(days.strftime('%Y-%m-%d'));result={}
    for policy in ['defensive5','momentum10']:
        decisions=pinned(monthly,f'decisions/{policy}.json')
        schedule={sessions[sessions.get_loc(d['signal_date'])+1].date().isoformat():i for i,d in enumerate(decisions)}
        if len(schedule)!=48 or any(day not in dates for day in schedule):
            raise ValueError('Unexpected monthly schedule')
        out={}
        for cost in [0,10,20]:
            ref=pinned(monthly,f'cases/{policy}_{cost}bps.json')
            rebuilt=run_technical_portfolio(frames,cal,start=cfg['start'],end=cfg['end'],decisions=decisions,cost_bps=cost)
            if ref!=rebuilt:
                raise ValueError('Archived control changed')
            common=(decisions,schedule,dates,symbols,opens,closes,allowed,signals,cost)
            origin,immediate_fills,original_units,*_=simulate(*common,all_pass=True)
            error=control_check(ref,origin,immediate_fills,original_units,symbols)
            delayed,dfills,_,slots,blocked,cap_count=simulate(*common)
            audit=compare_cash_ledger(delayed,dfills,symbols,closes,opens,dates)
            if any(e['status']=='filled' and e['signal_date'] not in signals[e['symbol']] for e in slots):
                raise ValueError('Unqualified slot filled')
            m={'immediate':summary(origin,immediate_fills,cfg),'wait':summary(delayed,dfills,cfg),
               'funnel':dict(Counter(e['status'] for e in slots)),'paired_slots':descriptive(slots),
               'source_exact':True,'all_pass_error':error,'ledger_error':audit,'reduced_budget_fills':cap_count,
               'blocked_monthly_actions':len(blocked),'raw_weekly_filtered_signals':verified}
            if cost==10:m.update(slots=slots,wait_fills=dfills,wait_curve=delayed)
            out[str(cost)]=m
        result[policy]=out
    return {'cases':result,'common_symbols':len(symbols),'common_dates':len(dates),'raw_strict_signals_verified':verified,
            'source_sha256':sha(Path(__file__)),'monthly_manifest_sha256':sha(monthly/'artifacts.json'),
            'and_manifest_sha256':sha(strict/'artifacts.json')}


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--monthly',required=True,type=Path);parser.add_argument('--strict',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();report=run(a.monthly,a.strict);write(a.output,report)
    for p,cases in report['cases'].items():
        r=cases['10'];print(p,r['funnel'],r['wait']['cagr_calendar'],r['wait']['maximum_drawdown'])
