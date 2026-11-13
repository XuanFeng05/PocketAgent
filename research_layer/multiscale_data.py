"""Expand the unchanged US100 into an auditable, liquidity-screened US500.

Source holdings and all downloaded observations remain under runtime_layer.
Selection uses provider availability/price/liquidity, never strategy outcomes.
The November 2026 holdings snapshot is not point-in-time historical membership.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import shutil
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from research_layer.elder_data import (
    NY, audit_snapshot, load_symbols, make_calendar, restore_symbol, save_json, sha256,
)

HOLDINGS_URL = 'https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv'
MARKET_HISTORY_CONTEXT = [
    {'symbol':'SW','verified_primary_US_listing_date':'2024-07-08',
     'source':'https://www.smurfitwestrock.com/newsroom/press-releases/2024/smurfit-westrock-makes-its-debut-in-new-york-and-london',
     'evidence_summary':'Issuer says the merger closed2024-07-05 and newSW trading onNYSE began2024-07-08. EarlierYahoo observations are not certified same-security NYSE history.'},
    {'symbol':'FERG','verified_first_US_NYSE_listing_date':'2021-03-08','verified_primary_NYSE_listing_date':'2022-05-12',
     'source':'https://www.corporate.ferguson.com/pressroom/news-releases/news-details/2022/Fergusons-Primary-Listing-Now-on-the-NYSE-As-It-Completes-Transfer-of-Listing-Category-05-12-2022/default.aspx',
     'evidence_summary':'Issuer confirms March2021 additionalNYSE listing and May2022 transfer of primarylisting. EarlierYahoo volumes are not certified NYSE history.'},
    {'symbol':'TLN','verified_public_OTC_Pink_start':'2023-06-23','verified_OTCQX_start':'2023-07-24','announced_Nasdaq_start':'2024-07-10',
     'sources':['https://www.sec.gov/Archives/edgar/data/1622536/000162827924000077/filename1.htm','https://www.sec.gov/Archives/edgar/data/1622536/000162828024031490/talen424b3.htm'],
     'evidence_summary':'SEC filings identify OTCmarket transitions and Nasdaqapproval. Yahoo begins2023-06-02 before the stated publicOTCstart; initial-history provenance needs reconciliation.'},
]


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_holdings(body: bytes):
    lines = body.decode('utf-8-sig').splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('Ticker,')), None)
    if header is None:
        raise ValueError('Holdings endpoint did not return the documented CSV header.')
    date_line = next((s for s in lines[:header] if s.startswith('Fund Holdings as of,')), None)
    if date_line is None:
        raise ValueError('Missing holdings observation date.')
    date = pd.to_datetime(date_line.split(',', 1)[1].strip('"')).date().isoformat()
    frame = pd.read_csv(io.StringIO('\n'.join(lines[header:])))
    required = {'Ticker', 'Name', 'Type', 'Asset Class', 'Market Value', 'Price', 'Location', 'Exchange', 'Currency'}
    if not required.issubset(frame.columns):
        raise ValueError('Holdings CSV columns changed.')
    rows, exclusions, seen = [], [], set()
    for _, r in frame.iterrows():
        if r['Asset Class'] != 'Equity' or r['Type'] != 'EQUITY':
            continue
        original = str(r['Ticker']).strip()
        symbol = original.replace(' ', '-').replace('.', '-')
        reason = None
        if not re.fullmatch(r'[A-Z][A-Z0-9-]{0,11}', symbol): reason = 'unsupported_ticker'
        elif r['Currency'] != 'USD' or r['Location'] != 'United States': reason = 'non_US_USD_equity'
        elif r['Exchange'] not in {'NYSE', 'NASDAQ', 'Cboe BZX'}: reason = 'non_primary_US_exchange'
        price = float(str(r['Price']).replace(',', ''))
        value = float(str(r['Market Value']).replace(',', ''))
        if reason is None and price < 10: reason = 'holdings_date_price_below_10'
        if reason is None and symbol in seen: reason = 'duplicate_normalized_symbol'
        row = {'symbol': symbol, 'provider_ticker': original, 'name': str(r['Name']),
               'sector': str(r.get('Sector', '')), 'holdings_price': price,
               'holding_market_value': value, 'holdings_as_of': date}
        if reason:
            exclusions.append({**row, 'reason': reason})
        else:
            seen.add(symbol); rows.append(row)
    rows.sort(key=lambda r: (-r['holding_market_value'], r['symbol']))
    if len(rows) < 500:
        raise ValueError(f'Fewer than500 valid eligible holdings: {len(rows)}')
    return date, rows, exclusions


def fetch_holdings(report_root: Path, timeout: float):
    path = report_root / 'iwb_holdings_latest.csv'
    if not path.exists():
        with urlopen(Request(HOLDINGS_URL, headers={'User-Agent': 'Mozilla/5.0'}), timeout=timeout) as response:
            body = response.read()
        parse_holdings(body)  # Do not cache an HTML/error response as a valid catalogue.
        path.write_bytes(body)
    date, rows, excluded = parse_holdings(path.read_bytes())
    provenance = {'provider': 'BlackRock iShares', 'fund': 'IWB', 'url': HOLDINGS_URL,
                  'holdings_as_of': date, 'snapshot_sha256': sha256(path),
                  'snapshot_recorded_utc': now(), 'eligible_catalogue': len(rows),
                  'preselection_exclusions': excluded,
                  'selection': 'Preserve original100; add first400 qualifying equities in descending IWB holding market value, tie ticker. No strategy returns used.',
                  'bias': 'Current survivor/provider-available sample. November2026 membership and endpoint liquidity are future information relative to2022-2025. Not point-in-time universe.'}
    meta = report_root / 'holdings_source.json'
    if not meta.exists(): save_json(meta, provenance)
    return rows


def reuse_original(config, root: Path):
    original = load_symbols(Path(config['preserve_universe_file']), 100)
    old = Path(config['reuse_data_root'])
    for symbol in original:
        m = json.loads((old / 'metadata' / f'{symbol}.json').read_text(encoding="utf-8"))
        if m['status'] != 'success' or m['start'] != config['start'] or m['end'] != config['end']:
            raise ValueError(f'Original100 metadata invalid: {symbol}')
        for folder, suffix, key in [('raw', '.json', 'raw_sha256'), ('daily', '.csv', 'daily_sha256')]:
            src = old / folder / (symbol + suffix)
            if sha256(src) != m[key]: raise ValueError(f'Original100 hash mismatch: {symbol}')
        for folder, suffix in [('raw', '.json'), ('daily', '.csv'), ('metadata', '.json')]:
            src = old / folder / (symbol + suffix); dst = root / folder / (symbol + suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and sha256(dst) != sha256(src): raise ValueError(f'Reused cache differs: {symbol}')
            if not dst.exists(): shutil.copyfile(src, dst)
    return original


def eligibility(frame, config):
    reasons = []
    last_price = float(frame.iloc[-1]['vendor_close'])
    median_dollars = float((frame.tail(60).vendor_close * frame.tail(60).volume).median())
    if last_price < config['minimum_price_USD']: reasons.append('requested_end_price_below_10')
    if len(frame) < 60: reasons.append('fewer_than_60_daily_observations')
    if median_dollars < config['minimum_last60_median_vendor_dollars']: reasons.append('last60_median_dollar_proxy_below_25million')
    return {'eligible': not reasons, 'exclusion_reasons': reasons,
            'last_requested_session_vendor_close': last_price,
            'last60_median_vendor_close_times_volume': median_dollars}


def download_candidate(row, config, root, calendar):
    symbol = row['symbol']; attempts = []
    for attempt in range(1, config['maximum_attempts'] + 1):
        try:
            m = restore_symbol(symbol, config['start'], config['end'], root, calendar, config['network_timeout_seconds'])
            frame = pd.read_csv(root / 'daily' / f'{symbol}.csv')
            screen = eligibility(frame, config)
            return {**row, 'status': 'success', **screen, 'quality': m['quality'],
                    'raw_sha256': m['raw_sha256'], 'daily_sha256': m['daily_sha256'],
                    'download_started_utc': m['download_started_utc'], 'download_finished_utc': m['download_finished_utc'],
                    'reused_validated_cache': m.get('reused_validated_cache', False), 'prior_failed_attempts': attempts}
        except Exception as exc:
            attempts.append({'attempt': attempt, 'at_utc': now(), 'type': type(exc).__name__, 'reason': str(exc)})
            # Parsing/calendar faults are deterministic; preserve, report, and do not repeatedly download them.
            if isinstance(exc, ValueError): break
            if attempt < config['maximum_attempts']: time.sleep(min(2 ** attempt, 8))
        finally:
            time.sleep(config['sleep_seconds_per_worker'])
    return {**row, 'status': 'failed', 'eligible': False, 'attempts': attempts,
            'exclusion_reasons': ['source_or_quality_failure'], 'last_error': attempts[-1]}


def expand(config, config_path):
    root = Path(config['data_root']); root.mkdir(parents=True, exist_ok=True)
    report = Path(config['report_root']); report.mkdir(parents=True, exist_ok=True)
    original = reuse_original(config, root)
    calendar = make_calendar(config['start'], config['end'])
    calpath = root / 'calendar.csv'
    if calpath.exists(): pd.testing.assert_frame_equal(pd.read_csv(calpath, keep_default_na=False), calendar, check_dtype=False)
    else: calendar.to_csv(calpath, index=False)
    catalogue = [r for r in fetch_holdings(report, config['network_timeout_seconds']) if r['symbol'] not in original]
    selected = list(original); outcomes = []; started = now(); cursor = 0
    # First8 exercise the real data path at2workers; subsequent waves may use4.
    while len(selected) < config['fixed_universe_count'] and cursor < len(catalogue):
        count = min(8 if cursor == 0 else 40, config['fixed_universe_count'] - len(selected))
        batch = catalogue[cursor:cursor + count]; cursor += len(batch)
        workers = min(2 if cursor <= 8 else config['workers'], len(batch))
        completed = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_candidate, row, config, root, calendar): row for row in batch}
            for future in as_completed(futures):
                row = future.result(); completed[row['symbol']] = row
                with (report / 'download_events.jsonl').open('a', encoding="utf-8") as handle: handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(json.dumps({'symbol': row['symbol'], 'status': row['status'], 'eligible': row['eligible'], 'attempted': len(outcomes) + len(completed), 'selected_before_wave': len(selected)}, ensure_ascii=False), flush=True)
        for item in batch:
            row = completed[item['symbol']]; outcomes.append(row)
            if row['eligible']: selected.append(row['symbol'])
        save_json(report / 'expansion_progress.json', {'started_utc': started, 'updated_utc': now(), 'selected_count': len(selected), 'attempted_new': len(outcomes), 'selected_symbols': selected, 'outcomes': outcomes})
        # A rate-limit wave is not bypassed by rotating providers or proxies.
        if any('429' in str(r.get('last_error', '')) for r in completed.values()):
            raise RuntimeError('Yahoo rate limit detected; preserve checkpoint and retry later.')
    if len(selected) != config['fixed_universe_count'] or len(set(selected)) != len(selected):
        raise ValueError(f'Could not obtain exactly500: {len(selected)}; explicit outcomes retained.')
    target = Path(config['universe_file']); text = '\n'.join(selected) + '\n'
    if target.exists() and target.read_text(encoding="utf-8") != text: raise ValueError('Refusing to change frozen US500 universe.')
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(text, encoding="utf-8")
    result = {'started_utc': started, 'finished_utc': now(), 'original_preserved': 100, 'added': 400,
              'count': 500, 'universe_sha256': sha256(target), 'config_sha256': sha256(config_path),
              'attempted_new': len(outcomes), 'excluded_new': [r for r in outcomes if not r['eligible']],
              'outcomes': outcomes, 'reused_original_snapshot': str(config['reuse_data_root']),
              'snapshot_note': 'Each raw/CSV/metadata retains actual retrieval time and hash. Original100 is reused without redownload; new400 has a later retrieval vintage.'}
    save_json(report / 'expansion_summary.json', result)
    return result


def download_frozen(config, only=None):
    """Reproduce an already frozen list without reselecting from live holdings."""
    symbols = load_symbols(Path(config['universe_file']), config['fixed_universe_count'])
    original = load_symbols(Path(config['preserve_universe_file']), 100)
    if symbols[:100] != original: raise ValueError('Frozen US500 must preserve the original100 in order.')
    requested = symbols if only is None else only
    if len(set(requested)) != len(requested) or not set(requested).issubset(symbols):
        raise ValueError('--only requires unique members of the frozen US500.')
    root = Path(config['data_root']); root.mkdir(parents=True, exist_ok=True)
    report = Path(config['report_root']); report.mkdir(parents=True, exist_ok=True)
    old = Path(config['reuse_data_root'])
    if all((old/'metadata'/f'{s}.json').exists() for s in original): reuse_original(config,root)
    calendar = make_calendar(config['start'], config['end'])
    path = root/'calendar.csv'
    if path.exists(): pd.testing.assert_frame_equal(pd.read_csv(path,keep_default_na=False),calendar,check_dtype=False)
    else: calendar.to_csv(path,index=False)
    rows=[];started=now()
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures={pool.submit(download_candidate,{'symbol':symbol,'scope':'frozen_US500_member'},config,root,calendar):symbol for symbol in requested}
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            with (report/'frozen_download_events.jsonl').open('a', encoding="utf-8") as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({'symbol':row['symbol'],'status':row['status'],'progress':len(rows),'total':len(requested)}),flush=True)
    result={'started_utc':started,'finished_utc':now(),'requested_symbols':requested,
            'success':sum(r['status']=='success' for r in rows),'failed':sum(r['status']!='success' for r in rows),
            'universe_sha256':sha256(Path(config['universe_file'])),
            'note':'Frozen membership never replaced due to current price/liquidity changes or failed requests.',
            'outcomes':rows}
    name='frozen_download_report.json' if only is None else 'retry_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json'
    save_json(report/name,result)
    return result


def diagnose_frame(symbol, frame, provider_meta):
    """Describe supplied zero-volume bars and large price moves; never alter them."""
    close=frame.close.to_numpy();previous=np.r_[np.nan,close[:-1]]
    returns=close/previous-1;gaps=frame.open.to_numpy()/previous-1
    def extremum(values,mask):
        valid=np.flatnonzero(np.isfinite(values)&mask)
        if not len(valid):return None
        index=valid[np.argmax(np.abs(values[valid]))]
        return {'date':str(frame.datetime.iloc[index]),'signed_return':float(values[index]),'absolute_return':float(abs(values[index]))}
    evaluation=frame.datetime.between('2022-01-01','2025-12-31').to_numpy()
    events=[]
    for index in np.flatnonzero(frame.volume.eq(0).to_numpy()):
        values={key:float(frame[key].iloc[index]) for key in ['open','high','low','close']}
        events.append({'symbol':symbol,'date':str(frame.datetime.iloc[index]),'all_OHLC_equal':len(set(values.values()))==1,
            'intraday_range_positive':values['high']>values['low'],
            'close_changed_from_prior':bool(index>0 and not np.isclose(values['close'],previous[index],rtol=1e-10,atol=1e-10)),
            'signed_close_return':float(returns[index]) if np.isfinite(returns[index]) else None,
            'signed_open_gap':float(gaps[index]) if np.isfinite(gaps[index]) else None,
            'is_tradeable':bool(frame.is_tradeable.iloc[index])})
    first=provider_meta.get('firstTradeDate')
    first_date=datetime.fromtimestamp(first,timezone.utc).astimezone(NY).date().isoformat() if first else None
    result={'symbol':symbol,'provider_symbol':provider_meta.get('symbol'),'provider_current_exchange':provider_meta.get('fullExchangeName'),
        'provider_firstTradeDate_epoch':first,'provider_firstTradeDate_NY':first_date,
        'observed_first_date':str(frame.datetime.iloc[0]),'observed_last_date':str(frame.datetime.iloc[-1]),
        'provider_firstTradeDate_matches_observed_when_within_request':None if first_date is None or first_date<'2020-01-01' else first_date==str(frame.datetime.iloc[0]),
        'zero_volume_count':len(events),'zero_volume_2022_2025':sum('2022-01-01'<=e['date']<='2025-12-31' for e in events),
        'zero_four_prices_equal':sum(e['all_OHLC_equal'] for e in events),'zero_with_intraday_range':sum(e['intraday_range_positive'] for e in events),
        'zero_close_changed_from_prior':sum(e['close_changed_from_prior'] for e in events),
        'largest_abs_adjusted_close_return_all_requested_dates':extremum(returns,np.ones(len(frame),dtype=bool)),
        'largest_abs_adjusted_open_gap_all_requested_dates':extremum(gaps,np.ones(len(frame),dtype=bool)),
        'largest_abs_adjusted_close_return_2022_2025':extremum(returns,evaluation),
        'largest_abs_adjusted_open_gap_2022_2025':extremum(gaps,evaluation)}
    return result,events


def audit_diagnostics(config):
    root=Path(config['data_root']);symbols=load_symbols(Path(config['universe_file']),500)
    outcomes=[];zeros=[]
    for symbol in symbols:
        meta=json.loads((root/'metadata'/f'{symbol}.json').read_text(encoding="utf-8"))
        raw=root/'raw'/f'{symbol}.json';daily=root/'daily'/f'{symbol}.csv'
        if sha256(raw)!=meta['raw_sha256'] or sha256(daily)!=meta['daily_sha256']:
            raise ValueError(f'QA diagnostic input hash mismatch: {symbol}')
        row,events=diagnose_frame(symbol,pd.read_csv(daily),meta['provider_meta'])
        row.update(raw_sha256=meta['raw_sha256'],daily_sha256=meta['daily_sha256'])
        outcomes.append(row);zeros.extend(events)
    def exceeds(row,key):
        return row[key] is not None and row[key]['absolute_return']>.5
    report={'audited_at_utc':now(),'universe_count':500,'universe_sha256':sha256(Path(config['universe_file'])),
        'raw_data_modified':False,'scope':'Data quality only;2026 remainsQA-only. No strategy results used.',
        'zero_volume_count':len(zeros),'zero_volume_symbols':[r['symbol'] for r in outcomes if r['zero_volume_count']],
        'zero_policy':'All supplied zero-volume rows are retained and nontradeable. These are distinct from missing exchange sessions. Yahoo zeros or flatOHLC do not independently certify a suspension or executable price.',
        'zero_known_market_history_context':MARKET_HISTORY_CONTEXT,
        'firstTradeDate_caveat':'CurrentYahoo exchange and firstTradeDate are vendor fields, not historical exchange/issuer-identity certification.',
        'outcomes':outcomes,'zero_volume_events':zeros,
        'provider_symbol_aliases':[{'requested':r['symbol'],'returned':r['provider_symbol']} for r in outcomes if r['provider_symbol']!=r['symbol']],
        'adjusted_close_abs_move_above_50pct':[r['symbol'] for r in outcomes if exceeds(r,'largest_abs_adjusted_close_return_all_requested_dates')],
        'adjusted_open_gap_above_50pct':[r['symbol'] for r in outcomes if exceeds(r,'largest_abs_adjusted_open_gap_all_requested_dates')]}
    save_json(Path(config['report_root'])/'quality_diagnostics.json',report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/research/us500_download.json')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--diagnostics-only',action='store_true',help='Read frozen source hashes, zero-volume patterns and return/gap extremes; no downloads.')
    parser.add_argument('--only',nargs='+',help='Retry only named members of an already frozen US500.')
    args = parser.parse_args(argv); path = Path(args.config); config = json.loads(path.read_text(encoding="utf-8"))
    if args.audit_only and args.only:parser.error('--audit-only cannot be combined with --only.')
    if args.diagnostics_only:
        if args.audit_only or args.only:parser.error('--diagnostics-only is a separate read-only action.')
        report=audit_diagnostics(config)
        print(json.dumps({'status':'completed','universe_count':report['universe_count'],'zero_volume_count':report['zero_volume_count']}),flush=True)
        return 0
    if not args.audit_only:
        if Path(config['universe_file']).exists():download_frozen(config,args.only)
        elif args.only:parser.error('Freeze the500-stock universe before retrying named members.')
        else:expand(config, path)
    if load_symbols(Path(config['universe_file']),500)[:100] != load_symbols(Path(config['preserve_universe_file']),100):
        raise ValueError('Audit refuses a US500 list that changed the original100.')
    quality = audit_snapshot(config, path)
    if quality['failed_stocks']==0:audit_diagnostics(config)
    print(json.dumps({'status': 'completed', 'successful_stocks': quality['successful_stocks'], 'failed_stocks': quality['failed_stocks'], 'rows': quality['rows']}), flush=True)
    return 0 if quality['failed_stocks'] == 0 else 2


if __name__ == '__main__': raise SystemExit(main())
