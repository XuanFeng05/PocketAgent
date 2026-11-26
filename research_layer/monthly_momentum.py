"""Eight fixed, past-close monthly research policies; not optimized parameters."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from feature_layer.triple_screen import _local_dates,_normalize_calendar
from research_layer.technical import normalize_technical_bars

POLICIES={"momentum10","buffer10","breadth10","vol12_10","defensive10","defensive5","equal100","cash"}
SETTINGS={"momentum_windows":[63,126],"trend_sma":126,"volatility_window":63,"volatility_target":.12,
          "breadth_minimum":.5,"base_slots":10,"concentrated_slots":5,"rank_retention_multiple":2,
          "rebalance":"month_end_signal_next_session_open","price_basis":"pre"}


def _date(value):
    v=pd.Timestamp(value)
    if pd.isna(v) or v.tzinfo is not None or v!=v.normalize():raise ValueError("Local midnight date required")
    return v


def _selection(ranked,previous,k,buffer):
    retained=[s for s in previous if s in ranked[:2*k]] if buffer else []
    return retained+[s for s in ranked if s not in retained][:k-len(retained)]


def _scale(history,weights,breadth,vol_control,breadth_control):
    sigma=0.
    if weights:
        portfolio_returns=history[list(weights)].mul(pd.Series(weights)).sum(axis=1)
        sigma=float(portfolio_returns.std(ddof=1)*math.sqrt(252))
    scale=min(1.,.12/sigma) if vol_control and sigma>0 else 1.
    if breadth_control and breadth<.5:scale=0.
    return scale,sigma


def build_monthly_decisions(bars_by_symbol,calendar,*,start,end,policy):
    if policy not in POLICIES:raise ValueError("Unknown policy")
    if not isinstance(bars_by_symbol,dict) or not bars_by_symbol:raise ValueError("Nonempty symbol mapping required")
    start,end=_date(start),_date(end)
    if start>end:raise ValueError("Invalid date range")
    frames={}
    for symbol,bars in bars_by_symbol.items():
        f=bars.copy() if isinstance(bars,pd.DataFrame) else pd.DataFrame(bars)
        if "datetime" not in f and "date" in f:f=f.rename(columns={"date":"datetime"})
        if "datetime" not in f:raise ValueError("Missing dates")
        f["datetime"]=_local_dates(f["datetime"],"bars");f=f.loc[f.datetime<=end].copy()
        f=normalize_technical_bars(f,min_rows=1)
        if f.symbol.nunique()!=1 or f.iloc[0].symbol!=symbol or "volume" not in f:raise ValueError("Symbol/volume mismatch")
        frames[symbol]=f.set_index("datetime")
    first=min(f.index[0] for f in frames.values());cal=_normalize_calendar(calendar,first,end)
    sessions=pd.DatetimeIndex(cal.loc[cal.is_trading_day,"date"])
    for symbol,f in frames.items():
        expected=sessions[(sessions>=f.index[0])&(sessions<=end)]
        if not f.index.equals(expected):raise ValueError(f"Missing/unexpected open session: {symbol}")
    month_ends=pd.Series(sessions,index=sessions).groupby(sessions.to_period("M")).max()
    dates=[]
    for d in month_ends:
        # The exchange calendar must cover the month, not merely the last input bar.
        if d.to_period("M").end_time.normalize()>cal.date.iloc[-1]:continue
        next_sessions=sessions[sessions>d]
        if len(next_sessions) and start<=next_sessions[0]<=end:dates.append(d)
    previous=[];decisions=[];n=len(frames)
    for date in dates:
        eligible=[];valid=above=0;tradeable=[];risk={};scores={}
        for symbol,f in frames.items():
            visible=f.loc[:date]
            if visible.empty or visible.index[-1]!=date:continue
            row=visible.iloc[-1];closes=visible.close.to_numpy(float)
            can_trade=bool(row.get("is_tradeable",True)) and row.volume>0
            if can_trade:tradeable.append(symbol)
            if len(closes)<127:continue
            valid+=1;trend=closes[-1]>closes[-126:].mean();above+=int(trend)
            r63=closes[-1]/closes[-64]-1;r126=closes[-1]/closes[-127]-1
            if trend and r63>0 and r126>0 and can_trade:
                scores[symbol]=.5*(math.log1p(r63)+math.log1p(r126));eligible.append(symbol)
                risk[symbol]=pd.Series(closes[-63:]/closes[-64:-1]-1,index=visible.index[-63:])
        ranked=sorted(eligible,key=lambda s:(-scores[s],s));breadth=above/valid if valid else 0.
        prior=previous[:];k=5 if policy=="defensive5" else 10;scale=1.;sigma=0.
        if policy=="cash":selected=[];weights={}
        elif policy=="equal100":selected=sorted(tradeable);weights={s:1/n for s in selected}
        else:
            selected=_selection(ranked,previous,k,policy!="momentum10");previous=selected[:]
            base={s:1/k for s in selected}
            history=pd.concat([risk[s].rename(s) for s in selected],axis=1) if selected else pd.DataFrame()
            if selected and (len(history)!=63 or history.isna().any().any()):raise ValueError("Risk history is not synchronized")
            scale,sigma=_scale(history,base,breadth,policy in {"vol12_10","defensive10","defensive5"},policy in {"breadth10","defensive10","defensive5"})
            weights={s:w*scale for s,w in base.items() if scale>0}
        if sum(weights.values())>1+1e-12:raise ValueError("Leveraged target")
        decisions.append({"signal_date":date.date().isoformat(),"weights":weights,
                          "metadata":{"policy":policy,"available_at":date.date().isoformat()+"T15:00:00",
                                      "eligible_count":len(ranked),"ranked_symbols":ranked,"scores":scores,
                                      "breadth":breadth,"breadth_numerator":above,"breadth_denominator":valid,
                                      "selected":selected,"previous_planned_selection":prior,"slots":k if policy not in {"cash","equal100"} else n,
                                      "planned_selection_scope":"buffer remembers planned names even during risk-off, not actual broker holdings",
                                      "estimated_portfolio_volatility":sigma,"risk_scale":scale,"target_exposure":sum(weights.values())}})
    return decisions
