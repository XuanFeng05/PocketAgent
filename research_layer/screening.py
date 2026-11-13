"""Vectorized training-prefix signal screening, never an execution backtest."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, time
import math
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from backtest_layer.data import normalize_daily_bars
from strategy_layer.registry import create_strategy

SUPPORTED_STRATEGIES = {"sma_trend", "breakout", "mean_reversion"}
DEFAULT_SETTINGS = {"train_fraction": .7, "train_bars": None, "forward_horizon": 5,
                    "max_candidates": 64, "max_bar_evaluations": 2_000_000}


def resolve_screen_settings(settings: Mapping[str, Any] | None = None) -> dict:
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("screening settings must be an object")
    raw = dict(settings or {})
    unknown = set(raw)-set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown screening settings: {sorted(unknown)}")
    result = {**DEFAULT_SETTINGS, **raw}
    for key, low, high in [("forward_horizon",1,2000),("max_candidates",1,64),("max_bar_evaluations",1,20_000_000)]:
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,Integral) or not low<=value<=high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
        result[key]=int(value)
    value=result["train_fraction"]
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<1:
        raise ValueError("train_fraction must be strictly between zero and one")
    result["train_fraction"]=float(value)
    if result["train_bars"] is not None:
        value=result["train_bars"]
        if isinstance(value,bool) or not isinstance(value,Integral) or value<2:
            raise ValueError("train_bars must be an integer >= 2, or null")
        result["train_bars"]=int(value)
    return result


class _RollingCache:
    def __init__(self, closes: pd.Series):
        self.closes=closes.reset_index(drop=True)
        self.values={}
        self.hits=0

    def get(self, operation: str, window: int) -> np.ndarray:
        key=(operation,int(window))
        if key not in self.values:
            if operation=="mean":
                values=self.closes.rolling(window,min_periods=window).mean()
            elif operation=="prior_max":
                values=self.closes.shift(1).rolling(window,min_periods=window).max()
            else:
                raise ValueError("Unknown rolling operation")
            self.values[key]=values.to_numpy(dtype=float)
        else:
            self.hits+=1
        return self.values[key]


def _score_and_signal(strategy: str, parameters: dict, cache: _RollingCache):
    # The registry is the authoritative validation contract for the underlying
    # strategy, even though this pass approximates entry conditions only.
    policy=create_strategy(strategy,parameters)
    closes=cache.closes.to_numpy(dtype=float)
    if strategy=="sma_trend":
        fast=cache.get("mean",policy.fast_window)
        slow=cache.get("mean",policy.slow_window)
        score=fast/slow-1
        signal=fast>slow
    elif strategy=="breakout":
        reference=cache.get("prior_max",policy.lookback)
        score=closes/reference-1
        signal=closes>reference
        # Match the rule's requirement that its exit window also has history.
        signal[:max(policy.lookback,policy.exit_window)]=False
        score[:max(policy.lookback,policy.exit_window)]=np.nan
    else:
        average=cache.get("mean",policy.window)
        score=1-closes/average
        signal=closes<=average*(1-policy.entry_deviation)
    return score,np.asarray(signal,dtype=bool)


def screen_candidates(bars: Any, strategy: str, candidates: list[dict], settings=None,
                      cancel_check: Callable[[], bool] | None = None) -> dict:
    resolved=resolve_screen_settings(settings)
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Screening supports only {sorted(SUPPORTED_STRATEGIES)}")
    if not isinstance(candidates,list) or not 1<=len(candidates)<=resolved["max_candidates"]:
        raise ValueError(f"Screening requires 1 to {resolved['max_candidates']} candidates")
    rows=[];identifiers=set()
    for item in candidates:
        if not isinstance(item,dict) or not isinstance(item.get("parameters"),dict):
            raise ValueError("Each candidate needs candidate_id and parameters")
        identifier=item.get("candidate_id")
        if not isinstance(identifier,str) or not identifier or identifier in identifiers:
            raise ValueError("Candidate IDs must be nonempty and unique")
        identifiers.add(identifier)
        rows.append({"candidate_id":identifier,"parameters":dict(item["parameters"])})
    rows.sort(key=lambda row:row["candidate_id"])
    frame=normalize_daily_bars(bars)
    count=len(frame)
    train_count=resolved["train_bars"] if resolved["train_bars"] is not None else math.floor(count*resolved["train_fraction"])
    horizon=resolved["forward_horizon"]
    if train_count>=count:
        raise ValueError("Screening must reserve at least one trailing bar")
    if train_count<=horizon:
        raise ValueError("Training prefix is too short for a matured forward label")
    estimated_work=train_count*len(rows)
    if estimated_work>resolved["max_bar_evaluations"]:
        raise ValueError(f"Screening needs {estimated_work} row-candidate evaluations, over the work budget")
    train=frame.iloc[:train_count].copy()
    closes=train["close"].to_numpy(dtype=float)
    opens=train["open"].to_numpy(dtype=float)
    label_count=train_count-horizon
    labels=closes[horizon:]/opens[1:label_count+1]-1
    dates=train["datetime"].dt.date.to_list()
    shared_samples=[{"sample_index":i,"signal_index":i,
                     "signal_time":datetime.combine(dates[i],time(15)).isoformat(),
                     "entry_index":i+1,"entry_time":datetime.combine(dates[i+1],time(9,30)).isoformat(),
                     "exit_index":i+horizon,"exit_time":datetime.combine(dates[i+horizon],time(15)).isoformat(),
                     "forward_return":float(value)} for i,value in enumerate(labels)]
    cache=_RollingCache(train["close"])
    trials=[];ranking=[]
    for item in rows:
        if cancel_check and cancel_check():
            raise InterruptedError("Vectorized screening cancelled")
        trial={**item,"approximate":True,"ranking_scope":"training_approximation"}
        try:
            score,signal=_score_and_signal(strategy,item["parameters"],cache)
            eligible=np.isfinite(score[:label_count]) & np.isfinite(labels)
            selected=np.flatnonzero(eligible & signal[:label_count])
            selected_returns=labels[selected]
            x,y=score[:label_count][eligible],labels[eligible]
            ic=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>0 and y.std()>0 else None
            if ic is not None and not math.isfinite(ic):ic=None
            metrics={"eligible_label_count":int(eligible.sum()),"signal_count":int(len(selected)),
                     "signal_frequency":float(len(selected)/eligible.sum()) if eligible.any() else None,
                     "mean_forward_return":float(selected_returns.mean()) if len(selected) else None,
                     "median_forward_return":float(np.median(selected_returns)) if len(selected) else None,
                     "hit_rate":float(np.mean(selected_returns>0)) if len(selected) else None,
                     "pearson_ic":ic}
            trial.update(status="completed",metrics=metrics,sample_indices=selected.tolist())
            if len(selected):
                ranking.append({"candidate_id":item["candidate_id"],"parameters":item["parameters"],
                                "score":metrics["mean_forward_return"],"signal_count":metrics["signal_count"],
                                "metric":"mean_forward_return","ranking_scope":"training_approximation"})
            else:
                trial["ranking_exclusion"]="no_matured_entry_signals"
        except (ValueError,TypeError,ArithmeticError,RuntimeError) as exc:
            trial.update(status="failed",error=str(exc),metrics=None,sample_indices=[])
        trials.append(trial)
    ranking.sort(key=lambda row:(-row["score"],row["candidate_id"]))
    for i,row in enumerate(ranking,1):row["rank"]=i
    return {"mode":"screen","approximate":True,"strategy_name":strategy,"settings":resolved,
            "status":"completed" if all(t["status"]=="completed" for t in trials) else "partial" if any(t["status"]=="completed" for t in trials) else "failed",
            "trials":trials,"ranking":ranking,"label_samples":shared_samples,
            "training_range":{"start":frame.iloc[0]["datetime"].date().isoformat(),
                              "end":frame.iloc[train_count-1]["datetime"].date().isoformat(),"bars":train_count,
                              "start_index":0,"end_index_exclusive":train_count},
            "reserved_range":{"start":frame.iloc[train_count]["datetime"].date().isoformat(),
                              "end":frame.iloc[-1]["datetime"].date().isoformat(),"bars":count-train_count,
                              "start_index":train_count,"end_index_exclusive":count,"evaluated":False},
            "cache_stats":{"rolling_computations":len(cache.values),"cache_hits":cache.hits,
                           "keys":[f"{op}:{window}" for op,window in sorted(cache.values)]},
            "estimated_bar_evaluations":estimated_work,
            "selection_scope":"training_only_approximate","automatic_promotion":False,
            "assumptions":[
                "This is vectorized entry-condition screening on a training prefix, not a net-cost strategy backtest; it creates no NAV or executable fills.",
                "A label observes the next available daily open through the forward_horizon-th later close. Its final close must remain inside the training prefix.",
                "Trailing reserved prices never form signals, scores or labels; the full input may still be normalized for data-quality checks.",
                "Signals are condition occurrences, not independent trades or a simulated position lifecycle; overlapping forward horizons are statistically dependent.",
                "Gross labels ignore fees, liquidity, ST/trading permission and execution feasibility. Surviving hypotheses require the full shared-executor backtest and out-of-sample validation.",
                "pearson_ic is a time-series Pearson correlation between the continuous rule score and matured gross labels, not cross-sectional rank IC.",
                "All candidates remain in trials, including failures and zero-signal cases; no candidate is automatically removed or promoted.",
            ]}
