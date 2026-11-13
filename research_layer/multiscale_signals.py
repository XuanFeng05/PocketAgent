"""Frozen daily signals and completed-week conditions for US500 research.

All published values are available at the row's NY session close. Consumers
must execute from the next session, never at the signal bar's opening price.
These are explicit technical-research rules, not claims of canonical strategy
implementations or independently profitable indicators.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research_layer.elder_engine import build_features as _validate_daily


FEATURE_VERSION = "us500_multiscale_close_features_v1"
SIGNAL_IDS = (
    "efi2_pullback", "efi3_pullback", "rsi2_oversold", "rsi14_oversold",
    "stochastic_recovery", "williams_oversold", "bollinger_reentry",
    "ema20_reclaim", "donchian20_breakout", "macd_bull_cross",
)
WEEKLY_FILTER_IDS = (
    "none", "hist_rising", "ema13_rising", "ema13_26_up", "adx_up",
    "rsi14_up", "impulse_nonred",
)
CONDITION_IDS = (
    "none", "atr_high", "atr_low", "adx_trend", "adx_range",
    "volume_high", "above_sma200", "below_sma200",
)

TECHNICAL_SOURCES = {
    "RSI": "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI",
    "MACD": "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd",
    "DMI_ADX": "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI",
    "Williams_R": "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/williams-r",
    "Bollinger": "https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands",
    "Slow_stochastic": "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full",
    "Force_index": "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/force-index",
    "Donchian": "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/price-channels",
}

SIGNAL_DEFINITIONS = {
    "efi2_pullback": "EMA2((close-previous close)*share volume)<0",
    "efi3_pullback": "EMA3((close-previous close)*share volume)<0",
    "rsi2_oversold": "Wilder RSI2<10; a state, not just its first crossing",
    "rsi14_oversold": "Wilder RSI14<30; a state, not just its first crossing",
    "stochastic_recovery": "Slow K=SMA3(fast K14)<20 and Slow K>Slow D=SMA3(Slow K); recovery state, not a required fresh cross",
    "williams_oversold": "Williams R14<-80; a state, not just its first crossing",
    "bollinger_reentry": "Previous close<previous lower band; current close>=current lower band; SMA20 minus2 population standard deviations",
    "ema20_reclaim": "Previous close<previous EMA20 and current close>=current EMA20",
    "donchian20_breakout": "Current close>highest high of the previous20 completed bars; current bar excluded from channel",
    "macd_bull_cross": "MACD12/26/9 histogram changes from<=0 to>0",
}


def wilder_mean(values: pd.Series, period: int) -> pd.Series:
    """SMA seed followed by Wilder recurrence, resetting after missing values."""
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError("Wilder period must be a positive integer")
    a = np.asarray(values, dtype=float)
    out = np.full(len(a), np.nan)
    seed: list[float] = []
    previous = np.nan
    for i, value in enumerate(a):
        if not np.isfinite(value):
            seed = []
            previous = np.nan
        elif np.isnan(previous):
            seed.append(float(value))
            if len(seed) == period:
                previous = float(np.mean(seed))
                out[i] = previous
        else:
            previous = (previous*(period-1)+value)/period
            out[i] = previous
    return pd.Series(out, index=values.index, name=values.name)


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = wilder_mean(delta.clip(lower=0), period)
    down = wilder_mean((-delta).clip(lower=0), period)
    total = up+down
    # Explicit neutral convention for a completely flat history avoids 0/0.
    result = 100*up/total.where(total.ne(0))
    return result.mask(total.eq(0), 50.)


def indicator_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Same parameter definitions on daily or completed-week OHLCV bars."""
    c, h, l, v = (bars[k].astype(float) for k in ("close", "high", "low", "volume"))
    f = pd.DataFrame(index=bars.index)
    force = c.diff()*v
    for period in (2, 3):
        f[f"efi{period}"] = force.ewm(span=period, adjust=False, min_periods=period).mean()
        f[f"rsi{period}"] = rsi(c, period)
    f["rsi14"] = rsi(c, 14)
    for period in (13, 20, 26):
        f[f"ema{period}"] = c.ewm(span=period, adjust=False, min_periods=period).mean()
    f["ema13_previous"] = f.ema13.shift()
    f["sma200"] = c.rolling(200, min_periods=200).mean()
    macd = c.ewm(span=12, adjust=False, min_periods=12).mean()-f.ema26
    f["macd"] = macd
    f["macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["hist"] = macd-f.macd_signal
    f["hist_previous"] = f["hist"].shift()
    lo14, hi14 = l.rolling(14, min_periods=14).min(), h.rolling(14, min_periods=14).max()
    span = hi14-lo14
    f["stoch_fast14"] = (100*(c-lo14)/span.where(span.ne(0))).mask(span.eq(0), 50.)
    f["stoch_k14_3"] = f.stoch_fast14.rolling(3, min_periods=3).mean()
    f["stoch_d14_3_3"] = f.stoch_k14_3.rolling(3, min_periods=3).mean()
    f["williams_r14"] = (-100*(hi14-c)/span.where(span.ne(0))).mask(span.eq(0), -50.)
    f["bb_mid20"] = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    f["bb_lower20"] = f.bb_mid20-2*sd
    f["bb_upper20"] = f.bb_mid20+2*sd
    f["donchian_high20"] = h.shift().rolling(20, min_periods=20).max()
    f["donchian_low20"] = l.shift().rolling(20, min_periods=20).min()
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    f["atr14"] = wilder_mean(tr, 14)
    up, down = h.diff(), -l.diff()
    dm_plus = up.where((up>down) & (up>0), 0.).mask(up.isna())
    dm_minus = down.where((down>up) & (down>0), 0.).mask(down.isna())
    # DMI's first movement observation needs a previous bar; all three smoothers
    # therefore seed from the same14 completed movements (indices1..14).
    tr_dm = tr.copy()
    if len(tr_dm):
        tr_dm.iloc[0] = np.nan
    sm_tr = wilder_mean(tr_dm, 14)
    for name, values in (("plus_di14", dm_plus), ("minus_di14", dm_minus)):
        f[name] = (100*wilder_mean(values, 14)/sm_tr.where(sm_tr.ne(0))).mask(sm_tr.eq(0), 0.)
    den = f.plus_di14+f.minus_di14
    dx = (100*(f.plus_di14-f.minus_di14).abs()/den.where(den.ne(0))).mask(den.eq(0), 0.)
    f["adx14"] = wilder_mean(dx, 14)
    f["atr_pct"] = f.atr14/c
    f["atr_pct_prior126_median"] = f.atr_pct.shift().rolling(126, min_periods=126).median()
    f["volume_prior20_mean"] = v.shift().rolling(20, min_periods=20).mean()
    return f


def _raw_signals(bars: pd.DataFrame, ind: pd.DataFrame) -> dict[str, pd.Series]:
    c = bars.close
    result = {
        "efi2_pullback": ind.efi2.lt(0),
        "efi3_pullback": ind.efi3.lt(0),
        "rsi2_oversold": ind.rsi2.lt(10),
        "rsi14_oversold": ind.rsi14.lt(30),
        "stochastic_recovery": ind.stoch_k14_3.lt(20) & ind.stoch_k14_3.gt(ind.stoch_d14_3_3),
        "williams_oversold": ind.williams_r14.lt(-80),
        "bollinger_reentry": c.shift().lt(ind.bb_lower20.shift()) & c.ge(ind.bb_lower20),
        "ema20_reclaim": c.shift().lt(ind.ema20.shift()) & c.ge(ind.ema20),
        "donchian20_breakout": c.gt(ind.donchian_high20),
        "macd_bull_cross": ind.hist_previous.le(0) & ind["hist"].gt(0),
    }
    assert tuple(result) == SIGNAL_IDS
    return result


def build_features(bars: Any, calendar: Any, warmup_weeks: int = 100) -> pd.DataFrame:
    """Return close-time features; no signal is shifted to its execution bar.

    The established validator rejects duplicate, missing open-session and
    invalid adjusted OHLC bars. Valid zero-volume rows remain untradeable.
    Partially observed IPO/boundary weeks do not count toward100-week warmup.
    """
    validated = _validate_daily(bars, calendar, warmup_weeks=warmup_weeks)
    keep = ["datetime", "symbol", "open", "high", "low", "close", "volume", "tradeable", "session_close", "adjust"]
    keep += [k for k in ("freq", "source", "adjust_factor", "vendor_open", "vendor_high", "vendor_low", "vendor_close") if k in validated]
    f = validated[keep].copy()
    ind = indicator_frame(f)
    f = pd.concat([f, ind], axis=1)
    f["is_tradeable"] = f.tradeable
    # Reuse the established validator's exact calendar completion mask. In
    # particular, a holiday need not have a missing session_close in the raw
    # calendar: its explicit is_trading_day flag is authoritative.
    completed_here = validated.session_close.eq(validated.weekly_available_at)
    rows = []
    for period, group in f.groupby(f.datetime.dt.to_period("W-SUN"), sort=True):
        if not completed_here.iloc[group.index[-1]]:
            continue
        rows.append({"weekly_available_at": group.session_close.iloc[-1],
                     "open": group.open.iloc[0], "high": group.high.max(), "low": group.low.min(),
                     "close": group.close.iloc[-1], "volume": group.volume.sum()})
    weekly = pd.DataFrame(rows)
    if len(weekly):
        wi = indicator_frame(weekly)
        ws = _raw_signals(weekly, wi)
        pub = pd.DataFrame({"weekly_available_at": weekly.weekly_available_at,
                            "weekly_completed_count": np.arange(1, len(weekly)+1),
                            "weekly_close": weekly.close})
        pub = pd.concat([pub, wi.add_prefix("weekly_")], axis=1)
        for key, mask in ws.items():
            pub[f"wsignal__{key}"] = mask
        f = pd.merge_asof(f, pub, left_on="session_close", right_on="weekly_available_at", direction="backward")
    else:
        f["weekly_available_at"] = pd.NaT
        f["weekly_completed_count"] = 0
        f["weekly_close"] = np.nan
        for key in ind:
            f[f"weekly_{key}"] = np.nan
        for key in SIGNAL_IDS:
            f[f"wsignal__{key}"] = False
    f["weekly_completed_count"] = f.weekly_completed_count.fillna(0).astype(int)
    f["ready_week100"] = f.weekly_completed_count.ge(warmup_weeks)
    f["ready"] = f.ready_week100 & f.atr14.gt(0) & f.hist_previous.notna() & f.sma200.notna()
    f["weekly_published"] = f.session_close.eq(f.weekly_available_at)
    # Source-provided vendor_close is already split-adjusted on this feed;
    # this is a declared research eligibility proxy, not historical nominal
    # price reconstruction. Fallback exists only for synthetic fixtures.
    f["universe_trade_eligible"] = f.get("vendor_close", f.close).ge(10)
    for key, mask in _raw_signals(f, f).items():
        f[f"signal__{key}"] = (mask & f.ready & f.universe_trade_eligible).astype(bool)
        f[f"wsignal__{key}"] = f[f"wsignal__{key}"].eq(True) & f.weekly_published & f.ready & f.universe_trade_eligible
    wh_up = f.weekly_hist.gt(f.weekly_hist_previous)
    we_up = f.weekly_ema13.gt(f.weekly_ema13_previous)
    week_valid = f.ready & f.weekly_hist_previous.notna() & f.weekly_ema13_previous.notna()
    filters = {
        "none": f.ready,
        "hist_rising": wh_up & week_valid,
        "ema13_rising": we_up & week_valid,
        "ema13_26_up": f.weekly_ema13.gt(f.weekly_ema26) & f.weekly_close.gt(f.weekly_ema13) & week_valid,
        "adx_up": f.weekly_adx14.gt(20) & f.weekly_plus_di14.gt(f.weekly_minus_di14) & f.ready,
        "rsi14_up": f.weekly_rsi14.gt(50) & f.ready,
        "impulse_nonred": ~(f.weekly_hist.lt(f.weekly_hist_previous) & f.weekly_ema13.lt(f.weekly_ema13_previous)) & week_valid,
    }
    for key, mask in filters.items():
        f[f"weekly__{key}"] = mask.astype(bool)
    conditions = {
        "none": f.ready,
        "atr_high": f.atr_pct.gt(f.atr_pct_prior126_median) & f.ready,
        "atr_low": f.atr_pct.le(f.atr_pct_prior126_median) & f.ready,
        "adx_trend": f.adx14.gt(25) & f.ready,
        "adx_range": f.adx14.lt(20) & f.ready,
        "volume_high": f.volume.gt(f.volume_prior20_mean) & f.ready,
        "above_sma200": f.close.gt(f.sma200) & f.ready,
        "below_sma200": f.close.le(f.sma200) & f.ready,
    }
    for key, mask in conditions.items():
        f[f"condition__{key}"] = mask.astype(bool)
    f.attrs.update(feature_version=FEATURE_VERSION, warmup_weeks=warmup_weeks,
                   signal_timing="Available at session close; trade no earlier than next session",
                   signal_definitions=SIGNAL_DEFINITIONS,
                   indicator_conventions="SMA-seeded Wilder RSI/ATR/DMI/ADX; recursive first-value EMA; population BB SD; flat RSI=50, stochastic=50, WilliamsR=-50, DMI=0")
    return f.reset_index(drop=True)


def _masks(features: pd.DataFrame, prefix: str, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    missing = [prefix+name for name in names if prefix+name not in features]
    if missing:
        raise ValueError(f"Missing prepared feature fields: {missing}")
    if any(features[prefix+name].isna().any() for name in names):
        raise ValueError("Prepared signal/condition masks cannot contain missing values")
    return {name: features[prefix+name].to_numpy(dtype=bool) for name in names}


def signal_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    return _masks(features, "signal__", SIGNAL_IDS)


def weekly_signal_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    return _masks(features, "wsignal__", SIGNAL_IDS)


def weekly_filter_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    return _masks(features, "weekly__", WEEKLY_FILTER_IDS)


def condition_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    return _masks(features, "condition__", CONDITION_IDS)
