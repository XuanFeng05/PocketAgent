from __future__ import annotations

import numpy as np
import pandas as pd

from feature_layer.normalizers.rolling import clip_feature_frame
from feature_layer.indicator_registry import indicator_lookback
from feature_layer.builders.indicator_calculations import calculate_indicator
from feature_layer.specs import DEFAULT_FEATURE_SPEC, FeatureSpec


META_OUTPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "datetime",
    "freq",
    "adjust",
)


def build_bar_features(
    bars: pd.DataFrame,
    *,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> pd.DataFrame:
    """
    Build model-facing market features for each visible bar.

    Raw price and volume columns are used for calculation but are not emitted as
    model input columns. The output keeps only metadata plus ratio/state fields.
    """
    working = _prepare_bars(bars)
    if working.empty:
        return pd.DataFrame(columns=list(META_OUTPUT_COLUMNS) + list(spec.market_feature_names))

    parts: list[pd.DataFrame] = []
    group_cols = [column for column in ("symbol", "freq", "adjust") if column in working.columns]
    for _, group in working.groupby(group_cols, sort=False, dropna=False):
        parts.append(_build_group_features(group.sort_values("datetime"), spec=spec))

    result = pd.concat(parts, ignore_index=True)
    result = clip_feature_frame(result, spec=spec)
    for column in spec.market_feature_names:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result[list(META_OUTPUT_COLUMNS) + list(spec.market_feature_names)]


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()

    working = bars.copy()
    if "datetime" not in working.columns and "bar_end" in working.columns:
        working["datetime"] = working["bar_end"]

    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required.difference(working.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["datetime"]).copy()

    for column in ["open", "high", "low", "close", "volume", "amount", "pctChg", "turn", "progress"]:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working["volume"] = working["volume"].fillna(0.0)
    working["amount"] = working["amount"].fillna(0.0)
    working["progress"] = working["progress"].fillna(1.0)

    if "symbol" not in working.columns:
        working["symbol"] = ""
    if "freq" not in working.columns:
        working["freq"] = ""
    if "adjust" not in working.columns:
        working["adjust"] = ""

    return working.sort_values(["symbol", "freq", "adjust", "datetime"]).reset_index(drop=True)


def _build_group_features(group: pd.DataFrame, *, spec: FeatureSpec) -> pd.DataFrame:
    result = group.loc[:, list(META_OUTPUT_COLUMNS)].copy()

    open_ = group["open"]
    high = group["high"]
    low = group["low"]
    close = group["close"]
    volume = group["volume"]
    amount = group["amount"]
    prev_close = close.shift(1)
    # Use an adaptive, no-lookahead seed for the first visible bar.
    # Extending the download window forever only moves the missing-reference
    # problem to an earlier bar, so the feature layer must define a stable
    # local reference for the first row.  Use previous close when available;
    # otherwise seed with the current open, falling back to current close.
    reference_close = _seeded_previous_close(prev_close, open_, close)
    current_pct = _safe_div(close, prev_close) - 1
    first_pct = _safe_div(close, reference_close) - 1
    pct_chg = group["pctChg"].where(group["pctChg"].notna(), current_pct)
    pct_chg = pct_chg.where(prev_close.notna() & (prev_close != 0), first_pct)

    result["pctChg"] = pct_chg
    result["log_ret"] = np.log(_safe_div(close, reference_close)).replace([np.inf, -np.inf], np.nan)
    result["open_close_ret"] = _safe_div(close, open_) - 1
    result["gap_ret"] = _safe_div(open_, reference_close) - 1
    result["high_low_range"] = _safe_div(high - low, reference_close)
    result["body_range"] = _safe_div((close - open_).abs(), reference_close)
    result["close_position"] = _safe_div_with_fill(close - low, high - low, fill=0.5)
    result["upper_shadow"] = _safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), reference_close)
    result["lower_shadow"] = _safe_div(pd.concat([open_, close], axis=1).min(axis=1) - low, reference_close)

    first_close = _first_valid_scalar(close)
    for window in (3, 5, 10, 20):
        reference = close.shift(window)
        if np.isfinite(first_close) and first_close != 0:
            reference = reference.where(reference.notna() & (reference != 0), first_close)
        result[f"ret_{window}"] = _safe_div(close, reference) - 1

    result["volatility_5"] = pct_chg.rolling(5, min_periods=2).std()
    result["volatility_20"] = pct_chg.rolling(20, min_periods=2).std()
    result["range_mean_20"] = result["high_low_range"].rolling(20, min_periods=1).mean()

    prior_volume_mean = volume.shift(1).rolling(20, min_periods=1).mean()
    prior_amount_mean = amount.shift(1).rolling(20, min_periods=1).mean()
    # Seed volume/amount ratios with the current bar when there is no prior
    # history, matching rolling-indicator behavior instead of emitting NaN.
    seeded_volume_mean = prior_volume_mean.where(prior_volume_mean.notna() & (prior_volume_mean != 0), volume)
    seeded_amount_mean = prior_amount_mean.where(prior_amount_mean.notna() & (prior_amount_mean != 0), amount)
    result["volume_ratio_20"] = _safe_div(volume, seeded_volume_mean)
    result["amount_ratio_20"] = _safe_div(amount, seeded_amount_mean)

    _add_turn_features(result, group["turn"])
    group_freq = str(group["freq"].iloc[0]) if len(group) else ""
    active_indicators = [
        item
        for item in spec.indicators
        if item.enabled and group_freq in item.frequencies
    ]
    for indicator in active_indicators:
        calculation = calculate_indicator(
            indicator,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        for name, values in calculation.model_fields.items():
            result[name] = values
    longest_lookback = max([20, *(indicator_lookback(item) for item in active_indicators)])
    result["history_coverage"] = (
        pd.Series(np.arange(1, len(group) + 1), index=group.index, dtype=float)
        / float(longest_lookback)
    ).clip(0.0, 1.0)
    sequence_window = max(1, int(spec.sequence_windows.get(group_freq, len(group) or 1)))
    result["sequence_valid_ratio"] = (
        pd.Series(np.arange(1, len(group) + 1), index=group.index, dtype=float)
        / float(sequence_window)
    ).clip(0.0, 1.0)
    result["progress"] = group["progress"]

    return result


def _add_turn_features(result: pd.DataFrame, turn: pd.Series) -> None:
    turn_lag1 = turn.shift(1)
    turn_ma20 = turn_lag1.rolling(20, min_periods=1).mean()
    turn_std20 = turn_lag1.rolling(20, min_periods=2).std()
    result["turn_lag1"] = turn_lag1
    result["turn_ma20"] = turn_ma20
    result["turn_z20"] = _safe_div(turn_lag1 - turn_ma20, turn_std20)


def _seeded_previous_close(prev_close: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
    reference = prev_close.where(prev_close.notna() & (prev_close != 0), open_)
    return reference.where(reference.notna() & (reference != 0), close)


def _first_valid_scalar(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def _safe_div_with_fill(numerator: pd.Series, denominator: pd.Series, *, fill: float) -> pd.Series:
    values = _safe_div(numerator, denominator)
    invalid = denominator.replace(0, np.nan).isna()
    return values.where(~invalid, float(fill))
