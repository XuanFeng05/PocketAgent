from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_layer.specs import IndicatorSpec


@dataclass(frozen=True)
class IndicatorCalculation:
    model_fields: dict[str, pd.Series]
    display_fields: dict[str, pd.Series]


def calculate_indicator(
    indicator: IndicatorSpec,
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> IndicatorCalculation:
    """Calculate one technical indicator for both model and chart consumers."""
    params = indicator.params
    prefix = indicator.id

    if indicator.kind == "macd":
        fast = close.ewm(span=params["fast"], adjust=False, min_periods=1).mean()
        slow = close.ewm(span=params["slow"], adjust=False, min_periods=1).mean()
        dif = fast - slow
        dea = dif.ewm(span=params["signal"], adjust=False, min_periods=1).mean()
        fields = {
            f"{prefix}_dif_pct": _safe_div(dif, close),
            f"{prefix}_dea_pct": _safe_div(dea, close),
            f"{prefix}_hist_pct": _safe_div(dif - dea, close),
        }
        return IndicatorCalculation(model_fields=fields, display_fields=fields)

    if indicator.kind == "kd":
        low_n = low.rolling(params["lookback"], min_periods=1).min()
        high_n = high.rolling(params["lookback"], min_periods=1).max()
        rsv = _safe_div_with_fill(close - low_n, high_n - low_n, fill=0.5).clip(0.0, 1.0)
        k = rsv.ewm(alpha=1 / params["smooth_k"], adjust=False, min_periods=1).mean()
        d = k.ewm(alpha=1 / params["smooth_d"], adjust=False, min_periods=1).mean()
        fields = {
            f"{prefix}_k_norm": k,
            f"{prefix}_d_norm": d,
            f"{prefix}_diff_norm": k - d,
        }
        return IndicatorCalculation(model_fields=fields, display_fields=fields)

    if indicator.kind == "efi":
        efi_reference = close.shift(1).where(close.shift(1).notna() & (close.shift(1) != 0), close)
        force = (close - efi_reference) * volume
        efi2 = force.ewm(span=params["fast"], adjust=False, min_periods=1).mean()
        efi13 = force.ewm(span=params["slow"], adjust=False, min_periods=1).mean()
        baseline = force.abs().shift(1).rolling(params["baseline"], min_periods=1).mean()
        model_fields = {
            f"{prefix}2_norm": _safe_div(efi2, baseline),
            f"{prefix}13_norm": _safe_div(efi13, baseline),
        }
        display_fields = {
            f"{prefix}2": efi2,
            f"{prefix}13": efi13,
        }
        return IndicatorCalculation(model_fields=model_fields, display_fields=display_fields)

    fast = close.ewm(span=params["fast"], adjust=False, min_periods=1).mean()
    slow = close.ewm(span=params["slow"], adjust=False, min_periods=1).mean()
    lower = pd.concat([fast, slow], axis=1).min(axis=1)
    upper = pd.concat([fast, slow], axis=1).max(axis=1)
    midpoint = (fast + slow) / 2
    channel_prefix = f"ema_channel_{prefix}"
    model_fields = {
        # When fast and slow EMA are identical at the seeded start, treat
        # price as centered in a zero-width channel instead of emitting NaN.
        f"{channel_prefix}_position": _safe_div_with_fill(close - lower, upper - lower, fill=0.5),
        f"{channel_prefix}_gap": _safe_div(close, midpoint) - 1,
        f"{channel_prefix}_width": _safe_div_with_fill(upper - lower, midpoint, fill=0.0),
        f"{channel_prefix}_slope": _safe_div_with_fill(midpoint, midpoint.shift(1), fill=1.0) - 1,
    }
    display_fields = {
        f"{prefix}_fast": fast,
        f"{prefix}_slow": slow,
    }
    return IndicatorCalculation(model_fields=model_fields, display_fields=display_fields)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _safe_div_with_fill(numerator: pd.Series, denominator: pd.Series, *, fill: float) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    values = numerator / denom
    return values.where(denom.notna(), float(fill))
