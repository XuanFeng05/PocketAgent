from __future__ import annotations

import numpy as np
import pandas as pd

from feature_layer.specs import DEFAULT_FEATURE_SPEC, FeatureSpec


def build_decision_context(
    decisions: pd.DataFrame,
    *,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> pd.DataFrame:
    """Build one row of model-facing time context for each decision."""
    columns = ["decision_id", "symbol", "decision_time", "stage", *spec.context_feature_names]
    if decisions is None or decisions.empty:
        return pd.DataFrame(columns=columns)

    result = decisions.loc[:, ["decision_id", "symbol", "decision_time", "stage"]].copy()
    timestamps = pd.to_datetime(result["decision_time"], errors="coerce")
    elapsed = timestamps.map(_elapsed_trading_minutes).astype(float)
    slots = np.ceil(elapsed / 5.0).clip(0, 48)
    minutes_of_day = timestamps.map(
        lambda ts: ts.hour * 60 + ts.minute if pd.notna(ts) else np.nan
    )

    result["bar_slot_norm"] = (slots / 48.0).clip(0.0, 1.0)
    result["day_progress"] = (elapsed / 240.0).clip(0.0, 1.0)
    result["is_morning_session"] = (
        (minutes_of_day >= 9 * 60 + 30) & (minutes_of_day <= 11 * 60 + 30)
    ).astype(float)
    result["is_afternoon_session"] = (
        (minutes_of_day >= 13 * 60) & (minutes_of_day <= 15 * 60)
    ).astype(float)
    result["minutes_to_close_norm"] = ((240.0 - elapsed) / 240.0).clip(0.0, 1.0)
    result["is_open_auction"] = result["stage"].astype(str).eq("open_auction").astype(float)
    return result[columns].reset_index(drop=True)


def _elapsed_trading_minutes(timestamp: pd.Timestamp) -> float:
    if pd.isna(timestamp):
        return 0.0
    minute = pd.Timestamp(timestamp).hour * 60 + pd.Timestamp(timestamp).minute
    if minute <= 9 * 60 + 30:
        return 0.0
    if minute <= 11 * 60 + 30:
        return float(minute - (9 * 60 + 30))
    if minute < 13 * 60:
        return 120.0
    if minute <= 15 * 60:
        return float(120 + minute - 13 * 60)
    return 240.0
