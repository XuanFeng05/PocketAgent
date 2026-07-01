from __future__ import annotations

import pandas as pd

from feature_layer.specs import DEFAULT_FEATURE_SPEC, FeatureSpec


def clip_feature_frame(
    frame: pd.DataFrame,
    *,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> pd.DataFrame:
    """Clip known model-input columns to stable ranges from the feature spec."""
    result = frame.copy()
    fields = (
        spec.market_fields
        + spec.context_fields
        + spec.portfolio_fields
        + spec.constraint_fields
        + spec.environment_fields
    )
    for field in fields:
        if field.clip is None or field.name not in result.columns:
            continue
        lower, upper = field.clip
        result[field.name] = pd.to_numeric(result[field.name], errors="coerce").clip(lower, upper)
    return result
