from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PaddedMarketSequence:
    """Fixed-size model sequence plus a mask that identifies real rows."""

    values: np.ndarray
    sequence_mask: np.ndarray
    valid_rows: int
    valid_ratio: float


def pad_market_sequence(
    frame: pd.DataFrame,
    *,
    feature_names: Iterable[str],
    window: int,
) -> PaddedMarketSequence:
    """Left-pad a real market sequence with zeros without inventing K-line rows."""
    size = int(window)
    if size <= 0:
        raise ValueError("Sequence window must be positive.")

    columns = [str(name) for name in feature_names]
    if not columns:
        raise ValueError("At least one feature name is required.")

    source = frame.tail(size).copy() if frame is not None else pd.DataFrame()
    for column in columns:
        if column not in source.columns:
            source[column] = 0.0
    numeric = (
        source[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    valid_rows = min(size, len(numeric))
    values = np.zeros((size, len(columns)), dtype=np.float32)
    sequence_mask = np.zeros(size, dtype=np.float32)
    if valid_rows:
        values[-valid_rows:] = numeric[-valid_rows:]
        sequence_mask[-valid_rows:] = 1.0

    return PaddedMarketSequence(
        values=values,
        sequence_mask=sequence_mask,
        valid_rows=valid_rows,
        valid_ratio=valid_rows / float(size),
    )
