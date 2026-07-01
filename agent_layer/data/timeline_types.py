from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimelineKey:
    decision_time: pd.Timestamp
    stage: str
    active_symbols: int
