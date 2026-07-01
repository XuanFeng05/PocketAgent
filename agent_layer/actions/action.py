from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class TradeDirection(IntEnum):
    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass(frozen=True)
class PortfolioAction:
    directions: np.ndarray
    sizes: np.ndarray

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions, dtype=np.int8)
        sizes = np.asarray(self.sizes, dtype=np.float32)
        if directions.ndim != 1 or sizes.ndim != 1 or directions.shape != sizes.shape:
            raise ValueError("Action directions and sizes must be equal-length vectors.")
        if not np.isfinite(sizes).all():
            raise ValueError("Action sizes must be finite.")
        if not np.isin(directions, [-1, 0, 1]).all():
            raise ValueError("Action directions must be SELL(-1), HOLD(0), or BUY(1).")
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "sizes", np.clip(sizes, 0.0, 1.0))

    @classmethod
    def hold(cls, symbol_count: int) -> "PortfolioAction":
        return cls(
            directions=np.zeros(symbol_count, dtype=np.int8),
            sizes=np.zeros(symbol_count, dtype=np.float32),
        )
