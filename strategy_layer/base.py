from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Mapping, Any
from types import MappingProxyType


@dataclass(frozen=True)
class AccountView:
    cash: float
    net_asset_value: float
    total_shares: int
    sellable_shares: int


@dataclass(frozen=True)
class BarView:
    symbol: str
    time: datetime
    closes: tuple[float, ...]
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))


@dataclass(frozen=True)
class OrderIntent:
    """Direction plus fraction of executable capacity, not target weight.

    A BUY fraction scales available cash / position headroom / liquidity.
    A SELL fraction scales sellable shares / liquidity. The executor owns
    fees, lot rounding and the final feasible quantity.
    """

    direction: int
    capacity_fraction: float = 1.0
    reason: str = "strategy_signal"

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError("Intent direction must be -1, 0 or 1.")
        if not 0 <= self.capacity_fraction <= 1:
            raise ValueError("Intent capacity_fraction must be in [0, 1].")


class Strategy(Protocol):
    name: str

    def on_close(self, market: BarView, account: AccountView) -> OrderIntent | None: ...
