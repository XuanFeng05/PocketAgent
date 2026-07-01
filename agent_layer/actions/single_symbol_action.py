from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from agent_layer.actions.action import PortfolioAction, TradeDirection


class SingleSymbolActionCode(IntEnum):
    HOLD = 0
    BUY_SMALL = 1
    BUY_MEDIUM = 2
    SELL_HALF = 3
    SELL_ALL = 4


@dataclass(frozen=True)
class SingleSymbolAction:
    """User-facing single-stock action contract.

    The current PPO implementation still uses the legacy PortfolioAction tensor
    carrier internally because the old rollout buffer expects direction/size
    arrays.  This class is the public single-stock contract and converts to a
    length-one PortfolioAction at the environment boundary.
    """

    code: SingleSymbolActionCode

    def to_portfolio_action(self) -> PortfolioAction:
        direction, size = single_symbol_code_to_direction_size(self.code)
        return PortfolioAction(
            directions=np.asarray([int(direction)], dtype=np.int8),
            sizes=np.asarray([float(size)], dtype=np.float32),
        )

    @classmethod
    def hold(cls) -> "SingleSymbolAction":
        return cls(SingleSymbolActionCode.HOLD)


def single_symbol_code_to_direction_size(
    code: SingleSymbolActionCode | int,
) -> tuple[TradeDirection, float]:
    value = SingleSymbolActionCode(int(code))
    if value == SingleSymbolActionCode.BUY_SMALL:
        return TradeDirection.BUY, 0.35
    if value == SingleSymbolActionCode.BUY_MEDIUM:
        return TradeDirection.BUY, 0.70
    if value == SingleSymbolActionCode.SELL_HALF:
        return TradeDirection.SELL, 0.50
    if value == SingleSymbolActionCode.SELL_ALL:
        return TradeDirection.SELL, 1.00
    return TradeDirection.HOLD, 0.0


def direction_size_to_single_symbol_code(direction: int, size: float) -> SingleSymbolActionCode:
    if int(direction) > 0:
        return SingleSymbolActionCode.BUY_MEDIUM if float(size) >= 0.5 else SingleSymbolActionCode.BUY_SMALL
    if int(direction) < 0:
        return SingleSymbolActionCode.SELL_ALL if float(size) >= 0.75 else SingleSymbolActionCode.SELL_HALF
    return SingleSymbolActionCode.HOLD
