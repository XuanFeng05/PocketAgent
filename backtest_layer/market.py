from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class MarketSnapshot:
    """Execution-visible facts, independent of neural-network observations.

    liquidity_amount is an explicitly documented liquidity estimate available
    at decision_time, never the eventual full-day turnover of an opening fill.
    AgentMarketStep remains structurally compatible with this contract.
    """

    decision_time: datetime
    symbols: tuple[str, ...]
    execution_prices: np.ndarray
    liquidity_amount: np.ndarray
    limit_reference_close: np.ndarray
    limit_pct: np.ndarray
    active_mask: np.ndarray
    is_tradeable: np.ndarray
    market_can_buy: np.ndarray
    market_can_sell: np.ndarray
    stage: str = "open_auction"
    runtime_contract: tuple[str, ...] = ()
