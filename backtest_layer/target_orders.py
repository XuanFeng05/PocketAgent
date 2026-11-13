"""Translate explicit target weights into the shared executor's capacity units."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Mapping

import numpy as np

from backtest_layer.account import PortfolioAccount
from backtest_layer.actions import PortfolioAction
from backtest_layer.execution import AshareExecutor
from backtest_layer.market import MarketSnapshot


def execute_target_weights(account: PortfolioAccount, market: MarketSnapshot,
                           target_weights: Mapping[str, float], executor: AshareExecutor) -> dict:
    unknown = set(target_weights) - set(market.symbols)
    if unknown:
        raise ValueError(f"Unknown target symbols: {sorted(unknown)}")
    weights = {symbol: float(target_weights.get(symbol, 0)) for symbol in market.symbols}
    if any(isinstance(v, bool) for v in target_weights.values()) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in weights.values()) or sum(weights.values()) > 1 + 1e-10:
        raise ValueError("Target weights must be finite, nonnegative and sum to at most one.")
    account.start_trading_day(market.decision_time.date())
    account.mark_to_market(market.symbols, market.execution_prices)
    opening_nav = account.net_asset_value
    lot = executor.config.lot_size
    directions = np.zeros(len(market.symbols), dtype=np.int8)
    sizes = np.zeros(len(market.symbols), dtype=float)
    intents = []
    for i, symbol in enumerate(market.symbols):
        price = float(market.execution_prices[i])
        position = account.position(symbol)
        # The executor's position cap constrains new purchases; it is not an
        # instruction to liquidate a held winner after ordinary price drift.
        target_value = opening_nav * weights[symbol]
        difference = position.market_value - target_value
        if price <= 0 or difference < price * lot - 1e-8:
            continue
        requested = min(position.total_shares, int(math.floor(difference / price / lot + 1e-10)) * lot)
        max_shares = executor._floor_lot(min(position.sellable_shares, executor._liquidity_shares(market, i, price)))
        directions[i] = -1
        sizes[i] = min(1.0, requested / max_shares) if max_shares else 1.0
        intents.append({"symbol": symbol, "side": "sell", "target_weight": weights[symbol],
                        "requested_shares": requested, "quantity_semantics": "target_weight_converted_to_capacity_fraction"})
    sells = executor.execute(account, market, PortfolioAction(directions, sizes))
    directions[:] = 0
    sizes[:] = 0
    post_sell_nav = account.net_asset_value
    for i, symbol in enumerate(market.symbols):
        price = float(market.execution_prices[i])
        position = account.position(symbol)
        target_value = post_sell_nav * min(weights[symbol], executor.config.max_position_ratio)
        needed = max(0., target_value - position.market_value)
        if weights[symbol] <= 0 or needed <= 1e-8:
            continue
        # Keep an unavailable target as a rejected intent for the audit trail.
        headroom = max(0., post_sell_nav * executor.config.max_position_ratio - position.market_value)
        capacity = min(headroom, executor._liquidity_cap(market, i))
        directions[i] = 1
        sizes[i] = min(1., needed / capacity) if capacity > 0 else 1.
        intents.append({"symbol": symbol, "side": "buy", "target_weight": weights[symbol],
                        "requested_value": needed, "quantity_semantics": "target_weight_converted_to_capacity_fraction"})
    buys = executor.execute(account, market, PortfolioAction(directions, sizes))
    fills = []
    for fill in (*sells.fills, *buys.fills):
        record = asdict(fill)
        record.update(direction=int(fill.direction), side="buy" if fill.direction > 0 else "sell",
                      total_fees=fill.total_fees, target_weight=weights[fill.symbol])
        fills.append(record)
    return {"fills": fills, "intents": intents, "nav_before": opening_nav,
            "nav_after": account.net_asset_value, "turnover_value": sells.turnover_value + buys.turnover_value}
