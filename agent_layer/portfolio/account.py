from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np


@dataclass
class Position:
    total_shares: int = 0
    sellable_shares: int = 0
    locked_shares: int = 0
    average_cost: float = 0.0
    last_price: float = 0.0
    holding_bars: int = 0

    @property
    def market_value(self) -> float:
        return float(self.total_shares) * float(self.last_price)


class PortfolioAccount:
    def __init__(self, universe: Iterable[str], *, initial_cash: float) -> None:
        symbols = tuple(str(symbol).upper() for symbol in universe)
        if not symbols:
            raise ValueError("Portfolio universe cannot be empty.")
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive.")
        self.universe = symbols
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions = {symbol: Position() for symbol in symbols}
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.current_trading_date: date | None = None

    @property
    def net_asset_value(self) -> float:
        return self.cash + sum(position.market_value for position in self.positions.values())

    def position(self, symbol: str) -> Position:
        return self.positions[str(symbol).upper()]

    def mark_to_market(self, symbols: Iterable[str], prices: np.ndarray) -> None:
        for symbol, price in zip(symbols, prices):
            value = float(price)
            if np.isfinite(value) and value > 0:
                self.position(symbol).last_price = value

    def start_trading_day(self, trading_date: date) -> None:
        if self.current_trading_date is not None and trading_date <= self.current_trading_date:
            return
        if self.current_trading_date is not None:
            for position in self.positions.values():
                position.sellable_shares += position.locked_shares
                position.locked_shares = 0
        self.current_trading_date = trading_date

    def advance_holding_bars(self) -> None:
        for position in self.positions.values():
            if position.total_shares > 0:
                position.holding_bars += 1

    def apply_buy(
        self,
        symbol: str,
        *,
        shares: int,
        price: float,
        fees: float,
    ) -> None:
        if shares <= 0:
            return
        gross = float(shares) * float(price)
        total_cost = gross + float(fees)
        if total_cost > self.cash + 1e-8:
            raise ValueError("Buy cost exceeds available cash.")
        position = self.position(symbol)
        old_basis = position.average_cost * position.total_shares
        position.total_shares += int(shares)
        position.locked_shares += int(shares)
        position.average_cost = (old_basis + total_cost) / position.total_shares
        position.last_price = float(price)
        self.cash -= total_cost
        self.total_fees += float(fees)

    def apply_sell(
        self,
        symbol: str,
        *,
        shares: int,
        price: float,
        fees: float,
    ) -> None:
        position = self.position(symbol)
        if shares <= 0:
            return
        if shares > position.sellable_shares or shares > position.total_shares:
            raise ValueError("Sell quantity exceeds sellable shares.")
        gross = float(shares) * float(price)
        self.cash += gross - float(fees)
        self.realized_pnl += gross - float(fees) - position.average_cost * shares
        self.total_fees += float(fees)
        position.total_shares -= int(shares)
        position.sellable_shares -= int(shares)
        position.last_price = float(price)
        if position.total_shares == 0:
            position.average_cost = 0.0
            position.holding_bars = 0

    def reset(self) -> None:
        self.cash = self.initial_cash
        self.positions = {symbol: Position() for symbol in self.universe}
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.current_trading_date = None
