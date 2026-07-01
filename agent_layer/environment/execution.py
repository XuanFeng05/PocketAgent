from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from agent_layer.actions import PortfolioAction, TradeDirection
from agent_layer.data import AgentMarketStep
from agent_layer.portfolio import PortfolioAccount


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 1_000_000.0
    lot_size: int = 100
    max_position_ratio: float = 0.20
    bar_participation_rate: float = 0.10
    auction_participation_rate: float = 0.02
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    base_slippage_rate: float = 0.0002
    impact_coefficient: float = 0.001
    maximum_slippage_rate: float = 0.005
    holding_bar_cap: int = 1000

    def __post_init__(self) -> None:
        if self.initial_cash <= 0 or self.lot_size <= 0 or self.holding_bar_cap <= 0:
            raise ValueError("Cash, lot size, and holding-bar cap must be positive.")
        for name in (
            "max_position_ratio",
            "bar_participation_rate",
            "auction_participation_rate",
        ):
            value = float(getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        for name in (
            "commission_rate",
            "minimum_commission",
            "stamp_duty_rate",
            "transfer_fee_rate",
            "base_slippage_rate",
            "impact_coefficient",
            "maximum_slippage_rate",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative.")


@dataclass(frozen=True)
class TradeFill:
    symbol: str
    direction: TradeDirection
    requested_size: float
    shares: int
    reference_price: float
    execution_price: float
    gross_value: float
    commission: float
    stamp_duty: float
    transfer_fee: float
    status: str
    reason: str | None = None

    @property
    def total_fees(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee


@dataclass(frozen=True)
class ExecutionResult:
    fills: tuple[TradeFill, ...]
    nav_before: float
    nav_after: float
    turnover_value: float

    @property
    def executed_fills(self) -> tuple[TradeFill, ...]:
        return tuple(fill for fill in self.fills if fill.status == "filled")


class AshareExecutor:
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def execute(
        self,
        account: PortfolioAccount,
        market: AgentMarketStep,
        action: PortfolioAction,
    ) -> ExecutionResult:
        if len(action.directions) != len(market.symbols):
            raise ValueError("Action size must match the fixed Agent universe.")
        account.start_trading_day(market.decision_time.date())
        account.mark_to_market(market.symbols, market.execution_prices)
        nav_before = account.net_asset_value
        fills: list[TradeFill] = []

        for index, symbol in enumerate(market.symbols):
            if action.directions[index] != TradeDirection.SELL:
                continue
            fills.append(
                self._execute_sell(
                    account,
                    market,
                    index=index,
                    symbol=symbol,
                    size=float(action.sizes[index]),
                )
            )

        buy_requests: list[tuple[int, str, float, float]] = []
        nav_after_sells = account.net_asset_value
        for index, symbol in enumerate(market.symbols):
            if action.directions[index] != TradeDirection.BUY:
                continue
            reason = self._blocked_reason(market, index=index, direction=TradeDirection.BUY)
            if reason:
                fills.append(self._blocked_fill(symbol, TradeDirection.BUY, float(action.sizes[index]), market, index, reason))
                continue
            size = float(action.sizes[index])
            price = float(market.execution_prices[index])
            position = account.position(symbol)
            headroom = max(0.0, nav_after_sells * self.config.max_position_ratio - position.market_value)
            liquidity_cap = self._liquidity_cap(market, index)
            requested = size * min(headroom, liquidity_cap)
            if requested <= 0 or price <= 0:
                fills.append(self._blocked_fill(symbol, TradeDirection.BUY, size, market, index, "no_buying_capacity"))
                continue
            buy_requests.append((index, symbol, size, requested))

        total_requested = sum(item[3] for item in buy_requests)
        cash_scale = min(1.0, account.cash / total_requested) if total_requested > 0 else 0.0
        for index, symbol, size, requested in buy_requests:
            fills.append(
                self._execute_buy(
                    account,
                    market,
                    index=index,
                    symbol=symbol,
                    size=size,
                    gross_budget=requested * cash_scale,
                )
            )

        # Positions are valued at the observable market price, not their slipped fill price.
        account.mark_to_market(market.symbols, market.execution_prices)
        turnover = sum(fill.gross_value for fill in fills if fill.status == "filled")
        return ExecutionResult(
            fills=tuple(fills),
            nav_before=nav_before,
            nav_after=account.net_asset_value,
            turnover_value=float(turnover),
        )

    def runtime_state(
        self,
        account: PortfolioAccount,
        market: AgentMarketStep,
    ) -> np.ndarray:
        nav = max(account.net_asset_value, 1e-12)
        values = np.zeros((len(market.symbols), len(market.runtime_contract)), dtype=np.float32)
        name_to_index = {name: index for index, name in enumerate(market.runtime_contract)}
        for row_index, symbol in enumerate(market.symbols):
            position = account.position(symbol)
            price = float(market.execution_prices[row_index])
            if not np.isfinite(price) or price <= 0:
                price = position.last_price
            position_value = position.total_shares * price
            sellable_value = position.sellable_shares * price
            unrealized = (price - position.average_cost) * position.total_shares
            liquidity_cap = self._liquidity_cap(market, row_index)
            headroom = max(0.0, nav * self.config.max_position_ratio - position_value)
            max_buy = min(account.cash, headroom, liquidity_cap) if market.market_can_buy[row_index] else 0.0
            max_sell = min(sellable_value, liquidity_cap) if market.market_can_sell[row_index] else 0.0
            can_buy = bool(max_buy >= price * self.config.lot_size and market.active_mask[row_index])
            can_sell = bool(max_sell > 0 and position.sellable_shares > 0 and market.active_mask[row_index])
            row = {
                "cash_ratio": account.cash / nav,
                "position_ratio": position_value / nav,
                "available_position_ratio": sellable_value / nav,
                "unrealized_pnl_ratio": unrealized / nav,
                "holding_bars_norm": min(position.holding_bars / max(1, self.config.holding_bar_cap), 1.0),
                "one_lot_nav_ratio": price * self.config.lot_size / nav if price > 0 else 0.0,
                "max_buy_value_ratio": max_buy / nav,
                "max_sell_value_ratio": max_sell / nav,
                "can_buy": float(can_buy),
                "can_sell": float(can_sell),
            }
            for name, value in row.items():
                if name in name_to_index:
                    values[row_index, name_to_index[name]] = float(value)
        return values

    def _execute_sell(
        self,
        account: PortfolioAccount,
        market: AgentMarketStep,
        *,
        index: int,
        symbol: str,
        size: float,
    ) -> TradeFill:
        reason = self._blocked_reason(market, index=index, direction=TradeDirection.SELL)
        position = account.position(symbol)
        if reason:
            return self._blocked_fill(symbol, TradeDirection.SELL, size, market, index, reason)
        if position.sellable_shares <= 0:
            return self._blocked_fill(symbol, TradeDirection.SELL, size, market, index, "t_plus_one_locked")
        reference_price = float(market.execution_prices[index])
        liquidity_shares = self._liquidity_shares(market, index, reference_price)
        requested = int(position.sellable_shares * size)
        if size >= 1.0 - 1e-6:
            shares = min(position.sellable_shares, liquidity_shares)
        else:
            shares = self._floor_lot(min(requested, liquidity_shares))
        if shares <= 0:
            return self._blocked_fill(symbol, TradeDirection.SELL, size, market, index, "below_sellable_lot")
        price = self._execution_price(market, index, TradeDirection.SELL, shares * reference_price)
        gross = shares * price
        commission, stamp, transfer = self._fees(gross, is_sell=True)
        account.apply_sell(symbol, shares=shares, price=price, fees=commission + stamp + transfer)
        return TradeFill(symbol, TradeDirection.SELL, size, shares, reference_price, price, gross, commission, stamp, transfer, "filled")

    def _execute_buy(
        self,
        account: PortfolioAccount,
        market: AgentMarketStep,
        *,
        index: int,
        symbol: str,
        size: float,
        gross_budget: float,
    ) -> TradeFill:
        reference_price = float(market.execution_prices[index])
        estimated_price = self._execution_price(market, index, TradeDirection.BUY, gross_budget)
        shares = self._floor_lot(gross_budget / max(estimated_price, 1e-12))
        while shares > 0:
            gross = shares * estimated_price
            commission, stamp, transfer = self._fees(gross, is_sell=False)
            if gross + commission + transfer <= account.cash + 1e-8:
                break
            shares -= self.config.lot_size
        if shares <= 0:
            return self._blocked_fill(symbol, TradeDirection.BUY, size, market, index, "below_affordable_lot")
        gross = shares * estimated_price
        commission, stamp, transfer = self._fees(gross, is_sell=False)
        account.apply_buy(symbol, shares=shares, price=estimated_price, fees=commission + transfer)
        return TradeFill(symbol, TradeDirection.BUY, size, shares, reference_price, estimated_price, gross, commission, stamp, transfer, "filled")

    def _blocked_reason(self, market: AgentMarketStep, *, index: int, direction: TradeDirection) -> str | None:
        if not market.active_mask[index]:
            return "inactive_symbol"
        if not market.is_tradeable[index]:
            return "not_tradeable"
        if direction == TradeDirection.BUY and not market.market_can_buy[index]:
            return "market_buy_blocked"
        if direction == TradeDirection.SELL and not market.market_can_sell[index]:
            return "market_sell_blocked"
        return None

    def _blocked_fill(
        self,
        symbol: str,
        direction: TradeDirection,
        size: float,
        market: AgentMarketStep,
        index: int,
        reason: str,
    ) -> TradeFill:
        reference = float(market.execution_prices[index])
        return TradeFill(symbol, direction, size, 0, reference, reference, 0.0, 0.0, 0.0, 0.0, "blocked", reason)

    def _participation_rate(self, market: AgentMarketStep) -> float:
        return self.config.auction_participation_rate if market.stage == "open_auction" else self.config.bar_participation_rate

    def _liquidity_cap(self, market: AgentMarketStep, index: int) -> float:
        return max(0.0, float(market.liquidity_amount[index])) * self._participation_rate(market)

    def _liquidity_shares(self, market: AgentMarketStep, index: int, price: float) -> int:
        if price <= 0:
            return 0
        return self._floor_lot(self._liquidity_cap(market, index) / price)

    def _execution_price(
        self,
        market: AgentMarketStep,
        index: int,
        direction: TradeDirection,
        gross_value: float,
    ) -> float:
        reference = float(market.execution_prices[index])
        amount = max(float(market.liquidity_amount[index]), 1e-12)
        participation = max(0.0, gross_value / amount)
        slippage = min(
            self.config.maximum_slippage_rate,
            self.config.base_slippage_rate + self.config.impact_coefficient * math.sqrt(participation),
        )
        raw = reference * (1.0 + slippage if direction == TradeDirection.BUY else 1.0 - slippage)
        lower = float(market.limit_reference_close[index]) * (1.0 - float(market.limit_pct[index]))
        upper = float(market.limit_reference_close[index]) * (1.0 + float(market.limit_pct[index]))
        bounded = min(max(raw, lower), upper)
        return math.ceil(bounded * 100.0) / 100.0 if direction == TradeDirection.BUY else math.floor(bounded * 100.0) / 100.0

    def _fees(self, gross_value: float, *, is_sell: bool) -> tuple[float, float, float]:
        commission = max(self.config.minimum_commission, gross_value * self.config.commission_rate)
        stamp = gross_value * self.config.stamp_duty_rate if is_sell else 0.0
        transfer = gross_value * self.config.transfer_fee_rate
        return commission, stamp, transfer

    def _floor_lot(self, value: float | int) -> int:
        return max(0, int(float(value) // self.config.lot_size) * self.config.lot_size)
