"""Deterministic FIFO research reporting, independent of account average cost.

Every filled sell must be covered by earlier buys or explicit initial lots.
Matching fragments, sell orders and position round trips are separate units.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import math
from typing import Any, Iterable, Mapping


def _decimal(value: Any, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not number.is_finite() or (number <= 0 if positive else number < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return number


def _shares(value: Any) -> int:
    number = _decimal(value, "shares", positive=True)
    if number != number.to_integral_value():
        raise ValueError("shares must be a positive integer")
    return int(number)


def _time(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, datetime.min.time())
    else:
        try:
            result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} requires an ISO date/time") from exc
    # Naive timestamps are exchange-local and are compared on their original
    # wall-clock scale. Mixing naive and offset-aware inputs is rejected below.
    return result


def _number(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Monetary result exceeds finite float range")
    return result


def _symbol(value: Any) -> str:
    result = str(value or "").strip().upper()
    if not result:
        raise ValueError("Every fill/initial lot requires a symbol")
    return result


def _fees(record: Mapping[str, Any], *, initial: bool = False) -> Decimal:
    components = [record.get(key, 0) for key in ("commission", "stamp_duty", "transfer_fee")]
    component_sum = sum((_decimal(value, "fee component") for value in components), Decimal(0))
    initial_default = record.get("remaining_buy_fees", record.get("fees", component_sum))
    total = _decimal(record.get("total_fees", initial_default if initial else component_sum), "total_fees")
    if "total_fees" in record and any(key in record for key in ("commission", "stamp_duty", "transfer_fee")):
        if abs(total - component_sum) > Decimal("0.00000001") * max(Decimal(1), total):
            raise ValueError("total_fees does not match commission/stamp_duty/transfer_fee")
    return total


def _direction(record: Mapping[str, Any]) -> int:
    side = record.get("side")
    direction = record.get("direction")
    if direction is None:
        if side not in {"buy", "sell"}:
            raise ValueError("Filled records require buy/sell side or direction +1/-1")
        return 1 if side == "buy" else -1
    if isinstance(direction, bool) or direction not in {-1, 1}:
        raise ValueError("Filled direction must be +1 or -1")
    if side is not None and side != ("buy" if direction == 1 else "sell"):
        raise ValueError("side and direction disagree")
    return int(direction)


def _rate(values: list[Decimal]) -> float | None:
    return sum(value > 0 for value in values) / len(values) if values else None


def build_fifo_report(
    fills: Iterable[Mapping[str, Any]], initial_lots: Iterable[Mapping[str, Any]] | None = None,
    final_prices: Mapping[str, Any] | None = None, as_of: Any = None,
) -> dict[str, Any]:
    """Build FIFO fragments, sell-order aggregates, flat-to-flat round trips.

    ``initial_lots`` require symbol, shares, entry_price (or price), entry_time
    (or buy_time), and optional lot_id/total_fees. They are pre-window holdings,
    ordered by entry time; they must predate all fills. ``final_prices`` only
    marks remaining lots and never creates synthetic sells. Rejected orders
    (status=blocked/cancelled/rejected) are counted separately and cost zero.
    """
    records = [dict(record) for record in fills]
    initial = [dict(record) for record in (initial_lots or [])]
    seen_ids: set[str] = set()
    rejected: list[dict] = []
    events = []
    times: list[datetime] = []
    for index, record in enumerate(records):
        fill_id = str(record.get("fill_id") or f"input_{index:08d}")
        if fill_id in seen_ids:
            raise ValueError(f"Duplicate fill_id: {fill_id}")
        seen_ids.add(fill_id)
        status = str(record.get("status", "filled"))
        if status in {"blocked", "cancelled", "rejected"}:
            if _fees(record) != 0:
                raise ValueError("Nonfilled records cannot carry fees in this report")
            rejected.append({"fill_id": fill_id, "symbol": _symbol(record.get("symbol")), "status": status, "reason": record.get("reason")})
            continue
        if status != "filled":
            raise ValueError(f"Unsupported fill status: {status}")
        timestamp = _time(record.get("execution_time", record.get("time", record.get("timestamp"))), "execution_time")
        sequence = record.get("sequence", record.get("seq", index))
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise ValueError("sequence/seq must be an integer")
        event = {"fill_id": fill_id, "symbol": _symbol(record.get("symbol")),
                 "shares": _shares(record.get("shares", record.get("quantity"))),
                 "price": _decimal(record.get("execution_price", record.get("price")), "execution_price", positive=True),
                 "fees": _fees(record), "direction": _direction(record), "time": timestamp,
                 "order_id": str(record.get("order_id") or record.get("signal_id") or fill_id),
                 "sequence": sequence, "input_index": index}
        times.append(timestamp)
        events.append(event)
    initial_events = []
    for index, record in enumerate(initial):
        lot_id = str(record.get("lot_id") or record.get("buy_fill_id") or f"initial_{index:08d}")
        if lot_id in seen_ids:
            raise ValueError(f"Duplicate initial lot/fill identifier: {lot_id}")
        seen_ids.add(lot_id)
        timestamp = _time(record.get("entry_time", record.get("buy_time")), "initial entry_time")
        initial_events.append({"fill_id": lot_id, "symbol": _symbol(record.get("symbol")),
                               "shares": _shares(record.get("shares", record.get("remaining_shares"))),
                               "price": _decimal(record.get("entry_price", record.get("price")), "entry_price", positive=True),
                               "fees": _fees(record, initial=True), "time": timestamp, "input_index": index})
        times.append(timestamp)
    end_time = _time(as_of, "as_of") if as_of is not None else None
    if end_time is not None:
        times.append(end_time)
    awareness = {t.tzinfo is not None and t.utcoffset() is not None for t in times}
    if len(awareness) > 1:
        raise ValueError("Do not mix naive exchange timestamps with timezone-aware timestamps")
    events.sort(key=lambda e: (e["time"], e["sequence"], e["input_index"]))
    initial_events.sort(key=lambda e: (e["time"], e["input_index"]))
    if events and any(e["time"] > events[0]["time"] for e in initial_events):
        raise ValueError("Initial lots must exist before the first report fill")
    if end_time is not None and any(t > end_time for t in times):
        raise ValueError("as_of precedes a fill or initial lot")
    prices = {_symbol(key): _decimal(value, "final_price", positive=True) for key, value in (final_prices or {}).items()}
    inventory: dict[str, deque] = defaultdict(deque)
    trips: dict[str, dict] = {}
    trip_counts: dict[str, int] = defaultdict(int)
    completed_trips: list[dict] = []
    fragments: list[dict] = []
    orders: dict[tuple[str, str], dict] = {}
    initial_fees = sum((event["fees"] for event in initial_events), Decimal(0))
    input_fees = sum((event["fees"] for event in events), Decimal(0))

    def add_lot(event: dict, is_initial: bool) -> None:
        symbol = event["symbol"]
        if not inventory[symbol]:
            trip_counts[symbol] += 1
            trips[symbol] = {"round_trip_id": f"{symbol}:{trip_counts[symbol]}", "symbol": symbol,
                             "entry_time": event["time"], "includes_initial_position": is_initial,
                             "net_pnl": Decimal(0), "entry_cost": Decimal(0), "fees": Decimal(0), "shares_sold": 0}
        trips[symbol]["includes_initial_position"] |= is_initial
        inventory[symbol].append({**event, "remaining": event["shares"], "remaining_fees": event["fees"],
                                  "initial": is_initial, "round_trip_id": trips[symbol]["round_trip_id"]})

    for event in initial_events:
        add_lot(event, True)
    for event in events:
        symbol = event["symbol"]
        if event["direction"] > 0:
            add_lot(event, False)
            continue
        if sum(lot["remaining"] for lot in inventory[symbol]) < event["shares"]:
            raise ValueError(f"Unmatched sell {event['fill_id']}: more shares sold than known FIFO inventory for {symbol}")
        remaining, sell_fees_remaining = event["shares"], event["fees"]
        order_key = symbol, event["order_id"]
        order = orders.setdefault(order_key, {"symbol": symbol, "order_id": event["order_id"], "sell_fill_ids": [],
                                             "shares": 0, "net_pnl": Decimal(0), "entry_cost": Decimal(0),
                                             "fees": Decimal(0), "fragment_count": 0})
        order["sell_fill_ids"].append(event["fill_id"])
        while remaining:
            lot = inventory[symbol][0]
            quantity = min(remaining, lot["remaining"])
            buy_fee = lot["remaining_fees"] if quantity == lot["remaining"] else lot["remaining_fees"] * Decimal(quantity) / Decimal(lot["remaining"])
            sell_fee = sell_fees_remaining if quantity == remaining else sell_fees_remaining * Decimal(quantity) / Decimal(remaining)
            entry_value, exit_value = lot["price"] * quantity, event["price"] * quantity
            entry_cost = entry_value + buy_fee
            pnl = exit_value - entry_value - buy_fee - sell_fee
            fragment = {"symbol": symbol, "fragment_id": f"fragment_{len(fragments)+1:08d}",
                        "round_trip_id": lot["round_trip_id"], "buy_fill_id": lot["fill_id"], "sell_fill_id": event["fill_id"],
                        "sell_order_id": event["order_id"], "shares": quantity, "entry_time": lot["time"].isoformat(),
                        "exit_time": event["time"].isoformat(), "entry_price": _number(lot["price"]), "exit_price": _number(event["price"]),
                        "buy_fees": _number(buy_fee), "sell_fees": _number(sell_fee), "entry_cost": _number(entry_cost),
                        "net_pnl": _number(pnl), "net_return": _number(pnl / entry_cost),
                        "holding_days": (event["time"] - lot["time"]).total_seconds() / 86400,
                        "initial_position": lot["initial"]}
            fragments.append(fragment)
            for aggregate in (order, trips[symbol]):
                aggregate["net_pnl"] += pnl
                aggregate["entry_cost"] += entry_cost
                aggregate["fees"] += buy_fee + sell_fee
            order["shares"] += quantity
            order["fragment_count"] += 1
            trips[symbol]["shares_sold"] += quantity
            lot["remaining"] -= quantity
            lot["remaining_fees"] -= buy_fee
            remaining -= quantity
            sell_fees_remaining -= sell_fee
            if not lot["remaining"]:
                inventory[symbol].popleft()
        if not inventory[symbol]:
            trip = trips.pop(symbol)
            completed_trips.append({**trip, "entry_time": trip["entry_time"].isoformat(), "exit_time": event["time"].isoformat(),
                                    "net_pnl": _number(trip["net_pnl"]), "entry_cost": _number(trip["entry_cost"]),
                                    "fees": _number(trip["fees"]), "net_return": _number(trip["net_pnl"] / trip["entry_cost"])})

    open_lots = []
    for symbol, lots in sorted(inventory.items()):
        for lot in lots:
            price = prices.get(symbol)
            entry_cost = lot["price"] * lot["remaining"] + lot["remaining_fees"]
            open_lots.append({"symbol": symbol, "lot_id": lot["fill_id"], "round_trip_id": lot["round_trip_id"],
                              "entry_time": lot["time"].isoformat(), "entry_price": _number(lot["price"]),
                              "remaining_shares": lot["remaining"], "remaining_buy_fees": _number(lot["remaining_fees"]),
                              "remaining_cost": _number(entry_cost), "initial_position": lot["initial"],
                              "mark_price": _number(price) if price is not None else None,
                              "market_value": _number(price * lot["remaining"]) if price is not None else None,
                              "unrealized_pnl": _number(price * lot["remaining"] - entry_cost) if price is not None else None,
                              "as_of": end_time.isoformat() if end_time is not None else None})
    sell_orders = [{**order, "net_pnl": _number(order["net_pnl"]), "entry_cost": _number(order["entry_cost"]),
                    "fees": _number(order["fees"]), "net_return": _number(order["net_pnl"] / order["entry_cost"])} for order in orders.values()]
    fragment_pnls = [Decimal(str(item["net_pnl"])) for item in fragments]
    gross_profit = sum((value for value in fragment_pnls if value > 0), Decimal(0))
    gross_loss = -sum((value for value in fragment_pnls if value < 0), Decimal(0))
    allocated = sum((Decimal(str(item["buy_fees"])) + Decimal(str(item["sell_fees"])) for item in fragments), Decimal(0))
    remaining_fees = sum((lot["remaining_fees"] for lots in inventory.values() for lot in lots), Decimal(0))
    total_fees = input_fees + initial_fees
    unrealized = sum((Decimal(str(item["unrealized_pnl"])) for item in open_lots if item["unrealized_pnl"] is not None), Decimal(0)) if all(item["unrealized_pnl"] is not None for item in open_lots) else None
    return {"schema_version": 1, "closed_trades": fragments, "open_lots": open_lots,
            "sell_orders": sell_orders, "round_trips": completed_trips, "rejected_orders": rejected,
            "summary": {
                "filled_buy_count": sum(event["direction"] == 1 for event in events),
                "filled_sell_count": sum(event["direction"] == -1 for event in events),
                "rejected_order_count": len(rejected), "blocked_order_count": sum(x["status"] == "blocked" for x in rejected),
                "closed_fragment_count": len(fragments), "sell_order_count": len(sell_orders),
                "completed_round_trip_count": len(completed_trips), "open_round_trip_count": len(trips),
                "fragment_win_rate": _rate(fragment_pnls),
                "sell_order_win_rate": _rate([Decimal(str(x["net_pnl"])) for x in sell_orders]),
                "round_trip_win_rate": _rate([Decimal(str(x["net_pnl"])) for x in completed_trips]),
                "realized_net_pnl": _number(sum(fragment_pnls, Decimal(0))),
                "mean_fragment_net_pnl": _number(sum(fragment_pnls, Decimal(0)) / len(fragment_pnls)) if fragment_pnls else None,
                "mean_fragment_net_return": sum(row["net_return"] for row in fragments) / len(fragments) if fragments else None,
                "mean_fragment_holding_days": sum(row["holding_days"] for row in fragments) / len(fragments) if fragments else None,
                "best_fragment_net_return": max((row["net_return"] for row in fragments), default=None),
                "worst_fragment_net_return": min((row["net_return"] for row in fragments), default=None),
                "unrealized_net_pnl": _number(unrealized) if unrealized is not None else None,
                "gross_profit": _number(gross_profit), "gross_loss": _number(gross_loss),
                "profit_factor": _number(gross_profit / gross_loss) if gross_loss else None,
                "profit_factor_status": "defined" if gross_loss else "no_losing_fragments",
                "input_fill_fees": _number(input_fees), "initial_lot_fees": _number(initial_fees),
                "total_fees": _number(total_fees), "allocated_fees": _number(allocated), "open_buy_fees": _number(remaining_fees),
                "fee_reconciliation_difference": _number(total_fees - allocated - remaining_fees),
                "open_lot_count": len(open_lots), "open_shares": sum(x["remaining_shares"] for x in open_lots),
            },
            "assumptions": [
                "FIFO is a reporting cost convention; it does not replace the account's average-cost realized PnL.",
                "closed_trades and fragment_win_rate count matched buy-lot/sell-fill fragments, not complete position trades.",
                "sell_orders group by symbol and order_id, otherwise signal_id, otherwise fill_id; rates count these sell orders.",
                "round_trips end only when a symbol's inventory returns to zero; initial-position trips are explicitly flagged.",
                "Equal-time fills use integer sequence/seq when supplied, otherwise original input order; no implicit buy-before-sell sorting.",
                "Open lots remain unrealized and are never forcibly closed. Marked unrealized PnL includes remaining buy fees but no hypothetical exit fees.",
                "Fees are allocated proportionally to matched shares. total_fees includes explicitly supplied initial-lot fees.",
                "profit_factor uses net fragment profits divided by net fragment losses; no losses returns null, never infinity.",
            ]}
