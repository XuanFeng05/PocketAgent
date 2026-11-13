"""CPU snapshots separating execution time from next-observation valuation."""
from typing import Any


def account_snapshot(account: Any, time: Any) -> dict[str, Any]:
    nav = float(account.net_asset_value)
    accounts = {}
    for symbol, p in account.positions.items():
        value = float(p.market_value)
        accounts[symbol] = {
            "cash": float(account.cash), "position_shares": int(p.total_shares),
            "sellable_shares": int(p.sellable_shares), "locked_shares": int(p.locked_shares),
            "average_cost": float(p.average_cost), "last_price": float(p.last_price),
            "position_value": value, "total_asset": nav,
            "position_ratio": value / nav if nav > 0 else 0.0,
            "total_fees": float(account.total_fees), "realized_pnl": float(account.realized_pnl),
        }
    return {"time": time, "nav": nav, "accounts": accounts}


def execution_valuation(info: dict[str, Any]) -> dict[str, Any]:
    snapshot = info.get("execution_snapshot")
    if not isinstance(snapshot, dict) or "time" not in snapshot or "nav" not in snapshot:
        raise ValueError("Environment must provide an execution-time valuation snapshot")
    return snapshot
