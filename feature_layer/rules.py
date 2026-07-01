from __future__ import annotations


def price_limit_pct_for_symbol(symbol: str, *, is_st: bool = False) -> float:
    """
    Return the effective A-share price limit ratio for one trading day.

    This is a trading-rule helper, not a model hyperparameter. ST status is a
    dated market fact and overrides the board-level limit for that day.
    """
    if bool(is_st):
        return 0.05
    normalized = str(symbol or "").upper().split(".", 1)[0]
    if normalized.startswith(("688", "689", "300", "301")):
        return 0.20
    if normalized.startswith(("8", "4", "920")):
        return 0.30
    return 0.10
