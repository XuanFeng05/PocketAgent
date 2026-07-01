from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def portfolio_metrics(
    daily_nav: Iterable[float],
    *,
    initial_nav: float | None = None,
    turnover_value: float = 0.0,
    total_fees: float = 0.0,
) -> dict[str, float]:
    values = list(daily_nav)
    if initial_nav is not None:
        values.insert(0, float(initial_nav))
    nav = np.asarray(values, dtype=np.float64)
    if nav.ndim != 1 or nav.size < 2 or not np.isfinite(nav).all() or np.any(nav <= 0):
        raise ValueError("At least two positive finite daily NAV values are required.")
    returns = nav[1:] / nav[:-1] - 1.0
    total_return = nav[-1] / nav[0] - 1.0
    periods = max(1, len(returns))
    annualized_return = (nav[-1] / nav[0]) ** (252.0 / periods) - 1.0
    volatility = float(returns.std(ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.0
    mean_return = float(returns.mean())
    return_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = mean_return / return_std * math.sqrt(252.0) if return_std > 0 else 0.0
    sortino = mean_return / downside_std * math.sqrt(252.0) if downside_std > 0 else 0.0
    running_peak = np.maximum.accumulate(nav)
    drawdown = nav / running_peak - 1.0
    maximum_drawdown = float(-drawdown.min())
    calmar = annualized_return / maximum_drawdown if maximum_drawdown > 0 else 0.0
    average_nav = float(nav.mean())
    return {
        "initial_nav": float(nav[0]),
        "final_nav": float(nav[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": volatility,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "maximum_drawdown": maximum_drawdown,
        "calmar": float(calmar),
        "positive_day_ratio": float((returns > 0).mean()),
        "return_observations": float(len(returns)),
        "turnover_ratio": float(turnover_value / average_nav) if average_nav > 0 else 0.0,
        "total_fees": float(total_fees),
    }
