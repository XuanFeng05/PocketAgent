"""Independent per-symbol research, explicitly separate from shared portfolios."""
from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from statistics import mean, median
from typing import Any

import pandas as pd

from backtest_layer.engine import run_comparison
from backtest_layer.execution import ExecutionConfig

DEFAULT_MAX_SYMBOLS = 20
HARD_MAX_SYMBOLS = 100
MAX_INPUT_ROWS = 500_000


def run_universe(
    bars: Any, strategy: str = "sma_trend", parameters: dict | None = None,
    execution_config: ExecutionConfig | dict | None = None,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Run the same rule and executable baselines on each independent account.

    Source-loading failures may be represented by a row carrying symbol and
    source_error. Other malformed data is validated per symbol by the existing
    backtest, so a bad stock cannot silently disappear from the experiment.
    """
    if isinstance(max_symbols, bool) or not isinstance(max_symbols, Integral) or not 1 <= int(max_symbols) <= HARD_MAX_SYMBOLS:
        raise ValueError(f"max_symbols must be an integer from 1 to {HARD_MAX_SYMBOLS}.")
    frame = bars.copy(deep=True) if isinstance(bars, pd.DataFrame) else pd.DataFrame(bars)
    if frame.empty or "symbol" not in frame:
        raise ValueError("Universe input needs nonempty symbol-labelled rows.")
    if len(frame) > MAX_INPUT_ROWS:
        raise ValueError(f"Universe input exceeds {MAX_INPUT_ROWS} rows; narrow the interval or symbols.")
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    if frame["symbol"].isin(["", "NAN", "NONE", "<NA>"]).any():
        raise ValueError("Every universe row requires a valid symbol identifier.")
    symbols = tuple(sorted(frame["symbol"].unique()))
    if len(symbols) > int(max_symbols):
        raise ValueError(f"Universe has {len(symbols)} symbols, over max_symbols={max_symbols}; no symbols were dropped.")
    execution = execution_config if isinstance(execution_config, ExecutionConfig) else ExecutionConfig(**dict(execution_config or {}))
    rows = []
    for index, symbol in enumerate(symbols, start=1):
        if cancel_check and cancel_check():
            raise InterruptedError("Universe research cancelled; completed symbol callbacks were delivered.")
        part = frame.loc[frame["symbol"].eq(symbol)].copy()
        row = {"symbol": symbol, "status": "running"}
        try:
            if "source_error" in part:
                errors = [str(value) for value in part["source_error"].dropna() if str(value).strip()]
                if errors:
                    raise ValueError("Source loading failed: " + "; ".join(dict.fromkeys(errors)))
                part = part.drop(columns="source_error")
            comparison = run_comparison(part, strategy, parameters, execution, symbol=symbol, cancel_check=cancel_check)
            report = comparison["strategy"]
            row.update(status="completed", result=comparison, metrics=report["metrics"],
                       excess_return=comparison["excess_return_vs_buy_hold"],
                       start=report["start"], end=report["end"], bars=report["bars"])
        except InterruptedError:
            raise
        except Exception as exc:
            row.update(status="failed", error=str(exc), result=None, metrics=None, excess_return=None)
        rows.append(row)
        if progress_callback:
            progress_callback({"event": "symbol_completed", "symbol": symbol,
                               "completed": index, "total": len(symbols), "symbol_result": row})
    valid = [row for row in rows if row["status"] == "completed"]
    returns = [row["metrics"]["total_return"] for row in valid]
    excess = [row["excess_return"] for row in valid]
    drawdowns = [row["metrics"]["maximum_drawdown"] for row in valid]
    chosen = valid[0]["result"] if valid else {"strategy": None, "baselines": {}}
    return {
        "mode": "universe", "status": "completed" if len(valid) == len(rows) else "partial" if valid else "failed",
        "strategy_name": strategy, "parameters": dict(parameters or {}),
        "symbols": list(symbols), "symbol_results": rows,
        "strategy": chosen["strategy"], "baselines": chosen["baselines"],
        "display_symbol": valid[0]["symbol"] if valid else None,
        "selection_scope": "selected_symbol_not_aggregate",
        "summary": {
            "total_symbols": len(rows), "completed_symbols": len(valid), "failed_symbols": len(rows)-len(valid),
            "mean_return": mean(returns) if returns else None, "median_return": median(returns) if returns else None,
            "minimum_return": min(returns) if returns else None, "maximum_return": max(returns) if returns else None,
            "positive_return_symbol_ratio": mean(value > 0 for value in returns) if returns else None,
            "mean_excess_return": mean(excess) if excess else None, "median_excess_return": median(excess) if excess else None,
            "win_rate_vs_buy_hold": mean(value > 0 for value in excess) if excess else None,
            "mean_maximum_drawdown": mean(drawdowns) if drawdowns else None,
            "worst_maximum_drawdown": max(drawdowns) if drawdowns else None,
            "periods_aligned": len({(row["start"], row["end"]) for row in valid}) <= 1 if valid else None,
            "return_distribution": [{"symbol": row["symbol"], "start": row["start"], "end": row["end"],
                                      "bars": row["bars"], "total_return": row["metrics"]["total_return"],
                                      "excess_return": row["excess_return"],
                                      "maximum_drawdown": row["metrics"]["maximum_drawdown"],
                                      "executed_trades": row["metrics"]["executed_trades"]} for row in valid],
        },
        "assumptions": [
            "Every symbol has its own full initial cash account; these are independent research trials, not positions in one portfolio.",
            "Means, medians and win rates summarize symbol outcomes; no average NAV curve or continuous portfolio return is manufactured.",
            "The top-level strategy and baselines display the first successful symbol in deterministic lexical order, not an aggregate or winner.",
            "All requested symbols, including source/data/strategy failures, remain in symbol_results; a budget excess rejects the request without truncation.",
            "Each symbol uses its observed input interval. periods_aligned flags differing boundaries; unequal periods are not directly comparable return horizons.",
            "The user-supplied universe must have an appropriate historical selection basis; this report does not certify freedom from survivorship bias.",
        ],
    }
