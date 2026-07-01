from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from agent_layer.actions import PortfolioAction
from agent_layer.environment import AshareTradingEnv, ExecutionConfig, SingleSymbolTradingEnv
from agent_layer.models import (
    MultiFrequencyMAPPO,
    market_steps_to_tensors,
    observation_state_to_tensors,
)
from evaluation_layer.metrics import portfolio_metrics


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    daily_nav: tuple[dict[str, Any], ...]
    orders: int
    blocked_orders: int
    steps: int

    def payload(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "daily_nav": list(self.daily_nav),
            "orders": self.orders,
            "blocked_orders": self.blocked_orders,
            "steps": self.steps,
        }


@torch.inference_mode()
def evaluate_policy(
    environment: AshareTradingEnv,
    model: MultiFrequencyMAPPO,
    *,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvaluationResult:
    target = torch.device(device)
    model.to(target)
    model.eval()
    observation, reset_info = environment.reset()
    initial_nav = float(reset_info["net_asset_value"])
    daily_values: dict[pd.Timestamp, float] = {
        observation.market.decision_time.normalize(): float(reset_info["net_asset_value"])
    }
    turnover = 0.0
    orders = 0
    blocked_orders = 0
    steps = 0
    total_steps = len(environment.keys)
    market_state_cache: dict[tuple[pd.Timestamp, str], torch.Tensor] = {}

    while observation is not None:
        if cancel_check and cancel_check():
            raise InterruptedError("Policy evaluation cancellation requested.")
        market_key = (
            observation.market.decision_time,
            observation.market.stage,
        )
        if market_key not in market_state_cache:
            market_chunk = [
                observation.market,
                *environment.prefetched_markets(),
            ]
            market_tensors = market_steps_to_tensors(market_chunk, device=target)
            encoded = model.encode_market(**market_tensors).detach()
            for index, market in enumerate(market_chunk):
                market_state_cache[(market.decision_time, market.stage)] = encoded[
                    index : index + 1
                ]
        state_tensors = observation_state_to_tensors(observation, device=target)
        action = model.act_from_market_state(
            market_state=market_state_cache.pop(market_key),
            deterministic=deterministic,
            **state_tensors,
        )
        next_observation, _, terminated, truncated, info = environment.step(
            PortfolioAction(
                action.directions.squeeze(0).detach().cpu().numpy(),
                action.sizes.squeeze(0).detach().cpu().numpy(),
            )
        )
        execution = info.get("execution")
        if execution is not None:
            turnover += float(execution.turnover_value)
            orders += len(execution.executed_fills)
            blocked_orders += sum(fill.status == "blocked" for fill in execution.fills)
        daily_values[observation.market.decision_time.normalize()] = float(
            info["net_asset_value"]
        )
        steps += 1
        if progress_callback:
            progress_callback(steps, total_steps)
        observation = next_observation
        if terminated or truncated:
            break

    ordered = tuple(
        {"date": str(date.date()), "nav": nav}
        for date, nav in sorted(daily_values.items())
    )
    metrics = portfolio_metrics(
        [item["nav"] for item in ordered],
        initial_nav=initial_nav,
        turnover_value=turnover,
        total_fees=environment.account.total_fees,
    )
    return EvaluationResult(metrics, ordered, orders, blocked_orders, steps)

# --- Persistent Evaluation Replay Runner -------------------------------------------------

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from agent_layer.experiments import load_agent_checkpoint
from agent_layer.data import AgentTimelineLoader, AgentMarketStep, open_agent_timeline_loader


@torch.inference_mode()
def run_policy_replay_to_files(
    *,
    run_dir: str | Path,
    model_path: str | Path,
    dataset_dir: str | Path,
    symbol: str,
    start: str,
    end: str,
    execution_config: ExecutionConfig,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
    stages: tuple[str, ...] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a single-symbol historical policy replay and persist event files.

    The runner is intentionally file based so the Visualization layer can poll a
    partially completed evaluation run while the backend evaluates as fast as it can.
    """

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    summary_path = output_dir / "summary.json"
    events_path.write_text("", encoding="utf-8")

    target = torch.device(device)
    model, checkpoint_payload = load_agent_checkpoint(model_path, map_location="cpu")
    loader = open_agent_timeline_loader(
        dataset_dir,
        universe=(str(symbol).upper(),),
        frequencies=model.frequencies,
        validate_store=False,
        use_market_cache=True,
        use_decision_cache=True,
    )
    expected_schema = str(checkpoint_payload.get("schema_hash") or "")
    if expected_schema and loader.schema_hash != expected_schema:
        raise ValueError(
            f"Feature schema mismatch: model {expected_schema} != dataset {loader.schema_hash}"
        )

    model.to(target)
    model.eval()
    environment = SingleSymbolTradingEnv(
        loader,
        start=start,
        end=end,
        stages=stages,
        execution_config=execution_config,
    )
    daily_bars_by_date = _daily_bars_by_date(loader, symbol=str(symbol).upper(), start=start, end=end)
    decision_bars_by_id = _decision_bars_by_id(loader, symbol=str(symbol).upper(), start=start, end=end)
    observation, reset_info = environment.reset()
    initial_nav = float(reset_info["net_asset_value"])
    target_symbol = str(symbol).upper()
    total_steps = len(environment.keys)
    market_state_cache: dict[tuple[pd.Timestamp, str], torch.Tensor] = {}
    nav_values: list[float] = [initial_nav]
    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    event_count = 0
    total_actions = 0
    orders = 0
    blocked_orders = 0
    blocked_buys = 0
    blocked_sells = 0
    turnover = 0.0
    peak_nav = initial_nav
    started = perf_counter()
    last_progress = 0.0

    with events_path.open("a", encoding="utf-8") as handle:
        while observation is not None:
            if cancel_check and cancel_check():
                raise InterruptedError("Evaluation cancellation requested.")
            market = observation.market
            market_key = (market.decision_time, market.stage)
            if market_key not in market_state_cache:
                market_chunk = [market, *environment.prefetched_markets()]
                market_tensors = market_steps_to_tensors(market_chunk, device=target)
                encoded = model.encode_market(**market_tensors).detach()
                for index, prefetched_market in enumerate(market_chunk):
                    market_state_cache[(prefetched_market.decision_time, prefetched_market.stage)] = encoded[
                        index : index + 1
                    ]
            state_tensors = observation_state_to_tensors(observation, device=target)
            evaluation = model.forward_from_market_state(
                market_state=market_state_cache.pop(market_key),
                **state_tensors,
            )
            action = model._action_from_evaluation(evaluation, deterministic=deterministic)
            direction_probs = torch.softmax(evaluation.direction_logits.detach(), dim=-1).squeeze(0).cpu().numpy()
            action_payload = PortfolioAction(
                action.directions.squeeze(0).detach().cpu().numpy(),
                action.sizes.squeeze(0).detach().cpu().numpy(),
            )
            index = _symbol_index(market, target_symbol)
            raw_direction = int(action_payload.directions[index]) if index is not None else 0
            raw_size = float(action_payload.sizes[index]) if index is not None else 0.0
            raw_action = _action_name(raw_direction)
            is_last_bar = event_count >= total_steps - 1
            synthetic_fills: list[dict[str, Any]] = []
            execution_action = action_payload
            if is_last_bar:
                execution_action = PortfolioAction.hold(len(market.symbols))
                if index is not None and raw_direction != 0:
                    synthetic_fills.append(_blocked_action_payload(
                        symbol=target_symbol,
                        side=raw_action,
                        size=raw_size,
                        market=market,
                        index=index,
                        reason="last_bar",
                    ))
            if raw_action != "hold":
                total_actions += 1
            next_observation, reward, terminated, truncated, info = environment.step(execution_action)
            execution = info.get("execution")
            fill_events: list[dict[str, Any]] = list(synthetic_fills)
            if execution is not None:
                turnover += float(execution.turnover_value)
                for fill in execution.fills:
                    fill_payload = _fill_payload(fill, environment.account.net_asset_value)
                    fill_events.append(fill_payload)
                    if fill.status == "filled":
                        orders += 1
                        trades.append({
                            "seq": event_count + 1,
                            "time": _iso(market.decision_time),
                            **fill_payload,
                        })
                    elif fill.status == "blocked":
                        blocked_orders += 1
                        if fill_payload.get("side") == "buy":
                            blocked_buys += 1
                        elif fill_payload.get("side") == "sell":
                            blocked_sells += 1
            for fill_payload in synthetic_fills:
                blocked_orders += 1
                if fill_payload.get("side") == "buy":
                    blocked_buys += 1
                elif fill_payload.get("side") == "sell":
                    blocked_sells += 1
            nav = float(info["net_asset_value"])
            nav_values.append(nav)
            peak_nav = max(peak_nav, nav)
            drawdown = 1.0 - nav / peak_nav if peak_nav > 0 else 0.0
            event_count += 1
            execution_summary = _execution_summary(
                fill_events,
                raw_action=raw_action,
                turnover=float(execution.turnover_value) if execution is not None else 0.0,
            )
            event = {
                "seq": event_count,
                "kind": "step",
                "time": _iso(market.decision_time),
                "stage": market.stage,
                "symbol": target_symbol,
                "bar": _event_bar(
                    market,
                    index,
                    decision_bars_by_id=decision_bars_by_id,
                ),
                "decision": {
                    "action": raw_action,
                    "raw_action": raw_action,
                    "final_action": execution_summary["final_action"],
                    "direction": raw_direction,
                    "size": raw_size,
                    "prob_sell": float(direction_probs[index, 0]) if index is not None else None,
                    "prob_hold": float(direction_probs[index, 1]) if index is not None else None,
                    "prob_buy": float(direction_probs[index, 2]) if index is not None else None,
                    "probabilities": {
                        "sell": float(direction_probs[index, 0]) if index is not None else None,
                        "hold": float(direction_probs[index, 1]) if index is not None else None,
                        "buy": float(direction_probs[index, 2]) if index is not None else None,
                    },
                    "value": float(action.value.squeeze().detach().cpu().item()) if action.value.numel() else None,
                },
                "execution": {
                    "fills": fill_events,
                    **execution_summary,
                },
                "account": _account_payload(environment.account, target_symbol),
                "performance": {
                    "initial_nav": initial_nav,
                    "nav": nav,
                    "return": nav / initial_nav - 1.0,
                    "drawdown": drawdown,
                    "reward": float(reward),
                    "total_fees": float(environment.account.total_fees),
                    "total_actions": total_actions,
                    "orders": orders,
                    "total_trades": orders,
                    "blocked_orders": blocked_orders,
                    "blocked_actions": blocked_orders,
                    "blocked_buys": blocked_buys,
                    "blocked_sells": blocked_sells,
                },
                "features": _feature_payload(market, index),
                "model_input": _model_input_payload(observation, index),
            }
            actions.append({
                "seq": event_count,
                "time": event["time"],
                "raw_action": event["decision"]["raw_action"],
                "final_action": event["decision"]["final_action"],
                "size": event["decision"]["size"],
                "executed": event["execution"]["executed"],
                "blocked": event["execution"]["blocked"],
            })
            equity_curve.append({
                "seq": event_count,
                "time": event["time"],
                "nav": nav,
                "return": event["performance"]["return"],
                "drawdown": drawdown,
            })
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            if event_count % 25 == 0:
                handle.flush()
            now_perf = perf_counter()
            if progress_callback and (event_count <= 1 or now_perf - last_progress >= 1.0 or event_count == total_steps):
                progress_callback({
                    "completed": event_count,
                    "total": total_steps,
                    "progress": event_count / max(1, total_steps),
                    "nav": nav,
                    "return": event["performance"]["return"],
                    "drawdown": drawdown,
                    "orders": orders,
                    "blocked_orders": blocked_orders,
                    "elapsed_seconds": now_perf - started,
                })
                last_progress = now_perf
            observation = next_observation
            if terminated or truncated:
                break
        handle.flush()

    summary = _summary_payload(
        nav_values=nav_values,
        initial_nav=initial_nav,
        turnover=turnover,
        total_fees=environment.account.total_fees,
        total_actions=total_actions,
        orders=orders,
        blocked_orders=blocked_orders,
        blocked_buys=blocked_buys,
        blocked_sells=blocked_sells,
        bars=daily_bars_by_date,
        equity_curve=equity_curve,
        trades=trades,
        actions=actions,
        steps=event_count,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def _daily_bars_by_date(
    loader: AgentTimelineLoader,
    *,
    symbol: str,
    start: str,
    end: str,
) -> dict[Any, dict[str, Any]]:
    source = getattr(loader, "_parts_source", None)
    if source is None:
        return {}
    try:
        with source.connect(symbols=(symbol,)) as conn:
            frame = conn.execute(
                "SELECT bar_datetime, adjust, open, high, low, close, volume, amount, pctChg "
                "FROM market_bars WHERE symbol = ? AND freq = 'daily' "
                "AND CAST(bar_datetime AS DATE) >= CAST(? AS DATE) "
                "AND CAST(bar_datetime AS DATE) <= CAST(? AS DATE) "
                "ORDER BY bar_datetime, CASE adjust WHEN 'pre' THEN 0 WHEN 'none' THEN 1 ELSE 2 END",
                [symbol, start, end],
            ).fetchdf()
    except Exception:
        return {}
    result: dict[Any, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        key = pd.Timestamp(row.bar_datetime).date()
        if key in result:
            continue
        result[key] = {
            "datetime": _iso(pd.Timestamp(row.bar_datetime)),
            "adjust": str(row.adjust),
            "open": _safe_float(row.open),
            "high": _safe_float(row.high),
            "low": _safe_float(row.low),
            "close": _safe_float(row.close),
            "volume": _safe_float(row.volume),
            "amount": _safe_float(row.amount),
            "pctChg": _safe_float(row.pctChg),
        }
    return result


def _decision_bars_by_id(
    loader: AgentTimelineLoader,
    *,
    symbol: str,
    start: str,
    end: str,
) -> dict[str, dict[str, Any]]:
    source = getattr(loader, "_parts_source", None)
    if source is None:
        return {}
    try:
        with source.connect(symbols=(symbol,)) as conn:
            frame = conn.execute(
                "SELECT d.decision_id, mb.bar_datetime, mb.adjust, mb.open, mb.high, "
                "mb.low, mb.close, mb.volume, mb.amount, mb.pctChg "
                "FROM decisions d "
                "JOIN market_bars mb ON mb.symbol = d.symbol "
                "AND mb.adjust = d.adjust AND mb.freq = '5min' "
                "AND mb.bar_datetime = d.source_bar_end "
                "WHERE d.symbol = ? "
                "AND d.decision_time >= CAST(? AS TIMESTAMP) "
                "AND d.decision_time <= CAST(? AS TIMESTAMP) "
                "ORDER BY d.decision_time",
                [symbol, start, _inclusive_end(end)],
            ).fetchdf()
    except Exception:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        result[str(row.decision_id)] = {
            "datetime": _iso(pd.Timestamp(row.bar_datetime)),
            "adjust": str(row.adjust),
            "freq": "5min",
            "open": _safe_float(row.open),
            "high": _safe_float(row.high),
            "low": _safe_float(row.low),
            "close": _safe_float(row.close),
            "volume": _safe_float(row.volume),
            "amount": _safe_float(row.amount),
            "pctChg": _safe_float(row.pctChg),
        }
    return result


def _inclusive_end(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp == timestamp.normalize():
        timestamp += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return timestamp


def _summary_payload(
    *,
    nav_values: list[float],
    initial_nav: float,
    turnover: float,
    total_fees: float,
    total_actions: int,
    orders: int,
    blocked_orders: int,
    blocked_buys: int,
    blocked_sells: int,
    bars: dict[Any, dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    steps: int,
) -> dict[str, Any]:
    try:
        metrics = portfolio_metrics(nav_values, initial_nav=None, turnover_value=turnover, total_fees=total_fees)
    except Exception:
        metrics = {
            "initial_nav": initial_nav,
            "final_nav": nav_values[-1] if nav_values else initial_nav,
            "total_return": (nav_values[-1] / initial_nav - 1.0) if nav_values and initial_nav > 0 else 0.0,
            "maximum_drawdown": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "total_fees": total_fees,
        }
    baseline = _buy_hold_baseline(bars, initial_nav)
    return {
        "metrics": metrics,
        "baseline": baseline,
        "total_actions": total_actions,
        "executed_trades": orders,
        "blocked_actions": blocked_orders,
        "blocked_buys": blocked_buys,
        "blocked_sells": blocked_sells,
        "block_rate": blocked_orders / total_actions if total_actions > 0 else 0.0,
        "total_fees": float(total_fees),
        "orders": orders,
        "blocked_orders": blocked_orders,
        "steps": steps,
        "equity_curve": equity_curve,
        "trades": trades,
        "actions": actions,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _buy_hold_baseline(bars: dict[Any, dict[str, Any]], initial_nav: float) -> dict[str, Any]:
    ordered = [item for _, item in sorted(bars.items(), key=lambda pair: str(pair[0]))]
    prices = [float(item.get("close") or 0.0) for item in ordered if float(item.get("close") or 0.0) > 0]
    if len(prices) < 2:
        return {"total_return": None, "final_nav": None}
    total_return = prices[-1] / prices[0] - 1.0
    return {"total_return": total_return, "final_nav": initial_nav * (1.0 + total_return)}


def _symbol_index(market: AgentMarketStep, symbol: str) -> int | None:
    try:
        return list(market.symbols).index(symbol)
    except ValueError:
        return None


def _event_bar(
    market: AgentMarketStep,
    index: int | None,
    *,
    decision_bars_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if index is not None:
        decision_id = market.decision_ids[index]
        if decision_id and decision_id in decision_bars_by_id:
            return decision_bars_by_id[decision_id]
    return _bar_from_market(market, index)


def _bar_from_market(market: AgentMarketStep, index: int | None) -> dict[str, Any]:
    price = float(market.execution_prices[index]) if index is not None else None
    is_open_auction = market.stage == "open_auction"
    return {
        "datetime": _iso(market.decision_time),
        "freq": "decision",
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": 0.0 if is_open_auction else (float(market.liquidity_volume[index]) if index is not None else None),
        "amount": 0.0 if is_open_auction else (float(market.liquidity_amount[index]) if index is not None else None),
    }


def _feature_payload(market: AgentMarketStep, index: int | None, *, limit_per_freq: int = 80) -> dict[str, dict[str, float | None]]:
    if index is None:
        return {}
    payload: dict[str, dict[str, float | None]] = {}
    for freq, values in market.market_sequences.items():
        names = market.feature_names.get(freq, ())
        if not names or values.shape[1] <= 0:
            continue
        latest = values[index, -1]
        payload[freq] = {
            str(name): _safe_float(latest[column_index])
            for column_index, name in enumerate(names[:limit_per_freq])
        }
    return payload


def _model_input_payload(observation: Any, index: int | None) -> dict[str, Any]:
    market = observation.market
    return {
        "market_sequences": _feature_payload(market, index),
        "decision_context": _named_vector_payload(
            market.decision_context_names,
            market.decision_context[index] if index is not None else None,
        ),
        "runtime_state": _named_vector_payload(
            market.runtime_contract,
            observation.runtime_state[index] if index is not None else None,
        ),
        "constraints": {
            "active": bool(market.active_mask[index]) if index is not None else False,
            "can_buy": bool(market.market_can_buy[index]) if index is not None else False,
            "can_sell": bool(market.market_can_sell[index]) if index is not None else False,
            "tradeable": bool(market.is_tradeable[index]) if index is not None else False,
            "limit_up": bool(market.is_limit_up[index]) if index is not None else False,
            "limit_down": bool(market.is_limit_down[index]) if index is not None else False,
            "zero_volume": bool(market.is_zero_volume[index]) if index is not None else False,
        },
    }


def _named_vector_payload(names: tuple[str, ...], values: Any) -> dict[str, float | None]:
    if values is None:
        return {}
    return {
        str(name): _safe_float(values[index])
        for index, name in enumerate(names)
    }


def _fill_payload(fill: Any, nav: float) -> dict[str, Any]:
    amount = float(fill.gross_value)
    return {
        "symbol": str(fill.symbol),
        "side": str(getattr(fill.direction, "name", fill.direction)).lower(),
        "requested_size": _safe_float(fill.requested_size),
        "shares": int(fill.shares),
        "reference_price": _safe_float(fill.reference_price),
        "price": _safe_float(fill.execution_price),
        "amount": amount,
        "commission": _safe_float(fill.commission),
        "stamp_duty": _safe_float(fill.stamp_duty),
        "transfer_fee": _safe_float(fill.transfer_fee),
        "fee": _safe_float(fill.total_fees),
        "slippage": _safe_float(float(fill.execution_price) - float(fill.reference_price)),
        "position_change_ratio": amount / max(float(nav), 1e-12),
        "status": str(fill.status),
        "reason": fill.reason,
    }


def _blocked_action_payload(
    *,
    symbol: str,
    side: str,
    size: float,
    market: AgentMarketStep,
    index: int,
    reason: str,
) -> dict[str, Any]:
    reference = float(market.execution_prices[index])
    return {
        "symbol": symbol,
        "side": side,
        "requested_size": _safe_float(size),
        "shares": 0,
        "reference_price": _safe_float(reference),
        "price": _safe_float(reference),
        "amount": 0.0,
        "commission": 0.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.0,
        "fee": 0.0,
        "slippage": 0.0,
        "position_change_ratio": 0.0,
        "status": "blocked",
        "reason": reason,
    }


def _execution_summary(
    fills: list[dict[str, Any]],
    *,
    raw_action: str,
    turnover: float,
) -> dict[str, Any]:
    filled = next((item for item in fills if item.get("status") == "filled"), None)
    blocked = next((item for item in fills if item.get("status") == "blocked"), None)
    primary = filled or blocked or None
    final_action = str(filled.get("side")) if filled else ("hold" if raw_action == "hold" else f"blocked_{raw_action}")
    return {
        "turnover": float(turnover),
        "executed": filled is not None,
        "blocked": blocked is not None,
        "blocked_reason": blocked.get("reason") if blocked else None,
        "final_action": final_action,
        "side": primary.get("side") if primary else raw_action,
        "shares": int(primary.get("shares") or 0) if primary else 0,
        "price": _safe_float(primary.get("price")) if primary else None,
        "amount": _safe_float(primary.get("amount")) if primary else 0.0,
        "fee": _safe_float(primary.get("fee")) if primary else 0.0,
        "slippage": _safe_float(primary.get("slippage")) if primary else 0.0,
        "position_change_ratio": _safe_float(primary.get("position_change_ratio")) if primary else 0.0,
    }


def _account_payload(account: Any, symbol: str) -> dict[str, Any]:
    position = account.position(symbol)
    nav = float(account.net_asset_value)
    position_value = float(position.market_value)
    return {
        "cash": float(account.cash),
        "position_shares": int(position.total_shares),
        "sellable_shares": int(position.sellable_shares),
        "locked_shares": int(position.locked_shares),
        "average_cost": float(position.average_cost),
        "last_price": float(position.last_price),
        "position_value": position_value,
        "total_asset": nav,
        "position_ratio": position_value / nav if nav > 0 else 0.0,
        "total_fees": float(account.total_fees),
        "realized_pnl": float(account.realized_pnl),
    }


def _action_name(direction: int) -> str:
    return "buy" if direction > 0 else "sell" if direction < 0 else "hold"


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()
