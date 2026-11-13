"""Pure target-weight selection based only on visible closing histories."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PortfolioSettings:
    strategy: str = "momentum_top_k"
    max_positions: int = 5
    lookback: int = 20
    rebalance_every: int = 5
    max_weight: float = .30
    minimum_momentum: float = 0.0
    require_membership: bool = False

    def __post_init__(self):
        if self.strategy not in {"momentum_top_k", "equal_weight_hold", "cash"}:
            raise ValueError("Unknown portfolio strategy.")
        for name, maximum in [("max_positions", 64), ("lookback", 2000), ("rebalance_every", 1000)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}.")
            object.__setattr__(self, name, int(value))
        if isinstance(self.max_weight, bool) or not math.isfinite(self.max_weight) or not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0,1].")
        if isinstance(self.minimum_momentum, bool) or not math.isfinite(self.minimum_momentum):
            raise ValueError("minimum_momentum must be finite.")
        if not isinstance(self.require_membership, bool):
            raise ValueError("require_membership must be boolean.")


def resolve_portfolio_settings(settings: Mapping[str, Any] | PortfolioSettings | None = None) -> PortfolioSettings:
    if isinstance(settings, PortfolioSettings):
        return settings
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("portfolio settings must be an object")
    try:
        return PortfolioSettings(**dict(settings or {}))
    except TypeError as exc:
        raise ValueError(f"Invalid portfolio settings: {exc}") from exc


def select_target_weights(histories: Mapping[str, Sequence[float]], eligible: Sequence[str],
                          settings: PortfolioSettings) -> tuple[dict[str, float], dict[str, float]]:
    names = sorted(set(eligible))
    scores = {}
    if settings.strategy == "cash":
        return {}, scores
    if settings.strategy == "momentum_top_k":
        for symbol in names:
            values = histories.get(symbol, ())
            if len(values) > settings.lookback:
                score = float(values[-1]) / float(values[-settings.lookback-1]) - 1
                if math.isfinite(score) and score >= settings.minimum_momentum:
                    scores[symbol] = score
        selected = sorted(scores, key=lambda name: (-scores[name], name))[:settings.max_positions]
    else:
        selected = names[:settings.max_positions]
    weight = min(settings.max_weight, 1 / len(selected)) if selected else 0
    return {symbol: weight for symbol in selected}, scores


def portfolio_strategy_catalog() -> list[dict]:
    defaults = asdict(PortfolioSettings())
    return [{"id": "momentum_top_k", "label": "动量前 K 组合", "defaults": defaults,
             "description": "使用此前已知收盘收益排序，按固定周期生成等权目标；共享一个现金账户。"},
            {"id": "equal_weight_hold", "label": "等权买入持有", "description": "首次可见合格股票组成初始篮子，各股票首次成交后持有，不定期再平衡。"},
            {"id": "cash", "label": "保持现金", "description": "共享账户保持空仓。"}]
