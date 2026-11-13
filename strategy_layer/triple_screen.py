"""Versioned specification for the long-only A-share technical Triple Screen."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class TripleScreenSettings:
    version: str = "triple_screen_pre_v1"
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    force_span: int = 2
    warmup_weeks: int = 100
    cost_bps: float = 0.0

    def __post_init__(self):
        if self.version not in {"triple_screen_pre_v1", "triple_screen_us_pre_v1"}:
            raise ValueError("Unsupported Triple Screen rule version.")
        for field in ("macd_fast", "macd_slow", "macd_signal", "force_span", "warmup_weeks"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 2000:
                raise ValueError(f"{field} must be an integer from 1 to 2000.")
        if self.macd_fast >= self.macd_slow or self.warmup_weeks < 2:
            raise ValueError("Require macd_fast < macd_slow and warmup_weeks >= 2.")
        if isinstance(self.cost_bps, bool) or not isinstance(self.cost_bps, Real) or not math.isfinite(self.cost_bps) or not 0 <= self.cost_bps < 10000:
            raise ValueError("cost_bps must be finite, >= 0 and < 10000.")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_us(self) -> bool:
        return self.version == "triple_screen_us_pre_v1"


def resolve_triple_settings(value: TripleScreenSettings | dict | None = None) -> TripleScreenSettings:
    if isinstance(value, TripleScreenSettings):
        return value
    if value is not None and not isinstance(value, dict):
        raise ValueError("Triple Screen settings must be an object.")
    try:
        return TripleScreenSettings(**(value or {}))
    except TypeError as exc:
        raise ValueError(f"Invalid Triple Screen settings: {exc}") from exc
