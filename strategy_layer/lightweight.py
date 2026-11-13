"""Small causal CPU strategies; no neural-network runtime or external service."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from strategy_layer.base import AccountView, BarView, OrderIntent


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _finite(value: Any, name: str, minimum: float, maximum: float, *, strictly_minimum: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value > maximum or (value <= minimum if strictly_minimum else value < minimum):
        raise ValueError(f"{name} is outside its finite supported range")
    return float(value)


def _known(context, key: str) -> float | None:
    value = context.get(key)
    return float(value) if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value) else None


@dataclass
class ContextMomentumStrategy:
    requires_context = True
    lookback: int = 5
    min_index_return: float = 0.0
    min_industry_return: float = 0.0
    require_industry: int = 1
    name: str = "context_momentum"
    _missing_entry_observations: int = field(default=0, init=False)

    def __post_init__(self):
        self.lookback = _integer(self.lookback, "lookback", 1, 2000)
        self.require_industry = _integer(self.require_industry, "require_industry", 0, 1)
        self.min_index_return = _finite(self.min_index_return, "min_index_return", -.99, 5)
        self.min_industry_return = _finite(self.min_industry_return, "min_industry_return", -.99, 5)

    def on_close(self, market: BarView, account: AccountView) -> OrderIntent | None:
        if len(market.closes) <= self.lookback:
            return None
        stock_return = market.closes[-1] / market.closes[-self.lookback-1] - 1
        context = market.context
        index_return = _known(context, "context_index_ret_5bar")
        industry_return = _known(context, "context_industry_ret_5bar")
        index_known = bool(context.get("context_index_available")) and index_return is not None
        industry_known = bool(context.get("context_industry_available")) and industry_return is not None
        # Missing external data blocks entries, never disables an otherwise
        # legal exit based on the observed stock trend or a known weak market.
        weak = stock_return <= 0 or (index_known and index_return < self.min_index_return)
        if self.require_industry and industry_known and industry_return < self.min_industry_return:
            weak = True
        if account.total_shares > 0 and weak:
            return OrderIntent(-1, reason="stock_or_known_context_weak")
        can_enter = index_known and index_return >= self.min_index_return
        if self.require_industry:
            can_enter = can_enter and industry_known and industry_return >= self.min_industry_return
        if account.total_shares == 0 and stock_return > 0 and (not index_known or (self.require_industry and not industry_known)):
            self._missing_entry_observations += 1
        if account.total_shares == 0 and stock_return > 0 and can_enter:
            return OrderIntent(1, reason="stock_and_visible_context_positive")
        return None

    def diagnostics(self) -> dict:
        return {"context_horizon": "5 observed daily bars", "missing_context": "blocks new entries; observed exits remain possible",
                "missing_context_entry_observations": self._missing_entry_observations}


@dataclass
class ConstrainedRandomStrategy:
    seed: int = 2026
    name: str = "constrained_random"
    _rng: Any = field(init=False, repr=False)

    def __post_init__(self):
        self.seed = _integer(self.seed, "seed", 0, 2**32-1)
        self._rng = np.random.default_rng(self.seed)

    def on_close(self, market: BarView, account: AccountView) -> OrderIntent | None:
        allowed = [0]
        context = market.context
        lot_size = int(context.get("lot_size", 100))
        minimum_commission = float(context.get("minimum_commission", 5))
        if account.total_shares == 0 and market.closes and account.cash >= market.closes[-1]*lot_size+minimum_commission and context.get("can_buy_at_close", True):
            allowed.append(1)
        if account.sellable_shares >= lot_size and context.get("can_sell_at_close", True):
            allowed.append(-1)
        choice = int(self._rng.choice(allowed))
        return OrderIntent(choice, reason="seeded_random_feasible_at_signal") if choice else None

    def diagnostics(self) -> dict:
        return {"seed": self.seed, "mask_scope": "known closing market/account state; next opening executor remains authoritative"}


@dataclass
class RidgeForecastStrategy:
    lags: int = 5
    train_window: int = 60
    min_samples: int = 20
    horizon: int = 1
    alpha: float = 1.0
    threshold: float = 0.001
    refit_interval: int = 5
    name: str = "ridge_forecast"
    _coef: Any = field(default=None, init=False, repr=False)
    _mean: Any = field(default=None, init=False, repr=False)
    _scale: Any = field(default=None, init=False, repr=False)
    _target_mean: float = field(default=0.0, init=False, repr=False)
    _fit_count: int = field(default=0, init=False)
    _last_fit_observations: int = field(default=0, init=False)
    _training_samples: int = field(default=0, init=False)
    _last_prediction: float | None = field(default=None, init=False)
    _fit_time: str | None = field(default=None, init=False)

    def __post_init__(self):
        self.lags = _integer(self.lags, "lags", 1, 20)
        self.train_window = _integer(self.train_window, "train_window", 10, 512)
        self.min_samples = _integer(self.min_samples, "min_samples", 5, self.train_window)
        self.horizon = _integer(self.horizon, "horizon", 1, 20)
        self.refit_interval = _integer(self.refit_interval, "refit_interval", 1, 100)
        self.alpha = _finite(self.alpha, "alpha", 0, 1e6, strictly_minimum=True)
        self.threshold = _finite(self.threshold, "threshold", 0, 1)

    def _fit(self, closes: np.ndarray, time: str) -> None:
        n = len(closes)
        # At close n-1, a sample originating at i is eligible only when its
        # complete label i+horizon has already arrived. Never fit on a target
        # computed beyond the supplied observed prefix.
        last_origin = n-1-self.horizon
        first_origin = max(self.lags, last_origin-self.train_window+1)
        if last_origin-first_origin+1 < self.min_samples:
            return
        returns = closes[1:] / closes[:-1] - 1
        origins = range(first_origin, last_origin+1)
        x = np.asarray([returns[i-self.lags:i] for i in origins], dtype=np.float64)
        y = np.asarray([closes[i+self.horizon]/closes[i]-1 for i in origins], dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Ridge training input must be finite")
        self._mean = x.mean(axis=0)
        self._scale = x.std(axis=0)
        self._scale = np.where(self._scale > 1e-12, self._scale, 1.0)
        normalized = (x-self._mean)/self._scale
        self._target_mean = float(y.mean())
        self._coef = np.linalg.solve(normalized.T@normalized+self.alpha*np.eye(self.lags),
                                     normalized.T@(y-self._target_mean))
        self._fit_count += 1
        self._last_fit_observations = n
        self._training_samples = len(y)
        self._fit_time = time

    def on_close(self, market: BarView, account: AccountView) -> OrderIntent | None:
        closes = np.asarray(market.closes, dtype=np.float64)
        if not np.isfinite(closes).all() or np.any(closes <= 0):
            raise ValueError("Ridge observations must be finite positive closes")
        if self._coef is None or len(closes)-self._last_fit_observations >= self.refit_interval:
            self._fit(closes, market.time.isoformat())
        if self._coef is None:
            self._last_prediction = None
            return None
        returns = closes[1:] / closes[:-1] - 1
        feature = returns[-self.lags:]
        prediction = float(((feature-self._mean)/self._scale)@self._coef+self._target_mean)
        if not math.isfinite(prediction):
            raise ValueError("Ridge prediction is not finite")
        self._last_prediction = prediction
        if prediction > self.threshold and account.total_shares == 0:
            return OrderIntent(1, reason="causal_ridge_forecast_positive")
        if prediction <= self.threshold and account.total_shares > 0:
            return OrderIntent(-1, reason="causal_ridge_forecast_not_positive")
        return None

    def diagnostics(self) -> dict:
        return {"model": "numpy_ridge", "deterministic": True, "fit_count": self._fit_count,
                "training_samples": self._training_samples, "last_fit_observations": self._last_fit_observations,
                "last_fit_time": self._fit_time, "last_prediction": self._last_prediction,
                "label_horizon_bars": self.horizon,
                "label_definition": "future close / current close - 1; not directly attainable next-open trading return",
                "update_rule": "online refit on matured historical labels only, including during validation; hyperparameters remain frozen",
                "scaler_fit": "eligible historical training samples only", "gpu_required": False}
