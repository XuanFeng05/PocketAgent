from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from feature_layer.indicator_registry import indicator_lookback
from feature_layer.specs import FeatureSpec


RAW_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pctChg",
    "turn",
    "progress",
)


class SnapshotFeatureEngine:
    """Vectorized as-of feature updates over one completed-bar history."""

    def __init__(
        self,
        times: np.ndarray,
        raw: np.ndarray,
        *,
        spec: FeatureSpec,
        freq: str,
    ) -> None:
        self.times = np.asarray(times, dtype=np.int64)
        self.raw = np.asarray(raw, dtype=np.float64)
        self.spec = spec
        self.freq = str(freq)
        self.columns = {
            name: self.raw[:, index] for index, name in enumerate(RAW_COLUMNS)
        }
        self.close = self.columns["close"]
        self.high = self.columns["high"]
        self.low = self.columns["low"]
        self.volume = np.nan_to_num(self.columns["volume"], nan=0.0)
        self.amount = np.nan_to_num(self.columns["amount"], nan=0.0)
        self.pct = _pct_series(
            self.columns["open"], self.close, self.columns["pctChg"]
        )
        previous = np.roll(self.close, 1)
        previous[0] = np.nan
        seeded_previous = _seeded_reference_array(previous, self.columns["open"], self.close)
        self.first_close = _first_finite(self.close)
        self.ranges = _divide_array(self.high - self.low, seeded_previous)
        self._states: dict[str, np.ndarray] = {}
        self.active = [
            item for item in spec.indicators if item.enabled and self.freq in item.frequencies
        ]
        for indicator in self.active:
            params = indicator.params
            prefix = indicator.id
            if indicator.kind == "macd":
                fast = _ewm(self.close, 2.0 / (params["fast"] + 1.0))
                slow = _ewm(self.close, 2.0 / (params["slow"] + 1.0))
                dif = fast - slow
                self._states[f"{prefix}:fast"] = fast
                self._states[f"{prefix}:slow"] = slow
                self._states[f"{prefix}:dea"] = _ewm(
                    dif, 2.0 / (params["signal"] + 1.0)
                )
            elif indicator.kind == "kd":
                rsv = _rolling_rsv(self.high, self.low, self.close, params["lookback"])
                k = _ewm(rsv, 1.0 / params["smooth_k"])
                self._states[f"{prefix}:k"] = k
                self._states[f"{prefix}:d"] = _ewm(k, 1.0 / params["smooth_d"])
            elif indicator.kind == "efi":
                force = np.zeros(len(self.close), dtype=np.float64)
                if len(force) > 1:
                    force[1:] = (self.close[1:] - self.close[:-1]) * self.volume[1:]
                self._states[f"{prefix}:force"] = force
                self._states[f"{prefix}:efi2"] = _ewm(
                    force, 2.0 / (params["fast"] + 1.0)
                )
                self._states[f"{prefix}:efi13"] = _ewm(
                    force, 2.0 / (params["slow"] + 1.0)
                )
            else:
                self._states[f"{prefix}:fast"] = _ewm(
                    self.close, 2.0 / (params["fast"] + 1.0)
                )
                self._states[f"{prefix}:slow"] = _ewm(
                    self.close, 2.0 / (params["slow"] + 1.0)
                )

    def build(
        self,
        stable_ends: Sequence[object],
        snapshots: Sequence[Mapping[str, object]],
        *,
        feature_names: Sequence[str],
    ) -> list[dict[str, float]]:
        if not snapshots:
            return []
        ends = np.asarray(
            [pd.Timestamp(value).value if value is not None and not pd.isna(value) else np.iinfo(np.int64).min for value in stable_ends],
            dtype=np.int64,
        )
        indices = np.searchsorted(self.times, ends, side="right") - 1
        current = np.asarray(
            [[_number(row.get(name)) for name in RAW_COLUMNS] for row in snapshots],
            dtype=np.float64,
        )
        current_columns = {
            name: current[:, index] for index, name in enumerate(RAW_COLUMNS)
        }
        open_ = current_columns["open"]
        high = current_columns["high"]
        low = current_columns["low"]
        close = current_columns["close"]
        volume = np.nan_to_num(current_columns["volume"], nan=0.0)
        amount = np.nan_to_num(current_columns["amount"], nan=0.0)
        previous_close = self._take(self.close, indices)
        reference_close = _seeded_reference_array(previous_close, open_, close)
        calculated_pct = _divide_array(close, previous_close) - 1.0
        first_pct = _divide_array(close, reference_close) - 1.0
        supplied_pct = current_columns["pctChg"]
        pct = np.where(np.isfinite(supplied_pct), supplied_pct, calculated_pct)
        pct = np.where(
            np.isfinite(previous_close) & (previous_close != 0), pct, first_pct
        )

        result: dict[str, np.ndarray] = {
            "pctChg": pct,
            "log_ret": np.log(_divide_array(close, reference_close)),
            "open_close_ret": _divide_array(close, open_) - 1.0,
            "gap_ret": _divide_array(open_, reference_close) - 1.0,
            "high_low_range": _divide_array(high - low, reference_close),
            "body_range": _divide_array(np.abs(close - open_), reference_close),
            "close_position": _divide_array_with_fill(close - low, high - low, fill=0.5),
            "upper_shadow": _divide_array(high - np.maximum(open_, close), reference_close),
            "lower_shadow": _divide_array(np.minimum(open_, close) - low, reference_close),
        }
        for window in (3, 5, 10, 20):
            reference = self._take(self.close, indices - window + 1)
            fallback = np.where(np.isfinite(self.first_close) & (self.first_close != 0), self.first_close, close)
            reference = np.where(np.isfinite(reference) & (reference != 0), reference, fallback)
            result[f"ret_{window}"] = _divide_array(close, reference) - 1.0

        result["volatility_5"] = self._combined_std(self.pct, indices, 4, pct)
        result["volatility_20"] = self._combined_std(self.pct, indices, 19, pct)
        result["range_mean_20"] = self._combined_mean(
            self.ranges, indices, 19, result["high_low_range"]
        )
        volume_mean = self._window_mean(self.volume, indices, 20)
        amount_mean = self._window_mean(self.amount, indices, 20)
        volume_mean = np.where(np.isfinite(volume_mean) & (volume_mean != 0), volume_mean, volume)
        amount_mean = np.where(np.isfinite(amount_mean) & (amount_mean != 0), amount_mean, amount)
        result["volume_ratio_20"] = _divide_array(volume, volume_mean)
        result["amount_ratio_20"] = _divide_array(amount, amount_mean)
        turn = self.columns["turn"]
        turn_lag = self._take(turn, indices)
        turn_mean = self._window_mean(turn, indices, 20)
        result["turn_lag1"] = turn_lag
        result["turn_ma20"] = turn_mean
        result["turn_z20"] = _divide_array(
            turn_lag - turn_mean, self._window_std(turn, indices, 20)
        )

        for indicator in self.active:
            params = indicator.params
            prefix = indicator.id
            if indicator.kind == "macd":
                fast = self._ewm_update(
                    self._states[f"{prefix}:fast"], indices, close,
                    2.0 / (params["fast"] + 1.0),
                )
                slow = self._ewm_update(
                    self._states[f"{prefix}:slow"], indices, close,
                    2.0 / (params["slow"] + 1.0),
                )
                dif = fast - slow
                dea = self._ewm_update(
                    self._states[f"{prefix}:dea"], indices, dif,
                    2.0 / (params["signal"] + 1.0),
                )
                result[f"{prefix}_dif_pct"] = _divide_array(dif, close)
                result[f"{prefix}_dea_pct"] = _divide_array(dea, close)
                result[f"{prefix}_hist_pct"] = _divide_array(dif - dea, close)
            elif indicator.kind == "kd":
                prior_low = self._window_extreme(self.low, indices, params["lookback"] - 1, "min")
                prior_high = self._window_extreme(self.high, indices, params["lookback"] - 1, "max")
                prior_low = np.where(np.isfinite(prior_low), prior_low, low)
                prior_high = np.where(np.isfinite(prior_high), prior_high, high)
                lower_bound = np.minimum(prior_low, low)
                upper_bound = np.maximum(prior_high, high)
                rsv = np.clip(
                    _divide_array_with_fill(close - lower_bound, upper_bound - lower_bound, fill=0.5),
                    0.0, 1.0,
                )
                k = self._ewm_update(
                    self._states[f"{prefix}:k"], indices, rsv,
                    1.0 / params["smooth_k"],
                )
                d = self._ewm_update(
                    self._states[f"{prefix}:d"], indices, k,
                    1.0 / params["smooth_d"],
                )
                result[f"{prefix}_k_norm"] = k
                result[f"{prefix}_d_norm"] = d
                result[f"{prefix}_diff_norm"] = k - d
            elif indicator.kind == "efi":
                force = (close - reference_close) * volume
                baseline = self._window_mean(
                    np.abs(self._states[f"{prefix}:force"]), indices, params["baseline"]
                )
                efi2 = self._ewm_update(
                    self._states[f"{prefix}:efi2"], indices, force,
                    2.0 / (params["fast"] + 1.0),
                )
                efi13 = self._ewm_update(
                    self._states[f"{prefix}:efi13"], indices, force,
                    2.0 / (params["slow"] + 1.0),
                )
                result[f"{prefix}2_norm"] = _divide_array(efi2, baseline)
                result[f"{prefix}13_norm"] = _divide_array(efi13, baseline)
            else:
                fast_state = self._states[f"{prefix}:fast"]
                slow_state = self._states[f"{prefix}:slow"]
                fast = self._ewm_update(
                    fast_state, indices, close, 2.0 / (params["fast"] + 1.0)
                )
                slow = self._ewm_update(
                    slow_state, indices, close, 2.0 / (params["slow"] + 1.0)
                )
                lower = np.minimum(fast, slow)
                upper = np.maximum(fast, slow)
                midpoint = (fast + slow) / 2.0
                prior_midpoint = (
                    self._take(fast_state, indices) + self._take(slow_state, indices)
                ) / 2.0
                channel = f"ema_channel_{prefix}"
                result[f"{channel}_position"] = _divide_array_with_fill(close - lower, upper - lower, fill=0.5)
                result[f"{channel}_gap"] = _divide_array(close, midpoint) - 1.0
                result[f"{channel}_width"] = _divide_array_with_fill(upper - lower, midpoint, fill=0.0)
                result[f"{channel}_slope"] = _divide_array_with_fill(midpoint, prior_midpoint, fill=1.0) - 1.0

        longest = max([20, *(indicator_lookback(item) for item in self.active)])
        result["history_coverage"] = np.clip((indices + 2) / float(longest), 0.0, 1.0)
        window = max(1, int(self.spec.sequence_windows.get(self.freq, 1)))
        result["sequence_valid_ratio"] = np.clip((indices + 2) / float(window), 0.0, 1.0)
        result["progress"] = current_columns["progress"]

        clips = {field.name: field.clip for field in self.spec.market_fields}
        outputs: list[dict[str, float]] = []
        for row_index in range(len(snapshots)):
            row: dict[str, float] = {}
            for name in feature_names:
                values = result.get(str(name))
                value = float(values[row_index]) if values is not None else 0.0
                if not np.isfinite(value):
                    value = 0.0
                bounds = clips.get(str(name))
                if bounds is not None:
                    value = float(np.clip(value, bounds[0], bounds[1]))
                row[str(name)] = value
            outputs.append(row)
        return outputs

    @staticmethod
    def _take(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        output = np.full(len(indices), np.nan, dtype=np.float64)
        valid = (indices >= 0) & (indices < len(values))
        output[valid] = values[indices[valid]]
        return output

    def _ewm_update(
        self, states: np.ndarray, indices: np.ndarray, current: np.ndarray, alpha: float
    ) -> np.ndarray:
        previous = self._take(states, indices)
        return np.where(np.isfinite(previous), alpha * current + (1.0 - alpha) * previous, current)

    def _window_values(self, values: np.ndarray, indices: np.ndarray, window: int) -> np.ndarray:
        width = max(1, int(window))
        offsets = np.arange(width - 1, -1, -1)
        positions = indices[:, None] - offsets[None, :]
        output = np.full(positions.shape, np.nan, dtype=np.float64)
        valid = (positions >= 0) & (positions < len(values))
        output[valid] = values[positions[valid]]
        return output

    def _window_mean(self, values: np.ndarray, indices: np.ndarray, window: int) -> np.ndarray:
        return _row_nanmean(self._window_values(values, indices, window))

    def _window_std(self, values: np.ndarray, indices: np.ndarray, window: int) -> np.ndarray:
        return _row_nanstd(self._window_values(values, indices, window))

    def _combined_mean(
        self, values: np.ndarray, indices: np.ndarray, prior_window: int, current: np.ndarray
    ) -> np.ndarray:
        combined = np.column_stack([self._window_values(values, indices, prior_window), current])
        return _row_nanmean(combined)

    def _combined_std(
        self, values: np.ndarray, indices: np.ndarray, prior_window: int, current: np.ndarray
    ) -> np.ndarray:
        combined = np.column_stack([self._window_values(values, indices, prior_window), current])
        return _row_nanstd(combined)

    def _window_extreme(
        self, values: np.ndarray, indices: np.ndarray, window: int, kind: str
    ) -> np.ndarray:
        selected = self._window_values(values, indices, window)
        finite = np.isfinite(selected)
        fill = np.inf if kind == "min" else -np.inf
        reduced = (
            np.min(np.where(finite, selected, fill), axis=1)
            if kind == "min"
            else np.max(np.where(finite, selected, fill), axis=1)
        )
        reduced[~finite.any(axis=1)] = np.nan
        return reduced


def build_snapshot_feature_row(
    history: np.ndarray,
    snapshot: Mapping[str, object],
    *,
    spec: FeatureSpec,
    freq: str,
    feature_names: Sequence[str],
) -> dict[str, float]:
    """Calculate only the final, partially formed higher-frequency feature row."""
    prior = np.asarray(history, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != len(RAW_COLUMNS):
        prior = np.empty((0, len(RAW_COLUMNS)), dtype=np.float64)
    current = np.asarray([_number(snapshot.get(name)) for name in RAW_COLUMNS], dtype=np.float64)
    raw = np.vstack([prior, current])
    columns = {name: raw[:, index] for index, name in enumerate(RAW_COLUMNS)}
    open_ = columns["open"]
    high = columns["high"]
    low = columns["low"]
    close = columns["close"]
    volume = np.nan_to_num(columns["volume"], nan=0.0)
    amount = np.nan_to_num(columns["amount"], nan=0.0)
    turn = columns["turn"]
    n = len(raw)
    previous_close = close[-2] if n >= 2 else np.nan
    current_close = close[-1]
    current_open = open_[-1]
    reference_close = _seeded_reference_scalar(previous_close, current_open, current_close)
    current_pct = _divide(current_close, previous_close) - 1.0
    first_pct = _divide(current_close, reference_close) - 1.0
    supplied_pct = columns["pctChg"][-1]
    pct = supplied_pct if np.isfinite(supplied_pct) else current_pct
    if not np.isfinite(previous_close) or previous_close == 0:
        pct = first_pct

    result: dict[str, float] = {
        "pctChg": pct,
        "log_ret": np.log(_divide(current_close, reference_close)),
        "open_close_ret": _divide(current_close, current_open) - 1.0,
        "gap_ret": _divide(current_open, reference_close) - 1.0,
        "high_low_range": _divide(high[-1] - low[-1], reference_close),
        "body_range": _divide(abs(current_close - current_open), reference_close),
        "close_position": _divide_with_fill(current_close - low[-1], high[-1] - low[-1], fill=0.5),
        "upper_shadow": _divide(high[-1] - max(current_open, current_close), reference_close),
        "lower_shadow": _divide(min(current_open, current_close) - low[-1], reference_close),
    }
    first_close = _first_finite(close)
    for window in (3, 5, 10, 20):
        reference = close[-(window + 1)] if n > window else np.nan
        if not np.isfinite(reference) or reference == 0:
            reference = first_close if np.isfinite(first_close) and first_close != 0 else current_close
        result[f"ret_{window}"] = _divide(current_close, reference) - 1.0

    pct_series = _pct_series(open_, close, columns["pctChg"])
    previous_series = np.roll(close, 1)
    previous_series[0] = np.nan
    range_series = _divide_array(high - low, _seeded_reference_array(previous_series, open_, close))
    result["volatility_5"] = _sample_std(pct_series[-5:])
    result["volatility_20"] = _sample_std(pct_series[-20:])
    result["range_mean_20"] = _nanmean(range_series[-20:])
    volume_mean = _nanmean(volume[:-1][-20:])
    amount_mean = _nanmean(amount[:-1][-20:])
    if not np.isfinite(volume_mean) or volume_mean == 0:
        volume_mean = volume[-1]
    if not np.isfinite(amount_mean) or amount_mean == 0:
        amount_mean = amount[-1]
    result["volume_ratio_20"] = _divide(volume[-1], volume_mean)
    result["amount_ratio_20"] = _divide(amount[-1], amount_mean)

    prior_turn = turn[:-1][-20:]
    result["turn_lag1"] = turn[-2] if n >= 2 else np.nan
    result["turn_ma20"] = _nanmean(prior_turn)
    result["turn_z20"] = _divide(
        result["turn_lag1"] - result["turn_ma20"],
        _sample_std(prior_turn),
    )

    active = [item for item in spec.indicators if item.enabled and freq in item.frequencies]
    for indicator in active:
        params = indicator.params
        prefix = indicator.id
        if indicator.kind == "macd":
            fast = _ewm(close, 2.0 / (params["fast"] + 1.0))
            slow = _ewm(close, 2.0 / (params["slow"] + 1.0))
            dif = fast - slow
            dea = _ewm(dif, 2.0 / (params["signal"] + 1.0))
            result[f"{prefix}_dif_pct"] = _divide(dif[-1], current_close)
            result[f"{prefix}_dea_pct"] = _divide(dea[-1], current_close)
            result[f"{prefix}_hist_pct"] = _divide(dif[-1] - dea[-1], current_close)
        elif indicator.kind == "kd":
            rsv = _rolling_rsv(high, low, close, params["lookback"])
            k = _ewm(rsv, 1.0 / params["smooth_k"])
            d = _ewm(k, 1.0 / params["smooth_d"])
            result[f"{prefix}_k_norm"] = k[-1]
            result[f"{prefix}_d_norm"] = d[-1]
            result[f"{prefix}_diff_norm"] = k[-1] - d[-1]
        elif indicator.kind == "efi":
            force = np.zeros(n, dtype=np.float64)
            if n > 1:
                force[1:] = (close[1:] - close[:-1]) * volume[1:]
            baseline = _nanmean(np.abs(force[:-1][-params["baseline"] :]))
            efi2 = _ewm(force, 2.0 / (params["fast"] + 1.0))
            efi13 = _ewm(force, 2.0 / (params["slow"] + 1.0))
            result[f"{prefix}2_norm"] = _divide(efi2[-1], baseline)
            result[f"{prefix}13_norm"] = _divide(efi13[-1], baseline)
        else:
            fast = _ewm(close, 2.0 / (params["fast"] + 1.0))
            slow = _ewm(close, 2.0 / (params["slow"] + 1.0))
            lower = min(fast[-1], slow[-1])
            upper = max(fast[-1], slow[-1])
            midpoint = (fast + slow) / 2.0
            channel = f"ema_channel_{prefix}"
            result[f"{channel}_position"] = _divide_with_fill(current_close - lower, upper - lower, fill=0.5)
            result[f"{channel}_gap"] = _divide(current_close, midpoint[-1]) - 1.0
            result[f"{channel}_width"] = _divide_with_fill(upper - lower, midpoint[-1], fill=0.0)
            result[f"{channel}_slope"] = (
                _divide_with_fill(midpoint[-1], midpoint[-2], fill=1.0) - 1.0 if n >= 2 else 0.0
            )

    longest = max([20, *(indicator_lookback(item) for item in active)])
    result["history_coverage"] = min(1.0, n / float(longest))
    result["sequence_valid_ratio"] = min(
        1.0, n / float(max(1, int(spec.sequence_windows.get(freq, n))))
    )
    result["progress"] = columns["progress"][-1]

    clips = {field.name: field.clip for field in spec.market_fields}
    output: dict[str, float] = {}
    for name in feature_names:
        value = float(result.get(name, 0.0))
        if not np.isfinite(value):
            value = 0.0
        bounds = clips.get(name)
        if bounds is not None:
            value = float(np.clip(value, bounds[0], bounds[1]))
        output[str(name)] = value
    return output


def _pct_series(open_: np.ndarray, close: np.ndarray, supplied: np.ndarray) -> np.ndarray:
    previous = np.roll(close, 1)
    previous[0] = np.nan
    calculated = _divide_array(close, previous) - 1.0
    first = _divide_array(close, open_) - 1.0
    result = np.where(np.isfinite(supplied), supplied, calculated)
    result = np.where(np.isfinite(previous) & (previous != 0), result, first)
    return result


def _rolling_rsv(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int
) -> np.ndarray:
    low_n = pd.Series(low).rolling(int(window), min_periods=1).min().to_numpy()
    high_n = pd.Series(high).rolling(int(window), min_periods=1).max().to_numpy()
    return np.clip(_divide_array_with_fill(close - low_n, high_n - low_n, fill=0.5), 0.0, 1.0)


def _ewm(values: np.ndarray, alpha: float) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .ewm(alpha=float(alpha), adjust=False, min_periods=1)
        .mean()
        .to_numpy()
    )


def _sample_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan


def _nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else np.nan


def _seeded_reference_scalar(previous: float, open_: float, close: float) -> float:
    if np.isfinite(previous) and previous != 0:
        return float(previous)
    if np.isfinite(open_) and open_ != 0:
        return float(open_)
    return float(close) if np.isfinite(close) else np.nan


def _seeded_reference_array(previous: np.ndarray, open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    previous = np.asarray(previous, dtype=np.float64)
    reference = np.where(np.isfinite(previous) & (previous != 0), previous, open_)
    return np.where(np.isfinite(reference) & (reference != 0), reference, close)


def _first_finite(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite[0]) if len(finite) else np.nan


def _divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _divide_with_fill(numerator: float, denominator: float, *, fill: float) -> float:
    value = _divide(numerator, denominator)
    return float(fill) if not np.isfinite(value) else value


def _divide_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan)
    np.divide(
        numerator,
        denominator,
        out=output,
        where=np.isfinite(denominator) & (denominator != 0),
    )
    return output


def _divide_array_with_fill(numerator: np.ndarray, denominator: np.ndarray, *, fill: float) -> np.ndarray:
    values = _divide_array(numerator, denominator)
    denom = np.asarray(denominator, dtype=np.float64)
    return np.where(np.isfinite(denom) & (denom != 0), values, float(fill))


def _number(value: object) -> float:
    try:
        return float(value) if value is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    sums = np.where(finite, values, 0.0).sum(axis=1)
    output = np.full(len(values), np.nan, dtype=np.float64)
    np.divide(sums, counts, out=output, where=counts > 0)
    return output


def _row_nanstd(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    means = _row_nanmean(values)
    centered = np.where(finite, values - means[:, None], 0.0)
    sums = np.square(centered).sum(axis=1)
    output = np.full(len(values), np.nan, dtype=np.float64)
    np.divide(sums, counts - 1, out=output, where=counts > 1)
    return np.sqrt(output)
