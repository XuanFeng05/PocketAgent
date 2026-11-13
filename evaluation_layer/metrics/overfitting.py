from __future__ import annotations

from itertools import combinations
import math
from statistics import NormalDist
from typing import Iterable

import numpy as np


def probabilistic_sharpe_ratio(
    estimated_sharpe: float,
    benchmark_sharpe: float,
    observations: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    if observations < 2:
        raise ValueError("At least two return observations are required.")
    if not all(math.isfinite(float(value)) for value in (estimated_sharpe, benchmark_sharpe, skewness, kurtosis)):
        raise ValueError("Sharpe estimates and moments must be finite.")
    denominator = 1.0 - skewness * estimated_sharpe + (
        (kurtosis - 1.0) / 4.0
    ) * estimated_sharpe**2
    if denominator <= 0:
        raise ValueError("Sharpe sampling variance is not positive.")
    statistic = (
        (estimated_sharpe - benchmark_sharpe)
        * math.sqrt(observations - 1)
        / math.sqrt(denominator)
    )
    return float(NormalDist().cdf(statistic))


def deflated_sharpe_ratio(
    estimated_sharpe: float,
    trial_sharpes: Iterable[float],
    observations: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    independent_trials: float | None = None,
) -> dict[str, float]:
    trials = np.asarray(list(trial_sharpes), dtype=np.float64)
    if trials.ndim != 1 or trials.size == 0 or not np.isfinite(trials).all():
        raise ValueError("All trial Sharpes must be finite; missing attempts must be accounted for explicitly.")
    count = float(trials.size if independent_trials is None else independent_trials)
    if not math.isfinite(count) or not 1 <= count <= trials.size:
        raise ValueError("independent_trials must be between 1 and the supplied trial count.")
    if count == 1 or trials.size == 1 or float(trials.std(ddof=1)) == 0.0:
        benchmark = 0.0
    else:
        deviation = float(trials.std(ddof=1))
        euler_gamma = 0.5772156649015329
        # DSR tests the zero-true-Sharpe null. The observed candidates' mean
        # must not be added to the expected maximum under that null.
        benchmark = max(0.0, deviation * (
            (1.0 - euler_gamma) * NormalDist().inv_cdf(1.0 - 1.0 / count)
            + euler_gamma
            * NormalDist().inv_cdf(1.0 - 1.0 / (count * math.e))
        ))
    probability = probabilistic_sharpe_ratio(
        estimated_sharpe,
        benchmark,
        observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    return {
        "deflated_sharpe_probability": probability,
        "selection_adjusted_benchmark_sharpe": benchmark,
        "trials": float(trials.size),
        "independent_trials_assumed": count,
    }


def probability_of_backtest_overfitting(
    period_returns: np.ndarray,
    *,
    partitions: int = 10,
) -> dict[str, float]:
    """Estimate CSCV probability that the best in-sample strategy ranks below median OOS."""
    values = np.asarray(period_returns, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("period_returns must be [periods, strategies] with at least two strategies.")
    if isinstance(partitions, bool) or not isinstance(partitions, int) or partitions < 2 or partitions > 12 or partitions % 2:
        raise ValueError("CSCV partitions must be an even integer from 2 to 12.")
    if values.shape[0] < max(4, partitions * 2):
        raise ValueError("There must be at least two return rows per CSCV partition.")
    blocks = [block for block in np.array_split(np.arange(values.shape[0]), partitions) if len(block)]
    half = partitions // 2
    logits: list[float] = []
    degradation: list[float] = []
    all_blocks = set(range(partitions))
    for selected in combinations(range(partitions), half):
        train_rows = np.concatenate([blocks[index] for index in selected])
        test_rows = np.concatenate([blocks[index] for index in sorted(all_blocks.difference(selected))])
        in_sample = _column_sharpes(values[train_rows])
        out_sample = _column_sharpes(values[test_rows])
        winner = int(np.nanargmax(in_sample))
        ties = np.isclose(out_sample, out_sample[winner], atol=1e-12, rtol=1e-12)
        below = (out_sample < out_sample[winner]) & ~ties
        rank = float(below.sum() + (ties.sum() + 1) / 2) / (values.shape[1] + 1.0)
        rank = min(max(rank, 1e-12), 1.0 - 1e-12)
        logits.append(math.log(rank / (1.0 - rank)))
        degradation.append(float(in_sample[winner] - out_sample[winner]))
    logit_array = np.asarray(logits)
    return {
        "probability_of_backtest_overfitting": float((logit_array <= 0).mean()),
        "mean_oos_degradation": float(np.mean(degradation)),
        "combinations": float(len(logits)),
    }


def _column_sharpes(returns: np.ndarray) -> np.ndarray:
    means = np.nanmean(returns, axis=0)
    deviations = np.nanstd(returns, axis=0, ddof=1)
    return np.divide(means, deviations, out=np.zeros_like(means), where=deviations > 0)
