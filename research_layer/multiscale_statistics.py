"""Synchronized date-block diagnostics for a registered multi-scale family.

These tests describe a fixed, already-viewed historical window. They do not
remove survivor selection, historical researcher discretion, or dependence on
the particular period. The primary test is an unstudentized centered max-mean
test against daily RF; it is not factor alpha or a probability of overfitting.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm

from research_layer.elder_analysis import _psr, _rf_aligned, holm_adjust, nav_returns


METHOD_SOURCES = {
    "block_bootstrap": "https://doi.org/10.1214/aos/1176347265",
    "data_snooping_family_test": "https://doi.org/10.1111/1468-0262.00152",
    "holm": "https://www.jstor.org/stable/4615733",
    "psr_dsr": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf",
}


def _circular_sums(values, length):
    """Sums beginning at every original row, wrapping once if necessary."""
    n = len(values)
    extended = np.concatenate((values, values[:length - 1]), axis=0)
    cumulative = np.concatenate((np.zeros((1, values.shape[1])), np.cumsum(extended, axis=0)), axis=0)
    return cumulative[np.arange(n) + length] - cumulative[np.arange(n)]


def circular_bootstrap_means(values, B, block, seed, batch_size=128):
    """Yield exact fixed-length circular-block means, synchronized by column.

    Equivalent to concatenating ceil(n/block) randomly selected date blocks,
    truncating at n rows, and averaging. Sparse counts of block *starts* avoid
    constructing a [replications, dates, candidates] sample cube. Random draws
    remain identical across batch sizes and to literal resampling.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Values must be a nonempty finite [dates,candidates] matrix")
    n = len(values)
    if not isinstance(B, int) or B < 1 or not isinstance(block, int) or not 1 <= block <= n:
        raise ValueError("Positive replications and a valid integer block length required")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive batch_size required")
    full, remainder = divmod(n, block)
    draws = full + bool(remainder)
    sums = _circular_sums(values, block)
    partial = _circular_sums(values, remainder) if remainder else None
    rng = np.random.default_rng(seed)
    for offset in range(0, B, batch_size):
        count = min(batch_size, B - offset)
        starts = rng.integers(0, n, size=(count, draws))
        weights = csr_matrix((np.ones(count * full),
                              (np.repeat(np.arange(count), full), starts[:, :full].reshape(-1))),
                             shape=(count, n))
        total = weights @ sums
        if remainder:
            total += partial[starts[:, -1]]
        yield offset, total / n


def block_family_test(excess, ids, selected_ids=(), B=39999, block=20,
                      seed=20261113, batch_size=128):
    """One-sided positive-mean tests with common centered family maxima.

    Individual percentile CIs are retained only for training-frozen choices.
    All candidates receive individual p, Holm p and single-step max-mean p.
    The latter is a centered, unstudentized bootstrap diagnostic; it is not
    asserted to implement studentized Romano-Wolf or Hansen SPA.
    """
    values = np.asarray(excess, dtype=float)
    ids, selected_ids = list(ids), list(selected_ids)
    if values.ndim != 2 or values.shape[1] != len(ids) or not ids:
        raise ValueError("Excess matrix columns must correspond to candidate IDs")
    if len(ids) != len(set(ids)) or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Candidate and selected IDs must be unique")
    if not set(selected_ids).issubset(ids):
        raise ValueError("Selected IDs must belong to the tested family")
    if B < 100:
        raise ValueError("At least100 bootstrap replications required")
    selected_idx = [ids.index(identifier) for identifier in selected_ids]
    selected_boot = np.empty((B, len(selected_idx)))
    maxima = np.empty(B)
    mu = values.mean(axis=0)
    exceed = np.zeros(len(ids), dtype=np.int64)
    started = time.monotonic()
    for offset, means in circular_bootstrap_means(values, B, block, seed, batch_size):
        count = len(means)
        centered = means - mu
        exceed += np.count_nonzero(centered >= mu, axis=0)
        maxima[offset:offset + count] = centered.max(axis=1)
        if selected_idx:
            selected_boot[offset:offset + count] = means[:, selected_idx]
    pvalues = (1 + exceed) / (B + 1)
    adjusted = holm_adjust(pvalues)
    sorted_maxima = np.sort(maxima)
    family_adjusted = (1 + B - np.searchsorted(sorted_maxima, mu, side="left")) / (B + 1)
    intervals = np.quantile(selected_boot, [.025, .975], axis=0) * 252 if selected_idx else None
    selected_lookup = {identifier: j for j, identifier in enumerate(selected_ids)}
    rows = []
    for j, identifier in enumerate(ids):
        s = selected_lookup.get(identifier)
        rows.append({"id": identifier, "mean_annualized_excess_rf": float(mu[j] * 252),
                     "positive_excess_one_sided_p": float(pvalues[j]),
                     "holm_adjusted_p": float(adjusted[j]),
                     "single_step_max_mean_adjusted_p": float(family_adjusted[j]),
                     "selected_on_training": s is not None,
                     "mean_annualized_excess_rf_individual_ci95":
                         [float(intervals[0, s]), float(intervals[1, s])] if s is not None else None})
    return {"block_sessions": block, "replications": B, "seed": seed,
            "method": "synchronized circular fixed-length date blocks; centered unstudentized max-mean",
            "candidate_count": len(ids), "sessions": len(values),
            "minimum_attainable_individual_p": 1 / (B + 1),
            "minimum_attainable_first_holm_p": min(1., len(ids) / (B + 1)),
            "holm_can_reject_at_5pct": len(ids) / (B + 1) <= .05,
            "elapsed_seconds": time.monotonic() - started,
            "candidate_results": rows,
            "family_centered_max_mean_test": {
                "observed_max_annualized_mean_excess": float(mu.max() * 252),
                "p_one_sided": float(family_adjusted[np.argmax(mu)]),
                "studentized": False},
            "confidence_interval_scope": "Individual percentile intervals for training-frozen IDs only; not simultaneous or selection-adjusted intervals."}


def assess_multiscale_family(nav_matrix, dates, rf_series, selected_ids, ids,
                            B=39999, blocks=(20, 40, 60), seed=20261113,
                            lifetime_trials_minimum=1612,
                            dsr_trial_scenarios=(1612, 5000), cash_interest=True):
    """Assess the complete candidate family, never only its selected winners."""
    dates = list(pd.to_datetime(dates).strftime("%Y-%m-%d"))
    ids, selected_ids = list(ids), list(selected_ids)
    values = np.asarray(nav_matrix, dtype=float)
    if values.ndim != 2 or values.shape != (len(dates), len(ids)):
        raise ValueError("NAV matrix shape must be [dates,candidates]")
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError("Dates must be unique and ordered")
    if lifetime_trials_minimum < len(ids):
        raise ValueError("Lifetime search lower bound cannot be below the tested family size")
    blocks = tuple(blocks)
    if not blocks or len(blocks) != len(set(blocks)):
        raise ValueError("Distinct block lengths required")
    if any(not isinstance(n, int) or n <= 1 for n in dsr_trial_scenarios):
        raise ValueError("DSR trial scenarios must be integers greater than1")
    rf = _rf_aligned(rf_series, dates)
    excess = nav_returns(values) - rf[:, None]
    # A sleeve that remains entirely in RF cash can differ from RF by a few
    # machine epsilons after NAV multiplication/division and500-sleeve means.
    # Do not turn that arithmetic noise into a statistically positive alpha.
    numerical_zero = 1e-12
    excess[np.abs(excess) <= numerical_zero] = 0.
    block_results = [block_family_test(excess, ids, selected_ids, B, length, seed) for length in blocks]
    mu, sd = excess.mean(axis=0), excess.std(axis=0, ddof=1)
    sharpe = np.divide(mu, sd, out=np.full(len(ids), np.nan), where=sd > 1e-12)
    finite = sharpe[np.isfinite(sharpe)]
    dispersion = float(finite.std(ddof=1)) if len(finite) > 1 else None
    gamma = .5772156649015329
    diagnostics = []
    for identifier in selected_ids:
        j = ids.index(identifier)
        row = {"id": identifier, "PSR_iid_approximation": _psr(excess[:, j]), "DSR_scenarios": []}
        if dispersion is not None:
            for n in dsr_trial_scenarios:
                threshold = dispersion * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * math.e)))
                row["DSR_scenarios"].append({"assumed_effective_independent_trials": n,
                    "daily_sharpe_null_max_threshold": float(threshold),
                    "DSR_iid_approximation": _psr(excess[:, j], threshold)})
        diagnostics.append(row)
    return {"status": "completed", "candidate_count": len(ids), "sessions": len(dates),
            "period": [dates[0], dates[-1]], "primary_block_sessions": blocks[0],
            "lifetime_recorded_strategy_trials_lower_bound": lifetime_trials_minimum,
            "absolute_daily_excess_numerical_zero_tolerance": numerical_zero,
            "cash_interest": "Prior-close idle cash earns the French daily RF proxy; not a broker cash-sweep quote."
                             if cash_interest else "Idle cash earns zero; RF comparison includes the resulting financing drag.",
            "block_sensitivity": block_results,
            "selected_PSR_DSR_diagnostics": diagnostics,
            "method_sources": METHOD_SOURCES,
            "limitations": [
                "All years were already viewed. This is retrospective chronological analysis, not an untouched prospective holdout.",
                "The bootstrap tests the current declared strategy family only. Historical72 trials and event/parameter exploration are disclosed but not jointly resampled; lifetime FWER is not established.",
                "Fixed survivor/current-universe selection and sample-period choice remain biases that these p-values cannot remove.",
                "Stocks are not independent replications. All candidates use identical sampled date blocks, preserving observed cross-candidate dependence after portfolio aggregation.",
                "Block tests assume sufficiently regular weak dependence;20/40/60-session sensitivity does not prove stationarity or eliminate regime risk.",
                "RF-excess mean is not factor alpha. Proportional transaction fees, data adjustments and cash-interest proxies are research assumptions.",
                "Individual percentile CIs describe frozen choices and are not selection-adjusted simultaneous confidence intervals.",
                "PSR/DSR use an IID approximation and hypothetical effective-independent-trial scenarios. Cross-candidate Sharpe dispersion is not a known null distribution, and DSR is not PBO.",
                "No single bootstrap block length or favorable candidate may be chosen after observing these diagnostics.",
            ]}
