"""Financial diagnostics for a registered, retrospective Elder research family.

This module never fills missing risk-free rates with zero. Statistical results
are conditional on the declared candidate family and the already-viewed market
history, not certification of a profitable or untouched out-of-sample strategy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


FRENCH_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def nav_returns(nav):
    values = np.asarray(nav, dtype=float)
    if values.ndim not in (1, 2) or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("NAV must be a nonempty positive finite one- or two-dimensional array")
    previous = np.concatenate((np.ones((1,) + values.shape[1:]), values[:-1]), axis=0)
    return values / previous - 1


def metrics(nav, rf=None) -> dict:
    """Metrics for close NAV with initial capital 1 preceding the first row.

    rf, if provided, must be already date-aligned daily simple returns. CAGR
    uses 252 sessions per year; Sharpe subtracts daily RF and uses sample SD.
    """
    nav = np.asarray(nav, dtype=float)
    if nav.ndim != 1:
        raise ValueError("metrics accepts one NAV series")
    r = nav_returns(nav)
    n = len(r)
    peaks = np.maximum.accumulate(np.r_[1., nav])[1:]
    dd = 1 - nav / peaks
    mdd = float(dd.max())
    cagr = float(nav[-1] ** (252 / n) - 1)
    volatility = float(r.std(ddof=1) * np.sqrt(252)) if n > 1 else None
    tail_cutoff = float(np.quantile(r, .05))
    streak = longest = 0
    for value in dd:
        streak = streak + 1 if value > 1e-12 else 0
        longest = max(longest, streak)
    out = {
        "total_return": float(nav[-1] - 1), "cagr": cagr,
        "maximum_drawdown": mdd, "volatility": volatility,
        "calmar": cagr / mdd if mdd > 1e-12 else None,
        "daily_VaR95_loss": -tail_cutoff,
        "daily_ES95_loss": float(-r[r <= tail_cutoff].mean()),
        "worst_day": float(r.min()), "longest_underwater_sessions": longest,
        "end_drawdown": float(dd[-1]), "sessions": n,
        "sharpe_excess_rf": None, "sortino_excess_rf": None,
        "mean_annualized_excess_rf": None, "rf_supplied": rf is not None,
    }
    if rf is not None:
        rates = np.asarray(rf, dtype=float)
        if rates.shape != r.shape or not np.isfinite(rates).all() or (rates <= -1).any():
            raise ValueError("RF must have exactly one finite valid daily return per NAV row")
        excess = r - rates
        sd = excess.std(ddof=1) if n > 1 else 0
        downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
        out.update(sharpe_excess_rf=float(excess.mean() / sd * np.sqrt(252)) if sd > 1e-12 else None,
                   sortino_excess_rf=float(excess.mean() / downside * np.sqrt(252)) if downside > 1e-12 else None,
                   mean_annualized_excess_rf=float(excess.mean() * 252))
    return out


def parse_french_daily(body: bytes, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Read official daily ZIP and convert provider percent values to decimals."""
    with zipfile.ZipFile(BytesIO(body)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected one official factor CSV in ZIP")
        raw = archive.read(members[0])
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if "Mkt-RF" in line and "RF" in line and "SMB" in line), None)
    if header is None:
        raise ValueError("Official factor header not found")
    rows = []
    for line in lines[header + 1:]:
        if re.match(r"^\s*\d{8}\s*,", line):
            rows.append(line)
        elif rows:
            break
    if not rows:
        raise ValueError("No daily observations in official CSV")
    frame = pd.read_csv(StringIO("\n".join([lines[header], *rows])), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame.index = pd.to_datetime(frame.index.astype(str).str.strip(), format="%Y%m%d")
    frame.index.name = "date"
    required = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    if set(required) - set(frame):
        raise ValueError("Missing factor columns")
    frame = frame.loc[start:end, required].astype(float)
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Empty, duplicate or unordered factor dates")
    if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() <= -99).any():
        raise ValueError("Missing or invalid factor value")
    frame /= 100.0
    frame.index = frame.index.strftime("%Y-%m-%d")
    return frame, {"zip_member": members[0], "provider_header": "\n".join(lines[:header]),
                   "provider_csv_sha256": hashlib.sha256(raw).hexdigest(),
                   "unit_conversion": "provider percentages divided by 100 once; daily simple returns"}


def fetch_french_rf(outdir, start="2022-01-01", end="2025-12-31") -> dict:
    """Bounded official-factor download, returning success metadata or failure.

    Files: official_ff5_daily.zip, ff5_daily.csv, manifest.json. Callers load RF
    from the success CSV and require exact date matches; failures are explicit.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "success" and prior.get("start") == start and prior.get("end") == end:
            zipped, csv_path = out / "official_ff5_daily.zip", out / "ff5_daily.csv"
            if zipped.exists() and csv_path.exists() and hashlib.sha256(zipped.read_bytes()).hexdigest() == prior["zip_sha256"] and hashlib.sha256(csv_path.read_bytes()).hexdigest() == prior["csv_sha256"]:
                return {**prior, "reused_verified_files": True}
            raise ValueError("Existing successful RF snapshot hash mismatch; refusing to overwrite")
    meta = {"source": "Kenneth French official US daily five-factor library",
            "url": FRENCH_DAILY_URL, "start": start, "end": end,
            "download_started_utc": datetime.now(timezone.utc).isoformat(),
            "network_timeout_seconds": 25, "outer_timeout_seconds": 60,
            "independent_benchmark_not_stock_price_source": True}
    script = "from urllib.request import Request,urlopen;import sys; r=urlopen(Request(sys.argv[1],headers={'User-Agent':'Mozilla/5.0'}),timeout=25);sys.stdout.buffer.write(r.read())"
    try:
        completed = subprocess.run([sys.executable, "-c", script, FRENCH_DAILY_URL], capture_output=True, timeout=60, check=True)
        body = completed.stdout
        (out / "official_ff5_daily.zip").write_bytes(body)
        frame, details = parse_french_daily(body, start, end)
        csv_path = out / "ff5_daily.csv"
        frame.to_csv(csv_path)
        meta.update(status="success", **details, rows=len(frame), first_date=frame.index[0], last_date=frame.index[-1],
                    csv_path=str(csv_path), zip_sha256=hashlib.sha256(body).hexdigest(),
                    csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                    download_finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        meta.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                    download_finished_utc=datetime.now(timezone.utc).isoformat())
        if isinstance(exc, subprocess.CalledProcessError):
            meta["stderr_tail"] = exc.stderr.decode("utf-8", errors="replace")[-2000:]
    _save(manifest_path, meta)
    return meta


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a finite vector in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def _rf_aligned(rf_series, dates):
    if rf_series is None:
        raise ValueError("Official daily RF required; zero substitution is prohibited")
    if isinstance(rf_series, pd.DataFrame):
        if "RF" not in rf_series:
            raise ValueError("RF column required")
        rf_series = rf_series["RF"]
    if isinstance(rf_series, pd.Series):
        rates = rf_series.copy()
        rates.index = pd.to_datetime(rates.index).strftime("%Y-%m-%d")
        if rates.index.has_duplicates:
            raise ValueError("Duplicate RF dates")
        missing = set(dates) - set(rates.index)
        if missing:
            raise ValueError(f"Missing RF dates: {sorted(missing)[:5]}")
        values = rates.loc[dates].to_numpy(dtype=float)
    else:
        values = np.asarray(rf_series, dtype=float)
    if values.shape != (len(dates),) or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Invalid aligned RF series")
    return values


def _psr(excess, threshold=0.0):
    r = np.asarray(excess, dtype=float)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 4 or sd <= 1e-12:
        return None
    sr = r.mean() / sd
    sk = float(skew(r, bias=False))
    ku = float(kurtosis(r, bias=False, fisher=False))
    denom = 1 - sk * sr + (ku - 1) * sr * sr / 4
    return float(norm.cdf((sr - threshold) * np.sqrt(n - 1) / np.sqrt(denom))) if denom > 0 else None


def assess_family(nav_matrix, dates, rf_series, selected_ids, ids, B=3000, block=20, seed=20261113) -> dict:
    """Synchronized circular block bootstrap and centered max-mean family test.

    The max-mean test is deliberately unstudentized: it tests positive daily
    RF-excess mean across this declared family and is not a factor-alpha test.
    Percentile intervals are individual, not simultaneous selection-adjusted
    confidence intervals. A later-window analysis remains retrospective.
    """
    values = np.asarray(nav_matrix, dtype=float)
    ids = list(ids)
    dates = list(pd.to_datetime(dates).strftime("%Y-%m-%d"))
    if values.ndim != 2 or values.shape != (len(dates), len(ids)):
        raise ValueError("NAV shape must be [dates, candidates]")
    if len(set(ids)) != len(ids) or len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("Unique candidate IDs and ordered unique dates are required")
    if not set(selected_ids).issubset(ids):
        raise ValueError("Selected ID not in candidate family")
    if not isinstance(B, int) or B < 100 or not isinstance(block, int) or not 1 <= block <= len(dates):
        raise ValueError("Require B>=100 and a block size within sample length")
    rf = _rf_aligned(rf_series, dates)
    excess = nav_returns(values) - rf[:, None]
    n, k = excess.shape
    mean = excess.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot_means = np.empty((B, k))
    for start in range(0, B, 50):
        count = min(50, B - start)
        starts = rng.integers(0, n, size=(count, math.ceil(n / block)))
        index = ((starts[:, :, None] + np.arange(block)) % n).reshape(count, -1)[:, :n]
        boot_means[start:start + count] = excess[index].mean(axis=1)
    centered = boot_means - mean
    p = (1 + (centered >= mean).sum(axis=0)) / (B + 1)
    ci = np.quantile(boot_means, [.025, .975], axis=0) * 252
    adjusted = holm_adjust(p)
    family_observed = float(mean.max())
    family_p = float((1 + (centered.max(axis=1) >= family_observed).sum()) / (B + 1))
    per_candidate = []
    for j, identifier in enumerate(ids):
        per_candidate.append({"id": identifier, "mean_annualized_excess_rf": float(mean[j] * 252),
                              "mean_annualized_excess_rf_individual_ci95": [float(ci[0, j]), float(ci[1, j])],
                              "positive_excess_one_sided_p": float(p[j]), "holm_adjusted_p": float(adjusted[j]),
                              "selected_on_training": identifier in selected_ids})
    sd = excess.std(axis=0, ddof=1)
    sr = np.divide(mean, sd, out=np.full(k, np.nan), where=sd > 1e-12)
    valid_sr = sr[np.isfinite(sr)]
    cross_case_sd = float(valid_sr.std(ddof=1)) if len(valid_sr) > 1 else None
    diagnostics = []
    gamma = .5772156649015329
    for identifier in selected_ids:
        j = ids.index(identifier)
        item = {"id": identifier, "PSR_iid_approximation": _psr(excess[:, j]), "DSR_scenarios": []}
        if cross_case_sd is not None:
            for effective_trials in [72, 200]:
                threshold = cross_case_sd * ((1 - gamma) * norm.ppf(1 - 1 / effective_trials) + gamma * norm.ppf(1 - 1 / (effective_trials * math.e)))
                item["DSR_scenarios"].append({"assumed_effective_independent_trials": effective_trials,
                                              "daily_sharpe_null_max_threshold": float(threshold),
                                              "DSR_iid_approximation": _psr(excess[:, j], threshold)})
        diagnostics.append(item)
    return {"status": "completed", "candidate_count": k, "sessions": n, "period": [dates[0], dates[-1]],
            "bootstrap": {"method": "synchronized circular fixed-length date blocks", "block_sessions": block,
                          "replications": B, "seed": seed, "dependence_preserved": "same sampled dates for every candidate"},
            "candidate_results": per_candidate,
            "family_centered_max_mean_test": {"observed_max_annualized_mean_excess": family_observed * 252,
                                                "p_one_sided": family_p, "studentized": False},
            "selected_PSR_DSR_diagnostics": diagnostics,
            "method_sources": {"RF": FRENCH_DAILY_URL, "PSR_DSR": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf"},
            "limitations": ["Already-viewed history; retrospective chronological diagnostic, not untouched holdout.",
                            "Candidate family and lifetime search history constrain multiple-testing interpretation.",
                            "PSR/DSR assume IID approximation and hypothetical effective independent trials; bootstrap is the primary serial-dependence diagnostic.",
                            "Cross-candidate Sharpe dispersion is a scenario input, not a known null distribution or a recovered effective trial count.",
                            "Idle cash earns zero in the engine; RF excess can be depressed by financing assumptions and is not factor alpha.",
                            "Individual confidence intervals do not adjust for selecting the best candidate."]}
