from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from data_layer.storage.duckdb_storage import load_kline_from_duckdb
from feature_layer.builders.aggregation import aggregate_ohlcv_from_base, normalize_frequency


COMPARE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
DEFAULT_TOLERANCE: dict[str, float] = {
    "open": 1e-6,
    "high": 1e-6,
    "low": 1e-6,
    "close": 1e-6,
    "volume": 1e-6,
    "amount": 1e-2,
}


def compare_aggregated_frequency(
    db_path: str | Path,
    *,
    symbol: str,
    target_freq: str,
    adjust: str = "pre",
    base_freq: str = "5min",
    start: str | None = None,
    end: str | None = None,
    tolerances: dict[str, float] | None = None,
    sample_limit: int = 20,
) -> dict[str, object]:
    """
    Compare direct target-frequency bars against bars aggregated from base_freq.

    This is a validation helper for the 5min-as-base design. It does not write
    generated bars back to storage; it only reports whether the generated
    target frequency matches already downloaded target-frequency data.
    """
    target = normalize_frequency(target_freq)
    base = normalize_frequency(base_freq)
    tol = {**DEFAULT_TOLERANCE, **(tolerances or {})}

    base_rows = load_kline_from_duckdb(
        db_path,
        symbol=symbol,
        freq=base,
        adjust=adjust,
        start=start,
        end=end,
    )
    direct_rows = load_kline_from_duckdb(
        db_path,
        symbol=symbol,
        freq=target,
        adjust=adjust,
        start=start,
        end=end,
    )

    if base_rows.empty or direct_rows.empty:
        return _empty_result(
            symbol=symbol,
            target_freq=target,
            adjust=adjust,
            base_freq=base,
            base_rows=len(base_rows),
            direct_rows=len(direct_rows),
        )

    generated = aggregate_ohlcv_from_base(base_rows, target, base_freq=base)
    generated = generated.loc[~generated["is_partial"].astype(bool)].copy()

    prepared_generated = _prepare_generated(generated, target)
    prepared_direct = _prepare_direct(direct_rows, target)
    merged = prepared_generated.merge(
        prepared_direct,
        on="compare_key",
        how="outer",
        suffixes=("_generated", "_direct"),
        indicator=True,
    )

    matched = merged.loc[merged["_merge"].eq("both")].copy()
    column_stats: dict[str, dict[str, object]] = {}
    mismatch_mask = pd.Series(False, index=matched.index)

    for column in COMPARE_COLUMNS:
        left = pd.to_numeric(matched[f"{column}_generated"], errors="coerce")
        right = pd.to_numeric(matched[f"{column}_direct"], errors="coerce")
        both_missing = left.isna() & right.isna()
        diff = (left - right).abs()
        missing_mismatch = (left.isna() ^ right.isna()) & ~both_missing
        column_mismatch = missing_mismatch | (diff.fillna(0.0) > tol[column])
        mismatch_mask = mismatch_mask | column_mismatch
        column_stats[column] = {
            "max_abs_diff": float(diff.max()) if diff.notna().any() else 0.0,
            "mismatch_count": int(column_mismatch.sum()),
            "tolerance": tol[column],
        }

    mismatch_rows = matched.loc[mismatch_mask].head(sample_limit)
    missing_generated = merged.loc[merged["_merge"].eq("right_only"), "compare_key"].astype(str).head(sample_limit).tolist()
    missing_direct = merged.loc[merged["_merge"].eq("left_only"), "compare_key"].astype(str).head(sample_limit).tolist()

    total_mismatches = int(mismatch_mask.sum())
    ok = (
        len(matched) > 0
        and total_mismatches == 0
        and int((merged["_merge"] == "right_only").sum()) == 0
        and int((merged["_merge"] == "left_only").sum()) == 0
    )

    return {
        "ok": bool(ok),
        "symbol": str(symbol).upper(),
        "adjust": adjust,
        "base_freq": base,
        "target_freq": target,
        "base_rows": int(len(base_rows)),
        "generated_rows": int(len(generated)),
        "direct_rows": int(len(direct_rows)),
        "matched_rows": int(len(matched)),
        "missing_generated_count": int((merged["_merge"] == "right_only").sum()),
        "missing_direct_count": int((merged["_merge"] == "left_only").sum()),
        "mismatch_rows": total_mismatches,
        "column_stats": column_stats,
        "missing_generated_keys": missing_generated,
        "missing_direct_keys": missing_direct,
        "mismatch_samples": _mismatch_samples(mismatch_rows),
    }


def compare_many_aggregated_frequencies(
    db_path: str | Path,
    *,
    symbols: Iterable[str],
    target_freqs: Iterable[str],
    adjust: str = "pre",
    base_freq: str = "5min",
    start: str | None = None,
    end: str | None = None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for symbol in symbols:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            continue
        for target in target_freqs:
            results.append(
                compare_aggregated_frequency(
                    db_path,
                    symbol=normalized_symbol,
                    target_freq=target,
                    adjust=adjust,
                    base_freq=base_freq,
                    start=start,
                    end=end,
                )
            )
    return results


def comparison_summary_frame(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "ok": result.get("ok"),
                "symbol": result.get("symbol"),
                "adjust": result.get("adjust"),
                "base_freq": result.get("base_freq"),
                "target_freq": result.get("target_freq"),
                "base_rows": result.get("base_rows"),
                "generated_rows": result.get("generated_rows"),
                "direct_rows": result.get("direct_rows"),
                "matched_rows": result.get("matched_rows"),
                "missing_generated_count": result.get("missing_generated_count"),
                "missing_direct_count": result.get("missing_direct_count"),
                "mismatch_rows": result.get("mismatch_rows"),
            }
        )
    return pd.DataFrame(rows)


def _prepare_generated(frame: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    result = frame.copy()
    result["compare_key"] = _comparison_key(result["bar_end"], target_freq)
    keep = ["compare_key", *COMPARE_COLUMNS]
    return result[keep]


def _prepare_direct(frame: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    result = frame.copy()
    result["compare_key"] = _comparison_key(result["datetime"], target_freq)
    keep = ["compare_key", *COMPARE_COLUMNS]
    return result[keep]


def _comparison_key(values: pd.Series, target_freq: str) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce")
    normalized = normalize_frequency(target_freq)
    if normalized in {"daily", "weekly", "monthly"}:
        return dt.dt.strftime("%Y-%m-%d")
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


def _mismatch_samples(frame: pd.DataFrame) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        item = {"compare_key": str(getattr(row, "compare_key"))}
        for column in COMPARE_COLUMNS:
            item[f"{column}_generated"] = getattr(row, f"{column}_generated")
            item[f"{column}_direct"] = getattr(row, f"{column}_direct")
        samples.append(item)
    return samples


def _empty_result(
    *,
    symbol: str,
    target_freq: str,
    adjust: str,
    base_freq: str,
    base_rows: int,
    direct_rows: int,
) -> dict[str, object]:
    return {
        "ok": False,
        "symbol": str(symbol).upper(),
        "adjust": adjust,
        "base_freq": base_freq,
        "target_freq": target_freq,
        "base_rows": int(base_rows),
        "generated_rows": 0,
        "direct_rows": int(direct_rows),
        "matched_rows": 0,
        "missing_generated_count": int(direct_rows),
        "missing_direct_count": 0,
        "mismatch_rows": 0,
        "column_stats": {},
        "missing_generated_keys": [],
        "missing_direct_keys": [],
        "mismatch_samples": [],
    }

