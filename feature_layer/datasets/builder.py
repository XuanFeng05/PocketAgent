from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_layer.storage.duckdb_storage import (
    KLINE_SELECT_COLUMNS,
    load_kline_from_duckdb,
    load_stock_liquidity_daily_from_duckdb,
    load_stock_status_daily_from_duckdb,
)
from feature_layer.datasets.market_parquet import (
    load_kline_from_market_parquet_cache,
    load_stock_liquidity_from_market_parquet_cache,
    load_stock_status_from_market_parquet_cache,
)
from feature_layer.builders.aggregation import INTRADAY_MINUTES, aggregate_ohlcv_from_base, normalize_frequency
from feature_layer.builders.bar_features import build_bar_features
from feature_layer.builders.decision_context import build_decision_context
from feature_layer.builders.decision_points import build_decision_points
from feature_layer.indicator_registry import indicator_lookback
from feature_layer.specs import DEFAULT_FEATURE_SPEC, FeatureSpec


DEFAULT_DATASET_FREQUENCIES: tuple[str, ...] = (
    "5min",
    "30min",
    "daily",
    "weekly",
)

FEATURE_WARMUP_REQUIREMENTS: dict[str, int] = {
    "returns_and_rolling": 20,
    "macd": 35,
    "kd": 9,
    "efi": 20,
    "ema_13_21": 21,
    "ema_144_169": 169,
}


@dataclass(frozen=True)
class FeatureDatasetConfig:
    trade_freq: str = "5min"
    frequencies: tuple[str, ...] = DEFAULT_DATASET_FREQUENCIES
    adjust: str = "none"
    include_open_auction: bool = True
    max_decisions: int | None = None
    decision_start: str | None = None
    decision_end: str | None = None
    require_limit_reference: bool = False
    materialize_sequences: bool = True


@dataclass(frozen=True)
class FeatureDataset:
    spec_name: str
    frequencies: tuple[str, ...]
    decisions: pd.DataFrame
    decision_context: pd.DataFrame
    constraints: pd.DataFrame
    market: dict[str, pd.DataFrame]
    requested_symbols: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "spec": self.spec_name,
            "decisions": int(len(self.decisions)),
            "decision_context_rows": int(len(self.decision_context)),
            "frequencies": list(self.frequencies),
            "market_rows": {freq: int(len(frame)) for freq, frame in self.market.items()},
            "requested_symbols": list(self.requested_symbols),
            "st_decisions": int(self.decisions.get("is_st", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        }


@dataclass(frozen=True)
class FeatureSourceFrames:
    """Source frames loaded once for both regular and compact feature builders."""

    base: pd.DataFrame
    daily: pd.DataFrame
    daily_status: pd.DataFrame
    requested_symbols: tuple[str, ...]
    status_coverage: dict[str, object]


def build_feature_dataset(
    base_bars: pd.DataFrame,
    *,
    daily_bars: pd.DataFrame | None = None,
    daily_status: pd.DataFrame | None = None,
    config: FeatureDatasetConfig | None = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> FeatureDataset:
    """
    Build table-shaped model inputs for every visible decision point.

    Frequency policy:
    - 5/15/30/60 minute states use visible completed 5min bars.
    - Daily state uses official completed daily bars plus the current visible
      intraday partial daily bar.
    - Weekly states use official completed daily bars plus that same current
      partial daily bar, then aggregate by calendar period.
    """
    cfg = config or FeatureDatasetConfig(
        trade_freq=spec.trade_frequency,
        frequencies=("5min", *spec.derived_frequencies),
    )
    frequencies = tuple(normalize_frequency(freq) for freq in cfg.frequencies)
    trade_freq = normalize_frequency(cfg.trade_freq)
    _validate_dataset_frequencies(frequencies, spec=spec)

    base = _prepare_kline_frame(base_bars, freq=trade_freq, adjust=cfg.adjust)
    daily = _prepare_kline_frame(daily_bars, freq="daily", adjust=cfg.adjust)

    decisions = build_decision_points(
        base,
        daily_status=daily_status,
        trade_freq=trade_freq,
        include_open_auction=cfg.include_open_auction,
        require_limit_reference=cfg.require_limit_reference,
    )
    if cfg.decision_start:
        decisions = decisions.loc[
            pd.to_datetime(decisions["decision_time"], errors="coerce") >= pd.Timestamp(cfg.decision_start)
        ].copy()
    if cfg.decision_end:
        end = pd.Timestamp(cfg.decision_end)
        if len(str(cfg.decision_end).strip()) == 10:
            end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        decisions = decisions.loc[
            pd.to_datetime(decisions["decision_time"], errors="coerce") <= end
        ].copy()
    decisions = _attach_decision_ids(decisions)
    if cfg.max_decisions is not None and int(cfg.max_decisions) > 0:
        decisions = decisions.head(int(cfg.max_decisions)).reset_index(drop=True)

    market_frames: dict[str, list[pd.DataFrame]] = {freq: [] for freq in frequencies}
    if cfg.materialize_sequences:
        for decision in decisions.itertuples(index=False):
            visible_by_freq = _visible_bars_by_frequency(
                decision=decision,
                base=base,
                daily=daily,
                frequencies=frequencies,
                trade_freq=trade_freq,
            )
            for freq, visible in visible_by_freq.items():
                part = _build_sequence_frame(
                    visible,
                    decision=decision,
                    freq=freq,
                    spec=spec,
                )
                if not part.empty:
                    market_frames[freq].append(part)

    market = {
        freq: (
            pd.concat(parts, ignore_index=True)
            if parts
            else _empty_market_frame(spec=spec)
        )
        for freq, parts in market_frames.items()
    }
    constraints = _constraints_frame(decisions, spec=spec)
    decision_context = build_decision_context(decisions, spec=spec)
    return FeatureDataset(
        spec_name=spec.name,
        frequencies=frequencies,
        decisions=decisions,
        decision_context=decision_context,
        constraints=constraints,
        market=market,
        requested_symbols=tuple(sorted(base["symbol"].dropna().astype(str).unique())),
    )


def load_feature_source_frames_from_duckdb(
    db_path: str | Path,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    adjust: str = "none",
    start: str | None = None,
    end: str | None = None,
    config: FeatureDatasetConfig | None = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> FeatureSourceFrames:
    """Load source bars and daily rule tables exactly once for a symbol batch."""

    cfg = config or FeatureDatasetConfig(
        trade_freq=spec.trade_frequency,
        frequencies=("5min", *spec.derived_frequencies),
        adjust=adjust,
    )
    selected_symbols = _normalize_symbols(symbol=symbol, symbols=symbols)
    base = load_kline_from_duckdb(
        db_path,
        symbol=selected_symbols[0] if len(selected_symbols) == 1 else None,
        symbols=selected_symbols if len(selected_symbols) != 1 else None,
        freq=normalize_frequency(cfg.trade_freq),
        adjust=cfg.adjust,
        start=None,
        end=end,
    )
    daily = load_kline_from_duckdb(
        db_path,
        symbol=selected_symbols[0] if len(selected_symbols) == 1 else None,
        symbols=selected_symbols if len(selected_symbols) != 1 else None,
        freq="daily",
        adjust=cfg.adjust,
        start=None,
        end=end,
    )
    liquidity = load_stock_liquidity_daily_from_duckdb(
        db_path, symbols=selected_symbols or None, start=None, end=end
    )
    daily = _attach_daily_turn(daily, liquidity)
    requested = tuple(selected_symbols or sorted(base["symbol"].dropna().astype(str).unique().tolist()))
    status = load_stock_status_daily_from_duckdb(
        db_path, symbols=requested, start=start, end=end
    )
    status_coverage = evaluate_market_status_coverage(
        base,
        status,
        symbols=requested,
        start=start,
        end=end,
    )
    if status_coverage["missing_days"]:
        examples = ", ".join(
            f"{item['symbol']}@{item['date']}"
            for item in status_coverage["missing_examples"][:10]
        )
        raise ValueError(
            f"Historical ST status is missing for {status_coverage['missing_days']} symbol-days"
            f" ({examples}). Re-run Download for the training range to backfill daily extensions."
        )
    return FeatureSourceFrames(
        base=base,
        daily=daily,
        daily_status=status,
        requested_symbols=requested,
        status_coverage=status_coverage,
    )


def load_feature_source_frames_from_parquet_cache(
    cache_dir: str | Path,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    adjust: str = "none",
    start: str | None = None,
    end: str | None = None,
    config: FeatureDatasetConfig | None = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> FeatureSourceFrames:
    """Load source frames from a symbol/frequency parquet market cache."""

    cfg = config or FeatureDatasetConfig(
        trade_freq=spec.trade_frequency,
        frequencies=("5min", *spec.derived_frequencies),
        adjust=adjust,
    )
    selected_symbols = _normalize_symbols(symbol=symbol, symbols=symbols)
    base = load_kline_from_market_parquet_cache(
        cache_dir,
        symbols=selected_symbols,
        freq=normalize_frequency(cfg.trade_freq),
        adjust=cfg.adjust,
        start=None,
        end=end,
    )
    daily = load_kline_from_market_parquet_cache(
        cache_dir,
        symbols=selected_symbols,
        freq="daily",
        adjust=cfg.adjust,
        start=None,
        end=end,
    )
    liquidity = load_stock_liquidity_from_market_parquet_cache(
        cache_dir, symbols=selected_symbols, start=None, end=end
    )
    daily = _attach_daily_turn(daily, liquidity)
    requested = tuple(selected_symbols or sorted(base["symbol"].dropna().astype(str).unique().tolist()))
    status = load_stock_status_from_market_parquet_cache(
        cache_dir, symbols=requested, start=start, end=end
    )
    status_coverage = evaluate_market_status_coverage(
        base,
        status,
        symbols=requested,
        start=start,
        end=end,
    )
    if status_coverage["missing_days"]:
        examples = ", ".join(
            f"{item['symbol']}@{item['date']}"
            for item in status_coverage["missing_examples"][:10]
        )
        raise ValueError(
            f"Historical ST status is missing for {status_coverage['missing_days']} symbol-days"
            f" ({examples}). Re-run Download for the training range to backfill daily extensions."
        )
    return FeatureSourceFrames(
        base=base,
        daily=daily,
        daily_status=status,
        requested_symbols=requested,
        status_coverage=status_coverage,
    )


def build_feature_dataset_from_frames(
    source: FeatureSourceFrames,
    *,
    adjust: str = "none",
    start: str | None = None,
    end: str | None = None,
    config: FeatureDatasetConfig | None = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> FeatureDataset:
    cfg = config or FeatureDatasetConfig(
        trade_freq=spec.trade_frequency,
        frequencies=("5min", *spec.derived_frequencies),
        adjust=adjust,
    )
    cfg = replace(
        cfg,
        decision_start=start,
        decision_end=end,
        require_limit_reference=True,
    )
    dataset = build_feature_dataset(
        source.base,
        daily_bars=source.daily,
        daily_status=source.daily_status,
        config=cfg,
        spec=spec,
    )
    return replace(dataset, requested_symbols=tuple(source.requested_symbols))


def build_feature_dataset_from_duckdb(
    db_path: str | Path,
    *,
    symbol: str | None = None,
    symbols: Iterable[str] | None = None,
    adjust: str = "none",
    start: str | None = None,
    end: str | None = None,
    config: FeatureDatasetConfig | None = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> FeatureDataset:
    source = load_feature_source_frames_from_duckdb(
        db_path,
        symbol=symbol,
        symbols=symbols,
        adjust=adjust,
        start=start,
        end=end,
        config=config,
        spec=spec,
    )
    return build_feature_dataset_from_frames(
        source,
        adjust=adjust,
        start=start,
        end=end,
        config=config,
        spec=spec,
    )


def evaluate_market_status_coverage(
    base: pd.DataFrame,
    status: pd.DataFrame,
    *,
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, object]:
    """Compare decision-session dates with dated ST facts."""
    requested = sorted({str(symbol).upper() for symbol in symbols or []})
    if base is None or base.empty or "symbol" not in base or "datetime" not in base:
        return {
            "required_days": 0,
            "covered_days": 0,
            "missing_days": 0,
            "missing_symbols": [],
            "missing_examples": [],
            "st_days": 0,
            "st_symbols": [],
        }

    sessions = base[["symbol", "datetime"]].copy()
    sessions["symbol"] = sessions["symbol"].astype(str).str.upper()
    sessions["date"] = pd.to_datetime(sessions["datetime"], errors="coerce").dt.normalize()
    sessions = sessions.dropna(subset=["symbol", "date"])
    if requested:
        sessions = sessions.loc[sessions["symbol"].isin(requested)]
    if start:
        sessions = sessions.loc[sessions["date"] >= pd.Timestamp(start).normalize()]
    if end:
        sessions = sessions.loc[sessions["date"] <= pd.Timestamp(end).normalize()]
    sessions = sessions[["symbol", "date"]].drop_duplicates()

    facts = status[[column for column in ("symbol", "date", "is_st") if column in status]].copy() if status is not None else pd.DataFrame()
    if facts.empty or not {"symbol", "date", "is_st"}.issubset(facts.columns):
        facts = pd.DataFrame(columns=["symbol", "date", "is_st"])
    else:
        facts["symbol"] = facts["symbol"].astype(str).str.upper()
        facts["date"] = pd.to_datetime(facts["date"], errors="coerce").dt.normalize()
        facts = facts.dropna(subset=["symbol", "date", "is_st"]).drop_duplicates(
            ["symbol", "date"], keep="last"
        )

    joined = sessions.merge(facts, on=["symbol", "date"], how="left", indicator=True)
    missing = joined.loc[joined["_merge"].ne("both"), ["symbol", "date"]]
    covered = joined.loc[joined["_merge"].eq("both")]
    return {
        "required_days": int(len(sessions)),
        "covered_days": int(len(covered)),
        "missing_days": int(len(missing)),
        "missing_symbols": sorted(missing["symbol"].dropna().astype(str).unique().tolist()),
        "missing_examples": [
            {"symbol": str(row.symbol), "date": str(pd.Timestamp(row.date).date())}
            for row in missing.head(20).itertuples(index=False)
        ],
        "st_days": int(covered["is_st"].fillna(False).astype(bool).sum()),
        "st_symbols": sorted(
            covered.loc[covered["is_st"].fillna(False).astype(bool), "symbol"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    }


def _attach_daily_turn(daily: pd.DataFrame, liquidity: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty or liquidity is None or liquidity.empty:
        return daily
    facts = liquidity[["symbol", "date", "turn"]].copy()
    facts["symbol"] = facts["symbol"].astype(str).str.upper()
    facts["date"] = pd.to_datetime(facts["date"], errors="coerce").dt.normalize()
    result = daily.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["_date"] = pd.to_datetime(result["datetime"], errors="coerce").dt.normalize()
    result = result.drop(columns=["turn"], errors="ignore").merge(
        facts.rename(columns={"date": "_date"}), on=["symbol", "_date"], how="left"
    )
    return result.drop(columns=["_date"])


def evaluate_frequency_warmup(
    base: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    frequencies: tuple[str, ...],
    trade_freq: str,
    start: str | None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    symbols: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Report pre-training history using counts only, without OHLC aggregation."""
    requested_symbols = (
        sorted({str(symbol).upper() for symbol in symbols})
        if symbols is not None
        else (sorted(base["symbol"].dropna().astype(str).str.upper().unique().tolist()) if not base.empty else [])
    )
    cutoff = pd.Timestamp(start) if start else None
    reports: list[dict[str, object]] = []
    for freq in frequencies:
        normalized = normalize_frequency(freq)
        feature_requirements = {
            "returns_and_rolling": FEATURE_WARMUP_REQUIREMENTS["returns_and_rolling"]
        }
        for indicator in spec.indicators:
            if indicator.enabled and normalized in indicator.frequencies:
                feature_requirements[indicator.id] = indicator_lookback(indicator)
        required = max(feature_requirements.values())
        counts = _frequency_bar_counts(
            base,
            daily,
            freq=normalized,
            trade_freq=normalize_frequency(trade_freq),
            cutoff=cutoff,
        )
        short = [symbol for symbol in requested_symbols if int(counts.get(symbol, 0)) < required]
        reports.append(
            {
                "freq": normalized,
                "required_bars": required,
                "available_bars": {symbol: int(counts.get(symbol, 0)) for symbol in requested_symbols},
                "short_symbols": short,
                "feature_requirements": feature_requirements,
            }
        )
    return reports


def _frequency_bar_counts(
    base: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    freq: str,
    trade_freq: str,
    cutoff: pd.Timestamp | None,
) -> dict[str, int]:
    """Count available bars per symbol without materializing aggregate OHLC rows."""
    if freq in {trade_freq, "15min", "30min", "60min"}:
        frame = base
    elif freq in {"daily", "weekly"}:
        frame = daily
    else:
        return {}
    if frame is None or frame.empty or "symbol" not in frame or "datetime" not in frame:
        return {}

    columns = ["symbol", "datetime"]
    if "adjust" in frame.columns:
        columns.append("adjust")
    working = frame[columns].copy()
    working["symbol"] = working["symbol"].astype(str).str.upper()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working.dropna(subset=["symbol", "datetime"])
    if cutoff is not None:
        working = working.loc[working["datetime"] < cutoff]
    if working.empty:
        return {}

    if freq == trade_freq or freq == "daily":
        return {str(key): int(value) for key, value in working.groupby("symbol", sort=False).size().items()}

    if freq in {"15min", "30min", "60min"}:
        source_minutes = INTRADAY_MINUTES[trade_freq]
        target_minutes = INTRADAY_MINUTES[freq]
        source_rows = target_minutes // source_minutes
        group_columns = ["symbol"]
        if "adjust" in working.columns:
            group_columns.append("adjust")
        working["_session"] = working["datetime"].dt.normalize()
        session_sizes = working.groupby([*group_columns, "_session"], sort=False).size()
        aggregate_counts = ((session_sizes + source_rows - 1) // source_rows).groupby(level="symbol").sum()
        return {str(key): int(value) for key, value in aggregate_counts.items()}

    working["_week"] = (
        working["datetime"] - pd.to_timedelta(working["datetime"].dt.weekday, unit="D")
    ).dt.normalize()
    weekly_counts = working.drop_duplicates(["symbol", "_week"]).groupby("symbol", sort=False).size()
    return {str(key): int(value) for key, value in weekly_counts.items()}


def _validate_dataset_frequencies(frequencies: tuple[str, ...], *, spec: FeatureSpec) -> None:
    allowed = {spec.base_frequency, *spec.derived_frequencies}
    unsupported = sorted(set(frequencies).difference(allowed))
    if unsupported:
        raise ValueError(
            f"Unsupported feature frequencies: {', '.join(unsupported)}. "
            f"feature_v1 supports: {', '.join(sorted(allowed))}."
        )


def _prepare_kline_frame(
    frame: pd.DataFrame | None,
    *,
    freq: str,
    adjust: str,
) -> pd.DataFrame:
    columns = list(KLINE_SELECT_COLUMNS) + ["turn", "progress"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = None

    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    result["freq"] = result["freq"].fillna(freq).map(normalize_frequency)
    result["adjust"] = result["adjust"].fillna(adjust).astype(str)
    result = result.loc[
        result["datetime"].notna()
        & result["symbol"].ne("")
        & result["freq"].eq(normalize_frequency(freq))
        & result["adjust"].eq(str(adjust))
    ].copy()

    for column in ["open", "high", "low", "close", "volume", "amount", "pctChg", "turn", "progress"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["volume"] = result["volume"].fillna(0.0)
    result["amount"] = result["amount"].fillna(0.0)
    result["progress"] = result["progress"].fillna(1.0)
    return result.sort_values(["symbol", "adjust", "datetime"]).reset_index(drop=True)


def _attach_decision_ids(decisions: pd.DataFrame) -> pd.DataFrame:
    result = decisions.copy()
    if result.empty:
        result.insert(0, "decision_id", pd.Series(dtype="string"))
        return result
    result["decision_time"] = pd.to_datetime(result["decision_time"], errors="coerce")
    ids = []
    for row in result.itertuples(index=False):
        timestamp = pd.Timestamp(row.decision_time).strftime("%Y%m%d%H%M%S")
        ids.append(f"{row.symbol}|{row.adjust}|{timestamp}|{row.stage}")
    result.insert(0, "decision_id", ids)
    return result


def _visible_bars_by_frequency(
    *,
    decision: object,
    base: pd.DataFrame,
    daily: pd.DataFrame,
    frequencies: tuple[str, ...],
    trade_freq: str,
) -> dict[str, pd.DataFrame]:
    symbol = str(decision.symbol).upper()
    adjust = str(decision.adjust)
    decision_time = pd.Timestamp(decision.decision_time)
    visible_cutoff = _visible_cutoff(decision)
    session_date = decision_time.date()

    symbol_base = base.loc[(base["symbol"].eq(symbol)) & (base["adjust"].eq(adjust))].copy()
    visible_base = symbol_base.loc[symbol_base["datetime"] <= visible_cutoff].copy()
    symbol_daily = daily.loc[(daily["symbol"].eq(symbol)) & (daily["adjust"].eq(adjust))].copy()
    completed_daily = symbol_daily.loc[symbol_daily["datetime"].dt.date < session_date].copy()
    partial_daily = _current_partial_daily(
        decision=decision,
        current_day_base=symbol_base.loc[
            (symbol_base["datetime"].dt.date == session_date)
            & (symbol_base["datetime"] <= visible_cutoff)
        ].copy(),
        trade_freq=trade_freq,
    )
    daily_parts = [part for part in (completed_daily, partial_daily) if not part.empty]
    daily_context = (
        pd.concat([part.dropna(axis=1, how="all") for part in daily_parts], ignore_index=True)
        if daily_parts
        else pd.DataFrame(columns=list(KLINE_SELECT_COLUMNS) + ["progress"])
    )

    result: dict[str, pd.DataFrame] = {}
    for freq in frequencies:
        if freq == trade_freq:
            result[freq] = visible_base.copy()
        elif freq in {"15min", "30min", "60min"}:
            result[freq] = aggregate_ohlcv_from_base(visible_base, freq, base_freq=trade_freq)
        elif freq == "daily":
            result[freq] = daily_context.copy()
        elif freq == "weekly":
            result[freq] = aggregate_ohlcv_from_base(daily_context, freq, base_freq="daily")
        else:
            raise ValueError(f"Unsupported dataset frequency: {freq}")
    return result


def _visible_cutoff(decision: object) -> pd.Timestamp:
    visible_bar_end = getattr(decision, "visible_bar_end", pd.NaT)
    if pd.notna(visible_bar_end):
        return pd.Timestamp(visible_bar_end)
    return pd.Timestamp(decision.decision_time)


def _current_partial_daily(
    *,
    decision: object,
    current_day_base: pd.DataFrame,
    trade_freq: str,
) -> pd.DataFrame:
    if not current_day_base.empty:
        partial = aggregate_ohlcv_from_base(current_day_base, "daily", base_freq=trade_freq)
        partial["datetime"] = pd.to_datetime(partial["bar_end"], errors="coerce").dt.normalize()
        partial["freq"] = "daily"
        return partial

    if str(decision.stage) != "open_auction":
        return pd.DataFrame(columns=KLINE_SELECT_COLUMNS + ["progress"])

    decision_time = pd.Timestamp(decision.decision_time)
    price = float(decision.execution_price)
    return pd.DataFrame(
        [
            {
                "symbol": str(decision.symbol).upper(),
                "datetime": decision_time.normalize(),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
                "amount": 0.0,
                "pctChg": None,
                "source": "open_auction_snapshot",
                "freq": "daily",
                "adjust": str(decision.adjust),
                "progress": 0.0,
            }
        ]
    )


def _build_sequence_frame(
    visible_bars: pd.DataFrame,
    *,
    decision: object,
    freq: str,
    spec: FeatureSpec,
) -> pd.DataFrame:
    if visible_bars.empty:
        return _empty_market_frame(spec=spec)

    bars = visible_bars.copy()
    if "datetime" not in bars.columns and "bar_end" in bars.columns:
        bars["datetime"] = bars["bar_end"]
    bars["freq"] = freq
    features = build_bar_features(bars, spec=spec)
    if features.empty:
        return _empty_market_frame(spec=spec)

    window = int(spec.sequence_windows.get(freq, len(features)))
    sequence = features.sort_values("datetime").tail(max(1, window)).copy()
    sequence["sequence_valid_ratio"] = min(1.0, len(sequence) / float(max(1, window)))
    sequence.insert(0, "decision_id", decision.decision_id)
    sequence.insert(1, "decision_time", pd.Timestamp(decision.decision_time))
    sequence.insert(2, "stage", str(decision.stage))
    sequence.insert(3, "sequence_index", range(len(sequence)))
    sequence = sequence.rename(columns={"datetime": "bar_datetime"})
    return sequence[_market_columns(spec=spec)]


def _constraints_frame(decisions: pd.DataFrame, *, spec: FeatureSpec) -> pd.DataFrame:
    columns = ["decision_id", *spec.constraint_feature_names]
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    result = decisions.copy()
    for column in spec.constraint_feature_names:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = result[column].astype(float)
    return result[columns].reset_index(drop=True)


def _market_columns(*, spec: FeatureSpec) -> list[str]:
    return [
        "decision_id",
        "decision_time",
        "stage",
        "sequence_index",
        "symbol",
        "bar_datetime",
        "freq",
        "adjust",
        *spec.market_feature_names,
    ]


def _empty_market_frame(*, spec: FeatureSpec) -> pd.DataFrame:
    return pd.DataFrame(columns=_market_columns(spec=spec))


def _normalize_symbols(*, symbol: str | None, symbols: Iterable[str] | None) -> list[str]:
    values: list[str] = []
    if symbol:
        values.extend(str(symbol).replace("\n", ",").split(","))
    if symbols:
        values.extend(str(item) for item in symbols)
    return [item.strip().upper() for item in values if item and item.strip()]
