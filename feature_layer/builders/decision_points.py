from __future__ import annotations

import pandas as pd

from feature_layer.builders.aggregation import INTRADAY_MINUTES, normalize_frequency
from feature_layer.rules import price_limit_pct_for_symbol
from feature_layer.specs import DecisionStage


def build_decision_points(
    base_bars: pd.DataFrame,
    *,
    daily_status: pd.DataFrame | None = None,
    trade_freq: str = "5min",
    include_open_auction: bool = True,
    require_limit_reference: bool = False,
) -> pd.DataFrame:
    """
    Build decision points from completed trade-frequency bars.

    `open_auction` rows use only the first bar's open plus previous visible
    history. `bar_close` rows use the completed bar close and the completed bar
    volume, so zero-volume and limit-state constraints are visible.

    Price-limit rules use the previous trading day's close throughout the
    session. Dated ST status overrides the board limit and blocks new buys.
    """
    bars = _prepare_bars(base_bars)
    if bars.empty:
        return _empty_decision_frame()

    freq = normalize_frequency(trade_freq)
    if freq not in INTRADAY_MINUTES:
        raise ValueError(f"Decision points require an intraday trade frequency: {trade_freq}")
    minutes = INTRADAY_MINUTES[freq]

    records: list[dict[str, object]] = []
    status_by_day = _prepare_status_lookup(daily_status)
    status_is_required = daily_status is not None
    group_cols = [column for column in ("symbol", "adjust") if column in bars.columns]
    bars["_session_date"] = bars["datetime"].dt.date
    previous_day_close: dict[tuple[object, ...], float | None] = {}
    previous_day_bar_end: dict[tuple[object, ...], pd.Timestamp | None] = {}

    for group_key, session in bars.groupby(group_cols + ["_session_date"], sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        meta_key = group_key[: len(group_cols)]
        session = session.sort_values("datetime")
        first = session.iloc[0]
        limit_reference_close = previous_day_close.get(meta_key)
        previous_visible_end = previous_day_bar_end.get(meta_key)
        if require_limit_reference and (
            limit_reference_close is None
            or pd.isna(limit_reference_close)
            or float(limit_reference_close) <= 0
        ):
            previous_day_close[meta_key] = float(session.iloc[-1]["close"])
            previous_day_bar_end[meta_key] = pd.Timestamp(session.iloc[-1]["datetime"])
            continue
        symbol = str(first["symbol"]).upper()
        session_date = pd.Timestamp(first["datetime"]).normalize()
        status_key = (symbol, session_date)
        status_known = not status_is_required or status_key in status_by_day
        is_st = bool(status_by_day.get(status_key, False))

        if include_open_auction:
            execution_price = float(first["open"])
            decision_time = pd.Timestamp(first["datetime"]) - pd.Timedelta(minutes=minutes)
            records.append(
                _decision_record(
                    row=first,
                    stage=DecisionStage.OPEN_AUCTION,
                    decision_time=decision_time,
                    execution_price=execution_price,
                    limit_reference_close=limit_reference_close,
                    is_st=is_st,
                    status_known=status_known,
                    visible_bar_end=previous_visible_end,
                    source_bar_end=pd.NaT,
                    is_zero_volume=False,
                )
            )

        for row in session.itertuples(index=False):
            execution_price = float(row.close)
            volume = float(row.volume or 0.0)
            records.append(
                _decision_record(
                    row=row,
                    stage=DecisionStage.BAR_CLOSE,
                    decision_time=pd.Timestamp(row.datetime),
                    execution_price=execution_price,
                    limit_reference_close=limit_reference_close,
                    is_st=is_st,
                    status_known=status_known,
                    visible_bar_end=pd.Timestamp(row.datetime),
                    source_bar_end=pd.Timestamp(row.datetime),
                    is_zero_volume=volume <= 0,
                )
            )
        previous_day_close[meta_key] = float(session.iloc[-1]["close"])
        previous_day_bar_end[meta_key] = pd.Timestamp(session.iloc[-1]["datetime"])

    return pd.DataFrame(records, columns=_decision_columns()).sort_values(
        ["symbol", "adjust", "decision_time", "stage"]
    ).reset_index(drop=True)


def _prepare_bars(base_bars: pd.DataFrame) -> pd.DataFrame:
    if base_bars.empty:
        return pd.DataFrame()

    required = {"datetime", "open", "close", "volume"}
    missing = required.difference(base_bars.columns)
    if missing:
        raise ValueError(f"Missing required decision columns: {sorted(missing)}")

    bars = base_bars.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    bars = bars.dropna(subset=["datetime", "open", "close"]).copy()
    for column in ["open", "close", "volume"]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars["volume"] = bars["volume"].fillna(0.0)
    if "symbol" not in bars.columns:
        bars["symbol"] = ""
    if "adjust" not in bars.columns:
        bars["adjust"] = ""
    return bars.sort_values(["symbol", "adjust", "datetime"]).reset_index(drop=True)


def _decision_record(
    *,
    row: object,
    stage: DecisionStage,
    decision_time: pd.Timestamp,
    execution_price: float,
    limit_reference_close: float | None,
    is_st: bool,
    status_known: bool,
    visible_bar_end: pd.Timestamp | None,
    source_bar_end: pd.Timestamp | None,
    is_zero_volume: bool,
) -> dict[str, object]:
    symbol = _value(row, "symbol")
    adjust = _value(row, "adjust")
    effective_limit_pct = price_limit_pct_for_symbol(str(symbol), is_st=is_st)
    is_limit_up, is_limit_down = _limit_flags(
        execution_price=execution_price,
        previous_close=limit_reference_close,
        limit_pct=effective_limit_pct,
    )
    is_tradeable = not bool(is_zero_volume)
    has_limit_reference = bool(
        limit_reference_close is not None
        and pd.notna(limit_reference_close)
        and float(limit_reference_close) > 0
    )
    return {
        "symbol": str(symbol),
        "adjust": str(adjust),
        "decision_time": decision_time,
        "stage": stage.value,
        "execution_price": execution_price,
        "limit_reference_close": limit_reference_close,
        "visible_bar_end": visible_bar_end,
        "source_bar_end": source_bar_end,
        "is_st": bool(is_st),
        "status_known": bool(status_known),
        "has_limit_reference": has_limit_reference,
        "limit_pct": effective_limit_pct,
        "market_can_buy": bool(status_known and has_limit_reference and not is_st and is_tradeable and not is_limit_up),
        "market_can_sell": bool(status_known and has_limit_reference and is_tradeable and not is_limit_down),
        "is_tradeable": bool(is_tradeable),
        "is_limit_up": bool(is_limit_up),
        "is_limit_down": bool(is_limit_down),
        "is_zero_volume": bool(is_zero_volume),
    }


def _limit_flags(
    *,
    execution_price: float,
    previous_close: float | None,
    limit_pct: float,
) -> tuple[bool, bool]:
    if previous_close is None or pd.isna(previous_close) or previous_close <= 0:
        return False, False
    upper = float(previous_close) * (1.0 + float(limit_pct) - 1e-4)
    lower = float(previous_close) * (1.0 - float(limit_pct) + 1e-4)
    return execution_price >= upper, execution_price <= lower


def _value(row: object, name: str) -> object:
    if isinstance(row, pd.Series):
        return row.get(name, "")
    return getattr(row, name, "")


def _prepare_status_lookup(daily_status: pd.DataFrame | None) -> dict[tuple[str, pd.Timestamp], bool]:
    if daily_status is None or daily_status.empty:
        return {}
    required = {"symbol", "date", "is_st"}
    missing = required.difference(daily_status.columns)
    if missing:
        raise ValueError(f"Missing required daily status columns: {sorted(missing)}")
    status = daily_status[["symbol", "date", "is_st"]].copy()
    status["symbol"] = status["symbol"].astype(str).str.upper()
    status["date"] = pd.to_datetime(status["date"], errors="coerce").dt.normalize()
    status = status.dropna(subset=["symbol", "date", "is_st"]).drop_duplicates(
        ["symbol", "date"], keep="last"
    )
    return {
        (str(row.symbol), pd.Timestamp(row.date)): bool(row.is_st)
        for row in status.itertuples(index=False)
    }


def _decision_columns() -> list[str]:
    return [
        "symbol",
        "adjust",
        "decision_time",
        "stage",
        "execution_price",
        "limit_reference_close",
        "visible_bar_end",
        "source_bar_end",
        "is_st",
        "status_known",
        "has_limit_reference",
        "limit_pct",
        "market_can_buy",
        "market_can_sell",
        "is_tradeable",
        "is_limit_up",
        "is_limit_down",
        "is_zero_volume",
    ]


def _empty_decision_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_decision_columns())
