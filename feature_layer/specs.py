from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class DecisionStage(str, Enum):
    """Decision moments supported by the first feature contract."""

    OPEN_AUCTION = "open_auction"
    BAR_CLOSE = "bar_close"


@dataclass(frozen=True)
class EmaChannelSpec:
    name: str
    fast_period: int
    slow_period: int


@dataclass(frozen=True)
class IndicatorSpec:
    id: str
    kind: str
    enabled: bool
    frequencies: tuple[str, ...]
    params: dict[str, int]
    render_target: str
    default_visible: bool


@dataclass(frozen=True)
class FeatureField:
    name: str
    group: str
    description: str
    clip: tuple[float, float] | None = None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str
    base_frequency: str
    trade_frequency: str
    derived_frequencies: tuple[str, ...]
    decision_stages: tuple[DecisionStage, ...]
    sequence_windows: dict[str, int]
    indicators: tuple[IndicatorSpec, ...]
    ema_channels: tuple[EmaChannelSpec, ...]
    market_fields: tuple[FeatureField, ...]
    context_fields: tuple[FeatureField, ...]
    portfolio_fields: tuple[FeatureField, ...]
    constraint_fields: tuple[FeatureField, ...]
    environment_fields: tuple[FeatureField, ...] = ()

    @property
    def market_feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.market_fields)

    @property
    def portfolio_feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.portfolio_fields)

    @property
    def context_feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.context_fields)

    @property
    def constraint_feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.constraint_fields)

    @property
    def environment_feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.environment_fields)

    @property
    def model_input_names(self) -> tuple[str, ...]:
        return (
            self.market_feature_names
            + self.context_feature_names
            + self.portfolio_feature_names
            + self.constraint_feature_names
            + self.environment_feature_names
        )

    @property
    def generated_feature_names(self) -> tuple[str, ...]:
        return self.market_feature_names + self.context_feature_names + self.constraint_feature_names


def _fields(rows: Iterable[tuple[str, str, str, tuple[float, float] | None]]) -> tuple[FeatureField, ...]:
    return tuple(
        FeatureField(name=name, group=group, description=description, clip=clip)
        for name, group, description, clip in rows
    )


EMA_CHANNELS_V1: tuple[EmaChannelSpec, ...] = (
    EmaChannelSpec(name="13_21", fast_period=13, slow_period=21),
    EmaChannelSpec(name="144_169", fast_period=144, slow_period=169),
)


INDICATORS_V1: tuple[IndicatorSpec, ...] = (
    IndicatorSpec(
        id="macd",
        kind="macd",
        enabled=True,
        frequencies=("5min", "15min", "30min", "60min", "daily", "weekly", "monthly"),
        params={"fast": 12, "slow": 26, "signal": 9},
        render_target="sub_panel",
        default_visible=True,
    ),
    IndicatorSpec(
        id="kd",
        kind="kd",
        enabled=True,
        frequencies=("5min", "15min", "30min", "60min", "daily", "weekly", "monthly"),
        params={"lookback": 9, "smooth_k": 3, "smooth_d": 3},
        render_target="sub_panel",
        default_visible=True,
    ),
    IndicatorSpec(
        id="efi",
        kind="efi",
        enabled=True,
        frequencies=("5min", "15min", "30min", "60min", "daily", "weekly", "monthly"),
        params={"fast": 2, "slow": 13, "baseline": 20},
        render_target="sub_panel",
        default_visible=False,
    ),
    IndicatorSpec(
        id="13_21",
        kind="ema_channel",
        enabled=True,
        frequencies=("5min", "15min", "30min", "60min", "daily", "weekly", "monthly"),
        params={"fast": 13, "slow": 21},
        render_target="main_overlay",
        default_visible=True,
    ),
    IndicatorSpec(
        id="144_169",
        kind="ema_channel",
        enabled=True,
        frequencies=("5min", "15min", "30min", "60min", "daily", "weekly", "monthly"),
        params={"fast": 144, "slow": 169},
        render_target="main_overlay",
        default_visible=True,
    ),
)


MARKET_FIELDS_V1: tuple[FeatureField, ...] = _fields(
    [
        ("pctChg", "price", "Close-to-previous-close return.", (-0.3, 0.3)),
        ("log_ret", "price", "Log close-to-previous-close return.", (-0.3, 0.3)),
        ("open_close_ret", "price", "Intrabar close-to-open return.", (-0.3, 0.3)),
        ("gap_ret", "price", "Open-to-previous-close gap.", (-0.3, 0.3)),
        ("high_low_range", "shape", "High-low range divided by previous close.", (0.0, 0.5)),
        ("body_range", "shape", "Absolute candle body divided by previous close.", (0.0, 0.5)),
        ("close_position", "shape", "Close location inside the high-low range.", (-1.0, 2.0)),
        ("upper_shadow", "shape", "Upper shadow divided by previous close.", (0.0, 0.5)),
        ("lower_shadow", "shape", "Lower shadow divided by previous close.", (0.0, 0.5)),
        ("ret_3", "trend", "Three-bar cumulative return.", (-0.5, 0.5)),
        ("ret_5", "trend", "Five-bar cumulative return.", (-0.5, 0.5)),
        ("ret_10", "trend", "Ten-bar cumulative return.", (-0.8, 0.8)),
        ("ret_20", "trend", "Twenty-bar cumulative return.", (-1.0, 1.0)),
        ("volatility_5", "volatility", "Five-bar return standard deviation.", (0.0, 0.3)),
        ("volatility_20", "volatility", "Twenty-bar return standard deviation.", (0.0, 0.3)),
        ("range_mean_20", "volatility", "Twenty-bar average high-low range.", (0.0, 0.5)),
        ("volume_ratio_20", "liquidity", "Current volume divided by prior 20-bar average volume.", (0.0, 20.0)),
        ("amount_ratio_20", "liquidity", "Current amount divided by prior 20-bar average amount.", (0.0, 20.0)),
        ("turn_lag1", "liquidity", "Previous visible daily turnover.", (0.0, 100.0)),
        ("turn_ma20", "liquidity", "Prior 20-day average turnover.", (0.0, 100.0)),
        ("turn_z20", "liquidity", "Previous turnover z-score against prior turnover history.", (-5.0, 5.0)),
        ("macd_dif_pct", "indicator", "MACD DIF divided by close.", (-0.3, 0.3)),
        ("macd_dea_pct", "indicator", "MACD DEA divided by close.", (-0.3, 0.3)),
        ("macd_hist_pct", "indicator", "MACD histogram divided by close.", (-0.3, 0.3)),
        ("kd_k_norm", "indicator", "KD K value scaled to 0-1.", (0.0, 1.0)),
        ("kd_d_norm", "indicator", "KD D value scaled to 0-1.", (0.0, 1.0)),
        ("kd_diff_norm", "indicator", "KD K-D spread scaled to roughly -1 to 1.", (-1.0, 1.0)),
        ("efi2_norm", "indicator", "EFI2 = EMA((close - REF(close, 1)) * volume, 2), normalized by prior absolute force.", (-10.0, 10.0)),
        ("efi13_norm", "indicator", "EFI13 = EMA((close - REF(close, 1)) * volume, 13), normalized by prior absolute force.", (-10.0, 10.0)),
        ("ema_channel_13_21_position", "indicator", "Close position inside EMA13/EMA21 channel.", (-5.0, 5.0)),
        ("ema_channel_13_21_gap", "indicator", "Close gap to EMA13/EMA21 channel midpoint.", (-0.5, 0.5)),
        ("ema_channel_13_21_width", "indicator", "EMA13/EMA21 channel width divided by midpoint.", (0.0, 1.0)),
        ("ema_channel_13_21_slope", "indicator", "EMA13/EMA21 midpoint one-bar return.", (-0.3, 0.3)),
        ("ema_channel_144_169_position", "indicator", "Close position inside EMA144/EMA169 channel.", (-5.0, 5.0)),
        ("ema_channel_144_169_gap", "indicator", "Close gap to EMA144/EMA169 channel midpoint.", (-0.5, 0.5)),
        ("ema_channel_144_169_width", "indicator", "EMA144/EMA169 channel width divided by midpoint.", (0.0, 1.0)),
        ("ema_channel_144_169_slope", "indicator", "EMA144/EMA169 midpoint one-bar return.", (-0.3, 0.3)),
        ("history_coverage", "availability", "Available bar history divided by the longest active indicator lookback.", (0.0, 1.0)),
        ("sequence_valid_ratio", "availability", "Real sequence rows divided by the configured sequence window.", (0.0, 1.0)),
        ("progress", "time", "Progress inside the current source period.", (0.0, 1.0)),
    ]
)


CONTEXT_FIELDS_V1: tuple[FeatureField, ...] = _fields(
    [
        ("bar_slot_norm", "decision_time", "Decision slot normalized inside the trading day.", (0.0, 1.0)),
        ("day_progress", "decision_time", "Trading-day progress visible at this decision.", (0.0, 1.0)),
        ("is_morning_session", "decision_time", "1 when the decision is in the morning session.", (0.0, 1.0)),
        ("is_afternoon_session", "decision_time", "1 when the decision is in the afternoon session.", (0.0, 1.0)),
        ("minutes_to_close_norm", "decision_time", "Remaining trading minutes divided by full-day trading minutes.", (0.0, 1.0)),
        ("is_open_auction", "decision_stage", "1 for the daily open-auction decision, otherwise 0.", (0.0, 1.0)),
    ]
)


PORTFOLIO_FIELDS_V1: tuple[FeatureField, ...] = _fields(
    [
        ("cash_ratio", "portfolio", "Cash divided by account equity.", (0.0, 1.0)),
        ("position_ratio", "portfolio", "Current symbol market value divided by account equity.", (0.0, 1.0)),
        ("available_position_ratio", "portfolio", "T+1 sellable position value divided by account equity.", (0.0, 1.0)),
        ("unrealized_pnl_ratio", "portfolio", "Unrealized PnL divided by account equity.", (-1.0, 1.0)),
        ("holding_bars_norm", "portfolio", "Holding duration normalized by the configured cap.", (0.0, 1.0)),
        ("one_lot_nav_ratio", "portfolio", "One-lot notional divided by account equity.", (0.0, 1.0)),
        ("max_buy_value_ratio", "portfolio", "Maximum buyable notional divided by account equity.", (0.0, 1.0)),
        ("max_sell_value_ratio", "portfolio", "Maximum sellable notional divided by account equity.", (0.0, 1.0)),
    ]
)


CONSTRAINT_FIELDS_V1: tuple[FeatureField, ...] = _fields(
    [
        ("market_can_buy", "market_constraint", "1 when volume, price-limit, previous-close, and dated ST rules allow increasing position.", (0.0, 1.0)),
        ("market_can_sell", "market_constraint", "1 when volume, price-limit, and previous-close rules allow reducing position, including legal ST exits.", (0.0, 1.0)),
        ("is_tradeable", "constraint", "1 when the bar can be traded by current data rules.", (0.0, 1.0)),
        ("is_limit_up", "constraint", "1 when buying is blocked by a limit-up state.", (0.0, 1.0)),
        ("is_limit_down", "constraint", "1 when selling is blocked by a limit-down state.", (0.0, 1.0)),
        ("is_zero_volume", "constraint", "1 when the completed bar has zero volume.", (0.0, 1.0)),
    ]
)


ENVIRONMENT_FIELDS_V1: tuple[FeatureField, ...] = _fields(
    [
        ("can_buy", "environment", "1 when both market rules and current account state allow buying.", (0.0, 1.0)),
        ("can_sell", "environment", "1 when both market rules and current account state allow selling.", (0.0, 1.0)),
    ]
)


FEATURE_SPEC_V1 = FeatureSpec(
    name="feature_v1",
    version="1.2",
    base_frequency="5min",
    trade_frequency="5min",
    derived_frequencies=("15min", "30min", "60min", "daily", "weekly", "monthly"),
    decision_stages=(DecisionStage.OPEN_AUCTION, DecisionStage.BAR_CLOSE),
    sequence_windows={
        "5min": 96,
        "15min": 64,
        "30min": 48,
        "60min": 32,
        "daily": 60,
        "weekly": 26,
        "monthly": 12,
    },
    indicators=INDICATORS_V1,
    ema_channels=EMA_CHANNELS_V1,
    market_fields=MARKET_FIELDS_V1,
    context_fields=CONTEXT_FIELDS_V1,
    portfolio_fields=PORTFOLIO_FIELDS_V1,
    constraint_fields=CONSTRAINT_FIELDS_V1,
    environment_fields=ENVIRONMENT_FIELDS_V1,
)


DEFAULT_FEATURE_SPEC = FEATURE_SPEC_V1
