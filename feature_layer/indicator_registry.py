from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import re
from typing import Any, Iterable

from feature_layer.specs import (
    DEFAULT_FEATURE_SPEC,
    EmaChannelSpec,
    FeatureField,
    FeatureSpec,
    IndicatorSpec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDICATOR_CONFIG_PATH = PROJECT_ROOT / "config" / "features" / "indicators.json"
SUPPORTED_INDICATOR_KINDS = ("macd", "kd", "efi", "ema_channel")
SUPPORTED_RENDER_TARGETS = ("main_overlay", "sub_panel")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,47}$")


def indicator_config_payload(
    path: str | Path = DEFAULT_INDICATOR_CONFIG_PATH,
) -> dict[str, Any]:
    indicators = load_indicator_specs(path)
    return {
        "path": _display_path(Path(path)),
        "supported_kinds": list(SUPPORTED_INDICATOR_KINDS),
        "supported_render_targets": list(SUPPORTED_RENDER_TARGETS),
        "available_frequencies": [
            DEFAULT_FEATURE_SPEC.base_frequency,
            *DEFAULT_FEATURE_SPEC.derived_frequencies,
        ],
        "indicators": [indicator_to_payload(item) for item in indicators],
    }


def load_indicator_specs(
    path: str | Path = DEFAULT_INDICATOR_CONFIG_PATH,
) -> tuple[IndicatorSpec, ...]:
    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_FEATURE_SPEC.indicators
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return validate_indicator_specs(payload.get("indicators", []))


def save_indicator_specs(
    payload: dict[str, Any],
    path: str | Path = DEFAULT_INDICATOR_CONFIG_PATH,
) -> dict[str, Any]:
    indicators = validate_indicator_specs(payload.get("indicators", []))
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": 2,
        "indicators": [indicator_to_payload(item) for item in indicators],
    }
    config_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return indicator_config_payload(config_path)


def active_feature_spec(
    *,
    path: str | Path = DEFAULT_INDICATOR_CONFIG_PATH,
    base_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> FeatureSpec:
    indicators = load_indicator_specs(path)
    return feature_spec_from_indicators(indicators, base_spec=base_spec)


def feature_spec_from_indicators(
    indicators: Iterable[IndicatorSpec],
    *,
    base_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> FeatureSpec:
    indicators = tuple(indicators)
    indicator_fields: list[FeatureField] = []
    ema_channels: list[EmaChannelSpec] = []
    for indicator in indicators:
        if not indicator.enabled:
            continue
        indicator_fields.extend(fields_for_indicator(indicator))
        if indicator.kind == "ema_channel":
            ema_channels.append(
                EmaChannelSpec(
                    name=indicator.id,
                    fast_period=indicator.params["fast"],
                    slow_period=indicator.params["slow"],
                )
            )

    core_fields = tuple(field for field in base_spec.market_fields if field.group != "indicator")
    return replace(
        base_spec,
        indicators=indicators,
        ema_channels=tuple(ema_channels),
        market_fields=core_fields + tuple(indicator_fields),
    )


def validate_indicator_specs(rows: Iterable[dict[str, Any]]) -> tuple[IndicatorSpec, ...]:
    allowed_frequencies = {
        DEFAULT_FEATURE_SPEC.base_frequency,
        *DEFAULT_FEATURE_SPEC.derived_frequencies,
    }
    result: list[IndicatorSpec] = []
    seen_ids: set[str] = set()
    seen_fields: set[str] = set()
    for raw in rows:
        indicator_id = str(raw.get("id") or "").strip()
        kind = str(raw.get("kind") or raw.get("type") or "").strip().lower()
        if not _ID_PATTERN.fullmatch(indicator_id):
            raise ValueError(
                f"Invalid indicator id: {indicator_id!r}. Use letters, numbers, and underscores."
            )
        if indicator_id in seen_ids:
            raise ValueError(f"Duplicate indicator id: {indicator_id}")
        if kind not in SUPPORTED_INDICATOR_KINDS:
            raise ValueError(f"Unsupported indicator kind: {kind}")

        frequencies = tuple(
            dict.fromkeys(str(item).strip() for item in raw.get("frequencies", []) if str(item).strip())
        )
        unsupported = sorted(set(frequencies).difference(allowed_frequencies))
        if unsupported:
            raise ValueError(
                f"Indicator {indicator_id} has unsupported frequencies: {', '.join(unsupported)}"
            )
        if not frequencies:
            raise ValueError(f"Indicator {indicator_id} must enable at least one frequency.")

        params = _validate_params(kind, raw.get("params") or {})
        default_target = _default_render_target(kind)
        render_target = str(raw.get("render_target") or default_target).strip().lower()
        if render_target not in SUPPORTED_RENDER_TARGETS:
            raise ValueError(f"Unsupported render target: {render_target}")
        if render_target != default_target:
            raise ValueError(
                f"{kind} indicators must use render_target={default_target}."
            )
        indicator = IndicatorSpec(
            id=indicator_id,
            kind=kind,
            enabled=bool(raw.get("enabled", True)),
            frequencies=frequencies,
            params=params,
            render_target=render_target,
            default_visible=(
                bool(raw["default_visible"])
                if "default_visible" in raw
                else kind != "efi"
            ),
        )
        fields = {field.name for field in fields_for_indicator(indicator)}
        collision = fields.intersection(seen_fields)
        if collision:
            raise ValueError(f"Indicator output field collision: {', '.join(sorted(collision))}")
        seen_fields.update(fields)
        seen_ids.add(indicator_id)
        result.append(indicator)
    return tuple(result)


def indicator_to_payload(indicator: IndicatorSpec) -> dict[str, Any]:
    return {
        "id": indicator.id,
        "kind": indicator.kind,
        "enabled": indicator.enabled,
        "frequencies": list(indicator.frequencies),
        "params": dict(indicator.params),
        "render_target": indicator.render_target,
        "default_visible": indicator.default_visible,
        "outputs": [field.name for field in fields_for_indicator(indicator)],
    }


def fields_for_indicator(indicator: IndicatorSpec) -> tuple[FeatureField, ...]:
    prefix = indicator.id
    if indicator.kind == "macd":
        rows = [
            (f"{prefix}_dif_pct", "DIF divided by close.", (-0.3, 0.3)),
            (f"{prefix}_dea_pct", "Signal line divided by close.", (-0.3, 0.3)),
            (f"{prefix}_hist_pct", "Histogram divided by close.", (-0.3, 0.3)),
        ]
    elif indicator.kind == "kd":
        rows = [
            (f"{prefix}_k_norm", "KD K value scaled to 0-1.", (0.0, 1.0)),
            (f"{prefix}_d_norm", "KD D value scaled to 0-1.", (0.0, 1.0)),
            (f"{prefix}_diff_norm", "KD K-D spread.", (-1.0, 1.0)),
        ]
    elif indicator.kind == "efi":
        rows = [
            (f"{prefix}2_norm", "EFI2 = EMA((close - REF(close, 1)) * volume, fast), normalized by prior absolute force.", (-10.0, 10.0)),
            (f"{prefix}13_norm", "EFI13 = EMA((close - REF(close, 1)) * volume, slow), normalized by prior absolute force.", (-10.0, 10.0)),
        ]
    else:
        channel_prefix = f"ema_channel_{prefix}"
        rows = [
            (f"{channel_prefix}_position", "Close position inside the EMA channel.", (-5.0, 5.0)),
            (f"{channel_prefix}_gap", "Close gap to the EMA channel midpoint.", (-0.5, 0.5)),
            (f"{channel_prefix}_width", "EMA channel width divided by midpoint.", (0.0, 1.0)),
            (f"{channel_prefix}_slope", "EMA channel midpoint one-bar return.", (-0.3, 0.3)),
        ]
    return tuple(
        FeatureField(name=name, group="indicator", description=description, clip=clip)
        for name, description, clip in rows
    )


def indicator_lookback(indicator: IndicatorSpec) -> int:
    params = indicator.params
    if indicator.kind == "macd":
        return params["slow"] + params["signal"]
    if indicator.kind == "kd":
        return params["lookback"]
    if indicator.kind == "efi":
        return max(params["fast"], params["slow"], params["baseline"])
    return params["slow"]


def _default_render_target(kind: str) -> str:
    return "main_overlay" if kind == "ema_channel" else "sub_panel"


def _validate_params(kind: str, raw: dict[str, Any]) -> dict[str, int]:
    defaults = {
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "kd": {"lookback": 9, "smooth_k": 3, "smooth_d": 3},
        "efi": {"fast": 2, "slow": 13, "baseline": 20},
        "ema_channel": {"fast": 13, "slow": 21},
    }[kind]
    params: dict[str, int] = {}
    for name, default in defaults.items():
        try:
            value = int(raw.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{kind}.{name} must be an integer.") from exc
        if value <= 0 or value > 5000:
            raise ValueError(f"{kind}.{name} must be between 1 and 5000.")
        params[name] = value
    if kind in {"macd", "ema_channel"} and params["fast"] >= params["slow"]:
        raise ValueError(f"{kind}.fast must be smaller than {kind}.slow.")
    return params


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)
